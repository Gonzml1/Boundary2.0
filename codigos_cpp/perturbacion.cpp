// Mandelbrot por teoría de perturbaciones, con rebasing, BLA y floatexp.
//
// Idea: para un píxel c = C + dc, con C el centro de la vista, escribimos z_n = Z_n + dz_n,
// donde Z_n es la órbita de C (la "referencia"). Restando las dos recurrencias:
//
//     dz_{n+1} = (2 Z_n + dz_n) dz_n + dc
//
// Z se calcula UNA vez en alta precisión (GMP); cada píxel itera solo dz, que es chico.
//
// - Rebasing (Zhuoran, 2021): si |Z_m + dz| < |dz| o se termina la órbita, se reinicia la
//   referencia (dz = Z_m + dz, m = 0). Elimina los glitches con una sola órbita.
// - BLA: tabla de saltos lineales que saltean muchas iteraciones de golpe (ver perturbacion_core.h).
// - floatexp: en zooms donde dc ya no entra en un double (~1e-300) se usa un double con
//   exponente aparte. Es más lento por paso, pero con BLA se hacen muy pocos pasos.
//
// Uso desde Python (core/backend_cpp.py):
//   h = pert_crear(c_re, c_im, max_iter, bits)      órbita de referencia (se cachea)
//   pert_preparar_bla(h, modo, |dc|max, eps)         tabla BLA para el modo (0 double, 1 floatexp)
//   pert_render(h, dc_base, step, ..., salida)        render en CPU
//   pert_exportar(h, ...)                             copia órbita y tabla para la GPU
//   pert_liberar(h)

#include <gmp.h>
#include <omp.h>
#include <vector>
#include <cstring>
#include "perturbacion_core.h"

#define EXPORT extern "C" __declspec(dllexport)

struct Perturbacion {
    int largo = 0;                    // largo L de la órbita (Z_0 .. Z_{L-1})
    std::vector<CFE> Z_fe;            // la órbita en floatexp (no pierde los valores muy chicos)
    std::vector<CD>  Z_d;             // la misma en double
    int modo = 0;                     // 0 = double, 1 = floatexp
    std::vector<Bla<CD, double>> bla_d;
    std::vector<Bla<CFE, FE>>    bla_fe;
    int niveles = 0;
    int off[BLA_MAX_NIVELES] = {0};
    int cnt[BLA_MAX_NIVELES] = {0};
};

static CFE mpf_a_cfe(const mpf_t re, const mpf_t im) {
    long e_re, e_im;
    double m_re = mpf_get_d_2exp(&e_re, re);
    double m_im = mpf_get_d_2exp(&e_im, im);
    if (m_re == 0.0) e_re = e_im;
    if (m_im == 0.0) e_im = e_re;
    long e = e_re > e_im ? e_re : e_im;
    return cfe_normalizar(ldexp(m_re, (int)(e_re - e)), ldexp(m_im, (int)(e_im - e)), (int)e);
}

// Calcula Z_0 = 0, Z_{n+1} = Z_n^2 + C con `prec_bits` bits. Corta cuando Z escapa (|Z| > 2).
EXPORT void* pert_crear(const char* c_re_str, const char* c_im_str, int max_iter, int prec_bits)
{
    Perturbacion* p = new Perturbacion();
    p->Z_fe.reserve(max_iter + 1);

    mpf_t c_re, c_im, z_re, z_im, z_re2, z_im2;
    mpf_init2(c_re, prec_bits);  mpf_init2(c_im, prec_bits);
    mpf_init2(z_re, prec_bits);  mpf_init2(z_im, prec_bits);
    mpf_init2(z_re2, prec_bits); mpf_init2(z_im2, prec_bits);

    mpf_set_str(c_re, c_re_str, 10);
    mpf_set_str(c_im, c_im_str, 10);
    mpf_set_ui(z_re, 0);
    mpf_set_ui(z_im, 0);
    p->Z_fe.push_back(complejo<CFE>(0.0, 0.0));

    for (int n = 1; n <= max_iter; ++n) {
        mpf_mul(z_re2, z_re, z_re);
        mpf_mul(z_im2, z_im, z_im);

        // z_im = 2 z_re z_im + c_im
        mpf_mul(z_im, z_im, z_re);
        mpf_mul_2exp(z_im, z_im, 1);
        mpf_add(z_im, z_im, c_im);

        // z_re = z_re^2 - z_im^2 + c_re
        mpf_sub(z_re, z_re2, z_im2);
        mpf_add(z_re, z_re, c_re);

        CFE z = mpf_a_cfe(z_re, z_im);
        p->Z_fe.push_back(z);
        if (a_double(norma2(z)) > 4.0) break;
    }
    mpf_clears(c_re, c_im, z_re, z_im, z_re2, z_im2, NULL);

    p->largo = (int)p->Z_fe.size();
    p->Z_d.resize(p->largo);
    for (int i = 0; i < p->largo; ++i) {
        const CFE& z = p->Z_fe[i];
        p->Z_d[i].re = ldexp(z.re, z.e);
        p->Z_d[i].im = ldexp(z.im, z.e);
    }
    return p;
}

EXPORT int pert_largo(void* h) { return static_cast<Perturbacion*>(h)->largo; }

EXPORT void pert_liberar(void* h) { delete static_cast<Perturbacion*>(h); }

// Arma la tabla BLA. dc_max = máximo |dc| de la imagen (mantisa, exponente).
// eps <= 0 desactiva el BLA. Devuelve la cantidad de niveles.
EXPORT int pert_preparar_bla(void* h, int modo, double dc_max_m, int dc_max_e, double eps)
{
    Perturbacion* p = static_cast<Perturbacion*>(h);
    p->modo = modo;
    int n0 = p->largo > 2 ? p->largo - 2 : 0;
    // Niveles: n0 + n0/2 + n0/4 + ... < 2 n0
    if (modo == 0) {
        p->bla_d.resize(2 * (size_t)n0 + 1);
        p->bla_fe.clear();
        p->niveles = construir_bla<CD, double>(p->Z_d.data(), p->largo, ldexp(dc_max_m, dc_max_e),
                                               eps, p->bla_d.data(), p->off, p->cnt);
    } else {
        p->bla_fe.resize(2 * (size_t)n0 + 1);
        p->bla_d.clear();
        p->niveles = construir_bla<CFE, FE>(p->Z_fe.data(), p->largo, fe_normalizar(dc_max_m, dc_max_e),
                                            eps, p->bla_fe.data(), p->off, p->cnt);
    }
    return p->niveles;
}

template<class C, class R>
static long long render(const Perturbacion* p, const C* Z, const Bla<C, R>* tabla,
                        FE bx, FE by, FE sx, FE sy,
                        int width, int height, int max_iter, int* output)
{
    long long pasos_total = 0;
    // dynamic: las filas que tocan el interior del conjunto tardan mucho más que las otras
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:pasos_total)
    for (int j = 0; j < height; ++j) {
        FE dc_im = by + sy * fe((double)j);
        long long pasos = 0;
        for (int i = 0; i < width; ++i) {
            FE dc_re = bx + sx * fe((double)i);
            C dc = complejo_fe<C>(dc_re, dc_im);
            output[j * width + i] = iterar_pixel<C, R>(Z, p->largo, tabla, p->off, p->cnt,
                                                        p->niveles, dc, max_iter, &pasos);
        }
        pasos_total += pasos;
    }
    return pasos_total;
}

// El píxel (i, j) tiene dc = (bx + i * sx, by + j * sy), todo pasado como (mantisa, exponente).
// output[j * width + i] = iteración de escape. Devuelve la cantidad total de pasos hechos.
EXPORT long long pert_render(void* h,
    double bx_m, int bx_e, double by_m, int by_e,
    double sx_m, int sx_e, double sy_m, int sy_e,
    int width, int height, int max_iter, int* output)
{
    const Perturbacion* p = static_cast<Perturbacion*>(h);
    FE bx = fe_normalizar(bx_m, bx_e), by = fe_normalizar(by_m, by_e);
    FE sx = fe_normalizar(sx_m, sx_e), sy = fe_normalizar(sy_m, sy_e);
    if (p->modo == 0)
        return render<CD, double>(p, p->Z_d.data(), p->bla_d.data(), bx, by, sx, sy,
                                  width, height, max_iter, output);
    return render<CFE, FE>(p, p->Z_fe.data(), p->bla_fe.data(), bx, by, sx, sy,
                           width, height, max_iter, output);
}

// --- Para la GPU: tamaños y copia de la órbita y la tabla del modo actual ---
EXPORT long long pert_bytes_orbita(void* h) {
    const Perturbacion* p = static_cast<Perturbacion*>(h);
    return (long long)p->largo * (p->modo == 0 ? sizeof(CD) : sizeof(CFE));
}

EXPORT long long pert_bytes_bla(void* h) {
    const Perturbacion* p = static_cast<Perturbacion*>(h);
    long long n = p->niveles ? p->off[p->niveles - 1] + p->cnt[p->niveles - 1] : 0;
    return n * (p->modo == 0 ? sizeof(Bla<CD, double>) : sizeof(Bla<CFE, FE>));
}

EXPORT int pert_exportar(void* h, void* orbita, void* bla, int* off, int* cnt) {
    const Perturbacion* p = static_cast<Perturbacion*>(h);
    if (p->modo == 0) std::memcpy(orbita, p->Z_d.data(), pert_bytes_orbita(h));
    else              std::memcpy(orbita, p->Z_fe.data(), pert_bytes_orbita(h));
    if (p->modo == 0) std::memcpy(bla, p->bla_d.data(), pert_bytes_bla(h));
    else              std::memcpy(bla, p->bla_fe.data(), pert_bytes_bla(h));
    for (int k = 0; k < BLA_MAX_NIVELES; ++k) { off[k] = p->off[k]; cnt[k] = p->cnt[k]; }
    return p->niveles;
}
