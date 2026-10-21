// Fractales por teoría de perturbaciones, con rebasing, BLA y floatexp.
//
// Idea (para Mandelbrot): para un píxel c = C + dc, con C el centro de la vista, escribimos
// z_n = Z_n + dz_n, donde Z_n es la órbita de C (la "referencia"). Restando las recurrencias:
//
//     dz_{n+1} = (2 Z_n + dz_n) dz_n + dc
//
// Z se calcula UNA vez en alta precisión (GMP); cada píxel itera solo dz, que es chico.
// Cada fórmula define su propia perturbación (ver perturbacion_core.h); acá solo vive el paso
// de la órbita de referencia en GMP.
//
// - Rebasing (Zhuoran, 2021): si la diferencia respecto de Z_0 queda más chica que dz, o se
//   termina la órbita, se reinicia la referencia. Elimina los glitches con una sola órbita.
// - BLA: tabla de saltos lineales que saltean muchas iteraciones de golpe.
// - floatexp: en zooms donde dc ya no entra en un double se usa un double con exponente aparte.
//
// Uso desde Python (core/backend_cpp.py):
//   h = pert_crear(formula, c_re, c_im, k_re, k_im, max_iter, bits, cancelar)   órbita de referencia
//   pert_preparar_bla(h, modo, |dc|max, eps)                           tabla BLA (0 double, 1 floatexp)
//   pert_render(h, ...)                                                render en CPU
//   pert_exportar(h, ...)                                              órbita y tabla para la GPU
//   pert_liberar(h)

#include <gmp.h>
#include <omp.h>
#include <vector>
#include <cstring>
#include "perturbacion_core.h"

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else      // Linux (render_nube/compilar_linux.sh)
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

// FORMULAS: el número de cada una (tiene que coincidir con FORMULAS en Python)
enum { FORMULA_MANDELBROT = 0 };

struct Perturbacion {
    int formula = FORMULA_MANDELBROT;
    int largo = 0;                    // largo L de la órbita (Z_0 .. Z_{L-1})
    CD k_d;                           // parámetro de la fórmula (Mandelbrot: el centro C)
    std::vector<CFE> Z_fe;            // la órbita en floatexp (no pierde los valores muy chicos)
    std::vector<CD>  Z_d;             // la misma en double
    std::vector<CFF> Z_ff;            // la misma en floatexp de mantisa float (modo 2)
    int modo = 0;                     // 0 = double, 1 = floatexp, 2 = floatexp con mantisa float
    std::vector<Bla<CD, double>> bla_d;
    std::vector<Bla<CFE, FE>>    bla_fe;
    std::vector<Bla<CFF, FF>>    bla_ff;
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

// --- Paso de la órbita de referencia en GMP, uno por fórmula ---
struct TemporalesGMP { mpf_t re2, im2; };

// Mandelbrot: Z_0 = 0, Z' = Z^2 + C
static void referencia_inicial_mandelbrot(mpf_t z_re, mpf_t z_im, const mpf_t, const mpf_t) {
    mpf_set_ui(z_re, 0);
    mpf_set_ui(z_im, 0);
}
static void referencia_paso_mandelbrot(mpf_t z_re, mpf_t z_im, const mpf_t c_re, const mpf_t c_im,
                                       TemporalesGMP& t) {
    mpf_mul(t.re2, z_re, z_re);
    mpf_mul(t.im2, z_im, z_im);
    // z_im = 2 z_re z_im + c_im
    mpf_mul(z_im, z_im, z_re);
    mpf_mul_2exp(z_im, z_im, 1);
    mpf_add(z_im, z_im, c_im);
    // z_re = z_re^2 - z_im^2 + c_re
    mpf_sub(z_re, t.re2, t.im2);
    mpf_add(z_re, z_re, c_re);
}

// Calcula la órbita de referencia de la fórmula con `prec_bits` bits. Corta cuando escapa.
// (c_re, c_im): el punto de la referencia (el centro de la vista).
// (k_re, k_im): parámetro de la fórmula (Mandelbrot no lo usa).
// Si *cancelar se pone en 1 mientras calcula, abandona y devuelve NULL.
EXPORT void* pert_crear(int formula, const char* c_re_str, const char* c_im_str,
                        const char* k_re_str, const char* k_im_str, int max_iter, int prec_bits,
                        volatile int* cancelar)
{
    Perturbacion* p = new Perturbacion();
    p->formula = formula;
    p->Z_fe.reserve(max_iter + 1);

    mpf_t c_re, c_im, z_re, z_im;
    TemporalesGMP t;
    mpf_init2(c_re, prec_bits);  mpf_init2(c_im, prec_bits);
    mpf_init2(z_re, prec_bits);  mpf_init2(z_im, prec_bits);
    mpf_init2(t.re2, prec_bits); mpf_init2(t.im2, prec_bits);
    mpf_set_str(c_re, c_re_str, 10);
    mpf_set_str(c_im, c_im_str, 10);
    (void)k_re_str; (void)k_im_str;

    // FORMULAS
    switch (formula) {
    default:
        p->k_d.re = mpf_get_d(c_re);
        p->k_d.im = mpf_get_d(c_im);
        referencia_inicial_mandelbrot(z_re, z_im, c_re, c_im);
    }
    p->Z_fe.push_back(mpf_a_cfe(z_re, z_im));

    bool cancelado = false;
    for (int n = 1; n <= max_iter; ++n) {
        if ((n & 1023) == 0 && cancelar && *cancelar) { cancelado = true; break; }
        switch (formula) {
        default: referencia_paso_mandelbrot(z_re, z_im, c_re, c_im, t);
        }
        CFE z = mpf_a_cfe(z_re, z_im);
        p->Z_fe.push_back(z);
        if (a_double(norma2(z)) > 4.0) break;
    }
    mpf_clears(c_re, c_im, z_re, z_im, t.re2, t.im2, NULL);
    if (cancelado) { delete p; return nullptr; }

    p->largo = (int)p->Z_fe.size();
    p->Z_d.resize(p->largo);
    for (int i = 0; i < p->largo; ++i) {
        const CFE& z = p->Z_fe[i];
        p->Z_d[i].re = ldexp(z.re, z.e);
        p->Z_d[i].im = ldexp(z.im, z.e);
    }
    p->Z_ff.resize(p->largo);
    for (int i = 0; i < p->largo; ++i) p->Z_ff[i] = a_cff(p->Z_fe[i]);
    return p;
}

EXPORT int pert_largo(void* h) { return static_cast<Perturbacion*>(h)->largo; }


EXPORT void pert_parametro(void* h, double* k_re, double* k_im) {
    const Perturbacion* p = static_cast<Perturbacion*>(h);
    *k_re = p->k_d.re; *k_im = p->k_d.im;
}

EXPORT void pert_liberar(void* h) { delete static_cast<Perturbacion*>(h); }

template<class F>
static int preparar_bla(Perturbacion* p, double dc_max_m, int dc_max_e, double eps) {
    if (p->modo == 0)
        return construir_bla<F, CD, double>(p->Z_d.data(), p->largo, ldexp(dc_max_m, dc_max_e),
                                            eps, p->bla_d.data(), p->off, p->cnt);
    if (p->modo == 1)
        return construir_bla<F, CFE, FE>(p->Z_fe.data(), p->largo, fe_normalizar(dc_max_m, dc_max_e),
                                         eps, p->bla_fe.data(), p->off, p->cnt);
    return construir_bla<F, CFF, FF>(p->Z_ff.data(), p->largo, ff_normalizar((float)dc_max_m, dc_max_e),
                                     eps, p->bla_ff.data(), p->off, p->cnt);
}

// Arma la tabla BLA. dc_max = máximo |dc| de la imagen (mantisa, exponente).
// eps <= 0 desactiva el BLA. Devuelve la cantidad de niveles.
EXPORT int pert_preparar_bla(void* h, int modo, double dc_max_m, int dc_max_e, double eps)
{
    Perturbacion* p = static_cast<Perturbacion*>(h);
    p->modo = modo;
    int n0 = p->largo > 2 ? p->largo - 2 : 0;
    // Niveles: n0 + n0/2 + n0/4 + ... < 2 n0
    p->bla_d.clear(); p->bla_fe.clear(); p->bla_ff.clear();
    if (modo == 0)      p->bla_d.resize(2 * (size_t)n0 + 1);
    else if (modo == 1) p->bla_fe.resize(2 * (size_t)n0 + 1);
    else                p->bla_ff.resize(2 * (size_t)n0 + 1);
    // FORMULAS
    switch (p->formula) {
    default: p->niveles = preparar_bla<Mandelbrot>(p, dc_max_m, dc_max_e, eps);
    }
    return p->niveles;
}

// Salidas del render: punteros a matrices width x height (nullptr si no se piden)
struct Salidas { int* iter; float* suave; float* distancia; float* nx; float* ny; };

template<class F, class C, class R>
static long long render(const Perturbacion* p, const C* Z, const Bla<C, R>* tabla, const Grilla& g,
                        int width, int height, int max_iter, int salidas, Salidas out,
                        volatile int* cancelar)
{
    const C k = complejo<C>(p->k_d.re, p->k_d.im);
    long long pasos_total = 0;
    // dynamic: las filas que tocan el interior del conjunto tardan mucho más que las otras
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:pasos_total)
    for (int j = 0; j < height; ++j) {
        if (cancelar && *cancelar) continue;      // un pedido más nuevo lo reemplazó: se abandona
        long long pasos = 0;
        for (int i = 0; i < width; ++i) {
            FE paso;
            C dc = dc_de_grilla<C>(g, i, j, &paso);
            Pixel px = iterar_pixel<F, C, R>(Z, p->largo, tabla, p->off, p->cnt, p->niveles,
                                             dc, k, max_iter, salidas, real_de_fe<R>(paso), &pasos);
            int idx = j * width + i;
            out.iter[idx] = px.n;
            if (out.suave) out.suave[idx] = px.suave;
            if (out.distancia) { out.distancia[idx] = px.distancia; out.nx[idx] = px.nx; out.ny[idx] = px.ny; }
        }
        pasos_total += pasos;
    }
    return pasos_total;
}

template<class F>
static long long render_formula(const Perturbacion* p, const Grilla& g,
                                int width, int height, int max_iter, int salidas, Salidas out,
                                volatile int* cancelar)
{
    if (p->modo == 0)
        return render<F, CD, double>(p, p->Z_d.data(), p->bla_d.data(), g,
                                     width, height, max_iter, salidas, out, cancelar);
    if (p->modo == 1)
        return render<F, CFE, FE>(p, p->Z_fe.data(), p->bla_fe.data(), g,
                                  width, height, max_iter, salidas, out, cancelar);
    return render<F, CFF, FF>(p, p->Z_ff.data(), p->bla_ff.data(), g,
                              width, height, max_iter, salidas, out, cancelar);
}

static long long render_grilla(const Perturbacion* p, const Grilla& g, int width, int height, int max_iter,
                               int salidas, Salidas out, volatile int* cancelar)
{
    // FORMULAS
    switch (p->formula) {
    default: return render_formula<Mandelbrot>(p, g, width, height, max_iter, salidas, out, cancelar);
    }
}

// El píxel (i, j) tiene dc = (bx + i * sx, by + j * sy), todo pasado como (mantisa, exponente).
// salidas: bits SALIDA_SUAVE / SALIDA_DERIVADA; las matrices que no se piden pueden ser NULL.
// Si *cancelar se pone en 1, deja de calcular filas (la salida queda incompleta).
// Devuelve la cantidad total de pasos hechos.
EXPORT long long pert_render(void* h,
    double bx_m, int bx_e, double by_m, int by_e,
    double sx_m, int sx_e, double sy_m, int sy_e,
    double paso_m, int paso_e,
    int width, int height, int max_iter, int salidas,
    int* iter, float* suave, float* distancia, float* nx, float* ny,
    volatile int* cancelar)
{
    const Perturbacion* p = static_cast<Perturbacion*>(h);
    Grilla g = {fe_normalizar(bx_m, bx_e), fe_normalizar(by_m, by_e),
                fe_normalizar(sx_m, sx_e), fe_normalizar(sy_m, sy_e), fe_normalizar(paso_m, paso_e),
                nullptr, nullptr, nullptr, nullptr, fe(0.0)};
    Salidas out = {iter,
                   (salidas & SALIDA_SUAVE) ? suave : nullptr,
                   (salidas & SALIDA_DERIVADA) ? distancia : nullptr, nx, ny};
    return render_grilla(p, g, width, height, max_iter, salidas, out, cancelar);
}

// Lo mismo sobre una grilla polar (la tira del mapa exponencial): el punto de la columna i y la
// fila j tiene dc = radio_j (cos_i, sin_i). width = cantidad de ángulos, height = de radios.
EXPORT long long pert_render_polar(void* h,
    const double* radio_m, const int* radio_e, const double* cos_t, const double* sin_t,
    double dtheta_m, int dtheta_e,
    int width, int height, int max_iter, int salidas,
    int* iter, float* suave, float* distancia, float* nx, float* ny,
    volatile int* cancelar)
{
    const Perturbacion* p = static_cast<Perturbacion*>(h);
    Grilla g = {fe(0.0), fe(0.0), fe(0.0), fe(0.0), fe(0.0),
                radio_m, radio_e, cos_t, sin_t, fe_normalizar(dtheta_m, dtheta_e)};
    Salidas out = {iter,
                   (salidas & SALIDA_SUAVE) ? suave : nullptr,
                   (salidas & SALIDA_DERIVADA) ? distancia : nullptr, nx, ny};
    return render_grilla(p, g, width, height, max_iter, salidas, out, cancelar);
}

// --- Para la GPU: tamaños y copia de la órbita y la tabla del modo actual ---
// Bytes del estado de un píxel a mitad de camino (la GPU lo guarda entre lanzamientos)
EXPORT int pert_tam_estado(int modo) {
    if (modo == 0) return (int)sizeof(EstadoPixel<CD, double>);
    if (modo == 1) return (int)sizeof(EstadoPixel<CFE, FE>);
    return (int)sizeof(EstadoPixel<CFF, FF>);
}

EXPORT long long pert_bytes_orbita(void* h) {
    const Perturbacion* p = static_cast<Perturbacion*>(h);
    return (long long)p->largo * (p->modo == 0 ? sizeof(CD) : p->modo == 1 ? sizeof(CFE) : sizeof(CFF));
}

EXPORT long long pert_bytes_bla(void* h) {
    const Perturbacion* p = static_cast<Perturbacion*>(h);
    long long n = p->niveles ? p->off[p->niveles - 1] + p->cnt[p->niveles - 1] : 0;
    return n * (p->modo == 0 ? sizeof(Bla<CD, double>) : p->modo == 1 ? sizeof(Bla<CFE, FE>) : sizeof(Bla<CFF, FF>));
}

EXPORT int pert_exportar(void* h, void* orbita, void* bla, int* off, int* cnt) {
    const Perturbacion* p = static_cast<Perturbacion*>(h);
    const void* Z = p->modo == 0 ? (const void*)p->Z_d.data() : p->modo == 1 ? (const void*)p->Z_fe.data()
                                                                               : (const void*)p->Z_ff.data();
    const void* tabla = p->modo == 0 ? (const void*)p->bla_d.data() : p->modo == 1 ? (const void*)p->bla_fe.data()
                                                                                   : (const void*)p->bla_ff.data();
    std::memcpy(orbita, Z, pert_bytes_orbita(h));
    std::memcpy(bla, tabla, pert_bytes_bla(h));
    for (int k = 0; k < BLA_MAX_NIVELES; ++k) { off[k] = p->off[k]; cnt[k] = p->cnt[k]; }
    return p->niveles;
}
