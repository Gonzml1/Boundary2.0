// Mandelbrot por teoría de perturbaciones con "rebasing".
//
// Idea: para un píxel c = C + dc, con C el centro de la vista, escribimos z_n = Z_n + dz_n,
// donde Z_n es la órbita de C (la "referencia"). Restando las dos recurrencias:
//
//     dz_{n+1} = (2 Z_n + dz_n) dz_n + dc
//
// Z se calcula UNA vez en alta precisión (GMP) y se guarda en double, porque |Z| <= 2.
// dz y dc son números chiquitos que el double representa bien aunque el zoom sea 1e-200,
// así que cada píxel itera en double, rápido y en paralelo.
//
// El problema clásico son los "glitches": cuando z_n pasa cerca de 0, dz deja de ser chico
// frente a Z y el double pierde precisión. El rebasing (Zhuoran, 2021) lo arregla así:
// si |Z_m + dz| < |dz|, o se termina la órbita de referencia, se reinicia la referencia
// (dz = Z_m + dz, m = 0). Como Z_0 = 0, eso equivale a seguir desde el valor real de z.
// Con una sola órbita de referencia y sin detección de glitches la imagen sale correcta.

#include <gmp.h>
#include <omp.h>

#define EXPORT extern "C" __declspec(dllexport)

// Calcula Z_0 = 0, Z_{n+1} = Z_n^2 + C con `prec_bits` bits de precisión.
// out_re/out_im deben tener lugar para max_iter + 1 valores.
// Devuelve el largo L de la órbita (Z_0 .. Z_{L-1}); corta cuando Z escapa (|Z| > 2).
EXPORT int calcular_orbita_referencia(
    const char* c_re_str, const char* c_im_str,
    int max_iter, int prec_bits,
    double* out_re, double* out_im)
{
    mpf_t c_re, c_im, z_re, z_im, z_re2, z_im2;
    mpf_init2(c_re, prec_bits);  mpf_init2(c_im, prec_bits);
    mpf_init2(z_re, prec_bits);  mpf_init2(z_im, prec_bits);
    mpf_init2(z_re2, prec_bits); mpf_init2(z_im2, prec_bits);

    mpf_set_str(c_re, c_re_str, 10);
    mpf_set_str(c_im, c_im_str, 10);
    mpf_set_ui(z_re, 0);
    mpf_set_ui(z_im, 0);
    out_re[0] = 0.0;
    out_im[0] = 0.0;

    int largo = max_iter + 1;
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

        out_re[n] = mpf_get_d(z_re);
        out_im[n] = mpf_get_d(z_im);
        if (out_re[n] * out_re[n] + out_im[n] * out_im[n] > 4.0) {
            largo = n + 1;
            break;
        }
    }

    mpf_clears(c_re, c_im, z_re, z_im, z_re2, z_im2, NULL);
    return largo;
}

// Itera cada píxel como delta respecto de la órbita de referencia.
// El píxel (i, j) tiene dc = (dc_x_base + i * step_x, dc_y_base + j * step_y).
// output[j * width + i] = iteración de escape (max_iter si no escapa), igual que el resto de los backends.
EXPORT void calcular_deltas_cpp(
    const double* Z_re, const double* Z_im, int largo_orbita,
    double dc_x_base, double dc_y_base,
    double step_x, double step_y,
    int width, int height, int max_iter,
    int* output)
{
    // dynamic: las filas que tocan el interior del conjunto tardan mucho más que las otras
    #pragma omp parallel for schedule(dynamic, 1)
    for (int j = 0; j < height; ++j) {
        const double dc_im = dc_y_base + j * step_y;
        for (int i = 0; i < width; ++i) {
            const double dc_re = dc_x_base + i * step_x;

            double dz_re = 0.0, dz_im = 0.0;
            int m = 0;  // posición en la órbita de referencia
            int n = 0;  // iteración real del píxel
            for (; n < max_iter; ++n) {
                // dz = (2 Z_m + dz) dz + dc
                const double a_re = 2.0 * Z_re[m] + dz_re;
                const double a_im = 2.0 * Z_im[m] + dz_im;
                const double t = a_re * dz_re - a_im * dz_im + dc_re;
                dz_im = a_re * dz_im + a_im * dz_re + dc_im;
                dz_re = t;
                ++m;

                const double z_re = Z_re[m] + dz_re;
                const double z_im = Z_im[m] + dz_im;
                const double z2 = z_re * z_re + z_im * z_im;
                if (z2 > 4.0) break;

                // Rebasing
                if (z2 < dz_re * dz_re + dz_im * dz_im || m == largo_orbita - 1) {
                    dz_re = z_re;
                    dz_im = z_im;
                    m = 0;
                }
            }
            output[j * width + i] = n;
        }
    }
}
