#include <gmp.h>
#include <stdlib.h>
#include <omp.h>

extern "C" {
    __declspec(dllexport) void calcular_orbita_referencia(
        const char* c_re_str, const char* c_im_str, 
        int max_iter, double* out_re, double* out_im) 
    {
        mpf_t c_re, c_im, z_re, z_im, z_re2, z_im2, temp;
        mpf_init2(c_re, 300); 
        mpf_init2(c_im, 300);
        mpf_init2(z_re, 300); mpf_init2(z_im, 300);
        mpf_init2(z_re2, 300); mpf_init2(z_im2, 300); mpf_init2(temp, 300);

        mpf_t old_z_re, old_z_im, diff_re, diff_im, eps;
        mpf_init2(old_z_re, 300); mpf_init2(old_z_im, 300);
        mpf_init2(diff_re, 300); mpf_init2(diff_im, 300);
        mpf_init2(eps, 300);
        
        mpf_set_str(eps, "1e-25", 10);

        mpf_set_str(c_re, c_re_str, 10);
        mpf_set_str(c_im, c_im_str, 10);
        mpf_set_d(z_re, 0.0);
        mpf_set_d(z_im, 0.0);

        int period_check_interval = 1;
        int steps_to_check = 1;

        for (int i = 0; i < max_iter; i++) {
            out_re[i] = mpf_get_d(z_re);
            out_im[i] = mpf_get_d(z_im);

            mpf_mul(z_re2, z_re, z_re);
            mpf_mul(z_im2, z_im, z_im);
            
            mpf_add(temp, z_re2, z_im2);
            if (mpf_cmp_d(temp, 4.0) > 0) {
                for (int j = i + 1; j < max_iter; j++) {
                    out_re[j] = out_re[i];
                    out_im[j] = out_im[i];
                }
                break;
            }

            if (i > 0) {
                mpf_sub(diff_re, z_re, old_z_re);
                mpf_abs(diff_re, diff_re);
                mpf_sub(diff_im, z_im, old_z_im);
                mpf_abs(diff_im, diff_im);

                if (mpf_cmp(diff_re, eps) < 0 && mpf_cmp(diff_im, eps) < 0) {
                    for (int j = i + 1; j < max_iter; j++) {
                        out_re[j] = out_re[i];
                        out_im[j] = out_im[i];
                    }
                    break;
                }
            }

            if (steps_to_check == i) {
                mpf_set(old_z_re, z_re);
                mpf_set(old_z_im, z_im);
                period_check_interval *= 2;
                steps_to_check += period_check_interval;
            }

            mpf_mul(z_im, z_im, z_re);
            mpf_mul_ui(z_im, z_im, 2);
            mpf_add(z_im, z_im, c_im);

            mpf_sub(z_re, z_re2, z_im2);
            mpf_add(z_re, z_re, c_re);
        }

        mpf_clears(c_re, c_im, z_re, z_im, z_re2, z_im2, temp, NULL);
        mpf_clears(old_z_re, old_z_im, diff_re, diff_im, eps, NULL);
    }

    __declspec(dllexport) void calcular_deltas_cpp(
            const double* Z_ref_re, const double* Z_ref_im,
            double dc_x_base, double dc_y_base,
            double step_x, double step_y,
            int width, int height, int max_iter,
            int* output) 
    {
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                // Se calcula Delta C sin usar las coordenadas absolutas
                double dc_re = dc_x_base + i * step_x;
                double dc_im = dc_y_base + j * step_y;

                double dz_re = 0.0;
                double dz_im = 0.0;

                int n = 0;
                while (n < max_iter) {
                    double z_ref_re = Z_ref_re[n];
                    double z_ref_im = Z_ref_im[n];

                    double z_tot_re = z_ref_re + dz_re;
                    double z_tot_im = z_ref_im + dz_im;

                    if (z_tot_re * z_tot_re + z_tot_im * z_tot_im > 4.0) {
                        break;
                    }

                    double dz_re_sq = dz_re * dz_re - dz_im * dz_im;
                    double dz_im_sq = 2.0 * dz_re * dz_im;

                    double dz_new_re = 2.0 * (z_ref_re * dz_re - z_ref_im * dz_im) + dz_re_sq + dc_re;
                    double dz_new_im = 2.0 * (z_ref_re * dz_im + z_ref_im * dz_re) + dz_im_sq + dc_im;

                    dz_re = dz_new_re;
                    dz_im = dz_new_im;
                    n++;
                }
                output[j * width + i] = n;
            }
        }
    }
}