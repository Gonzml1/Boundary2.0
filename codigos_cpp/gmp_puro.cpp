#include <gmp.h>
#include <stdlib.h>

extern "C" {
    __declspec(dllexport) void calcular_mandelbrot_gmp_puro(
        const char* xmin_str, const char* xmax_str, 
        const char* ymin_str, const char* ymax_str, 
        int width, int height, int max_iter, int* output) 
    {
        mpf_t xmin, xmax, ymin, ymax, step_x, step_y;
        mpf_init2(xmin, 256); mpf_init2(xmax, 256);
        mpf_init2(ymin, 256); mpf_init2(ymax, 256);
        mpf_init2(step_x, 256); mpf_init2(step_y, 256);

        mpf_set_str(xmin, xmin_str, 10);
        mpf_set_str(xmax, xmax_str, 10);
        mpf_set_str(ymin, ymin_str, 10);
        mpf_set_str(ymax, ymax_str, 10);

        // step_x = (xmax - xmin) / width
        mpf_sub(step_x, xmax, xmin);
        mpf_div_ui(step_x, step_x, width);

        // step_y = (ymax - ymin) / height
        mpf_sub(step_y, ymax, ymin);
        mpf_div_ui(step_y, step_y, height);

        mpf_t c_re, c_im, z_re, z_im, z_re2, z_im2, temp;
        mpf_init2(c_re, 256); mpf_init2(c_im, 256);
        mpf_init2(z_re, 256); mpf_init2(z_im, 256);
        mpf_init2(z_re2, 256); mpf_init2(z_im2, 256); mpf_init2(temp, 256);

        for (int y = 0; y < height; y++) {
            // c_im = ymin + y * step_y
            mpf_mul_ui(c_im, step_y, y);
            mpf_add(c_im, c_im, ymin);

            for (int x = 0; x < width; x++) {
                // c_re = xmin + x * step_x
                mpf_mul_ui(c_re, step_x, x);
                mpf_add(c_re, c_re, xmin);

                mpf_set_d(z_re, 0.0);
                mpf_set_d(z_im, 0.0);

                int iter;
                for (iter = 0; iter < max_iter; iter++) {
                    mpf_mul(z_re2, z_re, z_re);
                    mpf_mul(z_im2, z_im, z_im);
                    
                    mpf_add(temp, z_re2, z_im2);
                    if (mpf_cmp_d(temp, 4.0) > 0) break;

                    mpf_mul(z_im, z_im, z_re);
                    mpf_mul_ui(z_im, z_im, 2);
                    mpf_add(z_im, z_im, c_im);

                    mpf_sub(z_re, z_re2, z_im2);
                    mpf_add(z_re, z_re, c_re);
                }
                output[y * width + x] = iter;
            }
        }

        mpf_clears(xmin, xmax, ymin, ymax, step_x, step_y, c_re, c_im, z_re, z_im, z_re2, z_im2, temp, NULL);
    }
}