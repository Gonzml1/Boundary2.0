#include <vector>
#include <cstdlib>
#include <omp.h>
#include <boost/multiprecision/cpp_dec_float.hpp>

using mpf = boost::multiprecision::cpp_dec_float_50;

// Struct para números complejos en alta precisión
struct mp_complex {
    mpf re, im;
    mp_complex(mpf r = 0, mpf i_ = 0) : re(r), im(i_) {}
    mp_complex operator+(const mp_complex& o) const { return {re + o.re, im + o.im}; }
    mp_complex operator-(const mp_complex& o) const { return {re - o.re, im - o.im}; }
    mp_complex operator*(const mp_complex& o) const {
        return {re * o.re - im * o.im, re * o.im + im * o.re};
    }
    mpf abs2() const { return re * re + im * im; }
};

extern "C" {
    // Genera fractal usando aritmética arbitraria pura (lento)
    __declspec(dllexport)
    int* pertubacion(double xmin, double xmax, double ymin, double ymax,
                         int width, int height, int max_iter) {
        int* M = (int*)std::malloc(sizeof(int) * width * height);
        if (!M) return nullptr;

        mpf dx = (mpf(xmax) - xmin) / (width - 1);
        mpf dy = (mpf(ymax) - ymin) / (height - 1);

        #pragma omp parallel for collapse(2)
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                mpf x0 = xmin + mpf(i) * dx;
                mpf y0 = ymin + mpf(j) * dy;
                mp_complex z(0, 0), c(x0, y0);
                int n = 0;
                while (n < max_iter && z.abs2() <= 4) {
                    z = z * z + c;
                    ++n;
                }
                M[j * width + i] = n;
            }
        }
        return M;
    }

    // Genera fractal usando método de perturbación:
    // 1) Calcular órbita de referencia en alta precisión para un punto central.
    // 2) Para cada pixel, propagar la pequeña desviación con precisión doble.
    __declspec(dllexport)
    int* mandelbrot_perturb(double xc, double yc,
                              double xmin, double xmax, double ymin, double ymax,
                              int width, int height, int max_iter) {
        int* M = (int*)std::malloc(sizeof(int) * width * height);
        if (!M) return nullptr;

        // 1) Órbita de referencia en alta precisión
        std::vector<mp_complex> ref_orbit(max_iter + 1);
        mp_complex cref(xc, yc), zref(0, 0);
        ref_orbit[0] = zref;
        int ref_escape = max_iter;
        for (int k = 1; k <= max_iter; ++k) {
            zref = zref * zref + cref;
            ref_orbit[k] = zref;
            if (zref.abs2() > 4) { ref_escape = k; break; }
        }

        double dx = (xmax - xmin) / (width - 1);
        double dy = (ymax - ymin) / (height - 1);

        // 2) Pixel por pixel: propagación de la desviación
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                // delta inicial en doble precisión
                double x0 = xmin + i * dx;
                double y0 = ymin + j * dy;
                double dr = x0 - xc;
                double di = y0 - yc;
                double zr = 0.0, zi = 0.0;
                int n = 0;
                for (n = 1; n < max_iter; ++n) {
                    // Derivada de f(z) = z^2 + c en zref: f'(z) = 2*zref
                    double d_re = 2.0 * ref_orbit[n-1].re.convert_to<double>();
                    double d_im = 2.0 * ref_orbit[n-1].im.convert_to<double>();
                    // Propagar error linealmente: δ_{n} ≈ f'(z_{ref,n-1}) * δ_{n-1}
                    double dr_new = zr * dr - zi * di;
                    double di_new = zr * di + zi * dr;
                    dr = d_re * dr - d_im * di + dr_new;
                    di = d_re * di + d_im * dr + di_new;
                    
                    // Evolucionar aproximado de z
                    zr = ref_orbit[n].re.convert_to<double>() + dr;
                    zi = ref_orbit[n].im.convert_to<double>() + di;
                    
                    if (zr*zr + zi*zi > 4.0) break;
                }
                M[j * width + i] = n;
            }
        }
        return M;
    }

    __declspec(dllexport) void free_pertubacion(int* M) {
        std::free(M);
    }
}
