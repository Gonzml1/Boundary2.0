import os

import cupy as cp

# ---------------------------------------------------------
# Kernels Originales (Devuelven int32 - iteraciones secas)
# ---------------------------------------------------------

mandelbrot_kernel = cp.ElementwiseKernel(
    in_params='complex128 c, int32 max_iter',
    out_params='int32 result',
    operation="""
        complex<double> z = 0.0;  
        for (int i = 0; i < max_iter; ++i) {
            z = z*z + c;  
            if (real(z)*real(z) + imag(z)*imag(z) > 4.0) {
                result = i;
                return;
            }
        }
        result = max_iter;  
    """,
    name='mandelbrot_kernel'
)

mandelbrot_kernel_optimizado = cp.ElementwiseKernel(
    in_params='complex128 c, int32 max_iter',
    out_params='int32 result',
    operation="""
        double cx = real(c);
        double cy = imag(c);
        
        double q = (cx - 0.25) * (cx - 0.25) + cy * cy;
        if (q * (q + (cx - 0.25)) <= 0.25 * cy * cy || (cx + 1.0) * (cx + 1.0) + cy * cy <= 0.0625) {
            result = max_iter;
            return;
        }

        double x = 0.0;
        double y = 0.0;
        double x2 = 0.0;
        double y2 = 0.0;

        for (int i = 0; i < max_iter; ++i) {
            y = 2.0 * x * y + cy;
            x = x2 - y2 + cx;
            
            x2 = x * x;
            y2 = y * y;
            
            if (x2 + y2 > 4.0) {
                result = i;
                return;
            }
        }
        result = max_iter;  
   """,
    name='mandelbrot_kernel_opt'
)

# ---------------------------------------------------------
# Nuevos Kernels para Coloración Avanzada (Devuelven float64)
# ---------------------------------------------------------

mandelbrot_smooth_kernel = cp.ElementwiseKernel(
    in_params='complex128 c, int32 max_iter',
    out_params='float64 result',
    operation="""
        complex<double> z = 0.0;
        // Se usa un radio de escape mayor (100.0) para evitar errores de precisión en el logaritmo
        for (int i = 0; i < max_iter; ++i) {
            z = z*z + c;
            double z2 = real(z)*real(z) + imag(z)*imag(z);
            if (z2 > 100.0) {
                double log_z = log(z2) / 2.0;
                double nu = log2(log_z);
                result = (double)i + 1.0 - nu;
                return;
            }
        }
        result = (double)max_iter;
    """,
    name='mandelbrot_smooth_kernel'
)

mandelbrot_orbit_trap_kernel = cp.ElementwiseKernel(
    in_params='complex128 c, int32 max_iter',
    out_params='float64 result',
    operation="""
        complex<double> z = 0.0;
        double min_dist = 1e20; // Inicializar con un valor muy grande
        
        for (int i = 0; i < max_iter; ++i) {
            z = z*z + c;
            double current_dist = real(z)*real(z) + imag(z)*imag(z); // Distancia al origen
            if (current_dist < min_dist) {
                min_dist = current_dist;
            }
            if (current_dist > 4.0) {
                result = min_dist; // Mapear color basado en la distancia mínima
                return;
            }
        }
        result = min_dist;
    """,
    name='mandelbrot_orbit_trap_kernel'
)

mandelbrot_distance_estimator_kernel = cp.ElementwiseKernel(
    in_params='complex128 c, int32 max_iter',
    out_params='float64 result',
    operation="""
        complex<double> z = 0.0;
        complex<double> dz = 0.0; // Derivada
        
        for (int i = 0; i < max_iter; ++i) {
            dz = 2.0 * z * dz + 1.0; // Regla de la cadena: dz' = 2*z*dz + dc (donde dc = 1)
            z = z*z + c;
            
            double z2 = real(z)*real(z) + imag(z)*imag(z);
            if (z2 > 100.0) {
                double mod_z = sqrt(z2);
                double mod_dz = sqrt(real(dz)*real(dz) + imag(dz)*imag(dz));
                // Fórmula del estimador de distancia continua
                result = log(z2) * mod_z / mod_dz;
                return;
            }
        }
        result = 0.0; // Interior del fractal
    """,
    name='mandelbrot_distance_estimator_kernel'
)

# ---------------------------------------------------------
# Otros Fractales
# ---------------------------------------------------------

julia_kernel = cp.ElementwiseKernel(
    in_params='complex128 z, complex128 c, int32 max_iter',
    out_params='int32 result',
    operation="""
        complex<double> z_temp = z;  
        for (int i = 0; i < max_iter; ++i) {
            z_temp = z_temp * z_temp + c;  
            if (real(z_temp) * real(z_temp) + imag(z_temp) * imag(z_temp) > 4.0) {
                result = i;
                return;
            }
        }
        result = max_iter;  
    """,
    name='julia_kernel'
)

burning_kernel = cp.ElementwiseKernel(
    in_params='complex128 z, complex128 c, int32 max_iter',
    out_params='int32 result',
    operation="""
        complex<double> z_temp = z;
        for (int i = 0; i < max_iter; ++i) {
            double z_real = fabs(real(z_temp));
            double z_imag = fabs(imag(z_temp));
            z_temp = complex<double>(z_real * z_real - z_imag * z_imag + real(c),
                                     2.0 * z_real * z_imag + imag(c));
            if (real(z_temp) * real(z_temp) + imag(z_temp) * imag(z_temp) > 4.0) {
                result = i;
                return;
            }
        }
        result = max_iter;
    """,
    name='burning_kernel'
)

circulo_kernel = cp.ElementwiseKernel(
    in_params='complex128 z, complex128 c, int32 max_iter',
    out_params='int32 result',
    operation="""
        complex<double> z_temp = z;
        for (int i = 0; i < max_iter; ++i) {
            complex<double> z2 = z_temp * z_temp;
            complex<double> numerator = z2 - 1.00001 * z_temp;
            complex<double> c4 = c * c * c * c;
            z_temp = exp(numerator / c4);
            if (real(z_temp) * real(z_temp) + imag(z_temp) * imag(z_temp) > 4.0) {
                result = i;
                return;
            }
        }
        result = max_iter;
    """,
    name='circulo_kernel'
)

# Perturbación (rebasing + BLA + floatexp). La lógica está en codigos_cpp/perturbacion_core.h,
# compartida con la DLL de CPU: acá solo se envuelve en dos kernels, uno por tipo numérico.
# La órbita de referencia y la tabla BLA se arman en la DLL y se copian a la GPU.
_RUTA_HEADER_PERTURBACION = os.path.join(
    os.path.dirname(__file__), os.pardir, 'codigos_cpp', 'perturbacion_core.h'
)
_KERNELS_PERTURBACION = r"""
// Lanzamiento reanudable: cada hilo es un píxel. En el primer lanzamiento (iniciar = 1) arranca el
// píxel; en los siguientes retoma el estado guardado. Avanza como mucho `max_pasos` pasos: si
// termina escribe el resultado, si no guarda el estado y suma 1 a `pendientes`.
// Se lanza por franjas de `height` filas desde la fila `fila0` (las salidas apuntan al principio
// de la franja). dc se calcula igual que en la CPU (base + fila * paso), así las dos dan lo mismo.
template<class F, class C, class R>
__device__ void render_perturbacion(const C* Z, int largo, const Bla<C, R>* tabla,
                                    const int* off, const int* cnt, int niveles,
                                    double bx_m, int bx_e, double by_m, int by_e,
                                    double sx_m, int sx_e, double sy_m, int sy_e,
                                    double paso_m, int paso_e, double k_re, double k_im,
                                    const double* radio_m, const int* radio_e,
                                    const double* cos_t, const double* sin_t, double dtheta_m, int dtheta_e,
                                    int width, int height, int fila0, int max_iter, int salidas,
                                    EstadoPixel<C, R>* estados, int iniciar, long long max_pasos,
                                    int* iter, float* suave, float* distancia, float* nx, float* ny,
                                    unsigned long long* pasos, unsigned int* pendientes) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    // Grilla rectangular o polar (si radio_m no es nulo): igual que en la CPU
    Grilla g = {fe_normalizar(bx_m, bx_e), fe_normalizar(by_m, by_e), fe_normalizar(sx_m, sx_e),
                fe_normalizar(sy_m, sy_e), fe_normalizar(paso_m, paso_e),
                radio_m, radio_e, cos_t, sin_t, fe_normalizar(dtheta_m, dtheta_e)};
    FE paso_local;
    C dc = dc_de_grilla<C>(g, x, fila0 + y, &paso_local);

    EstadoPixel<C, R> e;
    if (iniciar) {
        iniciar_pixel<F, C, R>(e, Z, dc);
    } else {
        e = estados[idx];
        if (e.fin != PIXEL_EN_CURSO) return;
    }
    long long p = 0;
    avanzar_pixel<F, C, R>(e, Z, largo, tabla, off, cnt, niveles, dc, max_iter, salidas, max_pasos, &p);
    estados[idx] = e;
    atomicAdd(pasos, (unsigned long long)p);
    if (e.fin == PIXEL_EN_CURSO) {
        atomicAdd(pendientes, 1u);
        return;
    }
    Pixel px = terminar_pixel<F, C, R>(e, dc, complejo<C>(k_re, k_im), max_iter, salidas,
                                       real_de_fe<R>(paso_local));
    iter[idx] = px.n;
    if (salidas & SALIDA_SUAVE) suave[idx] = px.suave;
    if (salidas & SALIDA_DERIVADA) { distancia[idx] = px.distancia; nx[idx] = px.nx; ny[idx] = px.ny; }
}

#define ARGS(C, R) const C* Z, int largo, const Bla<C, R>* tabla, const int* off, const int* cnt, \
    int niveles, double bx_m, int bx_e, double by_m, int by_e, double sx_m, int sx_e, \
    double sy_m, int sy_e, double paso_m, int paso_e, double k_re, double k_im, \
    const double* radio_m, const int* radio_e, const double* cos_t, const double* sin_t, \
    double dtheta_m, int dtheta_e, \
    int width, int height, int fila0, int max_iter, int salidas, \
    EstadoPixel<C, R>* estados, int iniciar, long long max_pasos, \
    int* iter, float* suave, float* distancia, float* nx, float* ny, \
    unsigned long long* pasos, unsigned int* pendientes
#define PASAR Z, largo, tabla, off, cnt, niveles, bx_m, bx_e, by_m, by_e, sx_m, sx_e, sy_m, sy_e, \
    paso_m, paso_e, k_re, k_im, radio_m, radio_e, cos_t, sin_t, dtheta_m, dtheta_e, \
    width, height, fila0, max_iter, salidas, estados, iniciar, max_pasos, \
    iter, suave, distancia, nx, ny, pasos, pendientes

// Tres kernels (double / floatexp / floatexp con mantisa float) por fórmula
#define KERNELS_FORMULA(F, nombre) \
    extern "C" __global__ void perturbacion_##nombre##_double(ARGS(CD, double)) \
        { render_perturbacion<F, CD, double>(PASAR); } \
    extern "C" __global__ void perturbacion_##nombre##_floatexp(ARGS(CFE, FE)) \
        { render_perturbacion<F, CFE, FE>(PASAR); } \
    extern "C" __global__ void perturbacion_##nombre##_float(ARGS(CFF, FF)) \
        { render_perturbacion<F, CFF, FF>(PASAR); }

// FORMULAS
KERNELS_FORMULA(Mandelbrot, mandelbrot)
"""

_modulo_perturbacion = None

def kernel_perturbacion(formula: str, modo: int):
    """Kernel de la fórmula para el modo (0 = double, 1 = floatexp, 2 = floatexp con mantisa float).
    Se compila la primera vez."""
    global _modulo_perturbacion
    if _modulo_perturbacion is None:
        with open(_RUTA_HEADER_PERTURBACION, encoding='utf-8') as f:
            header = f.read()
        # --fmad=false: mismo redondeo que la DLL de CPU (compilada con -ffp-contract=off), así
        # CPU y GPU dan resultados idénticos y el video puede repartir cuadros entre las dos
        _modulo_perturbacion = cp.RawModule(code=header + _AYUDAS_GPU + _KERNELS_PERTURBACION,
                                            options=('--fmad=false',))
    return _modulo_perturbacion.get_function(
        f"perturbacion_{formula}_{('double', 'floatexp', 'float')[modo]}"
    )

_AYUDAS_GPU = ""


mandelbrot_espiral_kernel = cp.ElementwiseKernel(
    in_params='complex128 c, int32 max_iter',
    out_params='float64 result',
    operation="""
        complex<double> z = 0.0;
        double suma_espiral = 0.0;
        double frecuencia = 3.0; // Cambia este valor para más o menos brazos en la espiral
        
        for (int i = 0; i < max_iter; ++i) {
            z = z*z + c;
            
            // atan2 obtiene el ángulo (fase) del número complejo
            double angulo = atan2(imag(z), real(z));
            
            // Acumular la onda trigonométrica
            suma_espiral += sin(frecuencia * angulo);
            
            if (real(z)*real(z) + imag(z)*imag(z) > 100.0) {
                // Promediar la suma respecto a las iteraciones para suavizar
                result = suma_espiral / (double)(i + 1);
                return;
            }
        }
        result = suma_espiral / (double)max_iter;
    """,
    name='mandelbrot_espiral_kernel'
)

mandelbrot_triangulo_kernel = cp.ElementwiseKernel(
    in_params='complex128 c, int32 max_iter',
    out_params='float64 result',
    operation="""
        complex<double> z = 0.0;
        complex<double> z_prev = 0.0;
        double suma_triangulo = 0.0;
        
        for (int i = 0; i < max_iter; ++i) {
            z_prev = z;
            z = z*z + c;
            
            double mod_z = sqrt(real(z)*real(z) + imag(z)*imag(z));
            double mod_z_prev = sqrt(real(z_prev)*real(z_prev) + imag(z_prev)*imag(z_prev));
            
            complex<double> diff = z - z_prev;
            double mod_diff = sqrt(real(diff)*real(diff) + imag(diff)*imag(diff));
            
            // Promedio de la desigualdad del triángulo
            if (mod_z + mod_z_prev > 0.0) {
                suma_triangulo += mod_diff / (mod_z + mod_z_prev);
            }
            
            if (mod_z > 10.0) {
                result = suma_triangulo / (double)(i + 1);
                return;
            }
        }
        result = suma_triangulo / (double)max_iter;
    """,
    name='mandelbrot_triangulo_kernel'
)

mandelbrot_gauss_kernel = cp.ElementwiseKernel(
    in_params='complex128 c, int32 max_iter',
    out_params='float64 result',
    operation="""
        complex<double> z = 0.0;
        double suma_gauss = 0.0;
        for (int i = 0; i < max_iter; ++i) {
            z = z*z + c;
            double z2 = real(z)*real(z) + imag(z)*imag(z);
            suma_gauss += exp(-z2);
            if (z2 > 100.0) {
                result = suma_gauss;
                return;
            }
        }
        result = suma_gauss;
    """,
    name='mandelbrot_gauss_kernel'
)

mandelbrot_lyapunov_kernel = cp.ElementwiseKernel(
    in_params='complex128 c, int32 max_iter',
    out_params='float64 result',
    operation="""
        complex<double> z = 0.0;
        double lyapunov = 0.0;
        for (int i = 0; i < max_iter; ++i) {
            z = z*z + c;
            double mod_z = sqrt(real(z)*real(z) + imag(z)*imag(z));
            if (mod_z > 0.0) {
                lyapunov += log(2.0 * mod_z);
            }
            if (mod_z * mod_z > 100.0) {
                result = lyapunov / (double)(i + 1);
                return;
            }
        }
        result = lyapunov / (double)max_iter;
    """,
    name='mandelbrot_lyapunov_kernel'
)


mandelbrot_curvatura_kernel = cp.ElementwiseKernel(
    in_params='complex128 c, int32 max_iter',
    out_params='float64 result',
    operation="""
        complex<double> z = 0.0;
        complex<double> z_prev = 1.0; 
        double suma_angulo = 0.0;
        for (int i = 0; i < max_iter; ++i) {
            z_prev = z;
            z = z*z + c;
            
            double mag_prev = real(z_prev)*real(z_prev) + imag(z_prev)*imag(z_prev);
            if (mag_prev > 0.0001) {
                // Angulo de z / z_prev
                double r_z = real(z), i_z = imag(z);
                double r_p = real(z_prev), i_p = -imag(z_prev); // Conjugado
                double cross_r = r_z * r_p - i_z * i_p;
                double cross_i = r_z * i_p + i_z * r_p;
                suma_angulo += fabs(atan2(cross_i, cross_r));
            }
            
            if (real(z)*real(z) + imag(z)*imag(z) > 100.0) {
                result = suma_angulo / (double)(i + 1);
                return;
            }
        }
        result = suma_angulo / (double)max_iter;
    """,
    name='mandelbrot_curvatura_kernel'
)

mandelbrot_ejes_kernel = cp.ElementwiseKernel(
    in_params='complex128 c, int32 max_iter',
    out_params='float64 result',
    operation="""
        complex<double> z = 0.0;
        double min_dist = 1e20;
        for (int i = 0; i < max_iter; ++i) {
            z = z*z + c;
            double dist_x = fabs(real(z));
            double dist_y = fabs(imag(z));
            double dist = (dist_x < dist_y) ? dist_x : dist_y;
            
            if (dist < min_dist) min_dist = dist;
            
            if (real(z)*real(z) + imag(z)*imag(z) > 100.0) {
                result = min_dist;
                return;
            }
        }
        result = min_dist;
    """,
    name='mandelbrot_ejes_kernel'
)


mandelbrot_tablero_kernel = cp.ElementwiseKernel(
    in_params='complex128 c, int32 max_iter',
    out_params='float64 result',
    operation="""
        complex<double> z = 0.0;
        double suma_tablero = 0.0;
        double frecuencia = 5.0;
        for (int i = 0; i < max_iter; ++i) {
            z = z*z + c;
            
            suma_tablero += sin(real(z) * frecuencia) * sin(imag(z) * frecuencia);
            
            if (real(z)*real(z) + imag(z)*imag(z) > 100.0) {
                result = suma_tablero / (double)(i + 1);
                return;
            }
        }
        result = suma_tablero / (double)max_iter;
    """,
    name='mandelbrot_tablero_kernel'
)