import cupy as cp

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

#mandelbrot_kernel = cp.ElementwiseKernel(
#    in_params='complex128 c, int32 max_iter',
#    out_params='int32 result',
#    operation="""
#        double cx = real(c);
#        double cy = imag(c);
#        
#        // 1. OPTIMIZACIÓN MATEMÁTICA: Cardioid y Period-2 Bulb Check
#        // Si el punto está en las zonas más grandes del fractal, 
#        // sabemos que NUNCA va a escapar. Nos ahorramos el bucle completo.
#        double q = (cx - 0.25) * (cx - 0.25) + cy * cy;
#        if (q * (q + (cx - 0.25)) <= 0.25 * cy * cy || (cx + 1.0) * (cx + 1.0) + cy * cy <= 0.0625) {
#            result = max_iter;
#            return;
#        }
#
#        double x = 0.0;
#        double y = 0.0;
#        double x2 = 0.0;
#        double y2 = 0.0;
#
#        // 2. DESCOMPOSICIÓN ALGEBRAICA
#        // En lugar de usar la estructura 'complex', separamos reales e imaginarios.
#        // Reutilizamos x2 e y2 para no calcularlos dos veces (una para la suma y otra para el límite).
#        for (int i = 0; i < max_iter; ++i) {
#            y = 2.0 * x * y + cy;
#            x = x2 - y2 + cx;
#            
#            x2 = x * x;
#            y2 = y * y;
#            
#            // Verificación de escape
#            if (x2 + y2 > 4.0) {
#                result = i;
#                return;
#            }
#        }
#        result = max_iter;  
#    """,
#    name='mandelbrot_kernel_opt'
#)

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
            // z = exp((z^2 - 1.00001*z) / c^4)
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

newton_kernel = cp.ElementwiseKernel(
        in_params='complex128 c, int32 max_iter',
        out_params='int32 root_index, int32 iter_count',
        operation="""
            complex<double> z = c;
            complex<double> raices[3] = {
                complex<double>(1.0, 0.0),
                complex<double>(-0.5, 0.8660254),
                complex<double>(-0.5, -0.8660254)
            };
            double tolerancia = 1e-6;
            root_index = 0;
            iter_count = 0;

            for (int i = 0; i < max_iter; ++i) {
                complex<double> fz = z*z*z - 1.0;
                complex<double> dfz = 3.0*z*z;
                if (abs(dfz) < 1e-10) {  
                    root_index = 0;
                    iter_count = i;
                    return;
                }
                z = z - fz / dfz;

                for (int j = 0; j < 3; ++j) {
                    if (abs(z - raices[j]) < tolerancia) {
                        root_index = j + 1;
                        iter_count = i + 1;
                        return;
                    }
                }
            }
            root_index = 0;  
            iter_count = max_iter;
        """,
        name='newton_kernel'
    )

