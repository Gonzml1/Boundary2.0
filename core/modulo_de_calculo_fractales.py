import cupy as cp
import numpy as np
import time 
from OpenGL.GL import *
from .funciones_kernel import *
from functools import wraps
#from  .coimport mandelbrot
from .backend_cpp import *
from decimal import Decimal
from core.funciones_kernel import mandelbrot_perturbacion_kernel

ruta_dll_pert = os.path.join(os.path.dirname(__file__), '..', 'codigos_cpp', 'perturbacion.dll')
try:
    lib_pert = ctypes.CDLL(ruta_dll_pert)
    lib_pert.calcular_orbita_referencia.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    ]
    lib_pert.calcular_orbita_referencia.restype = None
except Exception as e:
    print(f"Advertencia: No se pudo cargar perturbacion.dll - {e}")

ruta_dll_gmp = os.path.join(os.path.dirname(__file__), '..', 'codigos_cpp', 'gmp_puro.dll')
try:
    lib_gmp_puro = ctypes.CDLL(ruta_dll_gmp)
    lib_gmp_puro.calcular_mandelbrot_gmp_puro.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS')
    ]
    lib_gmp_puro.calcular_mandelbrot_gmp_puro.restype = None
except Exception as e:
    print(f"Advertencia: No se pudo cargar gmp_puro.dll - {e}")

# cp.exp((z[matriz]**2 - 1.00001*z[matriz]) / C[matriz]**4) 
# z[matriz] = z[matriz]**2 + C[matriz]    

FRACTAL_REGISTRY: dict[str, dict[str, callable]] = {}

def register_fractal(fractal: str, calc: str) -> callable:
    """
    Decorador: registra la función en FRACTAL_REGISTRY bajo
    FRACTAL_REGISTRY[fractal][calc] = fn
    """
    def deco(fn) -> callable:
        # Si no existe la clave 'fractal', la creamos:
        if fractal not in FRACTAL_REGISTRY:
            FRACTAL_REGISTRY[fractal] = {}
        # Asociamos el nombre de cálculo a la función concreta:
        FRACTAL_REGISTRY[fractal][calc] = fn
        return fn
    return deco

#para añadir en un futuro
class calculos_mandelbrot:
    def __init__(self, xmin: float, xmax: float , ymin: float, ymax: float, 
                 width: int, height: int, max_iter: int, 
                 tipo_calculo: str, tipo_fractal: str) -> None:
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.tipo_calculo = tipo_calculo
        self.tipo_fractal = tipo_fractal
        self.x_np = np.linspace(self.xmin, self.xmax, self.width, dtype=np.float64)
        self.y_np = np.linspace(self.ymin, self.ymax, self.height, dtype=np.float64)
        

    @staticmethod
    def medir_tiempo(nombre) -> callable:
        """
        Decorador para medir el tiempo de ejecución de una función.
        """
        def decorador(func) -> callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> any:
                inicio = time.time()
                resultado = func(*args, **kwargs)
                fin = time.time()
                print(f"⏱️ Tiempo de ejecución de '{nombre}': {fin - inicio:.5f} segundos")
                return resultado
            return wrapper
        return decorador
        
    def actualizar_fractal(self,
            xmin: float,  xmax: float,
            ymin: float,  ymax: float,
            width: int,   height: int,
            max_iter: int,tipo_calculo: str, 
            tipo_fractal: str,
            real: float, imag: float) -> None:
        """
        Actualiza los parámetros del fractal.
        """
        
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.tipo_calculo = tipo_calculo
        self.tipo_fractal = tipo_fractal
        self.real = real
        self.imag = imag
        return None

    def calcular_fractal(self) -> np.ndarray:
        if self.tipo_fractal in FRACTAL_REGISTRY:
            if self.tipo_calculo in FRACTAL_REGISTRY[self.tipo_fractal]:
                M = FRACTAL_REGISTRY[self.tipo_fractal][self.tipo_calculo](self)
                return M
            else:
                raise ValueError(f"Tipo de cálculo '{self.tipo_calculo}' no soportado para el fractal '{self.tipo_fractal}'.")

    def _generar_malla_compleja_gpu(self):
        x = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
        y = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
        X, Y = cp.meshgrid(x, y)
        return (X + 1j * Y).ravel()
    


    ##############
    # Mandelbrot #
    ##############
    
    #@register_fractal("Mandelbrot", "GPU_Cupy_kernel")
    @medir_tiempo("Mandelbrot GPU")
    def hacer_mandelbrot_gpu(self) -> np.ndarray:
        C = self._generar_malla_compleja_gpu()
        C = C.ravel() 
        
        resultado = cp.empty(C.shape, dtype=cp.int32)
        
        try:
            mandelbrot_kernel(C, self.max_iter, resultado)
        except Exception as e:
            raise RuntimeError(f"Fallo en Kernel GPU: {e}")
            
        resultado = resultado.reshape((self.height, self.width))
        resultado_cpu = resultado.get()

        return resultado_cpu
    
    @register_fractal("Mandelbrot", "GPU_Cupy_kernel_optimizado")
    @medir_tiempo("Mandelbrot GPU optimizado")
    def hacer_mandelbrot_gpu(self) -> np.ndarray:
        C = self._generar_malla_compleja_gpu()
        C = C.ravel() 
        
        resultado = cp.empty(C.shape, dtype=cp.int32)
        
        try:
            mandelbrot_kernel_optimizado(C, self.max_iter, resultado)
        except Exception as e:
            raise RuntimeError(f"Fallo en Kernel GPU: {e}")
            
        resultado = resultado.reshape((self.height, self.width))
        resultado_cpu = resultado.get()

        return resultado_cpu
    
    @register_fractal("Mandelbrot", "CPU_cpp")
    @medir_tiempo("Mandelbrot CPP")
    def hacer_mandelbrot_cpp(self) -> np.ndarray:
        ptr = mandelbrot_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        free_mandelbrot(ptr)
        return img
    
    @register_fractal("Mandelbrot", "perturbacion")
    @medir_tiempo("Mandelrbot pertubation")
    def hacer_mandelbrot_perturbacion(self):
        # 1. Calcular el centro matemático con precisión absoluta
        c_re = (Decimal(self.xmin) + Decimal(self.xmax)) / Decimal('2')
        c_im = (Decimal(self.ymin) + Decimal(self.ymax)) / Decimal('2')

        # Convertir a strings de bytes para C++
        c_re_str = str(c_re).encode('utf-8')
        c_im_str = str(c_im).encode('utf-8')

        # 2. Preparar memoria para recibir la órbita de referencia
        Z_ref_re = np.zeros(self.max_iter, dtype=np.float64)
        Z_ref_im = np.zeros(self.max_iter, dtype=np.float64)

        # 3. Llamar a C++ (GMP) para calcular el centro
        lib_pert.calcular_orbita_referencia(c_re_str, c_im_str, self.max_iter, Z_ref_re, Z_ref_im)

        # Subir la órbita a la memoria de la placa de video
        d_Z_ref_re = cp.asarray(Z_ref_re)
        d_Z_ref_im = cp.asarray(Z_ref_im)

        # 4. Calcular el Delta base (Esquina superior izquierda vs Centro)
        delta_c_x_base = float(Decimal(self.xmin) - c_re)
        delta_c_y_base = float(Decimal(self.ymin) - c_im)

        # Calcular el salto por cada pixel (Step)
        step_x = float((Decimal(self.xmax) - Decimal(self.xmin)) / Decimal(self.width))
        step_y = float((Decimal(self.ymax) - Decimal(self.ymin)) / Decimal(self.height))

        # 5. Preparar la salida de la GPU
        output = cp.zeros((self.height, self.width), dtype=cp.int32)

        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(self.width / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(self.height / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # 6. Lanzar el kernel
        mandelbrot_perturbacion_kernel(
            blockspergrid, threadsperblock,
            (d_Z_ref_re, d_Z_ref_im,
            np.float64(delta_c_x_base), np.float64(delta_c_y_base),
            np.float64(step_x), np.float64(step_y),
            output, np.int32(self.max_iter), np.int32(self.width), np.int32(self.height))
        )

        # 7. Devolver a la CPU para colorear
        return output.get()
    
    @register_fractal("Mandelbrot", "gmp")
    @medir_tiempo("Gmp")
    def hacer_mandelbrot_gmp_puro(self):
        # Enviar los límites completos como strings para no perder precisión
        xmin_str = str(self.xmin).encode('utf-8')
        xmax_str = str(self.xmax).encode('utf-8')
        ymin_str = str(self.ymin).encode('utf-8')
        ymax_str = str(self.ymax).encode('utf-8')

        # Crear la matriz destino en la CPU
        output = np.zeros((self.height, self.width), dtype=np.int32)

        # Llamar a la DLL
        lib_gmp_puro.calcular_mandelbrot_gmp_puro(
            xmin_str, xmax_str, ymin_str, ymax_str,
            self.width, self.height, self.max_iter, output
        )

        return output
    #########
    # Julia #
    #########

    @register_fractal("Julia", "GPU_Cupy_kernel")
    def hacer_julia_gpu(self) -> np.ndarray:
        inicio = time.time()

        Z = self._generar_malla_compleja_gpu()
        Z = Z.ravel()   

        C = cp.full(Z.shape, complex(self.real, self.imag), dtype=cp.complex128)

        resultado = cp.empty(Z.shape, dtype=cp.int32)

        try:
            julia_kernel(Z, C, self.max_iter, resultado)
        except Exception as e:
            raise RuntimeError(f"Fallo en Kernel GPU: {e}")

        resultado = resultado.reshape((self.height, self.width))
        resultado_cpu = resultado.get()

        tiempo = time.time() - inicio
        print(f"Tiempo total: {tiempo:.5f} segundos")

        return resultado_cpu
    
    @register_fractal("Julia", "CPU_cpp")
    @medir_tiempo("Julia CPP")
    def hacer_julia_cpp(self) -> np.ndarray:
        ptr = julia_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter,
            self.real,   
            self.imag    
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        free_julia(ptr)
        return img

    ################
    # Burning Ship #
    ################
    
    @register_fractal("Burning Ship", "GPU_Cupy_kernel")
    def hacer_burning_gpu(self) -> np.ndarray:
        inicio = time.time()

        C = self._generar_malla_compleja_gpu()
        Z = cp.zeros_like(C, dtype=cp.complex128)  
        C = C.ravel()
        Z = Z.ravel()

        resultado = cp.empty(C.shape, dtype=cp.int32)

        try:
            burning_kernel(Z, C, self.max_iter, resultado)
        except Exception as e:
            raise RuntimeError(f"Fallo en Kernel GPU: {e}")
            

        resultado = resultado.reshape((self.height, self.width))
        resultado_cpu = resultado.get()
        tiempo = time.time() - inicio
        print(f"Tiempo total: {tiempo:.5f} segundos")

        return resultado_cpu
    
    @register_fractal("Burning Ship", "CPU_cpp")
    @medir_tiempo("Burning Ship CPP")
    def hacer_burning_cpp2(self) -> np.ndarray:
        ptr = burning_ship_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        free_burning_ship(ptr)
        return img
    
    ###########
    # Circulo #
    ###########
    
    @register_fractal("Circulo", "GPU_Cupy_kernel")
    def hacer_circulo_gpu(self) -> np.ndarray:
        inicio = time.time()

        C = self._generar_malla_compleja_gpu()
        C = C.ravel()   
        Z = cp.zeros_like(C, dtype=cp.complex128)  

        resultado = cp.empty(C.shape, dtype=cp.int32)

        try:
            circulo_kernel(Z, C, self.max_iter, resultado)
        except Exception as e:
            raise RuntimeError(f"Fallo en Kernel GPU: {e}")

        resultado = resultado.reshape((self.height, self.width))
        resultado_cpu = resultado.get()

        tiempo = time.time() - inicio
        print(f"Tiempo total: {tiempo:.5f} segundos")

        return resultado_cpu

    @register_fractal("Circulo", "CPU_cpp")
    @medir_tiempo("Circulo CPP")
    def hacer_circulo_cpp2(self) -> np.ndarray:
        ptr = circulo_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        free_circulo(ptr)
        return img

    
    
    