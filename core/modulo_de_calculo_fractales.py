import cupy as cp
import numpy as np
import time 
from OpenGL.GL import *
from .funciones_kernel import *
from functools import wraps
#from  .coimport mandelbrot
from .backend_cpp import *

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
                 width: int, height: int, max_iter: int, formula: str, 
                 tipo_calculo: str, tipo_fractal: str) -> None:
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.formula = formula
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
            max_iter: int, formula: str,
            tipo_calculo: str, tipo_fractal: str,
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
        self.formula = formula
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

    
    
    