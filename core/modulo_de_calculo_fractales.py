import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time 
from OpenGL.GL import *
from .funciones_kernel import *
import os
from functools import wraps
import ctypes
from gui.MandelbrotGUI import Ui_Boundary
from scipy.special import gamma
#from  .coimport mandelbrot
from .backend_cpp import *
import glfw

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
                 tipo_calculo: str, tipo_fractal: str, real: float, imag: float , 
                 ui=None) -> None:
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
        self.x_np = np.linspace(self.xmin, self.xmax, self.width, dtype=np.float64)
        self.y_np = np.linspace(self.ymin, self.ymax, self.height, dtype=np.float64)
        self.ui   = ui
        self.p  = 1.0
        self.nova_m = 1.0
        self.nova_k = 1.0
#        self.x_cp = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
#        self.y_cp = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
        if self.ui is not None:
            self._llenar_combo_fractales()

    def _llenar_combo_fractales(self) -> None:

        # Primero, limpias el comboBox por las dudas:
        self.ui.tipo_fractal_comboBox.clear()
        self.ui.tipo_calculo_comboBox.clear()

        for fractal in FRACTAL_REGISTRY:
            self.ui.tipo_fractal_comboBox.addItem(fractal)

        if FRACTAL_REGISTRY:
            primer_fractal = next(iter(FRACTAL_REGISTRY))
            for calc in FRACTAL_REGISTRY[primer_fractal]:
                self.ui.tipo_calculo_comboBox.addItem(calc)

        self.ui.tipo_fractal_comboBox.currentTextChanged.connect(
            self._on_fractal_cambiado
        )
    
    def _on_fractal_cambiado(self, nombre_fractal: str) -> None:
        """
        Se ejecuta cuando el usuario elige otro fractal;
        recarga el combo de 'cálculos' según lo registrado.
        """
        self.ui.tipo_calculo_comboBox.clear()
        if nombre_fractal in FRACTAL_REGISTRY:
            for calc in FRACTAL_REGISTRY[nombre_fractal]:
                self.ui.tipo_calculo_comboBox.addItem(calc)
    
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

    @staticmethod
    def convertir_formula_compleja(formula: str)-> tuple [str, str]:
        """
        Convierte una fórmula compleja como 'z**2 + C' en dos fórmulas para partes reales e imaginarias,
        usando variables zr, zi, Cr, Ci.
        """
        # Solo soportamos polinomios y suma con C por ahora.
        if formula.strip() == "z**2 + C":
            # (zr + i zi)^2 = (zr^2 - zi^2) + i(2*zr*zi)
            real_expr = "zr**2 - zi**2 + Cr"
            imag_expr = "2 * zr * zi + Ci"
            return real_expr, imag_expr
        elif formula.strip() == "z**2 + 0":  # Julia con constante embebida
            real_expr = "zr**2 - zi**2"
            imag_expr = "2 * zr * zi"
            return real_expr, imag_expr
        else:
            raise NotImplementedError(f"Fórmula no soportada todavía: {formula}")
    
    @staticmethod
    def transformar_expresion(expression: str, variables: str, mask_name :str ="matriz") -> str:
        """
        Aplica una máscara a las variables en la expresión.
        """
        for var in variables:
            expression = expression.replace(var, f"{var}[{mask_name}]")
        return expression

    ##############
    # Mandelbrot #
    ##############
    
    @register_fractal("Mandelbrot", "GPU_Cupy_kernel")
    @medir_tiempo("Mandelbrot GPU")
    def hacer_mandelbrot_gpu(self) -> np.ndarray:
        x = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
        y = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
        X, Y = cp.meshgrid(x, y)
        C = X + 1j * Y  
        C = C.ravel() 
        
        resultado = cp.empty(C.shape, dtype=cp.int32)
        
        try:
            mandelbrot_kernel(C, self.max_iter, resultado)
        except Exception as e:
            print(f"Error executing Julia kernel: {e}")
            return None
            
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

        x = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
        y = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
        X, Y = cp.meshgrid(x, y)
        Z = X + 1j * Y  
        Z = Z.ravel()   

        C = cp.full(Z.shape, complex(self.real, self.imag), dtype=cp.complex128)

        resultado = cp.empty(Z.shape, dtype=cp.int32)

        try:
            julia_kernel(Z, C, self.max_iter, resultado)
        except Exception as e:
            print(f"Error executing Julia kernel: {e}")
            return None

        resultado = resultado.reshape((self.height, self.width))
        resultado_cpu = resultado.get()

        tiempo = time.time() - inicio
        print(f"{self.max_iter} iteraciones")
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

        x = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
        y = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
        X, Y = cp.meshgrid(x, y)
        C = X + 1j * Y  
        Z = cp.zeros_like(C, dtype=cp.complex128)  
        C = C.ravel()
        Z = Z.ravel()

        resultado = cp.empty(C.shape, dtype=cp.int32)

        try:
            burning_kernel(Z, C, self.max_iter, resultado)
        except Exception as e:
            print(f"Error executing Burning Ship kernel: {e}")
            return None

        resultado = resultado.reshape((self.height, self.width))
        resultado_cpu = resultado.get()
        tiempo = time.time() - inicio
        print(f"{self.max_iter} iteraciones")
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

        x = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
        y = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
        X, Y = cp.meshgrid(x, y)
        C = X + 1j * Y  
        C = C.ravel()   
        Z = cp.zeros_like(C, dtype=cp.complex128)  

        resultado = cp.empty(C.shape, dtype=cp.int32)

        try:
            circulo_kernel(Z, C, self.max_iter, resultado)
        except Exception as e:
            print(f"Error executing Circulo kernel: {e}")
            return None

        resultado = resultado.reshape((self.height, self.width))
        resultado_cpu = resultado.get()

        tiempo = time.time() - inicio
        print(f"{self.max_iter} iteraciones")
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
    
    ##################
    # Newton-Raphson #
    ##################

    @register_fractal("Newton-Raphson", "GPU_Cupy_kernel")
    def hacer_newton_gpu(self) -> np.ndarray:
        inicio = time.time()

        x = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
        y = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
        X, Y = cp.meshgrid(x, y)
        C = X + 1j * Y
        C = C.ravel()

        root_index = cp.empty(C.shape, dtype=cp.int32)
        iter_count = cp.empty(C.shape, dtype=cp.int32)

        try:
            newton_kernel(C, self.max_iter, root_index, iter_count)
        except Exception as e:
            print(f"Error ejecutando el kernel de Newton: {e}")
            return None

        root_index = root_index.reshape((self.height, self.width))
        root_index_cpu = root_index.get()
        tiempo = time.time() - inicio
        print(f"{self.max_iter} iteraciones")
        print(f"Tiempo total: {tiempo:.5f} segundos")

        return root_index_cpu

    @register_fractal("Newton-Raphson", "CPU_cpp")
    @medir_tiempo("Newton CPP")
    def hacer_newton_cpp2(self) -> np.ndarray:
        ptr = newton_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        free_newton(ptr)
        return img
    
    
    