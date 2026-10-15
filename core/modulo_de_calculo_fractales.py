import time
import logging
import decimal
from decimal import Decimal
from functools import wraps
from typing import NamedTuple

import numpy as np

from . import backend_cpp as cpp
from .backend_cpp import BACKENDS_DISPONIBLES

log = logging.getLogger(__name__)

# Precisión de las coordenadas (dígitos decimales). 320 dígitos alcanzan para zooms
# de hasta ~1e-290, que es el límite de los deltas en double de la perturbación.
# DefaultContext es la plantilla de los hilos nuevos (p. ej. el hilo del video).
PRECISION_DECIMAL = 320
decimal.DefaultContext.prec = PRECISION_DECIMAL
decimal.getcontext().prec = PRECISION_DECIMAL

# --- GPU opcional: si no hay CuPy o no hay placa CUDA, se registran solo los backends de CPU ---
try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("no se encontró ninguna GPU CUDA")
    from .funciones_kernel import *
    BACKENDS_DISPONIBLES["gpu"] = True
except Exception as e:
    log.warning("GPU no disponible, se omiten los backends CuPy: %s", e)
    BACKENDS_DISPONIBLES["gpu"] = False


FRACTAL_REGISTRY: dict[str, dict[str, callable]] = {}

def register_fractal(fractal: str, calc: str, requiere: str | tuple[str, ...] = ()) -> callable:
    """Registra un método de cálculo. `requiere` son los backends ("gpu", "mandelbrot.dll", ...)
    que tienen que estar disponibles; si falta alguno, el método no aparece en la UI."""
    requisitos = (requiere,) if isinstance(requiere, str) else requiere
    def deco(fn) -> callable:
        faltantes = [r for r in requisitos if not BACKENDS_DISPONIBLES.get(r, False)]
        if faltantes:
            log.info("Se omite '%s / %s': falta %s", fractal, calc, ", ".join(faltantes))
            return fn
        FRACTAL_REGISTRY.setdefault(fractal, {})[calc] = fn
        return fn
    return deco


def _a_decimal(valor) -> Decimal:
    if isinstance(valor, Decimal):
        return valor
    return Decimal(str(valor))


def _bits_necesarios(paso: Decimal) -> int:
    """Bits de mantisa para que GMP distinga píxeles separados por `paso` (con margen)."""
    if paso == 0:
        return 128
    digitos = max(0, -paso.adjusted())
    return max(128, int(digitos * 3.33) + 64)


class OrbitaReferencia(NamedTuple):
    Z_re: np.ndarray
    Z_im: np.ndarray
    largo: int
    dc_x_base: float
    dc_y_base: float
    step_x: float
    step_y: float


class ProcesadorColores:
    """Metodos vectorizados para procesar la matriz de iteraciones (np.ndarray)"""
    
    @staticmethod
    def invertir_indice(matriz_iteraciones: np.ndarray, max_iter: int) -> np.ndarray:
        return max_iter - matriz_iteraciones

    @staticmethod
    def ecualizar_histograma(matriz_iteraciones: np.ndarray) -> np.ndarray:
        # Distribuye los colores uniformemente basándose en la frecuencia
        histograma, bins = np.histogram(matriz_iteraciones.flatten(), bins=256, density=True)
        cdf = histograma.cumsum()
        cdf_normalizado = cdf * 255 / cdf[-1]
        matriz_suave = np.interp(matriz_iteraciones.flatten(), bins[:-1], cdf_normalizado)
        return matriz_suave.reshape(matriz_iteraciones.shape)

    @staticmethod
    def bandas_ciclicas(matriz_iteraciones: np.ndarray, amplitud_banda: int = 20) -> np.ndarray:
        # Genera anillos de color basados en un módulo
        return matriz_iteraciones % amplitud_banda

    @staticmethod
    def mapeo_logaritmico(matriz_iteraciones: np.ndarray) -> np.ndarray:
        # Suaviza el salto brusco en zonas de muchas iteraciones
        return np.log(matriz_iteraciones + 1)

    @staticmethod
    def interpolacion_trigonometrica(matriz_iteraciones: np.ndarray, max_iter: int) -> np.ndarray:
        # Transiciones continuas oscilantes para evitar cortes secos de int
        return (np.cos(matriz_iteraciones / max_iter * np.pi) + 1) * 127.5


class calculos_mandelbrot:
    def __init__(self, xmin: float, xmax: float , ymin: float, ymax: float, 
                 width: int, height: int, max_iter: int, 
                 tipo_calculo: str, tipo_fractal: str) -> None:
        self.real = 0.0
        self.imag = 0.0
        self._cache_orbita = None
        self._fijar_limites(xmin, xmax, ymin, ymax)
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.tipo_calculo = tipo_calculo
        self.tipo_fractal = tipo_fractal

    def _fijar_limites(self, xmin, xmax, ymin, ymax) -> None:
        """Guarda los límites en Decimal (para perturbación y GMP) y en float (para el resto)."""
        self.xmin_dec, self.xmax_dec = _a_decimal(xmin), _a_decimal(xmax)
        self.ymin_dec, self.ymax_dec = _a_decimal(ymin), _a_decimal(ymax)
        self.xmin, self.xmax = float(self.xmin_dec), float(self.xmax_dec)
        self.ymin, self.ymax = float(self.ymin_dec), float(self.ymax_dec)

    @staticmethod
    def medir_tiempo(nombre) -> callable:
        def decorador(func) -> callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> any:
                inicio = time.perf_counter()
                resultado = func(*args, **kwargs)
                log.debug("Tiempo de '%s': %.5f s", nombre, time.perf_counter() - inicio)
                return resultado
            return wrapper
        return decorador
        
    def actualizar_fractal(self,
            xmin: Decimal, xmax: Decimal,
            ymin: Decimal, ymax: Decimal,
            width: int,   height: int,
            max_iter: int,tipo_calculo: str, 
            tipo_fractal: str,
            real: float, imag: float) -> None:
        
        self._fijar_limites(xmin, xmax, ymin, ymax)
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.tipo_calculo = tipo_calculo
        self.tipo_fractal = tipo_fractal
        self.real = float(real)
        self.imag = float(imag)
        return None

    def calcular_fractal(self) -> np.ndarray:
        metodos = FRACTAL_REGISTRY.get(self.tipo_fractal)
        if metodos is None:
            raise ValueError(f"Fractal '{self.tipo_fractal}' no registrado.")
        if self.tipo_calculo not in metodos:
            raise ValueError(f"Tipo de cálculo '{self.tipo_calculo}' no soportado para el fractal '{self.tipo_fractal}'.")
        return metodos[self.tipo_calculo](self)

    def _generar_malla_compleja_gpu(self):
        x = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
        y = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
        X, Y = cp.meshgrid(x, y)
        return (X + 1j * Y).ravel()

    ##############
    # Mandelbrot #
    ##############
    
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
    
    @register_fractal("Mandelbrot", "GPU_Cupy_kernel_optimizado", requiere="gpu")
    @medir_tiempo("Mandelbrot GPU optimizado")
    def hacer_mandelbrot_gpu2(self) -> np.ndarray:
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

    @register_fractal("Mandelbrot", "test1", requiere="gpu")
    @medir_tiempo("Mandelbrot GPU trigonométrico")
    def hacer_mandelbrot_gpu_trigonometrico(self) -> np.ndarray:
        C = self._generar_malla_compleja_gpu()
        C = C.ravel() 
        resultado = cp.empty(C.shape, dtype=cp.int32)
        try:
            mandelbrot_kernel_optimizado(C, self.max_iter, resultado)
        except Exception as e:
            raise RuntimeError(f"Fallo en Kernel GPU: {e}")
            
        resultado = resultado.reshape((self.height, self.width))
        resultado_cpu = resultado.get()
        resultado_cpu = ProcesadorColores.interpolacion_trigonometrica(resultado_cpu, self.max_iter)
        return resultado_cpu
    
    @register_fractal("Mandelbrot", "CPU_cpp", requiere="mandelbrot.dll")
    @medir_tiempo("Mandelbrot CPP")
    def hacer_mandelbrot_cpp(self) -> np.ndarray:
        ptr = cpp.mandelbrot_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        cpp.free_mandelbrot(ptr)
        return img
    
    # --- Perturbación ---
    # La órbita de referencia Z se calcula en alta precisión (GMP) en el centro de la vista;
    # cada píxel itera solo su diferencia dz en double. Ver codigos_cpp/perturbacion.cpp.

    def _preparar_perturbacion(self) -> OrbitaReferencia:
        c_re = (self.xmin_dec + self.xmax_dec) / 2
        c_im = (self.ymin_dec + self.ymax_dec) / 2
        # Mismo muestreo que np.linspace(xmin, xmax, width) del resto de los backends
        step_x = (self.xmax_dec - self.xmin_dec) / max(1, self.width - 1)
        step_y = (self.ymax_dec - self.ymin_dec) / max(1, self.height - 1)
        if abs(float(step_x)) < 1e-290:
            log.warning("Zoom fuera del rango de double (paso %.3e): la imagen va a salir rota", float(step_x))
        bits = _bits_necesarios(min(abs(step_x), abs(step_y)))

        # La calidad progresiva (low/medium/high) renderiza 3 veces el mismo centro:
        # se reutiliza la órbita si no cambió nada que la afecte.
        clave = (c_re, c_im, self.max_iter, bits)
        if self._cache_orbita is not None and self._cache_orbita[0] == clave:
            Z_re, Z_im, largo = self._cache_orbita[1]
        else:
            Z_re = np.zeros(self.max_iter + 1, dtype=np.float64)
            Z_im = np.zeros(self.max_iter + 1, dtype=np.float64)
            largo = cpp.calcular_orbita_referencia(
                format(c_re, 'f').encode('ascii'), format(c_im, 'f').encode('ascii'),
                self.max_iter, bits, Z_re, Z_im
            )
            self._cache_orbita = (clave, (Z_re, Z_im, largo))
            log.debug("Órbita de referencia: %d iteraciones a %d bits", largo - 1, bits)

        return OrbitaReferencia(
            Z_re, Z_im, largo,
            float(self.xmin_dec - c_re), float(self.ymin_dec - c_im),
            float(step_x), float(step_y)
        )

    @register_fractal("Mandelbrot", "perturbacion_cpu", requiere="perturbacion.dll")
    @medir_tiempo("Mandelbrot Perturbacion CPU")
    def hacer_mandelbrot_perturbacion_cpu(self) -> np.ndarray:
        ref = self._preparar_perturbacion()
        output = np.empty((self.height, self.width), dtype=np.int32)
        cpp.calcular_deltas_cpp(
            ref.Z_re, ref.Z_im, ref.largo,
            ref.dc_x_base, ref.dc_y_base,
            ref.step_x, ref.step_y,
            self.width, self.height, self.max_iter,
            output
        )
        return output

    @register_fractal("Mandelbrot", "perturbacion_gpu", requiere=("gpu", "perturbacion.dll"))
    @medir_tiempo("Mandelbrot Perturbacion GPU")
    def hacer_mandelbrot_perturbacion_gpu(self) -> np.ndarray:
        ref = self._preparar_perturbacion()
        Z_re = cp.asarray(ref.Z_re)
        Z_im = cp.asarray(ref.Z_im)
        output = cp.empty((self.height, self.width), dtype=cp.int32)
        bloque = (16, 16)
        grilla = ((self.width + 15) // 16, (self.height + 15) // 16)
        mandelbrot_perturbacion_kernel(grilla, bloque, (
            Z_re, Z_im, np.int32(ref.largo),
            np.float64(ref.dc_x_base), np.float64(ref.dc_y_base),
            np.float64(ref.step_x), np.float64(ref.step_y),
            np.int32(self.width), np.int32(self.height), np.int32(self.max_iter),
            output
        ))
        return output.get()

    @register_fractal("Mandelbrot", "gmp", requiere="gmp_puro.dll")
    @medir_tiempo("Gmp")
    def hacer_mandelbrot_gmp_puro(self) -> np.ndarray:
        step = (self.xmax_dec - self.xmin_dec) / max(1, self.width - 1)
        bits = _bits_necesarios(abs(step))
        coords = [format(v, 'f').encode('ascii')
                  for v in (self.xmin_dec, self.xmax_dec, self.ymin_dec, self.ymax_dec)]
        output = np.zeros((self.height, self.width), dtype=np.int32)
        cpp.calcular_mandelbrot_gmp_puro(
            *coords, self.width, self.height, self.max_iter, bits, output
        )
        return output

    @register_fractal("Mandelbrot", "GPU_Cupy_Espiral", requiere="gpu")
    @medir_tiempo("Mandelbrot Espiral GPU")
    def hacer_mandelbrot_espiral(self) -> np.ndarray:
        C = self._generar_malla_compleja_gpu().ravel() 
        
        # Debe ser float64 para los decimales de la onda trigonométrica
        resultado = cp.empty(C.shape, dtype=cp.float64) 
        
        try:
            mandelbrot_espiral_kernel(C, self.max_iter, resultado)
        except Exception as e:
            raise RuntimeError(f"Fallo en Kernel GPU: {e}")
            
        # Normalizar el resultado al rango de color 0-255 si es necesario
        res_cpu = resultado.get().reshape((self.height, self.width))
        res_normalizado = np.interp(res_cpu, (res_cpu.min(), res_cpu.max()), (0, 255))
        
        return res_normalizado

    @register_fractal("Mandelbrot", "GPU_Cupy_Triangulo", requiere="gpu")
    @medir_tiempo("Mandelbrot Triangulo GPU")
    def hacer_mandelbrot_triangulo(self) -> np.ndarray:
        C = self._generar_malla_compleja_gpu().ravel() 
        resultado = cp.empty(C.shape, dtype=cp.float64) 
        
        try:
            mandelbrot_triangulo_kernel(C, self.max_iter, resultado)
        except Exception as e:
            raise RuntimeError(f"Fallo en Kernel GPU: {e}")
            
        res_cpu = resultado.get().reshape((self.height, self.width))
        res_normalizado = np.interp(res_cpu, (res_cpu.min(), res_cpu.max()), (0, 255))
        
        return res_normalizado
    
    @register_fractal("Mandelbrot", "GPU_Cupy_Variado", requiere="gpu")
    @medir_tiempo("Mandelbrot GPU variado")
    def hacer_mandelbrot_generico(self) -> np.ndarray:
        C = self._generar_malla_compleja_gpu().ravel()
        resultado = cp.empty(C.shape, dtype=cp.float64)
        
        # CAMBIA EL KERNEL AQUI (ej. mandelbrot_gauss_kernel)
        mandelbrot_lyapunov_kernel(C, self.max_iter, resultado)
            
        res_cpu = resultado.get().reshape((self.height, self.width))
        
        # Normalizar para mapear a paleta (0 - 255)
        vmin, vmax = np.nanmin(res_cpu), np.nanmax(res_cpu)
        if vmax > vmin:
            res_normalizado = (res_cpu - vmin) / (vmax - vmin) * 255.0
        else:
            res_normalizado = np.zeros_like(res_cpu)
            
        return res_normalizado
    #########
    # Julia #
    #########

    @register_fractal("Julia", "GPU_Cupy_kernel", requiere="gpu")
    @medir_tiempo("Julia GPU")
    def hacer_julia_gpu(self) -> np.ndarray:
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
        return resultado_cpu
    
    @register_fractal("Julia", "CPU_cpp", requiere="julia.dll")
    @medir_tiempo("Julia CPP")
    def hacer_julia_cpp(self) -> np.ndarray:
        ptr = cpp.julia_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter,
            self.real,   
            self.imag    
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        cpp.free_julia(ptr)
        return img

    ################
    # Burning Ship #
    ################
    
    @register_fractal("Burning Ship", "GPU_Cupy_kernel", requiere="gpu")
    @medir_tiempo("Burning Ship GPU")
    def hacer_burning_gpu(self) -> np.ndarray:
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
        return resultado_cpu
    
    @register_fractal("Burning Ship", "CPU_cpp", requiere="burning_ship.dll")
    @medir_tiempo("Burning Ship CPP")
    def hacer_burning_cpp2(self) -> np.ndarray:
        ptr = cpp.burning_ship_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        cpp.free_burning_ship(ptr)
        return img
    
    ###########
    # Circulo #
    ###########
    
    @register_fractal("Circulo", "GPU_Cupy_kernel", requiere="gpu")
    @medir_tiempo("Circulo GPU")
    def hacer_circulo_gpu(self) -> np.ndarray:
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
        return resultado_cpu

    @register_fractal("Circulo", "CPU_cpp", requiere="circulo.dll")
    @medir_tiempo("Circulo CPP")
    def hacer_circulo_cpp2(self) -> np.ndarray:
        ptr = cpp.circulo_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        cpp.free_circulo(ptr)
        return img