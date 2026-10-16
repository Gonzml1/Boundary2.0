import math
import ctypes
import time
import logging
import decimal
from decimal import Decimal
from dataclasses import dataclass
from functools import wraps

import numpy as np

from . import backend_cpp as cpp
from .backend_cpp import BACKENDS_DISPONIBLES

log = logging.getLogger(__name__)

# Precisión de las coordenadas (dígitos decimales). Con floatexp la perturbación no tiene
# límite propio de zoom, así que el límite lo pone esto: ~1100 dígitos alcanzan para ~1e-1050.
# DefaultContext es la plantilla de los hilos nuevos (p. ej. el hilo del video).
PRECISION_DECIMAL = 1100
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


@dataclass
class Resultado:
    """Lo que devuelve un cálculo. Solo `iteraciones` es obligatorio: el resto lo llenan los
    backends que lo saben calcular (hoy, la perturbación) y solo si se pidió."""
    iteraciones: np.ndarray                  # (alto, ancho) int32; max_iter en el interior
    suave: np.ndarray | None = None          # iteración continua (float32)
    distancia: np.ndarray | None = None      # distancia al borde en píxeles (float32)
    normal_x: np.ndarray | None = None       # normal unitaria del relieve (float32)
    normal_y: np.ndarray | None = None

    def __array__(self, dtype=None, copy=None):
        # np.asarray(resultado) sigue devolviendo las iteraciones, como antes
        return np.asarray(self.iteraciones, dtype=dtype)


FRACTAL_REGISTRY: dict[str, dict[str, callable]] = {}
# (fractal, método) que calculan con precisión arbitraria y sirven a cualquier profundidad
METODOS_PROFUNDOS: set[tuple[str, str]] = set()
# Paso entre píxeles por debajo del cual un método en double ya no alcanza
LIMITE_DOUBLE = Decimal('1e-13')


def variante_rapida(fractal: str, metodo: str) -> str:
    """La variante en float del método si existe (más rápida y algo menos precisa), si no el mismo.
    Se usa donde no hace falta toda la precisión: las versiones borrosas y las vistas previas."""
    rapido = metodo + "_float"
    return rapido if rapido in FRACTAL_REGISTRY.get(fractal, {}) else metodo


def _a_gpu(arreglo: np.ndarray):
    """Copia a la GPU sin pasar por la memoria "pinned" de CuPy: su pool nunca devuelve esa
    memoria al sistema, y con órbitas y tablas de cientos de MB que cambian en cada cuadro de un
    video terminaba agotándola (cudaErrorMemoryAllocation / cudaErrorAlreadyMapped)."""
    en_gpu = cp.empty(arreglo.shape, dtype=arreglo.dtype)
    en_gpu.set(arreglo)
    return en_gpu


def liberar_memoria_gpu() -> None:
    """Devuelve al driver la memoria que CuPy tiene guardada en sus pools (sin uso)."""
    if BACKENDS_DISPONIBLES.get("gpu"):
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def es_falta_de_memoria(error: Exception) -> bool:
    if isinstance(error, MemoryError):          # numpy: "Unable to allocate ..."
        return True
    texto = str(error)
    return any(t in texto for t in ("cudaErrorMemoryAllocation", "out of memory", "cudaErrorAlreadyMapped",
                                    "OutOfMemory"))


@dataclass
class GrillaPolar:
    """Puntos de una franja de la tira del mapa exponencial (ver core/exponencial.py): la columna i
    y la fila j están en centro + radio_j (cos_i, sin_i)."""
    centro: tuple                # (Decimal, Decimal): el centro del zoom, usado como referencia
    radio_m: np.ndarray          # radio de cada fila = radio_m * 2**radio_e (float64 / int32)
    radio_e: np.ndarray
    cos_t: np.ndarray            # coseno y seno de cada columna (float64)
    sin_t: np.ndarray
    dtheta: tuple                # paso angular como (mantisa, exponente)
    paso_min: Decimal            # separación más chica entre puntos de esta franja (elige double/floatexp)
    paso_bits: Decimal           # la más chica de toda la tira: fija la precisión de la órbita, así
                                 # todas las franjas comparten la misma órbita de referencia
    e_radio_max: int             # exponente (base 2) del radio más grande


class RenderCancelado(Exception):
    """El cálculo se abandonó porque llegó un pedido más nuevo (ver calculos_mandelbrot.cancelar)."""


def register_fractal(fractal: str, calc: str, requiere: str | tuple[str, ...] = (),
                     profundo: bool = False) -> callable:
    """Registra un método de cálculo. `requiere` son los backends ("gpu", "mandelbrot.dll", ...)
    que tienen que estar disponibles; si falta alguno, el método no aparece en la UI.
    `profundo`: el método no pierde precisión al hacer zoom (perturbación, GMP)."""
    requisitos = (requiere,) if isinstance(requiere, str) else requiere
    def deco(fn) -> callable:
        faltantes = [r for r in requisitos if not BACKENDS_DISPONIBLES.get(r, False)]
        if faltantes:
            log.info("Se omite '%s / %s': falta %s", fractal, calc, ", ".join(faltantes))
            return fn
        FRACTAL_REGISTRY.setdefault(fractal, {})[calc] = fn
        if profundo:
            METODOS_PROFUNDOS.add((fractal, calc))
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


# --- Perturbación ---
# Tolerancia del BLA: un salto lineal se usa si el término que descarta es menor que
# BLA_EPSILON veces el que conserva. Más chico = más exacto y más lento.
BLA_EPSILON = 2.0 ** -40
# Por debajo de este paso entre píxeles se usa floatexp en vez de double
# (con margen: las normas |dz|^2 tienen que seguir entrando en un double).
UMBRAL_FLOATEXP = Decimal('1e-140')
# FORMULAS: número de cada fórmula en codigos_cpp/perturbacion.cpp (y nombre de sus kernels de GPU)
FORMULAS = {"mandelbrot": 0}
SALIDA_SUAVE, SALIDA_DERIVADA, DETECTAR_INTERIOR = 1, 2, 4
# La GPU avanza los píxeles en varios lanzamientos de ~este tiempo: Windows reinicia el driver
# si un kernel tarda más de 2 s, y entre lanzamientos se puede cancelar
SEGUNDOS_POR_LANZAMIENTO = 0.15
PASOS_INICIALES_POR_LANZAMIENTO = 1024
PASOS_MAX_POR_LANZAMIENTO = 1 << 20
# Píxeles por lanzamiento como máximo (el estado de cada uno ocupa ~100 bytes en la GPU)
PIXELES_MAX_POR_LANZAMIENTO = 4_000_000


def _a_floatexp(valor: Decimal) -> tuple[float, int]:
    """Decimal -> (mantisa, exponente) con valor = mantisa * 2**exponente, sin underflow."""
    if valor == 0:
        return 0.0, 0
    k = int(valor.copy_abs().adjusted() * 3.321928094887362)   # ~log2(|valor|)
    mantisa, e = math.frexp(float(valor * Decimal(2) ** -k))
    return mantisa, e + k


class _OrbitaGMP:
    """Órbita de referencia y tabla BLA, guardadas del lado de la DLL (perturbacion.cpp)."""

    def __init__(self, formula: str, c_re: Decimal, c_im: Decimal, k_re: Decimal, k_im: Decimal,
                 max_iter: int, bits: int, bandera_cancelar: ctypes.c_int):
        inicio = time.perf_counter()
        texto = lambda v: format(v, 'f').encode('ascii')
        self.formula = formula
        self.h = cpp.pert_crear(FORMULAS[formula], texto(c_re), texto(c_im), texto(k_re), texto(k_im),
                                max_iter, bits, ctypes.byref(bandera_cancelar))
        if not self.h:
            raise RenderCancelado()
        self.largo = cpp.pert_largo(self.h)
        k_re_d, k_im_d = ctypes.c_double(), ctypes.c_double()
        cpp.pert_parametro(self.h, ctypes.byref(k_re_d), ctypes.byref(k_im_d))
        self.k = (k_re_d.value, k_im_d.value)   # parámetro de la fórmula, en double
        self.clave_bla = None
        self.niveles = 0
        self._gpu = None
        log.debug("Órbita de referencia: %d iteraciones a %d bits (%.3f s)",
                  self.largo - 1, bits, time.perf_counter() - inicio)

    def preparar_bla(self, modo: int, dc_max: tuple[float, int], eps: float) -> None:
        clave = (modo, dc_max, eps)
        if clave == self.clave_bla:
            return
        self.niveles = cpp.pert_preparar_bla(self.h, modo, dc_max[0], dc_max[1], eps)
        self.clave_bla = clave
        self._gpu = None

    def en_gpu(self):
        """Copia a la GPU la órbita y la tabla del modo actual (una vez por tabla)."""
        if self._gpu is None:
            orbita = np.empty(cpp.pert_bytes_orbita(self.h), dtype=np.uint8)
            tabla = np.empty(max(1, cpp.pert_bytes_bla(self.h)), dtype=np.uint8)
            off = np.zeros(32, dtype=np.int32)
            cnt = np.zeros(32, dtype=np.int32)
            cpp.pert_exportar(self.h, orbita, tabla, off, cnt)
            self._gpu = tuple(_a_gpu(x) for x in (orbita, tabla, off, cnt))
        return self._gpu

    def __del__(self):
        if getattr(self, 'h', None) and cpp.pert_liberar:
            cpp.pert_liberar(self.h)
            self.h = None


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
        # Salidas extra que se le piden a los backends que las soportan (ver Resultado)
        self.pedir_suave = False
        self.pedir_derivada = False
        # Otro hilo la pone en 1 para abandonar el cálculo en curso (ver cancelar())
        self.bandera_cancelar = ctypes.c_int(0)
        # Si no es None, la perturbación calcula esta grilla polar en vez de la imagen rectangular
        self.grilla_polar: GrillaPolar | None = None
        # Pasos por lanzamiento en la GPU (se ajusta solo, ver _ajustar_pasos_por_lanzamiento)
        self._pasos_por_lanzamiento = PASOS_INICIALES_POR_LANZAMIENTO
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

    def cancelar(self) -> None:
        """Pide abandonar el cálculo en curso (se puede llamar desde otro hilo). Los backends de
        perturbación lo revisan seguido; el resto termina el cuadro igual."""
        self.bandera_cancelar.value = 1

    def reiniciar_cancelacion(self) -> None:
        """Vuelve a habilitar el cálculo después de una cancelación."""
        self.bandera_cancelar.value = 0

    def _revisar_cancelacion(self) -> None:
        if self.bandera_cancelar.value:
            raise RenderCancelado()

    def calcular_fractal(self) -> Resultado:
        metodos = FRACTAL_REGISTRY.get(self.tipo_fractal)
        if metodos is None:
            raise ValueError(f"Fractal '{self.tipo_fractal}' no registrado.")
        if self.tipo_calculo not in metodos:
            raise ValueError(f"Tipo de cálculo '{self.tipo_calculo}' no soportado para el fractal '{self.tipo_fractal}'.")
        resultado = metodos[self.tipo_calculo](self)
        self._revisar_cancelacion()
        if isinstance(resultado, Resultado):
            return resultado
        # Los backends simples devuelven solo la matriz de iteraciones (o de valores)
        return Resultado(np.asarray(resultado))

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

   # @register_fractal("Mandelbrot", "test1", requiere="gpu")
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
    # La órbita de referencia Z se calcula con GMP en el centro de la vista y cada píxel itera
    # solo su diferencia dz, salteando iteraciones con BLA. Ver codigos_cpp/perturbacion.cpp.
    bla_epsilon = BLA_EPSILON        # 0 desactiva el BLA
    detectar_interior = True         # cortar antes los píxeles del interior del conjunto
    modo_perturbacion = None         # None = automático, 0 = double, 1 = floatexp

    def _preparar_perturbacion(self, formula: str, modo_forzado: int | None = None):
        polar = self.grilla_polar
        if polar is None:
            c_re = (self.xmin_dec + self.xmax_dec) / 2
            c_im = (self.ymin_dec + self.ymax_dec) / 2
            # Mismo muestreo que np.linspace(xmin, xmax, width) del resto de los backends
            step_x = (self.xmax_dec - self.xmin_dec) / max(1, self.width - 1)
            step_y = (self.ymax_dec - self.ymin_dec) / max(1, self.height - 1)
            paso = min(abs(step_x), abs(step_y))
        else:
            # Tira del mapa exponencial: la referencia es el centro del zoom
            c_re, c_im = polar.centro
            paso = polar.paso_min
        if paso and paso.adjusted() < -(PRECISION_DECIMAL - 40):
            log.warning("Zoom más allá de la precisión de las coordenadas (paso %s)", f"{paso:.3e}")
        bits = _bits_necesarios(paso if polar is None else polar.paso_bits)
        k_re, k_im = _a_decimal(self.real), _a_decimal(self.imag)

        # La calidad progresiva (low/medium/high) renderiza varias veces el mismo centro:
        # se reutiliza la órbita si no cambió nada que la afecte.
        clave = (formula, c_re, c_im, k_re, k_im, self.max_iter, bits)
        if self._cache_orbita is None or self._cache_orbita[0] != clave:
            # Con muchas iteraciones la órbita y su tabla BLA ocupan cientos de MB (en RAM y en la
            # GPU): se suelta la anterior antes de armar la nueva para que no convivan las dos
            self._cache_orbita = None
            liberar_memoria_gpu()
            self._cache_orbita = (clave, _OrbitaGMP(formula, c_re, c_im, k_re, k_im, self.max_iter, bits,
                                                    self.bandera_cancelar))
        orbita = self._cache_orbita[1]

        # 0 = double, 1 = floatexp (zoom profundo), 2 = floatexp con mantisa float (métodos _float)
        modo = modo_forzado if modo_forzado is not None else self.modo_perturbacion
        if modo is None:
            modo = 1 if paso < UMBRAL_FLOATEXP else 0

        # Cota de |dc| en la imagen, redondeada hacia arriba a potencia de 2 para que la tabla
        # BLA sirva para todos los niveles de calidad (una cota mayor solo la hace más conservadora).
        if polar is None:
            dx = max(abs(self.xmin_dec - c_re), abs(self.xmax_dec - c_re))
            dy = max(abs(self.ymin_dec - c_im), abs(self.ymax_dec - c_im))
            _, e = _a_floatexp(dx + dy)
        else:
            e = polar.e_radio_max
        orbita.preparar_bla(modo, (0.5, e + 1), self.bla_epsilon)

        if polar is not None:
            return orbita, modo, None
        dc = tuple(_a_floatexp(v) for v in
                   (self.xmin_dec - c_re, self.ymin_dec - c_im, step_x, step_y, paso))
        return orbita, modo, dc

    def _log_perturbacion(self, orbita, modo, pasos, iteraciones) -> None:
        if log.isEnabledFor(logging.DEBUG):
            total = int(iteraciones.sum(dtype=np.int64)) + iteraciones.size
            log.debug("Perturbación %s %s: %d niveles BLA, %d pasos para %d iteraciones (x%.1f)",
                      orbita.formula, "floatexp" if modo else "double", orbita.niveles, pasos, total,
                      total / max(1, pasos))

    def _ajustar_pasos_por_lanzamiento(self, segundos: float) -> None:
        """Ajusta cuántos pasos avanza cada píxel por lanzamiento para que cada uno dure ~SEGUNDOS_POR_LANZAMIENTO."""
        if segundos > 0:
            factor = min(4.0, max(0.25, SEGUNDOS_POR_LANZAMIENTO / segundos))
            self._pasos_por_lanzamiento = int(min(PASOS_MAX_POR_LANZAMIENTO,
                                                  max(256, self._pasos_por_lanzamiento * factor)))

    def _salidas_pedidas(self) -> int:
        return ((SALIDA_SUAVE if self.pedir_suave else 0) | (SALIDA_DERIVADA if self.pedir_derivada else 0)
                | (DETECTAR_INTERIOR if self.detectar_interior else 0))

    def _resultado(self, iteraciones, suave, distancia, nx, ny, salidas) -> Resultado:
        con_derivada = bool(salidas & SALIDA_DERIVADA)
        return Resultado(iteraciones,
                         suave if salidas & SALIDA_SUAVE else None,
                         distancia if con_derivada else None,
                         nx if con_derivada else None,
                         ny if con_derivada else None)

    def _render_perturbacion(self, formula: str, en_gpu: bool, en_float: bool = False) -> Resultado:
        """Render por perturbación de cualquier fórmula de FORMULAS, en CPU o GPU."""
        orbita, modo, dc = self._preparar_perturbacion(formula, 2 if en_float else None)
        salidas = self._salidas_pedidas()
        forma = (self.height, self.width)
        # Las salidas que no se piden se reemplazan por una matriz mínima (no se escriben)
        forma_extra = lambda bit: forma if salidas & bit else (1, 1)

        if not en_gpu:
            iteraciones = np.empty(forma, dtype=np.int32)
            suave = np.empty(forma_extra(SALIDA_SUAVE), dtype=np.float32)
            distancia, nx, ny = (np.empty(forma_extra(SALIDA_DERIVADA), dtype=np.float32) for _ in range(3))
            if self.grilla_polar is None:
                pasos = cpp.pert_render(
                    orbita.h, *dc[0], *dc[1], *dc[2], *dc[3], *dc[4],
                    self.width, self.height, self.max_iter, salidas,
                    iteraciones, suave, distancia, nx, ny, ctypes.byref(self.bandera_cancelar)
                )
            else:
                g = self.grilla_polar
                if cpp.pert_render_polar is None:
                    raise RuntimeError("perturbacion.dll es de una versión vieja: recompilala con build_dlls.sh")
                pasos = cpp.pert_render_polar(
                    orbita.h, g.radio_m, g.radio_e, g.cos_t, g.sin_t, *g.dtheta,
                    self.width, self.height, self.max_iter, salidas,
                    iteraciones, suave, distancia, nx, ny, ctypes.byref(self.bandera_cancelar)
                )
        else:
            Z, tabla, off, cnt = orbita.en_gpu()
            iteraciones = cp.empty(forma, dtype=cp.int32)
            suave = cp.empty(forma_extra(SALIDA_SUAVE), dtype=cp.float32)
            distancia, nx, ny = (cp.empty(forma_extra(SALIDA_DERIVADA), dtype=cp.float32) for _ in range(3))
            pasos_gpu = cp.zeros(1, dtype=cp.uint64)
            kernel = kernel_perturbacion(formula, modo)
            fe_args = lambda fe: [np.float64(fe[0]), np.int32(fe[1])]
            if self.grilla_polar is None:
                grilla_args = [*fe_args(dc[0]), *fe_args(dc[1]), *fe_args(dc[2]), *fe_args(dc[3]), *fe_args(dc[4]),
                               np.float64(orbita.k[0]), np.float64(orbita.k[1]),
                               np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0), np.float64(0), np.int32(0)]
            else:
                g = self.grilla_polar
                cero = [np.float64(0), np.int32(0)]
                grilla_args = [*cero * 5, np.float64(orbita.k[0]), np.float64(orbita.k[1]),
                               _a_gpu(g.radio_m), _a_gpu(g.radio_e), _a_gpu(g.cos_t), _a_gpu(g.sin_t),
                               *fe_args(g.dtheta)]
            tam_estado = cpp.pert_tam_estado(modo)
            # Cada render arranca con pocos pasos por lanzamiento y se ajusta sobre la marcha: si
            # se heredara el valor de un render liviano, el primer lanzamiento de uno pesado
            # podría tardar segundos (sin poder cancelarse y con riesgo de que Windows reinicie el driver)
            self._pasos_por_lanzamiento = PASOS_INICIALES_POR_LANZAMIENTO
            # Franjas solo para no pasarse de memoria con el estado de cada píxel
            alto_franja = max(1, min(self.height, PIXELES_MAX_POR_LANZAMIENTO // max(1, self.width)))
            for fila in range(0, self.height, alto_franja):
                filas = min(alto_franja, self.height - fila)
                por_fila = lambda m: m[fila:fila + filas] if m.shape[0] > 1 else m
                estados = cp.empty(filas * self.width * tam_estado, dtype=cp.uint8)
                pendientes = cp.zeros(1, dtype=cp.uint32)
                bloque = (16, 16)
                grilla = ((self.width + 15) // 16, (filas + 15) // 16)
                iniciar = 1
                # Lanzamientos reanudables: cada uno avanza todos los píxeles como mucho
                # `pasos_por_lanzamiento` pasos. Ninguno dura más de lo que Windows tolera para un
                # kernel, se puede cancelar entre lanzamientos, y no hay franjas esperando a su
                # píxel más lento.
                while True:
                    self._revisar_cancelacion()
                    pendientes.fill(0)
                    inicio = time.perf_counter()
                    kernel(grilla, bloque, (
                        Z, np.int32(orbita.largo), tabla, off, cnt, np.int32(orbita.niveles),
                        *grilla_args,
                        np.int32(self.width), np.int32(filas), np.int32(fila), np.int32(self.max_iter),
                        np.int32(salidas), estados, np.int32(iniciar), np.int64(self._pasos_por_lanzamiento),
                        por_fila(iteraciones), por_fila(suave), por_fila(distancia), por_fila(nx), por_fila(ny),
                        pasos_gpu, pendientes
                    ))
                    cp.cuda.Stream.null.synchronize()
                    self._ajustar_pasos_por_lanzamiento(time.perf_counter() - inicio)
                    iniciar = 0
                    if int(pendientes.get()[0]) == 0:
                        break
            iteraciones, suave, distancia, nx, ny = (x.get() for x in (iteraciones, suave, distancia, nx, ny))
            pasos = int(pasos_gpu.get()[0])

        self._revisar_cancelacion()
        self._log_perturbacion(orbita, modo, pasos, iteraciones)
        return self._resultado(iteraciones, suave, distancia, nx, ny, salidas)

    @register_fractal("Mandelbrot", "perturbacion_cpu", requiere="perturbacion.dll", profundo=True)
    @medir_tiempo("Mandelbrot Perturbacion CPU")
    def hacer_mandelbrot_perturbacion_cpu(self) -> Resultado:
        return self._render_perturbacion("mandelbrot", en_gpu=False)

    @register_fractal("Mandelbrot", "perturbacion_gpu", requiere=("gpu", "perturbacion.dll"), profundo=True)
    @medir_tiempo("Mandelbrot Perturbacion GPU")
    def hacer_mandelbrot_perturbacion_gpu(self) -> Resultado:
        return self._render_perturbacion("mandelbrot", en_gpu=True)

    # Variante en float (24 bits de mantisa): en GPU es varias veces más rápida (las GeForce
    # hacen float ~64 veces más rápido que double), a cambio de algo de precisión en los píxeles
    # caóticos. En CPU no conviene (ahí double es igual de rápido), así que no se registra; la
    # versión de CPU (_render_perturbacion(..., en_gpu=False, en_float=True)) da exactamente lo
    # mismo y sirve para verificarla.
    @register_fractal("Mandelbrot", "perturbacion_gpu_float", requiere=("gpu", "perturbacion.dll"), profundo=True)
    @medir_tiempo("Mandelbrot Perturbacion GPU float")
    def hacer_mandelbrot_perturbacion_gpu_float(self) -> Resultado:
        return self._render_perturbacion("mandelbrot", en_gpu=True, en_float=True)

    @register_fractal("Mandelbrot", "gmp", requiere="gmp_puro.dll", profundo=True)
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

   # @register_fractal("Mandelbrot", "GPU_Cupy_Espiral", requiere="gpu")
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

    #@register_fractal("Mandelbrot", "GPU_Cupy_Triangulo", requiere="gpu")
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
    
    #@register_fractal("Mandelbrot", "GPU_Cupy_Variado", requiere="gpu")
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