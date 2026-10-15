import os
import ctypes
import logging

import numpy as np

log = logging.getLogger(__name__)

DIRECCION_DE_CARPETA = "codigos_cpp"
RUTA_CPP = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, DIRECCION_DE_CARPETA)
)
if hasattr(os, 'add_dll_directory') and os.path.isdir(RUTA_CPP):
    os.add_dll_directory(RUTA_CPP)

# Nombre de DLL -> True/False según se haya podido cargar
BACKENDS_DISPONIBLES: dict[str, bool] = {}


# --- Helper para carga de DLLs de fractales C++ ---
class CPPBackend:
    def __init__(self, dll_name: str, dll_dir: str = None):
        base_dir = dll_dir or RUTA_CPP
        dll_path = os.path.join(base_dir, dll_name)
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"DLL no encontrada: {dll_path}")
        try:
            self.lib = ctypes.CDLL(dll_path, winmode=0)
        except OSError as e:
            raise RuntimeError(f"Error cargando {dll_name}: {e}")

    def wrap_func(self, func_name: str, argtypes: list, restype):
        func = getattr(self.lib, func_name)
        func.argtypes = argtypes
        func.restype = restype
        return func


def cargar_backend(dll_name: str) -> CPPBackend | None:
    """Intenta cargar una DLL. Si falla, la marca como no disponible y devuelve None."""
    try:
        backend = CPPBackend(dll_name)
    except (FileNotFoundError, RuntimeError) as e:
        log.warning("Backend '%s' no disponible: %s", dll_name, e)
        BACKENDS_DISPONIBLES[dll_name] = False
        return None
    BACKENDS_DISPONIBLES[dll_name] = True
    log.info("Backend '%s' cargado", dll_name)
    return backend


# --- Firmas comunes ---
_ARGS_RECTANGULO = [ctypes.c_double, ctypes.c_double,
                    ctypes.c_double, ctypes.c_double,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int]
_PTR_INT = ctypes.POINTER(ctypes.c_int)
_ARRAY_F64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
_MATRIZ_I32 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS')

mandelbrot_cpp = free_mandelbrot = None
julia_cpp = free_julia = None
burning_ship_cpp = free_burning_ship = None
circulo_cpp = free_circulo = None
pert_crear = pert_parametro = pert_largo = pert_liberar = pert_preparar_bla = pert_render = None
pert_bytes_orbita = pert_bytes_bla = pert_exportar = None
calcular_mandelbrot_gmp_puro = None

# --- Instanciación de backends específicos ---
# Las DLL se compilan con build_dlls.sh y quedan en la carpeta 'codigos_cpp'.
_mandelbrot = cargar_backend('mandelbrot.dll')
if _mandelbrot:
    mandelbrot_cpp = _mandelbrot.wrap_func('mandelbrot', _ARGS_RECTANGULO, _PTR_INT)
    free_mandelbrot = _mandelbrot.wrap_func('free_mandelbrot', [_PTR_INT], None)

_julia = cargar_backend('julia.dll')
if _julia:
    julia_cpp = _julia.wrap_func(
        'julia', _ARGS_RECTANGULO + [ctypes.c_double, ctypes.c_double], _PTR_INT
    )
    free_julia = _julia.wrap_func('free_julia', [_PTR_INT], None)

_burning_ship = cargar_backend('burning_ship.dll')
if _burning_ship:
    burning_ship_cpp = _burning_ship.wrap_func('burning_ship', _ARGS_RECTANGULO, _PTR_INT)
    free_burning_ship = _burning_ship.wrap_func('free_burning_ship', [_PTR_INT], None)

_circulo = cargar_backend('circulo.dll')
if _circulo:
    circulo_cpp = _circulo.wrap_func('circulo', _ARGS_RECTANGULO, _PTR_INT)
    free_circulo = _circulo.wrap_func('free_circulo', [_PTR_INT], None)

_perturbacion = cargar_backend('perturbacion.dll')
if _perturbacion:
    # Ver codigos_cpp/perturbacion.cpp. Los números que pueden ser muy chicos (dc, pasos)
    # viajan como (mantisa double, exponente int) para no perderlos por underflow.
    _FE = [ctypes.c_double, ctypes.c_int]
    # (formula, c_re, c_im, k_re, k_im, max_iter, bits) -> handle
    pert_crear = _perturbacion.wrap_func(
        'pert_crear', [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                       ctypes.c_int, ctypes.c_int], ctypes.c_void_p
    )
    pert_parametro = _perturbacion.wrap_func(
        'pert_parametro', [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)], None
    )
    pert_largo = _perturbacion.wrap_func('pert_largo', [ctypes.c_void_p], ctypes.c_int)
    pert_liberar = _perturbacion.wrap_func('pert_liberar', [ctypes.c_void_p], None)
    pert_preparar_bla = _perturbacion.wrap_func(
        'pert_preparar_bla', [ctypes.c_void_p, ctypes.c_int, *_FE, ctypes.c_double], ctypes.c_int
    )
    _MATRIZ_F32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS')
    # (h, dc_x_base, dc_y_base, step_x, step_y, paso_pixel, width, height, max_iter, salidas,
    #  iter, suave, distancia, nx, ny) -> pasos
    pert_render = _perturbacion.wrap_func(
        'pert_render',
        [ctypes.c_void_p, *_FE, *_FE, *_FE, *_FE, *_FE,
         ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
         _MATRIZ_I32, _MATRIZ_F32, _MATRIZ_F32, _MATRIZ_F32, _MATRIZ_F32],
        ctypes.c_longlong
    )
    pert_bytes_orbita = _perturbacion.wrap_func('pert_bytes_orbita', [ctypes.c_void_p], ctypes.c_longlong)
    pert_bytes_bla = _perturbacion.wrap_func('pert_bytes_bla', [ctypes.c_void_p], ctypes.c_longlong)
    _BYTES = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS')
    _INTS = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
    pert_exportar = _perturbacion.wrap_func(
        'pert_exportar', [ctypes.c_void_p, _BYTES, _BYTES, _INTS, _INTS], ctypes.c_int
    )

_gmp_puro = cargar_backend('gmp_puro.dll')
if _gmp_puro:
    calcular_mandelbrot_gmp_puro = _gmp_puro.wrap_func(
        'calcular_mandelbrot_gmp_puro',
        [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
         ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
         _MATRIZ_I32],
        None
    )
