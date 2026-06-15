import os
import ctypes

DIRECCION_DE_CARPETA = "codigos_cpp"
# --- Helper para carga de DLLs de fractales C++ ---
class CPPBackend:
    def __init__(self, dll_name: str, dll_dir: str = None):
        # Determinar carpeta base de DLLs: por defecto ../codigos_cpp desde este archivo
        base_dir = dll_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, DIRECCION_DE_CARPETA)
        )
        dll_path = os.path.join(base_dir, dll_name)
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"DLL no encontrada: {dll_path}")
        try:
            self.lib = ctypes.CDLL(dll_path)
        except OSError as e:
            raise RuntimeError(f"Error cargando {dll_name}: {e}")

    def wrap_func(self, func_name: str, argtypes: list, restype):
        func = getattr(self.lib, func_name)
        func.argtypes = argtypes
        func.restype = restype
        return func

# --- Instanciación de backends específicos ---
# Hay que asegurarse de que la carpeta 'codigos_cpp' esté en el nivel superior del proyecto
# y contiene las DLL: mandelbrot.dll, circulo.dll, etc.
_mandelbrot_cpp = CPPBackend('mandelbrot.dll')
mandelbrot_cpp = _mandelbrot_cpp.wrap_func(
    'mandelbrot',
    [ctypes.c_double, ctypes.c_double,
     ctypes.c_double, ctypes.c_double,
     ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ctypes.POINTER(ctypes.c_int)
)
free_mandelbrot = _mandelbrot_cpp.wrap_func(
    'free_mandelbrot', [ctypes.POINTER(ctypes.c_int)], None
)

_julia_cpp = CPPBackend('julia.dll')
julia_cpp = _julia_cpp.wrap_func(
    'julia',
    [ctypes.c_double, ctypes.c_double,
     ctypes.c_double, ctypes.c_double,
     ctypes.c_int, ctypes.c_int, ctypes.c_int, 
     ctypes.c_double, ctypes.c_double],
    ctypes.POINTER(ctypes.c_int)
)
free_julia = _julia_cpp.wrap_func(
    'free_julia', [ctypes.POINTER(ctypes.c_int)], None
)

_burning_ship_cpp = CPPBackend('burning_ship.dll')
burning_ship_cpp = _burning_ship_cpp.wrap_func(
    'burning_ship',
    [ctypes.c_double, ctypes.c_double,
     ctypes.c_double, ctypes.c_double,
     ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ctypes.POINTER(ctypes.c_int)
)
free_burning_ship = _burning_ship_cpp.wrap_func(
    'free_burning_ship', [ctypes.POINTER(ctypes.c_int)], None
)

_newton_cpp = CPPBackend('newton.dll')
newton_cpp = _newton_cpp.wrap_func(
    'newton',
    [ctypes.c_double, ctypes.c_double,
     ctypes.c_double, ctypes.c_double,
     ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ctypes.POINTER(ctypes.c_int)
)
free_newton = _newton_cpp.wrap_func(
    'free_newton', [ctypes.POINTER(ctypes.c_int)], None
)

_circulo_cpp = CPPBackend('circulo.dll')
circulo_cpp = _circulo_cpp.wrap_func(
    'circulo',
    [ctypes.c_double, ctypes.c_double,
     ctypes.c_double, ctypes.c_double,
     ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ctypes.POINTER(ctypes.c_int)
)

free_circulo = _circulo_cpp.wrap_func(
    'free_circulo', [ctypes.POINTER(ctypes.c_int)], None
)


