# ABoundary

Explorador interactivo de fractales (Mandelbrot, Julia, Burning Ship y otros) hecho con PyQt5 + OpenGL.
Cada fractal se puede calcular con varios backends: kernels de GPU con CuPy, DLLs en C++ con OpenMP,
GMP de precisión arbitraria y **teoría de perturbaciones** para zooms profundos (hasta ~1e-290).

## Instalación

Requiere Windows, Python 3.11 o más nuevo y, si querés usar la GPU, una placa NVIDIA con CUDA.

```bash
pip install -r requirements.txt
pip install cupy-cuda12x   # opcional, solo para los backends GPU
```

### Compilar las DLL de C++

Las DLL no están en el repo. Se compilan con MinGW-w64 desde [MSYS2](https://www.msys2.org/).
En la terminal **MSYS2 UCRT64**:

```bash
pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-gmp
./build_dlls.sh
```

Quedan en `codigos_cpp/` y se linkean de forma estática, así que no hace falta copiar `libgomp-1.dll` ni otras.
Se compilan con `-march=native`: si las llevás a otra PC, recompilalas ahí.

## Uso

```bash
python iniciador.py           # normal
python iniciador.py --debug   # además muestra el tiempo de cada render
```

Si falta la GPU o alguna DLL, la app arranca igual: esos métodos no aparecen en el combo
y el log dice qué se omitió.

| Control | Acción |
|---|---|
| Rueda / `+` `-` | Zoom |
| Click izquierdo / derecho | Centrar y acercar / alejar |
| Click del medio + arrastrar, flechas, WASD | Mover |
| `P` / `O` | Paleta siguiente / anterior |
| `G` / `H` | Duplicar / dividir iteraciones |
| `B` / `N` | Duplicar / dividir clase de equivalencia de la paleta |
| `Q` | Activar/desactivar la calidad progresiva |
| `R` | Volver a la vista inicial |

## Estructura

```
iniciador.py                     punto de entrada
gui/                             ventana principal (gui.py) y la UI generada por Qt Designer
core/modulo_opengl.py            widget OpenGL: interacción, zoom y calidad progresiva
core/modulo_de_calculo_fractales.py  motor de cálculo y registro de fractales
core/funciones_kernel.py         kernels CuPy
core/backend_cpp.py              carga de las DLL
core/paletas.py                  paletas de color
codigos_cpp/                     fuentes C++ de los backends de CPU
```

### Agregar un fractal o un backend

El motor usa un registro: cualquier método de `calculos_mandelbrot` decorado con `@register_fractal`
aparece solo en la UI.

```python
@register_fractal("Mandelbrot", "Mi_metodo", requiere="gpu")   # o "mi.dll", o ("gpu", "mi.dll")
@medir_tiempo("Mi método")
def hacer_mi_metodo(self) -> np.ndarray:
    ...  # devolver una matriz (height, width) de iteraciones
```

Lo mismo con las paletas y `@register_palette` en `core/paletas.py`.

### Editar la interfaz

`gui/MandelbrotGUI.py` se genera a partir del `.ui`; no lo edites a mano:

```bash
pyuic5 -x gui/MandelbrotGUI.ui -o gui/MandelbrotGUI.py
```

## Zoom profundo: cómo funciona la perturbación

Las coordenadas se guardan como `Decimal` con 320 dígitos, y la UI las muestra con todos los
dígitos que hacen falta. Con double, el cálculo directo se rompe cerca de un zoom de 1e-13.

Los métodos `perturbacion_cpu` y `perturbacion_gpu` hacen lo siguiente:

1. Calculan la órbita del centro de la vista, `Z_n`, con GMP. La precisión se ajusta sola según el zoom.
2. Para cada píxel `c = C + dc` iteran solo la diferencia `dz_n = z_n - Z_n`, en double:
   `dz_{n+1} = (2 Z_n + dz_n) dz_n + dc`. Como `dz` y `dc` son chicos, el double los representa bien.
3. Usan **rebasing**: si `|Z_m + dz| < |dz|`, o si se termina la órbita de referencia, vuelven a empezar
   la referencia con `dz = Z_m + dz` y `m = 0`. Eso evita los "glitches" típicos de la perturbación
   con una sola órbita de referencia.

El resultado coincide píxel a píxel con el cálculo de GMP puro (`gmp`), pero es cientos de veces más rápido.
El límite es ~1e-290, porque más abajo `dc` deja de entrar en un double.
