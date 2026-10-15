# ABoundary

Explorador interactivo de fractales (Mandelbrot, Julia, Burning Ship y otros) hecho con PyQt5 + OpenGL.
Cada fractal se puede calcular con varios backends: kernels de GPU con CuPy, DLLs en C++ con OpenMP,
GMP de precisión arbitraria y **teoría de perturbaciones** (con BLA y floatexp) para zooms profundos
(probado hasta 1e-1000).

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

Las coordenadas se guardan como `Decimal` con 1100 dígitos, y la UI las muestra con todos los
dígitos que hacen falta. Con double, el cálculo directo se rompe cerca de un zoom de 1e-13.

Los métodos `perturbacion_cpu` y `perturbacion_gpu` comparten el mismo núcleo
([`codigos_cpp/perturbacion_core.h`](codigos_cpp/perturbacion_core.h)): la DLL lo compila con g++
y la GPU lo compila con NVRTC. Funciona así:

1. **Órbita de referencia.** Se calcula la órbita del centro de la vista, `Z_n`, con GMP. La precisión
   se ajusta sola según el zoom y la órbita se cachea mientras no cambie el centro.
2. **Perturbación.** Para cada píxel `c = C + dc` se itera solo la diferencia `dz_n = z_n - Z_n`:
   `dz_{n+1} = (2 Z_n + dz_n) dz_n + dc`.
3. **Rebasing.** Si `|Z_m + dz| < |dz|`, o si se termina la órbita de referencia, se reinicia la referencia
   con `dz = Z_m + dz` y `m = 0`. Eso evita los glitches con una sola órbita.
4. **BLA (bilinear approximation).** Mientras `dz` es chico frente a `Z`, un paso es casi lineal:
   `dz -> A dz + B dc`. Los pasos lineales se componen
   (`A = A2 A1`, `B = A2 B1 + B2`), así que se arma una tabla de saltos de 1, 2, 4, 8... iteraciones,
   cada uno con un radio de validez. Cada píxel usa el salto más largo que sea válido. La tolerancia es
   `BLA_EPSILON` (2⁻⁴⁰); con valores más grandes aparecen errores.
5. **floatexp.** Cuando el paso entre píxeles baja de 1e-140, `dc` y `dz` dejan de entrar cómodos
   en un double. Ahí se pasa a un double con exponente entero aparte, que no tiene underflow.
   Es más lento por paso, pero con BLA se hacen muy pocos pasos.

Resultados medidos (comparados píxel a píxel con el cálculo de GMP puro, `gmp`):

| Vista | Resultado | Tiempo |
|---|---|---|
| `c = i`, zoom 1e-40 y 1e-400 | idéntico a GMP | ms |
| Borde de minibrot, zoom 1e-30, 256.000 iteraciones | idéntico a GMP | ~0,1 s (32×18) |
| `c = i`, zoom 1e-1000, 1280×720 | idéntico a GMP en una muestra de 16×9 | CPU 0,36 s / GPU 0,16 s con BLA (sin BLA: 25,7 s / 2,3 s) |

En zooms poco profundos (< 1e-20) el BLA casi no saltea nada, porque `dz` deja de ser chico enseguida.
Ahí puede ser algo más lento que sin BLA. Para desactivarlo: `calculos_mandelbrot.bla_epsilon = 0`.

El límite práctico ahora es la precisión de las coordenadas (`PRECISION_DECIMAL`, ~1e-1050) y la
cantidad de iteraciones que pidas.
