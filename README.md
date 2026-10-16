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

`F1` muestra todos los atajos dentro de la app.

| Control | Acción |
|---|---|
| Rueda del mouse | Zoom hacia el cursor (el punto bajo el mouse queda quieto) |
| Clic izquierdo / derecho | Acercar / alejar en ese punto |
| Mantener el botón izquierdo (¼ s) y arrastrar | Zoom al rectángulo marcado. Si soltás antes, es el zoom por clic de siempre |
| Arrastrar con el del medio, flechas, WASD | Mover |
| `+` / `-` | Acercar / alejar al centro |
| `Backspace`, `Alt+Izquierda`, botón "atrás" del mouse | Vista anterior |
| `Shift+Backspace`, `Alt+Derecha` | Vista siguiente |
| `P` / `O` | Paleta siguiente / anterior |
| `G` / `H` | Duplicar / dividir iteraciones |
| `B` / `N` | Duplicar / dividir clase de equivalencia de la paleta |
| `Q` | Modo foto completa: cada imagen se calcula entera a resolución completa y recién ahí se muestra (sin versiones previas borrosas ni la imagen anterior corrida). Apagado, el render es progresivo |
| `R` | Volver a la vista inicial |

La barra de estado muestra el tamaño de la vista (`zoom 3.2e-145`), cuánto lleva calculando un
render largo, y un aviso si el método elegido no tiene precisión para esa profundidad. Las coordenadas
del cursor se muestran con todos los dígitos que corresponden a la profundidad.

Si se mueve la vista mientras se calcula, el cálculo en curso se abandona y arranca el nuevo. Cambiar
solo el coloreo (paleta, ciclo, luz o color del interior) no recalcula el fractal: se recolorea el
último cálculo, y volver a una vista reciente también sale del caché.

La app recuerda entre sesiones el tamaño de la ventana, el fractal y el método, la paleta, las opciones
de color y las últimas carpetas usadas. Al arrancar por primera vez elige `perturbacion_gpu` (o
`perturbacion_cpu` si no hay GPU), que son los métodos que soportan todas las opciones.

### Opciones de color (panel "4. COLOR")

Las tres se activan y desactivan por separado. Funcionan con los métodos de perturbación
(`perturbacion_cpu` / `perturbacion_gpu`); con los demás, la barra de estado avisa que no aplican.

- **Coloreo suave**: usa la iteración continua `n + 1 - log2(ln|z|)` en vez del entero. Así desaparecen
  las bandas. Se calcula siguiendo 4 iteraciones después del escape, sin cambiar el conteo de iteraciones.
- **Iluminación (estimador de distancia)**: sigue la derivada `dz/dc` durante la iteración (también a
  través de los saltos del BLA, donde se transforma como `der -> A der + B`). Con ella calcula la
  normal del "relieve" (dirección de `z / dz/dc`), que se ilumina desde el ángulo elegido, y la
  distancia al borde, que oscurece lo que está a menos de un píxel del conjunto.
- **Antialiasing (2x2)**: renderiza al doble de resolución y promedia. Cuesta 4 veces más, así que en
  pantalla se aplica solo a la calidad final.
- **Interior**: el color de los puntos que pertenecen al conjunto. Puede ser el de la paleta, negro,
  blanco o uno a elección.

Las opciones se respetan igual en pantalla, en "Guardar imagen" y en el video.

### Video de zoom (panel "6. EXPORTAR")

Genera un zoom que arranca con el fractal entero y termina exactamente en la vista actual. En el
diálogo se elige la resolución, la duración, los fotogramas por segundo, si el zoom arranca y frena
suave, y cuántos segundos queda quieto en la última imagen. Todos los cuadros usan las iteraciones de
la vista actual.

- **Vista previa**: reproduce el recorrido a baja resolución en unos segundos, antes de renderizar.
- Cada video se guarda en una carpeta nueva (`zoom_AAAA-MM-DD_HH-MM-SS`) con `frame_1.png`,
  `frame_2.png`, ... y, si está instalado ffmpeg, el `zoom.mp4`. El progreso muestra cuánto falta.
- **Continuar video…**: si se cancela (o se corta la luz), se puede retomar eligiendo su carpeta. Los
  cuadros que ya están no se recalculan: la carpeta guarda todos los parámetros en `video.json`. Si al
  cancelar quedó algún hueco (con CPU y GPU calculando a la vez, el cuadro que tenía la CPU puede quedar
  sin terminar), se rellena primero.
- **Rehacer cuadros…**: vuelve a calcular cuadros puntuales de un video, por ejemplo uno que se borró
  o que querés regenerar. Muestra cuáles faltan y acepta números y rangos (`17, 230-235`). Los cuadros
  salen idénticos a los originales y el `.mp4` se vuelve a armar si están todos.
- **Calcular también con la CPU** (activado por defecto): si el método es `perturbacion_gpu`, la CPU
  calcula cuadros al mismo tiempo con `perturbacion_cpu`. Las dos dan exactamente la misma imagen (ver
  más abajo), así que mezclar cuadros no se nota. Los PNG se comprimen en otros hilos mientras se
  calculan los cuadros siguientes.

#### Modo rápido (mapa exponencial)

En "Tipo de video" se puede elegir **Rápido (mapa exponencial)**. En coordenadas (logaritmo del radio,
ángulo) alrededor del centro del zoom, acercar la imagen equivale a desplazarse. Entonces, en vez de
calcular cada cuadro, se calcula una sola **tira**: cada fila es un anillo alrededor del destino, desde
el radio que cubre el conjunto entero hasta el más chico que hace falta en el último cuadro. Después
cada cuadro se arma deformando la tira (en GPU, ~5 ms por cuadro), sin calcular ninguna iteración.

- **El zoom queda centrado** en el destino desde el primer cuadro: el primer cuadro se aleja lo
  suficiente como para mostrar el conjunto entero.
- **Rotación**: "Rotación (vueltas)" hace girar la imagen durante el zoom, sin costo extra.
- **La tira se guarda** en la subcarpeta `tira/` del video, ya coloreada. Se puede cancelar y
  continuar, y rehacer cuadros sin recalcularla. Cambiar la paleta sí obliga a calcularla de nuevo.
- **Cuánto conviene** depende de cuántos cuadros dura cada duplicación del zoom; el diálogo lo estima
  antes de renderizar. Por ejemplo, en 1080p a 60 fps, un zoom a 1e-30 en 60 s calcula ~12 veces menos
  que cuadro por cuadro. Un zoom a 1e-300 en 30 s, en cambio, calcula más (hay tan pocos cuadros por
  duplicación que no hay nada que reusar): ahí conviene el modo normal.
- La tira se calcula con perturbación sobre una grilla polar (`pert_render_polar` en la DLL y el mismo
  kernel de GPU); los radios y ángulos van en tablas, así CPU y GPU siguen dando lo mismo. Un cuadro del
  mapa exponencial y el mismo cuadro calculado normalmente se ven iguales; difieren solo en los
  filamentos más finos, porque cada píxel muestrea un punto apenas distinto.

#### Qué método usar para un video

| Método | Velocidad | Precisión |
|---|---|---|
| `perturbacion_gpu` (+ CPU) | ~2× más rápido que solo GPU | La de siempre: idéntico a GMP salvo píxeles caóticos |
| `perturbacion_gpu_float` | ~3× (hasta 10× en zonas sin muchas iteraciones) | Mantisa de 24 bits: ~3 % de los píxeles cambia, todos en los filamentos caóticos (a simple vista no se distingue) |

Medido con un video de prueba de 20 cuadros en 960×540 a 20.000 iteraciones: 54,7 s antes de estas
mejoras, 20 a 30 s con `perturbacion_gpu` + CPU y 16,6 s con `perturbacion_gpu_float`.

### Favoritos

"Guardar" en el panel de favoritos guarda la vista con todos sus dígitos, las iteraciones, el
fractal, el método, la paleta y las opciones en `favoritos.json`, en la raíz del proyecto. Ese
archivo no se versiona.

## Estructura

```
iniciador.py                         punto de entrada
gui/gui.py                           ventana principal: conecta los controles con el widget
gui/dialogos.py                      diálogos del video (configuración, vista previa, progreso) y atajos
gui/MandelbrotGUI.ui                 la interfaz (Qt Designer); MandelbrotGUI.py se genera de acá
gui/tema_oscuro.py                   tema visual
core/modulo_opengl.py                widget OpenGL + hilo de render (calidad progresiva)
core/render.py                       pedido -> imagen: cálculo + paleta + suave + luz + antialiasing
core/modulo_de_calculo_fractales.py  motor de cálculo y registro de fractales
core/funciones_kernel.py             kernels CuPy
core/backend_cpp.py                  carga de las DLL
core/paletas.py                      paletas de color
core/video.py                        planificación del video de zoom y su plan guardado (video.json)
core/exponencial.py                  modo rápido del video: la tira del mapa exponencial y el armado de cuadros
core/favoritos.py                    ubicaciones guardadas
codigos_cpp/                         fuentes C++ (perturbacion_core.h: núcleo compartido CPU/GPU)
```

El cálculo nunca corre en el hilo de la interfaz. El widget arma un `PedidoRender` (una foto inmutable
de lo que hay que dibujar) y se lo pasa al hilo de render, que atiende siempre el más reciente. Mientras
llega la imagen nueva, el widget reproyecta la anterior, así el zoom y el arrastre responden al instante.

### Agregar un fractal o un backend

El motor usa un registro: cualquier método de `calculos_mandelbrot` decorado con `@register_fractal`
aparece solo en la UI.

```python
@register_fractal("Mandelbrot", "Mi_metodo", requiere="gpu")   # o "mi.dll", o ("gpu", "mi.dll")
@medir_tiempo("Mi método")
def hacer_mi_metodo(self) -> np.ndarray:
    ...  # devolver una matriz (height, width) de iteraciones, o un Resultado con más salidas
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
6. **Detección del interior.** Además se acumula `|dz_n/dz_1|^2 = prod |2 z_k|^2`. Dentro del conjunto la
   órbita cae en un ciclo atractor y ese producto tiende a cero; cuando baja de `1e-12` el píxel se da por
   interior y se corta ahí, en vez de hacer todas las iteraciones. Con iteraciones altas, las vistas con
   mucho interior salen hasta 15 veces más rápido. En las pruebas no marcó como interior ningún píxel que
   escapara.

**CPU y GPU dan exactamente lo mismo.** Las dos DLL/kernels se compilan sin fusionar multiplicaciones y
sumas por su cuenta (`-ffp-contract=off` en g++, `--fmad=false` en NVRTC) y las FMA se escriben a mano
en el núcleo, así las dos hacen las mismas operaciones con el mismo redondeo. Por eso el video puede
repartir cuadros entre las dos.

**La GPU calcula en lanzamientos reanudables.** Windows reinicia el driver de video si un kernel tarda
más de 2 segundos. En vez de partir la imagen en franjas (cada franja espera a su píxel más lento), cada
lanzamiento avanza todos los píxeles una cantidad limitada de pasos, guarda su estado y sigue en el
siguiente. Cada lanzamiento dura ~150 ms, y entre uno y otro se puede cancelar.

**`perturbacion_gpu_float`** usa un floatexp con mantisa `float` en vez de `double`: las GeForce hacen
cuentas en float unas 64 veces más rápido. La app lo usa sola en las versiones borrosas de la calidad
progresiva y en la vista previa del video; la imagen final usa el método elegido.

Se probó también usar como referencia el núcleo del minibrot cercano (encontrado con Newton) en vez del
centro: con rebasing ganaba apenas un 5-15 % y en algunas ubicaciones encontraba núcleos lejanos, así
que se descartó.

Resultados medidos (comparados píxel a píxel con el cálculo de GMP puro, `gmp`):

| Vista | Resultado | Tiempo |
|---|---|---|
| `c = i`, zoom 1e-40 y 1e-400 | idéntico a GMP | ms |
| Borde de minibrot, zoom 1e-30, 256.000 iteraciones | idéntico a GMP | ~0,1 s (32×18) |
| `c = i`, zoom 1e-1000, 1280×720 | idéntico a GMP en una muestra de 16×9 | CPU 0,36 s / GPU 0,16 s con BLA (sin BLA: 25,7 s / 2,3 s) |

En zooms poco profundos (< 1e-20) el BLA casi no saltea nada, porque `dz` deja de ser chico enseguida.
Ahí puede ser algo más lento que sin BLA. Para desactivarlo: `calculos_mandelbrot.bla_epsilon = 0`.

### Agregar una fórmula a la perturbación (Julia, Burning Ship, ...)

Todo el núcleo (rebasing, BLA, floatexp, suave, derivada, CPU y GPU) es genérico sobre la fórmula.
Una fórmula nueva necesita:

1. Una estructura en `perturbacion_core.h` con su perturbación, su derivada, su paso completo y sus
   coeficientes de BLA (o `tiene_bla = false`). La de `Mandelbrot` sirve de modelo.
2. Su paso de la órbita de referencia en GMP, en `perturbacion.cpp`.
3. Agregarla en los lugares marcados con `FORMULAS`: los `switch` de `perturbacion.cpp`,
   `KERNELS_FORMULA(...)` en `core/funciones_kernel.py` y el diccionario `FORMULAS` en Python.
4. Registrarla en dos líneas:

```python
@register_fractal("Julia", "perturbacion_gpu", requiere=("gpu", "perturbacion.dll"))
def hacer_julia_perturbacion_gpu(self):
    return self._render_perturbacion("julia", en_gpu=True)
```

El parámetro de la fórmula (la `c` de Julia) llega desde los campos "Julia Re(c) / Im(c)" de la interfaz.

El límite práctico ahora es la precisión de las coordenadas (`PRECISION_DECIMAL`, ~1e-1050) y la
cantidad de iteraciones que pidas.
