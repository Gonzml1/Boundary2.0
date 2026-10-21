# render_nube: videos de ABoundary sin interfaz

Un programa de consola que toma la carpeta de un video (con el `video.json` que guarda la app) y la
completa hasta el mp4. Sirve para renderizar en tu PC sin la ventana, o en una máquina alquilada con
varias GPUs, que es lo que lo hace mucho más rápido.

```bash
python render_nube/render_video.py CARPETA_DEL_VIDEO
```

- **Mismos cuadros que la app**: usa el mismo motor (`core/`). Probado cuadro por cuadro: idénticos en
  modo normal y rápido, con y sin antialiasing, luz y suave.
- **Usa todas las GPUs** de la máquina (cada una calcula franjas o tramos distintos) y, si el video lo
  pide, también la CPU mientras se calcula la tira.
- **Se corta y se retoma**: Ctrl+C (o que la máquina se apague) y volver a correr lo mismo sigue desde
  donde quedó. Las franjas de la tira y los tramos de video ya hechos no se repiten.
- **La carpeta es la misma que usa la app**: lo que calcula uno lo aprovecha el otro (por ejemplo, la
  tira que ya empezaste en tu PC).
- **No guarda un PNG por cuadro** (54 000 cuadros Full HD son ~200 GB): los cuadros van directo a
  ffmpeg en tramos de video, que al final se unen en `zoom.mp4` sin recodificar.

## Qué imprime

Una línea por evento, sin barras animadas (se lee bien por SSH o en un log):

```text
ABoundary · render sin interfaz
Video: zoom_2026-10-16_23-04-28 · 1920x1080 · 60 fps · 15 min 00 s · 54000 cuadros · modo rápido (mapa exponencial)
Cálculo: Mandelbrot con perturbacion_gpu_float, 4,194,304 iteraciones · antialiasing no
  GPU 0: NVIDIA GeForce RTX 4090
  GPU 1: NVIDIA GeForce RTX 4090
Tira: 762,642 filas x 6,976 ángulos · 0/1490 franjas ya calculadas · faltan 1490

[00:00:31] tira      1/1490 franjas  franja 0 (GPU 1) en 27.8 s
[00:00:33] tira      2/1490 franjas  franja 1 (GPU 0) en 29.1 s · faltan ~5 h 40 min
...
[05:41:10] tira: 1490 franjas calculadas en 5 h 41 min
Video: 54 tramos de 1000 cuadros · x264 calidad 18 · 0 ya hechos · faltan 54
[05:42:05] video  1000/54000 cuadros  tramo 1/54 (cuadros 1-1000, GPU 0) en 52 s · faltan ~23 min
...
[06:05:40] Video listo: .../zoom.mp4 (3.10 GB) · total 6 h 05 min
```

Termina con código 0 si el video quedó listo, 2 si se cortó (se sigue corriendo lo mismo) y 1 si hubo
un error.

## Opciones

| Opción | Qué hace |
|---|---|
| `--plan archivo.json` | Usa ese `video.json`: lo copia a la carpeta (que se crea si no existe) |
| `--gpus 0,2` | Qué GPUs usar (por defecto, todas) |
| `--sin-cpu` | No calcular también con la CPU |
| `--codificador` | `x264` (por defecto, el más compatible), `x265` (mitad de tamaño), `nvenc-h264` / `nvenc-h265` (en la GPU, más rápido) |
| `--calidad N` | CRF/CQ del codificador: más bajo = mejor y más pesado (por defecto 18 en x264) |
| `--tramo N` | Cuadros por tramo de video (por defecto, según el largo del video, hasta 1000) |
| `--solo-tira` | Modo rápido: calcular la tira y no armar el video |
| `--franjas A-B` | Modo rápido: calcular solo esas franjas (para repartir la tira entre varias máquinas) |
| `--guardar-cuadros` | Guardar también cada cuadro como PNG |
| `--detalle` | Mostrar también los mensajes internos del motor |

El codificador, la calidad y los cuadros por tramo se fijan la primera vez (en `tramos/tramos.json`)
y se respetan al retomar, para que todos los tramos se puedan unir. Para cambiarlos, borrá `tramos/`.

## Renderizar en la nube, paso a paso

La idea es alquilar **una máquina con varias GPUs** por unas horas, calcular ahí todo el video y
bajar solo el mp4. Cada franja de la tira se calcula sola, así que la velocidad crece casi en
proporción a la cantidad de GPUs.

1. **Crear el video en la app**: Video de zoom → elegí todo → **Solo guardar el plan…**. Queda una
   carpeta `zoom_...` con el `video.json` (y, si ya habías empezado en tu PC, su `tira/`).

2. **Alquilar la máquina** (vast.ai, RunPod u otro): Ubuntu con una imagen "CUDA" o "PyTorch",
   varias GPUs (por ejemplo 4 u 8 RTX 4090) y disco para la tira y el mp4. Cuánto pesa cada franja
   depende mucho de la imagen (de ~0,5 MB en zonas suaves hasta ~20 MB con mucho detalle y
   antialiasing): con 50 GB libres sobra para un video Full HD. Te dan un comando `ssh` para
   conectarte.

3. **Subir el proyecto y la carpeta del video** (desde tu PC):

   ```bash
   ssh -p PUERTO root@IP "git clone https://github.com/Gonzml1/Boundary2.0.git aboundary"
   scp -P PUERTO -r D:/Avideosfractales/zoom_2026-10-16_23-04-28 root@IP:aboundary/
   ```

4. **Instalar** (en la máquina, una sola vez; tarda unos minutos):

   ```bash
   cd aboundary
   bash render_nube/instalar.sh
   ```

   Instala g++, GMP, ffmpeg y CuPy, compila las bibliotecas de C++ para Linux y comprueba que cada GPU
   compile el kernel.

5. **Renderizar dentro de `tmux`**, así sigue aunque se corte tu conexión:

   ```bash
   tmux new -s render
   source render_nube/.venv/bin/activate
   python render_nube/render_video.py zoom_2026-10-16_23-04-28 2>&1 | tee render.log
   ```

   Para dejarlo corriendo: Ctrl+B y después D. Para volver a verlo: `tmux attach -t render`.

6. **Bajar el video** (desde tu PC) y **destruir la máquina** en la página del proveedor (si no, se
   sigue cobrando):

   ```bash
   scp -P PUERTO root@IP:aboundary/zoom_2026-10-16_23-04-28/zoom.mp4 .
   ```

Si la máquina se interrumpe (las más baratas pueden hacerlo), volvé a correr el paso 5 y sigue desde
donde quedó.

### Repartir entre varias máquinas

Cada máquina calcula una parte de la tira con `--franjas` (el resumen al arrancar dice cuántas hay):

```bash
python render_nube/render_video.py zoom_x --franjas 0-999      # máquina 1
python render_nube/render_video.py zoom_x --franjas 1000-1999  # máquina 2
```

Después se juntan las carpetas `tira/` en una máquina y se corre sin `--franjas` para armar el video.

## Archivos

```text
render_nube/
  render_video.py     el programa
  instalar.sh         prepara una máquina Linux con GPU
  compilar_linux.sh   compila perturbacion.so y gmp_puro.so en codigos_cpp/
  requirements.txt    dependencias de Python (sin PyQt)
```

En la carpeta del video agrega `tramos/` (los tramos de video y `tramos.json`) y `zoom.mp4`.
