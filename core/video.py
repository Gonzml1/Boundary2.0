"""Planificación de videos de zoom: desde una vista alejada del fractal hasta la vista actual."""
import os
import logging
import shutil
import subprocess
from datetime import datetime
from decimal import Decimal
from typing import Iterator, NamedTuple

log = logging.getLogger(__name__)

# Vista de partida: el conjunto de Mandelbrot entero
CENTRO_INICIAL = (Decimal('-0.5'), Decimal('0'))
ANCHO_MINIMO_INICIAL = Decimal('3.2')
ALTO_MINIMO_INICIAL = Decimal('2.6')


class Cuadro(NamedTuple):
    xmin: Decimal
    xmax: Decimal
    ymin: Decimal
    ymax: Decimal
    max_iter: int


def planificar_zoom(xmin: Decimal, xmax: Decimal, ymin: Decimal, ymax: Decimal,
                    ancho_px: int, alto_px: int, cuadros: int, max_iter: int) -> Iterator[Cuadro]:
    """Genera los cuadros de un zoom que arranca con el fractal entero y termina en la vista dada.

    - El ancho de la vista baja en forma geométrica (multiplicándose siempre por el mismo
      factor), así el zoom se ve a velocidad constante.
    - El destino arranca donde cae en la vista inicial y se desplaza de a poco hasta el centro
      de la pantalla, donde queda exacto en el último cuadro.
    - Todos los cuadros usan las mismas iteraciones que la vista final, así ninguna parte del
      recorrido queda mal resuelta por falta de iteraciones.
    - El alto se ajusta a la relación de aspecto de la resolución del video.
    """
    aspecto = Decimal(alto_px) / Decimal(ancho_px)
    cx_f, cy_f = (xmin + xmax) / 2, (ymin + ymax) / 2
    ancho_f = xmax - xmin
    cx_0, cy_0 = CENTRO_INICIAL
    ancho_0 = max(ANCHO_MINIMO_INICIAL, ALTO_MINIMO_INICIAL / aspecto, ancho_f)

    log_razon = (ancho_f / ancho_0).ln()          # <= 0
    px_0, py_0 = (cx_f - cx_0) / ancho_0, (cy_f - cy_0) / ancho_0
    for i in range(cuadros):
        t = Decimal(i) / Decimal(max(1, cuadros - 1))
        ancho = ancho_0 * (t * log_razon).exp()
        if i == cuadros - 1:
            ancho = ancho_f                        # que termine exacto, sin error de redondeo
        # Posición del destino en pantalla (en anchos de vista desde el centro): arranca donde
        # cae en la vista inicial y se mueve linealmente hasta el centro en el último cuadro.
        cx = cx_f - px_0 * (1 - t) * ancho
        cy = cy_f - py_0 * (1 - t) * ancho
        medio_ancho = ancho / 2
        medio_alto = ancho * aspecto / 2
        yield Cuadro(cx - medio_ancho, cx + medio_ancho, cy - medio_alto, cy + medio_alto, max_iter)


def crear_carpeta_video(base: str) -> str:
    """Crea una subcarpeta nueva para un video dentro de `base` (zoom_AAAA-MM-DD_HH-MM-SS).

    Cada video va en su propia carpeta para que sus cuadros nunca se mezclen con los de
    otro video (el explorador los intercala al ordenar y parece que la numeración salta).
    """
    nombre = datetime.now().strftime("zoom_%Y-%m-%d_%H-%M-%S")
    carpeta = os.path.join(base, nombre)
    sufijo = 2
    while os.path.exists(carpeta):              # dos videos en el mismo segundo
        carpeta = os.path.join(base, f"{nombre}_{sufijo}")
        sufijo += 1
    os.makedirs(carpeta)
    return carpeta


def armar_mp4(carpeta: str, fps: int, nombre: str = "zoom.mp4") -> str | None:
    """Une los frame_XXXXX.png en un mp4 con ffmpeg, si está instalado. Devuelve la ruta o None."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        log.warning("ffmpeg no está instalado: quedan solo los PNG en %s", carpeta)
        return None
    salida = os.path.join(carpeta, nombre)
    comando = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
               "-i", os.path.join(carpeta, "frame_%05d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               # libx264 + yuv420p necesita ancho y alto pares
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", salida]
    resultado = subprocess.run(comando, capture_output=True, text=True)
    if resultado.returncode != 0:
        log.error("ffmpeg falló: %s", resultado.stderr.strip())
        return None
    return salida
