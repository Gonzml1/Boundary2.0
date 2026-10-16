"""Planificación de videos de zoom: desde una vista alejada del fractal hasta la vista actual."""
import os
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from decimal import Decimal
from typing import Iterator, NamedTuple

log = logging.getLogger(__name__)

# Vista de partida: el conjunto de Mandelbrot entero
CENTRO_INICIAL = (Decimal('-0.5'), Decimal('0'))
ANCHO_MINIMO_INICIAL = Decimal('3.2')
ALTO_MINIMO_INICIAL = Decimal('2.6')

# Archivo con los parámetros del video, dentro de su carpeta (para poder retomarlo)
ARCHIVO_PLAN = "video.json"


class Cuadro(NamedTuple):
    xmin: Decimal
    xmax: Decimal
    ymin: Decimal
    ymax: Decimal
    max_iter: int


@dataclass(frozen=True)
class ConfigVideo:
    ancho: int
    alto: int
    duracion: float          # segundos de zoom
    fps: int
    suavizado: bool = False  # arranca y frena suave en vez de ir a velocidad constante
    pausa: float = 0.0       # segundos quieto en la última imagen
    usar_cpu: bool = True    # si el método es de GPU, la CPU calcula cuadros al mismo tiempo
    modo: str = "normal"     # "normal" (cuadro por cuadro) o "exponencial" (tira, ver core/exponencial.py)
    vueltas: float = 0.0     # solo en modo exponencial: vueltas que gira la imagen en todo el zoom

    @property
    def cuadros_zoom(self) -> int:
        return max(1, round(self.fps * self.duracion))

    @property
    def cuadros_pausa(self) -> int:
        return max(0, round(self.fps * self.pausa))

    @property
    def cuadros(self) -> int:
        return self.cuadros_zoom + self.cuadros_pausa


def nombre_cuadro(indice: int) -> str:
    """frame_1.png, frame_2.png, ... (sin ceros adelante, empezando en 1)."""
    return f"frame_{indice + 1}.png"


def _suavizar(u: Decimal) -> Decimal:
    """Smoothstep: velocidad cero al principio y al final, máxima en el medio."""
    return u * u * (3 - 2 * u)


def planificar_zoom(xmin: Decimal, xmax: Decimal, ymin: Decimal, ymax: Decimal,
                    ancho_px: int, alto_px: int, cuadros: int, max_iter: int,
                    suavizado: bool = False) -> Iterator[Cuadro]:
    """Genera los cuadros de un zoom que arranca con el fractal entero y termina en la vista dada.

    - El ancho de la vista baja en forma geométrica (multiplicándose siempre por el mismo
      factor), así el zoom se ve a velocidad constante. Con `suavizado`, arranca y frena suave.
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
        if suavizado:
            t = _suavizar(t)
        ancho = ancho_0 * (t * log_razon).exp()
        if i == cuadros - 1:
            ancho = ancho_f                        # que termine exacto, sin error de redondeo
        # Posición del destino en pantalla (en anchos de vista desde el centro): arranca donde
        # cae en la vista inicial y se mueve hasta el centro en el último cuadro.
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


# --- Plan guardado en la carpeta del video, para poder retomarlo ---

def guardar_plan(carpeta: str, pedido, config: ConfigVideo, nombre_paleta: str) -> None:
    """Guarda todo lo necesario para volver a generar exactamente los mismos cuadros."""
    datos = {
        "version": 1,
        "pedido": {
            "xmin": str(pedido.xmin), "xmax": str(pedido.xmax),
            "ymin": str(pedido.ymin), "ymax": str(pedido.ymax),
            "max_iter": pedido.max_iter, "tipo_fractal": pedido.tipo_fractal,
            "tipo_calculo": pedido.tipo_calculo, "real": pedido.real, "imag": pedido.imag,
            "paleta": pedido.paleta, "nombre_paleta": nombre_paleta,
            "clase_equiv": pedido.clase_equiv, "opciones": asdict(pedido.opciones),
        },
        "config": asdict(config),
    }
    with open(os.path.join(carpeta, ARCHIVO_PLAN), "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)


def cargar_plan(carpeta: str) -> tuple[dict, ConfigVideo] | None:
    """Lee el plan de un video. Devuelve (datos del pedido, config) o None si no hay plan válido."""
    ruta = os.path.join(carpeta, ARCHIVO_PLAN)
    try:
        with open(ruta, encoding="utf-8") as f:
            datos = json.load(f)
        return datos["pedido"], ConfigVideo(**datos["config"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        log.info("No hay un plan de video válido en %s: %s", carpeta, e)
        return None


def cuadros_hechos(carpeta: str, config: ConfigVideo) -> int:
    """Cuántos cuadros del video ya están guardados."""
    return config.cuadros - len(cuadros_faltantes(carpeta, config))


def cuadros_faltantes(carpeta: str, config: ConfigVideo) -> list[int]:
    """Índices (desde 0) de los cuadros que no están en la carpeta."""
    return [i for i in range(config.cuadros) if not os.path.exists(os.path.join(carpeta, nombre_cuadro(i)))]


def describir_cuadros(indices: list[int]) -> str:
    """Índices desde 0 -> texto con los números de cuadro (desde 1) agrupados: "17, 230-235"."""
    partes = []
    for i in sorted(set(indices)):
        if partes and i == partes[-1][1] + 1:
            partes[-1][1] = i
        else:
            partes.append([i, i])
    return ", ".join(f"{a + 1}" if a == b else f"{a + 1}-{b + 1}" for a, b in partes)


def leer_cuadros(texto: str, total: int) -> list[int]:
    """Texto como "17, 230-235" (números de cuadro desde 1) -> índices desde 0, ordenados.
    Lanza ValueError con un mensaje claro si algo no se entiende o se sale del video."""
    indices = set()
    for parte in texto.replace(";", ",").split(","):
        parte = parte.strip()
        if not parte:
            continue
        desde, _, hasta = parte.partition("-")
        try:
            a, b = int(desde), int(hasta) if hasta else int(desde)
        except ValueError:
            raise ValueError(f"No entiendo '{parte}': usá números y rangos, como 17, 230-235")
        if a > b:
            a, b = b, a
        if a < 1 or b > total:
            raise ValueError(f"'{parte}' se sale del video: los cuadros van de 1 a {total}")
        indices.update(range(a - 1, b))
    return sorted(indices)


def armar_mp4(carpeta: str, fps: int, nombre: str = "zoom.mp4") -> str | None:
    """Une los frame_1.png, frame_2.png... en un mp4 con ffmpeg, si está instalado. Devuelve la ruta o None."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        log.warning("ffmpeg no está instalado: quedan solo los PNG en %s", carpeta)
        return None
    salida = os.path.join(carpeta, nombre)
    comando = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-start_number", "1",
               "-i", os.path.join(carpeta, "frame_%d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               # libx264 + yuv420p necesita ancho y alto pares
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", salida]
    resultado = subprocess.run(comando, capture_output=True, text=True)
    if resultado.returncode != 0:
        log.error("ffmpeg falló: %s", resultado.stderr.strip())
        return None
    return salida
