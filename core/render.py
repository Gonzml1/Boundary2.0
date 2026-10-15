"""Del pedido a la imagen: calcula el fractal y lo colorea.

Lo usan por igual la pantalla (desde el hilo de render), "Guardar imagen" y el video, así que
las opciones de coloreo se ven igual en los tres lados.
"""
import time
import logging
from dataclasses import dataclass, field, replace
from decimal import Decimal

import numpy as np

from .modulo_de_calculo_fractales import calculos_mandelbrot, Resultado
from .paletas import PALETTE_REGISTRY

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpcionesColor:
    suave: bool = False            # coloreo suave (iteración continua)
    iluminacion: bool = False      # relieve con luz + bordes por estimador de distancia
    antialiasing: bool = False     # supermuestreo 2x2
    angulo_luz: float = 45.0       # grados, medido en el plano complejo
    altura_luz: float = 1.5        # más alto = sombras más suaves


@dataclass(frozen=True)
class PedidoRender:
    xmin: Decimal
    xmax: Decimal
    ymin: Decimal
    ymax: Decimal
    ancho: int
    alto: int
    max_iter: int
    tipo_fractal: str
    tipo_calculo: str
    real: float = 0.0              # parámetro de la fórmula (p. ej. el c de Julia)
    imag: float = 0.0
    paleta: int = 0
    clase_equiv: int = 128
    opciones: OpcionesColor = field(default_factory=OpcionesColor)
    # Contra qué max_iter se normalizan los colores (el video usa menos iteraciones en los
    # cuadros alejados pero colorea todos contra el máximo, para que los colores no salten)
    max_iter_color: int | None = None


@dataclass
class ImagenRender:
    rgb: np.ndarray                # (alto, ancho, 3) uint8; la fila 0 es ymin
    segundos: float
    aplicadas: frozenset           # opciones que el backend pudo aplicar ("suave", "iluminacion", ...)


FACTOR_ANTIALIASING = 2


def renderizar(motor: calculos_mandelbrot, pedido: PedidoRender) -> ImagenRender:
    """Calcula y colorea el pedido con el motor dado (cada hilo tiene que usar su propio motor)."""
    inicio = time.perf_counter()
    op = pedido.opciones
    factor = FACTOR_ANTIALIASING if op.antialiasing else 1
    ancho, alto = pedido.ancho * factor, pedido.alto * factor

    motor.actualizar_fractal(
        pedido.xmin, pedido.xmax, pedido.ymin, pedido.ymax,
        ancho, alto, pedido.max_iter, pedido.tipo_calculo, pedido.tipo_fractal,
        pedido.real, pedido.imag
    )
    motor.pedir_suave = op.suave
    motor.pedir_derivada = op.iluminacion
    resultado = motor.calcular_fractal()

    rgb, aplicadas = colorear(resultado, pedido)
    if factor > 1:
        # Promedio de cada bloque factor x factor (el color se promedia después de colorear)
        rgb = rgb.reshape(pedido.alto, factor, pedido.ancho, factor, 3).mean(axis=(1, 3))
        aplicadas = aplicadas | {"antialiasing"}
    rgb = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    return ImagenRender(rgb, time.perf_counter() - inicio, frozenset(aplicadas))


def colorear(resultado: Resultado, pedido: PedidoRender) -> tuple[np.ndarray, set]:
    """Aplica la paleta y, si se pidieron y el backend las calculó, el suavizado y la luz."""
    op = pedido.opciones
    aplicadas = set()
    max_iter_color = pedido.max_iter_color or pedido.max_iter

    iteraciones = np.asarray(resultado.iteraciones)
    interior = iteraciones >= pedido.max_iter
    if op.suave and resultado.suave is not None:
        valores = resultado.suave.astype(np.float64)
        aplicadas.add("suave")
    else:
        valores = iteraciones.astype(np.float64)
    valores[interior] = max_iter_color

    _, paleta = PALETTE_REGISTRY[pedido.paleta % len(PALETTE_REGISTRY)]
    rgb = paleta(valores / max_iter_color, max_iter_color, pedido.clase_equiv).astype(np.float32)

    if op.iluminacion and resultado.normal_x is not None:
        rgb *= _sombreado(resultado, op, interior)[..., None]
        aplicadas.add("iluminacion")
    return rgb, aplicadas


def _sombreado(resultado: Resultado, op: OpcionesColor, interior: np.ndarray) -> np.ndarray:
    """Factor de brillo por píxel: relieve iluminado (normal map) + bordes oscuros (distancia)."""
    # Relieve: la normal sale de z / (dz/dc). Luz en la dirección `angulo_luz`, a altura h.
    angulo = np.radians(op.angulo_luz)
    h = op.altura_luz
    t = (resultado.normal_x * np.cos(angulo) + resultado.normal_y * np.sin(angulo) + h) / (1.0 + h)
    t = np.clip(t, 0.0, 1.0)
    brillo = 0.3 + 0.9 * t
    # Bordes: a menos de un píxel del conjunto se oscurece, y el borde queda nítido
    cerca = np.clip(resultado.distancia, 0.0, 1.0)
    brillo *= 0.35 + 0.65 * np.sqrt(cerca)
    brillo[interior] = 1.0
    return brillo


def con_tamano(pedido: PedidoRender, ancho: int, alto: int, **cambios) -> PedidoRender:
    """El mismo pedido con otro tamaño en píxeles (y otros cambios opcionales)."""
    return replace(pedido, ancho=ancho, alto=alto, **cambios)
