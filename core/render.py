"""Del pedido a la imagen: calcula el fractal y lo colorea.

Lo usan por igual la pantalla (desde el hilo de render), "Guardar imagen" y el video, así que
las opciones de coloreo se ven igual en los tres lados.
"""
import os
import time
import logging
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from decimal import Decimal

import numpy as np
from PIL import Image

from .modulo_de_calculo_fractales import calculos_mandelbrot, Resultado, liberar_memoria_gpu, es_falta_de_memoria
from .paletas import PALETTE_REGISTRY

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpcionesColor:
    suave: bool = False            # coloreo suave (iteración continua)
    iluminacion: bool = False      # relieve con luz + bordes por estimador de distancia
    antialiasing: bool = False     # supermuestreo 2x2
    angulo_luz: float = 45.0       # grados, medido en el plano complejo
    altura_luz: float = 1.5        # más alto = sombras más suaves
    interior: tuple | None = None  # color (r, g, b) del interior del conjunto; None = el de la paleta


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
    # Contra qué max_iter se normalizan los colores (None = el mismo max_iter del cálculo)
    max_iter_color: int | None = None


@dataclass
class ImagenRender:
    rgb: np.ndarray                # (alto, ancho, 3) uint8; la fila 0 es ymin
    segundos: float
    aplicadas: frozenset           # opciones que el backend pudo aplicar ("suave", "iluminacion", ...)
    recalculado: bool = True       # False si se reusó un cálculo anterior y solo se recoloreó


FACTOR_ANTIALIASING = 2


class Renderizador:
    """Calcula y colorea pedidos con su propio motor (cada hilo tiene que usar el suyo).

    Guarda los últimos resultados calculados: si llega un pedido que solo cambia el coloreo
    (paleta, ciclo, luz, color del interior), se recolorea sin volver a calcular el fractal.
    """

    def __init__(self, tam_cache: int = 8):
        self.motor = calculos_mandelbrot(Decimal(-2), Decimal(1), Decimal(-1), Decimal(1), 1, 1, 1, "", "")
        self.tam_cache = tam_cache
        # clave de cálculo -> (pidió suave, pidió derivada, Resultado)
        self._cache: OrderedDict = OrderedDict()

    def cancelar(self) -> None:
        """Abandona el cálculo en curso (se puede llamar desde otro hilo)."""
        self.motor.cancelar()

    def reiniciar_cancelacion(self) -> None:
        self.motor.reiniciar_cancelacion()

    def renderizar(self, pedido: PedidoRender) -> ImagenRender:
        inicio = time.perf_counter()
        op = pedido.opciones
        factor = FACTOR_ANTIALIASING if op.antialiasing else 1
        clave = (pedido.xmin, pedido.xmax, pedido.ymin, pedido.ymax, pedido.ancho * factor,
                 pedido.alto * factor, pedido.max_iter, pedido.tipo_fractal, pedido.tipo_calculo,
                 pedido.real, pedido.imag)

        guardado = self._cache.get(clave)
        if guardado and guardado[0] >= op.suave and guardado[1] >= op.iluminacion:
            resultado, recalculado = guardado[2], False
            self._cache.move_to_end(clave)
        else:
            try:
                resultado = _calcular(self.motor, pedido, factor)
            except Exception as e:
                if not es_falta_de_memoria(e):
                    raise
                # Sin memoria (GPU o RAM bloqueada): se sueltan los cachés y se reintenta una vez
                log.warning("Falta de memoria al calcular (%s): se libera memoria y se reintenta", e)
                self._cache.clear()
                self.motor._cache_orbita = None
                liberar_memoria_gpu()
                resultado = _calcular(self.motor, pedido, factor)
            recalculado = True
            if self.tam_cache:
                self._cache[clave] = (op.suave, op.iluminacion, resultado)
                while len(self._cache) > self.tam_cache:
                    self._cache.popitem(last=False)

        rgb, aplicadas = _componer(resultado, pedido, factor)
        return ImagenRender(rgb, time.perf_counter() - inicio, aplicadas, recalculado)


def renderizar(motor: calculos_mandelbrot, pedido: PedidoRender) -> ImagenRender:
    """Calcula y colorea el pedido con el motor dado, sin caché."""
    inicio = time.perf_counter()
    factor = FACTOR_ANTIALIASING if pedido.opciones.antialiasing else 1
    rgb, aplicadas = _componer(_calcular(motor, pedido, factor), pedido, factor)
    return ImagenRender(rgb, time.perf_counter() - inicio, aplicadas)


def _calcular(motor: calculos_mandelbrot, pedido: PedidoRender, factor: int) -> Resultado:
    motor.actualizar_fractal(
        pedido.xmin, pedido.xmax, pedido.ymin, pedido.ymax,
        pedido.ancho * factor, pedido.alto * factor, pedido.max_iter,
        pedido.tipo_calculo, pedido.tipo_fractal, pedido.real, pedido.imag
    )
    motor.pedir_suave = pedido.opciones.suave
    motor.pedir_derivada = pedido.opciones.iluminacion
    return motor.calcular_fractal()


def _componer(resultado: Resultado, pedido: PedidoRender, factor: int) -> tuple[np.ndarray, frozenset]:
    rgb, aplicadas = colorear(resultado, pedido)
    if factor > 1:
        # Promedio de cada bloque factor x factor (el color se promedia después de colorear)
        rgb = rgb.reshape(pedido.alto, factor, pedido.ancho, factor, 3).mean(axis=(1, 3))
        aplicadas = aplicadas | {"antialiasing"}
    rgb = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    return rgb, frozenset(aplicadas)


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
    if op.interior is not None:
        rgb[interior] = op.interior
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


def guardar_imagen(ruta: str, rgb: np.ndarray) -> None:
    """Guarda la imagen en RGB (sin canal de transparencia), en el formato de la extensión.

    Se escribe a un archivo temporal y después se renombra: si se corta a la mitad nunca
    queda un archivo roto con el nombre final (el video usa eso para saber qué cuadros
    ya están hechos)."""
    formato = "JPEG" if ruta.lower().endswith((".jpg", ".jpeg")) else "PNG"
    temporal = ruta + ".parcial"
    # La fila 0 de la imagen es ymin: se invierte para que no salga al revés
    Image.fromarray(rgb[::-1]).save(temporal, format=formato)
    os.replace(temporal, ruta)


def con_tamano(pedido: PedidoRender, ancho: int, alto: int, **cambios) -> PedidoRender:
    """El mismo pedido con otro tamaño en píxeles (y otros cambios opcionales)."""
    return replace(pedido, ancho=ancho, alto=alto, **cambios)
