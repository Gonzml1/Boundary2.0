"""Mapa exponencial: videos de zoom calculando cada zona del fractal una sola vez.

En coordenadas (log del radio, ángulo) alrededor del centro del zoom, acercar la imagen es
desplazarse. Se calcula una "tira": cada fila es un anillo alrededor del centro, del radio que
cubre el primer cuadro (el fractal entero) al más chico que hace falta en el último. Después
cada cuadro del video se arma leyendo la tira, sin calcular ninguna iteración.

La tira se guarda por franjas (PNG ya coloreados) en la carpeta del video: se puede retomar, y
cambiar duración, fps, frenado, pausa o rotación no requiere recalcularla.
"""
import os
import math
import logging
from dataclasses import dataclass, replace
from decimal import Decimal

import numpy as np
from PIL import Image

from .modulo_de_calculo_fractales import GrillaPolar, FRACTAL_REGISTRY
from .render import colorear, PedidoRender, a_numpy_uint8, COLOREAR_EN_GPU

log = logging.getLogger(__name__)

FILAS_POR_FRANJA = 512
# Cada franja se calcula y colorea de a partes de a lo sumo estos puntos: colorear necesita
# varios arreglos temporales por punto, y de una vez (7 M de puntos con antialiasing) eran picos
# de ~1 GB que podían chocar con el límite de memoria de Windows. Con 2,5 M el pico baja ~40 % y
# además va más rápido (cada parte entra en un solo lote de la GPU). Las partes usan la misma
# tabla BLA y el mismo modo que la franja entera, así que el resultado es idéntico.
PUNTOS_POR_PARTE = 2_500_000
# Puntos de la tira por píxel en el borde del cuadro (en el centro siempre hay más): 1 alcanza
# para que se vea nítido; con antialiasing se duplica y cada píxel promedia 2x2 muestras
DENSIDAD = 1.0
DENSIDAD_ANTIALIASING = 2.0
# Región que tiene que entrar entera en el primer cuadro (el conjunto de Mandelbrot)
CONJUNTO_X = (Decimal('-2.1'), Decimal('0.6'))
CONJUNTO_Y = Decimal('1.25')
LN2 = math.log(2.0)


@dataclass(frozen=True)
class GeometriaTira:
    centro_re: Decimal
    centro_im: Decimal
    ancho_px: int
    alto_px: int
    n_angulos: int             # columnas de la tira
    filas: int                 # filas de la tira (radios, del más grande al más chico)
    ln_radio_max: float        # ln del radio de la fila 0
    du: float                  # paso en ln(radio) entre filas = paso angular (muestreo parejo)
    ln_paso_inicial: float     # ln del tamaño de un píxel en el primer cuadro
    ln_paso_final: float       # ... y en el último
    ancho_inicial: Decimal

    @property
    def franjas(self) -> int:
        return -(-self.filas // FILAS_POR_FRANJA)

    @property
    def puntos(self) -> int:
        return self.filas * self.n_angulos


def _ln(valor: Decimal) -> float:
    return float(valor.ln())


def geometria_tira(pedido: PedidoRender, ancho_px: int, alto_px: int) -> GeometriaTira:
    """La tira para un zoom centrado en el centro de la vista del pedido, desde una vista que
    muestra el conjunto entero hasta la vista del pedido, en cuadros de ancho_px x alto_px."""
    cx, cy = (pedido.xmin + pedido.xmax) / 2, (pedido.ymin + pedido.ymax) / 2
    ancho_final = pedido.xmax - pedido.xmin
    aspecto = Decimal(alto_px) / Decimal(ancho_px)
    # Primer cuadro: centrado en el destino y con el conjunto entero adentro
    medio_ancho = max(cx - CONJUNTO_X[0], CONJUNTO_X[1] - cx, (abs(cy) + CONJUNTO_Y) / aspecto)
    ancho_inicial = max(2 * medio_ancho, ancho_final)

    radio_px = math.hypot((ancho_px - 1) / 2, (alto_px - 1) / 2)     # centro -> esquina, en píxeles
    densidad = DENSIDAD_ANTIALIASING if pedido.opciones.antialiasing else DENSIDAD
    n_angulos = int(math.ceil(2 * math.pi * radio_px * densidad / 64) * 64)
    du = 2 * math.pi / n_angulos
    ln_paso_inicial = _ln(ancho_inicial / (ancho_px - 1))
    ln_paso_final = _ln(ancho_final / (ancho_px - 1))
    ln_radio_max = ln_paso_inicial + math.log(radio_px * 1.01)
    ln_radio_min = ln_paso_final + math.log(0.25)
    filas = int(math.ceil((ln_radio_max - ln_radio_min) / du)) + 2
    return GeometriaTira(cx, cy, ancho_px, alto_px, n_angulos, filas, ln_radio_max, du,
                         ln_paso_inicial, ln_paso_final, ancho_inicial)


def ln_pasos_cuadros(geo: GeometriaTira, cuadros: int, suavizado: bool) -> list[float]:
    """ln del tamaño de píxel de cada cuadro: zoom geométrico (o suavizado) de principio a fin."""
    resultado = []
    for i in range(cuadros):
        t = i / max(1, cuadros - 1)
        if suavizado:
            t = t * t * (3 - 2 * t)
        resultado.append(geo.ln_paso_inicial + t * (geo.ln_paso_final - geo.ln_paso_inicial))
    return resultado


def metodo_para_tira(pedido: PedidoRender) -> str:
    """La tira se calcula con perturbación (es lo que sabe calcular grillas polares)."""
    metodos = FRACTAL_REGISTRY.get(pedido.tipo_fractal, {})
    if pedido.tipo_calculo.startswith("perturbacion") and pedido.tipo_calculo in metodos:
        return pedido.tipo_calculo
    for candidato in ("perturbacion_gpu", "perturbacion_cpu"):
        if candidato in metodos:
            return candidato
    raise ValueError(f"El modo rápido necesita un método de perturbación para '{pedido.tipo_fractal}'")


# ============================================================
#                    CÁLCULO DE LA TIRA
# ============================================================

def _radios(geo: GeometriaTira, desde: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Radio de las filas desde..desde+n como (mantisa, exponente), sin underflow."""
    log2_r = (geo.ln_radio_max - (desde + np.arange(n)) * geo.du) / LN2
    e = np.floor(log2_r).astype(np.int64) + 1
    m = np.exp2(log2_r - e)                     # en [0.5, 1)
    return np.ascontiguousarray(m, dtype=np.float64), np.ascontiguousarray(e, dtype=np.int32)


def _angulos(geo: GeometriaTira) -> tuple[np.ndarray, np.ndarray]:
    th = np.arange(geo.n_angulos) * (2 * math.pi / geo.n_angulos)
    return np.ascontiguousarray(np.cos(th)), np.ascontiguousarray(np.sin(th))


def _decimal_fe(m: float, e: int) -> Decimal:
    return Decimal(m) * Decimal(2) ** int(e)


def calcular_franja(motor, pedido: PedidoRender, geo: GeometriaTira, franja: int) -> np.ndarray:
    """Calcula y colorea una franja de la tira: (filas, n_angulos, 3) uint8, fila 0 = radio mayor."""
    desde = franja * FILAS_POR_FRANJA
    n = min(FILAS_POR_FRANJA, geo.filas - desde)
    radio_m, radio_e = _radios(geo, desde, n)
    cos_t, sin_t = _angulos(geo)
    dtheta = math.frexp(2 * math.pi / geo.n_angulos)
    du = Decimal(geo.du)
    m_ult, e_ult = _radios(geo, geo.filas - 1, 1)
    # Valores de la franja entera (no de cada parte): misma precisión, mismo modo y misma tabla BLA
    paso_min = _decimal_fe(radio_m[-1], radio_e[-1]) * du
    paso_bits = _decimal_fe(m_ult[0], e_ult[0]) * du
    e_radio_max = int(radio_e[0])
    op = pedido.opciones
    metodo = metodo_para_tira(pedido)
    salida = np.empty((n, geo.n_angulos, 3), dtype=np.uint8)
    filas_por_parte = max(1, PUNTOS_POR_PARTE // geo.n_angulos)
    for inicio in range(0, n, filas_por_parte):
        fin = min(n, inicio + filas_por_parte)
        motor.actualizar_fractal(geo.centro_re, geo.centro_re, geo.centro_im, geo.centro_im,
                                 geo.n_angulos, fin - inicio, pedido.max_iter, metodo,
                                 pedido.tipo_fractal, pedido.real, pedido.imag)
        motor.pedir_suave = op.suave
        motor.pedir_derivada = op.iluminacion
        motor.dejar_en_gpu = COLOREAR_EN_GPU
        motor.grilla_polar = GrillaPolar(
            centro=(geo.centro_re, geo.centro_im),
            radio_m=np.ascontiguousarray(radio_m[inicio:fin]), radio_e=np.ascontiguousarray(radio_e[inicio:fin]),
            cos_t=cos_t, sin_t=sin_t, dtheta=dtheta,
            paso_min=paso_min, paso_bits=paso_bits, e_radio_max=e_radio_max,
        )
        try:
            resultado = motor.calcular_fractal()
        finally:
            motor.grilla_polar = None
        rgb, _ = colorear(resultado, pedido)
        salida[inicio:fin] = a_numpy_uint8(rgb)
        del resultado, rgb
    return salida


def ruta_franja(carpeta_tira: str, franja: int) -> str:
    return os.path.join(carpeta_tira, f"franja_{franja:05d}.png")


def guardar_franja(ruta: str, rgb: np.ndarray) -> None:
    temporal = ruta + ".parcial"
    Image.fromarray(rgb).save(temporal, format="PNG", compress_level=1)
    os.replace(temporal, ruta)


def franjas_para_cuadro(geo: GeometriaTira, ln_paso: float) -> range:
    """Franjas de la tira que hacen falta para armar un cuadro con ese tamaño de píxel."""
    k_lo, k_hi = _filas_para_cuadro(geo, ln_paso)
    return range(k_lo // FILAS_POR_FRANJA, k_hi // FILAS_POR_FRANJA + 1)


def _filas_para_cuadro(geo: GeometriaTira, ln_paso: float) -> tuple[int, int]:
    w, h = geo.ancho_px, geo.alto_px
    r_max = math.hypot((w - 1) / 2 + 0.5, (h - 1) / 2 + 0.5)
    k0 = (geo.ln_radio_max - ln_paso) / geo.du             # fila de radio = 1 píxel
    k_lo = max(0, int(math.floor(k0 - math.log(r_max) / geo.du)) - 1)
    k_hi = min(geo.filas - 1, int(math.ceil(k0 - math.log(0.25) / geo.du)) + 1)
    return k_lo, k_hi


# ============================================================
#                    ARMADO DE CUADROS
# ============================================================

def armar_cuadro(geo: GeometriaTira, ln_paso: float, rotacion: float, cargar_franja, antialiasing: bool,
                 xp=np) -> np.ndarray:
    """Arma un cuadro leyendo la tira. cargar_franja(i) devuelve la franja i como arreglo de xp.
    Devuelve (alto, ancho, 3) uint8 con la fila 0 abajo (ymin), como el resto de la app."""
    w, h, n = geo.ancho_px, geo.alto_px, geo.n_angulos
    k_lo, k_hi = _filas_para_cuadro(geo, ln_paso)
    f_lo, f_hi = k_lo // FILAS_POR_FRANJA, k_hi // FILAS_POR_FRANJA
    banda = xp.concatenate([cargar_franja(f) for f in range(f_lo, f_hi + 1)], axis=0)
    filas_banda = banda.shape[0]
    plano = banda.reshape(-1, 3).astype(xp.float32)
    inicio = f_lo * FILAS_POR_FRANJA
    k0 = (geo.ln_radio_max - ln_paso) / geo.du - inicio      # fila (en la banda) de radio = 1 píxel

    ys, xs = xp.mgrid[0:h, 0:w].astype(xp.float32)
    submuestras = ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)) if antialiasing else ((0.0, 0.0),)
    acumulado = xp.zeros((h, w, 3), dtype=xp.float32)
    for ox, oy in submuestras:
        px = xs - (w - 1) / 2 + ox
        py = ys - (h - 1) / 2 + oy
        r = xp.maximum(xp.sqrt(px * px + py * py), 0.25)
        k = xp.clip(xp.float32(k0) - xp.log(r) / xp.float32(geo.du), 0, filas_banda - 1.001)
        col = xp.mod((xp.arctan2(py, px) + xp.float32(rotacion)) * xp.float32(n / (2 * math.pi)), n)
        k_ent = k.astype(xp.int64)
        c_ent = col.astype(xp.int64) % n
        fk = (k - k_ent)[..., None]
        fc = (col - xp.floor(col))[..., None]
        c_sig = (c_ent + 1) % n
        k_sig = xp.minimum(k_ent + 1, filas_banda - 1)
        a = plano[k_ent * n + c_ent]
        b = plano[k_ent * n + c_sig]
        c = plano[k_sig * n + c_ent]
        d = plano[k_sig * n + c_sig]
        acumulado += (a * (1 - fc) + b * fc) * (1 - fk) + (c * (1 - fc) + d * fc) * fk
    rgb = xp.clip(acumulado / len(submuestras) + 0.5, 0, 255).astype(xp.uint8)
    return rgb.get() if hasattr(rgb, "get") else rgb
