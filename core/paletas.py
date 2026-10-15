from functools import lru_cache
from typing import Callable

import numpy as np
import matplotlib

PALETTE_REGISTRY: list[tuple[str, Callable]] = []

def register_palette(palette_name: str):
    def deco(fn):
        PALETTE_REGISTRY.append((palette_name, fn))
        return fn
    return deco


# Todas las paletas reciben norm = iteración / max_iter. Con el coloreo suave la iteración
# tiene parte fraccionaria, así que las tablas de colores (LUT) se interpolan en vez de
# indexarse con un entero: con valores enteros el resultado es el mismo de siempre.

@lru_cache(maxsize=64)
def _lut(nombre: str, n: int) -> np.ndarray:
    """LUT de n colores (n, 3) en float 0..255 del colormap de Matplotlib (se cachea)."""
    return matplotlib.colormaps[nombre].resampled(n)(np.arange(n))[:, :3] * 255.0


def _interpolar(lut: np.ndarray, pos: np.ndarray, ciclico: bool) -> np.ndarray:
    """Color en la posición `pos` (en unidades de entradas de la LUT), interpolando linealmente."""
    n = len(lut)
    base = np.floor(pos)
    f = (pos - base)[..., None]
    i0 = base.astype(np.int64)
    if ciclico:
        i0 %= n
        i1 = (i0 + 1) % n
    else:
        i0 = np.clip(i0, 0, n - 1)
        i1 = np.minimum(i0 + 1, n - 1)
    return (lut[i0] * (1.0 - f) + lut[i1] * f).astype(np.uint8)


def _paleta_lineal(nombre: str, norm: np.ndarray) -> np.ndarray:
    """norm ∈ [0, 1] recorre una vez el colormap (LUT de 256). Sin interpolar: con 256 niveles
    el suave ya se nota, y así los colormaps cualitativos (Set1, Paired...) no mezclan colores."""
    indices = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
    return _lut(nombre, 256)[indices].astype(np.uint8)


def _paleta_ciclica(nombre: str, norm: np.ndarray, max_iter: int, ciclo: int) -> np.ndarray:
    """El colormap se repite cada `ciclo` iteraciones."""
    iters = np.clip(norm * max_iter, 0, max_iter)
    return _interpolar(_lut(nombre, ciclo), iters, ciclico=True)


@register_palette("Iteraciones variables (Twilight Shifted)")
def _paleta_iters_variable_twilight(norm: np.ndarray, max_iter: int, clase_equiv: int, t_actual: float=0.0, thickness: float=0.0) -> np.ndarray:
    """Twilight shifted que se repite cada `clase_equiv` iteraciones."""
    return _paleta_ciclica('twilight_shifted', norm, max_iter, max(2, clase_equiv))


@register_palette("Rojo→Amarillo→Blanco")
def _paleta_rojo_amarillo(norm: np.ndarray, max_iter: int, clase_equiv: int, t_actual: float=0.0, thickness: float=0.0) -> np.ndarray:
    """
    Rojo→Amarillo→Blanco:
    - R siempre 255
    - G crece linealmente de 0→255 para norm en [0,0.5], luego se mantiene 255
    - B se activa solo para norm≥0.5, crece de 0→255 en [0.5,1]
    """
    r = np.uint8(255 * np.ones_like(norm))
    g = np.uint8((np.clip(norm * 2, 0, 1) * 255).clip(0, 255))
    b = np.uint8((np.clip((norm - 0.5) * 2, 0, 1) * 255).clip(0, 255))
    return np.dstack([r, g, b])

@register_palette("HSV")
def _paleta_hsv(norm: np.ndarray, max_iter: int, clase_equiv: int, t_actual: float=0.0, thickness: float=0.0) -> np.ndarray:
    """Colormap 'hsv' de Matplotlib recorrido una vez."""
    return _paleta_lineal('hsv', norm)

@register_palette("Púrpura Psicodélica")
def _paleta_psicodelica(norm: np.ndarray, max_iter: int, clase_equiv: int, t_actual: float=0.0, thickness: float=0.0) -> np.ndarray:
    """
    Púrpura psicodélica usando funciones sinusoidales:
    - Tres ciclos de color, fases desplazadas en R,G,B
    """
    r = np.uint8((0.5 + 0.5 * np.sin(2 * np.pi * norm * 3 + 0)) * 255)
    g = np.uint8((0.5 + 0.5 * np.sin(2 * np.pi * norm * 3 + 2)) * 255)
    b = np.uint8((0.5 + 0.5 * np.sin(2 * np.pi * norm * 3 + 4)) * 255)
    return np.dstack([r, g, b])

@register_palette("Accent")
def _paleta_accent(norm: np.ndarray, max_iter: int, clase_equiv: int, t_actual: float=0.0, thickness: float=0.0) -> np.ndarray:
    """Colormap 'Accent' de Matplotlib (colores brillantes)."""
    return _paleta_lineal('Accent', norm)

@register_palette("Dark2")
def _paleta_dark2(norm: np.ndarray, max_iter: int, clase_equiv: int, t_actual: float=0.0, thickness: float=0.0) -> np.ndarray:
    """Colormap 'Dark2' de Matplotlib (colores oscuros y saturados)."""
    return _paleta_lineal('Dark2', norm)

@register_palette("Set1")
def _paleta_set1(norm: np.ndarray, max_iter: int, clase_equiv: int, t_actual: float=0.0, thickness: float=0.0) -> np.ndarray:
    """Colormap 'Set1' de Matplotlib (colores brillantes y saturados)."""
    return _paleta_lineal('Set1', norm)

@register_palette("Set2")
def _paleta_set2(norm: np.ndarray, max_iter: int, clase_equiv: int, t_actual: float=0.0, thickness: float=0.0) -> np.ndarray:
    """Colormap 'Set2' de Matplotlib (colores suaves y agradables)."""
    return _paleta_lineal('Set2', norm)

@register_palette("Set3")
def _paleta_set3(norm: np.ndarray, max_iter: int, clase_equiv: int, t_actual: float=0.0, thickness: float=0.0) -> np.ndarray:
    """Colormap 'Set3' de Matplotlib (colores variados y agradables)."""
    return _paleta_lineal('Set3', norm)

@register_palette("prism")
def _paleta_prism(norm: np.ndarray, max_iter: int, clase_equiv: int, t_actual: float=0.0, thickness: float=0.0) -> np.ndarray:
    """Colormap 'prism' de Matplotlib."""
    return _paleta_lineal('prism', norm)

@register_palette("Prism LUT")
def _palette_prism_from_norm(norm: np.ndarray, max_iter: int, clase_equiv: int, t_actual: float=0.0, thickness: float=0.0) -> np.ndarray:
    """Paleta 'prism' con una LUT de 256 colores."""
    return _paleta_lineal('prism', norm)

@register_palette("Paired")
def _paleta_paired(norm: np.ndarray, max_iter: int, clase_equiv: int, t_actual: float=0.0, thickness: float=0.0) -> np.ndarray:
    """Colormap 'Paired' de Matplotlib (cualitativo, pares de colores contrastantes)."""
    return _paleta_lineal('Paired', norm)

@register_palette("Pastel1")
def _paleta_pastel1(norm: np.ndarray, max_iter: int, clase_equiv: int, t_actual: float=0.0, thickness: float=0.0) -> np.ndarray:
    """Colormap 'Pastel1' de Matplotlib (cualitativo, colores suaves)."""
    return _paleta_lineal('Pastel1', norm)

@register_palette("Iteraciones (HSV ciclo 64)")
def _paleta_iters_hsv(norm: np.ndarray, max_iter: int, clase_equiv: int, t_actual: float=0.0, thickness: float=0.0) -> np.ndarray:
    """HSV que se repite cada 64 iteraciones."""
    return _paleta_ciclica('hsv', norm, max_iter, 64)


@register_palette("Grises cíclico")
def _paleta_grises_ciclico(norm: np.ndarray, max_iter: int, clase_equiv: int, t_actual: float=0.0, thickness: float=0.0) -> np.ndarray:
    """Rampa de gris de 0 a 255 que se repite cada 64 iteraciones."""
    ciclo = 64
    pos = np.clip(norm * max_iter, 0, max_iter) % ciclo
    gray = np.uint8((pos / (ciclo - 1) * 255).clip(0, 255))
    return np.dstack([gray, gray, gray])


@register_palette("Iteraciones (Plasma ciclo 64)")
def _paleta_iters_plasma(norm: np.ndarray, max_iter: int, clase_equiv: int, t_actual: float=0.0, thickness: float=0.0) -> np.ndarray:
    """Plasma que se repite cada 64 iteraciones."""
    return _paleta_ciclica('plasma', norm, max_iter, 64)


@register_palette("Iteraciones (Viridis ciclo 64)")
def _paleta_iters_viridis(norm: np.ndarray, max_iter: int, clase_equiv: int, t_actual: float=0.0, thickness: float=0.0) -> np.ndarray:
    """Viridis que se repite cada 64 iteraciones."""
    return _paleta_ciclica('viridis', norm, max_iter, 64)

@register_palette("Bandas RGB")
def _paleta_bandas_rgb(norm: np.ndarray, max_iter: int, clase_equiv: int, t_actual: float=0.0, thickness: float=0.0) -> np.ndarray:
    """
    Bandas semilineales: divide norm en 3 franjas, con degradado lineal dentro de cada franja:
    - franja0: rojo crece de 0→1
    - franja1: verde crece de 0→1
    - franja2: azul crece de 0→1
    """
    pos = norm * 3  # pos ∈ [0,3)
    # En la primera porción (pos < 1): r=pos, g=0,b=0
    # Segunda (1 ≤ pos < 2): r=2-pos, g=pos-1, b=0
    # Tercera (2 ≤ pos < 3): r=0, g=3-pos, b=pos-2
    r = np.where(pos < 1, pos, np.where(pos < 2, 2 - pos, 0))
    g = np.where(pos < 1, 0, np.where(pos < 2, pos - 1, 3 - pos))
    b = np.where(pos < 2, 0, pos - 2)
    rgb = np.dstack([r.clip(0,1), g.clip(0,1), b.clip(0,1)])
    return np.uint8(rgb * 255)
