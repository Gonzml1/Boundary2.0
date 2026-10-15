"""Ubicaciones guardadas: vista (con todos sus dígitos), iteraciones, método, paleta y opciones."""
import os
import json
import logging
from dataclasses import dataclass, asdict, field

log = logging.getLogger(__name__)

RUTA_FAVORITOS = os.path.join(os.path.dirname(__file__), os.pardir, "favoritos.json")


@dataclass
class Favorito:
    nombre: str
    # Los límites se guardan como texto: en zooms profundos tienen cientos de dígitos
    xmin: str
    xmax: str
    ymin: str
    ymax: str
    max_iter: int
    tipo_fractal: str
    tipo_calculo: str
    paleta: str                 # nombre de la paleta (el índice puede cambiar si se agregan paletas)
    clase_equiv: int
    real: str = "0"             # parámetro de la fórmula (Julia)
    imag: str = "0"
    opciones: dict = field(default_factory=dict)   # campos de OpcionesColor


def cargar(ruta: str = RUTA_FAVORITOS) -> list[Favorito]:
    if not os.path.exists(ruta):
        return []
    try:
        with open(ruta, encoding="utf-8") as f:
            datos = json.load(f)
    except (OSError, ValueError) as e:
        log.error("No se pudieron leer los favoritos de %s: %s", ruta, e)
        return []
    favoritos = []
    for d in datos:
        try:
            favoritos.append(Favorito(**d))
        except TypeError as e:
            log.warning("Favorito ignorado (%s): %s", e, d.get("nombre", "?"))
    return favoritos


def guardar(favoritos: list[Favorito], ruta: str = RUTA_FAVORITOS) -> None:
    # Se escribe a un temporal y se reemplaza, para no romper el archivo si algo falla a la mitad
    temporal = ruta + ".tmp"
    with open(temporal, "w", encoding="utf-8") as f:
        json.dump([asdict(fav) for fav in favoritos], f, ensure_ascii=False, indent=2)
    os.replace(temporal, ruta)
