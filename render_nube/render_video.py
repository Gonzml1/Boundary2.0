#!/usr/bin/env python3
"""ABoundary · render de videos sin interfaz.

Toma la carpeta de un video (con su video.json, el mismo que guarda la app) y la completa hasta
el mp4: calcula lo que falta con todas las GPUs de la máquina (y la CPU si se quiere), arma los
cuadros y los codifica en tramos de video que al final se unen. Se puede cortar con Ctrl+C y
volver a correr: sigue desde donde quedó. Ver render_nube/README.md.

    python render_nube/render_video.py CARPETA [--plan video.json] [opciones]
"""
import argparse
import gc
import json
import logging
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, RAIZ)

import numpy as np
from PIL import Image

from core.modulo_de_calculo_fractales import (BACKENDS_DISPONIBLES, RenderCancelado, es_falta_de_memoria,
                                             liberar_memoria_gpu)
from core.render import Renderizador, guardar_imagen
from core import exponencial
from core.video import (ARCHIVO_PLAN, cargar_plan, planificar_zoom, nombre_cuadro, metodos_para_video,
                        formatear_duracion, pedido_desde_plan)

HAY_GPU = BACKENDS_DISPONIBLES.get("gpu", False)
if HAY_GPU:
    import cupy as cp

log = logging.getLogger("render_nube")

CARPETA_TRAMOS = "tramos"
ARCHIVO_TRAMOS = "tramos.json"      # cómo se cortó el video en tramos (se respeta al retomar)
NOMBRE_VIDEO = "zoom.mp4"
FRANJAS_EN_MEMORIA = 64             # franjas cargadas por GPU al armar cuadros (como la app)
FRANJAS_EN_ESPERA = 6               # franjas calculadas esperando a guardarse, por cada unidad
HILOS_GUARDADO = 4
VENTANA_TIEMPOS = 30                # el tiempo restante se estima con los últimos pasos
CUADROS_ADELANTE = 24               # modo rápido: se leen del disco las franjas de los próximos cuadros
CUADROS_EN_COLA = 4                 # cuadros listos esperando a ffmpeg (la GPU arma el siguiente mientras)

CODIFICADORES = {
    # nombre: (argumentos de ffmpeg con {calidad}, calidad por defecto)
    "x264": (["-c:v", "libx264", "-preset", "medium", "-crf", "{calidad}"], 18),
    "x265": (["-c:v", "libx265", "-preset", "medium", "-crf", "{calidad}", "-tag:v", "hvc1"], 20),
    "nvenc-h264": (["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "{calidad}", "-b:v", "0"], 19),
    "nvenc-h265": (["-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "{calidad}", "-b:v", "0",
                    "-tag:v", "hvc1"], 21),
}


# ============================================================
#                         CONSOLA
# ============================================================

class Consola:
    """Una línea por evento, con el tiempo desde el arranque: se lee bien por SSH o en un log."""

    def __init__(self):
        self.inicio = time.monotonic()
        self._candado = threading.Lock()

    def linea(self, texto: str) -> None:
        s = int(time.monotonic() - self.inicio)
        with self._candado:
            print(f"[{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}] {texto}", flush=True)

    def texto(self, texto: str = "") -> None:
        with self._candado:
            print(texto, flush=True)


class Etapa:
    """Cuenta los pasos de una etapa (franjas, cuadros) e imprime cada uno con el tiempo restante."""

    def __init__(self, consola: Consola, nombre: str, total: int, unidad: str):
        self.consola, self.nombre, self.total, self.unidad = consola, nombre, total, unidad
        self.hechos = 0
        self.inicio = time.monotonic()
        self._tiempos = deque([(self.inicio, 0)], maxlen=VENTANA_TIEMPOS)
        self._candado = threading.Lock()

    def paso(self, detalle: str, cantidad: int = 1) -> None:
        with self._candado:
            self.hechos += cantidad
            ahora = time.monotonic()
            self._tiempos.append((ahora, self.hechos))
            (t0, a0), (t1, a1) = self._tiempos[0], self._tiempos[-1]
            falta = ""
            if a1 > a0 and self.hechos < self.total:
                falta = f" · faltan ~{formatear_duracion((t1 - t0) / (a1 - a0) * (self.total - self.hechos))}"
            ancho = len(str(self.total))
            texto = f"{self.nombre:<6} {self.hechos:>{ancho}}/{self.total} {self.unidad}  {detalle}{falta}"
        self.consola.linea(texto)

    def duracion(self) -> str:
        return formatear_duracion(time.monotonic() - self.inicio)


# ============================================================
#                  UNIDADES DE CÁLCULO (GPUs y CPU)
# ============================================================

class Unidad:
    """Una GPU (o la CPU) con su propio motor. Todo lo que hace corre dentro de su placa."""

    def __init__(self, nombre: str, metodo: str, placa: int | None):
        self.nombre, self.metodo, self.placa = nombre, metodo, placa
        self.renderizador = Renderizador(tam_cache=0)
        self.segundos_por_paso: float | None = None     # promedio de lo que tarda (para repartir)

    def medir(self, segundos: float) -> None:
        anterior = self.segundos_por_paso
        self.segundos_por_paso = segundos if anterior is None else 0.7 * anterior + 0.3 * segundos

    def ejecutar(self, funcion, *args) -> None:
        if self.placa is None:
            funcion(self, *args)
        else:
            with cp.cuda.Device(self.placa):
                funcion(self, *args)

    def xp(self):
        return cp if HAY_GPU else np


def crear_unidades(pedido, config, args, consola: Consola) -> list[Unidad]:
    metodos = metodos_para_video(pedido, config.usar_cpu and not args.sin_cpu)
    metodo_gpu = metodos[0]
    unidades = []
    if "_gpu" in metodo_gpu:
        if not HAY_GPU:
            raise SystemExit(f"El video usa '{metodo_gpu}' pero no hay GPU (CuPy/CUDA) en esta máquina.")
        cantidad = cp.cuda.runtime.getDeviceCount()
        placas = list(range(cantidad)) if args.gpus == "todas" else [int(g) for g in args.gpus.split(",")]
        for p in placas:
            if not 0 <= p < cantidad:
                raise SystemExit(f"No existe la GPU {p} (hay {cantidad}).")
            nombre = cp.cuda.runtime.getDeviceProperties(p)["name"].decode()
            unidades.append(Unidad(f"GPU {p}", metodo_gpu, p))
            consola.texto(f"  GPU {p}: {nombre}")
        if len(metodos) > 1:
            unidades.append(Unidad("CPU", metodos[1], None))
            consola.texto(f"  CPU: {os.cpu_count()} hilos (calcula a la par con '{metodos[1]}')")
    else:
        unidades.append(Unidad("CPU", metodo_gpu, None))
        consola.texto(f"  CPU: {os.cpu_count()} hilos")
    return unidades


# ============================================================
#                       EL RENDER
# ============================================================

class Render:
    def __init__(self, carpeta: str, args, consola: Consola):
        self.carpeta, self.args, self.consola = carpeta, args, consola
        datos, self.config = cargar_plan(carpeta)
        self.pedido = pedido_desde_plan(datos, self.config)
        self.cortar = threading.Event()
        self.error: BaseException | None = None
        self.unidades: list[Unidad] = []

    # --- control ---

    def pedir_corte(self) -> None:
        self.cortar.set()
        for u in self.unidades:
            u.renderizador.cancelar()

    def fallo(self, unidad: Unidad, que: str, error: BaseException) -> None:
        if not isinstance(error, RenderCancelado):
            self.consola.linea(f"ERROR en {unidad.nombre} ({que}): {error}")
            self.consola.texto("".join(traceback.format_exception(type(error), error, error.__traceback__)))
            if self.error is None:
                self.error = error
        self.pedir_corte()

    def _en_paralelo(self, funcion, unidades=None) -> None:
        """Corre funcion(unidad) en un hilo por unidad; el hilo principal queda libre para Ctrl+C."""
        hilos = [threading.Thread(target=u.ejecutar, args=(funcion,), daemon=True) for u in (unidades or self.unidades)]
        for h in hilos:
            h.start()
        for h in hilos:
            while h.is_alive():
                h.join(0.25)

    # --- todo ---

    def correr(self) -> int:
        c, p = self.config, self.pedido
        modo = "rápido (mapa exponencial)" if c.modo == "exponencial" else "normal (cuadro por cuadro)"
        self.consola.texto("ABoundary · render sin interfaz")
        self.consola.texto(f"Video: {os.path.basename(os.path.abspath(self.carpeta))} · {c.ancho}x{c.alto} · "
                           f"{c.fps} fps · {formatear_duracion(c.cuadros / c.fps)} · {c.cuadros} cuadros · modo {modo}")
        self.consola.texto(f"Cálculo: {p.tipo_fractal} con {p.tipo_calculo}, {p.max_iter:,} iteraciones · "
                           f"antialiasing {'sí' if p.opciones.antialiasing else 'no'}")
        self.unidades = crear_unidades(p, c, self.args, self.consola)

        if c.modo == "exponencial":
            if not self._tira():
                return self._salida()
            if self.args.solo_tira or self.args.franjas:
                return self._salida()
        elif self.args.solo_tira or self.args.franjas:
            raise SystemExit("--solo-tira y --franjas son para videos en modo rápido (exponencial).")
        if not self._tramos():
            return self._salida()
        return self._unir()

    def _salida(self) -> int:
        if self.error is not None:
            self.consola.linea("Se detuvo por un error. Revisalo y volvé a correr lo mismo para seguir.")
            return 1
        if self.cortar.is_set():
            self.consola.linea("Cortado. Lo hecho quedó guardado: volvé a correr lo mismo para seguir.")
            return 2
        return 0

    # ========================================================
    #   1) La tira (modo rápido)
    # ========================================================

    def _geometria(self):
        c = self.config
        self.geo = exponencial.geometria_tira(self.pedido, c.ancho, c.alto)
        self.ln_pasos = exponencial.ln_pasos_cuadros(self.geo, c.cuadros_zoom, c.suavizado)
        self.carpeta_tira = os.path.join(self.carpeta, "tira")
        os.makedirs(self.carpeta_tira, exist_ok=True)

    def _tira(self) -> bool:
        """Calcula las franjas que faltan. Devuelve True si se puede seguir con los cuadros."""
        self._geometria()
        necesarias = sorted({f for ln in self.ln_pasos for f in exponencial.franjas_para_cuadro(self.geo, ln)})
        existe = lambda f: os.path.exists(exponencial.ruta_franja(self.carpeta_tira, f))
        elegidas = necesarias
        if self.args.franjas:
            desde, _, hasta = self.args.franjas.partition("-")
            a, b = int(desde), int(hasta or desde)
            elegidas = [f for f in necesarias if a <= f <= b]
        faltan = [f for f in elegidas if not existe(f)]
        hechas = sum(existe(f) for f in necesarias)
        self.consola.texto(f"Tira: {self.geo.filas:,} filas x {self.geo.n_angulos:,} ángulos · "
                           f"{hechas}/{len(necesarias)} franjas ya calculadas"
                           + (f" · este pedido: franjas {self.args.franjas}" if self.args.franjas else "")
                           + f" · faltan {len(faltan)}")
        self.consola.texto()
        if faltan:
            etapa = Etapa(self.consola, "tira", len(faltan), "franjas")
            pendientes = deque(faltan)
            candado = threading.Lock()
            pedido = self.pedido

            with ThreadPoolExecutor(HILOS_GUARDADO) as guardado:
                def guardar(unidad, f, rgb, segundos, espacio):
                    try:
                        exponencial.guardar_franja(exponencial.ruta_franja(self.carpeta_tira, f), rgb)
                        etapa.paso(f"franja {f} ({unidad.nombre}) en {segundos:.1f} s")
                    except Exception as e:
                        self.fallo(unidad, f"guardando la franja {f}", e)
                    finally:
                        espacio.release()

                def calcular(unidad):
                    espacio = threading.Semaphore(FRANJAS_EN_ESPERA)
                    while not self.cortar.is_set():
                        with candado:
                            if self._conviene_esperar(unidad, len(pendientes)):
                                return
                            f = pendientes.popleft() if pendientes else None
                        if f is None:
                            return
                        inicio = time.monotonic()
                        try:
                            rgb = self._franja_con_reintentos(unidad, replace(pedido, tipo_calculo=unidad.metodo), f)
                        except Exception as e:
                            self.fallo(unidad, f"franja {f}", e)
                            return
                        unidad.medir(time.monotonic() - inicio)
                        espacio.acquire()
                        guardado.submit(guardar, unidad, f, rgb, time.monotonic() - inicio, espacio)

                self._en_paralelo(calcular)
            if self.cortar.is_set():
                return False
            self.consola.linea(f"tira: {len(faltan)} franjas calculadas en {etapa.duracion()}")
        incompletas = [f for f in necesarias if not existe(f)]
        if incompletas:
            self.consola.linea(f"Faltan {len(incompletas)} franjas de la tira (por ejemplo la {incompletas[0]}): "
                               f"los cuadros se arman cuando esté completa.")
            return False
        return True

    def _conviene_esperar(self, unidad: Unidad, pendientes: int) -> bool:
        """La CPU es mucho más lenta que una GPU: no toma otra franja si las GPUs van a terminar
        todas las que quedan antes de que ella termine esa (si no, el final la espera)."""
        gpus = [u for u in self.unidades if u.placa is not None]
        if unidad.placa is not None or not gpus or unidad.segundos_por_paso is None:
            return False
        tiempos = [u.segundos_por_paso for u in gpus if u.segundos_por_paso is not None]
        if not tiempos:
            return False
        ritmo_gpus = sum(1 / t for t in tiempos)                  # franjas por segundo entre todas
        return unidad.segundos_por_paso > pendientes / ritmo_gpus

    def _franja_con_reintentos(self, unidad: Unidad, pedido, f: int, intentos: int = 3):
        """Como en la app: si falta memoria por un momento, libera lo que puede y reintenta."""
        for intento in range(intentos):
            try:
                return exponencial.calcular_franja(unidad.renderizador.motor, pedido, self.geo, f)
            except Exception as e:
                if not es_falta_de_memoria(e) or intento == intentos - 1:
                    raise
                self.consola.linea(f"AVISO {unidad.nombre}: falta memoria en la franja {f}, se reintenta ({e})")
                unidad.renderizador.motor._cache_orbita = None
                gc.collect()
                liberar_memoria_gpu()
                time.sleep(2 * (intento + 1))

    # ========================================================
    #   2) Cuadros, codificados en tramos de video
    # ========================================================

    def _config_tramos(self) -> dict:
        """Cómo se corta el video en tramos. Se guarda al empezar y se respeta al retomar, así
        los tramos ya hechos siguen sirviendo y todos se pueden unir sin recodificar."""
        carpeta = os.path.join(self.carpeta, CARPETA_TRAMOS)
        os.makedirs(carpeta, exist_ok=True)
        ruta = os.path.join(carpeta, ARCHIVO_TRAMOS)
        if os.path.exists(ruta):
            with open(ruta, encoding="utf-8") as f:
                datos = json.load(f)
            pedido = (self.args.codificador, self.args.calidad, self.args.tramo)
            if any(v is not None and v != datos[k] for v, k in zip(pedido, ("codificador", "calidad", "cuadros_por_tramo"))):
                self.consola.texto(f"Aviso: el video ya tiene tramos hechos con {datos['codificador']}, calidad "
                                   f"{datos['calidad']}, {datos['cuadros_por_tramo']} cuadros por tramo: se sigue igual "
                                   f"(para cambiarlo, borrá la carpeta {CARPETA_TRAMOS}).")
            return datos
        codificador = self.args.codificador or "x264"
        unidades = max(1, len(self.unidades))
        por_tramo = self.args.tramo or max(30, min(1000, math.ceil(self.config.cuadros / (4 * unidades))))
        datos = {"codificador": codificador, "calidad": self.args.calidad or CODIFICADORES[codificador][1],
                 "cuadros_por_tramo": por_tramo}
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2)
        return datos

    def _ruta_tramo(self, k: int) -> str:
        return os.path.join(self.carpeta, CARPETA_TRAMOS, f"tramo_{k + 1:05d}.mp4")

    def _tramos(self) -> bool:
        c = self.config
        ajustes = self._config_tramos()
        self.ajustes_tramos = ajustes
        por_tramo = ajustes["cuadros_por_tramo"]
        self.n_tramos = math.ceil(c.cuadros / por_tramo)
        faltan = [k for k in range(self.n_tramos) if not os.path.exists(self._ruta_tramo(k))]
        self.consola.texto(f"Video: {self.n_tramos} tramos de {por_tramo} cuadros · {ajustes['codificador']} calidad "
                           f"{ajustes['calidad']} · {self.n_tramos - len(faltan)} ya hechos · faltan {len(faltan)}")
        self.consola.texto()
        if not faltan:
            return True
        if c.modo != "exponencial":
            self.plan = list(planificar_zoom(self.pedido.xmin, self.pedido.xmax, self.pedido.ymin, self.pedido.ymax,
                                             c.ancho, c.alto, c.cuadros_zoom, self.pedido.max_iter, c.suavizado))
        cuadros = sum(min(c.cuadros, (k + 1) * por_tramo) - k * por_tramo for k in faltan)
        etapa = Etapa(self.consola, "video", cuadros, "cuadros")
        pendientes = deque(faltan)
        candado = threading.Lock()

        def trabajar(unidad):
            estado = {"franjas": {}, "leyendo": {}, "banda": {}, "ultimo": None,
                      "lector": ThreadPoolExecutor(2), "guardado": ThreadPoolExecutor(2)}
            try:
                trabajar_tramos(unidad, estado)
            finally:
                estado["lector"].shutdown(wait=False, cancel_futures=True)
                estado["guardado"].shutdown(wait=True)

        def trabajar_tramos(unidad, estado):
            while not self.cortar.is_set():
                with candado:
                    if not pendientes:
                        return
                    k = pendientes.popleft()
                inicio = time.monotonic()
                try:
                    hechos = self._tramo(unidad, k, estado)
                except Exception as e:
                    self.fallo(unidad, f"tramo {k + 1}", e)
                    return
                if hechos:
                    desde = k * por_tramo
                    etapa.paso(f"tramo {k + 1}/{self.n_tramos} (cuadros {desde + 1}-{desde + hechos}, {unidad.nombre}) "
                               f"en {formatear_duracion(time.monotonic() - inicio)}", hechos)

        # Los tramos son largos: si hay GPUs, solo ellas (la CPU tomaría uno entero y el final la esperaría)
        unidades = [u for u in self.unidades if u.placa is not None] or self.unidades
        self._en_paralelo(trabajar, unidades)
        if self.cortar.is_set():
            return False
        self.consola.linea(f"video: {cuadros} cuadros en {etapa.duracion()}")
        return True

    def _tramo(self, unidad: Unidad, k: int, estado: dict) -> int:
        """Genera los cuadros del tramo k y los codifica con ffmpeg. Devuelve cuántos cuadros
        tiene, o 0 si se cortó (el tramo a medias se descarta y se rehace al retomar)."""
        c = self.config
        por_tramo = self.ajustes_tramos["cuadros_por_tramo"]
        indices = range(k * por_tramo, min(c.cuadros, (k + 1) * por_tramo))
        destino = self._ruta_tramo(k)
        temporal = destino + ".parcial"
        argumentos, _ = CODIFICADORES[self.ajustes_tramos["codificador"]]
        argumentos = [a.format(calidad=self.ajustes_tramos["calidad"]) for a in argumentos]
        comando = [ffmpeg(), "-y", "-loglevel", "error",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{c.ancho}x{c.alto}", "-r", str(c.fps), "-i", "-",
                   *argumentos, "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                   "-f", "mp4", temporal]
        proceso = subprocess.Popen(comando, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        # Un hilo le pasa los cuadros a ffmpeg mientras la GPU arma los siguientes
        cola: queue.Queue = queue.Queue(CUADROS_EN_COLA)
        fallo_escritura = []

        def escribir():
            try:
                while (rgb := cola.get()) is not None:
                    proceso.stdin.write(np.ascontiguousarray(rgb[::-1]).tobytes())   # fila 0 = arriba
            except Exception as e:
                fallo_escritura.append(e)
                while cola.get() is not None:      # que el productor no se quede trabado
                    pass

        escritor = threading.Thread(target=escribir, daemon=True)
        escritor.start()
        try:
            try:
                for i in indices:
                    if self.cortar.is_set() or fallo_escritura:
                        break
                    rgb = self._cuadro(unidad, i, estado)
                    png = os.path.join(self.carpeta, nombre_cuadro(i))
                    if self.args.guardar_cuadros and not os.path.exists(png):
                        estado["guardado"].submit(guardar_imagen, png, rgb)
                    cola.put(rgb)
            finally:
                cola.put(None)
                escritor.join()
            proceso.stdin.close()
            errores = proceso.stderr.read().decode(errors="replace").strip()
            codigo = proceso.wait()
        except BaseException:
            proceso.kill()
            proceso.wait()
            raise
        finally:
            proceso.stderr.close()
        if fallo_escritura and not self.cortar.is_set():
            _borrar(temporal)
            raise RuntimeError(f"ffmpeg dejó de recibir cuadros en el tramo {k + 1}: {fallo_escritura[0]} {errores}")
        if self.cortar.is_set():
            _borrar(temporal)
            return 0
        if codigo != 0:
            _borrar(temporal)
            raise RuntimeError(f"ffmpeg falló en el tramo {k + 1}: {errores}")
        os.replace(temporal, destino)
        return len(indices)

    def _cuadro(self, unidad: Unidad, i: int, estado: dict) -> np.ndarray:
        """El cuadro i como (alto, ancho, 3) uint8, fila 0 abajo (como guarda la app)."""
        c = self.config
        png = os.path.join(self.carpeta, nombre_cuadro(i))
        if os.path.exists(png):                      # hecho antes (por la app o con --guardar-cuadros)
            return np.asarray(Image.open(png).convert("RGB"))[::-1]
        i = min(i, c.cuadros_zoom - 1)               # la pausa final repite el último cuadro del zoom
        if estado["ultimo"] is not None and estado["ultimo"][0] == i:
            return estado["ultimo"][1]
        if c.modo == "exponencial":
            xp = unidad.xp()
            franjas, leyendo = estado["franjas"], estado["leyendo"]
            leer = lambda f: np.asarray(Image.open(exponencial.ruta_franja(self.carpeta_tira, f)).convert("RGB"))

            # Leer un PNG de franja tarda ~0.1 s: se piden de antemano las de los próximos cuadros
            adelante = min(i + CUADROS_ADELANTE, c.cuadros_zoom - 1)
            for f in exponencial.franjas_para_cuadro(self.geo, self.ln_pasos[adelante]):
                if f not in franjas and f not in leyendo:
                    leyendo[f] = estado["lector"].submit(leer, f)

            def cargar(f):
                if f not in franjas:
                    arr = leyendo.pop(f).result() if f in leyendo else leer(f)
                    if len(franjas) >= FRANJAS_EN_MEMORIA:
                        franjas.pop(next(iter(franjas)))
                    franjas[f] = xp.asarray(arr)
                return franjas[f]

            t = i / max(1, c.cuadros_zoom - 1)
            rgb = exponencial.armar_cuadro(self.geo, self.ln_pasos[i], 2 * math.pi * c.vueltas * t, cargar,
                                           self.pedido.opciones.antialiasing, xp, estado["banda"])
        else:
            cuadro = self.plan[i]
            pedido = replace(self.pedido, xmin=cuadro.xmin, xmax=cuadro.xmax, ymin=cuadro.ymin, ymax=cuadro.ymax,
                             max_iter=cuadro.max_iter, tipo_calculo=unidad.metodo)
            rgb = unidad.renderizador.renderizar(pedido).rgb
        estado["ultimo"] = (i, rgb)
        return rgb

    # ========================================================
    #   3) Unir los tramos
    # ========================================================

    def _unir(self) -> int:
        lista = os.path.join(self.carpeta, CARPETA_TRAMOS, "lista.txt")
        with open(lista, "w", encoding="utf-8") as f:
            for k in range(self.n_tramos):
                f.write(f"file '{os.path.basename(self._ruta_tramo(k))}'\n")
        destino = os.path.join(self.carpeta, NOMBRE_VIDEO)
        temporal = destino + ".parcial"
        resultado = subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lista,
                                    "-c", "copy", "-movflags", "+faststart", "-f", "mp4", temporal],
                                   capture_output=True, text=True)
        if resultado.returncode != 0:
            _borrar(temporal)
            self.consola.linea(f"ERROR uniendo los tramos: {resultado.stderr.strip()}")
            return 1
        os.replace(temporal, destino)
        tamano = os.path.getsize(destino) / 1e9
        self.consola.linea(f"Video listo: {destino} ({tamano:.2f} GB) · total "
                           f"{formatear_duracion(time.monotonic() - self.consola.inicio)}")
        return 0


def _borrar(ruta: str) -> None:
    try:
        os.remove(ruta)
    except OSError:
        pass


def ffmpeg() -> str:
    ruta = shutil.which("ffmpeg")
    if ruta is None:
        raise SystemExit("Falta ffmpeg (en Linux: sudo apt install ffmpeg).")
    return ruta


# ============================================================
#                       LÍNEA DE COMANDOS
# ============================================================

def preparar_carpeta(carpeta: str, plan: str | None) -> None:
    """Con --plan, crea la carpeta (si hace falta) y le copia el video.json."""
    destino = os.path.join(carpeta, ARCHIVO_PLAN)
    if plan:
        with open(plan, encoding="utf-8") as f:
            nuevo = json.load(f)
        if os.path.exists(destino):
            with open(destino, encoding="utf-8") as f:
                if json.load(f) != nuevo:
                    raise SystemExit(f"La carpeta ya tiene otro {ARCHIVO_PLAN}: usá otra carpeta o no pases --plan.")
        else:
            os.makedirs(carpeta, exist_ok=True)
            shutil.copyfile(plan, destino)
    if cargar_plan(carpeta) is None:
        raise SystemExit(f"{carpeta} no tiene un {ARCHIVO_PLAN} válido (pasalo con --plan).")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Renderiza un video de ABoundary sin interfaz: lee el video.json de la carpeta, calcula lo que "
                    "falta y arma el mp4. Se puede cortar (Ctrl+C) y volver a correr para seguir.")
    parser.add_argument("carpeta", help="carpeta del video (la misma que usa la app)")
    parser.add_argument("--plan", help="video.json a usar: se copia a la carpeta (que se crea si no existe)")
    parser.add_argument("--gpus", default="todas", help="GPUs a usar: 'todas' (por defecto) o una lista, como 0,1,3")
    parser.add_argument("--sin-cpu", action="store_true", help="no calcular también con la CPU")
    parser.add_argument("--solo-tira", action="store_true", help="modo rápido: calcular solo la tira, sin armar el video")
    parser.add_argument("--franjas", metavar="A-B",
                        help="modo rápido: calcular solo esas franjas de la tira (para repartirla entre máquinas)")
    parser.add_argument("--codificador", choices=list(CODIFICADORES),
                        help="x264 (por defecto, el más compatible), x265 (la mitad de tamaño), nvenc-h264 / "
                             "nvenc-h265 (por hardware en la GPU)")
    parser.add_argument("--calidad", type=int, help="CRF/CQ: más bajo = mejor y más pesado (por defecto 18 en x264)")
    parser.add_argument("--tramo", type=int, help="cuadros por tramo de video (por defecto, según el largo del video)")
    parser.add_argument("--guardar-cuadros", action="store_true",
                        help="guardar también cada cuadro como PNG (ocupa mucho: ~3 MB por cuadro en Full HD)")
    parser.add_argument("--detalle", action="store_true", help="mostrar también los mensajes internos")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.detalle else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if not args.detalle:
        logging.getLogger("core").setLevel(logging.ERROR)

    preparar_carpeta(args.carpeta, args.plan)
    consola = Consola()
    render = Render(args.carpeta, args, consola)

    def al_cortar(signum, frame):
        if render.cortar.is_set():
            consola.linea("Corte forzado.")
            os._exit(2)
        consola.linea("Cortando… (los cálculos en curso se abandonan; Ctrl+C otra vez para salir ya)")
        render.pedir_corte()

    signal.signal(signal.SIGINT, al_cortar)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, al_cortar)        # la máquina alquilada se apaga
    return render.correr()


if __name__ == "__main__":
    sys.exit(main())
