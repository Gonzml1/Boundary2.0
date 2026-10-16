"""Diálogos de la ventana principal: configuración, vista previa y progreso del video, y atajos."""
import os
import time
import shutil
import logging
import gc
import math
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace, asdict
from decimal import Decimal

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
from PIL import Image

from core.modulo_de_calculo_fractales import (RenderCancelado, FRACTAL_REGISTRY, variante_rapida,
                                             es_falta_de_memoria, liberar_memoria_gpu)
from core.paletas import PALETTE_REGISTRY
from core.render import Renderizador, PedidoRender, OpcionesColor, guardar_imagen
from core import exponencial
from core.video import (ConfigVideo, planificar_zoom, armar_mp4, nombre_cuadro, guardar_plan,
                        cargar_plan, cuadros_hechos, cuadros_faltantes, describir_cuadros, leer_cuadros)

log = logging.getLogger(__name__)

MODOS_VIDEO = {"Normal (cuadro por cuadro)": "normal", "Rápido (mapa exponencial)": "exponencial"}
FRANJAS_EN_MEMORIA = 64      # franjas de la tira que se guardan cargadas al armar los cuadros


def modulo_arreglos():
    """CuPy si hay GPU (armar los cuadros es mucho más rápido), si no NumPy."""
    try:
        from core.modulo_de_calculo_fractales import BACKENDS_DISPONIBLES
        if BACKENDS_DISPONIBLES.get("gpu"):
            import cupy
            return cupy
    except Exception:
        pass
    return np


RESOLUCIONES = {"Full HD (1920x1080)": (1920, 1080), "4K (3840x2160)": (3840, 2160), "HD (1280x720)": (1280, 720)}
ANCHO_VISTA_PREVIA = 320
FPS_MAX_VISTA_PREVIA = 15
# Hilos que comprimen y guardan los PNG mientras se calculan los cuadros siguientes, y cuántos
# cuadros calculados pueden esperar a ser guardados (para no llenar la memoria)
HILOS_GUARDADO = 2
MAX_CUADROS_EN_ESPERA = 6


def metodos_para_video(pedido: PedidoRender, usar_cpu: bool) -> list[str]:
    """Métodos que calculan los cuadros en paralelo: el elegido y, si es de GPU y existe, su
    versión de CPU (dan la misma imagen, así que los cuadros se pueden repartir)."""
    metodos = [pedido.tipo_calculo]
    if usar_cpu and "_gpu" in pedido.tipo_calculo:
        cpu = pedido.tipo_calculo.replace("_gpu", "_cpu")      # perturbacion_gpu_float -> perturbacion_cpu_float
        if cpu in FRACTAL_REGISTRY.get(pedido.tipo_fractal, {}):
            metodos.append(cpu)
    return metodos


def formatear_duracion(segundos: float) -> str:
    segundos = max(0, round(segundos))
    if segundos < 60:
        return f"{segundos} s"
    if segundos < 3600:
        return f"{segundos // 60} min {segundos % 60:02d} s"
    return f"{segundos // 3600} h {segundos % 3600 // 60:02d} min"


def pedido_desde_plan(datos: dict, config: ConfigVideo) -> PedidoRender:
    """Rearma el pedido guardado en el video.json de un video."""
    nombres = [nombre for nombre, _ in PALETTE_REGISTRY]
    paleta = nombres.index(datos["nombre_paleta"]) if datos.get("nombre_paleta") in nombres else datos["paleta"]
    opciones = {**asdict(OpcionesColor()), **datos.get("opciones", {})}
    if opciones.get("interior") is not None:
        opciones["interior"] = tuple(opciones["interior"])
    return PedidoRender(
        Decimal(datos["xmin"]), Decimal(datos["xmax"]), Decimal(datos["ymin"]), Decimal(datos["ymax"]),
        config.ancho, config.alto, datos["max_iter"], datos["tipo_fractal"], datos["tipo_calculo"],
        datos.get("real", 0.0), datos.get("imag", 0.0), paleta, datos["clase_equiv"],
        OpcionesColor(**opciones),
    )


# ============================================================
#                       HILOS DEL VIDEO
# ============================================================

class WorkerVideo(QThread):
    """Renderiza un zoom desde el fractal entero hasta la vista del pedido (ver core/video.py).

    - Si el método es de GPU y tiene versión de CPU, las dos calculan cuadros a la vez.
    - Los PNG se comprimen y guardan en otros hilos mientras se calculan los cuadros siguientes.
    - Si en la carpeta ya hay cuadros hechos (un video cancelado), los saltea.
    - Con `rehacer` (índices desde 0) calcula solo esos cuadros, aunque ya existan."""
    empezo = pyqtSignal(int, int)    # cuadros ya hechos, total
    progreso = pyqtSignal(int, int)  # cuadros hechos, total
    terminado = pyqtSignal(str)      # mensaje final

    def __init__(self, pedido: PedidoRender, carpeta: str, config: ConfigVideo,
                 rehacer: list[int] | None = None):
        super().__init__()
        self.pedido = pedido
        self.carpeta = carpeta
        self.config = config
        self.rehacer = rehacer
        self.metodos = metodos_para_video(pedido, config.usar_cpu)
        self._renderizadores = [Renderizador(tam_cache=0) for _ in self.metodos]
        self._candado = threading.Lock()
        self._hechos = 0
        self._error = None

    def cancelar(self) -> None:
        """Corta en los cuadros en curso (no espera a que terminen)."""
        self.requestInterruption()
        for r in self._renderizadores:
            r.cancelar()

    def _contar_hecho(self, total: int) -> None:
        with self._candado:
            self._hechos += 1
            hechos = self._hechos
        self.progreso.emit(hechos, total)

    def run(self):
        if self.config.modo == "exponencial":
            self._run_exponencial()
            return
        p, cfg = self.pedido, self.config
        guardar_plan(self.carpeta, p, cfg, PALETTE_REGISTRY[p.paleta][0])
        plan = list(planificar_zoom(p.xmin, p.xmax, p.ymin, p.ymax, p.ancho, p.alto,
                                    cfg.cuadros_zoom, p.max_iter, cfg.suavizado))
        if self.rehacer is None:
            # Todo el video, salteando los cuadros que ya están (y rellenando los huecos)
            total = cfg.cuadros
            self._hechos = cuadros_hechos(self.carpeta, cfg)
            a_calcular = [i for i in range(cfg.cuadros_zoom)
                          if not os.path.exists(os.path.join(self.carpeta, nombre_cuadro(i)))]
            pausa = [i for i in range(cfg.cuadros_zoom, cfg.cuadros)
                     if not os.path.exists(os.path.join(self.carpeta, nombre_cuadro(i)))]
        else:
            # Solo los cuadros pedidos, aunque ya existan
            total = len(self.rehacer)
            self._hechos = 0
            a_calcular = [i for i in self.rehacer if i < cfg.cuadros_zoom]
            pausa = [i for i in self.rehacer if i >= cfg.cuadros_zoom]
        self.empezo.emit(self._hechos, total)
        log.info("Video con %s", " + ".join(self.metodos))
        pendientes = iter([(i, plan[i]) for i in a_calcular])
        espacio = threading.Semaphore(MAX_CUADROS_EN_ESPERA)

        def guardar(ruta, rgb):
            try:
                guardar_imagen(ruta, rgb)
                self._contar_hecho(total)
            except Exception as e:
                self._error = e
                self.cancelar()
            finally:
                espacio.release()

        def calcular(metodo, renderizador, guardado):
            while not self.isInterruptionRequested():
                with self._candado:
                    siguiente = next(pendientes, None)
                if siguiente is None:
                    return
                i, cuadro = siguiente
                # Mismas iteraciones (y colores) que la vista actual; solo cambia la región
                pedido = replace(p, xmin=cuadro.xmin, xmax=cuadro.xmax, ymin=cuadro.ymin, ymax=cuadro.ymax,
                                 max_iter=cuadro.max_iter, tipo_calculo=metodo)
                try:
                    rgb = renderizador.renderizar(pedido).rgb
                except RenderCancelado:
                    return
                except Exception as e:
                    log.exception("Falló el cuadro %d con %s", i + 1, metodo)
                    self._error = e
                    self.cancelar()
                    return
                espacio.acquire()
                guardado.submit(guardar, os.path.join(self.carpeta, nombre_cuadro(i)), rgb)

        with ThreadPoolExecutor(HILOS_GUARDADO) as guardado:
            hilos = [threading.Thread(target=calcular, args=(m, r, guardado), daemon=True)
                     for m, r in zip(self.metodos, self._renderizadores)]
            for hilo in hilos:
                hilo.start()
            for hilo in hilos:
                hilo.join()

        if self._error is not None:
            self.terminado.emit(f"El video se detuvo por un error: {self._error}")
            return
        if self.isInterruptionRequested():
            self.terminado.emit(
                f"Video cancelado en {self._hechos}/{total} cuadros. Para seguirlo: Video de zoom → "
                f"Continuar video… y elegí la carpeta {self.carpeta}")
            return

        # Pausa final: copias del último cuadro del zoom (no hace falta volver a calcularlo)
        ultimo = os.path.join(self.carpeta, nombre_cuadro(cfg.cuadros_zoom - 1))
        if pausa and os.path.exists(ultimo):
            for i in pausa:
                shutil.copyfile(ultimo, os.path.join(self.carpeta, nombre_cuadro(i)))
                self._contar_hecho(total)

        # El mp4 se arma solo si están todos los cuadros (ffmpeg corta en el primer hueco)
        faltan = cuadros_faltantes(self.carpeta, cfg)
        if faltan:
            cuantos = "falta 1 cuadro" if len(faltan) == 1 else f"faltan {len(faltan)} cuadros"
            self.terminado.emit(f"Listo, pero {cuantos} ({describir_cuadros(faltan)}): "
                                f"completalos con Continuar video… o Rehacer cuadros…")
            return
        mp4 = armar_mp4(self.carpeta, cfg.fps)
        self.terminado.emit(f"Video listo: {mp4}" if mp4 else f"Cuadros listos en {self.carpeta}")


    def _run_exponencial(self):
        """Modo rápido: calcula la tira (por franjas, con CPU y GPU a la vez) y arma cada cuadro
        leyéndola. Las franjas y los cuadros ya hechos se saltean (se puede retomar)."""
        cfg = self.config
        guardar_plan(self.carpeta, self.pedido, cfg, PALETTE_REGISTRY[self.pedido.paleta][0])
        pedido = replace(self.pedido, tipo_calculo=exponencial.metodo_para_tira(self.pedido))
        geo = exponencial.geometria_tira(pedido, cfg.ancho, cfg.alto)
        carpeta_tira = os.path.join(self.carpeta, "tira")
        os.makedirs(carpeta_tira, exist_ok=True)
        ln_pasos = exponencial.ln_pasos_cuadros(geo, cfg.cuadros_zoom, cfg.suavizado)
        existe = lambda i: os.path.exists(os.path.join(self.carpeta, nombre_cuadro(i)))
        if self.rehacer is None:
            cuadros = [i for i in range(cfg.cuadros_zoom) if not existe(i)]
            pausa = [i for i in range(cfg.cuadros_zoom, cfg.cuadros) if not existe(i)]
        else:
            cuadros = [i for i in self.rehacer if i < cfg.cuadros_zoom]
            pausa = [i for i in self.rehacer if i >= cfg.cuadros_zoom]
        necesarias = sorted({f for i in cuadros for f in exponencial.franjas_para_cuadro(geo, ln_pasos[i])})
        franjas = [f for f in necesarias if not os.path.exists(exponencial.ruta_franja(carpeta_tira, f))]
        total = len(franjas) + len(cuadros) + len(pausa)
        self._hechos = 0
        self.empezo.emit(0, total)
        log.info("Video rápido: tira de %d x %d (%d franjas por calcular), %d cuadros por armar",
                 geo.filas, geo.n_angulos, len(franjas), len(cuadros))

        # 1) La tira, por franjas: los métodos (GPU y su versión de CPU) se reparten las franjas
        metodos = metodos_para_video(pedido, cfg.usar_cpu)
        self._renderizadores = [Renderizador(tam_cache=0) for _ in metodos]
        pendientes = iter(franjas)
        espacio = threading.Semaphore(MAX_CUADROS_EN_ESPERA)

        def guardar(ruta, rgb, funcion):
            try:
                funcion(ruta, rgb)
                self._contar_hecho(total)
            except Exception as e:
                self._error = e
                self.cancelar()
            finally:
                espacio.release()

        def calcular(metodo, renderizador, guardado):
            while not self.isInterruptionRequested():
                with self._candado:
                    f = next(pendientes, None)
                if f is None:
                    return
                try:
                    rgb = self._franja_con_reintentos(renderizador, replace(pedido, tipo_calculo=metodo), geo, f)
                except RenderCancelado:
                    return
                except Exception as e:
                    log.exception("Falló la franja %d con %s", f, metodo)
                    self._error = e
                    self.cancelar()
                    return
                espacio.acquire()
                guardado.submit(guardar, exponencial.ruta_franja(carpeta_tira, f), rgb, exponencial.guardar_franja)

        with ThreadPoolExecutor(HILOS_GUARDADO) as guardado:
            hilos = [threading.Thread(target=calcular, args=(m, r, guardado), daemon=True)
                     for m, r in zip(metodos, self._renderizadores)]
            for hilo in hilos:
                hilo.start()
            for hilo in hilos:
                hilo.join()

            # 2) Los cuadros: se arman leyendo la tira (sin calcular iteraciones)
            if self._error is None and not self.isInterruptionRequested():
                xp = modulo_arreglos()
                cargadas = {}

                def cargar(f):
                    if f not in cargadas:
                        if len(cargadas) >= FRANJAS_EN_MEMORIA:
                            cargadas.pop(next(iter(cargadas)))
                        arr = np.asarray(Image.open(exponencial.ruta_franja(carpeta_tira, f)).convert("RGB"))
                        cargadas[f] = xp.asarray(arr)
                    return cargadas[f]

                for i in cuadros:
                    if self.isInterruptionRequested() or self._error is not None:
                        break
                    t = i / max(1, cfg.cuadros_zoom - 1)
                    try:
                        rgb = exponencial.armar_cuadro(geo, ln_pasos[i], 2 * math.pi * cfg.vueltas * t, cargar,
                                                       pedido.opciones.antialiasing, xp)
                    except Exception as e:
                        log.exception("Falló el armado del cuadro %d", i + 1)
                        self._error = e
                        break
                    espacio.acquire()
                    guardado.submit(guardar, os.path.join(self.carpeta, nombre_cuadro(i)), rgb, guardar_imagen)

        if self._error is not None:
            self.terminado.emit(f"El video se detuvo por un error: {self._error}")
            return
        if self.isInterruptionRequested():
            self.terminado.emit(
                f"Video cancelado ({self._hechos}/{total} pasos). Para seguirlo: Video de zoom → "
                f"Continuar video… y elegí la carpeta {self.carpeta}")
            return
        self._terminar(pausa, total)

    def _franja_con_reintentos(self, renderizador, pedido, geo, f, intentos: int = 3):
        """Si falta memoria (la RAM o la memoria comprometida de Windows se llenó por un momento),
        libera lo que puede, espera un poco y reintenta en vez de cortar el video."""
        for intento in range(intentos):
            try:
                return exponencial.calcular_franja(renderizador.motor, pedido, geo, f)
            except Exception as e:
                if not es_falta_de_memoria(e) or intento == intentos - 1:
                    raise
                log.warning("Falta de memoria en la franja %d (%s): se libera memoria y se reintenta", f, e)
                renderizador.motor._cache_orbita = None
                gc.collect()
                liberar_memoria_gpu()
                time.sleep(2 * (intento + 1))

    def _terminar(self, pausa, total):
        """Pausa final (copias del último cuadro del zoom) y mp4 si están todos los cuadros."""
        cfg = self.config
        ultimo = os.path.join(self.carpeta, nombre_cuadro(cfg.cuadros_zoom - 1))
        if pausa and os.path.exists(ultimo):
            for i in pausa:
                shutil.copyfile(ultimo, os.path.join(self.carpeta, nombre_cuadro(i)))
                self._contar_hecho(total)
        faltan = cuadros_faltantes(self.carpeta, cfg)
        if faltan:
            cuantos = "falta 1 cuadro" if len(faltan) == 1 else f"faltan {len(faltan)} cuadros"
            self.terminado.emit(f"Listo, pero {cuantos} ({describir_cuadros(faltan)}): "
                                f"completalos con Continuar video… o Rehacer cuadros…")
            return
        mp4 = armar_mp4(self.carpeta, cfg.fps)
        self.terminado.emit(f"Video listo: {mp4}" if mp4 else f"Cuadros listos en {self.carpeta}")


class WorkerVistaPrevia(QThread):
    """Renderiza el recorrido del video a baja resolución, en memoria."""
    cuadro_listo = pyqtSignal(int, int, object)   # índice, total, QImage

    def __init__(self, pedido: PedidoRender, config: ConfigVideo):
        super().__init__()
        self.pedido = pedido
        self.config = config
        self._renderizador = Renderizador(tam_cache=0)

    def cancelar(self) -> None:
        self.requestInterruption()
        self._renderizador.cancelar()

    def _run_exponencial(self):
        p, cfg = self.pedido, self.config
        pedido = replace(p, tipo_calculo=exponencial.metodo_para_tira(p))
        geo = exponencial.geometria_tira(pedido, cfg.ancho, cfg.alto)
        franjas = []
        for f in range(geo.franjas):
            if self.isInterruptionRequested():
                return
            franjas.append(exponencial.calcular_franja(self._renderizador.motor, pedido, geo, f))
        ln_pasos = exponencial.ln_pasos_cuadros(geo, cfg.cuadros_zoom, cfg.suavizado)
        imagen = None
        for i in range(cfg.cuadros):
            if self.isInterruptionRequested():
                return
            if i < cfg.cuadros_zoom:
                t = i / max(1, cfg.cuadros_zoom - 1)
                rgb = exponencial.armar_cuadro(geo, ln_pasos[i], 2 * math.pi * cfg.vueltas * t,
                                               lambda f: franjas[f], False)[::-1].copy()
                alto, ancho, _ = rgb.shape
                imagen = QImage(rgb.data, ancho, alto, 3 * ancho, QImage.Format_RGB888).copy()
            self.cuadro_listo.emit(i, cfg.cuadros, imagen)

    def run(self):
        if self.config.modo == "exponencial":
            try:
                self._run_exponencial()
            except RenderCancelado:
                pass
            except Exception:
                log.exception("Falló la vista previa del video")
            return
        p, cfg = self.pedido, self.config
        plan = list(planificar_zoom(p.xmin, p.xmax, p.ymin, p.ymax, p.ancho, p.alto,
                                    cfg.cuadros_zoom, p.max_iter, cfg.suavizado))
        total = cfg.cuadros
        imagen = None
        try:
            for i in range(total):
                if self.isInterruptionRequested():
                    return
                if i < len(plan):
                    c = plan[i]
                    rgb = self._renderizador.renderizar(
                        replace(p, xmin=c.xmin, xmax=c.xmax, ymin=c.ymin, ymax=c.ymax)).rgb[::-1].copy()
                    alto, ancho, _ = rgb.shape
                    imagen = QImage(rgb.data, ancho, alto, 3 * ancho, QImage.Format_RGB888).copy()
                self.cuadro_listo.emit(i, total, imagen)   # en la pausa se repite el último
        except RenderCancelado:
            return
        except Exception:
            log.exception("Falló la vista previa del video")


# ============================================================
#                       DIÁLOGOS DEL VIDEO
# ============================================================

class VentanaVistaPrevia(QtWidgets.QDialog):
    """Reproduce el recorrido del video a baja resolución mientras se va calculando."""

    def __init__(self, pedido: PedidoRender, config: ConfigVideo, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Vista previa del video")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self._cuadros = []
        self._total = config.cuadros
        self._actual = 0

        self.imagen = QtWidgets.QLabel()
        self.imagen.setAlignment(Qt.AlignCenter)
        self.imagen.setFixedSize(pedido.ancho, pedido.alto)
        self.estado = QtWidgets.QLabel()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.imagen)
        layout.addWidget(self.estado)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._avanzar)
        self._timer.start(round(1000 / config.fps))

        self._worker = WorkerVistaPrevia(pedido, config)
        self._worker.cuadro_listo.connect(self._al_llegar_cuadro)
        self._worker.start()

    def _al_llegar_cuadro(self, indice, total, imagen):
        if imagen is not None:
            self._cuadros.append(QPixmap.fromImage(imagen))

    def _avanzar(self):
        if not self._cuadros:
            self.estado.setText("Calculando…")
            return
        self._actual = (self._actual + 1) % len(self._cuadros)
        self.imagen.setPixmap(self._cuadros[self._actual])
        if len(self._cuadros) < self._total:
            self.estado.setText(f"Calculando vista previa {len(self._cuadros)}/{self._total}")
        else:
            self.estado.setText(f"Cuadro {self._actual + 1}/{self._total} (se repite)")

    def done(self, resultado):
        self._timer.stop()
        self._worker.cancelar()
        self._worker.wait()
        super().done(resultado)


class DialogoVideo(QtWidgets.QDialog):
    """Configuración del video de zoom. Al aceptar, `accion` es "renderizar" o "continuar"
    (en ese caso `carpeta_continuar` tiene la carpeta del video a seguir)."""

    def __init__(self, ajustes, tamano_actual: tuple[int, int], crear_pedido, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video de zoom")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.ajustes = ajustes
        self.crear_pedido = crear_pedido      # (ancho, alto) -> PedidoRender de la vista actual
        self.accion = None
        self.carpeta_continuar = None
        self.cuadros_rehacer = None
        self.resoluciones = {f"Actual ({tamano_actual[0]}x{tamano_actual[1]})": tamano_actual, **RESOLUCIONES}

        self.resolucion = QtWidgets.QComboBox()
        self.resolucion.addItems(list(self.resoluciones))
        self.resolucion.setCurrentText(ajustes.value("video/resolucion", "Full HD (1920x1080)"))
        self.duracion = QtWidgets.QDoubleSpinBox()
        self.duracion.setRange(0.1, 3600.0)
        self.duracion.setDecimals(1)
        self.duracion.setSuffix(" s")
        self.duracion.setValue(float(ajustes.value("video/duracion", 10.0)))
        self.fps = QtWidgets.QSpinBox()
        self.fps.setRange(1, 240)
        self.fps.setValue(int(ajustes.value("video/fps", 30)))
        self.suavizado = QtWidgets.QCheckBox("Arranque y frenado suaves")
        self.suavizado.setChecked(ajustes.value("video/suavizado", False, type=bool))
        self.pausa = QtWidgets.QDoubleSpinBox()
        self.pausa.setRange(0.0, 60.0)
        self.pausa.setDecimals(1)
        self.pausa.setSuffix(" s")
        self.pausa.setValue(float(ajustes.value("video/pausa", 0.0)))
        self.modo = QtWidgets.QComboBox()
        self.modo.addItems(list(MODOS_VIDEO))
        self.modo.setCurrentText(ajustes.value("video/modo", "Normal (cuadro por cuadro)"))
        self.modo.setToolTip("Rápido: calcula cada zona del fractal una sola vez y arma los cuadros "
                             "deformando esa 'tira'. El zoom queda centrado en el destino desde el principio.")
        self.vueltas = QtWidgets.QDoubleSpinBox()
        self.vueltas.setRange(-50.0, 50.0)
        self.vueltas.setDecimals(2)
        self.vueltas.setSingleStep(0.25)
        self.vueltas.setValue(float(ajustes.value("video/vueltas", 0.0)))
        self.vueltas.setToolTip("Vueltas que gira la imagen en todo el zoom (solo en el modo rápido)")
        self.usar_cpu = QtWidgets.QCheckBox("Calcular también con la CPU (más rápido)")
        self.usar_cpu.setChecked(ajustes.value("video/usar_cpu", True, type=bool))
        self.resumen = QtWidgets.QLabel()

        formulario = QtWidgets.QFormLayout()
        formulario.addRow("Resolución", self.resolucion)
        formulario.addRow("Duración del zoom", self.duracion)
        formulario.addRow("Fotogramas por segundo", self.fps)
        formulario.addRow("", self.suavizado)
        formulario.addRow("Pausa en la última imagen", self.pausa)
        formulario.addRow("", self.usar_cpu)
        formulario.addRow("Tipo de video", self.modo)
        formulario.addRow("Rotación (vueltas)", self.vueltas)

        boton_previa = QtWidgets.QPushButton("Vista previa")
        boton_continuar = QtWidgets.QPushButton("Continuar video…")
        boton_rehacer = QtWidgets.QPushButton("Rehacer cuadros…")
        boton_rehacer.setToolTip("Volver a calcular cuadros puntuales de un video ya hecho (por ejemplo, uno que se borró)")
        boton_rehacer.clicked.connect(self._rehacer)
        boton_renderizar = QtWidgets.QPushButton("Renderizar…")
        boton_cancelar = QtWidgets.QPushButton("Cancelar")
        boton_previa.clicked.connect(self._vista_previa)
        boton_continuar.clicked.connect(self._continuar)
        boton_renderizar.clicked.connect(self._renderizar)
        boton_cancelar.clicked.connect(self.reject)
        botones = QtWidgets.QHBoxLayout()
        for b in (boton_previa, boton_continuar, boton_rehacer):
            botones.addWidget(b)
        botones.addStretch()
        for b in (boton_renderizar, boton_cancelar):
            botones.addWidget(b)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(formulario)
        layout.addWidget(self.resumen)
        layout.addLayout(botones)

        for w in (self.duracion, self.fps, self.pausa):
            w.valueChanged.connect(self._actualizar_resumen)
        self.modo.currentIndexChanged.connect(self._actualizar_resumen)
        self.resolucion.currentIndexChanged.connect(self._actualizar_resumen)
        self._actualizar_resumen()

    def config(self) -> ConfigVideo:
        ancho, alto = self.resoluciones[self.resolucion.currentText()]
        return ConfigVideo(ancho, alto, self.duracion.value(), self.fps.value(),
                           self.suavizado.isChecked(), self.pausa.value(), self.usar_cpu.isChecked(),
                           MODOS_VIDEO[self.modo.currentText()], self.vueltas.value())

    def _actualizar_resumen(self):
        c = self.config()
        texto = f"{c.cuadros_zoom} cuadros de zoom"
        if c.cuadros_pausa:
            texto += f" + {c.cuadros_pausa} de pausa"
        texto = f"{texto}  ·  video de {formatear_duracion(c.cuadros / c.fps)}"
        rapido = c.modo == "exponencial"
        self.vueltas.setEnabled(rapido)
        if rapido:
            # Cuánto se calcula con la tira contra cuadro por cuadro
            try:
                pedido = self.crear_pedido(c.ancho, c.alto)
                geo = exponencial.geometria_tira(pedido, c.ancho, c.alto)
                normal = c.cuadros_zoom * c.ancho * c.alto * (4 if pedido.opciones.antialiasing else 1)
                razon = normal / geo.puntos
                comparacion = (f"~{razon:.1f} veces menos cálculo" if razon >= 1.2 else
                               "casi lo mismo: no conviene" if razon >= 0.85 else
                               f"~{1 / razon:.1f} veces MÁS cálculo: para un zoom tan rápido conviene el modo normal")
                texto += (f"\nTira de {geo.puntos / 1e6:,.0f} M puntos contra {normal / 1e6:,.0f} M "
                          f"cuadro por cuadro ({comparacion})")
            except Exception as e:
                texto += f"\n{e}"
        self.resumen.setText(texto)

    def _guardar_ajustes(self):
        self.ajustes.setValue("video/resolucion", self.resolucion.currentText())
        self.ajustes.setValue("video/duracion", self.duracion.value())
        self.ajustes.setValue("video/fps", self.fps.value())
        self.ajustes.setValue("video/suavizado", self.suavizado.isChecked())
        self.ajustes.setValue("video/pausa", self.pausa.value())
        self.ajustes.setValue("video/usar_cpu", self.usar_cpu.isChecked())
        self.ajustes.setValue("video/modo", self.modo.currentText())
        self.ajustes.setValue("video/vueltas", self.vueltas.value())

    def _vista_previa(self):
        c = self.config()
        ancho = ANCHO_VISTA_PREVIA
        alto = max(1, round(ancho * c.alto / c.ancho))
        fps = min(c.fps, FPS_MAX_VISTA_PREVIA)
        previa = replace(c, ancho=ancho, alto=alto, fps=fps)
        pedido = self.crear_pedido(ancho, alto)
        # Vista previa: sin antialiasing y con la variante en float del método (más rápida)
        pedido = replace(pedido, opciones=replace(pedido.opciones, antialiasing=False),
                         tipo_calculo=variante_rapida(pedido.tipo_fractal, pedido.tipo_calculo))
        VentanaVistaPrevia(pedido, previa, self).exec()

    def _renderizar(self):
        self._guardar_ajustes()
        self.accion = "renderizar"
        self.accept()

    def _elegir_video(self, titulo: str):
        """Pide la carpeta de un video y devuelve (carpeta, config), o None."""
        carpeta = QtWidgets.QFileDialog.getExistingDirectory(
            self, f"{titulo}: elegí la carpeta del video (zoom_...)", self.ajustes.value("carpetas/videos", ""))
        if not carpeta:
            return None
        plan = cargar_plan(carpeta)
        if plan is None:
            QtWidgets.QMessageBox.warning(self, titulo, "Esa carpeta no tiene un video (falta video.json).")
            return None
        return carpeta, plan[1]

    def _continuar(self):
        elegido = self._elegir_video("Continuar video")
        if elegido is None:
            return
        carpeta, config = elegido
        faltan = cuadros_faltantes(carpeta, config)
        if not faltan:
            QtWidgets.QMessageBox.information(self, "Continuar video", "Ese video ya tiene todos sus cuadros.")
            return
        if QtWidgets.QMessageBox.question(
                self, "Continuar video",
                f"Video de {config.ancho}x{config.alto}: hay {config.cuadros - len(faltan)}/{config.cuadros} "
                f"cuadros hechos.\nFaltan: {describir_cuadros(faltan)}\n\n¿Calcular los que faltan?"
        ) != QtWidgets.QMessageBox.Yes:
            return
        self.carpeta_continuar = carpeta
        self.accion = "continuar"
        self.accept()

    def _rehacer(self):
        elegido = self._elegir_video("Rehacer cuadros")
        if elegido is None:
            return
        carpeta, config = elegido
        faltan = cuadros_faltantes(carpeta, config)
        texto_faltan = describir_cuadros(faltan) if faltan else "ninguno"
        while True:
            texto, ok = QtWidgets.QInputDialog.getText(
                self, "Rehacer cuadros",
                f"Video de {config.ancho}x{config.alto} con {config.cuadros} cuadros.\n"
                f"Faltan: {texto_faltan}\n\nCuadros a rehacer (por ejemplo: 17, 230-235):",
                text=describir_cuadros(faltan))
            if not ok:
                return
            try:
                cuadros = leer_cuadros(texto, config.cuadros)
            except ValueError as e:
                QtWidgets.QMessageBox.warning(self, "Rehacer cuadros", str(e))
                continue
            if cuadros:
                break
        self.carpeta_continuar = carpeta
        self.cuadros_rehacer = cuadros
        self.accion = "rehacer"
        self.accept()


class DialogoProgresoVideo(QtWidgets.QDialog):
    """Progreso del video: mensaje, barra y, abajo, cuántas imágenes van, cuánto falta y Cancelar."""
    cancelado = pyqtSignal()
    VENTANA_TIEMPOS = 30          # el tiempo restante se estima con los últimos cuadros

    def __init__(self, texto: str, total: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Renderizando video")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self._terminado = False
        self._tiempos = deque(maxlen=self.VENTANA_TIEMPOS)

        mensaje = QtWidgets.QLabel(texto)
        mensaje.setAlignment(Qt.AlignCenter)
        self.barra = QtWidgets.QProgressBar()
        self.barra.setMinimumWidth(360)
        self.contador = QtWidgets.QLabel()
        self.boton_cancelar = QtWidgets.QPushButton("Cancelar")
        self.boton_cancelar.clicked.connect(self._cancelar)

        abajo = QtWidgets.QHBoxLayout()
        abajo.addWidget(self.contador)
        abajo.addStretch()
        abajo.addWidget(self.boton_cancelar)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(mensaje)
        layout.addWidget(self.barra)
        layout.addLayout(abajo)
        self.actualizar(0, total)

    def empezar(self, hechos: int, total: int) -> None:
        """Arranca la medición de tiempos (los cuadros ya hechos de un video retomado no cuentan)."""
        self._tiempos.clear()
        self._tiempos.append((time.monotonic(), hechos))
        self.actualizar(hechos, total)

    def actualizar(self, actual: int, total: int) -> None:
        self.barra.setMaximum(total)
        self.barra.setValue(actual)
        texto = f"{actual}/{total} imágenes"
        if actual and self._tiempos and self._tiempos[-1][1] != actual:
            self._tiempos.append((time.monotonic(), actual))
        if len(self._tiempos) >= 2:
            (t0, a0), (t1, a1) = self._tiempos[0], self._tiempos[-1]
            if a1 > a0:
                texto += f"  ·  faltan ~{formatear_duracion((t1 - t0) / (a1 - a0) * (total - actual))}"
        self.contador.setText(texto)

    def _cancelar(self) -> None:
        # El hilo abandona el cuadro en curso y después avisa; recién ahí se cierra el diálogo
        self.boton_cancelar.setEnabled(False)
        self.contador.setText("Cancelando…")
        self.cancelado.emit()

    def reject(self):
        # Esc o la X de la ventana cancelan el video, igual que el botón
        if self._terminado:
            super().reject()
        else:
            self._cancelar()

    def terminar(self) -> None:
        self._terminado = True
        self.accept()


# ============================================================
#                       ATAJOS
# ============================================================

ATAJOS = [
    ("Rueda del mouse", "Zoom hacia el cursor"),
    ("Clic izquierdo / derecho", "Acercar / alejar en ese punto"),
    ("Mantener y arrastrar con el izquierdo", "Zoom al rectángulo marcado"),
    ("Arrastrar con el del medio", "Mover la vista"),
    ("Flechas, WASD", "Mover la vista"),
    ("+  /  −", "Acercar / alejar al centro"),
    ("Backspace, Alt+Izquierda, botón atrás del mouse", "Vista anterior"),
    ("Shift+Backspace, Alt+Derecha", "Vista siguiente"),
    ("P  /  O", "Paleta siguiente / anterior"),
    ("G  /  H", "Duplicar / dividir iteraciones"),
    ("B  /  N", "Duplicar / dividir el ciclo de la paleta"),
    ("Q", "Modo foto completa / progresivo"),
    ("R", "Vista inicial"),
    ("F1", "Esta ayuda"),
]


def mostrar_atajos(parent) -> None:
    filas = "".join(f"<tr><td style='padding:2px 18px 2px 0'><b>{tecla}</b></td><td>{accion}</td></tr>"
                    for tecla, accion in ATAJOS)
    dialogo = QtWidgets.QMessageBox(parent)
    dialogo.setWindowTitle("Atajos")
    dialogo.setTextFormat(Qt.RichText)
    dialogo.setText(f"<table>{filas}</table>")
    dialogo.exec()
