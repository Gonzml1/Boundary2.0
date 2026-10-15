"""Diálogos de la ventana principal: configuración, vista previa y progreso del video, y atajos."""
import os
import time
import shutil
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace, asdict
from decimal import Decimal

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from core.modulo_de_calculo_fractales import RenderCancelado, FRACTAL_REGISTRY, variante_rapida
from core.paletas import PALETTE_REGISTRY
from core.render import Renderizador, PedidoRender, OpcionesColor, guardar_imagen
from core.video import (ConfigVideo, planificar_zoom, armar_mp4, nombre_cuadro, guardar_plan,
                        cargar_plan, cuadros_hechos)

log = logging.getLogger(__name__)

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
    - Si en la carpeta ya hay cuadros hechos (un video cancelado), los saltea."""
    empezo = pyqtSignal(int, int)    # cuadros ya hechos, total
    progreso = pyqtSignal(int, int)  # cuadros hechos, total
    terminado = pyqtSignal(str)      # mensaje final

    def __init__(self, pedido: PedidoRender, carpeta: str, config: ConfigVideo):
        super().__init__()
        self.pedido = pedido
        self.carpeta = carpeta
        self.config = config
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
        p, cfg = self.pedido, self.config
        total = cfg.cuadros
        guardar_plan(self.carpeta, p, cfg, PALETTE_REGISTRY[p.paleta][0])
        self._hechos = cuadros_hechos(self.carpeta, cfg)
        self.empezo.emit(self._hechos, total)
        log.info("Video con %s", " + ".join(self.metodos))

        plan = planificar_zoom(p.xmin, p.xmax, p.ymin, p.ymax, p.ancho, p.alto,
                               cfg.cuadros_zoom, p.max_iter, cfg.suavizado)
        pendientes = iter([(i, c) for i, c in enumerate(plan)
                           if not os.path.exists(os.path.join(self.carpeta, nombre_cuadro(i)))])
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

        # Pausa final: copias del último cuadro (no hace falta volver a calcularlo)
        ultimo = os.path.join(self.carpeta, nombre_cuadro(cfg.cuadros_zoom - 1))
        for i in range(cfg.cuadros_zoom, total):
            ruta = os.path.join(self.carpeta, nombre_cuadro(i))
            if not os.path.exists(ruta):
                shutil.copyfile(ultimo, ruta)
                self._contar_hecho(total)

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

    def run(self):
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

        boton_previa = QtWidgets.QPushButton("Vista previa")
        boton_continuar = QtWidgets.QPushButton("Continuar video…")
        boton_renderizar = QtWidgets.QPushButton("Renderizar…")
        boton_cancelar = QtWidgets.QPushButton("Cancelar")
        boton_previa.clicked.connect(self._vista_previa)
        boton_continuar.clicked.connect(self._continuar)
        boton_renderizar.clicked.connect(self._renderizar)
        boton_cancelar.clicked.connect(self.reject)
        botones = QtWidgets.QHBoxLayout()
        for b in (boton_previa, boton_continuar):
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
        self._actualizar_resumen()

    def config(self) -> ConfigVideo:
        ancho, alto = self.resoluciones[self.resolucion.currentText()]
        return ConfigVideo(ancho, alto, self.duracion.value(), self.fps.value(),
                           self.suavizado.isChecked(), self.pausa.value(), self.usar_cpu.isChecked())

    def _actualizar_resumen(self):
        c = self.config()
        texto = f"{c.cuadros_zoom} cuadros de zoom"
        if c.cuadros_pausa:
            texto += f" + {c.cuadros_pausa} de pausa"
        self.resumen.setText(f"{texto}  ·  video de {formatear_duracion(c.cuadros / c.fps)}")

    def _guardar_ajustes(self):
        self.ajustes.setValue("video/resolucion", self.resolucion.currentText())
        self.ajustes.setValue("video/duracion", self.duracion.value())
        self.ajustes.setValue("video/fps", self.fps.value())
        self.ajustes.setValue("video/suavizado", self.suavizado.isChecked())
        self.ajustes.setValue("video/pausa", self.pausa.value())
        self.ajustes.setValue("video/usar_cpu", self.usar_cpu.isChecked())

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

    def _continuar(self):
        carpeta = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Elegí la carpeta del video sin terminar (zoom_...)",
            self.ajustes.value("carpetas/videos", ""))
        if not carpeta:
            return
        plan = cargar_plan(carpeta)
        if plan is None:
            QtWidgets.QMessageBox.warning(self, "Continuar video",
                                          "Esa carpeta no tiene un video para continuar (falta video.json).")
            return
        _, config = plan
        hechos = cuadros_hechos(carpeta, config)
        if QtWidgets.QMessageBox.question(
                self, "Continuar video",
                f"Video de {config.ancho}x{config.alto}: hay {hechos}/{config.cuadros} cuadros hechos.\n"
                f"¿Seguir desde ahí?") != QtWidgets.QMessageBox.Yes:
            return
        self.carpeta_continuar = carpeta
        self.accion = "continuar"
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
    ("Arrastrar con el izquierdo", "Zoom al rectángulo marcado"),
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
