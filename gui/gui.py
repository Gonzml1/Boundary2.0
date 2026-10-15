import os
import logging
from dataclasses import replace, asdict
from decimal import Decimal, InvalidOperation

import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QInputDialog, QProgressDialog, QMessageBox

import gui.tema_oscuro as ts
from gui.MandelbrotGUI import Ui_Boundary
from core.modulo_de_calculo_fractales import FRACTAL_REGISTRY, calculos_mandelbrot
from core.modulo_opengl import MandelbrotWidget, VISTA_INICIAL
from core.paletas import PALETTE_REGISTRY
from core.render import renderizar, PedidoRender, OpcionesColor
from core.video import planificar_zoom, armar_mp4
from core import favoritos

log = logging.getLogger(__name__)

RESOLUCIONES = {"Actual": None, "Full HD (1920x1080)": (1920, 1080), "4K (3840x2160)": (3840, 2160)}


def _guardar_png(ruta: str, rgb: np.ndarray) -> None:
    # La fila 0 de la imagen es ymin: se invierte para que no salga al revés
    plt.imsave(ruta, rgb[::-1])


# --- HILO EN SEGUNDO PLANO PARA EL VIDEO ---
class WorkerVideo(QThread):
    """Renderiza un zoom desde el fractal entero hasta la vista del pedido (ver core/video.py)."""
    progreso = pyqtSignal(int, int)  # frame_actual, total_frames
    terminado = pyqtSignal(str)      # mensaje final

    def __init__(self, pedido: PedidoRender, carpeta: str, fps: int, duracion: float):
        super().__init__()
        self.pedido = pedido
        self.carpeta = carpeta
        self.fps = fps
        self.cuadros = max(1, round(fps * duracion))

    def run(self):
        p = self.pedido
        # Motor EXCLUSIVO para este hilo, para no pisarse con el de la pantalla
        motor = calculos_mandelbrot(p.xmin, p.xmax, p.ymin, p.ymax, p.ancho, p.alto,
                                    p.max_iter, p.tipo_calculo, p.tipo_fractal)
        plan = planificar_zoom(p.xmin, p.xmax, p.ymin, p.ymax, p.ancho, p.alto, self.cuadros, p.max_iter)

        for i, cuadro in enumerate(plan):
            if self.isInterruptionRequested():
                self.terminado.emit(f"Video cancelado: quedaron {i} cuadros en {self.carpeta}")
                return
            # Los cuadros alejados usan menos iteraciones, pero se colorean contra el máximo final
            pedido = replace(p, xmin=cuadro.xmin, xmax=cuadro.xmax, ymin=cuadro.ymin, ymax=cuadro.ymax,
                             max_iter=cuadro.max_iter, max_iter_color=p.max_iter)
            imagen = renderizar(motor, pedido)
            _guardar_png(os.path.join(self.carpeta, f"frame_{i:05d}.png"), imagen.rgb)
            self.progreso.emit(i + 1, self.cuadros)

        mp4 = armar_mp4(self.carpeta, self.fps)
        self.terminado.emit(f"Video listo: {mp4}" if mp4 else f"Cuadros listos en {self.carpeta}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Boundary()
        self.ui.setupUi(self)
        ts.tema_oscuro(QtWidgets.QApplication.instance())

        # La vista OpenGL ocupa el contenedor de la izquierda
        self.mandelbrot = MandelbrotWidget(self.ui.contenedor_grafico)
        layout = QtWidgets.QVBoxLayout(self.ui.contenedor_grafico)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.mandelbrot)

        self.favoritos = favoritos.cargar()

        # Indicador fijo del modo de render, a la derecha de la barra de estado
        self.indicador_modo = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.indicador_modo)
        self._al_cambiar_modo_completo(False)

        self.inicializar_combos()
        self._conectar_mvc()
        self._conectar_controles()
        self._actualizar_combo_favoritos()

        self.sincronizar_hacia_motor()
        self.mandelbrot.setFocus()

    # --- COMBOS ---
    def inicializar_combos(self):
        self.ui.tipo_fractal_comboBox.addItems(list(FRACTAL_REGISTRY.keys()))
        if self.ui.tipo_fractal_comboBox.count() > 0:
            self.actualizar_combo_calculo(self.ui.tipo_fractal_comboBox.currentText())
        self.ui.paleta_comboBox.addItems([nombre for nombre, _ in PALETTE_REGISTRY])

    def actualizar_combo_calculo(self, fractal):
        combo = self.ui.tipo_calculo_comboBox
        anterior = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(list(FRACTAL_REGISTRY.get(fractal, {}).keys()))
        # Si el método anterior existe también para este fractal, se mantiene
        if combo.findText(anterior) >= 0:
            combo.setCurrentText(anterior)
        combo.blockSignals(False)
        es_julia = fractal == "Julia"
        for w in (self.ui.julia_re_entrada, self.ui.julia_im_entrada,
                  self.ui.julia_re_label, self.ui.julia_im_label):
            w.setEnabled(es_julia)

    # --- ARQUITECTURA MVC ---
    def _conectar_mvc(self):
        self.mandelbrot.coordenadas_raton_cambiadas.connect(self.actualizar_textos_coordenadas)
        self.mandelbrot.limites_zoom_cambiados.connect(self.actualizar_cajas_zoom)
        self.mandelbrot.parametros_matematicos_cambiados.connect(self.actualizar_cajas_matematicas)
        self.mandelbrot.paleta_cambiada.connect(self._al_cambiar_paleta_desde_teclado)
        self.mandelbrot.render_terminado.connect(self._al_terminar_render)
        self.mandelbrot.modo_completo_cambiado.connect(self._al_cambiar_modo_completo)
        self.mandelbrot.render_fallido.connect(
            lambda mensaje: self.statusBar().showMessage(f"Error al calcular: {mensaje}", 10000))

    def _conectar_controles(self):
        ui = self.ui
        ui.tipo_fractal_comboBox.currentTextChanged.connect(self.actualizar_combo_calculo)
        for combo in (ui.tipo_fractal_comboBox, ui.tipo_calculo_comboBox, ui.paleta_comboBox):
            combo.currentIndexChanged.connect(self.sincronizar_hacia_motor)
        for entrada in (ui.max_iter_entrada, ui.clase_equiv_entrada, ui.zoom_in_factor_entrada,
                        ui.zoom_out_factor_entrada, ui.julia_re_entrada, ui.julia_im_entrada):
            entrada.editingFinished.connect(self.sincronizar_hacia_motor)
        # Los límites solo se aplican al terminar de editarlos (sino el aspecto los corrige a mitad)
        for entrada in (ui.xmin_entrada, ui.xmax_entrada, ui.ymin_entrada, ui.ymax_entrada):
            entrada.editingFinished.connect(self.aplicar_vista_desde_cajas)
        for casilla in (ui.suave_checkBox, ui.iluminacion_checkBox, ui.antialiasing_checkBox):
            casilla.toggled.connect(self.sincronizar_hacia_motor)
        ui.angulo_luz_slider.valueChanged.connect(self._al_mover_luz)

        ui.boton_duplicar.clicked.connect(lambda: self._multiplicar_entrada(ui.max_iter_entrada, 2))
        ui.boton_dividir.clicked.connect(lambda: self._multiplicar_entrada(ui.max_iter_entrada, 0.5))
        ui.boton_duplicar_clase_equiv.clicked.connect(lambda: self._multiplicar_entrada(ui.clase_equiv_entrada, 2))
        ui.boton_dividir_clase_equiv.clicked.connect(lambda: self._multiplicar_entrada(ui.clase_equiv_entrada, 0.5))

        ui.boton_favorito_ir.clicked.connect(self.ir_a_favorito)
        ui.boton_favorito_guardar.clicked.connect(self.guardar_favorito)
        ui.boton_favorito_borrar.clicked.connect(self.borrar_favorito)

        ui.boton_guardar.clicked.connect(self.guardar_imagen_dialogo)
        ui.boton_hacer_fractal.clicked.connect(self.video_dialogo)
        ui.boton_resetear.clicked.connect(self.resetear_vista)

    def opciones_desde_ui(self) -> OpcionesColor:
        ui = self.ui
        return OpcionesColor(
            suave=ui.suave_checkBox.isChecked(),
            iluminacion=ui.iluminacion_checkBox.isChecked(),
            antialiasing=ui.antialiasing_checkBox.isChecked(),
            angulo_luz=float(ui.angulo_luz_slider.value()),
        )

    def sincronizar_hacia_motor(self, *args):
        """Pasa al motor todo lo de los controles, menos los límites de la vista."""
        ui = self.ui
        try:
            parametros = dict(
                max_iter=max(1, int(ui.max_iter_entrada.text())),
                clase_equiv=max(1, int(ui.clase_equiv_entrada.text())),
                zoom_in=Decimal(ui.zoom_in_factor_entrada.text()),
                zoom_out=Decimal(ui.zoom_out_factor_entrada.text()),
                real=Decimal(ui.julia_re_entrada.text()),
                imag=Decimal(ui.julia_im_entrada.text()),
            )
        except (ValueError, InvalidOperation):
            self.statusBar().showMessage("Hay un valor inválido en los controles", 5000)
            return
        ui.angulo_luz_slider.setEnabled(ui.iluminacion_checkBox.isChecked())
        self.mandelbrot.configurar(
            tipo_fractal=ui.tipo_fractal_comboBox.currentText(),
            tipo_calculo=ui.tipo_calculo_comboBox.currentText(),
            palette_index=max(0, ui.paleta_comboBox.currentIndex()),
            opciones=self.opciones_desde_ui(),
            **parametros,
        )

    def aplicar_vista_desde_cajas(self):
        try:
            limites = [Decimal(e.text()) for e in (self.ui.xmin_entrada, self.ui.xmax_entrada,
                                                    self.ui.ymin_entrada, self.ui.ymax_entrada)]
        except InvalidOperation:
            self.statusBar().showMessage("Límite de la vista inválido", 5000)
            return
        self.mandelbrot.fijar_vista(*limites)

    def _multiplicar_entrada(self, entrada, factor):
        try:
            entrada.setText(str(max(1, int(int(entrada.text()) * factor))))
        except ValueError:
            return
        self.sincronizar_hacia_motor()

    def _al_mover_luz(self, valor):
        self.ui.angulo_luz_label.setText(f"Luz {valor}°")
        self.sincronizar_hacia_motor()

    def _al_cambiar_modo_completo(self, activo):
        if activo:
            self.indicador_modo.setText("FOTO COMPLETA  [Q]")
            self.indicador_modo.setStyleSheet(f"color: {ts.ACENTO};")
        else:
            self.indicador_modo.setText("PROGRESIVO  [Q]")
            self.indicador_modo.setStyleSheet(f"color: {ts.TEXTO_TENUE};")

    def _al_cambiar_paleta_desde_teclado(self, indice):
        self.ui.paleta_comboBox.blockSignals(True)
        self.ui.paleta_comboBox.setCurrentIndex(indice)
        self.ui.paleta_comboBox.blockSignals(False)

    def _al_terminar_render(self, segundos, aplicadas, ancho, alto, final):
        if not final:
            return
        pedidas = self.opciones_desde_ui()
        texto = f"{ancho}x{alto} en {segundos * 1000:.0f} ms"
        # Avisa si se pidió algo que el método elegido no sabe calcular
        faltan = [nombre for clave, nombre, activo in (("suave", "coloreo suave", pedidas.suave),
                                                       ("iluminacion", "iluminación", pedidas.iluminacion))
                  if activo and clave not in aplicadas]
        if faltan:
            texto += f"   ({', '.join(faltan)}: solo con los métodos de perturbación)"
        self.statusBar().showMessage(texto)

    def actualizar_textos_coordenadas(self, real, imag):
        self.ui.label_coordenadas.setText(f"Re: {real:.16f}, Im: {imag:.16f}")

    def actualizar_cajas_zoom(self, xmin, xmax, ymin, ymax):
        self.ui.xmin_entrada.setText(str(xmin))
        self.ui.xmax_entrada.setText(str(xmax))
        self.ui.ymin_entrada.setText(str(ymin))
        self.ui.ymax_entrada.setText(str(ymax))
        for entrada in (self.ui.xmin_entrada, self.ui.xmax_entrada, self.ui.ymin_entrada, self.ui.ymax_entrada):
            entrada.setCursorPosition(0)

    def actualizar_cajas_matematicas(self, max_iter, clase_equiv):
        self.ui.max_iter_entrada.setText(str(max_iter))
        self.ui.clase_equiv_entrada.setText(str(clase_equiv))

    def resetear_vista(self):
        self.mandelbrot.resetear()

    # --- FAVORITOS ---
    def _actualizar_combo_favoritos(self, seleccionar: str | None = None):
        combo = self.ui.favoritos_comboBox
        combo.clear()
        combo.addItems([f.nombre for f in self.favoritos])
        if seleccionar is not None:
            combo.setCurrentText(seleccionar)
        hay = bool(self.favoritos)
        self.ui.boton_favorito_ir.setEnabled(hay)
        self.ui.boton_favorito_borrar.setEnabled(hay)

    def guardar_favorito(self):
        nombre, ok = QInputDialog.getText(self, "Guardar favorito", "Nombre:")
        nombre = nombre.strip()
        if not ok or not nombre:
            return
        m = self.mandelbrot
        xmin, xmax, ymin, ymax = m.limites_como_texto()
        nuevo = favoritos.Favorito(
            nombre=nombre, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            max_iter=m.max_iter, tipo_fractal=m.tipo_fractal, tipo_calculo=m.tipo_calculo,
            paleta=PALETTE_REGISTRY[m.palette_index][0], clase_equiv=m.clase_equiv,
            real=str(m.real), imag=str(m.imag), opciones=asdict(m.opciones),
        )
        existente = next((i for i, f in enumerate(self.favoritos) if f.nombre == nombre), None)
        if existente is not None:
            if QMessageBox.question(self, "Guardar favorito", f"Ya existe '{nombre}'. ¿Reemplazarlo?") != QMessageBox.Yes:
                return
            self.favoritos[existente] = nuevo
        else:
            self.favoritos.append(nuevo)
        favoritos.guardar(self.favoritos)
        self._actualizar_combo_favoritos(seleccionar=nombre)
        self.statusBar().showMessage(f"Favorito '{nombre}' guardado", 5000)

    def ir_a_favorito(self):
        indice = self.ui.favoritos_comboBox.currentIndex()
        if not 0 <= indice < len(self.favoritos):
            return
        fav = self.favoritos[indice]
        ui = self.ui
        # Se cargan los controles y después se sincroniza todo de una vez
        for w in (ui.tipo_fractal_comboBox, ui.tipo_calculo_comboBox, ui.paleta_comboBox,
                  ui.suave_checkBox, ui.iluminacion_checkBox, ui.antialiasing_checkBox, ui.angulo_luz_slider):
            w.blockSignals(True)
        if ui.tipo_fractal_comboBox.findText(fav.tipo_fractal) >= 0:
            ui.tipo_fractal_comboBox.setCurrentText(fav.tipo_fractal)
            self.actualizar_combo_calculo(fav.tipo_fractal)
        if ui.tipo_calculo_comboBox.findText(fav.tipo_calculo) >= 0:
            ui.tipo_calculo_comboBox.setCurrentText(fav.tipo_calculo)
        else:
            self.statusBar().showMessage(f"El método '{fav.tipo_calculo}' no está disponible acá", 8000)
        if ui.paleta_comboBox.findText(fav.paleta) >= 0:
            ui.paleta_comboBox.setCurrentText(fav.paleta)
        op = OpcionesColor(**{**asdict(OpcionesColor()), **fav.opciones})
        ui.suave_checkBox.setChecked(op.suave)
        ui.iluminacion_checkBox.setChecked(op.iluminacion)
        ui.antialiasing_checkBox.setChecked(op.antialiasing)
        ui.angulo_luz_slider.setValue(int(op.angulo_luz))
        ui.angulo_luz_label.setText(f"Luz {int(op.angulo_luz)}°")
        for w in (ui.tipo_fractal_comboBox, ui.tipo_calculo_comboBox, ui.paleta_comboBox,
                  ui.suave_checkBox, ui.iluminacion_checkBox, ui.antialiasing_checkBox, ui.angulo_luz_slider):
            w.blockSignals(False)
        ui.max_iter_entrada.setText(str(fav.max_iter))
        ui.clase_equiv_entrada.setText(str(fav.clase_equiv))
        ui.julia_re_entrada.setText(fav.real)
        ui.julia_im_entrada.setText(fav.imag)
        self.mandelbrot.xmin, self.mandelbrot.xmax = Decimal(fav.xmin), Decimal(fav.xmax)
        self.mandelbrot.ymin, self.mandelbrot.ymax = Decimal(fav.ymin), Decimal(fav.ymax)
        self.mandelbrot.corregir_aspect_ratio()
        self.sincronizar_hacia_motor()

    def borrar_favorito(self):
        indice = self.ui.favoritos_comboBox.currentIndex()
        if not 0 <= indice < len(self.favoritos):
            return
        nombre = self.favoritos[indice].nombre
        if QMessageBox.question(self, "Borrar favorito", f"¿Borrar '{nombre}'?") != QMessageBox.Yes:
            return
        del self.favoritos[indice]
        favoritos.guardar(self.favoritos)
        self._actualizar_combo_favoritos()

    # --- LÓGICA DE EXPORTACIÓN (Ventanas de Diálogo) ---
    def _elegir_resolucion(self, titulo):
        opcion, ok = QInputDialog.getItem(self, titulo, "Seleccionar resolución:", list(RESOLUCIONES), 0, False)
        if not ok:
            return None
        return RESOLUCIONES[opcion] or self.mandelbrot.tamano_render()

    def guardar_imagen_dialogo(self):
        resolucion = self._elegir_resolucion("Guardar imagen")
        if resolucion is None: return
        ruta, _ = QFileDialog.getSaveFileName(
            self, "Guardar imagen", f"fractal_{self.mandelbrot.tipo_fractal}.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not ruta: return

        self.sincronizar_hacia_motor()
        pedido = self.mandelbrot.pedido_actual(*resolucion)
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Motor propio: el de la pantalla vive en el hilo de render
            motor = calculos_mandelbrot(*VISTA_INICIAL, 1, 1, 1, pedido.tipo_calculo, pedido.tipo_fractal)
            imagen = renderizar(motor, pedido)
            _guardar_png(ruta, imagen.rgb)
        except Exception as e:
            log.exception("No se pudo guardar la imagen")
            QMessageBox.critical(self, "Guardar imagen", f"No se pudo guardar la imagen:\n{e}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        log.info("Imagen guardada en %dx%d: %s", pedido.ancho, pedido.alto, ruta)
        self.statusBar().showMessage(f"Imagen guardada: {ruta}", 10000)

    def video_dialogo(self):
        carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta para el video")
        if not carpeta: return

        resolucion = self._elegir_resolucion("Video")
        if resolucion is None: return

        duracion, ok_dur = QInputDialog.getDouble(self, "Video", "Duración del video (segundos):", 10.0, 0.1, 3600.0, 1)
        if not ok_dur: return

        fps, ok_fps = QInputDialog.getInt(self, "Video", "Fotogramas por segundo:", 30, 1, 240)
        if not ok_fps: return

        self.sincronizar_hacia_motor()
        cuadros = max(1, round(fps * duracion))
        render_w, render_h = resolucion

        self.progreso_ui = QProgressDialog(
            f"Zoom desde el fractal entero hasta la vista actual\n{cuadros} cuadros de {render_w}x{render_h}",
            "Cancelar", 0, cuadros, self)
        self.progreso_ui.setWindowTitle("Renderizando video")
        self.progreso_ui.setMinimumDuration(0)
        self.progreso_ui.show()

        self.worker = WorkerVideo(self.mandelbrot.pedido_actual(render_w, render_h), carpeta, fps, duracion)
        self.worker.progreso.connect(self.progreso_ui.setValue)
        self.worker.terminado.connect(self._video_terminado)
        self.progreso_ui.canceled.connect(self.worker.requestInterruption)
        self.worker.start()

    def _video_terminado(self, mensaje):
        self.progreso_ui.close()
        log.info(mensaje)
        self.statusBar().showMessage(mensaje, 15000)

    def closeEvent(self, event):
        self.mandelbrot.detener()
        if getattr(self, "worker", None) is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait()
        super().closeEvent(event)
