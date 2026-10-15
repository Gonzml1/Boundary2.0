import time
import logging
from dataclasses import asdict
from decimal import Decimal, InvalidOperation

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QSettings, QTimer
from PyQt5.QtGui import QColor, QKeySequence
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QInputDialog, QMessageBox, QShortcut

import gui.tema_oscuro as ts
from gui.MandelbrotGUI import Ui_Boundary
from gui.dialogos import (DialogoVideo, DialogoProgresoVideo, WorkerVideo, pedido_desde_plan,
                          mostrar_atajos, formatear_duracion)
from core.modulo_de_calculo_fractales import FRACTAL_REGISTRY, METODOS_PROFUNDOS, LIMITE_DOUBLE
from core.modulo_opengl import MandelbrotWidget
from core.paletas import PALETTE_REGISTRY
from core.render import Renderizador, OpcionesColor, guardar_imagen
from core.video import crear_carpeta_video, cargar_plan
from core import favoritos

log = logging.getLogger(__name__)

RESOLUCIONES_IMAGEN = {"Actual": None, "Full HD (1920x1080)": (1920, 1080), "4K (3840x2160)": (3840, 2160)}
# Métodos preferidos al arrancar (el primero disponible): son los que soportan todas las opciones
METODOS_PREFERIDOS = ("perturbacion_gpu", "perturbacion_cpu")
# Opciones del color del interior: nombre -> color (None = el de la paleta; "elegir" = diálogo)
COLORES_INTERIOR = {"Color de la paleta": None, "Negro": (0, 0, 0), "Blanco": (255, 255, 255),
                    "Personalizado…": "elegir"}
SEGUNDOS_ANTES_DE_MOSTRAR_CALCULO = 0.5


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Boundary()
        self.ui.setupUi(self)
        ts.tema_oscuro(QtWidgets.QApplication.instance())
        # Preferencias del usuario (en Windows quedan en el registro, HKCU\Software\ABoundary)
        self.ajustes = QSettings("ABoundary", "ABoundary")

        # La vista OpenGL ocupa el contenedor de la izquierda
        self.mandelbrot = MandelbrotWidget(self.ui.contenedor_grafico)
        layout = QtWidgets.QVBoxLayout(self.ui.contenedor_grafico)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.mandelbrot)

        self.favoritos = favoritos.cargar()
        self._color_personalizado = (40, 40, 40)
        self._indice_interior = 0
        self._inicio_calculo = None
        self.worker = None

        self._crear_barra_de_estado()
        self.inicializar_combos()
        self._restaurar_preferencias()
        self._conectar_mvc()
        self._conectar_controles()
        self._crear_atajos()
        self._actualizar_combo_favoritos()

        self.sincronizar_hacia_motor()
        self._actualizar_indicadores_vista()
        self.mandelbrot.setFocus()

    # --- BARRA DE ESTADO ---
    def _crear_barra_de_estado(self):
        barra = self.statusBar()
        self.indicador_calculo = QtWidgets.QLabel()
        self.indicador_aviso = QtWidgets.QLabel()
        self.indicador_aviso.setStyleSheet("color: #e0a060;")
        self.indicador_zoom = QtWidgets.QLabel()
        self.indicador_modo = QtWidgets.QLabel()
        ayuda = QtWidgets.QLabel("F1 atajos")
        ayuda.setStyleSheet(f"color: {ts.TEXTO_TENUE};")
        for w in (self.indicador_calculo, self.indicador_aviso, self.indicador_zoom, self.indicador_modo, ayuda):
            barra.addPermanentWidget(w)
        self._al_cambiar_modo_completo(False)
        # Mientras un render tarda, muestra hace cuánto está calculando
        self._timer_calculo = QTimer(self)
        self._timer_calculo.setInterval(200)
        self._timer_calculo.timeout.connect(self._actualizar_indicador_calculo)

    def _actualizar_indicador_calculo(self):
        if self._inicio_calculo is None:
            return
        transcurrido = time.monotonic() - self._inicio_calculo
        if transcurrido >= SEGUNDOS_ANTES_DE_MOSTRAR_CALCULO:
            self.indicador_calculo.setText(f"calculando… {transcurrido:.1f} s")

    def _al_empezar_render(self):
        self._inicio_calculo = time.monotonic()
        self._timer_calculo.start()

    def _parar_indicador_calculo(self):
        self._inicio_calculo = None
        self._timer_calculo.stop()
        self.indicador_calculo.clear()

    def _actualizar_indicadores_vista(self, *args):
        m = self.mandelbrot
        self.indicador_zoom.setText(f"zoom {m.ancho_vista():.2e}")
        # Aviso si el método elegido no tiene precisión para esta profundidad
        profundo = (m.tipo_fractal, m.tipo_calculo) in METODOS_PROFUNDOS
        if m.tipo_calculo and not profundo and m.paso_pixel() < LIMITE_DOUBLE:
            self.indicador_aviso.setText("⚠ este método no alcanza a esta profundidad: usá perturbación")
        else:
            self.indicador_aviso.clear()

    # --- COMBOS ---
    def inicializar_combos(self):
        self.ui.tipo_fractal_comboBox.addItems(list(FRACTAL_REGISTRY.keys()))
        if self.ui.tipo_fractal_comboBox.count() > 0:
            self.actualizar_combo_calculo(self.ui.tipo_fractal_comboBox.currentText())
        self.ui.paleta_comboBox.addItems([nombre for nombre, _ in PALETTE_REGISTRY])
        self.ui.interior_comboBox.addItems(list(COLORES_INTERIOR))

    def _metodo_por_defecto(self, fractal: str) -> str | None:
        metodos = list(FRACTAL_REGISTRY.get(fractal, {}))
        guardado = self.ajustes.value("fractal/metodo", "")
        for candidato in (guardado, *METODOS_PREFERIDOS):
            if candidato in metodos:
                return candidato
        return metodos[0] if metodos else None

    def actualizar_combo_calculo(self, fractal):
        combo = self.ui.tipo_calculo_comboBox
        anterior = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(list(FRACTAL_REGISTRY.get(fractal, {}).keys()))
        # Si el método anterior existe también para este fractal se mantiene; si no, el preferido
        elegido = anterior if combo.findText(anterior) >= 0 else self._metodo_por_defecto(fractal)
        if elegido:
            combo.setCurrentText(elegido)
        combo.blockSignals(False)
        es_julia = fractal == "Julia"
        for w in (self.ui.julia_re_entrada, self.ui.julia_im_entrada,
                  self.ui.julia_re_label, self.ui.julia_im_label):
            w.setEnabled(es_julia)

    # --- PREFERENCIAS ---
    def _restaurar_preferencias(self):
        a, ui = self.ajustes, self.ui
        geometria = a.value("ventana/geometria")
        if geometria is not None:
            self.restoreGeometry(geometria)
        fractal = a.value("fractal/tipo", "")
        if ui.tipo_fractal_comboBox.findText(fractal) >= 0:
            ui.tipo_fractal_comboBox.setCurrentText(fractal)
        self.actualizar_combo_calculo(ui.tipo_fractal_comboBox.currentText())
        paleta = a.value("color/paleta", "")
        if ui.paleta_comboBox.findText(paleta) >= 0:
            ui.paleta_comboBox.setCurrentText(paleta)
        ui.clase_equiv_entrada.setText(str(a.value("color/ciclo", 128)))
        ui.suave_checkBox.setChecked(a.value("color/suave", False, type=bool))
        ui.iluminacion_checkBox.setChecked(a.value("color/iluminacion", False, type=bool))
        ui.antialiasing_checkBox.setChecked(a.value("color/antialiasing", False, type=bool))
        angulo = int(a.value("color/angulo_luz", 45))
        ui.angulo_luz_slider.setValue(angulo)
        ui.angulo_luz_label.setText(f"Luz {angulo}°")
        personalizado = a.value("color/interior_personalizado", "")
        if QColor(personalizado).isValid():
            c = QColor(personalizado)
            self._color_personalizado = (c.red(), c.green(), c.blue())
        interior = a.value("color/interior", "Color de la paleta")
        if ui.interior_comboBox.findText(interior) >= 0:
            ui.interior_comboBox.setCurrentText(interior)
        self._indice_interior = ui.interior_comboBox.currentIndex()
        ui.julia_re_entrada.setText(str(a.value("fractal/julia_re", "-0.8")))
        ui.julia_im_entrada.setText(str(a.value("fractal/julia_im", "0.156")))
        if a.value("render/modo_completo", False, type=bool):
            self.mandelbrot.toggle_modo_completo()
            self._al_cambiar_modo_completo(True)

    def _guardar_preferencias(self):
        a, ui = self.ajustes, self.ui
        a.setValue("ventana/geometria", self.saveGeometry())
        a.setValue("fractal/tipo", ui.tipo_fractal_comboBox.currentText())
        a.setValue("fractal/metodo", ui.tipo_calculo_comboBox.currentText())
        a.setValue("fractal/julia_re", ui.julia_re_entrada.text())
        a.setValue("fractal/julia_im", ui.julia_im_entrada.text())
        a.setValue("color/paleta", ui.paleta_comboBox.currentText())
        a.setValue("color/ciclo", ui.clase_equiv_entrada.text())
        a.setValue("color/suave", ui.suave_checkBox.isChecked())
        a.setValue("color/iluminacion", ui.iluminacion_checkBox.isChecked())
        a.setValue("color/antialiasing", ui.antialiasing_checkBox.isChecked())
        a.setValue("color/angulo_luz", ui.angulo_luz_slider.value())
        a.setValue("color/interior", ui.interior_comboBox.currentText())
        a.setValue("color/interior_personalizado", QColor(*self._color_personalizado).name())
        a.setValue("render/modo_completo", self.mandelbrot.modo_completo)

    # --- ARQUITECTURA MVC ---
    def _conectar_mvc(self):
        m = self.mandelbrot
        m.coordenadas_raton_cambiadas.connect(self.actualizar_textos_coordenadas)
        m.limites_zoom_cambiados.connect(self.actualizar_cajas_zoom)
        m.limites_zoom_cambiados.connect(self._actualizar_indicadores_vista)
        m.parametros_matematicos_cambiados.connect(self.actualizar_cajas_matematicas)
        m.paleta_cambiada.connect(self._al_cambiar_paleta_desde_teclado)
        m.render_empezado.connect(self._al_empezar_render)
        m.render_terminado.connect(self._al_terminar_render)
        m.modo_completo_cambiado.connect(self._al_cambiar_modo_completo)
        m.render_fallido.connect(self._al_fallar_render)

    def _conectar_controles(self):
        ui = self.ui
        ui.tipo_fractal_comboBox.currentTextChanged.connect(self.actualizar_combo_calculo)
        for combo in (ui.tipo_fractal_comboBox, ui.tipo_calculo_comboBox, ui.paleta_comboBox):
            combo.currentIndexChanged.connect(self.sincronizar_hacia_motor)
        ui.interior_comboBox.currentIndexChanged.connect(self._al_elegir_interior)
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

    def _crear_atajos(self):
        # A nivel ventana, así andan aunque el foco esté en otro control
        QShortcut(QKeySequence(Qt.Key_F1), self, lambda: mostrar_atajos(self))
        QShortcut(QKeySequence("Alt+Left"), self, self.mandelbrot.atras)
        QShortcut(QKeySequence("Alt+Right"), self, self.mandelbrot.adelante)

    def color_interior(self):
        valor = COLORES_INTERIOR[self.ui.interior_comboBox.currentText()]
        return self._color_personalizado if valor == "elegir" else valor

    def opciones_desde_ui(self) -> OpcionesColor:
        ui = self.ui
        return OpcionesColor(
            suave=ui.suave_checkBox.isChecked(),
            iluminacion=ui.iluminacion_checkBox.isChecked(),
            antialiasing=ui.antialiasing_checkBox.isChecked(),
            angulo_luz=float(ui.angulo_luz_slider.value()),
            interior=self.color_interior(),
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
        self._actualizar_indicadores_vista()

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

    def _al_elegir_interior(self, indice):
        if COLORES_INTERIOR[self.ui.interior_comboBox.currentText()] == "elegir":
            color = QtWidgets.QColorDialog.getColor(QColor(*self._color_personalizado), self, "Color del interior")
            if not color.isValid():
                # Se canceló: vuelve a la opción anterior
                self.ui.interior_comboBox.blockSignals(True)
                self.ui.interior_comboBox.setCurrentIndex(self._indice_interior)
                self.ui.interior_comboBox.blockSignals(False)
                return
            self._color_personalizado = (color.red(), color.green(), color.blue())
        self._indice_interior = indice
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

    def _al_terminar_render(self, segundos, aplicadas, ancho, alto, final, recalculado):
        if not final:
            return
        self._parar_indicador_calculo()
        pedidas = self.opciones_desde_ui()
        accion = "en" if recalculado else "recoloreado en"
        texto = f"{ancho}x{alto} {accion} {segundos * 1000:.0f} ms"
        # Avisa si se pidió algo que el método elegido no sabe calcular
        faltan = [nombre for clave, nombre, activo in (("suave", "coloreo suave", pedidas.suave),
                                                       ("iluminacion", "iluminación", pedidas.iluminacion))
                  if activo and clave not in aplicadas]
        if faltan:
            texto += f"   ({', '.join(faltan)}: solo con los métodos de perturbación)"
        self.statusBar().showMessage(texto)

    def _al_fallar_render(self, mensaje):
        self._parar_indicador_calculo()
        self.statusBar().showMessage(f"Error al calcular: {mensaje}", 10000)

    def actualizar_textos_coordenadas(self, real, imag):
        self.ui.label_coordenadas.setText(f"Re: {real}    Im: {imag}")

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
        controles = (ui.tipo_fractal_comboBox, ui.tipo_calculo_comboBox, ui.paleta_comboBox,
                     ui.interior_comboBox, ui.suave_checkBox, ui.iluminacion_checkBox,
                     ui.antialiasing_checkBox, ui.angulo_luz_slider)
        # Se cargan los controles y después se sincroniza todo de una vez
        for w in controles:
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
        self._seleccionar_interior(op.interior)
        for w in controles:
            w.blockSignals(False)
        ui.max_iter_entrada.setText(str(fav.max_iter))
        ui.clase_equiv_entrada.setText(str(fav.clase_equiv))
        ui.julia_re_entrada.setText(fav.real)
        ui.julia_im_entrada.setText(fav.imag)
        self.mandelbrot.xmin, self.mandelbrot.xmax = Decimal(fav.xmin), Decimal(fav.xmax)
        self.mandelbrot.ymin, self.mandelbrot.ymax = Decimal(fav.ymin), Decimal(fav.ymax)
        self.mandelbrot.corregir_aspect_ratio()
        self.sincronizar_hacia_motor()

    def _seleccionar_interior(self, color):
        """Pone el combo del interior en la opción que corresponde al color guardado."""
        color = tuple(color) if color is not None else None
        for nombre, valor in COLORES_INTERIOR.items():
            if valor == color:
                self.ui.interior_comboBox.setCurrentText(nombre)
                break
        else:
            self._color_personalizado = color
            self.ui.interior_comboBox.setCurrentText("Personalizado…")
        self._indice_interior = self.ui.interior_comboBox.currentIndex()

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

    # --- EXPORTACIÓN ---
    def _elegir_resolucion(self, titulo):
        opcion, ok = QInputDialog.getItem(self, titulo, "Seleccionar resolución:", list(RESOLUCIONES_IMAGEN), 0, False)
        if not ok:
            return None
        return RESOLUCIONES_IMAGEN[opcion] or self.mandelbrot.tamano_render()

    def guardar_imagen_dialogo(self):
        resolucion = self._elegir_resolucion("Guardar imagen")
        if resolucion is None: return
        carpeta = self.ajustes.value("carpetas/imagenes", "")
        sugerido = f"fractal_{self.mandelbrot.tipo_fractal}.png"
        ruta, _ = QFileDialog.getSaveFileName(
            self, "Guardar imagen", f"{carpeta}/{sugerido}" if carpeta else sugerido,
            "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not ruta: return
        self.ajustes.setValue("carpetas/imagenes", ruta.rsplit("/", 1)[0])

        self.sincronizar_hacia_motor()
        pedido = self.mandelbrot.pedido_actual(*resolucion)
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Renderizador propio: el de la pantalla vive en el hilo de render
            imagen = Renderizador(tam_cache=0).renderizar(pedido)
            guardar_imagen(ruta, imagen.rgb)
        except Exception as e:
            log.exception("No se pudo guardar la imagen")
            QMessageBox.critical(self, "Guardar imagen", f"No se pudo guardar la imagen:\n{e}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        log.info("Imagen guardada en %dx%d: %s", pedido.ancho, pedido.alto, ruta)
        self.statusBar().showMessage(f"Imagen guardada: {ruta}", 10000)

    def video_dialogo(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Video", "Ya hay un video renderizándose.")
            return
        self.sincronizar_hacia_motor()
        dialogo = DialogoVideo(self.ajustes, self.mandelbrot.tamano_render(), self.mandelbrot.pedido_actual, self)
        if dialogo.exec() != QtWidgets.QDialog.Accepted:
            return

        if dialogo.accion == "continuar":
            carpeta = dialogo.carpeta_continuar
            datos, config = cargar_plan(carpeta)
            pedido = pedido_desde_plan(datos, config)
        else:
            config = dialogo.config()
            base = QFileDialog.getExistingDirectory(
                self, "Seleccionar dónde guardar el video (se crea una subcarpeta nueva)",
                self.ajustes.value("carpetas/videos", ""))
            if not base:
                return
            self.ajustes.setValue("carpetas/videos", base)
            try:
                # Cada video en su propia subcarpeta, para no mezclar cuadros con videos anteriores
                carpeta = crear_carpeta_video(base)
            except OSError as e:
                QMessageBox.critical(self, "Video", f"No se pudo crear la carpeta del video:\n{e}")
                return
            pedido = self.mandelbrot.pedido_actual(config.ancho, config.alto)

        self.statusBar().showMessage(f"Guardando el video en {carpeta}", 10000)
        texto = (f"Zoom desde el fractal entero hasta la vista elegida\n"
                 f"{config.cuadros} cuadros de {config.ancho}x{config.alto} "
                 f"({formatear_duracion(config.cuadros / config.fps)} de video)")
        self.progreso_ui = DialogoProgresoVideo(texto, config.cuadros, self)
        self.progreso_ui.show()

        self.worker = WorkerVideo(pedido, carpeta, config)
        self.worker.empezo.connect(self.progreso_ui.empezar)
        self.worker.progreso.connect(self.progreso_ui.actualizar)
        self.worker.terminado.connect(self._video_terminado)
        self.progreso_ui.cancelado.connect(self.worker.cancelar)
        self.worker.start()

    def _video_terminado(self, mensaje):
        self.progreso_ui.terminar()
        log.info(mensaje)
        self.statusBar().showMessage(mensaje, 30000)

    def closeEvent(self, event):
        self._guardar_preferencias()
        self.mandelbrot.detener()
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancelar()
            self.worker.wait()
        super().closeEvent(event)
