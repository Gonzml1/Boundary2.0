import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (QMainWindow, QGraphicsEllipseItem, QGraphicsScene, 
                             QGraphicsView, QFileDialog, QInputDialog, QApplication)
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtCore import QPointF, QTimer

import core.funciones_ui as md
import gui.tema_oscuro as ts
from gui.MandelbrotGUI import Ui_Boundary
from OpenGL.GL import *
from core.modulo_de_calculo_fractales import FRACTAL_REGISTRY

class Punto(QGraphicsEllipseItem):
    def __init__(self, callback):
        super().__init__(-10, -10, 20, 20)
        self.setBrush(QBrush(QColor("white")))
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges, True)
        self.callback = callback

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionChange:
            new_x = max(0, min(200, value.x()))
            new_y = max(0, min(200, value.y()))
            self.callback(new_x, new_y)
            return QPointF(new_x, new_y)
        return super().itemChange(change, value) 

class GraphicsViewFlechas(QGraphicsView):
    def __init__(self, *args, punto=None, ui=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.punto = punto
        self.ui = ui

    def keyPressEvent(self, event):
        try:
            paso = float(self.ui.paso_entrada.text())
        except Exception:
            paso = 1
        pos = self.punto.pos()
        if event.key() == QtCore.Qt.Key_Left:
            self.punto.setPos(max(0, pos.x() - paso), pos.y())
        elif event.key() == QtCore.Qt.Key_Right:
            self.punto.setPos(min(200, pos.x() + paso), pos.y())
        elif event.key() == QtCore.Qt.Key_Up:
            self.punto.setPos(pos.x(), max(0, pos.y() - paso))
        elif event.key() == QtCore.Qt.Key_Down:
            self.punto.setPos(pos.x(), min(200, pos.y() + paso))
        super().keyPressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Boundary()
        self.ui.setupUi(self)
        
        # Tema oscuro
        ts.tema_oscuro(QtWidgets.QApplication.instance())

        # Mostrar fractal
        self.mandelbrot = md.mostrar_fractal_opengl(self.ui)

        # Escena y Punto
        self.scene = QGraphicsScene(0, 0, 200, 200)
        self.punto = Punto(self.actualizar_coordenadas)
        self.scene.addItem(self.punto)
        self.punto.setPos(100, 100)

        # Configurar Minimapa
        self.ui.graphicsView.__class__ = GraphicsViewFlechas
        self.ui.graphicsView.punto = self.punto
        self.ui.graphicsView.ui = self.ui 
        self.ui.graphicsView.setScene(self.scene)
        self.ui.graphicsView.setSceneRect(0, 0, 200, 200)
        self.ui.graphicsView.setFixedSize(200, 200)
        self.ui.graphicsView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.ui.graphicsView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.ui.graphicsView.setInteractive(True)

        self.ui.formula_entrada.setText("z**2 + c")
        self.scene.changed.connect(self.al_interactuar)
        self.ui.graphicsView.setFocus()
        
        self.inicializar_combos()
        self._conectar_mvc()
        self._conectar_botones()
        
        # Sincronización inicial
        self.sincronizar_hacia_motor()

    def inicializar_combos(self):
        try:
            self.ui.tipo_fractal_comboBox.currentTextChanged.disconnect()
        except TypeError:
            pass

        self.ui.tipo_fractal_comboBox.clear()
        self.ui.tipo_fractal_comboBox.addItems(list(FRACTAL_REGISTRY.keys()))
        
        if self.ui.tipo_fractal_comboBox.count() > 0:
            primer_fractal = self.ui.tipo_fractal_comboBox.currentText()
            self.actualizar_combo_calculo(primer_fractal)
        
        self.ui.tipo_fractal_comboBox.currentTextChanged.connect(self.actualizar_combo_calculo)

    def actualizar_combo_calculo(self, fractal):
        self.ui.tipo_calculo_comboBox.clear()
        if fractal in FRACTAL_REGISTRY:
            self.ui.tipo_calculo_comboBox.addItems(list(FRACTAL_REGISTRY[fractal].keys()))
            if self.ui.tipo_calculo_comboBox.count() > 0:
                self.ui.tipo_calculo_comboBox.setCurrentIndex(0)

    # --- ARQUITECTURA MVC ---
    def _conectar_mvc(self):
        # Escuchar al Motor
        self.mandelbrot.coordenadas_raton_cambiadas.connect(self.actualizar_textos_coordenadas)
        self.mandelbrot.limites_zoom_cambiados.connect(self.actualizar_cajas_zoom)
        self.mandelbrot.parametros_matematicos_cambiados.connect(self.actualizar_cajas_matematicas)

        # Escuchar a la UI y avisar al Motor
        self.ui.slider_iteraciones.valueChanged.connect(self.sincronizar_hacia_motor)
        self.ui.tipo_fractal_comboBox.currentTextChanged.connect(self.sincronizar_hacia_motor)
        self.ui.tipo_calculo_comboBox.currentTextChanged.connect(self.sincronizar_hacia_motor)
        self.ui.cmap_comboBox.currentTextChanged.connect(self.sincronizar_hacia_motor)

    def _conectar_botones(self):
        self.ui.boton_guardar.clicked.connect(self.guardar_imagen_dialogo)
        self.ui.boton_hacer_fractal.clicked.connect(self.video_dialogo)
        
        self.ui.boton_duplicar.clicked.connect(self.duplicar_iter)
        self.ui.boton_dividir.clicked.connect(self.dividir_iter)
        self.ui.boton_duplicar_clase_equiv.clicked.connect(self.duplicar_clase)
        self.ui.boton_dividir_clase_equiv.clicked.connect(self.dividir_clase)
        self.ui.boton_resetear.clicked.connect(self.resetear_vista)

    # --- MÉTODOS DE SINCRONIZACIÓN ---
    def sincronizar_hacia_motor(self, *args):
        # Lee la fuente de verdad (UI) y actualiza el estado del motor
        try:
            cmap = self.ui.cmap_comboBox.currentText()
            xmin = float(self.ui.xmin_entrada.text())
            xmax = float(self.ui.xmax_entrada.text())
            ymin = float(self.ui.ymin_entrada.text())
            ymax = float(self.ui.ymax_entrada.text())
            width = int(self.ui.width_entrada.text())
            height = int(self.ui.high_entrada.text())
            max_iter = int(self.ui.max_iter_entrada.text())
            formula = self.ui.formula_entrada.text()
            tipo_calculo = self.ui.tipo_calculo_comboBox.currentText()
            tipo_fractal = self.ui.tipo_fractal_comboBox.currentText()
            zoom_in = float(self.ui.zoom_in_factor_entrada.text())
            zoom_out = float(self.ui.zoom_out_factor_entrada.text())
            clase_equiv = int(self.ui.clase_equiv_entrada.text())
            real = float(self.ui.real_julia_entrada.text())
            imag = float(self.ui.im_julia_entrada.text())
            
            self.mandelbrot.actualizar_estado_desde_ui(
                cmap, width, height, max_iter, formula, tipo_calculo, tipo_fractal, 
                zoom_in, zoom_out, clase_equiv, real, imag
            )
            # Resincronizar límites por si hubo cambios manuales directos en el código
            self.mandelbrot.xmin, self.mandelbrot.xmax = xmin, xmax
            self.mandelbrot.ymin, self.mandelbrot.ymax = ymin, ymax
            self.mandelbrot.update()
        except ValueError:
            pass # Ignorar si el usuario está escribiendo un número incompleto ("-")

    def actualizar_textos_coordenadas(self, real, imag):
        self.ui.label_coordenadas.setText(f"Re: {real:.16f}, Im: {imag:.16f}")

    def actualizar_cajas_zoom(self, xmin, xmax, ymin, ymax):
        self.ui.xmin_entrada.setText(str(xmin))
        self.ui.xmax_entrada.setText(str(xmax))
        self.ui.ymin_entrada.setText(str(ymin))
        self.ui.ymax_entrada.setText(str(ymax))

    def actualizar_cajas_matematicas(self, max_iter, clase_equiv):
        self.ui.max_iter_entrada.setText(str(max_iter))
        self.ui.clase_equiv_entrada.setText(str(clase_equiv))

    def actualizar_coordenadas(self, x, y):
        x_real = (x / 100) * 2 - 2
        y_real = -((y / 100) * 2 - 2)
        self.ui.real_julia_entrada.setText(f"{x_real:.5f}")
        self.ui.im_julia_entrada.setText(f"{y_real:.5f}")
        self.ui.label_coordenadas2.setText(f"Re: {x_real:.5f}, Im: {y_real:.5f}")

    def al_interactuar(self, *args):
        self.mandelbrot.interaccion_rapida()

    # --- LÓGICA DE BOTONES ---
    def duplicar_iter(self):
        self.ui.max_iter_entrada.setText(str(int(self.ui.max_iter_entrada.text()) * 2))
        self.sincronizar_hacia_motor()

    def dividir_iter(self):
        self.ui.max_iter_entrada.setText(str(max(1, int(self.ui.max_iter_entrada.text()) // 2)))
        self.sincronizar_hacia_motor()

    def duplicar_clase(self):
        self.ui.clase_equiv_entrada.setText(str(int(self.ui.clase_equiv_entrada.text()) * 2))
        self.sincronizar_hacia_motor()

    def dividir_clase(self):
        self.ui.clase_equiv_entrada.setText(str(max(1, int(self.ui.clase_equiv_entrada.text()) // 2)))
        self.sincronizar_hacia_motor()

    def resetear_vista(self):
        self.actualizar_cajas_zoom(-2.0, 1.2, -0.9, 0.9)
        self.actualizar_cajas_matematicas(256, 128)
        self.sincronizar_hacia_motor()

    # --- LÓGICA DE EXPORTACIÓN (Ventanas de Diálogo) ---
    def guardar_imagen_dialogo(self):
        resoluciones = ["Actual (como se ve en pantalla)", "Full HD (1920x1080)", "4K (3840x2160)"]
        res_elegida, ok_res = QInputDialog.getItem(self, "Guardar Imagen", "Seleccionar resolución:", resoluciones, 0, False)
        if not ok_res: return

        if "Full HD" in res_elegida: render_w, render_h = 1920, 1080
        elif "4K" in res_elegida: render_w, render_h = 3840, 2160
        else: render_w, render_h = self.mandelbrot.width, self.mandelbrot.height

        ruta, _ = QFileDialog.getSaveFileName(
            self, "Guardar imagen", f"fractal_{self.mandelbrot.tipo_fractal}.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not ruta: return

        self.sincronizar_hacia_motor()
        rgb = self.mandelbrot.obtener_frame_rgb(render_w, render_h)
        plt.imsave(ruta, rgb.astype(np.uint8))
        print(f"Imagen guardada en {render_w}x{render_h}: {ruta}")
        self.mandelbrot.interaccion_rapida()

    def video_dialogo(self):
        carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta para los frames")
        if not carpeta: return

        frames, ok_frames = QInputDialog.getInt(self, "Video", "Cantidad de frames a renderizar:", 60, 1, 10000)
        if not ok_frames: return

        factor_zoom, ok_zoom = QInputDialog.getDouble(self, "Video", "Factor de zoom por frame:", 1.03, 0.001, 10.0, 4)
        if not ok_zoom: return

        resoluciones = ["Actual (como se ve en pantalla)", "Full HD (1920x1080)", "4K (3840x2160)"]
        res_elegida, ok_res = QInputDialog.getItem(self, "Video", "Seleccionar resolución:", resoluciones, 0, False)
        if not ok_res: return

        if "Full HD" in res_elegida: render_w, render_h = 1920, 1080
        elif "4K" in res_elegida: render_w, render_h = 3840, 2160
        else: render_w, render_h = self.mandelbrot.width, self.mandelbrot.height

        self.sincronizar_hacia_motor()
        
        # Guardar coordenadas originales para restaurarlas al final
        orig_xmin, orig_xmax = self.mandelbrot.xmin, self.mandelbrot.xmax
        orig_ymin, orig_ymax = self.mandelbrot.ymin, self.mandelbrot.ymax

        print(f"Iniciando renderizado de {frames} frames en: {carpeta}")

        for i in range(frames):
            rgb = self.mandelbrot.obtener_frame_rgb(render_w, render_h)
            ruta = os.path.join(carpeta, f"frame_{i:04d}.png")
            plt.imsave(ruta, rgb.astype(np.uint8))
            
            print(f"Frame {i+1}/{frames} guardado. (Faltan: {frames - (i+1)})")
            QApplication.processEvents() # Mantiene la app sin congelarse

            # Aplicar zoom
            c_x = (self.mandelbrot.xmin + self.mandelbrot.xmax) / 2
            c_y = (self.mandelbrot.ymin + self.mandelbrot.ymax) / 2
            dx = (self.mandelbrot.xmax - self.mandelbrot.xmin) * factor_zoom / 2
            dy = (self.mandelbrot.ymax - self.mandelbrot.ymin) * factor_zoom / 2
            self.mandelbrot.xmin, self.mandelbrot.xmax = c_x - dx, c_x + dx
            self.mandelbrot.ymin, self.mandelbrot.ymax = c_y - dy, c_y + dy

        # Restaurar vista
        self.mandelbrot.xmin, self.mandelbrot.xmax = orig_xmin, orig_xmax
        self.mandelbrot.ymin, self.mandelbrot.ymax = orig_ymin, orig_ymax
        self.actualizar_cajas_zoom(orig_xmin, orig_xmax, orig_ymin, orig_ymax)
        self.mandelbrot.update()
        print("¡Secuencia completada!")