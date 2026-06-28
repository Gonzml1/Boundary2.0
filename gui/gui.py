import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (QMainWindow, QGraphicsScene, 
                             QFileDialog, QInputDialog, QProgressDialog)
from PyQt5.QtCore import QThread, pyqtSignal

import core.funciones_ui as md
import gui.tema_oscuro as ts
from gui.MandelbrotGUI import Ui_Boundary
from OpenGL.GL import *
from core.modulo_de_calculo_fractales import FRACTAL_REGISTRY, calculos_mandelbrot
from core.paletas import PALETTE_REGISTRY
from decimal import Decimal, getcontext
getcontext().prec= 100

# --- HILO EN SEGUNDO PLANO PARA EL VIDEO ---
class WorkerVideo(QThread):
    progreso = pyqtSignal(int, int)  # frame_actual, total_frames
    terminado = pyqtSignal()

    def __init__(self, estado, carpeta, frames, render_w, render_h, factor_zoom):
        super().__init__()
        self.estado = estado
        self.carpeta = carpeta
        self.frames = frames
        self.render_w = render_w
        self.render_h = render_h
        self.factor_zoom = factor_zoom

    def run(self):
        # Solución a la Condición de Carrera:
        # Creamos un motor matemático EXCLUSIVO para este hilo secundario.
        motor_clon = calculos_mandelbrot(
            self.estado['xmin'], self.estado['xmax'],
            self.estado['ymin'], self.estado['ymax'],
            self.render_w, self.render_h,
            self.estado['max_iter'], self.estado['formula'],
            self.estado['tipo_calculo'], self.estado['tipo_fractal']
        )
        motor_clon.real = self.estado.get('real', 0.0)
        motor_clon.imag = self.estado.get('imag', 0.0)
        
        xmin, xmax = self.estado['xmin'], self.estado['xmax']
        ymin, ymax = self.estado['ymin'], self.estado['ymax']

        _, func_paleta = PALETTE_REGISTRY[self.estado['palette_index']]

        for i in range(self.frames):
            motor_clon.actualizar_fractal(
                xmin, xmax, ymin, ymax,
                self.render_w, self.render_h,
                self.estado['max_iter'], self.estado['formula'],
                self.estado['tipo_calculo'], self.estado['tipo_fractal'],
                motor_clon.real, motor_clon.imag
            )
            
            data = motor_clon.calcular_fractal()
            norm = data.astype(float) / self.estado['max_iter']
            rgb = func_paleta(norm, self.estado['max_iter'], self.estado['clase_equiv'])
            
            ruta = os.path.join(self.carpeta, f"frame_{i:04d}.png")
            plt.imsave(ruta, rgb.astype(np.uint8))
            
            # Avisamos a la UI cómo vamos
            self.progreso.emit(i + 1, self.frames)

            # Zoom Matemático
            c_x, c_y = (xmin + xmax) / 2, (ymin + ymax) / 2
            dx = (xmax - xmin) * self.factor_zoom / 2
            dy = (ymax - ymin) * self.factor_zoom / 2
            xmin, xmax = c_x - dx, c_x + dx
            ymin, ymax = c_y - dy, c_y + dy

        self.terminado.emit()



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Boundary()
        self.ui.setupUi(self)
        
        ts.tema_oscuro(QtWidgets.QApplication.instance())

        self.mandelbrot = md.mostrar_fractal_opengl(self.ui)

        self.scene = QGraphicsScene(0, 0, 200, 200)




        self.scene.changed.connect(self.al_interactuar)

        self.inicializar_combos()
        self._conectar_mvc()
        self._conectar_botones()
        
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
        self.mandelbrot.coordenadas_raton_cambiadas.connect(self.actualizar_textos_coordenadas)
        self.mandelbrot.limites_zoom_cambiados.connect(self.actualizar_cajas_zoom)
        self.mandelbrot.parametros_matematicos_cambiados.connect(self.actualizar_cajas_matematicas)

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


    def sincronizar_hacia_motor(self, *args):
        try:
            xmin = Decimal(self.ui.xmin_entrada.text())
            xmax = Decimal(self.ui.xmax_entrada.text())
            ymin = Decimal(self.ui.ymin_entrada.text())
            ymax = Decimal(self.ui.ymax_entrada.text())
            zoom_in = Decimal(self.ui.zoom_in_factor_entrada.text())
            zoom_out = Decimal(self.ui.zoom_out_factor_entrada.text())
            real = 0.1
            imag = 0.1
            cmap = self.ui.cmap_comboBox.currentText()
            width = int(self.ui.width_entrada.text())
            height = int(self.ui.high_entrada.text())
            max_iter = int(self.ui.max_iter_entrada.text())
            tipo_calculo = self.ui.tipo_calculo_comboBox.currentText()
            tipo_fractal = self.ui.tipo_fractal_comboBox.currentText()
            clase_equiv = int(self.ui.clase_equiv_entrada.text())

            
            self.mandelbrot.actualizar_estado_desde_ui(
                cmap, width, height, max_iter, tipo_calculo, tipo_fractal, 
                zoom_in, zoom_out, clase_equiv, real, imag
            )
            self.mandelbrot.xmin, self.mandelbrot.xmax = xmin, xmax
            self.mandelbrot.ymin, self.mandelbrot.ymax = ymin, ymax

            self.mandelbrot.corregir_aspect_ratio()

            self.mandelbrot.update()
        except ValueError:
            pass 

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

        resoluciones = ["Actual", "Full HD (1920x1080)", "4K (3840x2160)"]
        res_elegida, ok_res = QInputDialog.getItem(self, "Video", "Seleccionar resolución:", resoluciones, 0, False)
        if not ok_res: return

        if "Full HD" in res_elegida: render_w, render_h = 1920, 1080
        elif "4K" in res_elegida: render_w, render_h = 3840, 2160
        else: render_w, render_h = self.mandelbrot.width, self.mandelbrot.height

        self.sincronizar_hacia_motor()
        
        self.progreso_ui = QProgressDialog("Renderizando video en 2do plano...", "Cancelar", 0, frames, self)
        self.progreso_ui.setWindowTitle("GPU Procesando")
        self.progreso_ui.setCancelButton(None) 
        self.progreso_ui.show()

        # Extraemos una foto del estado actual para mandársela al hilo secundario
        estado_actual = {
            'xmin': self.mandelbrot.xmin, 'xmax': self.mandelbrot.xmax,
            'ymin': self.mandelbrot.ymin, 'ymax': self.mandelbrot.ymax,
            'max_iter': self.mandelbrot.max_iter, 'formula': self.mandelbrot.formula,
            'tipo_calculo': self.mandelbrot.tipo_calculo, 'tipo_fractal': self.mandelbrot.tipo_fractal,
            'real': getattr(self.mandelbrot, 'real', 0.0), 'imag': getattr(self.mandelbrot, 'imag', 0.0),
            'clase_equiv': getattr(self.mandelbrot, 'clase_equiv', 128),
            'palette_index': self.mandelbrot.palette_index
        }

        # Lanzamos el proceso asíncrono
        self.worker = WorkerVideo(estado_actual, carpeta, frames, render_w, render_h, factor_zoom)
        self.worker.progreso.connect(self.progreso_ui.setValue)
        self.worker.terminado.connect(self._video_terminado)
        self.worker.start()

    def _video_terminado(self):
        self.progreso_ui.close()
        print("¡Secuencia completada!")