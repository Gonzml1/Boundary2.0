import numpy as np
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from OpenGL.GL import *
from core.modulo_de_calculo_fractales import calculos_mandelbrot
from core.paletas import PALETTE_REGISTRY

class MandelbrotWidget(QOpenGLWidget):
    # --- SEÑALES MVC ---
    coordenadas_raton_cambiadas = pyqtSignal(float, float)
    limites_zoom_cambiados = pyqtSignal(float, float, float, float)
    parametros_matematicos_cambiados = pyqtSignal(int, int)

    def __init__(self, cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, zoom_in, zoom_out, clase_equiv, real, imag):
        super().__init__()
        self.cmap = cmap    
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.formula = formula
        self.tipo_calculo = tipo_calculo
        self.tipo_fractal = tipo_fractal
        self.zoom_in = zoom_in
        self.zoom_out = zoom_out
        self.clase_equiv = clase_equiv
        self.real = real
        self.imag = imag

        self.zoom_factor = 1.0
        self.dragging = False
        self.last_pos = None
        self.mandelbrot = calculos_mandelbrot(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter, self.formula, self.tipo_calculo, self.tipo_fractal)                               

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        
        self.palettes = PALETTE_REGISTRY 
        self.palette_index = 0

        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self.ejecutar_alta_resolucion)
        self.is_preview_mode = False
        self.resolucion_adaptativa_activa = True

    def actualizar_estado_desde_ui(self, cmap, width, height, max_iter, formula, tipo_calculo, tipo_fractal, zoom_in, zoom_out, clase_equiv, real, imag):
        self.cmap = cmap
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.formula = formula
        self.tipo_calculo = tipo_calculo
        self.tipo_fractal = tipo_fractal
        self.zoom_in = zoom_in
        self.zoom_out = zoom_out
        self.clase_equiv = clase_equiv
        self.real = real
        self.imag = imag

    def interaccion_rapida(self):
        if self.resolucion_adaptativa_activa:
            self.is_preview_mode = True
            self.render_timer.stop()
            self.update() 
            self.render_timer.start(500) 
        else:
            self.is_preview_mode = False
            self.update()

    def ejecutar_alta_resolucion(self):
        self.is_preview_mode = False
        self.update()

    def next_palette(self):
        self.palette_index = (self.palette_index + 1) % len(self.palettes)
        self.update()

    def previous_palette(self):
        self.palette_index = (self.palette_index - 1) % len(self.palettes)
        self.update()

    def obtener_frame_rgb(self, render_w, render_h):
        self.mandelbrot.actualizar_fractal(self.xmin, self.xmax, self.ymin, self.ymax, render_w, render_h, self.max_iter, self.formula, self.tipo_calculo, self.tipo_fractal, self.real, self.imag)
        data = self.mandelbrot.calcular_fractal()
        norm = data.astype(float) / self.max_iter
        _, func = self.palettes[self.palette_index]
        return func(norm, self.max_iter, self.clase_equiv, getattr(self, 't_actual', 0.0), getattr(self, 'thickness', 0.0))

    def initializeGL(self):
        glClearColor(0, 0, 0, 1)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    def pixel_a_complejo(self, x, y):
        real = self.xmin + (x / self.width) * (self.xmax - self.xmin)
        imag = self.ymin + (1.0 - (y / self.height)) * (self.ymax - self.ymin) # Invertimos Y
        return real, imag

    def corregir_aspect_ratio(self):
        if self.height == 0: return
        aspect_ratio = self.width / self.height
        centro_y = (self.ymin + self.ymax) / 2
        ancho_actual = self.xmax - self.xmin
        alto_corregido = ancho_actual / aspect_ratio
        
        self.ymin = centro_y - alto_corregido / 2
        self.ymax = centro_y + alto_corregido / 2
        self.limites_zoom_cambiados.emit(self.xmin, self.xmax, self.ymin, self.ymax)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()

        factor_escala = 4.0 if (self.is_preview_mode or getattr(self, 'modo_baja_res_fijo', False)) else 1.0
        render_w = int(self.width / factor_escala)
        render_h = int(self.height / factor_escala)

        self.mandelbrot.actualizar_fractal(self.xmin, self.xmax, self.ymin, self.ymax, render_w, render_h, self.max_iter, self.formula, self.tipo_calculo, self.tipo_fractal, self.real, self.imag)
        x = np.linspace(self.xmin, self.xmax, render_w)
        y = np.linspace(self.ymin, self.ymax, render_h)
        X, Y = np.meshgrid(x, y)
        self.mandelbrot.Z = X + 1j * Y

        try:
            data = self.mandelbrot.calcular_fractal()
        except RuntimeError as e:
            print(e)
            glClearColor(1.0, 0.0, 0.0, 1.0) 
            glClear(GL_COLOR_BUFFER_BIT)
            return

        norm = data / self.max_iter 
        _, func = self.palettes[self.palette_index]
        rgb = func(norm, self.max_iter, self.clase_equiv, getattr(self, 't_actual', 0.0), getattr(self, 'thickness', 0.0))

        glPixelZoom(factor_escala, factor_escala)
        glDrawPixels(render_w, render_h, GL_RGB, GL_UNSIGNED_BYTE, rgb)
        glPixelZoom(1.0, 1.0)

    def resizeGL(self, w, h):
        self.width = w
        self.height = h
        glViewport(0, 0, w, h)
        self.corregir_aspect_ratio()
        self.update()

    def wheelEvent(self, event):
        zoom = 0.9 if event.angleDelta().y() > 0 else 1.1
        self._aplicar_zoom(zoom)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._centrar_y_zoom(event.x(), self.height - event.y(), self.zoom_in)
        elif event.button() == Qt.RightButton:
            self._centrar_y_zoom(event.x(), self.height - event.y(), self.zoom_out)
        elif event.button() == Qt.MiddleButton:
            self.dragging = True
            self.last_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.dragging = False

    def mouseMoveEvent(self, event):
        x, y = event.x(), event.y()
        real, imag = self.pixel_a_complejo(x, y)
        self.coordenadas_raton_cambiadas.emit(real, imag) 

        if self.dragging:
            c0_real, c0_imag = self.pixel_a_complejo(self.last_pos.x(), self.last_pos.y())
            dx = c0_real - real
            dy = c0_imag - imag
            self.xmin += dx
            self.xmax += dx
            self.ymin += dy
            self.ymax += dy
            self.last_pos = event.pos()
            self.limites_zoom_cambiados.emit(self.xmin, self.xmax, self.ymin, self.ymax)
            self.interaccion_rapida()

    def _aplicar_zoom(self, factor):
        c_x, c_y = (self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2
        dx = (self.xmax - self.xmin) * factor / 2
        dy = (self.ymax - self.ymin) * factor / 2
        self.xmin, self.xmax = c_x - dx, c_x + dx
        self.ymin, self.ymax = c_y - dy, c_y + dy
        self.limites_zoom_cambiados.emit(self.xmin, self.xmax, self.ymin, self.ymax)
        self.interaccion_rapida()

    def _centrar_y_zoom(self, px, py, factor):
        # py ya viene invertido desde mousePressEvent
        c_x = self.xmin + (px / self.width) * (self.xmax - self.xmin)
        c_y = self.ymin + (py / self.height) * (self.ymax - self.ymin)
        self.xmin = c_x - (c_x - self.xmin) * factor
        self.xmax = c_x + (self.xmax - c_x) * factor
        self.ymin = c_y - (c_y - self.ymin) * factor
        self.ymax = c_y + (self.ymax - c_y) * factor
        self.limites_zoom_cambiados.emit(self.xmin, self.xmax, self.ymin, self.ymax)
        self.interaccion_rapida()

    def keyPressEvent(self, event):
        move = 0.05
        dx, dy = (self.xmax - self.xmin) * move, (self.ymax - self.ymin) * move
        
        if event.key() in (Qt.Key_Left, Qt.Key_A): self.xmin, self.xmax = self.xmin - dx, self.xmax - dx
        elif event.key() in (Qt.Key_Right, Qt.Key_D): self.xmin, self.xmax = self.xmin + dx, self.xmax + dx
        elif event.key() in (Qt.Key_Up, Qt.Key_W): self.ymin, self.ymax = self.ymin + dy, self.ymax + dy
        elif event.key() in (Qt.Key_Down, Qt.Key_S): self.ymin, self.ymax = self.ymin - dy, self.ymax - dy
        elif event.key() == Qt.Key_Plus: self._aplicar_zoom(self.zoom_in)
        elif event.key() == Qt.Key_Minus: self._aplicar_zoom(self.zoom_out)
        elif event.key() == Qt.Key_P: self.next_palette()
        elif event.key() == Qt.Key_O: self.previous_palette()
        elif event.key() == Qt.Key_R: 
            self.xmin, self.xmax, self.ymin, self.ymax = -2.0, 1.2, -0.9, 0.9
            self.max_iter, self.clase_equiv = 256, 128
            self.corregir_aspect_ratio()
            self.parametros_matematicos_cambiados.emit(self.max_iter, self.clase_equiv)
        elif event.key() == Qt.Key_G: 
            self.max_iter = int(self.max_iter * 2)
            self.parametros_matematicos_cambiados.emit(self.max_iter, self.clase_equiv)
        elif event.key() == Qt.Key_H: 
            self.max_iter = max(1, int(self.max_iter / 2))
            self.parametros_matematicos_cambiados.emit(self.max_iter, self.clase_equiv)
        elif event.key() == Qt.Key_B: 
            self.clase_equiv = int(self.clase_equiv * 2)
            self.parametros_matematicos_cambiados.emit(self.max_iter, self.clase_equiv)
        elif event.key() == Qt.Key_N: 
            self.clase_equiv = max(1, int(self.clase_equiv / 2))
            self.parametros_matematicos_cambiados.emit(self.max_iter, self.clase_equiv)
        elif event.key() == Qt.Key_Q:
            self.resolucion_adaptativa_activa = not self.resolucion_adaptativa_activa
            if not self.resolucion_adaptativa_activa:
                self.is_preview_mode = False
                self.render_timer.stop()

        if event.key() in (Qt.Key_Left, Qt.Key_A, Qt.Key_Right, Qt.Key_D, Qt.Key_Up, Qt.Key_W, Qt.Key_Down, Qt.Key_S):
            self.limites_zoom_cambiados.emit(self.xmin, self.xmax, self.ymin, self.ymax)

        self.interaccion_rapida()