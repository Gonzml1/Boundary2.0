import numpy as np
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from core.modulo_de_calculo_fractales import calculos_mandelbrot
from gui.MandelbrotGUI import Ui_Boundary
from matplotlib import cm
from typing import Callable
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog
import matplotlib.pyplot as plt
from PyQt5.QtCore import QTimer
PALETTE_REGISTRY: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = []



def register_palette(palette_name: str) -> Callable[[Callable[[np.ndarray], np.ndarray]], Callable[[np.ndarray], np.ndarray]]:
    """
    Decorador que registra (nombre, función) en PALETTE_REGISTRY.
    La función _aún no_ está ligada a ninguna instancia.
    """
    def deco(fn: Callable[[np.ndarray], np.ndarray]):
        PALETTE_REGISTRY.append((palette_name, fn))
        return fn
    return deco

class MandelbrotWidget(QOpenGLWidget):
    def __init__(self,cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal,zoom_in, zoom_out, boundary=Ui_Boundary):
        super().__init__()
        self.cmap           =       cmap    
        self.xmin           =       xmin
        self.xmax           =       xmax
        self.ymin           =       ymin
        self.ymax           =       ymax
        self.width          =       width
        self.height         =       height
        self.max_iter       =       max_iter
        self.formula        =       formula
        self.tipo_calculo   =       tipo_calculo
        self.tipo_fractal   =       tipo_fractal
        self.ui             =       boundary
        self.zoom_in        =       zoom_in
        self.zoom_out       =       zoom_out
        self.zoom_factor    =       1.0
        self.dragging       =       False
        self.last_pos       =       None
        self.mandelbrot     =       calculos_mandelbrot(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter,self.formula, self.tipo_calculo, self.tipo_fractal)                               

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.ui.boton_hacer_fractal.clicked.connect(lambda : self.update())
        self.ui.slider_iteraciones.valueChanged.connect(self.update)
        self.actualizar_parametros()
        self.palettes = []
        self.clase_equiv = self.ui.clase_equiv_entrada.text()
        for name, fn in PALETTE_REGISTRY:
            bound_fn = fn.__get__(self, type(self))
            self.palettes.append((name, bound_fn))
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.palette_index = 0
        self.ui.boton_guardar.clicked.connect(lambda: self.guardar_imagen())
        self.linkeo_botones()
        self.ui.boton_resetear.clicked.connect(lambda: self.reset_view())
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self.ejecutar_alta_resolucion)
        self.is_preview_mode = False
        
        self.resolucion_adaptativa_activa = True

    def interaccion_rapida(self):
            if self.resolucion_adaptativa_activa:
                # Comportamiento normal: bajamos la calidad y arrancamos el reloj
                self.is_preview_mode = True
                self.render_timer.stop()
                self.update() 
                self.render_timer.start(500) 
            else:
                # Sistema desactivado: renderizamos todo al máximo nivel siempre
                self.is_preview_mode = False
                self.update()

    def ejecutar_alta_resolucion(self):
        """Restaura la calidad al máximo cuando el usuario deja de moverse."""
        self.is_preview_mode = False
        self.update()
  

    ######################
    # Paletas de colores #
    ######################
    
    @register_palette("Iteraciones variables (Bandas RGB variable)")
    def _paleta_bandas_rgb_equiv(self, norm: np.ndarray) -> np.ndarray:
        """
        Paleta RGB cíclica tipo bandas con clase_equiv franjas.
        - Divide [0,1] en `self.clase_equiv` franjas.
        - Dentro de cada franja: transición lineal R→G→B→R...
        """
        iters = np.uint32((norm * self.max_iter).clip(0, self.max_iter))
        mod = iters % self.clase_equiv
        pos = mod / self.clase_equiv * 3  # Escalamos a [0,3)

        r = np.where(pos < 1, pos, np.where(pos < 2, 2 - pos, 0))
        g = np.where(pos < 1, 0, np.where(pos < 2, pos - 1, 3 - pos))
        b = np.where(pos < 2, 0, pos - 2)

        rgb = np.stack([r.clip(0,1), g.clip(0,1), b.clip(0,1)], axis=-1)
        return (rgb * 255).astype(np.uint8)

    @register_palette("Iteracion varibales (YlGnBu variable)")
    def _paleta_ylgnbu(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'YlGnBu' de Matplotlib (secuencial amarillo→verde→azul).
        """
        iters = np.uint32((norm * self.max_iter).clip(0, self.max_iter))
        cycle = self.clase_equiv
        mod = iters % cycle
        cmap = cm.get_cmap('YlGnBu', cycle)
        lut = (cmap(np.arange(cycle))[:, :3] * 255).astype(np.uint8)
        return lut[mod]
    
    @register_palette("Iteraciones variables (Viridis variable)")
    def _pallete_iters_variable_viridis(self, norm: np.ndarray) -> np.ndarray:
        """
        - Reconstruye iter ∈ [0..max_iter] desde norm
        - Usa iter % 64 para indexar un LUT de viridis de self.clase_equiv colores
        """
        iters = np.uint32((norm * self.max_iter).clip(0, self.max_iter))
        cycle = self.clase_equiv
        mod = iters % cycle
        cmap = cm.get_cmap('viridis', cycle)
        lut = (cmap(np.arange(cycle))[:, :3] * 255).astype(np.uint8)
        return lut[mod]

    @register_palette("Iteraciones variables (Plasma variable)")
    def _pallete_iters_variable_plasma(self, norm: np.ndarray) -> np.ndarray:
        """
        - Reconstruye iter ∈ [0..max_iter] desde norm
        - Usa iter % 64 para indexar un LUT de plasma de self.clase_equiv colores
        """
        iters = np.uint32((norm * self.max_iter).clip(0, self.max_iter))
        cycle = self.clase_equiv
        cmap= cm.get_cmap('plasma', cycle)
        lut = (cmap(np.arange(cycle))[:, :3] * 255).astype(np.uint8)
        return lut[iters % cycle]

    @register_palette("Iteraciones variables (Grises)")
    def _pallete_iters_variable_grises(self, norm: np.ndarray) -> np.ndarray:
        """
        - Reconstruye iter ∈ [0..max_iter] desde norm
        - Usa iter % 64 para indexar un LUT de grises cíclico de self.clase_equiv colores
        """
        iters = np.uint32((norm * self.max_iter).clip(0, self.max_iter))
        cycle = self.clase_equiv
        mod = iters % cycle
        gray = np.uint8(((mod.astype(float) / (cycle - 1)) * 255).clip(0, 255))
        return np.dstack([gray, gray, gray])
    
    @register_palette("Iteraciones variables (Twilight Shifted)")
    def _paleta_iters_variable_twilight(self, norm: np.ndarray) -> np.ndarray:
        """
        - Reconstruye iter ∈ [0..max_iter] desde norm
        - Usa iter % 512 para indexar un LUT de twilight_shifted de 512 colores
        """
        iters = np.uint32((norm * self.max_iter).clip(0, self.max_iter))
        cycle = self.clase_equiv
        cmap = cm.get_cmap('twilight_shifted', cycle)
        lut = (cmap(np.arange(cycle))[:, :3] * 255).astype(np.uint8)
        return lut[iters % cycle]     


    def next_palette(self):
        """
        Incrementa palette_index y actualiza el widget.
        """
        self.palette_index = (self.palette_index + 1) % len(self.palettes)
        self.update()  # Fuerza un repaint (llamará de nuevo a paintGL)

    def previous_palette(self):
        """
        Decrementa palette_index y actualiza el widget.
        """
        self.palette_index = (self.palette_index - 1) % len(self.palettes)
        self.update()
        
    # Opcional: atajar una tecla para cambiar


    # ——— paintGL revisitado, usando el índice de paleta ———
    def guardar_imagen(self) -> None:
        ruta, _ = QFileDialog.getSaveFileName(
            None,
            "Guardar imagen",
            f"fractal_{self.xmin:.16f}_{self.xmax:.16f}_{self.ymin:.16f}_{self.ymax:.16f}_{self.tipo_fractal}_{self.max_iter}.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;Todos los archivos (*)"
        )
        if not ruta:
            return

        # 1) Generar los datos del fractal
        self.mandelbrot.actualizar_fractal(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            4*self.width, 4*self.height,
            self.max_iter,
            self.formula, self.tipo_calculo,
            self.tipo_fractal,
            self.real, self.imag
        )
        data = self.mandelbrot.calcular_fractal()  # np.ndarray de enteros [0..max_iter]

        # 2) Normalizar a [0,1]
        norm = data.astype(float) / self.max_iter

        # 3) Elegir la paleta actual
        name, func = self.palettes[self.palette_index]

        # 4) Aplicar la paleta → rgb uint8 (H,W,3)
        rgb = func(norm)

        # 5) Invertir verticalmente si lo hacés en paintGL

        # 6) Guardar la imagen directamente como RGB (sin pasar cmap)
        # plt.imsave admite uint8 RGB si no le pasás cmap
        plt.imsave(ruta, rgb)
        print(f"Imagen guardada en: {ruta}")

    def linkeo_botones(self):
        self.ui.boton_dividir.clicked.connect(lambda : self.dividir())
        self.ui.boton_duplicar.clicked.connect(lambda : self.duplicar())
        self.ui.boton_dividir_clase_equiv.clicked.connect(lambda : self.dividir_clase_equiv())
        self.ui.boton_duplicar_clase_equiv.clicked.connect(lambda : self.duplicar_clase_equiv())


        self.ui.slider_iteraciones.valueChanged.connect(lambda value: self.ui.max_iter_entrada.setText(str(value)))
    
    def duplicar(self):
        self.ui.max_iter_entrada.setText(str(int(int(self.ui.max_iter_entrada.text())*2)))
        self.update()
        
    def dividir(self):
        self.ui.max_iter_entrada.setText(str(int(int(self.ui.max_iter_entrada.text())/2)))
        self.update()
    
    def dividir_clase_equiv(self=Ui_Boundary()):
        self.ui.clase_equiv_entrada.setText(str(int(int(self.ui.clase_equiv_entrada.text())/2)))
        self.update()
    
    def duplicar_clase_equiv(self=Ui_Boundary()):
        self.ui.clase_equiv_entrada.setText(str(int(int(self.ui.clase_equiv_entrada.text())*2)))
        self.update()
    
    def reset_view(self):
        """
        Resetea la vista a los valores iniciales.
        """
        self.xmin = -2.0
        self.xmax = 1.2
        self.ymin = -0.9
        self.ymax = 0.9
        self.max_iter = 256
        self.width = 1000
        self.height = 600
        self.zoom_factor = 1.0
        self.clase_equiv = 128
        self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter, self.clase_equiv)
        self.update()

    def mostrar_parametros(self, xmin, xmax, ymin, ymax, width, height, max_iter, clase_equiv):
        """
        Muestra los parámetros del fractal en la UI.
        """
        self.ui.xmin_entrada.setText(f"{xmin}")
        self.ui.xmax_entrada.setText(f"{xmax}")
        self.ui.ymin_entrada.setText(f"{ymin}")
        self.ui.ymax_entrada.setText(f"{ymax}")
        self.ui.width_entrada.setText(f"{width}")
        self.ui.high_entrada.setText(f"{height}")
        self.ui.max_iter_entrada.setText(f"{max_iter}")
        self.ui.clase_equiv_entrada.setText(f"{clase_equiv}")


    def actualizar_parametros(self) -> None:
        """
        Actualiza los parámetros del fractal según los valores de la UI.
        """
        
        self.cmap           =   str(self.ui.cmap_comboBox.currentText())
        self.zoom_in        =   float(self.ui.zoom_in_factor_entrada.text())
        self.zoom_out       =   float(self.ui.zoom_out_factor_entrada.text()) 
        self.width          =   int(self.ui.width_entrada.text())
        self.height         =   int(self.ui.high_entrada.text())
        self.max_iter       =   int(self.ui.max_iter_entrada.text())
        self.tipo_calculo   =   str(self.ui.tipo_calculo_comboBox.currentText())
        self.tipo_fractal   =   str(self.ui.tipo_fractal_comboBox.currentText())
        self.formula        =   str(self.ui.formula_entrada.text())
        self.real           =   float(self.ui.real_julia_entrada.text())
        self.imag           =   float(self.ui.im_julia_entrada.text())
        self.clase_equiv    =   int(self.ui.clase_equiv_entrada.text())
        self.mandelbrot.actualizar_fractal(
                self.xmin, self.xmax,
                self.ymin, self.ymax,
                self.width, self.height,
                self.max_iter,
                self.formula, self.tipo_calculo,
                self.tipo_fractal,
                self.real, self.imag
            )


    def initializeGL(self):
        glClearColor(0, 0, 0, 1)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    def mouseMoveEvent(self, event):
        x = event.x()
        y = event.y()
        real, imag = self.pixel_a_complejo(x, y)
        self.ui.label_coordenadas.setText(f"Re: {real:.16f}, Im: {imag:.16f}")

        if self.dragging:
            x0, y0 = self.last_pos.x(), self.last_pos.y()
            c0_real, c0_imag = self.pixel_a_complejo(x0, y0)
            dx = c0_real - real
            dy = c0_imag - imag
            self.xmin += dx
            self.xmax += dx
            self.ymin += dy
            self.ymax += dy
            self.last_pos = event.pos()
            self.actualizar_parametros()
            self.mostrar_parametros(
                self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter, self.clase_equiv
            )
            
            self.interaccion_rapida()
            
    
    def pixel_a_complejo(self, x, y):
        real = self.xmin + (x / self.width) * (self.xmax - self.xmin)
        imag = self.ymin + (y / self.height) * (self.ymax - self.ymin)
        return real, imag
    
    
    def paintGL(self):
            if str(self.ui.generador_comboBox.currentText()) == "Sucesion":
                glClear(GL_COLOR_BUFFER_BIT)
                glLoadIdentity()

                # 1) Actualizar parámetros base desde la UI
                self.actualizar_parametros()

                # --- SISTEMA DE RESOLUCIÓN ADAPTATIVA ---
                factor_escala = 1.0
                if getattr(self, 'is_preview_mode', False) or getattr(self, 'modo_baja_res_fijo', False):
                    factor_escala = 4.0  # Calcula 16 veces menos píxeles para volar a máximos FPS
                    self.mandelbrot.width = int(self.width / factor_escala)
                    self.mandelbrot.height = int(self.height / factor_escala)

                # 2) Definir malla dinámica según la resolución calculada
                x = np.linspace(self.xmin, self.xmax, self.mandelbrot.width)
                y = np.linspace(self.ymin, self.ymax, self.mandelbrot.height)
                X, Y = np.meshgrid(x, y)

                    

                self.mandelbrot.Z = X + 1j * Y

                # 4) Calcular el fractal
                try:
                    data = self.mandelbrot.calcular_fractal()
                except RuntimeError as e:
                    print(e)
                    # Aquí puedes pintar la pantalla de un color sólido para indicar error
                    glClearColor(1.0, 0.0, 0.0, 1.0) 
                    glClear(GL_COLOR_BUFFER_BIT)
                    return

                # 5) Normalizar y colorear (usando la iteración adaptativa)
                norm = data / self.mandelbrot.max_iter 
                name, func = self.palettes[self.palette_index]
                rgb = func(norm)[::-1, :, :]

                # 6) Dibujar estirando los píxeles mágicamente con OpenGL
                glPixelZoom(factor_escala, factor_escala)
                glDrawPixels(self.mandelbrot.width, self.mandelbrot.height, GL_RGB, GL_UNSIGNED_BYTE, rgb)
                glPixelZoom(1.0, 1.0) # Restaurar la escala para el próximo frame

            
    def resizeGL(self, w, h):
        self.width = w
        self.height = h
        glViewport(0, 0, w, h)
        self.update()
  
    def wheelEvent(self, event):
        zoom = 0.9 if event.angleDelta().y() > 0 else 1.1
        cx = (self.xmin + self.xmax) / 2
        cy = (self.ymin + self.ymax) / 2
        dx = (self.xmax - self.xmin) * zoom / 2
        dy = (self.ymax - self.ymin) * zoom / 2
        self.xmin, self.xmax = cx - dx, cx + dx
        self.ymin, self.ymax = cy - dy, cy + dy
        
        self.actualizar_parametros()
        self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter, self.clase_equiv)
        self.interaccion_rapida()

    def mousePressEvent(self, event):
        
        if event.button() == Qt.LeftButton:

            x_pixel = event.x()
            y_pixel = event.y()  

            c_x = self.xmin + (x_pixel / self.width) * (self.xmax - self.xmin)
            c_y = self.ymin + (y_pixel / self.height) * (self.ymax - self.ymin)

            self.xmin = c_x - (c_x - self.xmin) * self.zoom_in
            self.xmax = c_x + (self.xmax - c_x) * self.zoom_in
            self.ymin = c_y - (c_y - self.ymin) * self.zoom_in
            self.ymax = c_y + (self.ymax - c_y) * self.zoom_in
            self.actualizar_parametros()
            self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter, self.clase_equiv)
            self.interaccion_rapida()
            
        elif event.button() == Qt.RightButton:

            x_pixel = event.x()
            y_pixel = event.y()  

            c_x = self.xmin + (x_pixel / self.width) * (self.xmax - self.xmin)
            c_y = self.ymin + (y_pixel / self.height) * (self.ymax - self.ymin)

            self.xmin = c_x - (c_x - self.xmin) * self.zoom_out
            self.xmax = c_x + (self.xmax - c_x) * self.zoom_out
            self.ymin = c_y - (c_y - self.ymin) * self.zoom_out
            self.ymax = c_y + (self.ymax - c_y) * self.zoom_out
            
            self.actualizar_parametros()
            self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter, self.clase_equiv)
            self.interaccion_rapida()

        elif event.button() == Qt.MiddleButton:
            self.dragging = True
            self.last_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.dragging = False
    
    def keyPressEvent(self, event):
        
        if str(self.ui.generador_comboBox.currentText()) == "Sucesion":
            move = 0.05
            dx = (self.xmax - self.xmin) * move
            dy = (self.ymax - self.ymin) * move
            
            if event.key() in (Qt.Key_Left, Qt.Key_A):
                self.xmin -= dx
                self.xmax -= dx
                self.interaccion_rapida()
                
            elif event.key() in (Qt.Key_Right, Qt.Key_D):
                self.xmin += dx
                self.xmax += dx
                self.interaccion_rapida()
                
            elif event.key() in (Qt.Key_Up, Qt.Key_W):
                self.ymin -= dy
                self.ymax -= dy
                self.interaccion_rapida()
                
            elif event.key() in (Qt.Key_Down, Qt.Key_S):
                self.ymin += dy
                self.ymax += dy
                self.interaccion_rapida()

            elif event.key() == Qt.Key_Plus:
                # Calcular el punto central actual
                c_x = (self.xmin + self.xmax) / 2
                c_y = (self.ymin + self.ymax) / 2

                # Ajustar los límites en torno al centro con el factor de zoom
                dx = (self.xmax - self.xmin) * self.zoom_in / 2
                dy = (self.ymax - self.ymin) * self.zoom_in / 2
                self.xmin, self.xmax = c_x - dx, c_x + dx
                self.ymin, self.ymax = c_y - dy, c_y + dy

                # Refrescar parámetros y repintar
                self.actualizar_parametros()
                self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax,
                                        self.width, self.height, self.max_iter, self.clase_equiv)
                self.interaccion_rapida()
                
            elif event.key() == Qt.Key_Minus:
                c_x = (self.xmin + self.xmax) / 2
                c_y = (self.ymin + self.ymax) / 2

                # Ajustar los límites en torno al centro con el factor de zoom
                dx = (self.xmax - self.xmin) * self.zoom_out / 2
                dy = (self.ymax - self.ymin) * self.zoom_out / 2
                self.xmin, self.xmax = c_x - dx, c_x + dx
                self.ymin, self.ymax = c_y - dy, c_y + dy

                # Refrescar parámetros y repintar
                self.actualizar_parametros()
                self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax,
                                        self.width, self.height, self.max_iter, self.clase_equiv)
                self.interaccion_rapida()
                
            elif event.key() == Qt.Key_P:    
                self.next_palette()
            
            elif event.key() == Qt.Key_O:
                self.previous_palette()
                
            elif event.key() == Qt.Key_R:
                self.reset_view()

            elif event.key() == Qt.Key_G:
                self.duplicar()
            
            elif event.key() == Qt.Key_H:
                self.dividir()

            elif event.key() == Qt.Key_B:
                self.duplicar_clase_equiv()

            elif event.key() == Qt.Key_N:
                self.dividir_clase_equiv()
            if event.key() == Qt.Key_F:
                self.fluido_activo = not self.fluido_activo
                # para que se repinte inmediatamente:
                self.update()
            elif event.key() == Qt.Key_Q:
                self.resolucion_adaptativa_activa = not self.resolucion_adaptativa_activa
                
                estado = "ACTIVADA (Baja calidad al moverse)" if self.resolucion_adaptativa_activa else "DESACTIVADA (Siempre máxima calidad)"
                print(f"Resolución adaptativa: {estado}")
                
                # Si acabamos de apagar el sistema, frenamos el reloj y forzamos alta resolución
                if not self.resolucion_adaptativa_activa:
                    self.is_preview_mode = False
                    self.render_timer.stop()
                    self.update()
            