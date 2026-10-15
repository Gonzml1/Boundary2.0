import logging
import threading
from dataclasses import replace
from decimal import Decimal

from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from OpenGL.GL import *

from core.modulo_de_calculo_fractales import calculos_mandelbrot
from core.paletas import PALETTE_REGISTRY
from core.render import renderizar, PedidoRender, OpcionesColor

log = logging.getLogger(__name__)

# Vista inicial
VISTA_INICIAL = (Decimal('-2.0'), Decimal('1.2'), Decimal('-0.9'), Decimal('0.9'))

# Calidad progresiva: cada pedido se renderiza primero chico y después a tamaño completo.
# Divisor de resolución de cada nivel (el último es la calidad final, con antialiasing si está activo).
NIVELES_CALIDAD = (8, 2, 1)
NIVEL_FINAL = len(NIVELES_CALIDAD) - 1


class HiloRender(QThread):
    """Renderiza en segundo plano. Atiende siempre el pedido más reciente: si llegan varios
    mientras está ocupado, los viejos se descartan."""
    listo = pyqtSignal(object, object)    # (PedidoPantalla, ImagenRender)
    fallo = pyqtSignal(object, str)       # (PedidoPantalla, mensaje)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._candado = threading.Lock()
        self._hay_pedido = threading.Event()
        self._pendiente = None
        self._salir = False

    def pedir(self, pedido_pantalla) -> None:
        with self._candado:
            self._pendiente = pedido_pantalla
        self._hay_pedido.set()

    def detener(self) -> None:
        self._salir = True
        self._hay_pedido.set()
        self.wait()

    def run(self):
        # Motor propio del hilo: el motor guarda estado (cachés de órbita) y no se comparte
        motor = calculos_mandelbrot(*VISTA_INICIAL, 1, 1, 1, "", "")
        while True:
            self._hay_pedido.wait()
            with self._candado:
                pp, self._pendiente = self._pendiente, None
                self._hay_pedido.clear()
            if self._salir:
                return
            if pp is None:
                continue
            try:
                self.listo.emit(pp, renderizar(motor, pp.pedido))
            except Exception as e:
                log.exception("Fallo calculando '%s / %s'", pp.pedido.tipo_fractal, pp.pedido.tipo_calculo)
                self.fallo.emit(pp, str(e))


class PedidoPantalla:
    """Un pedido de la pantalla: la versión de la vista que lo originó y su nivel de calidad."""
    __slots__ = ("version", "nivel", "pedido")

    def __init__(self, version: int, nivel: int, pedido: PedidoRender):
        self.version, self.nivel, self.pedido = version, nivel, pedido


class MandelbrotWidget(QOpenGLWidget):
    # --- SEÑALES MVC ---
    coordenadas_raton_cambiadas = pyqtSignal(float, float)
    # Los límites viajan como texto para no perder precisión en zooms profundos
    limites_zoom_cambiados = pyqtSignal(str, str, str, str)
    parametros_matematicos_cambiados = pyqtSignal(int, int)    # max_iter, clase_equiv
    paleta_cambiada = pyqtSignal(int)
    # segundos, opciones aplicadas, ancho, alto, ¿es la calidad final?
    render_terminado = pyqtSignal(float, object, int, int, bool)
    render_fallido = pyqtSignal(str)
    modo_completo_cambiado = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.xmin, self.xmax, self.ymin, self.ymax = VISTA_INICIAL
        self.max_iter = 256
        self.tipo_calculo = ""
        self.tipo_fractal = ""
        self.real = Decimal('-0.8')
        self.imag = Decimal('0.156')
        self.palette_index = 0
        self.clase_equiv = 128
        self.opciones = OpcionesColor()
        self.zoom_in = Decimal('0.5')
        self.zoom_out = Decimal('2.0')
        # Modo foto completa (tecla Q): siempre se calcula la vista entera a resolución completa
        # y no se muestra nada nuevo hasta que está lista (sin versiones previas ni reproyección)
        self.modo_completo = False

        self.dragging = False
        self.last_pos = None

        # Versión de la vista: sube con cada cambio. Solo los resultados de la versión actual
        # disparan el siguiente nivel de calidad.
        self._version = 0
        # Última imagen recibida y la vista que representa (para reproyectarla mientras llega la nueva)
        self._textura = None
        self._imagen_pendiente = None
        self._rect_imagen = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._hilo = HiloRender(self)
        self._hilo.listo.connect(self._al_terminar_render)
        self._hilo.fallo.connect(self._al_fallar_render)
        self._hilo.start()

    # ============================================================
    #                    ESTADO Y PEDIDOS
    # ============================================================

    def configurar(self, **parametros) -> None:
        """Cambia parámetros que no son la vista (max_iter, tipo_calculo, opciones, ...) y re-renderiza."""
        for nombre, valor in parametros.items():
            if not hasattr(self, nombre):
                raise AttributeError(nombre)
            setattr(self, nombre, valor)
        self.solicitar_render()

    def fijar_vista(self, xmin: Decimal, xmax: Decimal, ymin: Decimal, ymax: Decimal) -> None:
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.corregir_aspect_ratio()
        self.solicitar_render()

    def resetear(self) -> None:
        self.max_iter, self.clase_equiv = 256, 128
        self.parametros_matematicos_cambiados.emit(self.max_iter, self.clase_equiv)
        self.fijar_vista(*VISTA_INICIAL)

    def tamano_render(self) -> tuple[int, int]:
        """Tamaño de la vista en píxeles físicos (considera el escalado de pantalla de Windows)."""
        escala = self.devicePixelRatioF()
        return max(1, round(self.width() * escala)), max(1, round(self.height() * escala))

    def pedido_actual(self, ancho: int | None = None, alto: int | None = None, **cambios) -> PedidoRender:
        """El pedido que representa lo que se ve, opcionalmente con otro tamaño (para exportar)."""
        if ancho is None:
            ancho, alto = self.tamano_render()
        return PedidoRender(
            self.xmin, self.xmax, self.ymin, self.ymax, ancho, alto, self.max_iter,
            self.tipo_fractal, self.tipo_calculo, float(self.real), float(self.imag),
            self.palette_index, self.clase_equiv, self.opciones, **cambios
        )

    def solicitar_render(self) -> None:
        """Pide un render nuevo de la vista actual, empezando por la calidad más baja."""
        if not self.tipo_fractal or not self.tipo_calculo:
            return
        self._version += 1
        self._pedir_nivel(NIVEL_FINAL if self.modo_completo else 0)
        self.update()      # reproyecta la imagen vieja mientras tanto

    # Compatibilidad con el nombre anterior
    interaccion_rapida = solicitar_render

    def _pedir_nivel(self, nivel: int) -> None:
        ancho, alto = self.tamano_render()
        divisor = NIVELES_CALIDAD[nivel]
        pedido = self.pedido_actual(max(1, ancho // divisor), max(1, alto // divisor))
        if nivel != NIVEL_FINAL:
            # El antialiasing solo a resolución completa: en las previas sería trabajo tirado
            pedido = replace(pedido, opciones=replace(pedido.opciones, antialiasing=False))
        self._hilo.pedir(PedidoPantalla(self._version, nivel, pedido))

    def _al_terminar_render(self, pp: PedidoPantalla, imagen) -> None:
        p = pp.pedido
        self._imagen_pendiente = imagen.rgb
        self._rect_imagen = (p.xmin, p.xmax, p.ymin, p.ymax)
        final = pp.nivel == NIVEL_FINAL
        self.render_terminado.emit(imagen.segundos, imagen.aplicadas, p.ancho, p.alto, final)
        if pp.version == self._version and not final:
            self._pedir_nivel(pp.nivel + 1)
        self.update()

    def _al_fallar_render(self, pp: PedidoPantalla, mensaje: str) -> None:
        if pp.version == self._version:
            self.render_fallido.emit(mensaje)

    def detener(self) -> None:
        self._hilo.detener()

    # ============================================================
    #                    PALETAS Y CALIDAD
    # ============================================================

    def toggle_modo_completo(self):
        """Activa/desactiva el modo foto completa (tecla Q)."""
        self.modo_completo = not self.modo_completo
        self.modo_completo_cambiado.emit(self.modo_completo)
        self.solicitar_render()

    def next_palette(self):
        self._cambiar_paleta(1)

    def previous_palette(self):
        self._cambiar_paleta(-1)

    def _cambiar_paleta(self, paso: int):
        self.palette_index = (self.palette_index + paso) % len(PALETTE_REGISTRY)
        self.paleta_cambiada.emit(self.palette_index)
        self.solicitar_render()

    # ============================================================
    #                    COORDENADAS
    # ============================================================

    def pixel_a_complejo(self, x, y):
        prop_x = Decimal(x) / Decimal(max(1, self.width()))
        prop_y = Decimal('1') - (Decimal(y) / Decimal(max(1, self.height())))
        real = self.xmin + prop_x * (self.xmax - self.xmin)
        imag = self.ymin + prop_y * (self.ymax - self.ymin)
        return real, imag

    def corregir_aspect_ratio(self):
        if self.height() == 0:
            return
        aspect_ratio = Decimal(self.width()) / Decimal(self.height())
        centro_y = (self.ymin + self.ymax) / Decimal('2')
        ancho_actual = self.xmax - self.xmin
        alto_corregido = ancho_actual / aspect_ratio
        self.ymin = centro_y - alto_corregido / Decimal('2')
        self.ymax = centro_y + alto_corregido / Decimal('2')
        self._emitir_limites()

    def _emitir_limites(self):
        self.limites_zoom_cambiados.emit(*self.limites_como_texto())

    def limites_como_texto(self) -> tuple[str, str, str, str]:
        # 20 dígitos más allá del tamaño de la vista: suficiente para no perder nada al releerlos
        span = min(abs(self.xmax - self.xmin), abs(self.ymax - self.ymin))
        decimales = max(0, 20 - span.adjusted()) if span else 20
        def formatear(valor: Decimal) -> str:
            texto = format(valor, f'.{decimales}f')
            return texto.rstrip('0').rstrip('.') if '.' in texto else texto
        return tuple(formatear(v) for v in (self.xmin, self.xmax, self.ymin, self.ymax))

    # ============================================================
    #                    OPENGL
    # ============================================================

    def initializeGL(self):
        glClearColor(0x1c / 255, 0x1c / 255, 0x1f / 255, 1)  # mismo fondo que el tema
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        self._textura = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._textura)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        if self._imagen_pendiente is not None:
            rgb = self._imagen_pendiente
            self._imagen_pendiente = None
            glBindTexture(GL_TEXTURE_2D, self._textura)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.shape[1], rgb.shape[0], 0,
                         GL_RGB, GL_UNSIGNED_BYTE, rgb)
        if self._rect_imagen is None:
            return

        # Dónde cae la última imagen dentro de la vista actual (0..1 en cada eje). Si la vista
        # cambió desde que se pidió, la imagen se ve corrida/escalada hasta que llegue la nueva.
        ixmin, ixmax, iymin, iymax = self._rect_imagen
        ancho, alto = self.xmax - self.xmin, self.ymax - self.ymin
        if ancho == 0 or alto == 0:
            return
        if self.modo_completo:
            # Sin reproyectar: la foto anterior queda entera y quieta hasta que llega la nueva
            x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0
        else:
            x0, x1 = float((ixmin - self.xmin) / ancho), float((ixmax - self.xmin) / ancho)
            y0, y1 = float((iymin - self.ymin) / alto), float((iymax - self.ymin) / alto)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._textura)
        glColor3f(1, 1, 1)
        # La fila 0 de la imagen es ymin: va abajo
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x0, y0)
        glTexCoord2f(1, 0); glVertex2f(x1, y0)
        glTexCoord2f(1, 1); glVertex2f(x1, y1)
        glTexCoord2f(0, 1); glVertex2f(x0, y1)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def resizeGL(self, w, h):
        self.corregir_aspect_ratio()
        self.solicitar_render()

    # ==================== EVENTOS DE MOUSE Y TECLADO ====================

    def wheelEvent(self, event):
        zoom = 0.9 if event.angleDelta().y() > 0 else 1.1
        self._aplicar_zoom(zoom)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._centrar_y_zoom(event.x(), self.height() - event.y(), self.zoom_in)
        elif event.button() == Qt.RightButton:
            self._centrar_y_zoom(event.x(), self.height() - event.y(), self.zoom_out)
        elif event.button() == Qt.MiddleButton:
            self.dragging = True
            self.last_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.dragging = False

    def mouseMoveEvent(self, event):
        x, y = event.x(), event.y()
        real, imag = self.pixel_a_complejo(x, y)
        self.coordenadas_raton_cambiadas.emit(float(real), float(imag))

        if self.dragging:
            c0_real, c0_imag = self.pixel_a_complejo(self.last_pos.x(), self.last_pos.y())
            dx = c0_real - real
            dy = c0_imag - imag
            self.xmin += dx
            self.xmax += dx
            self.ymin += dy
            self.ymax += dy
            self.last_pos = event.pos()
            self._emitir_limites()
            self.solicitar_render()

    def _aplicar_zoom(self, factor):
        factor_dec = Decimal(str(factor))
        c_x, c_y = (self.xmin + self.xmax) / Decimal('2'), (self.ymin + self.ymax) / Decimal('2')
        dx = (self.xmax - self.xmin) * factor_dec / Decimal('2')
        dy = (self.ymax - self.ymin) * factor_dec / Decimal('2')
        self.xmin, self.xmax = c_x - dx, c_x + dx
        self.ymin, self.ymax = c_y - dy, c_y + dy
        self._emitir_limites()
        self.solicitar_render()

    def _centrar_y_zoom(self, px, py, factor):
        factor_dec = Decimal(str(factor))
        prop_x = Decimal(px) / Decimal(max(1, self.width()))
        prop_y = Decimal(py) / Decimal(max(1, self.height()))
        c_x = self.xmin + prop_x * (self.xmax - self.xmin)
        c_y = self.ymin + prop_y * (self.ymax - self.ymin)
        self.xmin = c_x - (c_x - self.xmin) * factor_dec
        self.xmax = c_x + (self.xmax - c_x) * factor_dec
        self.ymin = c_y - (c_y - self.ymin) * factor_dec
        self.ymax = c_y + (self.ymax - c_y) * factor_dec
        self._emitir_limites()
        self.solicitar_render()

    def keyPressEvent(self, event):
        move = Decimal('0.05')
        dx, dy = (self.xmax - self.xmin) * move, (self.ymax - self.ymin) * move
        tecla = event.key()

        if tecla in (Qt.Key_Left, Qt.Key_A):
            self.xmin, self.xmax = self.xmin - dx, self.xmax - dx
        elif tecla in (Qt.Key_Right, Qt.Key_D):
            self.xmin, self.xmax = self.xmin + dx, self.xmax + dx
        elif tecla in (Qt.Key_Up, Qt.Key_W):
            self.ymin, self.ymax = self.ymin + dy, self.ymax + dy
        elif tecla in (Qt.Key_Down, Qt.Key_S):
            self.ymin, self.ymax = self.ymin - dy, self.ymax - dy
        elif tecla == Qt.Key_Plus:
            self._aplicar_zoom(self.zoom_in)
            return
        elif tecla == Qt.Key_Minus:
            self._aplicar_zoom(self.zoom_out)
            return
        elif tecla == Qt.Key_P:
            self.next_palette()
            return
        elif tecla == Qt.Key_O:
            self.previous_palette()
            return
        elif tecla == Qt.Key_R:
            self.resetear()
            return
        elif tecla in (Qt.Key_G, Qt.Key_H, Qt.Key_B, Qt.Key_N):
            if tecla == Qt.Key_G:
                self.max_iter = int(self.max_iter * 2)
            elif tecla == Qt.Key_H:
                self.max_iter = max(1, int(self.max_iter / 2))
            elif tecla == Qt.Key_B:
                self.clase_equiv = int(self.clase_equiv * 2)
            else:
                self.clase_equiv = max(1, int(self.clase_equiv / 2))
            self.parametros_matematicos_cambiados.emit(self.max_iter, self.clase_equiv)
        elif tecla == Qt.Key_Q:
            self.toggle_modo_completo()
            return
        else:
            super().keyPressEvent(event)
            return

        if tecla in (Qt.Key_Left, Qt.Key_A, Qt.Key_Right, Qt.Key_D,
                     Qt.Key_Up, Qt.Key_W, Qt.Key_Down, Qt.Key_S):
            self._emitir_limites()
        self.solicitar_render()
