from gui.MandelbrotGUI import Ui_Boundary
from PyQt5 import QtWidgets
from core.modulo_opengl import MandelbrotWidget

def mostrar_fractal_opengl(ui):
    try:
        # Obtener los 16 valores exactos desde la UI
        cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, zoom_in, zoom_out, clase_equiv, real, imag = obtener_datos(ui)
        
        mandelbrot_widget = MandelbrotWidget(
            cmap, xmin, xmax, ymin, ymax, width, height, max_iter, 
            formula, tipo_calculo, tipo_fractal, zoom_in, zoom_out, 
            clase_equiv, real, imag
        )

        if ui.grafico_openGLWidget.layout() is None:
            layout = QtWidgets.QVBoxLayout(ui.grafico_openGLWidget)
            layout.setContentsMargins(0, 0, 0, 0)
            ui.grafico_openGLWidget.setLayout(layout)
        else:
            layout = ui.grafico_openGLWidget.layout()

        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        layout.addWidget(mandelbrot_widget)
        return mandelbrot_widget
    except Exception as e:
        # Imprimimos el error real en consola para no fallar en silencio
        print(f"Error crítico al cargar OpenGL: {e}")
        raise e

def obtener_datos(ui):
    cmap          =   str(ui.cmap_comboBox.currentText())
    xmin          =   float(ui.xmin_entrada.text())
    xmax          =   float(ui.xmax_entrada.text())
    ymin          =   float(ui.ymin_entrada.text())
    ymax          =   float(ui.ymax_entrada.text())
    width         =   int(ui.width_entrada.text())
    height        =   int(ui.high_entrada.text())
    max_iter      =   int(ui.max_iter_entrada.text())
    tipo_calculo  =   str(ui.tipo_calculo_comboBox.currentText())
    tipo_fractal  =   str(ui.tipo_fractal_comboBox.currentText())
    formula       =   str(ui.formula_entrada.text())
    zoom_out      =   float(ui.zoom_out_factor_entrada.text())
    zoom_in       =   float(ui.zoom_in_factor_entrada.text())
    clase_equiv   =   int(ui.clase_equiv_entrada.text())
    real          =   float(ui.real_julia_entrada.text())
    imag          =   float(ui.im_julia_entrada.text())
    
    return cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, zoom_in, zoom_out, clase_equiv, real, imag