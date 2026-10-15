import os
import sys

from PyQt5 import QtWidgets
from PyQt5.QtGui import QPalette, QColor, QFont

# Paleta: fondo casi negro, texto gris claro, acento beige/marrón
FONDO = "#1c1c1f"
FONDO_CAMPO = "#141416"
BORDE = "#2c2c31"
BORDE_HOVER = "#4a4a50"
TEXTO = "#d6d6d6"
TEXTO_TENUE = "#8c8c8c"
ACENTO = "#d4b4a4"          # beige rosado (líneas y topes)
ACENTO_OSCURO = "#4e4440"   # marrón (rellenos)

TIPOGRAFIA = "Consolas"
# QSS necesita barras "/" también en Windows
FLECHA_COMBO = os.path.join(os.path.dirname(__file__), "flecha_combo.svg").replace("\\", "/")

HOJA_DE_ESTILO = f"""
* {{
    font-family: "{TIPOGRAFIA}", "Cascadia Mono", "Courier New", monospace;
    font-size: 9pt;
    color: {TEXTO};
}}
QMainWindow, QWidget {{
    background-color: {FONDO};
}}
QLabel {{
    background: transparent;
    color: {TEXTO_TENUE};
}}
QLineEdit {{
    background-color: {FONDO_CAMPO};
    border: 1px solid {BORDE};
    border-radius: 0px;
    padding: 3px 5px;
    selection-background-color: {ACENTO_OSCURO};
    selection-color: {TEXTO};
}}
QLineEdit:hover {{ border-color: {BORDE_HOVER}; }}
QLineEdit:focus {{ border-color: {ACENTO}; }}

QPushButton {{
    background-color: {ACENTO_OSCURO};
    border: none;
    border-top: 2px solid {ACENTO};
    padding: 4px 10px;
    color: {TEXTO};
}}
QPushButton:hover {{ background-color: #5e524d; }}
QPushButton:pressed {{ background-color: #3a3330; }}

QComboBox {{
    background-color: {FONDO_CAMPO};
    border: 1px solid {BORDE};
    border-radius: 0px;
    padding: 3px 5px;
}}
QComboBox:hover {{ border-color: {BORDE_HOVER}; }}
QComboBox:focus {{ border-color: {ACENTO}; }}
QComboBox::drop-down {{ border: none; width: 16px; }}
QComboBox::down-arrow {{ image: url("{FLECHA_COMBO}"); width: 8px; height: 5px; }}
QComboBox QAbstractItemView {{
    background-color: {FONDO_CAMPO};
    border: 1px solid {BORDE};
    selection-background-color: {ACENTO_OSCURO};
    outline: none;
}}

QSlider::groove:horizontal {{
    height: 6px;
    background: #2a2a2e;
}}
QSlider::sub-page:horizontal {{ background: {ACENTO_OSCURO}; }}
QSlider::handle:horizontal {{
    width: 3px;
    margin: -4px 0;
    background: {ACENTO};
}}

QOpenGLWidget {{ border: 1px solid {BORDE}; }}

/* Paneles con título numerado chico arriba a la izquierda */
QGroupBox {{
    border: 1px solid {BORDE};
    margin-top: 14px;
    padding: 8px 6px 6px 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 0px;
    padding: 0 2px;
    color: {TEXTO_TENUE};
}}
QLabel:disabled, QLineEdit:disabled, QCheckBox:disabled, QPushButton:disabled {{ color: #555559; }}
QPushButton:disabled {{ background-color: #2a2a2e; border-top-color: #3a3a3f; }}

QCheckBox {{ spacing: 6px; background: transparent; }}
QCheckBox::indicator {{
    width: 11px; height: 11px;
    border: 1px solid {BORDE_HOVER};
    background: {FONDO_CAMPO};
}}
QCheckBox::indicator:hover {{ border-color: {ACENTO}; }}
QCheckBox::indicator:checked {{ background: {ACENTO}; border-color: {ACENTO}; }}

QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}

QMenuBar, QStatusBar {{
    background-color: {FONDO};
    color: {TEXTO_TENUE};
    border-top: 1px solid {BORDE};
}}
QMenuBar {{ border-top: none; border-bottom: 1px solid {BORDE}; }}

QProgressBar {{
    background-color: #2a2a2e;
    border: none;
    text-align: center;
    color: {TEXTO};
}}
QProgressBar::chunk {{ background-color: {ACENTO_OSCURO}; }}

QToolTip {{
    background-color: {FONDO_CAMPO};
    border: 1px solid {BORDE};
    color: {TEXTO};
}}

QScrollBar:vertical, QScrollBar:horizontal {{ background: {FONDO}; width: 8px; height: 8px; }}
QScrollBar::handle {{ background: {BORDE_HOVER}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
"""


def tema_oscuro(app=None):
    """Aplica el tema oscuro (estilo instrumento de medición) a la aplicación Qt dada.

    Si ``app`` es ``None`` se intenta usar la instancia de
    :class:`~PyQt5.QtWidgets.QApplication` existente. En caso de no
    existir, se crea una nueva.
    """
    if app is None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

    paleta = QPalette()
    paleta.setColor(QPalette.ColorRole.Window, QColor(FONDO))
    paleta.setColor(QPalette.ColorRole.WindowText, QColor(TEXTO))
    paleta.setColor(QPalette.ColorRole.Base, QColor(FONDO_CAMPO))
    paleta.setColor(QPalette.ColorRole.AlternateBase, QColor(FONDO))
    paleta.setColor(QPalette.ColorRole.ToolTipBase, QColor(FONDO_CAMPO))
    paleta.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXTO))
    paleta.setColor(QPalette.ColorRole.Text, QColor(TEXTO))
    paleta.setColor(QPalette.ColorRole.Button, QColor(ACENTO_OSCURO))
    paleta.setColor(QPalette.ColorRole.ButtonText, QColor(TEXTO))
    paleta.setColor(QPalette.ColorRole.BrightText, QColor("#e06c6c"))
    paleta.setColor(QPalette.ColorRole.Highlight, QColor(ACENTO_OSCURO))
    paleta.setColor(QPalette.ColorRole.HighlightedText, QColor(TEXTO))

    QtWidgets.QApplication.setStyle("Fusion")
    app.setPalette(paleta)
    app.setFont(QFont(TIPOGRAFIA, 9))
    app.setStyleSheet(HOJA_DE_ESTILO)
