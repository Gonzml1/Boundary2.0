import sys
import argparse
import logging

from PyQt5 import QtWidgets


def configurar_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explorador de fractales ABoundary")
    parser.add_argument("--debug", action="store_true",
                        help="muestra mensajes de depuración (tiempos de cada render, órbitas, etc.)")
    args, qt_args = parser.parse_known_args()
    configurar_logging(args.debug)

    # Se importa después de configurar logging para ver qué backends se cargaron
    import gui.tema_oscuro as ts
    from gui.gui import MainWindow

    app = QtWidgets.QApplication(sys.argv[:1] + qt_args)
    ts.tema_oscuro(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
