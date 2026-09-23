"""HSI System entry point."""

from __future__ import annotations
import logging
import qdarktheme
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
import sklearn
import sys
import os
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
print(sys.getswitchinterval())


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _init_pyqtgraph_theme():
    """Configure pyqtgraph defaults to match qdarktheme dark palette.

    Must be called before any PlotWidget is constructed.
    """
    pg.setConfigOptions(
        antialias=True,
        background="#1a1a1a",
        foreground="#e0e0e0",
    )


def main():
    _init_pyqtgraph_theme()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(qdarktheme.load_stylesheet("dark"))

    from windows.main_window import MainWindow

    main_win = MainWindow()
    main_win.resize(1400, 900)
    main_win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        input("Press Enter...")
