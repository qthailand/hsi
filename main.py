"""HSI System entry point."""

from __future__ import annotations
import logging
import qdarktheme
import pyqtgraph as pg
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication
from modules.ui_kit.theme import ACCENT_DARK, PLOT_BACKGROUND, PLOT_FOREGROUND, dashboard_stylesheet
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
    """Configure pyqtgraph defaults to match the dashboard card colours.

    Must be called before any PlotWidget is constructed.
    """
    pg.setConfigOptions(
        antialias=True,
        background=PLOT_BACKGROUND,
        foreground=PLOT_FOREGROUND,
    )


def main():
    from modules.compute_runtime import initialize_compute
    initialize_compute()
    _init_pyqtgraph_theme()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(dashboard_stylesheet(qdarktheme.load_stylesheet("dark")))
    palette = app.palette()
    palette.setColor(QPalette.Highlight, QColor(ACCENT_DARK))   # MDI title bars and other Fusion highlights
    app.setPalette(palette)

    from windows.main_window import MainWindow

    main_win = MainWindow(persist_settings=True)   # remember camera/stage settings for the next start
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
