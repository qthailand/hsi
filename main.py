"""HSI System entry point."""

from __future__ import annotations
import os
import sys
# Thread caps for the BLAS/OpenMP pools: read once when numpy/sklearn load, so they must be
# set before any of the imports below.
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import logging
import traceback
import qdarktheme
import pyqtgraph as pg
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication, QMessageBox
from modules.ui_kit.theme import ACCENT_DARK, PLOT_BACKGROUND, PLOT_FOREGROUND, dashboard_stylesheet
import sklearn
print(sys.getswitchinterval())


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class _ErrorReporter(QObject):
    """Shows uncaught-exception messages in the GUI thread, whichever thread raised."""

    report = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._boxes = []        # open messages, kept referenced while shown
        # Queued even from the GUI thread: the hook may run inside a paint or another slot.
        self.report.connect(self._show, Qt.QueuedConnection)

    def _show(self, text: str):
        """Show ``text`` in a non-modal box; the event loop (and an in-RAM scan) goes on."""
        box = QMessageBox(QMessageBox.Critical, "Unexpected Error",
                          "An unexpected error occurred. The application keeps running; "
                          "save your scan data.\n\n" + text)
        box.setModal(False)
        box.finished.connect(lambda *_: self._boxes.remove(box))
        self._boxes.append(box)
        box.show()


_reporter = None


def _excepthook(exc_type, exc, tb):
    """Log an uncaught exception and report it instead of aborting the process.

    With PyQt5 >= 5.5 an exception escaping a slot calls ``qFatal`` unless
    ``sys.excepthook`` is replaced, which would lose a scan that is only in RAM.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
    if _reporter is not None:
        _reporter.report.emit("".join(traceback.format_exception_only(exc_type, exc)).strip())


def install_excepthook():
    """Route uncaught exceptions to :func:`_excepthook` (call in the GUI thread, before the event loop)."""
    global _reporter
    if _reporter is None:
        _reporter = _ErrorReporter()
    sys.excepthook = _excepthook


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
    install_excepthook()
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
        traceback.print_exc()
        input("Press Enter...")
