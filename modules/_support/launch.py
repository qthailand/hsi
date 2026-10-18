"""Stand-alone launcher used by ``python -m modules.<name>``."""
from __future__ import annotations

from typing import Callable, Tuple


def launch(widget_factory: Callable[[], object], title: str, size: Tuple[int, int] = (1200, 900)) -> int:
    """Show ``widget_factory()`` in its own main window and run the Qt event loop.

    Args:
        widget_factory: Zero-argument callable returning the ``QWidget`` to show.
            It is called after the ``QApplication`` exists.
        title: Window title.
        size: Initial ``(width, height)`` in pixels.

    Returns:
        The process exit code from ``QApplication.exec_()``.

    Example:
        >>> # python -m modules.wavelength_calibration
        >>> # launch(lambda: WavelengthCalibrationWidget(), 'Wavelength Calibration')  # doctest: +SKIP

    Note:
        PyQt5 is imported here, not at module import time, so importing a module
        package never pulls in Qt.
    """
    from PyQt5.QtWidgets import QApplication, QMainWindow

    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    window.setWindowTitle(title)
    window.setCentralWidget(widget_factory())
    window.resize(*size)
    window.show()
    return app.exec_()
