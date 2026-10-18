"""Embeddable widget of the module.

Inputs are methods/slots, outputs are signals; the widget never reaches into a main
window and does not rely on an application stylesheet.

Threading:
    GUI thread only.
"""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from .core import scale


class TemplateWidget(QWidget):
    """Shows a value and announces when it changes.

    Signals:
        valueChanged(float): emitted by :meth:`set_value` with the scaled value.

    Args:
        parent: Optional Qt parent.
    """

    valueChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = QLabel("-")
        layout = QVBoxLayout(self)
        layout.addWidget(self._label)

    def set_value(self, value: float, factor: float = 1.0) -> None:
        """Display ``value * factor`` and emit :attr:`valueChanged`.

        Args:
            value: Input value.
            factor: Multiplier applied before display.
        """
        result = scale([value], factor)[0]
        self._label.setText(f"{result:g}")
        self.valueChanged.emit(result)
