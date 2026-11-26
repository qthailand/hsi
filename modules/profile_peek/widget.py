"""``ProfilePeek``: a small popup card with one calibration curve, a few facts and *Open ▸*.

A status card opens it next to itself; a click outside or Esc closes it, *Open ▸* closes it and
emits :attr:`ProfilePeek.openRequested` (the host then opens the full viewer).

Data conventions:
    Shows a :class:`~modules.profile_peek.core.PeekData`; the curve's arrays are plotted as given.

Threading:
    A QWidget: create and call it on the GUI thread only.

Example:
    >>> from PyQt5.QtWidgets import QApplication
    >>> app = QApplication.instance() or QApplication([])
    >>> from modules.profile_peek import ProfilePeek, text_peek
    >>> peek = ProfilePeek()
    >>> peek.set_data(text_peek("Wavelength", "Stored", ["bin 1"]))
    >>> peek.title_label.text()
    'Wavelength'
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPalette
from PyQt5.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QToolButton,
                             QVBoxLayout, QWidget)

from .core import COLORS, PeekData

#: Popup width (px); the plot is this wide less the margins.
PEEK_WIDTH = 320
#: Plot height (px).
PLOT_HEIGHT = 150
#: Gap between the anchor and the popup (px).
GAP = 6


def _decade_ticks(y):
    """Ticks of a log axis at whole decades only (the minor labels overlap in a plot this small).

    Returns ``[[(log10 value, label), ...], []]`` for ``AxisItem.setTicks``, or ``None`` (automatic
    ticks) when ``y`` has no positive finite value.
    """
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y) & (y > 0)]
    if not y.size:
        return None
    lo, hi = int(np.floor(np.log10(y.min()))), int(np.ceil(np.log10(y.max())))
    return [[(float(e), f"{10.0 ** e:g}") for e in range(lo, hi + 1)], []]


class ProfilePeek(QFrame):
    """Popup card: coloured dot + title, a curve, a few lines, an optional warning and *Open ▸*.

    Signals:
        openRequested(): *Open ▸* was pressed (the popup has closed already).
        actionRequested(str): one of :attr:`PeekData.actions` was pressed: its key (closed already).

    Attributes:
        title_label, lines_label, warning_label (QLabel), open_btn (QPushButton),
        plot (``pyqtgraph.PlotWidget``): exposed for hosts and tests.
    """

    openRequested = pyqtSignal()
    actionRequested = pyqtSignal(str)

    def __init__(self, parent=None):
        """Build an empty peek (call :meth:`set_data`, then :meth:`popup`)."""
        super().__init__(parent)
        import pyqtgraph as pg
        self.setObjectName("profilePeek")
        self.setFrameShape(QFrame.StyledPanel)
        self.setAutoFillBackground(True)
        self.setFixedWidth(PEEK_WIDTH)
        self._data = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(6)

        head = QHBoxLayout()
        head.setSpacing(8)
        self.dot = QLabel()
        self.dot.setFixedSize(10, 10)
        self.title_label = QLabel()
        bold = QFont(self.title_label.font())
        bold.setBold(True)
        self.title_label.setFont(bold)
        self.close_btn = QToolButton()
        self.close_btn.setText("×")
        self.close_btn.setAutoRaise(True)
        self.close_btn.setToolTip("Close (Esc)")
        self.close_btn.clicked.connect(self.close)
        head.addWidget(self.dot)
        head.addWidget(self.title_label, 1)
        head.addWidget(self.close_btn)
        layout.addLayout(head)

        # the host's pyqtgraph background (pg.setConfigOption), like every other plot of the host
        self.plot = pg.PlotWidget(background="default")
        self.plot.setFixedHeight(PLOT_HEIGHT)
        self.plot.setMouseEnabled(False, False)
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)
        item = self.plot.getPlotItem()
        item.showGrid(x=True, y=True, alpha=0.15)
        small = QFont(self.font())
        small.setPointSizeF(max(6.0, small.pointSizeF() * 0.8))
        text = pg.mkColor(pg.getConfigOption("foreground"))
        for side in ("bottom", "left"):
            axis = item.getAxis(side)
            axis.setTickFont(small)
            axis.setTextPen(text)
            axis.setPen(pg.mkPen(text.darker(150)))
            axis.label.setFont(small)
        item.getAxis("left").setWidth(44)
        item.getAxis("bottom").setHeight(36)           # room for the tick labels and the caption
        self._curve = item.plot([], [], connect="finite")
        self._guide = pg.InfiniteLine(angle=0, movable=False,
                                      pen=pg.mkPen(text.darker(160), style=Qt.DashLine))
        item.addItem(self._guide)
        layout.addWidget(self.plot)

        self.lines_label = QLabel()
        self.lines_label.setWordWrap(True)
        self.lines_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.lines_label)

        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        warn = self.warning_label.palette()
        warn.setColor(QPalette.WindowText, Qt.darkYellow)
        self.warning_label.setPalette(warn)
        layout.addWidget(self.warning_label)

        foot = QHBoxLayout()
        self._actions_row = QHBoxLayout()
        self._actions_row.setSpacing(6)
        self.action_buttons = {}
        foot.addLayout(self._actions_row)
        foot.addStretch(1)
        self.open_btn = QPushButton("Open ▸")
        self.open_btn.setDefault(True)
        self.open_btn.clicked.connect(self._open)
        foot.addWidget(self.open_btn)
        layout.addLayout(foot)

    # ------------------------------------------------------------------ content
    def data(self):
        """The :class:`PeekData` shown, or ``None``."""
        return self._data

    def set_data(self, data: PeekData):
        """Show ``data``: title, colour, curve (hidden when ``None``), lines, warning, button text."""
        import pyqtgraph as pg
        self._data = data
        color = COLORS.get(data.color, COLORS["cyan"])
        self.dot.setStyleSheet(f"background: {color}; border-radius: 5px;")
        self.title_label.setText(data.title)
        self.lines_label.setText("\n".join(data.lines))
        self.lines_label.setVisible(bool(data.lines))
        self.warning_label.setText(f"⚠ {data.warning}" if data.warning else "")
        self.warning_label.setVisible(bool(data.warning))
        self.open_btn.setText(data.open_label)
        self.open_btn.setVisible(bool(data.open_label))
        self._set_actions(data.actions)
        curve = data.curve
        self.plot.setVisible(curve is not None)
        if curve is not None:
            item = self.plot.getPlotItem()
            y = np.asarray(curve.y, dtype=np.float64)
            if curve.log_y:
                y = np.where(y > 0, y, np.nan)
            item.setLogMode(x=False, y=bool(curve.log_y))
            item.getAxis("left").setTicks(_decade_ticks(y) if curve.log_y else None)
            self._curve.setData(np.asarray(curve.x, dtype=np.float64), y, connect="finite")
            self._curve.setPen(pg.mkPen(color, width=2))
            self._guide.setVisible(curve.reference is not None and not curve.log_y)
            if curve.reference is not None:
                self._guide.setValue(float(curve.reference))
            item.setLabel("bottom", curve.x_label)
            item.setLabel("left", curve.y_label)
            item.enableAutoRange()
        self.adjustSize()

    # ---------------------------------------------------------------- placement
    def popup(self, anchor: QWidget = None):
        """Show as a popup beside ``anchor`` (below it, else above; kept on the anchor's screen).

        Without an anchor it opens at the cursor. A click outside or Esc closes it.
        """
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.adjustSize()
        size = self.size()          # the real (fixed-width) size; sizeHint() asks for the plot's 600 px
        if anchor is not None:
            top_left = anchor.mapToGlobal(QPoint(0, 0))
            below = top_left + QPoint(0, anchor.height() + GAP)
            above = top_left - QPoint(0, size.height() + GAP)
            screen = anchor.screen().availableGeometry() if hasattr(anchor, "screen") else \
                QApplication.desktop().availableGeometry(anchor)
            pos = below if below.y() + size.height() <= screen.bottom() else above
        else:
            from PyQt5.QtGui import QCursor
            pos = QCursor.pos()
            screen = QApplication.desktop().availableGeometry(pos)
        x = min(max(pos.x(), screen.left()), screen.right() - size.width())
        y = min(max(pos.y(), screen.top()), screen.bottom() - size.height())
        self.move(x, y)
        self.show()
        focus = self.open_btn if not self.open_btn.isHidden() else next(iter(self.action_buttons.values()), None)
        if focus is not None:
            focus.setFocus()

    def _set_actions(self, actions):
        while self._actions_row.count():
            widget = self._actions_row.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        self.action_buttons = {}
        for key, label in actions:
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, key=key: self._action(key))
            self._actions_row.addWidget(button)
            self.action_buttons[key] = button
        if self.open_btn.isHidden() and self.action_buttons:     # Enter = the first action
            next(iter(self.action_buttons.values())).setDefault(True)

    def _action(self, key: str):
        self.close()
        self.actionRequested.emit(key)

    def _open(self):
        self.close()
        self.openRequested.emit()

    def keyPressEvent(self, event):
        """Enter = the focused button, else *Open ▸* (or the first action); Esc closes (Qt's popup
        default). Outside a dialog a focused button ignores Enter, so it is pressed here."""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            focused = QApplication.focusWidget()
            buttons = [self.open_btn, *self.action_buttons.values()]
            if any(focused is b for b in buttons) and focused.isVisible() and focused.isEnabled():
                focused.click()
            elif not self.open_btn.isHidden():
                self._open()
            elif self.action_buttons:
                self._action(next(iter(self.action_buttons)))
            return
        super().keyPressEvent(event)
