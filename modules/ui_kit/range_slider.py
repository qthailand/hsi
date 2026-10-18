# widgets/range_slider.py
"""Minimal dual-handle range slider widget.

Data conventions:
    Both handle positions are normalised floats in ``0.0 .. 1.0`` (fractions of
    the track, not pixels).  ``low < high`` is kept while dragging, with a
    minimum gap of 0.01.

Threading:
    GUI thread only.

Example:
    >>> from modules.ui_kit import RangeSlider            # doctest: +SKIP
    >>> slider = RangeSlider(low=0.02, high=0.98)         # doctest: +SKIP
    >>> slider.range_changed.connect(print)               # doctest: +SKIP
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush
from PyQt5.QtWidgets import QWidget, QSizePolicy


class RangeSlider(QWidget):
    """Compact dual-handle slider: drag the left/right handle to set low/high.

    Args:
        low: Initial lower handle position, fraction ``0.0 .. 1.0`` (clamped).
        high: Initial upper handle position, fraction ``0.0 .. 1.0`` (clamped).
        parent: Optional parent widget.

    Signals:
        range_changed(float, float): ``(low, high)`` in ``0.0 .. 1.0``.  Emitted
            on every mouse move while a handle is dragged and by :meth:`reset`.
            It is NOT emitted by :meth:`set_range` or by the constructor.

    The widget is fixed to 28 px height and expands horizontally.  Only the left
    mouse button drags; the handle nearest to the press position is picked.
    """

    range_changed = pyqtSignal(float, float)   # emit (low, high) in 0.0–1.0

    def __init__(self, low: float = 0.02, high: float = 0.98, parent=None):
        super().__init__(parent)
        self._low = max(0.0, min(1.0, low))
        self._high = max(0.0, min(1.0, high))
        self._dragging = None           # 'low', 'high', or None
        self._handle_radius = 6

        self.setMinimumHeight(28)
        self.setMaximumHeight(28)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)

    # ── public API ────────────────────────────────────────

    @property
    def low(self) -> float:
        """float: Current lower handle position, fraction ``0.0 .. 1.0``."""
        return self._low

    @property
    def high(self) -> float:
        """float: Current upper handle position, fraction ``0.0 .. 1.0``."""
        return self._high

    def set_range(self, low: float, high: float):
        """Set both handles programmatically WITHOUT emitting ``range_changed``.

        Args:
            low: New lower position, clamped to ``0.0 .. 1.0``.
            high: New upper position, at least ``low + 0.01`` (and at most 1.0 before
                that minimum gap is applied).
        """
        self._low = max(0.0, min(1.0, low))
        self._high = max(self._low + 0.01, min(1.0, high))
        self.update()

    def reset(self, low: float = 0.02, high: float = 0.98):
        """Move both handles to the given values and emit ``range_changed``.

        Args:
            low: New lower position (fraction, stored as given, not clamped).
            high: New upper position (fraction, stored as given, not clamped).
        """
        self._low = low
        self._high = high
        self.update()
        self.range_changed.emit(self._low, self._high)

    # ── geometry helpers ──────────────────────────────────

    def _track_rect(self) -> QRectF:
        r = self._handle_radius
        return QRectF(r, self.height() // 2 - 2, self.width() - 2 * r, 4)

    def _val_to_x(self, val: float) -> float:
        tr = self._track_rect()
        return tr.left() + val * tr.width()

    def _x_to_val(self, x: float) -> float:
        tr = self._track_rect()
        if tr.width() <= 0:
            return 0.0
        return max(0.0, min(1.0, (x - tr.left()) / tr.width()))

    # ── painting ──────────────────────────────────────────

    def paintEvent(self, event):
        """Qt paint handler: draws the track, the selected range and both handles."""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        tr = self._track_rect()
        r = self._handle_radius

        # background track
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(60, 60, 60)))
        p.drawRoundedRect(tr, 2, 2)

        # selected range
        x_low = self._val_to_x(self._low)
        x_high = self._val_to_x(self._high)
        sel_rect = QRectF(x_low, tr.top(), x_high - x_low, tr.height())
        p.setBrush(QBrush(QColor(79, 195, 247, 140)))
        p.drawRoundedRect(sel_rect, 2, 2)

        # low handle
        p.setPen(QPen(QColor(79, 195, 247), 2))
        p.setBrush(QBrush(QColor(30, 30, 30)))
        p.drawEllipse(QRectF(x_low - r, tr.center().y() - r, 2 * r, 2 * r))

        # high handle
        p.setPen(QPen(QColor(79, 195, 247), 2))
        p.setBrush(QBrush(QColor(30, 30, 30)))
        p.drawEllipse(QRectF(x_high - r, tr.center().y() - r, 2 * r, 2 * r))

        p.end()

    # ── mouse interaction ─────────────────────────────────

    def mousePressEvent(self, event):
        """Qt handler: on left press select the handle nearest to the cursor for dragging."""
        if event.button() != Qt.LeftButton:
            return

        x = event.x()
        x_low = self._val_to_x(self._low)
        x_high = self._val_to_x(self._high)

        # ★ pick closest handle
        if abs(x - x_low) < abs(x - x_high):
            self._dragging = 'low'
        else:
            self._dragging = 'high'

    def mouseMoveEvent(self, event):
        """Qt handler: while dragging, move the selected handle and emit ``range_changed``.

        The handles never cross; they keep a minimum gap of 0.01.
        """
        if self._dragging is None:
            return

        val = self._x_to_val(event.x())

        if self._dragging == 'low':
            self._low = min(val, self._high - 0.01)
        elif self._dragging == 'high':
            self._high = max(val, self._low + 0.01)

        self.update()
        self.range_changed.emit(self._low, self._high)

    def mouseReleaseEvent(self, event):
        """Qt handler: end the drag."""
        self._dragging = None