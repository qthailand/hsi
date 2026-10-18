"""Vertical sharpness sidebar: large monospace reading plus a bar indicator.

Data conventions:
    The score is the unit-less number from :func:`modules.sharpness.sharpness_score`
    (larger = sharper). The bar shows ``score / peak`` where ``peak`` is the running
    maximum, which decays slowly (factor 0.9995 per :meth:`SharpnessPanel.set_value`
    call, floor 1.0).

Threading:
    GUI thread only. Feed it from a signal such as
    ``CameraPreviewThread.sharpness_ready`` (about 5 Hz).

Example:
    >>> # panel = SharpnessPanel()
    >>> # thread.sharpness_ready.connect(panel.set_value)
    >>> # panel.set_scanning(True)   # number turns amber during a scan
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QWidget, QSizePolicy

# ── palette ─────────────────────────────────────────────────────
_BG = QColor("#1a1a1a")
_BORDER = QColor("#333333")
_BAR_BG = QColor("#222222")
_BAR_BORDER = QColor("#3a3a3a")
_NUM_NORMAL = QColor("#e0e0e0")
_NUM_SCANNING = QColor("#ffa726")
_IDLE_TEXT = QColor("#555555")

# bar fill colours by ratio
_FILL_LOW = QColor("#546e7a")
_FILL_MID = QColor("#78909c")
_FILL_HIGH = QColor("#66bb6a")
_FILL_PEAK = QColor("#81c784")

_PEAK_DECAY = 0.9995          # per set_value call (~5 Hz → half-life ~2.3 min)
_FONT_FAMILY = "monospace"


class SharpnessPanel(QWidget):
    """Fixed-width sidebar that shows live sharpness next to the image.

    Layout::

        ┌──────────┐  48 px
        │          │
        │   274    │  ← large number (auto-formatted)
        │          │
        │ ████████ │  ← vertical bar (score / peak)
        │ ████████ │
        │ ████████ │
        │          │
        └──────────┘

    The widget is 48 px wide, expands vertically (minimum 80 px) and paints itself with
    fixed dark colours (it does not follow the application palette). It shows an em dash
    until the first :meth:`set_value`. It has no signals. Constructible with no arguments.
    The host only needs to call :meth:`set_value` with scores; GUI thread only.
    """

    def __init__(self, parent: QWidget | None = None):
        """Create an idle panel.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)

        self.setFixedWidth(48)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setMinimumHeight(80)

        self._score: float = 0.0
        self._peak: float = 1.0
        self._scanning: bool = False
        self._has_data: bool = False

    # ── public API ──────────────────────────────────────────────

    def set_value(self, score: float):
        """Show a new sharpness score and update the peak reference.

        Args:
            score: Sharpness score (unit-less, >= 0). A score above the peak raises the
                peak; otherwise the peak decays slightly.
        """
        self._score = score
        self._has_data = True

        if score > self._peak:
            self._peak = score
        else:
            self._peak = max(self._peak * _PEAK_DECAY, 1.0)

        self.update()

    def set_scanning(self, scanning: bool):
        """Colour the number amber while a scan runs.

        Args:
            scanning: True during a scan.
        """
        if scanning != self._scanning:
            self._scanning = scanning
            self.update()

    def reset(self):
        """Forget the score and the peak and return to the idle (em dash) display."""
        self._score = 0.0
        self._peak = 1.0
        self._has_data = False
        self.update()

    # ── paint ───────────────────────────────────────────────────

    def paintEvent(self, _event):
        """Paint the number and the score bar (Qt override; not called directly)."""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # background
        p.fillRect(0, 0, w, h, _BG)

        # left border
        p.setPen(QPen(_BORDER, 1))
        p.drawLine(0, 0, 0, h)

        # ── idle state ──────────────────────────────────────────
        if not self._has_data:
            font = QFont(_FONT_FAMILY, 10)
            font.setStyleHint(QFont.Monospace)
            p.setFont(font)
            p.setPen(_IDLE_TEXT)
            p.drawText(0, 0, w, h, Qt.AlignCenter, "\u2014")  # em-dash
            p.end()
            return

        # ── number ──────────────────────────────────────────────
        p.setPen(_NUM_SCANNING if self._scanning else _NUM_NORMAL)

        font = QFont(_FONT_FAMILY, 13, QFont.Bold)
        font.setStyleHint(QFont.Monospace)
        p.setFont(font)

        if self._score >= 100:
            text = f"{self._score:.0f}"
        elif self._score >= 10:
            text = f"{self._score:.1f}"
        else:
            text = f"{self._score:.1f}"

        p.drawText(0, 6, w, 36, Qt.AlignHCenter | Qt.AlignTop, text)

        # ── bar geometry ────────────────────────────────────────
        bar_x = 8
        bar_w = w - 16
        bar_top = 50
        bar_bottom = h - 10
        bar_h = max(1, bar_bottom - bar_top)

        # bar background
        p.fillRect(bar_x, bar_top, bar_w, bar_h, _BAR_BG)

        # bar fill (bottom → up, ratio = score / peak)
        ratio = min(1.0, self._score / max(self._peak, 1.0))
        fill_h = int(bar_h * ratio)

        if fill_h > 0:
            if ratio > 0.95:
                fill = _FILL_PEAK
            elif ratio > 0.80:
                fill = _FILL_HIGH
            elif ratio > 0.50:
                fill = _FILL_MID
            else:
                fill = _FILL_LOW
            p.fillRect(bar_x, bar_bottom - fill_h, bar_w, fill_h, fill)

        # peak reference line (dashed, at top of bar)
        p.setPen(QPen(QColor("#607d8b"), 1, Qt.DashLine))
        p.drawLine(bar_x, bar_top, bar_x + bar_w, bar_top)

        # bar border
        p.setPen(QPen(_BAR_BORDER, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(bar_x, bar_top, bar_w, bar_h)

        p.end()