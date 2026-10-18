# widgets/histogram_plot.py
"""Lightweight intensity histogram widget drawn with QPainter only (no pyqtgraph).

Data conventions:
    ``update_histogram`` takes a raw frame (any shape, integer dtype); values
    are scaled to 8 bit using the sensor bit depth, giving 256 bins.

Threading:
    GUI thread only.

Example:
    >>> import numpy as np
    >>> from modules.live_view import HistogramPlot                 # doctest: +SKIP
    >>> hist = HistogramPlot(bit_depth=12)                          # doctest: +SKIP
    >>> hist.update_histogram(np.random.randint(0, 4096, (100, 100)))  # doctest: +SKIP
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QFont, QFontMetrics
from PyQt5.QtWidgets import QWidget

from .saturation import saturated_mask


class HistogramPlot(QWidget):
    """Filled bar histogram: X = intensity (8-bit scaled), Y = pixel count.

    Bins at or above 99 % of full scale are drawn red and the share of saturated
    pixels is printed below the plot.  When a saturation ``ceiling`` is given to
    :meth:`update_histogram` (dark-subtracted frames clip below full scale), pixels
    at >= 99 % of their own ceiling are counted instead and drawn red wherever they
    fall.  Uses pure QPainter - no pyqtgraph.

    Args:
        bit_depth: Sensor bit depth in bits (full scale ``2**bit_depth - 1``).
        parent: Optional parent widget.

    Note:
        Its size limits are set in the constructor (minimum height 220 px).
    """

    _N_BINS = 256
    _8BIT_MAX = 255.0

    def __init__(self, bit_depth: int = 16, parent=None):
        super().__init__(parent)

        self._bit_depth = int(bit_depth)
        self._raw_max = float((1 << self._bit_depth) - 1)
        self._scale = self._8BIT_MAX / self._raw_max
        self._sat_threshold_8bit = int(self._8BIT_MAX * 0.99)

        # data
        self._counts: np.ndarray = np.zeros(self._N_BINS, dtype=np.float64)
        self._max_count: float = 1.0
        self._sat_pct: float = 0.0
        self._sat_counts: np.ndarray | None = None   # saturated pixels per bin (ceiling mode)
        self._has_data: bool = False

        # layout margins
        self._ml = 60
        self._mr = 12
        self._mt = 12
        self._mb = 45
        self._sat_label_h = 18          # ★ sat label height reserve

        # ★ NO layout, NO child widgets — pure paint

        self.setMinimumHeight(220)
        self.setMaximumHeight(160)

    # ── public API ────────────────────────────────────────────────

    def set_bit_depth(self, bit_depth: int):
        """Change the sensor bit depth used to scale values to 8 bit.

        Args:
            bit_depth: Bits; takes effect on the next :meth:`update_histogram`.
        """
        self._bit_depth = int(bit_depth)
        self._raw_max = float((1 << self._bit_depth) - 1)
        self._scale = self._8BIT_MAX / self._raw_max

    def update_histogram(self, frame: np.ndarray, ceiling: np.ndarray | None = None):
        """Recompute and repaint the histogram from a raw frame.

        Args:
            frame: Raw frame of any shape (it is flattened); values are in sensor
                counts.  Frames above 500 000 pixels are subsampled by 4 for speed.
                Not modified.
            ceiling: Optional per-pixel saturation ceiling of ``frame``'s shape (see
                :func:`~modules.live_view.saturation.saturated_mask`); another shape
                or ``None`` falls back to 99 % of full scale.
        """
        flat = frame.ravel()
        flat_ceiling = None
        if ceiling is not None and np.shape(ceiling) == frame.shape:
            flat_ceiling = np.asarray(ceiling).ravel()

        # downsample for speed
        if flat.size > 500_000:
            flat = flat[::4]
            if flat_ceiling is not None:
                flat_ceiling = flat_ceiling[::4]

        # normalize to 0-255
        flat_8bit = (flat.astype(np.float32) * self._scale)
        flat_8bit = np.clip(flat_8bit, 0, 255).astype(np.uint8)

        # histogram
        counts = np.bincount(flat_8bit, minlength=self._N_BINS).astype(np.float64)
        self._counts = counts
        self._max_count = float(counts.max()) if counts.max() > 0 else 1.0
        self._has_data = True

        # saturation %
        if flat_ceiling is not None:
            sat = saturated_mask(flat, flat_ceiling)
            n_sat = int(np.count_nonzero(sat))
            self._sat_counts = np.bincount(flat_8bit[sat], minlength=self._N_BINS).astype(np.float64)
        else:
            n_sat = int(np.count_nonzero(flat >= self._raw_max * 0.99))
            self._sat_counts = None
        self._sat_pct = (n_sat / flat.size) * 100.0 if flat.size > 0 else 0.0

        self.update()

    # ── paint ─────────────────────────────────────────────────────

    def paintEvent(self, event):
        """Qt handler: draw grid, axes, saturation zone, bars and saturation label."""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # background
        p.fillRect(0, 0, w, h, QColor(15, 15, 15))

        # ★ histogram area = widget minus sat_label at bottom
        plot_bottom = h - self._sat_label_h
        plot_w = w - self._ml - self._mr
        plot_h = plot_bottom - self._mt - self._mb

        if plot_w < 1 or plot_h < 1:
            self._paint_sat_label(p, w, h)
            p.end()
            return

        # ── grid ──────────────────────────────────────────────────
        p.setPen(QPen(QColor(40, 40, 40)))
        major_x = 8
        major_y = 4

        for i in range(major_x + 1):
            x = self._ml + i * plot_w / major_x
            p.drawLine(int(x), self._mt, int(x), self._mt + plot_h)

        for i in range(major_y + 1):
            y = self._mt + plot_h - i * plot_h / major_y
            p.drawLine(self._ml, int(y), self._ml + plot_w, int(y))

        # ── axis ──────────────────────────────────────────────────
        p.setPen(QPen(QColor(180, 180, 180)))
        p.drawLine(self._ml, self._mt, self._ml, self._mt + plot_h)
        p.drawLine(self._ml, self._mt + plot_h, self._ml + plot_w, self._mt + plot_h)

        # ── X ticks ───────────────────────────────────────────────
        for i in range(major_x + 1):
            x = self._ml + i * plot_w / major_x
            val = int(255 * i / major_x)
            p.drawText(int(x) - 10, self._mt + plot_h + 15, str(val))

        # ── X label ───────────────────────────────────────────────
        p.drawText(
            self._ml + plot_w // 2 - 40,
            self._mt + plot_h + 30,
            "Intensity (8-bit)",
        )

        # ── Y ticks ───────────────────────────────────────────────
        if self._has_data:
            metrics = p.fontMetrics()
            for i in range(major_y + 1):
                y = self._mt + plot_h - i * plot_h / major_y
                value = self._max_count * i / major_y
                if value >= 1_000_000:
                    text = f"{value / 1e6:.1f}M"
                elif value >= 1000:
                    text = f"{value / 1e3:.0f}K"
                else:
                    text = f"{value:.0f}"
                tw = metrics.horizontalAdvance(text)
                p.drawText(int(self._ml) - tw - 5, int(y) + 4, text)

        # ── saturation zone (red background) ─────────────────────
        sat_x_start = self._ml + (self._sat_threshold_8bit / 255.0) * plot_w
        sat_x_end = self._ml + plot_w
        p.fillRect(
            QRectF(sat_x_start, self._mt, sat_x_end - sat_x_start, plot_h),
            QColor(244, 67, 54, 40),
        )

        # ── bars ──────────────────────────────────────────────────
        if self._has_data and self._max_count > 0:
            bar_w = plot_w / self._N_BINS
            if bar_w < 0.5:
                bar_w = 0.5

            color_normal = QColor(79, 195, 247, 160)
            color_sat = QColor(244, 67, 54, 200)

            p.setPen(Qt.NoPen)

            for i in range(self._N_BINS):
                c = self._counts[i]
                if c <= 0:
                    continue

                bar_h = (c / self._max_count) * plot_h
                if bar_h < 0.3:
                    continue

                x = self._ml + i * bar_w
                y = self._mt + plot_h - bar_h

                if self._sat_counts is None:
                    p.setBrush(color_sat if i >= self._sat_threshold_8bit else color_normal)
                    p.drawRect(QRectF(x, y, bar_w + 0.5, bar_h))
                    continue
                p.setBrush(color_normal)
                p.drawRect(QRectF(x, y, bar_w + 0.5, bar_h))
                sat_h = (self._sat_counts[i] / self._max_count) * plot_h
                if sat_h >= 0.3:
                    p.setBrush(color_sat)
                    p.drawRect(QRectF(x, self._mt + plot_h - sat_h, bar_w + 0.5, sat_h))

        # ── sat label (painted at bottom) ─────────────────────────
        self._paint_sat_label(p, w, h)

        p.end()

    def _paint_sat_label(self, p: QPainter, w: int, h: int):
        """★ Paint saturation label at bottom — no child widget needed."""
        label_y = h - self._sat_label_h

        if self._sat_pct > 0.01:
            text = f"Saturated: {self._sat_pct:.2f}%"
            p.setPen(QPen(QColor(239, 83, 80)))  # red
            font = p.font()
            font.setBold(True)
            font.setPointSize(8)
            p.setFont(font)
        else:
            text = "Saturated: 0.00%"
            p.setPen(QPen(QColor(158, 158, 158)))  # gray
            font = p.font()
            font.setBold(False)
            font.setPointSize(8)
            p.setFont(font)

        p.drawText(self._ml, label_y + 13, text)
