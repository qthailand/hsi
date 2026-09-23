# widgets/spectrum.py

"""Spectrum plotting widgets backed by pyqtgraph.

Public API is identical to the old matplotlib version so callers
(CubeViewerWindow, ScanController, ScanWindow) require zero changes.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import QWidget, QVBoxLayout

# ── pen / color constants ───────────────────────────────────────────
_PEN_LIVE = pg.mkPen(color="#4fc3f7", width=1.4)
_PEN_ROI = pg.mkPen(color="#ffd54f", width=1.3)
_PEN_MARKER = pg.mkPen(
    color="#ef5350", width=1.2, style=Qt.DashLine,
)
_GRID_ALPHA = 0.18
_AXIS_LABEL_STYLE = {"color": "#bdbdbd", "font-size": "10px"}
_TITLE_STYLE = {"color": "#e0e0e0", "font-size": "11px"}
_TICK_FONT = QFont("sans-serif", 8)


# ── helpers ──────────────────────────────────────────────────────────
def _apply_axis_fonts(pw: pg.PlotWidget):
    """Set tick font on both axes."""
    for name in ("bottom", "left"):
        axis = pw.getAxis(name)
        axis.setPen(pg.mkPen(color="#616161", width=1))
        axis.setTextPen(pg.mkPen(color="#bdbdbd"))
        axis.setStyle(tickFont=_TICK_FONT)


def _clamp_y_range(
    pw: pg.PlotWidget,
    spectrum: np.ndarray,
):
    """Set Y range from data with 8 % padding.  X range is untouched."""
    ymin = float(np.min(spectrum))
    ymax = float(np.max(spectrum))
    if ymax <= ymin:
        ymax = ymin + 1.0
    margin = (ymax - ymin) * 0.08 + 1e-6
    pw.setYRange(ymin - margin, ymax + margin, padding=0)


# ═══════════════════════════════════════════════════════════════════
#  LiveLineSpectrumWidget — lightweight QPainter (no pyqtgraph)
# ═══════════════════════════════════════════════════════════════════

class LiveLineSpectrumWidget(QWidget):
    """High-performance live spectrum using pure QPainter.

    Y-axis: auto-scale UP to next bit-depth ceiling, never down.
    Prevents jitter while ensuring data is always visible.
    """

    _BIT_CEILINGS = [
        (255,       255),       # 8-bit
        (1023,      1023),      # 10-bit
        (4095,      4095),      # 12-bit
        (65535,     65535),     # 16-bit
    ]

    def __init__(self):
        super().__init__()

        self._n_points: int = 0
        self._data: np.ndarray | None = None
        self._wl_min: float = 0.0
        self._wl_max: float = 1.0
        self._use_wavelengths: bool = False
        self._xlabel: str = "Band"

        # ★ Y range
        self._y_max: float = 65535.0
        self._manual_bit_depth: int | None = None

        # layout
        self._ml = 55
        self._mr = 15
        self._mt = 18
        self._mb = 40

        self._major_x = 10
        self._major_y = 8
        self._minor_x = 5
        self._minor_y = 5

        self._polygon = QPolygonF()
        self._x_cache: np.ndarray = np.zeros(0)

        self.setMinimumHeight(220)
        self.setMinimumWidth(200)

    # ── public API ────────────────────────────────────────────────

    def set_bit_depth(self, bit_depth: int):
        """Manual override — lock Y range."""
        self._manual_bit_depth = int(bit_depth)
        self._y_max = float(2 ** int(bit_depth) - 1)
        if self._data is not None and self._n_points > 0:
            self._update_polygon_points()
            self.update()

    def clear(self):
        self._data = None
        self._n_points = 0
        self._polygon.clear()
        self._x_cache = np.zeros(0)
        # ★ reset Y to 16-bit default
        self._y_max = 65535.0
        self.update()

    def update_spectrum(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray | None = None,
    ):
        if spectrum is None:
            return

        spectrum = np.asarray(spectrum, dtype=np.float32).ravel()
        if spectrum.size == 0:
            return

        # ★ auto-scale Y (UP only, never down)
        if self._manual_bit_depth is None:
            self._auto_scale_y(spectrum)

        # resolve axis
        use_wl = (
            wavelengths is not None
            and np.asarray(wavelengths).shape[0] == spectrum.shape[0]
        )

        if use_wl:
            wl = np.asarray(wavelengths, dtype=np.float32)
            self._wl_min = float(wl[0])
            self._wl_max = float(wl[-1])
            self._use_wavelengths = True
            self._xlabel = "Wavelength (nm)"
        else:
            self._wl_min = 0.0
            self._wl_max = float(spectrum.shape[0] - 1)
            self._use_wavelengths = False
            self._xlabel = "Band"

        # detect dimension change
        new_n = spectrum.shape[0]
        if new_n != self._n_points:
            self._n_points = new_n
            self._rebuild_polygon()
            self._rebuild_x_cache()

        self._data = spectrum
        self._update_polygon_points()
        self.update()

    # ── auto-scale Y (UP only) ───────────────────────────────────

    def _auto_scale_y(self, spectrum: np.ndarray):
        """Scale Y UP to next bit-depth ceiling if data exceeds current y_max.

        Never scales DOWN — prevents jitter.
        """
        data_max = float(np.max(spectrum))

        # ★ ถ้า data เกิน y_max ปัจจุบัน → step up
        if data_max > self._y_max:
            for ceiling, y_max in self._BIT_CEILINGS:
                if data_max <= ceiling * 1.05:
                    self._y_max = y_max
                    return
            # fallback: 16-bit
            self._y_max = 65535.0

        # ★ ถ้า y_max สูงเกิน data มาก → snap down ให้พอดี
        # (เช่น data=200 แต่ y_max=65535 → ปรับลงเป็น 255)
        if data_max < self._y_max * 0.1:
            for ceiling, y_max in self._BIT_CEILINGS:
                if data_max <= ceiling * 1.05:
                    self._y_max = y_max
                    return

    # ── internal ──────────────────────────────────────────────────

    def _rebuild_polygon(self):
        self._polygon = QPolygonF()
        for _ in range(self._n_points):
            self._polygon.append(QPointF())

    def _rebuild_x_cache(self):
        plot_w = self.width() - self._ml - self._mr
        if plot_w < 1:
            plot_w = 1
        self._x_cache = np.linspace(
            float(self._ml),
            float(self._ml + plot_w),
            self._n_points,
        )

    def _update_polygon_points(self):
        if self._data is None or self._n_points == 0:
            return

        vmax = self._y_max
        plot_h = self.height() - self._mt - self._mb
        if plot_h < 1:
            plot_h = 1

        y = self._mt + plot_h - np.clip(self._data / vmax, 0.0, 1.0) * plot_h

        for i in range(self._n_points):
            self._polygon[i] = QPointF(float(self._x_cache[i]), float(y[i]))

    def resizeEvent(self, event):
        if self._n_points > 0:
            self._rebuild_x_cache()
            if self._data is not None:
                self._update_polygon_points()
        super().resizeEvent(event)

    # ── paint ─────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        p.fillRect(self.rect(), QColor(15, 15, 15))

        plot_w = self.width() - self._ml - self._mr
        plot_h = self.height() - self._mt - self._mb

        if plot_w < 1 or plot_h < 1:
            p.end()
            return

        dx_major = plot_w / self._major_x
        dy_major = plot_h / self._major_y

        # minor grid
        p.setPen(QPen(QColor(35, 35, 35)))
        for major in range(self._major_x):
            x0 = self._ml + major * dx_major
            for k in range(1, self._minor_x):
                x = x0 + k * dx_major / self._minor_x
                p.drawLine(int(x), self._mt, int(x), self._mt + plot_h)
        for major in range(self._major_y):
            y0 = self._mt + plot_h - major * dy_major
            for k in range(1, self._minor_y):
                y = y0 - k * dy_major / self._minor_y
                p.drawLine(self._ml, int(y), self._ml + plot_w, int(y))

        # major grid
        p.setPen(QPen(QColor(60, 60, 60)))
        for i in range(self._major_x + 1):
            x = self._ml + i * dx_major
            p.drawLine(int(x), self._mt, int(x), self._mt + plot_h)
        for i in range(self._major_y + 1):
            y = self._mt + plot_h - i * dy_major
            p.drawLine(self._ml, int(y), self._ml + plot_w, int(y))

        # axis
        p.setPen(QPen(QColor(220, 220, 220)))
        p.drawLine(self._ml, self._mt, self._ml, self._mt + plot_h)
        p.drawLine(self._ml, self._mt + plot_h, self._ml + plot_w, self._mt + plot_h)

        # X ticks
        for i in range(self._major_x + 1):
            x = self._ml + i * dx_major
            wl = self._wl_min + (self._wl_max - self._wl_min) * i / self._major_x
            p.drawText(int(x) - 15, self._mt + plot_h + 18, f"{wl:.0f}")

        # Y ticks — right-aligned
        vmax = self._y_max
        metrics = p.fontMetrics()
        for i in range(self._major_y + 1):
            y = self._mt + plot_h - i * dy_major
            value = vmax * i / self._major_y
            text = f"{value:.0f}"
            tw = metrics.horizontalAdvance(text)
            p.drawText(int(self._ml) - tw - 6, int(y) + 4, text)

        # X label
        p.drawText(self._ml + plot_w // 2 - 50, self.height() - 8, self._xlabel)

        # Y label
        p.save()
        p.translate(12, self._mt + plot_h // 2)
        p.rotate(-90)
        tw = metrics.horizontalAdvance("Intensity")
        p.drawText(-tw // 2, 0, "Intensity")
        p.restore()

        # spectrum line
        if self._data is not None and self._n_points > 1:
            p.setPen(QPen(QColor(79, 195, 247), 1.6))
            p.drawPolyline(self._polygon)

        p.end()


# ── SpectrumPlot ─────────────────────────────────────────────────────
class SpectrumPlot(QWidget):
    """Static ROI / cube spectrum with a movable band marker.

    API
    ---
    clear()
    plot(spectrum, wavelengths=None, current_band=None)
    """

    def __init__(self):
        super().__init__()

        self.pw = pg.PlotWidget()
        self.pw.setMinimumHeight(160)
        self.pw.showGrid(x=True, y=True, alpha=_GRID_ALPHA)
        self.pw.setLabel("bottom", "Wavelength (nm)", **_AXIS_LABEL_STYLE)
        self.pw.setLabel("left", "Mean intensity", **_AXIS_LABEL_STYLE)
        self.pw.setTitle("ROI Spectrum", **_TITLE_STYLE)
        _apply_axis_fonts(self.pw)
        plot_item = self.pw.getPlotItem()
        plot_item.layout.setRowMinimumHeight(3, 45)

        self.curve = self.pw.plot(
            pen=_PEN_ROI,
            autoDownsample=True,
            clipToView=True,
        )

        self.band_line = pg.InfiniteLine(
            angle=90, movable=False, pen=_PEN_MARKER,
        )
        self.band_line.setVisible(False)
        self.pw.addItem(self.band_line)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.pw)
        self.setLayout(layout)

    def clear(self):
        self.curve.clear()
        self.band_line.setVisible(False)

    def plot(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray | None = None,
        current_band: int | None = None,
    ):
        if spectrum is None:
            return

        spectrum = np.asarray(spectrum, dtype=np.float32)
        if spectrum.size == 0:
            return

        marker_pos: float | None = None

        use_wl = (
            wavelengths is not None
            and np.asarray(wavelengths).shape[0] == spectrum.shape[0]
        )

        if use_wl:
            x = np.asarray(wavelengths, dtype=np.float32)
            xlabel = "Wavelength (nm)"
            if current_band is not None and 0 <= current_band < len(x):
                marker_pos = float(x[current_band])
        else:
            x = np.arange(spectrum.shape[0], dtype=np.float32)
            xlabel = "Band"
            if current_band is not None and 0 <= current_band < len(x):
                marker_pos = float(current_band)

        self.curve.setData(x, spectrum, connect="finite")

        if marker_pos is not None:
            self.band_line.setPos(marker_pos)
            self.band_line.setVisible(True)
        else:
            self.band_line.setVisible(False)

        self.pw.setLabel("bottom", xlabel, **_AXIS_LABEL_STYLE)
        self.pw.autoRange(padding=0.04)
