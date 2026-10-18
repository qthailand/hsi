# widgets/spectrum.py

"""Spectrum plotting widgets: a fast live line spectrum and a static ROI/cube spectrum.

Data conventions:
    A spectrum is a 1-D array over wavelength bands (intensity in counts).  The
    optional ``wavelengths`` array is in nm, same length, ascending; without it
    the x axis is the band index.

Threading:
    GUI thread only.  ``SpectrumPlot`` uses pyqtgraph.

Example:
    >>> import numpy as np
    >>> from modules.live_view import LiveLineSpectrumWidget, SpectrumPlot   # doctest: +SKIP
    >>> live = LiveLineSpectrumWidget()                                      # doctest: +SKIP
    >>> live.update_spectrum(np.random.rand(300) * 4000, np.linspace(400, 1000, 300))  # doctest: +SKIP
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from .saturation import saturated_mask

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
    """High-performance live spectrum drawn with QPainter (no pyqtgraph).

    The Y axis starts at 16-bit full scale (65535) and, unless a bit depth was
    fixed with :meth:`set_bit_depth`, jumps UP to the next bit-depth ceiling
    (255, 1023, 4095, 65535) when the data exceeds it, and snaps down only when the
    data stays below 10 % of the current maximum.  Takes no constructor arguments.

    With a saturation ``ceiling`` (see :meth:`update_spectrum`) the level a clipped
    pixel can reach is drawn as a dashed red line, saturated points are drawn red and
    the number of saturated bands is shown, because after dark subtraction a clipped
    pixel sits at ``full scale - dark``, well below the top of the axis.
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

        # saturation ceiling (same length as the data) and its drawing state
        self._ceiling: np.ndarray | None = None
        self._ceiling_polygon = QPolygonF()
        self._sat_runs: list[tuple[int, int]] = []
        self._n_saturated: int = 0

        self.setMinimumHeight(220)
        self.setMinimumWidth(200)

    # ── public API ────────────────────────────────────────────────

    def set_bit_depth(self, bit_depth: int):
        """Lock the Y axis to a sensor bit depth (disables auto-scaling).

        Args:
            bit_depth: Bits; the axis then runs ``0 .. 2**bit_depth - 1``.
        """
        self._manual_bit_depth = int(bit_depth)
        self._y_max = float(2 ** int(bit_depth) - 1)
        if self._data is not None and self._n_points > 0:
            self._update_polygon_points()
            self.update()

    def clear(self):
        """Remove the spectrum and reset the Y range to 16-bit full scale."""
        self._data = None
        self._n_points = 0
        self._polygon.clear()
        self._x_cache = np.zeros(0)
        self._ceiling = None
        self._ceiling_polygon.clear()
        self._sat_runs = []
        self._n_saturated = 0
        # ★ reset Y to 16-bit default
        self._y_max = 65535.0
        self.update()

    def update_spectrum(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray | None = None,
        ceiling: np.ndarray | None = None,
    ):
        """Show a new spectrum.

        Args:
            spectrum: 1-D intensities (any numeric dtype; converted to float32 and
                flattened); ``None`` or empty arrays are ignored.
            wavelengths: Optional 1-D wavelengths in nm, same length as ``spectrum``.
                If omitted or the length differs the x axis is labelled "Band".
            ceiling: Optional 1-D saturation ceiling, same length as ``spectrum``: the
                value a clipped pixel shows at each band. Points at >= 99 % of it are
                drawn red. Omitted or of another length: no saturation display.
        """
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
        self._set_ceiling(ceiling)
        self._update_polygon_points()
        self.update()

    def _set_ceiling(self, ceiling):
        """Store the ceiling and find the runs of saturated points of ``self._data``."""
        if ceiling is not None:
            ceiling = np.asarray(ceiling, dtype=np.float32).ravel()
            if ceiling.shape != self._data.shape:
                ceiling = None
        self._ceiling = ceiling
        if ceiling is None:
            self._sat_runs = []
            self._n_saturated = 0
            return
        mask = saturated_mask(self._data, ceiling)
        self._n_saturated = int(np.count_nonzero(mask))
        edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
        self._sat_runs = list(zip(edges[0::2].tolist(), edges[1::2].tolist()))

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

        self._ceiling_polygon = QPolygonF()
        if self._ceiling is not None:
            yc = self._mt + plot_h - np.clip(self._ceiling / vmax, 0.0, 1.0) * plot_h
            for i in range(self._n_points):
                self._ceiling_polygon.append(QPointF(float(self._x_cache[i]), float(yc[i])))

    def resizeEvent(self, event):
        """Qt handler: recompute the x positions and the polyline for the new size."""
        if self._n_points > 0:
            self._rebuild_x_cache()
            if self._data is not None:
                self._update_polygon_points()
        super().resizeEvent(event)

    # ── paint ─────────────────────────────────────────────────────

    def paintEvent(self, event):
        """Qt handler: draw grid, axes, tick labels and the spectrum polyline."""
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

        # saturation ceiling (dashed) under the spectrum
        if self._ceiling_polygon.size() > 1:
            p.setPen(QPen(QColor(239, 83, 80, 170), 1.0, Qt.DashLine))
            p.drawPolyline(self._ceiling_polygon)

        # spectrum line
        if self._data is not None and self._n_points > 1:
            p.setPen(QPen(QColor(79, 195, 247), 1.6))
            p.drawPolyline(self._polygon)

        # saturated points: red over the spectrum, plus a count
        if self._sat_runs and self._polygon.size() == self._n_points:
            p.setPen(QPen(QColor(255, 82, 82), 2.6))
            for start, stop in self._sat_runs:
                if stop - start > 1:
                    p.drawPolyline(self._polygon.mid(start, stop - start))
                else:
                    p.drawPoint(self._polygon[start])
            text = f"SATURATED {self._n_saturated} band" + ("s" if self._n_saturated != 1 else "")
            p.setPen(QPen(QColor(255, 82, 82)))
            tw = metrics.horizontalAdvance(text)
            p.drawText(self._ml + plot_w - tw - 6, self._mt + 14, text)

        p.end()


# ── SpectrumPlot ─────────────────────────────────────────────────────
class SpectrumPlot(QWidget):
    """Static ROI / cube spectrum (pyqtgraph) with a marker for the current band.

    API:
        ``clear()`` and ``plot(spectrum, wavelengths=None, current_band=None)``.
        The underlying ``pyqtgraph.PlotWidget`` is exposed as ``pw`` and the curve
        as ``curve`` for customisation.  Takes no constructor arguments.
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
        """Remove the curve and hide the band marker."""
        self.curve.clear()
        self.band_line.setVisible(False)

    def plot(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray | None = None,
        current_band: int | None = None,
    ):
        """Plot a spectrum and auto-range the view.

        Args:
            spectrum: 1-D intensities (converted to float32); ``None``/empty is ignored.
                NaN/inf values break the line instead of being drawn.
            wavelengths: Optional 1-D wavelengths in nm, same length as ``spectrum``.
                Otherwise the x axis is the band index.
            current_band: Optional band index at which a dashed vertical marker is
                drawn; hidden when ``None`` or out of range.
        """
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
