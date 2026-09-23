# widgets/cube_preview.py
"""Realtime cube preview with QLabel (fit+center) — no OpenGL dependency."""

from __future__ import annotations

import logging

import numpy as np
from matplotlib import colormaps

from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
)

from workers.cube_render import CubeRenderWorker
from widgets.range_slider import RangeSlider
from widgets.zoom_pan import ZoomPanMixin
logger = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.WARNING)

RGB_TARGET_WAVELENGTHS_VIS = (645.0, 555.0, 465.0)
RGB_TARGET_WAVELENGTHS_SWIR = (1500.0, 1300.0, 1100.0)


def find_closest_band(wavelengths: np.ndarray, target_nm: float) -> int:
    return int(np.argmin(np.abs(wavelengths - target_nm)))


# ══════════════════════════════════════════════════════════════
#  tab20 colormap — shared between label dots + spectrum lines
# ══════════════════════════════════════════════════════════════

_TAB20_COUNT = 20
_tab20_cmap = colormaps["tab20"]
_RADIUS = 2
TAB20_HEX: list[str] = []
TAB20_QCOLORS: list[QColor] = []

for _i in range(_TAB20_COUNT):
    r, g, b, a = _tab20_cmap(_i / _TAB20_COUNT)
    TAB20_HEX.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    TAB20_QCOLORS.append(QColor(int(r * 255), int(g * 255), int(b * 255)))


# ══════════════════════════════════════════════════════════════
#  Centered Pixmap Label (fit + center + dot markers)
# ══════════════════════════════════════════════════════════════

# ★ แทนที่ CenteredPixmapLabel class เดิมใน cube_preview.py

class CenteredPixmapLabel(QLabel, ZoomPanMixin):
    """QLabel with fit/center pixmap, click dots, and zoom/pan support.

    Emits:
        clicked(int, int)  — image pixel coords on left click
        zoom_changed(float) — zoom percent
    """

    clicked = pyqtSignal(int, int)
    zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_zoom_pan()

        self._pixmap: QPixmap | None = None
        self._click_points: list[tuple[int, int]] = []
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(1, 1)
        self.setStyleSheet("background-color: #0d0d0d;")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # ── ZoomPanMixin overrides ────────────────────────────

    def _zp_image_size(self) -> tuple[int, int]:
        if self._pixmap is None or self._pixmap.isNull():
            return 0, 0
        return self._pixmap.width(), self._pixmap.height()

    def _zp_on_zoom_changed(self):
        self.zoom_changed.emit(self.zoom_percent)

    # ── public API ────────────────────────────────────────

    def setPixmapScaled(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self.update()

    def getPixmap(self) -> QPixmap | None:
        return self._pixmap

    def add_click_point(self, ix: int, iy: int):
        self._click_points.append((ix, iy))
        self.update()

    def clear_click_points(self):
        self._click_points.clear()
        self.update()

    # ── painting ──────────────────────────────────────────

    def paintEvent(self, event):
        if self._pixmap is None or self._pixmap.isNull():
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(13, 13, 13))
            painter.end()
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(13, 13, 13))

        # ★ zoom/pan transform
        ox, oy, sx, sy = self._zp_transform()

        img_w = self._pixmap.width()
        img_h = self._pixmap.height()
        scaled_w = int(img_w * sx)
        scaled_h = int(img_h * sy)

        if scaled_w > 0 and scaled_h > 0:
            scaled = self._pixmap.scaled(
                scaled_w, scaled_h,
                Qt.IgnoreAspectRatio, Qt.SmoothTransformation,
            )
            painter.drawPixmap(int(ox), int(oy), scaled)

        # draw click dots + labels
        for i, (ix, iy) in enumerate(self._click_points):
            color = TAB20_QCOLORS[i % len(TAB20_QCOLORS)]
            cx, cy = self._zp_map_image_to_widget(ix, iy)

            if cx < -20 or cy < -20 or cx > self.width() + 20 or cy > self.height() + 20:
                continue

            radius = 3
            painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QRectF(
                cx - radius, cy - radius,
                2 * radius, 2 * radius,
            ))

            label = f"({ix},{iy})"
            font = painter.font()
            font.setPixelSize(10)
            painter.setFont(font)
            metrics = painter.fontMetrics()

            tw = metrics.horizontalAdvance(label) + 8
            th = metrics.height() + 4
            lx = cx + radius + 4
            ly = cy - th // 2

            if lx + tw > self.width():
                lx = cx - radius - 4 - tw
            if ly < 0:
                ly = 0

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 160)))
            painter.drawRoundedRect(QRectF(lx, ly, tw, th), 3, 3)

            painter.setPen(QPen(QColor(255, 255, 255, 230)))
            painter.drawText(QRectF(lx, ly, tw, th), Qt.AlignCenter, label)

        painter.end()

    # ── coordinate mapping (zoom-aware) ───────────────────

    def map_widget_to_image(self, wx: int, wy: int, img_w: int, img_h: int):
        """Map widget coords to image coords — zoom/pan aware."""
        return self._zp_map_widget_to_image(wx, wy)

    # ── mouse events ──────────────────────────────────────

    def wheelEvent(self, event):
        self._zp_wheel_event(event)

    def mouseDoubleClickEvent(self, event):
        if self._zp_mouse_double_click(event):
            return

    def mousePressEvent(self, event):
        # middle button → pan
        if self._zp_mouse_press(event):
            return

        # left button → emit clicked signal
        if event.button() == Qt.LeftButton:
            ix, iy = self._zp_map_widget_to_image(event.x(), event.y())
            if ix is not None and iy is not None:
                self.clicked.emit(ix, iy)

    def mouseMoveEvent(self, event):
        if self._zp_mouse_move(event):
            return

    def mouseReleaseEvent(self, event):
        if self._zp_mouse_release(event):
            return


# ══════════════════════════════════════════════════════════════
#  Realtime Cube Preview Widget (scan mode — frame by frame)
# ══════════════════════════════════════════════════════════════

class RealtimeCubePreviewWidget(QWidget):
    """RGB composite viewer with 1-click-1-spectrum + tab20 legend.

    Layout::

        ┌──────────────────────────────────────────────┐
        │                                              │
        │        CenteredPixmapLabel (RGB)             │
        │           ● (x,y) dot markers                │
        │                                              │
        ├──────────────────────────────────────────────┤
        │  ──●══════════════════════════●── 2.0–98.0%  │
        ├──────────────────────────────────────────────┤
        │  Frames: 1029  │ Point 2: x=120 y=67 (2)    │
        │  [Clear Spectra]                               │
        ├──────────────────────────────────────────────┤
        │  Legend: ── (45,23)  ── (120,67)             │
        │  Spectrum plot                               │
        │   465.1nm  555.3nm     645.2nm               │
        └──────────────────────────────────────────────┘
    """

    def __init__(
        self,
        rgb_target_wavelengths: tuple[float, float, float] = (645.0, 555.0, 465.0),
    ):
        super().__init__()

        self._bands: int = 0
        self._wavelengths: np.ndarray | None = None
        self._current_shape: tuple[int, int] = (0, 0)
        self._rgb_targets = rgb_target_wavelengths
        self._rgb_band_indices: list[int] = [0, 0, 0]
        self._suppress_line_emit = False
        self._raw_rgb8: np.ndarray | None = None
        self._pending_click: tuple[int, int] | None = None

        self._spectrum_history: list[tuple[int, int, np.ndarray]] = []
        self._saturation_mask: np.ndarray | None = None  # ★ (n_frames, H)

        import pyqtgraph as pg
        self._pg = pg

        label_style = {"color": "#bdbdbd", "font-size": "9px"}
        title_style = {"color": "#e0e0e0", "font-size": "10px"}

        # worker thread
        self._worker = CubeRenderWorker()
        self._worker.rgb_ready.connect(self._on_rgb_ready)
        self._worker.frames_count_ready.connect(self._on_frames_count)
        self._worker.spectrum_ready.connect(self._on_spectrum_ready)
        self._worker.start()

        # image label
        self.image_label = CenteredPixmapLabel()
        self.image_label.setMinimumHeight(150)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.clicked.connect(self._on_image_clicked)

        # stretch controls
        self._stretch_label = QLabel("2.0% \u2013 98.0%")
        self._stretch_label.setStyleSheet("font-size: 10px; color: #9e9e9e;")
        self._stretch_label.setFixedWidth(90)

        self._stretch_slider = RangeSlider(low=0.02, high=0.98)
        self._stretch_slider.range_changed.connect(self._on_stretch_changed)

        stretch_row = QHBoxLayout()
        stretch_row.setContentsMargins(0, 0, 0, 0)
        stretch_row.setSpacing(6)
        stretch_row.addWidget(self._stretch_slider, stretch=1)
        stretch_row.addWidget(self._stretch_label)

        # info labels
        self.frame_label = QLabel("Frames: 0")
        self.pixel_label = QLabel("Click image for spectrum")
        self.pixel_label.setStyleSheet("font-size: 11px; color: #9e9e9e;")

        info_row = QHBoxLayout()
        info_row.addWidget(self.frame_label)
        info_row.addStretch()
        info_row.addWidget(self.pixel_label)

        # clear button
        self._btn_clear = QPushButton("Clear Spectra")
        self._btn_clear.setFixedHeight(24)
        self._btn_clear.clicked.connect(self._clear_spectrum_history)

        # spectrum plot
        self.spec_plot = pg.PlotWidget()
        self.spec_plot.setMinimumHeight(100)
        self.spec_plot.setMaximumHeight(200)
        self.spec_plot.showGrid(x=True, y=True, alpha=0.15)
        self.spec_plot.setTitle("Pixel spectrum", **title_style)
        self.spec_plot.setLabel("bottom", "Wavelength (nm)", **label_style)
        self.spec_plot.setLabel("left", "Intensity", **label_style)
        self.spec_plot.addLegend()

        for axis_name in ("bottom", "left"):
            axis = self.spec_plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(color="#616161"))
            axis.setTextPen(pg.mkPen(color="#bdbdbd"))

        # draggable R/G/B band lines
        channels = [
            ("R", "#ef5350", pg.mkPen(color="#ef5350", width=2.0)),
            ("G", "#66bb6a", pg.mkPen(color="#66bb6a", width=2.0)),
            ("B", "#42a5f5", pg.mkPen(color="#42a5f5", width=2.0)),
        ]

        self._rgb_lines: list = []
        for i, (_label, _color, _pen) in enumerate(channels):
            line = pg.InfiniteLine(
                pos=self._rgb_targets[i],
                angle=90, movable=True, pen=_pen,
                label="{value:.1f} nm",
                labelOpts={
                    "position": 0.08, "color": _color,
                    "fill": pg.mkBrush(30, 30, 30, 200),
                    "movable": True,
                },
            )
            line.sigPositionChanged.connect(self._on_line_moved)
            line.setVisible(True)
            self.spec_plot.addItem(line)
            self._rgb_lines.append(line)

        # main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        main_layout.addWidget(self.image_label, stretch=1)
        main_layout.addLayout(stretch_row)
        main_layout.addLayout(info_row)
        main_layout.addWidget(self._btn_clear)
        main_layout.addWidget(self.spec_plot)
        self.setLayout(main_layout)

    # ── public API ────────────────────────────────────────

    def reset(self):
        self._worker.reset()
        self._bands = 0
        self._wavelengths = None
        self._current_shape = (0, 0)
        self._raw_rgb8 = None
        self._pending_click = None
        self._spectrum_history.clear()
        self._saturation_mask = None  # ★
        self.frame_label.setText("Frames: 0")
        self.pixel_label.setText("Click image for spectrum")
        self.spec_plot.clear()
        self.spec_plot.addLegend()
        for i, line in enumerate(self._rgb_lines):
            line.setValue(self._rgb_targets[i])
            line.setVisible(False)
            self.spec_plot.addItem(line)
        self.image_label.clear_click_points()
        self.image_label.setPixmapScaled(QPixmap())

    def push_frame(self, frame: np.ndarray, wavelengths: np.ndarray | None, bands: int):
        if self._bands != bands and bands > 0:
            self._bands = bands
            self._wavelengths = wavelengths
            self._worker.set_wavelengths(wavelengths)
            self._init_band_lines(bands, wavelengths)
        self._worker.push_frame(frame)

    def cleanup(self):
        self._worker.stop()

    # ── band line initialization ──────────────────────────

    def _init_band_lines(self, bands: int, wavelengths: np.ndarray | None):
        self._suppress_line_emit = True
        if wavelengths is not None and len(wavelengths) == bands:
            wl = np.asarray(wavelengths, dtype=np.float32)
            for i, target_nm in enumerate(self._rgb_targets):
                bi = find_closest_band(wl, target_nm)
                self._rgb_band_indices[i] = bi
                self._rgb_lines[i].setValue(float(wl[bi]))
                self._rgb_lines[i].setVisible(True)
            self.spec_plot.setXRange(float(wl[0]), float(wl[-1]), padding=0.02)
        else:
            for i, frac in enumerate([0.25, 0.50, 0.75]):
                bi = int(bands * frac)
                self._rgb_band_indices[i] = bi
                self._rgb_lines[i].setValue(float(bi))
                self._rgb_lines[i].setVisible(True)
            self.spec_plot.setXRange(0, bands, padding=0.02)
        self._suppress_line_emit = False
        self._push_rgb_bands()

    # ── line drag handler ─────────────────────────────────

    def _on_line_moved(self):
        if self._suppress_line_emit:
            return
        for i, line in enumerate(self._rgb_lines):
            raw_x = line.value()
            if self._wavelengths is not None and len(self._wavelengths) == self._bands:
                wl = np.asarray(self._wavelengths, dtype=np.float32)
                bi = int(np.argmin(np.abs(wl - raw_x)))
                snapped = float(wl[bi])
            else:
                bi = max(0, min(self._bands - 1, int(round(raw_x))))
                snapped = float(bi)
            if self._rgb_band_indices[i] != bi:
                self._rgb_band_indices[i] = bi
                self._suppress_line_emit = True
                line.setValue(snapped)
                self._suppress_line_emit = False
                self._push_rgb_bands()
                return

    def _push_rgb_bands(self):
        self._worker.set_rgb_bands(*self._rgb_band_indices)

    # ── spectrum history ──────────────────────────────────

    def _clear_spectrum_history(self):
        self._spectrum_history.clear()
        self.image_label.clear_click_points()
        self._redraw_all_spectra()

    def _redraw_all_spectra(self):
        saved_lines = []
        for line in self._rgb_lines:
            saved_lines.append((line.value(), line.isVisible()))

        self.spec_plot.clear()
        self.spec_plot.addLegend()

        for i, (val, vis) in enumerate(saved_lines):
            color = ["#ef5350", "#66bb6a", "#42a5f5"][i]
            self._rgb_lines[i] = self._pg.InfiniteLine(
                pos=val, angle=90, movable=True,
                pen=self._pg.mkPen(color=color, width=2.0),
                label="{value:.1f} nm",
                labelOpts={
                    "position": 0.08, "color": color,
                    "fill": self._pg.mkBrush(30, 30, 30, 200),
                    "movable": True,
                },
            )
            self._rgb_lines[i].sigPositionChanged.connect(self._on_line_moved)
            self._rgb_lines[i].setVisible(vis)
            self.spec_plot.addItem(self._rgb_lines[i])

        for i, (px, py, spectrum) in enumerate(self._spectrum_history):
            color = TAB20_HEX[i % len(TAB20_HEX)]
            pen = self._pg.mkPen(color=color, width=1.5)

            if self._wavelengths is not None and len(self._wavelengths) == len(spectrum):
                x_data = self._wavelengths
            else:
                x_data = np.arange(len(spectrum), dtype=np.float32)

            self.spec_plot.plot(
                x_data, spectrum, pen=pen,
                name=f"({px},{py})", connect="finite",
            )

        if self._spectrum_history:
            all_spectra = np.stack([s for _, _, s in self._spectrum_history])
            ymin = float(all_spectra.min())
            ymax = float(all_spectra.max())
            if ymax <= ymin:
                ymax = ymin + 1.0
            margin = (ymax - ymin) * 0.08 + 1e-6
            self.spec_plot.setYRange(ymin - margin, ymax + margin, padding=0)

    # ── stretch ───────────────────────────────────────────

    @staticmethod
    def _sample_for_percentile(band: np.ndarray, max_samples: int = 200_000) -> np.ndarray:
        """Return a bounded-size sample of `band` for percentile estimation.

        As the pushbroom scan grows, the accumulated image gets wider every
        frame — computing np.percentile on the full array on every new
        frame would make per-frame cost grow with scan length (O(N) per
        frame, O(N^2) total). A strided sample keeps the per-frame cost
        roughly constant regardless of how long the scan has run, while
        still giving a good percentile estimate for display stretching.
        """
        flat = band.ravel()
        if flat.size <= max_samples:
            return flat
        step = flat.size // max_samples
        return flat[::step]

    def _on_stretch_changed(self, low: float, high: float):
        self._stretch_label.setText(f"{low*100:.1f}% \u2013 {high*100:.1f}%")
        if self._raw_rgb8 is None:
            return
        rgb8 = self._raw_rgb8
        stretched = np.empty_like(rgb8)
        for ch in range(3):
            band = rgb8[:, :, ch].astype(np.float32)
            sample = self._sample_for_percentile(band)
            lo = np.percentile(sample, low * 100)
            hi = np.percentile(sample, high * 100)
            if hi <= lo:
                hi = lo + 1.0
            out = (band - lo) * (255.0 / (hi - lo))
            stretched[:, :, ch] = np.clip(out, 0, 255).astype(np.uint8)
        self._display_rgb(stretched)

    def set_saturation_mask(self, mask: np.ndarray | None):
        """★ Set saturation overlay. mask shape = (n_frames, H). True = saturated."""
        self._saturation_mask = mask
        if mask is not None:
            logger.info(
                "Saturation mask: shape=%s dtype=%s any=%s count=%d",
                mask.shape, mask.dtype, mask.any(), np.count_nonzero(mask),
            )
        else:
            logger.info("Saturation mask: None")
        if self._raw_rgb8 is not None:
            logger.info(
                "RGB image: shape=%s", self._raw_rgb8.shape,
            )
            self._on_stretch_changed(
                self._stretch_slider.low, self._stretch_slider.high,
            )

    def _display_rgb(self, rgb8: np.ndarray):
        display = rgb8

        # ★ saturation overlay — white (clipped = no data)
        if self._saturation_mask is not None:
            h, w = rgb8.shape[:2]
            mask_t = self._saturation_mask.T  # (552, N)
            sh, sw = mask_t.shape
            if sh == h and sw >= w:
                display = rgb8.copy()
                mask_2d = mask_t[:, :w]  # (h, w)
                display[mask_2d] = 255   # ★ ขาวโพลนทุก channel

        h, w, ch = display.shape
        fmt = QImage.Format_RGB888 if ch == 3 else QImage.Format_RGBA8888
        bytes_per_line = ch * w
        qimg = QImage(display.data, w, h, bytes_per_line, fmt).copy()
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmapScaled(pixmap)

    # ── slots from worker ─────────────────────────────────

    def _on_rgb_ready(self, rgb: np.ndarray, ih: int, iw: int):
        self._current_shape = (ih, iw)
        if rgb.dtype == np.float64 or (rgb.size > 0 and rgb.max() <= 1.0):
            rgb8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        else:
            rgb8 = np.clip(rgb, 0, 255).astype(np.uint8)
        self._raw_rgb8 = rgb8.copy()
        self._on_stretch_changed(
            self._stretch_slider.low, self._stretch_slider.high,
        )

    def _on_frames_count(self, n: int):
        self.frame_label.setText(f"Frames: {n}")

    def _on_spectrum_ready(self, x_data, spectrum, img_x, img_y):
        self.pixel_label.setText(f"Pixel  x={img_x}  y={img_y}")

    # ── 1-click spectrum ──────────────────────────────────

    def _on_image_clicked(self, img_x: int, img_y: int):
        # img_x/img_y already come from CenteredPixmapLabel's own
        # left-click handling (zoom/pan-aware mapping, and it only emits
        # this signal when a middle-click pan isn't in progress).
        h, w = self._current_shape
        if h == 0 or w == 0:
            return

        self._pending_click = (img_x, img_y)

        try:
            self._worker.spectrum_ready.disconnect(self._on_spectrum_for_click)
        except TypeError:
            pass
        self._worker.spectrum_ready.connect(self._on_spectrum_for_click)
        self._worker.request_spectrum(img_x, img_y)

    def _on_spectrum_for_click(self, x_data, spectrum, img_x, img_y):
        # NOTE: img_x/img_y from the signal are the coords this particular
        # spectrum was actually computed for. self._pending_click is only
        # used as a "is a click still outstanding?" guard — using it for
        # the coordinates themselves is wrong if the user clicks a second
        # point before the first result comes back (pending_click would
        # already have been overwritten with the newer click).
        if self._pending_click is None:
            return

        self._pending_click = None

        try:
            self._worker.spectrum_ready.disconnect(self._on_spectrum_for_click)
        except TypeError:
            pass

        px, py = img_x, img_y
        self._spectrum_history.append((px, py, spectrum.copy()))
        self.image_label.add_click_point(px, py)

        idx = len(self._spectrum_history)
        self.pixel_label.setText(
            f"Point {idx}: x={px}  y={py}  ({len(self._spectrum_history)} spectra)"
        )
        self._redraw_all_spectra()