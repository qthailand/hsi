# widgets/cube_preview.py

from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
)

from workers.cube_render import CubeRenderWorker

logger = logging.getLogger("HSI")

# ── pen / style constants ──────────────────────────────────────────
_PEN_R = pg.mkPen(color="#ef5350", width=2.0)
_PEN_G = pg.mkPen(color="#66bb6a", width=2.0)
_PEN_B = pg.mkPen(color="#42a5f5", width=2.0)
_PEN_PIXEL = pg.mkPen(color="#29b6f6", width=1.0, style=Qt.DashLine)
_LABEL_STYLE = {"color": "#bdbdbd", "font-size": "9px"}
_TITLE_STYLE = {"color": "#e0e0e0", "font-size": "10px"}

_CHANNELS = [
    ("R", "#ef5350", _PEN_R),
    ("G", "#66bb6a", _PEN_G),
    ("B", "#42a5f5", _PEN_B),
]


def find_closest_band(
    wavelengths: np.ndarray,
    target_nm: float,
) -> int:
    return int(np.argmin(np.abs(wavelengths - target_nm)))


class RealtimeCubePreviewWidget(QWidget):
    """RGB composite viewer with 3 draggable band lines showing wavelength.

    Layout::

        ┌──────────────────────────────────────────────┐
        │  pg.ImageView (RGB composite + histogram)    │
        ├──────────────────────────────────────────────┤
        │  Frames: 1029  │  Click image for spectrum   │
        ├──────────────────────────────────────────────┤
        │  Spectrum plot                               │
        │      pixel spectrum (cyan dashed)            │
        │       │       │            │                  │
        │    465.1nm  555.3nm     645.2nm               │
        │    (red)    (green)     (blue) ← draggable    │
        └──────────────────────────────────────────────┘
    """

    def __init__(
        self,
        rgb_target_wavelengths: tuple[float, float, float] = (
            645.0, 555.0, 465.0,
        ),
    ):
        super().__init__()

        self._bands: int = 0
        self._wavelengths: np.ndarray | None = None
        self._first_image: bool = True
        self._current_shape: tuple[int, int] = (0, 0)
        self._rgb_targets = rgb_target_wavelengths
        self._rgb_band_indices: list[int] = [0, 0, 0]
        self._suppress_line_emit = False

        # ── worker thread ─────────────────────────────────────────
        self._worker = CubeRenderWorker()
        self._worker.rgb_ready.connect(self._on_rgb_ready)
        self._worker.frames_count_ready.connect(self._on_frames_count)
        self._worker.spectrum_ready.connect(self._on_spectrum_ready)
        self._worker.start()

        # ── pg.ImageView ──────────────────────────────────────────
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.setMinimumHeight(200)

        self.image_view.getImageItem().scene().sigMouseClicked.connect(
            self._on_image_clicked
        )

        # ── info labels ───────────────────────────────────────────
        self.frame_label = QLabel("Frames: 0")
        self.pixel_label = QLabel("Click image to see spectrum")
        self.pixel_label.setStyleSheet("font-size: 11px; color: #9e9e9e;")

        info_row = QHBoxLayout()
        info_row.addWidget(self.frame_label)
        info_row.addStretch()
        info_row.addWidget(self.pixel_label)

        # ── spectrum plot with draggable R/G/B lines ──────────────
        self.spec_plot = pg.PlotWidget()
        self.spec_plot.setMinimumHeight(120)
        self.spec_plot.setMaximumHeight(180)
        self.spec_plot.showGrid(x=True, y=True, alpha=0.15)
        self.spec_plot.setTitle("Pixel spectrum", **_TITLE_STYLE)
        self.spec_plot.setLabel("bottom", "Wavelength (nm)", **_LABEL_STYLE)
        self.spec_plot.setLabel("left", "Intensity", **_LABEL_STYLE)

        for axis_name in ("bottom", "left"):
            axis = self.spec_plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(color="#616161"))
            axis.setTextPen(pg.mkPen(color="#bdbdbd"))

        self.spec_curve = self.spec_plot.plot(pen=_PEN_PIXEL)

        # R, G, B draggable lines — label shows wavelength
        self._rgb_lines: list[pg.InfiniteLine] = []

        for i, (_label, _color, _pen) in enumerate(_CHANNELS):
            wavelength_default = self._rgb_targets[i]
            line = pg.InfiniteLine(
                pos=self._rgb_targets[i],      # ← ใช้ target จาก config
                angle=90,
                movable=True,
                pen=_pen,
                label="{value:.1f} nm",
                labelOpts={
                    "position": 0.08,
                    "color": _color,
                    "fill": pg.mkBrush(30, 30, 30, 200),
                    "movable": True,
                },
            )
            line.sigPositionChanged.connect(self._on_line_moved)
            line.setVisible(True)               # ← แสดงเลย ไม่ต้องรอ frame
            self.spec_plot.addItem(line)
            self._rgb_lines.append(line)

        # ── layout ────────────────────────────────────────────────
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self.image_view, stretch=1)
        layout.addLayout(info_row)
        layout.addWidget(self.spec_plot)
        self.setLayout(layout)

        # self.setVisible(False)

    # ── public API ────────────────────────────────────────────────

    def reset(self):
        self._worker.reset()
        self._bands = 0
        self._wavelengths = None
        self._first_image = True
        self._current_shape = (0, 0)
        self.frame_label.setText("Frames: 0")
        self.pixel_label.setText("Click image to see spectrum")
        self.spec_curve.clear()
        for i, line in enumerate(self._rgb_lines):
            line.setValue(self._rgb_targets[i])  # ← คืนค่า default ที่มีความหมาย
            line.setVisible(False)

    def push_frame(
        self,
        frame: np.ndarray,
        wavelengths: np.ndarray | None,
        bands: int,
    ):
        if self._bands != bands and bands > 0:
            self._bands = bands
            self._wavelengths = wavelengths
            self._worker.set_wavelengths(wavelengths)
            self._init_band_lines(bands, wavelengths)
        self._worker.push_frame(frame)

    def cleanup(self):
        self._worker.stop()

    # ── band line initialization ──────────────────────────────────

    def _init_band_lines(self, bands: int, wavelengths: np.ndarray | None):
        """Place R/G/B lines at target wavelengths."""
        self._suppress_line_emit = True

        if wavelengths is not None and len(wavelengths) == bands:
            wl = np.asarray(wavelengths, dtype=np.float32)
            for i, target_nm in enumerate(self._rgb_targets):
                bi = find_closest_band(wl, target_nm)
                self._rgb_band_indices[i] = bi
                self._rgb_lines[i].setValue(float(wl[bi]))
                self._rgb_lines[i].setVisible(True)

            self.spec_plot.setXRange(
                float(wl[0]), float(wl[-1]), padding=0.02,
            )
        else:
            for i, frac in enumerate([0.25, 0.50, 0.75]):
                bi = int(bands * frac)
                self._rgb_band_indices[i] = bi
                self._rgb_lines[i].setValue(float(bi))
                self._rgb_lines[i].setVisible(True)

            self.spec_plot.setXRange(0, bands, padding=0.02)

        self._suppress_line_emit = False
        self._push_rgb_bands()

    # ── line drag handler ─────────────────────────────────────────

    def _on_line_moved(self):
        if self._suppress_line_emit:
            return

        for i, line in enumerate(self._rgb_lines):
            raw_x = line.value()

            if (
                self._wavelengths is not None
                and len(self._wavelengths) == self._bands
            ):
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

    # ── slots from worker ─────────────────────────────────────────

    def _on_rgb_ready(self, rgb: np.ndarray, ih: int, iw: int):
        self._current_shape = (ih, iw)

        # ★ คำนวณ levels จากข้อมูลจริงทุกครั้ง
        if rgb.dtype == np.uint16:
            lo, hi = 0, 65535
        elif rgb.max() <= 1.0:
            lo, hi = 0.0, 1.0          # RGB ถูก normalize แล้ว (0-1)
        else:
            lo = float(np.percentile(rgb, 1))
            hi = float(np.percentile(rgb, 99))

        self.image_view.setImage(
            rgb,
            autoLevels=False,
            autoRange=self._first_image,
            levels=(lo, hi),            # ★ บังคับ set range
        )
        self._first_image = False

    def _on_frames_count(self, n: int):
        self.frame_label.setText(f"Frames: {n}")

    def _on_spectrum_ready(self, x_data, spectrum, img_x, img_y):
        self.pixel_label.setText(f"Pixel  x={img_x}  y={img_y}")

        use_wl = (
            self._wavelengths is not None
            and len(self._wavelengths) == len(spectrum)
        )
        xlabel = "Wavelength (nm)" if use_wl else "Band"

        self.spec_curve.setData(x_data, spectrum, connect="finite")
        self.spec_plot.setLabel("bottom", xlabel, **_LABEL_STYLE)

        ymin = float(np.min(spectrum))
        ymax = float(np.max(spectrum))
        if ymax <= ymin:
            ymax = ymin + 1.0
        margin = (ymax - ymin) * 0.08 + 1e-6
        self.spec_plot.setYRange(ymin - margin, ymax + margin, padding=0)

    # ── click-to-spectrum ─────────────────────────────────────────

    def _on_image_clicked(self, event):
        if event.button() != Qt.LeftButton:
            return

        img_item = self.image_view.getImageItem()
        pos = img_item.mapFromScene(event.scenePos())
        img_x = int(pos.x())
        img_y = int(pos.y())

        h, w = self._current_shape
        if 0 <= img_x < w and 0 <= img_y < h:
            self._worker.request_spectrum(img_x, img_y)
