"""``SmileModelViewDialog``: look at the Smile (or Smile + keystone) model in use.

Shown when the scan window's *Smile* status card is clicked. Read from the model's maps only
(``map_x`` / ``map_y``: the corrected pixel ``(Y, X)`` reads the raw frame at
``(map_y, map_x)``), so a smile-only model and a joint model are drawn the same way.

Top: the distortion grid. Straight dotted rows and columns are the corrected image; the curves
are where it reads them in the raw frame: orange = rows (keystone + tilt), cyan = columns
(smile), green dashed = the crop kept (column margin, row margin). The model's stored target (or
lamp) frame is the background. *Exaggerate* scales the displacement so a small distortion shows.
Bottom: the column shift ``map_x - X`` (smile) and the row shift ``map_y - Y`` (keystone) as
images.

Threading:
    GUI thread only.

Example:
    >>> from modules.smile.model_view import SmileModelViewDialog          # doctest: +SKIP
    >>> SmileModelViewDialog(load_calibration(path), path).show()          # doctest: +SKIP
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

__all__ = ["SmileModelViewDialog", "grid_curves"]

#: Grid lines along each axis.
GRID_LINES = 13
#: Points per grid curve.
CURVE_POINTS = 240


def _nice_gain(value: float) -> float:
    """Largest of 1, 2, 5, 10, 20, 50, ... not above ``value`` (at least 1)."""
    gain = 1.0
    for step in (1, 2, 5, 10, 20, 50, 100, 200, 500):
        if step <= value:
            gain = float(step)
    return gain


def _sample(map_, ys, xs):
    """Bilinear sample of ``map_`` at float pixel positions ``(ys, xs)`` (clipped to the frame)."""
    h, w = map_.shape
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    y0 = np.minimum(np.floor(ys).astype(int), h - 2) if h > 1 else np.zeros_like(ys, int)
    x0 = np.minimum(np.floor(xs).astype(int), w - 2) if w > 1 else np.zeros_like(xs, int)
    fy, fx = ys - y0, xs - x0
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    return ((1 - fy) * (1 - fx) * map_[y0, x0] + (1 - fy) * fx * map_[y0, x1]
            + fy * (1 - fx) * map_[y1, x0] + fy * fx * map_[y1, x1])


def grid_curves(map_x, map_y, n: int = GRID_LINES, points: int = CURVE_POINTS):
    """Destination grid and where the remap reads it.

    Returns:
        ``(rows, cols, row_curves, col_curves)``: the grid row / column positions and, per line,
        ``(dst_x, dst_y, src_x, src_y)`` arrays of ``points`` samples.
    """
    h, w = map_x.shape
    rows = np.linspace(0.03 * (h - 1), 0.97 * (h - 1), n)
    cols = np.linspace(0.04 * (w - 1), 0.96 * (w - 1), n)
    xs = np.linspace(0, w - 1, points)
    ys = np.linspace(0, h - 1, points)
    row_curves = [(xs, np.full_like(xs, Y), _sample(map_x, np.full_like(xs, Y), xs),
                   _sample(map_y, np.full_like(xs, Y), xs)) for Y in rows]
    col_curves = [(np.full_like(ys, X), ys, _sample(map_x, ys, np.full_like(ys, X)),
                   _sample(map_y, ys, np.full_like(ys, X))) for X in cols]
    return rows, cols, row_curves, col_curves


class SmileModelViewDialog(QDialog):
    """Plots of one loaded ``SmileCalibration`` (see the module docstring).

    Args:
        calibration: The model (``map_x``, ``_get_map_y()``, ``margin``, ``row_margin``,
            ``frames``, ``summary()``).
        path: Its file, for the title (or ``""``).
        open_page: Called by the *Open Smile page* button, or ``None`` for no button.
        parent: Parent widget.
    """

    def __init__(self, calibration, path: str = "", open_page: Optional[Callable[[], None]] = None, parent=None):
        super().__init__(parent)
        self.calibration = calibration
        name = os.path.basename(path) if path else "Smile model"
        self.setWindowTitle(f"Smile model {name}")
        layout = QVBoxLayout(self)
        kind = "smile + keystone" if getattr(calibration, "has_keystone", False) else "smile only"
        h, w = (int(v) for v in calibration.image_shape)
        left, right = (int(v) for v in calibration.margin)
        top, bottom = (int(v) for v in calibration.row_margin)
        parts = [f"<b>{name}</b>", kind, f"{w}×{h} px", f"interp {calibration.interp}",
                 f"created {calibration.created}", f"crop: columns {left} / {right}, rows {top} / {bottom}"]
        if np.isfinite(getattr(calibration, "rms_before", np.nan)):
            parts.append(f"straightness {calibration.rms_before:.3f} → {calibration.rms_after:.3f} px")
        self.summary_label = QLabel(" · ".join(parts))
        self.summary_label.setWordWrap(True)
        self.summary_label.setToolTip(calibration.summary())
        layout.addWidget(self.summary_label)

        bar = QHBoxLayout()
        self.chk_exaggerate = QCheckBox("Exaggerate displacement")
        self.chk_exaggerate.setToolTip("Scale the displacement so a small distortion is visible")
        self.chk_exaggerate.toggled.connect(self._draw)
        bar.addWidget(self.chk_exaggerate)
        bar.addStretch(1)
        if open_page is not None:
            self.btn_open = QPushButton("Open Smile page")
            self.btn_open.clicked.connect(lambda: open_page())
            bar.addWidget(self.btn_open)
        layout.addLayout(bar)

        self.figure = Figure(figsize=(9.5, 8.5), layout="tight")
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)
        self._map_x = np.asarray(calibration.map_x, dtype=np.float64)
        self._map_y = np.asarray(calibration._get_map_y(), dtype=np.float64)
        frames = getattr(calibration, "frames", None) or {}
        self.background_kind = next((k for k in ("target", "lamp", "white") if frames.get(k) is not None), None)
        self._background = (np.asarray(frames[self.background_kind], np.float32)
                            if self.background_kind else None)
        self._draw()
        self.resize(980, 900)

    def _draw(self, *_):
        fig = self.figure
        fig.clear()
        grid_ax = fig.add_subplot(2, 2, (1, 2))
        dx_ax = fig.add_subplot(2, 2, 3)
        dy_ax = fig.add_subplot(2, 2, 4)
        mx, my = self._map_x, self._map_y
        h, w = mx.shape
        extent = (-.5, w - .5, h - .5, -.5)
        if self._background is not None and self._background.shape == (h, w):
            bg = self._background
            lo, hi = np.percentile(bg, (1, 99.5))
            grid_ax.imshow(bg, cmap="gray", aspect="auto", vmin=lo - 0.8 * (hi - lo), vmax=hi, alpha=.6,
                           extent=extent, interpolation="nearest")
        rows, cols, row_curves, col_curves = grid_curves(mx, my)
        biggest = max(float(np.hypot(sx - dx, sy - dy).max()) for dx, dy, sx, sy in row_curves + col_curves)
        spacing = min((h - 1) / GRID_LINES, (w - 1) / GRID_LINES)
        gain = _nice_gain(0.6 * spacing / max(biggest, 1e-6)) if self.chk_exaggerate.isChecked() else 1.0
        for Y in rows:
            grid_ax.plot([0, w - 1], [Y, Y], color="#34495e", lw=.7, ls=(0, (2, 3)), alpha=.7, zorder=2)
        for X in cols:
            grid_ax.plot([X, X], [0, h - 1], color="#34495e", lw=.7, ls=(0, (2, 3)), alpha=.7, zorder=2)
        for i, (dx, dy, sx, sy) in enumerate(row_curves):
            grid_ax.plot(dx + gain * (sx - dx), dy + gain * (sy - dy), color="#ff8c1a", lw=1.5, zorder=3,
                         label="rows: keystone + tilt" if i == 0 else None)
        for i, (dx, dy, sx, sy) in enumerate(col_curves):
            grid_ax.plot(dx + gain * (sx - dx), dy + gain * (sy - dy), color="#1fb5d6", lw=1.5, zorder=3,
                         label="columns: smile" if i == 0 else None)
        left, right = (int(v) for v in self.calibration.margin)
        top, bottom = (int(v) for v in self.calibration.row_margin)
        c0, c1, r0, r1 = left - .5, w - right - .5, top - .5, h - bottom - .5
        grid_ax.plot([c0, c1, c1, c0, c0], [r0, r0, r1, r1, r0], color="#2ecc71", lw=1, ls="--", zorder=4,
                     label="crop kept")
        grid_ax.set_xlim(-.5, w - .5)
        grid_ax.set_ylim(h - .5, -.5)
        grid_ax.set_xlabel("Column (spectral)")
        grid_ax.set_ylabel("Row (slit)")
        scale = f"displacement ×{gain:g}" if gain != 1 else "true scale"
        background = f", background: stored {self.background_kind} frame" if self.background_kind else ""
        grid_ax.set_title(f"Distortion grid ({scale}, largest {biggest:.1f} px{background}): the corrected image "
                          "reads each straight dotted line from the curve drawn here", fontsize=9, loc="left")
        legend = grid_ax.legend(fontsize=7.5, loc="upper right", framealpha=.85)
        legend.set_zorder(5)

        X = np.arange(w, dtype=np.float64)[None, :]
        Y = np.arange(h, dtype=np.float64)[:, None]
        for ax, shift, title in ((dx_ax, mx - X, "Column shift map_x − X (smile), px"),
                                 (dy_ax, my - Y, "Row shift map_y − Y (keystone), px")):
            spread = float(np.nanmax(np.abs(shift))) or 1.0
            image = ax.imshow(shift, cmap="RdBu_r", vmin=-spread, vmax=spread, aspect="auto", extent=extent,
                              interpolation="nearest")
            ax.set_title(title, fontsize=9, loc="left")
            ax.set_xlabel("Column (spectral)")
            ax.set_ylabel("Row (slit)")
            fig.colorbar(image, ax=ax)
        self.canvas.draw_idle()
