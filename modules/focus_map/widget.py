"""Live focus-map window: where a line-target frame is sharp, and one number to focus by.

The host feeds frames (``begin_grab`` / ``accept_snapshot`` / ``snapshot_failed``, the same
one-shot protocol as Smile Analysis); with *Live* ticked the widget asks for the next frame
as soon as one has been processed. Frames can also be opened from ``.png`` / ``.npy`` files.

Threading:
    GUI thread only. The map of one 1024 x 1280 frame takes about 50 ms, computed in the GUI
    thread between frames.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .core import FocusMap, focus_advice, focus_map

#: Scores kept in the history strip.
HISTORY = 240

_BG = "#1e1f22"
_PANEL = "#26282c"
_TEXT = "#e0e0e0"
_DIM = "#9aa0a6"
_GOOD = "#66bb6a"
_BAD = "#ef5350"
_ACCENT = "#ffb74d"


def _big_label(size: int, colour: str = _TEXT, bold: bool = True) -> QLabel:
    lab = QLabel("—")
    font = QFont("Consolas")
    font.setPointSize(size)
    font.setBold(bold)
    lab.setFont(font)
    lab.setStyleSheet(f"color: {colour};")
    return lab


class FocusMapWidget(QWidget):
    """Bar-contrast map of a line-target frame with the best-focus curve and a focus score.

    Signals:
        grabRequested(): the widget wants one camera frame (the host answers with
            :meth:`begin_grab` and later :meth:`accept_snapshot` or :meth:`snapshot_failed`).

    Args:
        parent: Optional Qt parent.
    """

    grabRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet(f"QWidget {{ background: {_BG}; color: {_TEXT}; }}"
                           f"QPushButton {{ background: #33363b; border: 1px solid #44474d; padding: 4px 10px; }}"
                           f"QPushButton:checked {{ background: #2e7d32; }}"
                           f"QPushButton:disabled {{ color: #666; }}")
        self._request_id = None
        self._waiting = False
        self._wavelength_provider: Optional[Callable] = None
        self._to_nm: Optional[Callable] = None
        self._history = deque(maxlen=HISTORY)
        self._peak = 0.0
        self._last: Optional[FocusMap] = None
        self._im = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._request_next)

        # ── controls ───────────────────────────────────────────
        self.btn_live = QPushButton("Live")
        self.btn_live.setCheckable(True)
        self.btn_live.setToolTip("Grab frames continuously and update the map")
        self.btn_live.toggled.connect(self._on_live)
        self.btn_grab = QPushButton("Grab once")
        self.btn_grab.clicked.connect(self._request_next)
        self.btn_open = QPushButton("Open image…")
        self.btn_open.clicked.connect(self._open_file)
        self.btn_reset = QPushButton("Reset peak")
        self.btn_reset.clicked.connect(self.reset_peak)
        self.spin_interval = QDoubleSpinBox()
        self.spin_interval.setRange(0.0, 5.0)
        self.spin_interval.setSingleStep(0.1)
        self.spin_interval.setValue(0.2)
        self.spin_interval.setSuffix(" s")
        self.spin_interval.setToolTip("Pause between live frames")
        self.lab_status = QLabel("Point the camera at the bar target (bars along the spectrum), then press Live.")
        self.lab_status.setStyleSheet(f"color: {_DIM};")
        top = QHBoxLayout()
        for w in (self.btn_live, self.btn_grab, QLabel("every"), self.spin_interval, self.btn_open, self.btn_reset):
            top.addWidget(w)
        top.addWidget(self.lab_status, stretch=1)

        # ── plots ──────────────────────────────────────────────
        self.fig = Figure(figsize=(9, 6), facecolor=_BG)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.38, left=0.07, right=0.98, top=0.86, bottom=0.08)
        self.ax_map = self.fig.add_subplot(gs[0])
        self.ax_hist = self.fig.add_subplot(gs[1])
        for ax in (self.ax_map, self.ax_hist):
            ax.set_facecolor(_PANEL)
            ax.tick_params(colors=_DIM, labelsize=8)
            for s in ax.spines.values():
                s.set_color("#444")
        self.ax_map.set_ylabel("row (slit)", color=_DIM, fontsize=9)
        self.ax_map.set_xlabel("column (spectral)", color=_DIM, fontsize=9)
        self.ax_hist.set_ylabel("worst-zone", color=_DIM, fontsize=9)
        self.ax_hist.set_xlabel("frames (latest at right) - turn the focus until this is highest", color=_DIM, fontsize=8)
        self._sec = None
        (self._pts,) = self.ax_map.plot([], [], "o", ms=3, color="#4dd0e1", label="sharpest column per row")
        (self._fit,) = self.ax_map.plot([], [], "-", lw=1.5, color="white", alpha=.9, label="best-focus curve")
        self._mid = self.ax_map.axvline(np.nan, color=_ACCENT, ls="--", lw=1, label="spectrum middle")
        self.ax_map.legend(loc="lower right", fontsize=7, facecolor=_PANEL, edgecolor="#444", labelcolor=_TEXT)
        (self._hline,) = self.ax_hist.plot([], [], "-", color=_GOOD, lw=1.4)
        self._hpeak = self.ax_hist.axhline(np.nan, color=_ACCENT, ls=":", lw=1)

        # ── numbers ────────────────────────────────────────────
        side = QFrame()
        side.setStyleSheet(f"QFrame {{ background: {_PANEL}; border-radius: 6px; }}")
        side.setFixedWidth(270)
        g = QGridLayout(side)
        g.setContentsMargins(12, 12, 12, 12)
        cap = lambda t: self._caption(t)
        self.lab_score = _big_label(34, _GOOD)
        self.lab_trend = _big_label(26, _DIM)
        self.lab_peak = _big_label(11, _ACCENT, bold=False)
        self.lab_balance = _big_label(11, _DIM, bold=False)
        self.lab_centre = _big_label(18)
        self.lab_edge = _big_label(18)
        self.lab_tilt = _big_label(10, _DIM, bold=False)
        self.lab_tilt.setWordWrap(True)
        self.lab_advice = QLabel("")
        self.lab_advice.setWordWrap(True)
        self.lab_advice.setStyleSheet(f"color: {_TEXT}; font-size: 12px;")
        g.addWidget(cap("Worst-zone contrast (×100)"), 0, 0, 1, 2)
        g.addWidget(self.lab_score, 1, 0)
        g.addWidget(self.lab_trend, 1, 1, Qt.AlignRight)
        g.addWidget(self.lab_peak, 2, 0, 1, 2)
        g.addWidget(self.lab_balance, 3, 0, 1, 2)
        g.addWidget(cap("Sharpest at slit centre"), 4, 0, 1, 2)
        g.addWidget(self.lab_centre, 5, 0, 1, 2)
        g.addWidget(cap("Sharpest at slit ends"), 6, 0, 1, 2)
        g.addWidget(self.lab_edge, 7, 0, 1, 2)
        g.addWidget(self.lab_tilt, 8, 0, 1, 2)
        g.addWidget(self.lab_advice, 9, 0, 1, 2)
        g.setRowStretch(10, 1)

        body = QHBoxLayout()
        body.addWidget(self.canvas, stretch=1)
        body.addWidget(side)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addLayout(body, stretch=1)

    @staticmethod
    def _caption(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setStyleSheet(f"color: {_DIM}; font-size: 11px;")
        return lab

    # ── host interface ────────────────────────────────────────
    def set_wavelength_provider(self, provider: Optional[Callable]) -> None:
        """``provider(camera_bin) -> (column -> nm)`` or ``None``; asked for every frame (``None``: columns only)."""
        self._wavelength_provider = provider

    def set_grab_available(self, available: bool) -> None:
        """Enable or disable *Live* / *Grab once* (no camera: files only)."""
        self.btn_live.setEnabled(available)
        self.btn_grab.setEnabled(available)
        if not available:
            self.btn_live.setChecked(False)

    def begin_grab(self, request_id) -> None:
        """Remember the pending request; only a snapshot with this id is accepted."""
        self._request_id = request_id
        self._waiting = True

    def accept_snapshot(self, packet) -> None:
        """Process the requested frame (``packet.image`` 2-D ``(slit, wavelength)``, ``packet.camera_bin``)."""
        if not self._waiting or packet.request_id != self._request_id:
            return
        self._waiting = False
        self.set_frame(packet.image, getattr(packet, "camera_bin", None))
        self._schedule()

    def snapshot_failed(self, request_id, message: str) -> None:
        """Show why a frame did not come; in live mode try again after a second."""
        if request_id is not None and request_id != self._request_id:
            return
        self._waiting = False
        self.lab_status.setText(str(message))
        if self.btn_live.isChecked():
            self._timer.start(1000)

    def set_frame(self, image: np.ndarray, camera_bin: Optional[int] = None) -> Optional[FocusMap]:
        """Compute and show the focus map of ``image``; returns it (``None`` if it could not be measured)."""
        self._to_nm = None
        if self._wavelength_provider is not None:
            try:
                self._to_nm = self._wavelength_provider(camera_bin)
            except Exception:  # noqa: BLE001 - a missing calibration only drops the nm labels
                self._to_nm = None
        try:
            fm = focus_map(image)
        except ValueError as exc:
            self.lab_status.setText(f"Cannot measure this frame: {exc}")
            return None
        self._last = fm
        self._show(fm)
        h, w = fm.frame_shape
        self.lab_status.setText(f"{w} x {h}" + (f", bin {camera_bin}" if camera_bin else "")
                                + f", bar period {fm.period:.1f} px")
        return fm

    def reset_peak(self) -> None:
        """Forget the peak score and the history (after changing exposure, gain or target)."""
        self._peak = 0.0
        self._history.clear()
        self._hline.set_data([], [])
        self._hpeak.set_ydata([np.nan, np.nan])
        self.lab_peak.setText("peak —")
        self.canvas.draw_idle()

    # ── live loop ─────────────────────────────────────────────
    def _on_live(self, on: bool) -> None:
        self.btn_grab.setEnabled(not on)
        if on and not self._waiting:
            self._request_next()
        elif not on:
            self._timer.stop()

    def _schedule(self) -> None:
        if self.btn_live.isChecked():
            self._timer.start(int(1000 * self.spin_interval.value()))

    def _request_next(self) -> None:
        if self._waiting:
            return
        self.grabRequested.emit()

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open line-target frame", "", "Images (*.png *.tif *.tiff *.npy)")
        if not path:
            return
        try:
            if path.lower().endswith(".npy"):
                img = np.load(path)
            else:
                import matplotlib.image as mpimg
                img = mpimg.imread(path)
            img = np.asarray(img, np.float32)
            if img.ndim == 3:
                img = img[..., :3].mean(axis=2)
        except Exception as exc:  # noqa: BLE001
            self.lab_status.setText(f"Cannot open {path}: {exc}")
            return
        self.set_frame(img)

    # ── drawing ───────────────────────────────────────────────
    def _show(self, fm: FocusMap) -> None:
        h, w = fm.frame_shape
        vmax = max(1e-6, float(np.nanpercentile(fm.contrast, 99)))
        extent = (-0.5, w - 0.5, h - 0.5, -0.5)
        if self._im is None or self._im.get_array().shape != fm.contrast.shape:
            if self._im is not None:
                self._im.remove()
            self._im = self.ax_map.imshow(fm.contrast, aspect="auto", cmap="magma", vmin=0, vmax=vmax,
                                          extent=extent, interpolation="nearest", zorder=1)
        else:
            self._im.set_data(fm.contrast)
            self._im.set_clim(0, vmax)
            self._im.set_extent(extent)
        self.ax_map.set_xlim(-0.5, w - 0.5)
        self.ax_map.set_ylim(h - 0.5, -0.5)
        self._pts.set_data(fm.best_col, fm.row_samples)
        rr = np.linspace(fm.lit_rows[0], fm.lit_rows[1], 200)
        self._fit.set_data(fm.fitted_col(rr), rr)
        self._mid.set_xdata([0.5 * (fm.lit_cols[0] + fm.lit_cols[1])] * 2)
        to_nm = self._to_nm
        if self._sec is not None:
            self._sec.remove()
            self._sec = None
        if to_nm is not None:
            a = float(to_nm(1.0) - to_nm(0.0))
            b = float(to_nm(0.0))
            if a != 0:
                self._sec = self.ax_map.secondary_xaxis("top", functions=(lambda c: a * c + b, lambda n: (n - b) / a))
                self._sec.tick_params(colors=_DIM, labelsize=8)
                self._sec.set_xlabel("wavelength (nm)", color=_DIM, fontsize=9)
        self.ax_map.set_title("Bar contrast - bright = sharp bars" + ("" if to_nm else "   (no wavelength calibration)"),
                              color=_TEXT, fontsize=10, pad=24 if to_nm else 6)

        score = 100.0 * fm.worst
        self._history.append(score)
        self._peak = max(self._peak, score)
        ys = np.array(self._history)
        self._hline.set_data(np.arange(len(ys)), ys)
        self._hpeak.set_ydata([self._peak, self._peak])
        self.ax_hist.set_xlim(0, max(HISTORY, len(ys)) - 1)
        self.ax_hist.set_ylim(0, max(1e-3, 1.15 * self._peak))
        self.canvas.draw_idle()

        near_peak = score >= 0.97 * self._peak
        self.lab_score.setText(f"{score:.2f}")
        self.lab_score.setStyleSheet(f"color: {_GOOD if near_peak else _TEXT};")
        if len(ys) >= 6:
            d = ys[-3:].mean() - ys[-6:-3].mean()
            tol = 0.02 * max(1e-6, self._peak)
            arrow, colour = ("▲", _GOOD) if d > tol else (("▼", _BAD) if d < -tol else ("●", _DIM))
        else:
            arrow, colour = "●", _DIM
        self.lab_trend.setText(arrow)
        self.lab_trend.setStyleSheet(f"color: {colour};")
        self.lab_peak.setText(f"peak {self._peak:.2f}")
        self.lab_balance.setText(f"even-ness (worst / best) {100 * fm.balance:.0f} %")
        fmt = (lambda c: f"{float(to_nm(c)):.0f} nm") if to_nm else (lambda c: f"col {c:.0f}")
        if fm.fit is None:
            self.lab_centre.setText("—")
            self.lab_edge.setText("—")
            self.lab_tilt.setText("")
        else:
            lo, hi = fm.lit_cols

            def shown(c):
                # the parabola may run past the lit spectrum at the slit ends: say so instead of a number off the sensor
                if c > hi:
                    return "> " + fmt(hi)
                if c < lo:
                    return "< " + fmt(lo)
                return fmt(c)
            self.lab_centre.setText(shown(fm.centre_col))
            self.lab_edge.setText(shown(fm.edge_col))
            top, bottom = fm.fitted_col(np.array(fm.lit_rows, float))
            self.lab_tilt.setText(f"top {shown(top)} / bottom {shown(bottom)}")
        self.lab_advice.setText(focus_advice(fm, to_nm))

    # ── teardown ──────────────────────────────────────────────
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Stop the live loop."""
        self._timer.stop()
        self.btn_live.setChecked(False)
        super().closeEvent(event)
