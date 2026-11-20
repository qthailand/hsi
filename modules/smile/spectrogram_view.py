# smile/spectrogram_view.py
"""Spectrogram view of a line target for the keystone page (Qt + matplotlib).

Shows the normalised target frame (slit rows x spectral columns, x axis in nm when a wavelength
calibration is known) with every traced bar (:class:`~modules.smile.keystone.TargetLine`) drawn
on top, its measured points, a draggable reference wavelength and, below, the displacement
``dy(lambda)`` of the selected bar at true scale.

Displacement scale: keystone is usually below a few pixels, invisible on a 1000-row frame. The
view can draw each bar's deviation from straight magnified by a gain ``G`` (``Auto`` picks one
from the bar spacing); the image is warped by the same field so bars and stripes stay together.
Every number shown (read-out, dy plot, statistics) is in true pixels.

Data conventions:
    ``dy`` of a bar at column ``x`` = measured row - straight row, where straight passes through
    the bar's row at the reference column ``x_ref`` with slope 0, or with the common tilt slope
    when "Remove common tilt" is on (keystone only).

Threading: GUI thread only.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .ui_theme import colors, is_dark

__all__ = ["SpectrogramView", "bar_deviation"]

_GAINS = [("Auto", 0), ("x1", 1), ("x10", 10), ("x25", 25), ("x50", 50)]
_MAX_COLS, _MAX_ROWS = 720, 900          # display decimation of the image
_RMS_WARN_PX = 0.25                       # same limit as the re-trace residual (RETRACE_WARN_PX)


def _accent(dark: bool):
    """(bar colour, selected colour) readable on the theme's plot background."""
    return ("#2fd3e6", "#f5a524") if dark else ("#0b8fa3", "#d9770b")


def bar_deviation(bar, xs: np.ndarray, w: int, x_ref: float, slope: float):
    """Measured row, straight row and deviation of *bar* at columns *xs* (NaN outside its span).

    Returns:
        tuple: ``(row, straight, dev, row_ref)``: arrays like *xs* and the bar's row at ``x_ref``.
    """
    xs = np.asarray(xs, float)
    lo, hi = float(np.min(bar.x)), float(np.max(bar.x))
    row_ref = float(bar.y_at(np.array([np.clip(x_ref, lo, hi)]), w)[0])
    row = bar.y_at(xs, w).astype(float)
    row[(xs < lo) | (xs > hi)] = np.nan
    straight = row_ref + slope * (xs - x_ref)
    return row, straight, row - straight, row_ref


def _nice_gain(value: float) -> float:
    """Largest 1-2-5 step <= *value* (at least 1)."""
    if not np.isfinite(value) or value <= 1:
        return 1.0
    exp = 10 ** math.floor(math.log10(value))
    for m in (5, 2, 1):
        if m * exp <= value:
            return float(m * exp)
    return float(exp)


class SpectrogramView(QWidget):
    """Interactive spectrogram of a line target with its traced bars (see the module docstring)."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        from .widget import PlotCanvas          # lazy: widget.py imports keystone_view -> this module
        self._norm: Optional[np.ndarray] = None
        self._bars: List = []
        self._rows_ref = np.zeros(0)
        self._x_ref = 0.0
        self._x_ref0 = 0.0
        self._slope = 0.0
        self._wl: Optional[Callable[[float], float]] = None
        self._sel = 0
        self._gain = 1.0
        self._drag = False
        self._cache_key = None
        self._img = None
        self._cols = np.zeros(0)
        self._disp: List[np.ndarray] = []
        self._ax = self._ax_prof = self._ref_line = self._ref_text = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("Displacement"))
        self._gain_group = QButtonGroup(self)
        self._gain_group.setExclusive(True)
        for i, (text, _g) in enumerate(_GAINS):
            b = QToolButton()
            b.setText(text)
            b.setCheckable(True)
            b.setAutoRaise(False)
            b.setToolTip("Draw each bar's deviation from straight magnified by this gain (the image is "
                         "warped the same way). x1 = true scale. Numbers are always true pixels.")
            self._gain_group.addButton(b, i)
            row.addWidget(b)
        self._gain_group.button(0).setChecked(True)
        self._gain_group.buttonClicked.connect(lambda _b: self._redraw())
        self.chk_tilt = QCheckBox("Remove common tilt")
        self.chk_tilt.setChecked(True)
        self.chk_tilt.setToolTip("Measure dy from a line with the common tilt of all bars (keystone only); "
                                 "off: from a horizontal line (tilt + keystone)")
        self.chk_points = QCheckBox("Points")
        self.chk_points.setChecked(True)
        self.chk_points.setToolTip("Measured bar centres")
        self.chk_fit = QCheckBox("Fit lines")
        self.chk_fit.setChecked(True)
        self.chk_fit.setToolTip("Fitted bar curves")
        for c in (self.chk_tilt, self.chk_points, self.chk_fit):
            c.toggled.connect(lambda _on: self._redraw())
            row.addWidget(c)
        row.addSpacing(8)
        row.addWidget(QLabel("Bar"))
        self.combo_bar = QComboBox()
        self.combo_bar.setMinimumContentsLength(4)
        self.combo_bar.currentIndexChanged.connect(self._on_bar_combo)
        row.addWidget(self.combo_bar)
        self.lbl_ref = QLabel("λref")
        row.addWidget(self.lbl_ref)
        self.spin_ref = QDoubleSpinBox()
        self.spin_ref.setDecimals(1)
        self.spin_ref.setKeyboardTracking(False)
        self.spin_ref.setToolTip("Reference wavelength (dy = 0 there). You can also drag the dashed line.")
        self.spin_ref.valueChanged.connect(self._on_ref_spin)
        row.addWidget(self.spin_ref)
        row.addStretch()
        self.btn_export = QPushButton("Export PNG")
        self.btn_export.setToolTip("Export this view as a presentation-quality PNG (300 dpi).")
        self.btn_export.clicked.connect(self._on_export)
        row.addWidget(self.btn_export)
        lay.addLayout(row)
        self.canvas = PlotCanvas(self, figsize=(10, 7))
        lay.addWidget(self.canvas, stretch=1)
        self.lbl_hover = QLabel(" ")
        self.lbl_hover.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_hover.setStyleSheet("font-family: Consolas, 'JetBrains Mono', monospace;")
        lay.addWidget(self.lbl_hover)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self._set_controls_enabled(False)
        self.clear("Load or grab a line-target frame: its spectrogram is drawn here")

    # ------------------------------------------------------------------ public
    def clear(self, message: str = "") -> None:
        """Drop the data and show *message*."""
        self._norm, self._bars = None, []
        self._cache_key = None
        self._set_controls_enabled(False)
        self.canvas.clear()
        ax = self.canvas.fig.add_subplot(111)
        ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes, color="#5d6d7e",
                fontsize=10, wrap=True)
        ax.set_axis_off()
        self._ax = self._ax_prof = None
        self.canvas.draw_idle()

    def set_data(self, preview: dict, wavelength: Optional[Callable[[float], float]] = None) -> None:
        """Show a :func:`~modules.smile.keystone_view.target_preview` payload.

        Args:
            preview: Needs ``norm``, ``bars``, ``x_ref``; ``fan`` (for the common tilt) is optional.
            wavelength: ``column -> nm`` of the stored wavelength calibration, or ``None`` (columns).
        """
        norm = np.asarray(preview["norm"], np.float32)
        h, w = norm.shape
        x_ref = float(preview.get("x_ref", (w - 1) / 2.0))
        bars = list(preview.get("bars") or [])
        if bars:
            rows = np.array([float(b.y_at(np.array([np.clip(x_ref, b.x.min(), b.x.max())]), w)[0]) for b in bars])
            order = np.argsort(rows)
            bars = [bars[i] for i in order]
        fan = preview.get("fan") or {}
        self._slope = (float(fan["coef"][0]) / max(1.0, fan["xr"] - fan["xl"])) if "coef" in fan else 0.0
        keep = self._sel if self._norm is not None and self._norm.shape == norm.shape else None
        self._norm, self._bars, self._wl = norm, bars, wavelength
        self._x_ref = self._x_ref0 = x_ref
        self._cache_key = None
        if keep is None or keep >= len(bars):
            keep = len(bars) // 2
        self._sel = keep
        self.combo_bar.blockSignals(True)
        self.combo_bar.clear()
        self.combo_bar.addItems([self._name(i) for i in range(len(bars))])
        self.combo_bar.setCurrentIndex(self._sel if bars else -1)
        self.combo_bar.blockSignals(False)
        self.spin_ref.blockSignals(True)
        lo, hi = self._to_axis(0.0), self._to_axis(w - 1.0)
        self.spin_ref.setRange(min(lo, hi), max(lo, hi))
        self.spin_ref.setSuffix(" nm" if wavelength else " col")
        self.spin_ref.setValue(self._to_axis(x_ref))
        self.spin_ref.blockSignals(False)
        self.lbl_ref.setText("λref" if wavelength else "Ref column")
        self._set_controls_enabled(bool(bars))
        self._redraw()

    def select_bar(self, index: int) -> None:
        if 0 <= index < len(self._bars) and index != self._sel:
            self._sel = index
            self.combo_bar.blockSignals(True)
            self.combo_bar.setCurrentIndex(index)
            self.combo_bar.blockSignals(False)
            self._redraw()

    @property
    def gain(self) -> float:
        """Displacement gain used by the last draw."""
        return self._gain

    # ------------------------------------------------------------------ helpers
    def _set_controls_enabled(self, on: bool) -> None:
        for wdg in (self.chk_tilt, self.chk_points, self.chk_fit, self.combo_bar, self.spin_ref, self.btn_export,
                    *self._gain_group.buttons()):
            wdg.setEnabled(on)

    def _name(self, i: int) -> str:
        return "B%02d" % (i + 1)

    def _to_axis(self, col: float) -> float:
        return float(self._wl(col)) if self._wl else float(col)

    def _from_axis(self, value: float) -> float:
        if not self._wl or self._norm is None:
            return float(value)
        w = self._norm.shape[1]
        a, b = float(self._wl(0.0)), float(self._wl(w - 1.0))
        if abs(b - a) < 1e-12:
            return float(value)
        return (float(value) - a) / (b - a) * (w - 1.0)        # stored calibrations are linear

    def _on_bar_combo(self, index: int) -> None:
        self.select_bar(index)

    def _on_ref_spin(self, value: float) -> None:
        if self._norm is None:
            return
        w = self._norm.shape[1]
        self._x_ref = float(np.clip(self._from_axis(value), 0, w - 1))
        self._redraw()

    def _slope_used(self) -> float:
        return self._slope if self.chk_tilt.isChecked() else 0.0

    def _compute(self):
        """Display columns, per-bar (row, straight, dev) there, the gain and the warped image (cached)."""
        norm, bars = self._norm, self._bars
        h, w = norm.shape
        cs = max(1, int(math.ceil(w / _MAX_COLS)))
        rs = max(1, int(math.ceil(h / _MAX_ROWS)))
        cols = np.arange(0, w, cs, dtype=float)
        slope = self._slope_used()
        curves = [bar_deviation(b, cols, w, self._x_ref, slope) for b in bars]
        self._rows_ref = np.array([c[3] for c in curves])
        choice = _GAINS[max(0, self._gain_group.checkedId())][1]
        if choice:
            gain = float(choice)
        else:
            biggest = max((float(np.nanmax(np.abs(c[2]))) for c in curves if np.isfinite(c[2]).any()), default=0.0)
            spacing = float(np.median(np.diff(self._rows_ref))) if len(bars) > 1 else 0.25 * h
            gain = _nice_gain(0.35 * spacing / max(biggest, 1e-3))
        key = (gain, round(self._x_ref, 2), slope, norm.shape, id(norm))
        if key != self._cache_key:
            rows_out = np.arange(0, h, rs, dtype=float)
            img = np.empty((rows_out.size, cols.size), np.float32)
            src_rows = np.arange(h, dtype=float)
            for j, x in enumerate(cols):
                col = norm[:, int(x)]
                if gain == 1.0 or not curves:
                    img[:, j] = np.interp(rows_out, src_rows, col)
                    continue
                pos = np.array([c[0][j] for c in curves])
                dev = np.array([c[2][j] for c in curves])
                ok = np.isfinite(pos) & np.isfinite(dev)
                if ok.sum() < 1:
                    img[:, j] = np.interp(rows_out, src_rows, col)
                    continue
                disp = pos[ok] + (gain - 1.0) * dev[ok]
                o = np.argsort(disp)
                shift = np.interp(rows_out, disp[o], dev[ok][o])
                img[:, j] = np.interp(rows_out - (gain - 1.0) * shift, src_rows, col)
            self._img, self._cache_key = img, key
        self._gain, self._cols = gain, cols
        return cols, curves, gain

    def _stretch(self):
        img = self._img
        lit = img[img > 0]
        if lit.size < 16:
            return 0.0, 1.0
        lo, hi = np.percentile(lit, [1, 99.5])
        return float(lo - 0.25 * (hi - lo)), float(hi + 0.18 * (hi - lo))     # keep the bright stripes grey

    def _xticks(self, ax) -> None:
        """Ticks in nm (when calibrated) at 1-2-5 steps, positioned in columns."""
        if not self._wl:
            ax.set_xlabel("Column (spectral)")
            return
        from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
        w = self._norm.shape[1]
        a, b = self._to_axis(0.0), self._to_axis(w - 1.0)
        nm = MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]).tick_values(min(a, b), max(a, b))
        nm = nm[(nm >= min(a, b)) & (nm <= max(a, b))]
        ax.xaxis.set_major_locator(FixedLocator([self._from_axis(v) for v in nm]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: "%.0f" % self._to_axis(x)))
        ax.set_xlabel("Wavelength (nm)")

    # ------------------------------------------------------------------ drawing
    def _redraw(self) -> None:
        if self._norm is None:
            return
        import matplotlib.patheffects as pe
        from matplotlib.lines import Line2D
        dark = is_dark(self)
        pal = colors(self)
        cyan, orange = _accent(dark)
        ink = "#ffffff" if dark else "#1b2631"
        h, w = self._norm.shape
        bars = self._bars
        cols, curves, gain = self._compute()
        fig = self.canvas.fig
        self.canvas.clear()
        gs = fig.add_gridspec(2, 2, height_ratios=[3.3, 1.35], width_ratios=[4.4, 1.0])
        right = gs[0, 1].subgridspec(2, 1, height_ratios=[2.3, 1.0])
        ax = fig.add_subplot(gs[0, 0])
        axp = fig.add_subplot(right[0], sharey=ax)
        axl = fig.add_subplot(right[1])
        axd = fig.add_subplot(gs[1, 0], sharex=ax)
        axs = fig.add_subplot(gs[1, 1])
        self._ax, self._ax_prof, self._ax_dy = ax, axp, axd

        # ---- spectrogram
        vmin, vmax = self._stretch()
        x0, x1 = float(cols[0]), float(cols[-1])
        ax.imshow(self._img, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax, interpolation="bilinear",
                  extent=(x0 - .5, x1 + .5, h - .5, -.5))
        self._disp = []
        n = len(bars)
        label_step = max(1, int(math.ceil(n / 12.0)))
        order = [k for k in range(n) if k != self._sel] + ([self._sel] if n else [])
        for k in order:
            row, straight, dev, _r0 = curves[k]
            disp = straight + gain * dev
            sel = k == self._sel
            col = orange if sel else cyan
            if self.chk_fit.isChecked():
                line, = ax.plot(cols, disp, color=col, lw=2.2 if sel else 1.1, alpha=1.0 if sel else .9, zorder=4 if sel else 3)
                if sel:
                    line.set_path_effects([pe.Stroke(linewidth=6, foreground=orange, alpha=.25), pe.Normal()])
            if self.chk_points.isChecked():
                b = bars[k]
                step = max(1, int(math.ceil(len(b.x) / 60.0)))
                bx, by = np.asarray(b.x[::step], float), np.asarray(b.y[::step], float)
                s_b = _r0 + self._slope_used() * (bx - self._x_ref)
                ax.scatter(bx, s_b + gain * (by - s_b), s=16 if sel else 7, color=col, edgecolors="#0b1520",
                           linewidths=.4, zorder=5 if sel else 3)
            if sel or k % label_step == 0:
                fin = np.where(np.isfinite(disp))[0]
                if fin.size:
                    ax.text(cols[fin[-1]] - 0.01 * w, disp[fin[-1]] - 0.012 * h, self._name(k), ha="right",
                            va="bottom", fontsize=7.5, fontweight="bold" if sel else "normal", zorder=6,
                            color=orange if sel else "#dce7f3", family="monospace",
                            bbox=dict(boxstyle="round,pad=0.15", fc=(0.03, 0.06, 0.1, .7), ec="none"))
        self._disp = [straight + gain * dev for _row, straight, dev, _r in curves]
        self._ref_line = ax.axvline(self._x_ref, color="#ffffff", lw=1.3, ls=(0, (5, 4)), alpha=.9, zorder=6)
        self._ref_text = ax.text(self._x_ref, 0.995 * h, self._ref_label(), ha="center", va="bottom", fontsize=8,
                                 color="#ffffff", zorder=7,
                                 bbox=dict(boxstyle="round,pad=0.3", fc=(0.03, 0.06, 0.1, .85), ec="none"))
        badge = "TRUE SCALE" if gain == 1 else "dy x%g" % gain
        ax.text(0.012, 0.015, badge, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold",
                family="monospace", color="#3ddc84" if gain == 1 else orange, zorder=7,
                bbox=dict(boxstyle="round,pad=0.35", fc=(0.03, 0.06, 0.1, .8), ec="none"))
        ax.set_xlim(x0 - .5, x1 + .5)
        ax.set_ylim(h - .5, -.5)
        ax.set_ylabel("Slit position (row)")
        ax.tick_params(labelbottom=False)
        tilt = "common tilt removed" if self.chk_tilt.isChecked() else "tilt + keystone"
        ax.set_title("Spectrogram - spatial x wavelength   (%d bars, %s%s)" % (
            n, tilt, "" if gain == 1 else ", bar displacement drawn x%g" % gain), fontsize=10, loc="left")

        # ---- profile across the slit at x_ref (true scale)
        c = int(round(np.clip(self._x_ref, 0, w - 1)))
        prof = np.mean(self._norm[:, max(0, c - 2):min(w, c + 3)], axis=1)
        axp.imshow(np.repeat(prof[:, None], 8, axis=1), cmap="gray", aspect="auto", vmin=vmin, vmax=vmax,
                   extent=(0, 1, h - .5, -.5), interpolation="bilinear")
        p_lo, p_hi = float(np.percentile(prof, 1)), float(np.percentile(prof, 99.5))
        pn = 0.06 + 0.88 * np.clip((prof - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)
        axp.plot(pn, np.arange(h), color=cyan, lw=1.0)
        axp.axvline(0.5, color=orange, lw=1, ls=(0, (3, 3)))
        for k, r in enumerate(self._rows_ref):
            sel = k == self._sel
            axp.plot([1.0, 1.08], [r, r], color=orange if sel else pal["muted"], lw=2 if sel else .8, clip_on=False)
        if n:
            axp.text(1.1, self._rows_ref[self._sel], self._name(self._sel), color=orange, fontsize=7.5,
                     va="center", ha="left", clip_on=False, family="monospace")
        axp.set_xlim(0, 1)
        axp.set_xticks([])
        axp.tick_params(labelleft=False)
        axp.set_title("Target across slit\n@ " + self._ref_label(), fontsize=8.5)

        # ---- legend
        axl.set_axis_off()
        handles = [Line2D([], [], color=cyan, lw=1.4, marker="o", ms=3.5, label="Traced bar (all)"),
                   Line2D([], [], color=orange, lw=2.4, marker="o", ms=4.5, label="Selected bar"),
                   Line2D([], [], color=ink if dark else "#5d6d7e", lw=1.2, ls=(0, (5, 4)), label="λref (dy = 0)")]
        axl.legend(handles=handles, loc="center", fontsize=8, frameon=True, handlelength=2.4, borderpad=.8)

        # ---- dy(lambda), true scale
        lim = 0.0
        for k, (row, straight, dev, _r) in enumerate(curves):
            if k == self._sel:
                continue
            axd.plot(cols, dev, color=cyan, lw=.8, alpha=.28)
            if np.isfinite(dev).any():
                lim = max(lim, float(np.nanmax(np.abs(dev))))
        stats = {}
        if n:
            row, straight, dev, r0 = curves[self._sel]
            b = bars[self._sel]
            bx, by = np.asarray(b.x, float), np.asarray(b.y, float)
            s_b = r0 + self._slope_used() * (bx - self._x_ref)
            resid = by - b.y_at(bx, w)
            axd.scatter(bx, by - s_b, s=6, color=orange, alpha=.45, lw=0, zorder=3)
            axd.plot(cols, dev, color=orange, lw=2.0, zorder=4)
            fin = np.isfinite(dev)
            if fin.any():
                lim = max(lim, float(np.nanmax(np.abs(dev))), float(np.percentile(np.abs(by - s_b), 99)))
                stats = dict(lo=float(np.nanmin(dev)), hi=float(np.nanmax(dev)), rms=float(np.sqrt(np.mean(resid ** 2))),
                             npts=int(bx.size), row=r0)
        lim = max(0.25, 1.15 * lim)
        axd.axhline(0, color=pal["muted"], lw=.8)
        axd.axvline(self._x_ref, color=ink if dark else "#5d6d7e", lw=1, ls=(0, (5, 4)), alpha=.75)
        axd.plot([self._x_ref], [0], marker="o", ms=5, color=ink if dark else "#1b2631", zorder=5)
        axd.set_ylim(-lim, lim)
        axd.set_ylabel("dy (px)")
        axd.grid(alpha=.25)
        axd.set_title("dy(λ) of %s - true scale; other bars faint" % (self._name(self._sel) if n else "-"),
                      fontsize=9, loc="left")
        self._xticks(axd)

        # ---- statistics
        axs.set_axis_off()
        if stats:
            p2p = stats["hi"] - stats["lo"]
            worst, wk = 0.0, 0
            for k, (_row, _s, dev, _r) in enumerate(curves):
                if np.isfinite(dev).any():
                    v = float(np.nanmax(dev) - np.nanmin(dev))
                    if v > worst:
                        worst, wk = v, k
            fg, muted = pal["text"], pal["muted"]
            axs.text(0.02, 0.98, "Bar " + self._name(self._sel), fontsize=10, fontweight="bold", va="top", color=fg,
                     transform=axs.transAxes)
            axs.text(0.02, 0.80, "%.2f px" % p2p, fontsize=19, fontweight="bold", va="top", color=orange,
                     family="monospace", transform=axs.transAxes)
            axs.text(0.02, 0.55, "peak-to-peak dy", fontsize=7.5, va="top", color=muted, transform=axs.transAxes)
            good = stats["rms"] < _RMS_WARN_PX
            lines = [("Row @ ref", "%.1f" % stats["row"]),
                     ("dy min / max", "%+.2f / %+.2f" % (stats["lo"], stats["hi"])),
                     ("Fit RMS", "%.3f px" % stats["rms"]),
                     ("Points", "%d" % stats["npts"]),
                     ("Worst bar", "%s %.2f px" % (self._name(wk), worst))]
            y = 0.42
            for k_, v_ in lines:
                axs.text(0.02, y, k_, fontsize=7.5, color=muted, va="top", transform=axs.transAxes)
                axs.text(0.98, y, v_, fontsize=7.5, color=(pal["good"] if good else pal["warn"]) if k_ == "Fit RMS"
                         else fg, va="top", ha="right", family="monospace", transform=axs.transAxes)
                y -= 0.105
        self.canvas.fig.set_layout_engine("constrained")
        self.canvas.draw_idle()

    def _ref_label(self) -> str:
        return ("λref %.0f nm" % self._to_axis(self._x_ref)) if self._wl else ("ref col %.0f" % self._x_ref)

    # ------------------------------------------------------------------ mouse
    def _near_ref(self, event) -> bool:
        if self._ax is None or event.inaxes is not self._ax or event.x is None:
            return False
        px = self._ax.transData.transform((self._x_ref, 0))[0]
        return abs(event.x - px) < 8

    def _nearest_bar(self, x: float, y: float, use_display: bool) -> int:
        if not self._bars or self._cols.size == 0:
            return -1
        if not use_display:
            return int(np.argmin(np.abs(self._rows_ref - y)))
        j = int(np.clip(np.searchsorted(self._cols, x), 0, self._cols.size - 1))
        d = np.array([abs(dp[j] - y) if np.isfinite(dp[j]) else np.inf for dp in self._disp])
        return int(np.argmin(d)) if np.isfinite(d).any() else -1

    def _on_press(self, event) -> None:
        if self._norm is None or event.button != 1 or event.xdata is None:
            return
        if self._near_ref(event):
            self._drag = True
            return
        if event.inaxes is self._ax:
            k = self._nearest_bar(event.xdata, event.ydata, True)
        elif event.inaxes is self._ax_prof:
            k = self._nearest_bar(event.xdata, event.ydata, False)
        else:
            return
        if k >= 0:
            self.select_bar(k)

    def _on_motion(self, event) -> None:
        if self._norm is None or self._ax is None:
            return
        h, w = self._norm.shape
        if self._drag:
            if event.inaxes is self._ax and event.xdata is not None:
                self._x_ref = float(np.clip(event.xdata, 0, w - 1))
                self._ref_line.set_xdata([self._x_ref, self._x_ref])
                self._ref_text.set_x(self._x_ref)
                self._ref_text.set_text(self._ref_label())
                self.canvas.draw_idle()
            return
        self.canvas.setCursor(Qt.SizeHorCursor if self._near_ref(event) else Qt.CrossCursor)
        if event.inaxes is not self._ax or event.xdata is None:
            self.lbl_hover.setText(" ")
            return
        x, y = float(event.xdata), float(event.ydata)
        j = int(np.clip(np.searchsorted(self._cols, x), 0, self._cols.size - 1))
        i_img = int(np.clip(round(y / h * self._img.shape[0]), 0, self._img.shape[0] - 1))
        where = ("λ %.1f nm" % self._to_axis(x)) if self._wl else ""
        text = "%s  col %d  row %.0f  I %.3f" % (where, round(x), y, float(self._img[i_img, j]))
        k = self._nearest_bar(x, y, True)
        if k >= 0:
            _row, _s, dev, _r = bar_deviation(self._bars[k], np.array([x]), w, self._x_ref, self._slope_used())
            if np.isfinite(dev[0]):
                text += "   |   %s  dy %+.3f px (true)" % (self._name(k), dev[0])
        self.lbl_hover.setText(text.strip())

    def _on_release(self, event) -> None:
        if not self._drag:
            return
        self._drag = False
        self.spin_ref.blockSignals(True)
        self.spin_ref.setValue(self._to_axis(self._x_ref))
        self.spin_ref.blockSignals(False)
        self._redraw()

    # ------------------------------------------------------------------ export
    def _on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export spectrogram as PNG", "Keystone_Spectrogram.png",
                                              "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        self.canvas.fig.savefig(path, format="png", dpi=300, facecolor=self.canvas.fig.get_facecolor())
