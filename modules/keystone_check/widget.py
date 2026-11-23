"""Keystone Check window: residual keystone of a corrected cube, measured on its horizontal edges.

Open an ENVI cube (or take one from the host, e.g. an open cube viewer), press *Check*; the
measurement runs on a worker thread and the *Measurement* tab shows the verdict, a table of the
wavelength zones and two plots (shift vs wavelength per edge; shift map).

The *Band fusion* tab works as soon as a cube is set: pick a centre band and drag the two side
sliders out from it to stack more and more neighbouring bands onto it (cumulative). Where the
bands line up an edge stays sharp and grey; keystone shows as coloured fringes, a doubled edge
or a signed stripe (see :mod:`modules.keystone_check.fusion`). A row profile beside the image
(averaged over the visible scan columns) shows the left side / centre / right side offsets.

Threading:
    GUI thread only; :func:`modules.keystone_check.check_keystone` and the band-fusion reads run
    on ``QThread`` workers that are joined on close.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
                             QProgressBar, QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem,
                             QTabWidget, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from .band_slider import BandWindowSlider
from .core import PASS_PX, KeystoneCheck, check_keystone
from .fusion import BandFusion, FusionParts, fusion_image, row_profiles

_BG = "#1e1f22"
_PANEL = "#26282c"
_TEXT = "#e0e0e0"
_DIM = "#9aa0a6"
_GOOD = "#66bb6a"
_BAD = "#ffb74d"
_LEFT = "#4fc3f7"            # bands below the centre band (as on the band slider)
_RIGHT = "#ef5350"           # bands above it
_FUSED = "#ffca28"           # profile of the summed bands
_MODES = (("Sum (grayscale)", "sum"),
          ("Difference: right side - left side", "difference"))


class _Worker(QThread):
    """Runs :func:`check_keystone` (``done(result)`` / ``failed(message)`` / ``progressed(percent)``)."""

    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progressed = pyqtSignal(int)

    def __init__(self, cube, wavelengths, ref_nm, row_margin):
        super().__init__()
        self._args = (cube, wavelengths, ref_nm, row_margin)

    def run(self) -> None:
        """Measure (worker thread)."""
        cube, wl, ref, margin = self._args
        try:
            self.done.emit(check_keystone(cube, wl, ref_nm=ref, row_margin=margin, progress=self.progressed.emit))
        except Exception as exc:  # noqa: BLE001 - reported in the window
            self.failed.emit(str(exc))


class _FusionWorker(QThread):
    """Reads / fuses bands for the preview (``done((fusion, parts))`` / ``failed(message)`` / ``progressed(percent)``)."""

    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progressed = pyqtSignal(int)

    def __init__(self, fusion: BandFusion, request: Tuple[int, int, int]):
        super().__init__()
        self._fusion, self._request = fusion, request

    def run(self) -> None:
        """Fuse (worker thread)."""
        try:
            self.done.emit((self._fusion, self._fusion.compose(*self._request, progress=self.progressed.emit)))
        except Exception as exc:  # noqa: BLE001 - reported in the window
            self.failed.emit(str(exc))


class KeystoneCheckWidget(QWidget):
    """Check a corrected cube for residual keystone.

    Signals:
        checked(object): the :class:`~modules.keystone_check.KeystoneCheck` of the last run.

    Args:
        parent: Optional Qt parent.
    """

    checked = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet(f"QWidget {{ background: {_BG}; color: {_TEXT}; }}"
                           "QPushButton { background: #33363b; border: 1px solid #44474d; padding: 4px 10px; }"
                           "QPushButton:disabled { color: #666; }"
                           "QTabBar::tab { background: #2b2d31; padding: 5px 14px; border: 1px solid #3a3a3a; }"
                           "QTabBar::tab:selected { background: #3a3d42; }")
        self._cube = None
        self._wl = None
        self._margin = (0, 0)
        self._name = ""
        self._worker: Optional[_Worker] = None
        self._sources: Callable[[], List[Tuple[str, object, object, dict]]] = lambda: []
        self.result: Optional[KeystoneCheck] = None

        self.btn_open = QPushButton("Open cube…")
        self.btn_open.clicked.connect(self._open_file)
        self.combo_source = QComboBox()
        self.combo_source.setMinimumWidth(260)
        self.combo_source.setToolTip("Cubes open in the application (cube viewers)")
        self.btn_use = QPushButton("Use")
        self.btn_use.clicked.connect(self._use_source)
        self.btn_refresh = QPushButton("↻")
        self.btn_refresh.setToolTip("Refresh the list of open cubes")
        self.btn_refresh.clicked.connect(self.refresh_sources)
        self.spin_ref = QDoubleSpinBox()
        self.spin_ref.setRange(300.0, 3000.0)
        self.spin_ref.setValue(1300.0)
        self.spin_ref.setSuffix(" nm")
        self.spin_ref.setToolTip("Reference band: shifts are measured against it")
        self.btn_check = QPushButton("Check")
        self.btn_check.setEnabled(False)
        self.btn_check.clicked.connect(self.run_check)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(160)
        self.progress.setVisible(False)
        top = QHBoxLayout()
        for w in (self.btn_open, QLabel("or"), self.combo_source, self.btn_use, self.btn_refresh,
                  QLabel("reference"), self.spin_ref, self.btn_check, self.progress):
            top.addWidget(w)
        top.addStretch(1)

        self.lbl_cube = QLabel("Open a corrected cube of a scene with sharp edges across the slit "
                               "(checkerboard, printed page, bar target).")
        self.lbl_cube.setStyleSheet(f"color: {_DIM};")
        self.lbl_verdict = QLabel("")
        self.lbl_verdict.setWordWrap(True)
        self.lbl_verdict.setStyleSheet("font-size: 14px; font-weight: bold;")

        self.fig = Figure(figsize=(12, 5), facecolor=_BG)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["zone (nm)", "median px", "95 % px", "max px", "worst row"])
        self.table.setMinimumWidth(470)
        self.table.setMaximumWidth(520)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet(f"QTableWidget {{ background: {_PANEL}; gridline-color: #3a3a3a; }}")
        page_measure = QWidget()
        body = QHBoxLayout(page_measure)
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self.canvas, stretch=1)
        body.addWidget(self.table)

        self.tabs = QTabWidget()
        self.tabs.addTab(page_measure, "Measurement")
        self.tabs.addTab(self._build_fusion_page(), "Band fusion")

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.lbl_cube)
        lay.addWidget(self.lbl_verdict)
        lay.addWidget(self.tabs, stretch=1)

    # ── band-fusion preview ────────────────────────────────────
    def _build_fusion_page(self) -> QWidget:
        self._fusion: Optional[BandFusion] = None
        self._fusion_worker: Optional[_FusionWorker] = None
        self._fusion_pending = False
        self._parts: Optional[FusionParts] = None
        self._im = None
        self._im_kind = ""
        self._marks: list = []
        self._fuse_timer = QTimer(self)
        self._fuse_timer.setSingleShot(True)
        self._fuse_timer.setInterval(120)
        self._fuse_timer.timeout.connect(self.update_preview)
        self._profile_timer = QTimer(self)
        self._profile_timer.setSingleShot(True)
        self._profile_timer.setInterval(0)
        self._profile_timer.timeout.connect(self._on_view_changed)

        self.band_slider = BandWindowSlider()
        self.band_slider.changed.connect(self._on_window_changed)
        self.lbl_centre = QLabel("centre: -")
        self.lbl_centre.setMinimumWidth(150)
        self.lbl_centre.setStyleSheet("font-weight: bold;")
        self.combo_mode = QComboBox()
        for text, _ in _MODES:
            self.combo_mode.addItem(text)
        self.combo_mode.currentIndexChanged.connect(lambda _i: self._fuse_timer.start())
        self.chk_edges = QCheckBox("mark measured edges")
        self.chk_edges.setChecked(True)
        self.chk_edges.setToolTip("Ticks at the edges found by Check (grey = thin, left out)")
        self.chk_edges.toggled.connect(self._redraw_overlays)
        self.progress_fusion = QProgressBar()
        self.progress_fusion.setMaximumWidth(140)
        self.progress_fusion.setVisible(False)
        self.lbl_left = QLabel("left: none")
        self.lbl_left.setStyleSheet(f"color: {_LEFT};")
        self.lbl_right = QLabel("right: none")
        self.lbl_right.setStyleSheet(f"color: {_RIGHT};")
        row1 = QHBoxLayout()
        row1.addWidget(self.lbl_left)
        row1.addWidget(self.lbl_centre)
        row1.addWidget(self.lbl_right)
        row1.addStretch(1)
        row1.addWidget(self.combo_mode)
        row1.addWidget(self.chk_edges)
        row1.addWidget(self.progress_fusion)

        hint = QLabel("Drag the ▲ left / ▲ right out of the centre band to sum more bands onto it (cumulative); "
                      "click the spectrum or drag the white marker to move the centre. Bands that line up keep the "
                      "edges as sharp as the centre band; keystone fills the edges in (wider, ramped or doubled) "
                      "as bands are added. Zoom into an edge with the magnifier: the profile compares the centre "
                      "band (white) with the sum (amber).")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_DIM};")

        self.fig_fusion = Figure(figsize=(12, 5), facecolor=_BG)
        self.canvas_fusion = FigureCanvasQTAgg(self.fig_fusion)
        self.canvas_fusion.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs = self.fig_fusion.add_gridspec(1, 2, width_ratios=[5, 1.4], wspace=0.04)
        self.ax_img = self.fig_fusion.add_subplot(gs[0])
        self.ax_prof = self.fig_fusion.add_subplot(gs[1], sharey=self.ax_img)
        self._style_fusion_axes()
        self.fig_fusion.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.08)
        self._clear_image_axes()
        self.toolbar_fusion = NavigationToolbar2QT(self.canvas_fusion, self)
        self.toolbar_fusion.setStyleSheet("background: #c8c8c8;")

        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 4, 0, 0)
        v.addWidget(self.band_slider)
        v.addLayout(row1)
        v.addWidget(hint)
        v.addWidget(self.toolbar_fusion)
        v.addWidget(self.canvas_fusion, stretch=1)
        return page

    def _clear_image_axes(self) -> None:
        """Clear the image axes; ``clear()`` drops the limit callbacks, so the row profile is hooked up again."""
        self.ax_img.clear()
        # deferred: drawing the profile inside a limit callback re-enters it (shared y axis) and recurses
        self.ax_img.callbacks.connect("xlim_changed", lambda _ax: self._profile_timer.start())
        self.ax_img.callbacks.connect("ylim_changed", lambda _ax: self._profile_timer.start())
        self._style_fusion_axes()

    def _on_view_changed(self) -> None:
        """The image was zoomed / panned: profile over the new visible columns and rows."""
        self._draw_profile()
        self.canvas_fusion.draw_idle()

    def _style_fusion_axes(self) -> None:
        for ax in (self.ax_img, self.ax_prof):
            ax.set_facecolor(_PANEL)
            ax.tick_params(colors=_DIM, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color("#444")
        self.ax_prof.tick_params(labelleft=False)

    def set_fusion(self, centre_nm: Optional[float] = None, left: Optional[int] = None,
                   right: Optional[int] = None) -> None:
        """Set the preview's centre band (nearest to *centre_nm*) and side band counts (scripts / tests)."""
        if self._wl is None:
            return
        self.band_slider.set_window(None if centre_nm is None else int(np.argmin(np.abs(self._wl - centre_nm))),
                                    left, right)

    def _on_window_changed(self, c: int, nl: int, nr: int) -> None:
        if self._wl is None:
            return
        self.lbl_centre.setText(f"centre {self._wl[c]:.1f} nm (#{c})")
        self.lbl_left.setText(f"left {nl} bands {self._wl[c - nl]:.0f}-{self._wl[c - 1]:.0f} nm" if nl
                              else "left: none")
        self.lbl_right.setText(f"right {nr} bands {self._wl[c + 1]:.0f}-{self._wl[c + nr]:.0f} nm" if nr
                               else "right: none")
        self._fuse_timer.start()

    def update_preview(self, synchronous: bool = False) -> None:
        """Fuse the bands chosen on the sliders and show them (on the preview worker unless *synchronous*)."""
        if self._fusion is None:
            return
        request = self.band_slider.window()
        if synchronous:
            try:
                self._on_fused((self._fusion, self._fusion.compose(*request)))
            except Exception as exc:  # noqa: BLE001
                self._on_fusion_failed(str(exc))
            return
        if self._fusion_worker is not None and self._fusion_worker.isRunning():
            self._fusion_pending = True                 # run again with the latest sliders when it ends
            return
        self._fusion_pending = False
        self._fusion_worker = _FusionWorker(self._fusion, request)
        self._fusion_worker.progressed.connect(self._on_fusion_progress)
        self._fusion_worker.done.connect(self._on_fused)
        self._fusion_worker.failed.connect(self._on_fusion_failed)
        self._fusion_worker.finished.connect(self._on_fusion_finished)
        self._fusion_worker.start()

    def _on_fusion_progress(self, p: int) -> None:
        self.progress_fusion.setVisible(p < 100)
        self.progress_fusion.setValue(p)

    def _on_fusion_finished(self) -> None:
        self.progress_fusion.setVisible(False)
        if self._fusion_pending:
            self.update_preview()

    def _on_fusion_failed(self, message: str) -> None:
        self.progress_fusion.setVisible(False)
        self.lbl_centre.setText(f"preview: {message}")

    def _on_fused(self, payload) -> None:
        fusion, parts = payload
        if fusion is not self._fusion:                  # a cube set while this one was reading
            return
        if (parts.centre_band, parts.n_left, parts.n_right) != self.band_slider.window():
            return       # stale: the sliders moved since; the timer / pending re-run brings the current one
        if self._parts is None:
            self.band_slider.set_spectrum(fusion.levels())     # levels were read on the worker
        self._parts = parts
        mode = _MODES[self.combo_mode.currentIndex()][1]
        img = fusion_image(parts, mode)
        kind = mode
        if self._im is not None and self._im_kind == kind:
            self._im.set_data(img)
        else:
            had = self._im is not None
            lims = (self.ax_img.get_xlim(), self.ax_img.get_ylim())
            self._clear_image_axes()
            self._marks = []
            h, w = img.shape[:2]
            aspect = "equal" if 0.2 < w / max(h, 1) < 5 else "auto"
            if kind == "difference":
                self._im = self.ax_img.imshow(img, cmap="RdBu_r", interpolation="nearest", aspect=aspect)
            else:
                self._im = self.ax_img.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest",
                                              aspect=aspect)
            self._im_kind = kind
            self.ax_img.set_xlabel("scan column", color=_DIM, fontsize=9)
            self.ax_img.set_ylabel("slit row", color=_DIM, fontsize=9)
            if had:
                self.ax_img.set_xlim(lims[0])
                self.ax_img.set_ylim(lims[1])
        if kind == "difference":
            sub = img[:: max(1, img.shape[0] // 400), :: max(1, img.shape[1] // 400)]
            v = max(float(np.percentile(np.abs(sub), 99.5)), 0.02)
            self._im.set_clim(-v, v)
        wl, c = self._wl, parts.centre_band

        def side(n: int, a: int, b: int) -> str:
            return f"{n} bands {wl[a]:.0f}-{wl[b]:.0f} nm" if n else "none"
        self.ax_img.set_title(f"centre {wl[c]:.1f} nm  |  left {side(parts.n_left, c - parts.n_left, c - 1)}"
                              f"  |  right {side(parts.n_right, c + 1, c + parts.n_right)}", color=_TEXT, fontsize=9)
        self._redraw_overlays()

    def _redraw_overlays(self, *_args) -> None:
        self._draw_marks()
        self._draw_profile()
        self.canvas_fusion.draw_idle()

    def _draw_profile(self) -> None:
        if self._parts is None:
            return
        ax = self.ax_prof
        for line in list(ax.lines):
            line.remove()
        x0, x1 = sorted(self.ax_img.get_xlim())
        prof = row_profiles(self._parts, (int(np.floor(x0 + 0.5)), int(np.ceil(x1 + 0.5))))
        rows = np.arange(self._parts.centre.shape[0])
        y0, y1 = sorted(self.ax_img.get_ylim())
        vis = (rows >= y0) & (rows <= y1)
        lo, hi = np.inf, -np.inf
        for key, colour, lw in (("fused", _FUSED, 1.6), ("centre", _TEXT, 1.0)):
            p = prof[key]
            # no autoscale: the y axis is shared with the image and must keep its zoom
            ax.plot(p, rows, color=colour, lw=lw, drawstyle="steps-mid", scalex=False, scaley=False)
            if vis.any():
                lo, hi = min(lo, float(p[vis].min())), max(hi, float(p[vis].max()))
        if np.isfinite(lo) and hi > lo:
            pad = 0.05 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad)
        ax.set_title("row profile: centre / sum", color=_TEXT, fontsize=9)
        if self.result is not None and self.chk_edges.isChecked():
            from matplotlib.transforms import blended_transform_factory
            tr = blended_transform_factory(ax.transAxes, ax.transData)
            for r, thin in zip(self.result.rows, self.result.thin):
                if y0 <= r <= y1:
                    ax.plot((0.0, 1.0), (r, r), transform=tr, color="#888" if thin else _GOOD, lw=0.6,
                            ls=":" if thin else "--", scalex=False, scaley=False)

    def _draw_marks(self) -> None:
        for m in self._marks:
            m.remove()
        self._marks = []
        if self._im is None or self.result is None or not self.chk_edges.isChecked():
            return
        from matplotlib.transforms import blended_transform_factory
        tr = blended_transform_factory(self.ax_img.transAxes, self.ax_img.transData)
        for r, thin in zip(self.result.rows, self.result.thin):
            for xa in ((0.0, 0.015), (0.985, 1.0)):
                self._marks += self.ax_img.plot(xa, (r, r), transform=tr, color="#888" if thin else _GOOD,
                                                lw=1.5, solid_capstyle="butt", scalex=False, scaley=False)

    # ── inputs ─────────────────────────────────────────────────
    def set_source_provider(self, provider: Callable[[], List[Tuple[str, object, object, dict]]]) -> None:
        """``provider() -> [(name, cube, wavelengths, metadata), ...]``: cubes the host has open."""
        self._sources = provider
        self.refresh_sources()

    def refresh_sources(self) -> None:
        """Re-read the host's open cubes into the list."""
        self.combo_source.clear()
        try:
            items = list(self._sources() or [])
        except Exception:  # noqa: BLE001
            items = []
        for name, *_ in items:
            self.combo_source.addItem(str(name))
        self.combo_source.setEnabled(bool(items))
        self.btn_use.setEnabled(bool(items))
        if not items:
            self.combo_source.addItem("(no open cube)")

    def set_cube(self, cube, wavelengths, metadata: Optional[dict] = None, name: str = "") -> None:
        """Use *cube* ``(slit, scan, wavelength)`` with its band *wavelengths* (nm).

        ``metadata``: header fields as writer config keys; ``keystone_row_margin`` is left out of
        the measurement unless ``keystone_margin_rows`` says the rows were cropped already.
        """
        from modules.cube_io.writers import keystone_row_margin
        meta = dict(metadata or {})
        margin = keystone_row_margin(meta)
        if str(meta.get("keystone_margin_rows", "")).startswith("cropped"):
            margin = (0, 0)
        self._cube, self._wl, self._margin, self._name = cube, np.asarray(wavelengths, float), margin, name
        h, w, b = cube.shape
        self.lbl_cube.setText(f"{name or 'cube'}: {h} x {w} x {b}, {self._wl.min():.0f}-{self._wl.max():.0f} nm"
                              + (f", keystone margin rows {margin[0]} / {margin[1]} left out" if sum(margin) else ""))
        self.btn_check.setEnabled(True)
        self.lbl_verdict.setText("")
        self.result = None
        self._start_fusion(cube, margin)

    def _start_fusion(self, cube, margin) -> None:
        """New cube for the band-fusion preview: centre on the reference band, 3 bands each side."""
        try:
            self._fusion = BandFusion(cube, row_margin=margin)
        except ValueError as exc:
            self._fusion = None
            self.lbl_centre.setText(str(exc))
            return
        self._parts = None
        self._im = None
        self._marks = []
        self._clear_image_axes()
        for line in list(self.ax_prof.lines):
            line.remove()
        self._style_fusion_axes()
        c = int(np.argmin(np.abs(self._wl - float(self.spin_ref.value()))))
        self.band_slider.set_bands(self._wl)
        self.band_slider.set_max_side(self._fusion.max_side)
        self.band_slider.set_window(c, 3, 3, emit=False)
        self._on_window_changed(*self.band_slider.window())

    def _use_source(self) -> None:
        i = self.combo_source.currentIndex()
        try:
            items = list(self._sources() or [])
        except Exception:  # noqa: BLE001
            items = []
        if 0 <= i < len(items):
            name, cube, wl, meta = items[i]
            if wl is None:
                self.lbl_cube.setText(f"{name}: no wavelengths")
                return
            self.set_cube(cube, wl, meta, name)

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open corrected cube", "", "ENVI header (*.hdr);;All files (*)")
        if not path:
            return
        try:
            from modules.cube_io import envi_scan_metadata, load_envi_cube_for_viewer
            hdr = path if path.lower().endswith(".hdr") else path + ".hdr"
            cube, wl = load_envi_cube_for_viewer(hdr)
            meta = envi_scan_metadata(hdr[:-4])
        except Exception as exc:  # noqa: BLE001
            self.lbl_cube.setText(f"Cannot open {path}: {exc}")
            return
        if wl is None:
            self.lbl_cube.setText(f"{path}: the header has no wavelengths")
            return
        self.set_cube(cube, wl, meta, path.replace("\\", "/").split("/")[-1])

    # ── run ────────────────────────────────────────────────────
    def run_check(self, synchronous: bool = False) -> bool:
        """Measure the cube (on a worker thread unless *synchronous*); returns whether it started."""
        if self._cube is None or (self._worker is not None and self._worker.isRunning()):
            return False
        self.btn_check.setEnabled(False)
        self.lbl_verdict.setText("Measuring…")
        self.lbl_verdict.setStyleSheet("font-size: 14px; font-weight: bold;")
        args = (self._cube, self._wl, float(self.spin_ref.value()), self._margin)
        if synchronous:
            try:
                self._on_done(check_keystone(*args[:2], ref_nm=args[2], row_margin=args[3]))
            except Exception as exc:  # noqa: BLE001
                self._on_failed(str(exc))
            return True
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self._worker = _Worker(*args)
        self._worker.progressed.connect(self.progress.setValue)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()
        return True

    def wait(self, ms: int = 120000) -> bool:
        """Block until a running check ends (tests / scripts); True when none is running any more."""
        if self._worker is not None and self._worker.isRunning():
            return bool(self._worker.wait(ms))
        return True

    def _on_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.btn_check.setEnabled(self._cube is not None)
        self.lbl_verdict.setText(f"Cannot check: {message}")
        self.lbl_verdict.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {_BAD};")

    def _on_done(self, res: KeystoneCheck) -> None:
        self.result = res
        self.progress.setVisible(False)
        self.btn_check.setEnabled(True)
        self.lbl_verdict.setText(res.verdict() + f"  ({int((~res.thin).sum())} edges"
                                 + (f", {int(res.thin.sum())} thin left out" if res.thin.any() else "") + ")")
        self.lbl_verdict.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {_GOOD if res.passed else _BAD};")
        self.table.setRowCount(len(res.zones))
        for i, z in enumerate(res.zones):
            vals = [f"{z['lo']:.0f}-{z['hi']:.0f}", f"{z['median']:.3f}", f"{z['p95']:.3f}", f"{z['max']:.2f}",
                    f"{z['worst_row']:.0f}"]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if j == 2 and z["p95"] >= PASS_PX:
                    item.setForeground(_qcolor(_BAD))
                self.table.setItem(i, j, item)
        self.table.resizeColumnsToContents()
        self._plot(res)
        self._redraw_overlays()
        self.checked.emit(res)

    def _plot(self, res: KeystoneCheck) -> None:
        import matplotlib
        self.fig.clear()
        ax1, ax2 = self.fig.subplots(1, 2)
        for ax in (ax1, ax2):
            ax.set_facecolor(_PANEL)
            ax.tick_params(colors=_DIM, labelsize=8)
            for s in ax.spines.values():
                s.set_color("#444")
        h = float(self._cube.shape[0]) if self._cube is not None else float(res.rows.max() + 1)
        cmap = matplotlib.colormaps["viridis"]
        for r, s, thin in zip(res.rows, res.shifts, res.thin):
            ax1.plot(res.wavelengths, s, lw=0.6 if thin else 0.9, color="#666" if thin else cmap(r / h),
                     ls=":" if thin else "-")
        for y in (-PASS_PX, PASS_PX):
            ax1.axhline(y, color=_BAD, ls="--", lw=0.8)
        lim = max(1.0, min(3.0, float(np.nanmax(np.abs(res.shifts[~res.thin]))) if (~res.thin).any() else 1.0))
        ax1.set_ylim(-lim, lim)
        ax1.set_xlabel("wavelength (nm)", color=_DIM, fontsize=9)
        ax1.set_ylabel(f"edge row shift vs {res.ref_nm:.0f} nm (px)", color=_DIM, fontsize=9)
        ax1.set_title("each edge (colour = slit row; grey dotted = thin, left out)", color=_TEXT, fontsize=9)
        ok = ~res.thin
        if ok.any():
            X = np.repeat(res.wavelengths[None, :], ok.sum(), 0)
            Y = np.repeat(res.rows[ok][:, None], res.wavelengths.size, 1)
            m = ax2.scatter(X.ravel(), Y.ravel(), c=res.shifts[ok].ravel(), cmap="RdBu_r", vmin=-1, vmax=1, s=6)
            cb = self.fig.colorbar(m, ax=ax2)
            cb.ax.tick_params(colors=_DIM, labelsize=8)
            cb.set_label("px (+ = lower in the frame)", color=_DIM, fontsize=8)
        ax2.set_ylim(h, 0)
        ax2.set_xlabel("wavelength (nm)", color=_DIM, fontsize=9)
        ax2.set_ylabel("slit row", color=_DIM, fontsize=9)
        ax2.set_title("shift map (reliable edges)", color=_TEXT, fontsize=9)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Wait for a running measurement before the window goes."""
        self._fuse_timer.stop()
        for worker in (self._worker, self._fusion_worker):
            if worker is not None and worker.isRunning():
                worker.wait()
        super().closeEvent(event)


def _qcolor(hex_colour: str):
    from PyQt5.QtGui import QColor
    return QColor(hex_colour)
