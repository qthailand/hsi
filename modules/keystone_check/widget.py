"""Keystone Check window: residual keystone of a corrected cube, measured on its horizontal edges.

Open an ENVI cube (or take one from the host, e.g. an open cube viewer), press *Check*; the
measurement runs on a worker thread and the window shows the verdict, a table of the
wavelength zones and two plots (shift vs wavelength per edge; shift map).

Threading:
    GUI thread only; :func:`modules.keystone_check.check_keystone` runs on a ``QThread`` that is
    joined on close.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QProgressBar,
                             QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .core import PASS_PX, KeystoneCheck, check_keystone

_BG = "#1e1f22"
_PANEL = "#26282c"
_TEXT = "#e0e0e0"
_DIM = "#9aa0a6"
_GOOD = "#66bb6a"
_BAD = "#ffb74d"


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
                           "QPushButton:disabled { color: #666; }")
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
        body = QHBoxLayout()
        body.addWidget(self.canvas, stretch=1)
        body.addWidget(self.table)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.lbl_cube)
        lay.addWidget(self.lbl_verdict)
        lay.addLayout(body, stretch=1)

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
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait()
        super().closeEvent(event)


def _qcolor(hex_colour: str):
    from PyQt5.QtGui import QColor
    return QColor(hex_colour)
