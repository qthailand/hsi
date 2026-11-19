# smile/keystone_view.py
"""Keystone page of the smile calibration widget (Qt).

The page collects a line-target frame (and optionally a white frame), runs
:func:`modules.smile.keystone.analyze_keystone` on the smile analysis of the
lamp frame in a worker thread and shows the report and three check plots.
The owning :class:`~modules.smile.widget.SmileCorrectionWidget` keeps the
corrector: it receives the result through :attr:`KeystonePage.keystoneReady`
and attaches it (one combined smile + keystone map).

Data conventions:
    Frames are 2-D ``(slit rows, spectral columns)`` with the same orientation,
    bin and size as the lamp frame. Grabbed frames are oriented and
    dark-subtracted by the host; files are used as stored.

Threading:
    Widgets live on the GUI thread; only :class:`KeystoneWorker.run` runs in a
    worker thread and touches no widget.
"""
from __future__ import annotations

import html
import logging
import os
from typing import Callable, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .core import load_frame
from .keystone import RETRACE_WARN_PX, analyze_keystone, keystone_report

logger = logging.getLogger(__name__)

__all__ = ["KeystonePage", "KeystoneWorker"]

_IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All (*.*)"
_POLARITIES = [("Dark bars on white", "dark"), ("Bright bars on dark", "bright")]
_PLOTS = [
    ("Target  Before / After", "Keystone_Target_Before_After.png"),
    ("Row Shift Map", "Keystone_Row_Shift_Map.png"),
    ("Re-trace Residual", "Keystone_Retrace_Residual.png"),
]


def _ascii(text: str) -> str:
    """User-visible mismatch texts stay ASCII (``CameraSetting.mismatch`` uses a not-equal sign)."""
    return text.replace("≠", "!=").replace("·", "-").replace("×", "x")


def _setting_problems(reference, other) -> List[str]:
    """Differences between two camera settings (empty when one is unknown or they agree)."""
    if reference is None or other is None:
        return []
    try:
        return [_ascii(p) for p in reference.mismatch(other)]
    except Exception:  # noqa: BLE001
        return []


def _num(v) -> float:
    """JSON value -> float (``None`` = not measured -> NaN)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


class KeystoneWorker(QThread):
    """Run :func:`analyze_keystone` off the GUI thread.

    Signals:
        done: ``(smile_analysis, KeystoneAnalysis)`` on success.
        failed: Error text.
    """

    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, analysis, target: np.ndarray, kwargs: dict):
        """Args:
            analysis: The lamp's :class:`~modules.smile.analysis.SmileAnalysis` (read only).
            target: Line-target frame (a private copy).
            kwargs: Passed to :func:`analyze_keystone` (``white``, ``polarity``, ``inputs``).
        """
        super().__init__()
        self.analysis = analysis
        self._target = target
        self._kwargs = kwargs

    def run(self) -> None:
        """Compute the joint geometry; emit ``done`` or ``failed``."""
        try:
            self.done.emit((self.analysis, analyze_keystone(self.analysis, self._target, **self._kwargs)))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Keystone analysis failed")
            self.failed.emit(str(exc))


class KeystonePage(QWidget):
    """Line target (+ white) -> joint smile + keystone geometry, with report and plots.

    Signals:
        grabRequested: ``"target"`` or ``"white"``; the owner asks the host for one frame and
            hands it back through :meth:`set_frame`.
        keystoneReady: The :class:`~modules.smile.keystone.KeystoneAnalysis` just computed for
            the current smile analysis (the owner attaches it to its corrector).
        statusChanged: Message for the owner's status bar.

    Attributes:
        target, white: Frames ``float32 (H, W)`` or ``None``.
        target_setting, white_setting: ``CameraSetting`` they were taken with (``None`` = unknown).
        result: Latest :class:`KeystoneAnalysis` of this session, or ``None``.
        result_setting: Camera setting of the target used for :attr:`result`.
    """

    grabRequested = pyqtSignal(str)
    keystoneReady = pyqtSignal(object)
    statusChanged = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None,
                 setting_for_file: Optional[Callable[[Tuple[int, int]], object]] = None):
        """Args:
            parent: Parent widget.
            setting_for_file: ``shape -> CameraSetting or None`` for frames loaded from disk.
        """
        super().__init__(parent)
        self._setting_for_file = setting_for_file
        self._smile = None                  # SmileAnalysis of the lamp
        self._lamp_shape = None
        self._lamp_setting = None
        self.target = self.white = None
        self.target_setting = self.white_setting = None
        self._target_source = self._white_source = ""
        self.result = None
        self.result_setting = None
        self._pending_setting = None
        self._worker: Optional[KeystoneWorker] = None
        self._grab_enabled = False
        self._build_ui()
        self._clear_plots("Run Analyze Keystone to see the check plots")
        self._refresh()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        from .widget import PlotCanvas          # lazy: widget.py imports this module

        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        row = QHBoxLayout()
        self.btn_load_target = QPushButton("Load target...")
        self.btn_load_target.setToolTip("Line-target frame(s); several files are averaged")
        self.btn_load_target.clicked.connect(self._on_load_target)
        self.btn_grab_target = QPushButton("Grab target")
        self.btn_grab_target.setToolTip("One oriented post-dark frame of the line target (same setting as the lamp)")
        self.btn_grab_target.clicked.connect(lambda: self.grabRequested.emit("target"))
        self.btn_load_white = QPushButton("Load white...")
        self.btn_load_white.setToolTip("Optional: same light and distance with the target removed (several files "
                                       "are averaged); divides out the illumination")
        self.btn_load_white.clicked.connect(self._on_load_white)
        self.btn_grab_white = QPushButton("Grab white")
        self.btn_grab_white.clicked.connect(lambda: self.grabRequested.emit("white"))
        self.btn_clear_white = QPushButton("Clear white")
        self.btn_clear_white.clicked.connect(self.clear_white)
        self.combo_polarity = QComboBox()
        for label, value in _POLARITIES:
            self.combo_polarity.addItem(label, value)
        self.combo_polarity.setToolTip("Bars darker (default) or brighter than the background")
        self.btn_analyze = QPushButton("Analyze Keystone")
        self.btn_analyze.setStyleSheet("font-weight:bold;")
        self.btn_analyze.clicked.connect(lambda: self.analyze())
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setFixedWidth(100)
        self.busy.setVisible(False)
        for wdg in (self.btn_load_target, self.btn_grab_target, self.btn_load_white, self.btn_grab_white,
                    self.btn_clear_white):
            row.addWidget(wdg)
        row.addWidget(QLabel("Bars:"))
        row.addWidget(self.combo_polarity)
        row.addStretch()
        row.addWidget(self.btn_analyze)
        row.addWidget(self.busy)
        root.addLayout(row)

        self.lbl_frames = QLabel()
        self.lbl_frames.setWordWrap(True)
        self.lbl_frames.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        root.addWidget(self.lbl_frames)
        self.lbl_state = QLabel()
        self.lbl_state.setWordWrap(True)
        self.lbl_state.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        root.addWidget(self.lbl_state)

        split = QSplitter(Qt.Vertical)
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setLineWrapMode(QTextEdit.WidgetWidth)
        self.report.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 9pt;")
        self.report.setMinimumHeight(90)
        split.addWidget(self.report)
        self.plot_tabs = QTabWidget()
        self._canvases: List[PlotCanvas] = []
        for index, (name, _file) in enumerate(_PLOTS):
            page = QWidget(self.plot_tabs)
            lay = QVBoxLayout(page)
            lay.setContentsMargins(2, 2, 2, 2)
            export_row = QHBoxLayout()
            export_row.addStretch()
            button = QPushButton("Export PNG")
            button.setToolTip("Export this plot as a presentation-quality PNG (300 dpi).")
            button.clicked.connect(lambda _checked=False, i=index: self._on_export_plot(i))
            export_row.addWidget(button)
            lay.addLayout(export_row)
            canvas = PlotCanvas(page)
            self._canvases.append(canvas)
            lay.addWidget(canvas, stretch=1)
            self.plot_tabs.addTab(page, name)
        split.addWidget(self.plot_tabs)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        split.setSizes([180, 540])
        root.addWidget(split, stretch=1)
        self.report.setPlainText("Keystone: measure the along-slit geometry with a line target (bars across the "
                                 "slit) after the smile analysis of the lamp frame. Apply / Save then write ONE "
                                 "smile + keystone model.")

    # ------------------------------------------------------------- context
    def set_context(self, analysis, lamp_shape=None, lamp_setting=None, reason: str = "") -> None:
        """New lamp frame or new smile analysis: the previous keystone result no longer applies.

        Target / white frames are kept while they still match the lamp frame (size and, if
        known, camera setting); otherwise they are dropped.

        Args:
            analysis: Smile analysis of the lamp (``None`` before analysis).
            lamp_shape: ``(H, W)`` of the lamp frame (``None`` = no frame).
            lamp_setting: Its ``CameraSetting`` (``None`` = unknown).
            reason: Shown in the page state line when a result was dropped.
        """
        had_result = self.result is not None
        self._smile = analysis
        self._lamp_shape = tuple(lamp_shape) if lamp_shape is not None else None
        self._lamp_setting = lamp_setting
        dropped = []
        for kind in ("target", "white"):
            frame = getattr(self, kind)
            if frame is None or self._lamp_shape is None:
                continue
            if tuple(frame.shape) != self._lamp_shape or _setting_problems(self._lamp_setting,
                                                                          getattr(self, kind + "_setting")):
                self._set_frame_attr(kind, None, None, "")
                dropped.append(kind)
        self.clear_result(reason if had_result else "")
        if dropped:
            self.lbl_state.setText(self.lbl_state.text() + "  " + " and ".join(dropped).capitalize()
                                   + " frame dropped: it does not match the new lamp frame.")

    def clear_result(self, reason: str = "") -> None:
        """Forget the computed keystone (report and plots return to the placeholder)."""
        self.result = None
        self.result_setting = None
        self._clear_plots("Run Analyze Keystone to see the check plots")
        if reason:
            self.report.setPlainText(reason)
        self._refresh(reason)

    def set_grab_enabled(self, enabled: bool) -> None:
        """Grab buttons follow the owner's grab availability / pending request."""
        self._grab_enabled = bool(enabled)
        busy = self._worker is not None             # same rule as _refresh: no grabs while analysing
        self.btn_grab_target.setEnabled(self._grab_enabled and not busy)
        self.btn_grab_white.setEnabled(self._grab_enabled and not busy)

    # -------------------------------------------------------------- frames
    def _set_frame_attr(self, kind, frame, setting, source) -> None:
        setattr(self, kind, None if frame is None else np.asarray(frame, dtype=np.float32))
        setattr(self, kind + "_setting", setting if frame is not None else None)
        setattr(self, "_%s_source" % kind, source if frame is not None else "")

    def set_frame(self, kind: str, frame: Optional[np.ndarray], setting=None, source: str = "") -> None:
        """Set the line-target (``kind="target"``) or white (``"white"``) frame.

        A keystone already computed stays attached to the owner's corrector until the next
        *Analyze Keystone* (Apply checks the setting of the target it was computed from).
        """
        if kind not in ("target", "white"):
            raise ValueError(f"Unknown keystone frame kind {kind!r}")
        if frame is not None and np.asarray(frame).ndim != 2:
            raise ValueError("A keystone frame must be 2-D (rows, columns)")
        self._set_frame_attr(kind, frame, setting, source)
        if frame is not None:
            h, w = self.target.shape if kind == "target" else self.white.shape
            self._status(f"{kind.capitalize()} frame {w}x{h} ({source or 'array'})")
        self._refresh()

    def set_target(self, frame, setting=None, source: str = "") -> None:
        """Shortcut for :meth:`set_frame` ``("target", ...)``."""
        self.set_frame("target", frame, setting, source)

    def set_white(self, frame, setting=None, source: str = "") -> None:
        """Shortcut for :meth:`set_frame` ``("white", ...)``."""
        self.set_frame("white", frame, setting, source)

    def clear_white(self) -> None:
        """Analyse without a white frame (the illumination is flattened along the slit only)."""
        self.set_frame("white", None)
        self._status("White frame cleared")

    def load_files(self, kind: str, paths: List[str]) -> None:
        """Load (and average) image files as the target or white frame.

        Raises:
            ValueError: Files of different sizes.
        """
        frames = [load_frame(p) for p in paths]
        if any(f.shape != frames[0].shape for f in frames):
            raise ValueError("All files must have the same size to be averaged")
        frame = frames[0] if len(frames) == 1 else np.mean(np.stack(frames), axis=0).astype(np.float32)
        setting = None
        if self._setting_for_file is not None:
            try:
                setting = self._setting_for_file(frame.shape)
            except Exception:  # noqa: BLE001
                setting = None
        names = ", ".join(os.path.basename(p) for p in paths)
        source = names if len(paths) == 1 else f"mean of {len(paths)}: {names}"
        self.set_frame(kind, frame, setting, source)

    def _on_load(self, kind: str) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, f"Keystone {kind} frame(s)", "", _IMAGE_FILTER)
        if paths:
            try:
                self.load_files(kind, paths)
            except Exception as exc:  # noqa: BLE001
                self._status(f"Load {kind} failed: {exc}")

    def _on_load_target(self) -> None:
        self._on_load("target")

    def _on_load_white(self) -> None:
        self._on_load("white")

    # ------------------------------------------------------------ analysis
    @property
    def running(self) -> bool:
        """True while a keystone worker is alive."""
        return self._worker is not None

    @property
    def polarity(self) -> str:
        """``"dark"`` or ``"bright"`` bars."""
        return self.combo_polarity.currentData()

    def set_polarity(self, polarity: str) -> None:
        """Select ``"dark"`` or ``"bright"`` bars."""
        index = self.combo_polarity.findData(polarity)
        if index < 0:
            raise ValueError(f"Unknown polarity {polarity!r}")
        self.combo_polarity.setCurrentIndex(index)

    def check(self) -> Tuple[bool, str]:
        """Whether *Analyze Keystone* can run now, and why not."""
        if self._worker is not None:
            return False, "Keystone analysis running..."
        if self._smile is None:
            return False, "Run the smile analysis of a lamp frame first."
        if self.target is None:
            return False, "Load or grab a line-target frame (bars across the slit)."
        lamp_shape = tuple(self._smile.frame_orig.shape)
        for kind, frame, setting in (("Target", self.target, self.target_setting),
                                     ("White", self.white, self.white_setting)):
            if frame is None:
                continue
            if tuple(frame.shape) != lamp_shape:
                return False, (f"{kind} frame {frame.shape[1]}x{frame.shape[0]} differs from the lamp frame "
                               f"{lamp_shape[1]}x{lamp_shape[0]}: use the same camera setting.")
            problems = _setting_problems(self._lamp_setting, setting)
            if problems:
                return False, f"{kind} frame was taken at another camera setting than the lamp: " + "; ".join(problems)
        return True, ""

    def analyze(self, synchronous: bool = False) -> bool:
        """Start the keystone analysis (worker thread, or here when ``synchronous``).

        Returns:
            ``False`` when it cannot run (reason in the state line / status).
        """
        ok, why = self.check()
        if not ok:
            self._status("Keystone not analysed: " + why)
            return False
        inputs = dict(target=self._target_source or "array", polarity=self.polarity)
        if self.white is not None:
            inputs["white"] = self._white_source or "array"
        kwargs = dict(white=None if self.white is None else self.white.copy(), polarity=self.polarity, inputs=inputs)
        self._pending_setting = self.target_setting
        analysis = self._smile
        if synchronous:
            try:
                k = analyze_keystone(analysis, self.target.copy(), **kwargs)
            except Exception as exc:  # noqa: BLE001
                self._on_failed(str(exc))
                return False
            self._on_done((analysis, k))
            return True
        worker = KeystoneWorker(analysis, self.target.copy(), kwargs)
        worker.done.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        self.busy.setVisible(True)
        self._status("Analyzing keystone (bar tracing + joint fit)...")
        self._refresh()
        worker.start()
        return True

    def _on_done(self, payload) -> None:
        analysis, k = payload
        if analysis is not self._smile:
            self._status("Smile analysis changed while the keystone was computed: run Analyze Keystone again")
            return
        self.result = k
        self.result_setting = self._pending_setting
        self.show_report(keystone_report(k), k.warnings)
        self._plot_target(k)
        self._plot_shift_map(k.map_y, k.model.dy.box, k.row_margin, k.margin)
        self._plot_retrace(k.qc)
        s = k.summary
        self._status(f"Keystone: max {s['keystone_max_px']:.2f} px, tilt {s['tilt_deg']:.3f} deg, "
                     f"{len(k.warnings)} warning(s)")
        self._refresh()
        self.keystoneReady.emit(k)

    def _on_failed(self, message: str) -> None:
        self.report.setPlainText("Keystone analysis failed:\n" + message)
        self._status("Keystone analysis failed: " + message)

    def _on_finished(self) -> None:
        if self._worker is not None:
            self._worker.wait()
            self._worker.deleteLater()
        self._worker = None
        self.busy.setVisible(False)
        # keep the worker's outcome (result / failure / discarded) on the state line, not "Ready"
        self._refresh(self.lbl_state.text())

    def wait_worker(self) -> None:
        """Block until a running keystone worker has finished (owner's ``closeEvent``)."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait()

    # ------------------------------------------------------------- display
    def show_report(self, text: str, warnings: Optional[List[str]] = None) -> None:
        """Show a keystone report; ``Warning:`` lines are highlighted."""
        rows = []
        for line in text.splitlines():
            escaped = html.escape(line)
            if line.startswith("Warning:"):
                rows.append(f"<span style='color:#c0392b;font-weight:bold'>{escaped}</span>")
            else:
                rows.append(escaped)
        if warnings is not None and not warnings:
            rows.append("<span style='color:#1e8449'>No warnings.</span>")
        self.report.setHtml("<pre style='margin:0'>" + "<br>".join(rows) + "</pre>")

    def show_stored(self, keystone: Optional[dict], map_y: Optional[np.ndarray] = None, source: str = "") -> None:
        """Show the keystone stored in a loaded model file (``None``: the file is smile only)."""
        self.result = None
        self.result_setting = None
        self._clear_plots("Loaded model: the target plots need Analyze Keystone")
        if not keystone:
            self.report.setPlainText(f"Loaded model {source}: smile only (no keystone).")
            self._refresh()
            return
        try:
            text = keystone_report(keystone)
        except Exception as exc:  # noqa: BLE001
            text = f"Stored keystone report unreadable: {exc}"
        self.show_report(f"Loaded model {source}\n" + text, keystone.get("warnings"))
        model = keystone.get("model") or {}
        dy = (model.get("dy") or {}).get("box")
        if map_y is not None:
            self._plot_shift_map(np.asarray(map_y), dy, keystone.get("row_margin") or (0, 0),
                                 keystone.get("margin") or (0, 0))
        self._plot_retrace(keystone.get("qc") or {})
        self._refresh("Showing the keystone of the loaded model.")

    def _clear_plots(self, message: str) -> None:
        for canvas in self._canvases:
            canvas.clear()
            ax = canvas.fig.add_subplot(111)
            ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            canvas.refresh()

    def _plot_target(self, k) -> None:
        canvas = self._canvases[0]
        canvas.clear()
        fig = canvas.fig
        before = np.asarray(k.target_norm, dtype=np.float32)
        after = np.asarray(k.target_corr, dtype=np.float32)
        finite = before[np.isfinite(before)]
        vmin, vmax = (np.percentile(finite, [1, 99.5]) if finite.size else (0, 1))
        h, w = before.shape
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1, sharey=ax1)
        ax1.imshow(before, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
        ax1.set_title("Before: normalised target (raw geometry); lines = corrected bar rows", fontsize=9)
        ax2.imshow(after, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
        ax2.set_title("After smile + keystone remap: bars should follow the lines", fontsize=9)
        for Y in k.model.target_Y:
            for ax in (ax1, ax2):
                ax.axhline(Y, color="#e67e22", lw=.6, alpha=.8)
        left, right = k.margin
        top, bottom = k.row_margin
        for x in (left - .5, w - right - .5):
            ax2.axvline(x, color="#2980b9", lw=.8, ls="--")
        for y in (top - .5, h - bottom - .5):
            ax2.axhline(y, color="#2980b9", lw=.8, ls="--")
        ax1.set_ylabel("Row (slit)")
        ax2.set_ylabel("Row (slit)")
        ax2.set_xlabel("Column (spectral)")
        canvas.refresh()

    def _plot_shift_map(self, map_y, box, row_margin, margin) -> None:
        canvas = self._canvases[1]
        canvas.clear()
        ax = canvas.fig.add_subplot(111)
        map_y = np.asarray(map_y, dtype=np.float32)
        h, w = map_y.shape
        dy = map_y - np.arange(h, dtype=np.float32)[:, None]
        lim = float(np.nanmax(np.abs(dy))) if np.isfinite(dy).any() else 1.0
        lim = max(lim, 1e-3)
        im = ax.imshow(dy, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto",
                       extent=(-.5, w - .5, h - .5, -.5))
        canvas.fig.colorbar(im, ax=ax, label="Row shift map_y - Y (px)")
        try:
            levels = np.linspace(-lim, lim, 11)
            cs = ax.contour(np.arange(w), np.arange(h), dy, levels=levels, colors="k", linewidths=.4)
            ax.clabel(cs, fmt="%.1f", fontsize=6)
        except Exception:  # noqa: BLE001  (flat map: no contour)
            pass
        if box is not None and len(box) == 4:
            c0, c1, r0, r1 = (float(v) for v in box)
            ax.plot([c0, c1, c1, c0, c0], [r0, r0, r1, r1, r0], color="#27ae60", lw=1, ls="--",
                    label="measured bars")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(-.5, w - .5)
        ax.set_ylim(h - .5, -.5)
        ax.set_xlabel("Column (spectral)")
        ax.set_ylabel("Row (slit)")
        ax.set_title(f"Keystone row shift (source row - destination row); border rows {tuple(row_margin)}, "
                     f"columns {tuple(margin)}", fontsize=9)
        canvas.refresh()

    def _plot_retrace(self, qc: dict) -> None:
        canvas = self._canvases[2]
        canvas.clear()
        fig = canvas.fig
        bars = qc.get("retrace_target_per_bar") or []
        lamps = qc.get("retrace_lamp_per_line") or []
        ax1 = fig.add_subplot(211)
        if bars:
            y = np.array([_num(p.get("Y")) for p in bars])
            ax1.plot(y, [_num(p.get("rms_px")) for p in bars], "o-", ms=3, label="RMS")
            ax1.plot(y, [_num(p.get("max_px")) for p in bars], "x", ms=3, alpha=.5, label="max |dev|")
        ax1.axhline(RETRACE_WARN_PX, color="#e67e22", lw=.8, ls="--", label=f"{RETRACE_WARN_PX:g} px")
        ax1.axhline(2 * RETRACE_WARN_PX, color="#c0392b", lw=.8, ls=":", label=f"{2 * RETRACE_WARN_PX:g} px")
        ax1.set(xlabel="Bar row (corrected)", ylabel="Re-trace residual (px)",
                title="Target bars re-traced on the corrected frame (0 = straight)")
        ax1.grid(alpha=.2)
        ax1.legend(fontsize=7)
        ax2 = fig.add_subplot(212)
        if lamps:
            x = np.array([_num(p.get("X")) for p in lamps])
            ax2.plot(x, [_num(p.get("rms_px")) for p in lamps], "s-", ms=3, color="#8e44ad", label="RMS")
            ax2.plot(x, [_num(p.get("max_px")) for p in lamps], "x", ms=3, color="#8e44ad", alpha=.5,
                     label="max |dev|")
        ax2.axhline(RETRACE_WARN_PX, color="#e67e22", lw=.8, ls="--")
        ax2.set(xlabel="Lamp line column (corrected)", ylabel="Re-trace residual (px)",
                title="Lamp lines re-traced on the corrected frame")
        ax2.grid(alpha=.2)
        if lamps:
            ax2.legend(fontsize=7)
        canvas.refresh()

    def _on_export_plot(self, index: int) -> None:
        """Save one plot as a 300-dpi PNG."""
        path, _ = QFileDialog.getSaveFileName(self, "Export plot as PNG", _PLOTS[index][1], "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            self._canvases[index].fig.savefig(path, format="png", dpi=300, bbox_inches="tight", facecolor="white")
            self._status(f"Exported plot PNG: {path} (300 dpi)")
        except Exception as exc:  # noqa: BLE001
            self._status(f"Plot export failed: {exc}")

    # ------------------------------------------------------------- helpers
    def _status(self, message: str) -> None:
        self.lbl_state.setText(message)
        self.statusChanged.emit(message)

    def _describe(self, kind: str) -> str:
        frame = getattr(self, kind)
        if frame is None:
            return f"{kind.capitalize()}: none"
        h, w = frame.shape
        return f"{kind.capitalize()}: {w}x{h} ({getattr(self, '_%s_source' % kind) or 'array'})"

    def _refresh(self, message: str = "") -> None:
        """Enable buttons and update the frame / state lines."""
        ok, why = self.check()
        self.btn_analyze.setEnabled(ok)
        self.btn_clear_white.setEnabled(self.white is not None)
        busy = self._worker is not None
        for b in (self.btn_load_target, self.btn_load_white):
            b.setEnabled(not busy)
        self.btn_grab_target.setEnabled(self._grab_enabled and not busy)
        self.btn_grab_white.setEnabled(self._grab_enabled and not busy)
        self.combo_polarity.setEnabled(not busy)
        frames = self._describe("target") + "   |   " + self._describe("white")
        if self.white is None:
            frames += " (optional, recommended)"
        self.lbl_frames.setText(frames)
        if message:
            self.lbl_state.setText(message)
        elif self.result is not None:
            pass                                        # keep the result status
        elif not ok:
            self.lbl_state.setText(why)
        else:
            self.lbl_state.setText("Ready: Analyze Keystone")
