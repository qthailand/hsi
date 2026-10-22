"""Non-modal Segment dialog of the cube viewer (Phase 2): classify, separability, compare.

``SegmentDialog(viewer)`` is a thin view over the viewer's
:class:`~modules.cube_viewer.segment_session.SegmentSession`; every computation runs in
the session's own worker thread, so the dialog and the ROI tools stay responsive.

Tabs:
    * **Classify** - method dropdown (every ``segment.METHODS`` entry; methods that need
      scikit-learn are disabled with a tooltip when it is missing), the method's own
      parameters, the estimated time (``session.estimate_seconds``), Run / Cancel with a
      progress bar, one reject slider (global multiplier 0.5-3x of the per-class default
      thresholds for angle / distance / divergence scores, minimum confidence 0-1 for
      probabilities; re-thresholds the stored maps, never recomputes), 'Show overlay',
      opacity, 'Clear segmentation', the stale badge and a per-class result table.
    * **Separability** - spectral-angle and JM matrices (cells shaded by value), the
      per-band ANOVA-F curve against wavelength, the ROI-consistency confusion matrix with
      its mode, and the warnings.
    * **Compare** - check list of methods, Run compare / Cancel, a results table with the
      best row (macro-F1, then accuracy) highlighted and 'Use this method'.

Threading:
    GUI thread only.  The viewer owns the single instance and closes it in its
    ``closeEvent`` (the session's ``shutdown`` stops the jobs).

Example:
    >>> dlg = SegmentDialog(viewer)                     # doctest: +SKIP
    >>> dlg.show()                                      # doctest: +SKIP
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import qtawesome as qta
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                             QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                             QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QSlider,
                             QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                             QWidget)

from modules.cube_viewer import ml as _ml
from modules.cube_viewer import segment as _segment
from modules.cube_viewer.segment_session import (JOB_COMPARE, JOB_RUN, JOB_SEPARABILITY,
                                                 SCALE_RANGE)
from modules.ui_kit.theme import ACCENT, CARD, PLOT_BACKGROUND, PLOT_FOREGROUND

__all__ = ("SegmentDialog", "RETHRESHOLD_DELAY_MS")

#: Debounce of the reject slider (one re-threshold costs about 20 ms on 1024x1762).
RETHRESHOLD_DELAY_MS = 60
_STALE_COLOR = "#e0a030"
_WARN_COLOR = "#e0a030"
_INT_RANGE = (1, 500)
_FLOAT_RANGE = (0.0, 1e6)


def _fmt_seconds(s) -> str:
    """Human-readable duration."""
    if s is None or not np.isfinite(s):
        return "-"
    if s < 1.0:
        return f"{s * 1000:.0f} ms"
    if s < 120.0:
        return f"{s:.1f} s"
    return f"{s / 60:.1f} min"


def _shade(fraction: float) -> QColor:
    """Cell colour from the card background (0) to the accent (1)."""
    f = min(max(float(fraction), 0.0), 1.0) if np.isfinite(fraction) else 0.0
    a, b = QColor(CARD), QColor(ACCENT)
    return QColor(int(a.red() + (b.red() - a.red()) * f),
                  int(a.green() + (b.green() - a.green()) * f),
                  int(a.blue() + (b.blue() - a.blue()) * f))


def _item(text, align=Qt.AlignRight | Qt.AlignVCenter) -> QTableWidgetItem:
    """Read-only table item."""
    it = QTableWidgetItem(str(text))
    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
    it.setTextAlignment(int(align))
    return it


def _table(columns=()) -> QTableWidget:
    """Read-only, row-selecting table."""
    t = QTableWidget(0, len(columns))
    t.setHorizontalHeaderLabels(list(columns))
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setSelectionMode(QAbstractItemView.SingleSelection)
    t.verticalHeader().setVisible(False)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    return t


class SegmentDialog(QDialog):
    """Non-modal dialog driving ``viewer.segment_session`` (see the module docstring).

    Args:
        viewer: The owning :class:`~modules.cube_viewer.window.CubeViewerWindow`.
    """

    def __init__(self, viewer):
        super().__init__(viewer)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setModal(False)
        self.setWindowTitle("Segment")
        self.setWindowIcon(qta.icon("mdi.shape-plus", color=PLOT_FOREGROUND))
        self.viewer = viewer
        self.session = viewer.segment_session
        self._param_widgets: dict = {}
        self._compare_rows: list = []
        self._separability = None
        self._last_job = JOB_RUN
        self._reject_timer = QTimer(self)
        self._reject_timer.setSingleShot(True)
        self._reject_timer.setInterval(RETHRESHOLD_DELAY_MS)
        self._reject_timer.timeout.connect(self.apply_reject)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_classify(), "Classify")
        self.tabs.addTab(self._build_separability(), "Separability")
        self.tabs.addTab(self._build_compare(), "Compare")
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.addWidget(self.tabs)
        lay.addWidget(self.status)
        self.resize(560, 680)

        s = self.session
        s.state_changed.connect(self.refresh)
        viewer.spectra_panel.segment_ready_changed.connect(self.refresh)
        s.progress.connect(self._on_progress)
        s.result_ready.connect(self._on_result)
        s.separability_ready.connect(self.show_separability)
        s.compare_ready.connect(self.show_compare)
        s.failed.connect(self._on_failed)
        s.confirm_needed.connect(self._on_confirm_needed)
        self._select_method_in_combo(s.method)
        self._on_method_changed()
        self.refresh()

    # ── Classify tab ──────────────────────────────────────

    def _build_classify(self) -> QWidget:
        """Method, parameters, run, reject, overlay and result table."""
        w = QWidget()
        lay = QVBoxLayout(w)
        self.method_combo = QComboBox()
        model = self.method_combo.model()
        for key, info in _segment.METHODS.items():
            self.method_combo.addItem(info.label, key)
            row = self.method_combo.count() - 1
            tip = info.description or info.label
            if not info.available:
                model.item(row).setEnabled(False)
                tip = "Needs scikit-learn (not installed). " + tip
            self.method_combo.setItemData(row, tip, Qt.ToolTipRole)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        form = QFormLayout()
        form.addRow("Method", self.method_combo)
        lay.addLayout(form)
        self.method_help = QLabel("")
        self.method_help.setWordWrap(True)
        lay.addWidget(self.method_help)
        self.params_box = QGroupBox("Parameters")
        self.params_form = QFormLayout(self.params_box)
        lay.addWidget(self.params_box)
        self.estimate_label = QLabel("")
        lay.addWidget(self.estimate_label)

        row = QHBoxLayout()
        self.btn_run = QPushButton(qta.icon("mdi.play", color=PLOT_FOREGROUND), "Run")
        self.btn_run.clicked.connect(self.run)
        self.btn_cancel = QPushButton(qta.icon("mdi.stop", color=PLOT_FOREGROUND), "Cancel")
        self.btn_cancel.clicked.connect(lambda: self.session.cancel(JOB_RUN))
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.progress, 1)
        lay.addLayout(row)

        box = QGroupBox("Result")
        g = QVBoxLayout(box)
        rrow = QHBoxLayout()
        self.reject_label = QLabel("Reject")
        self.reject_slider = QSlider(Qt.Horizontal)
        self.reject_slider.valueChanged.connect(self._on_reject_moved)
        self.reject_value = QLabel("")
        self.reject_value.setMinimumWidth(110)
        rrow.addWidget(self.reject_label)
        rrow.addWidget(self.reject_slider, 1)
        rrow.addWidget(self.reject_value)
        g.addLayout(rrow)
        orow = QHBoxLayout()
        self.chk_overlay = QCheckBox("Show overlay")
        self.chk_overlay.setChecked(self.session.overlay_visible)
        self.chk_overlay.toggled.connect(self.session.set_overlay_visible)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(round(self.session.opacity * 100)))
        self.opacity_slider.valueChanged.connect(lambda v: self.session.set_opacity(v / 100.0))
        self.btn_clear = QPushButton(qta.icon("mdi.eraser", color=PLOT_FOREGROUND), "Clear segmentation")
        self.btn_clear.clicked.connect(self.viewer.clear_segmentation)
        orow.addWidget(self.chk_overlay)
        orow.addWidget(QLabel("Opacity"))
        orow.addWidget(self.opacity_slider, 1)
        orow.addWidget(self.btn_clear)
        g.addLayout(orow)
        self.result_label = QLabel("No result")
        self.result_label.setWordWrap(True)
        g.addWidget(self.result_label)
        self.result_table = _table(("", "Class", "Pixels", "%"))
        g.addWidget(self.result_table, 1)
        lay.addWidget(box, 1)
        return w

    def current_method(self) -> str:
        """Method key selected in the dropdown."""
        return self.method_combo.currentData()

    def _select_method_in_combo(self, key) -> bool:
        """Select ``key`` in the dropdown (False when absent or disabled)."""
        i = self.method_combo.findData(key)
        if i < 0 or not _segment.METHODS[key].available:
            return False
        self.method_combo.setCurrentIndex(i)
        return True

    def select_method(self, key: str) -> bool:
        """Select method ``key`` (dropdown and session); False when unknown or unavailable."""
        ok = self._select_method_in_combo(key)
        if ok:
            self.tabs.setCurrentIndex(0)
        return ok

    def _defaults(self, key) -> dict:
        """Default parameters of ``key``."""
        if key == "random_forest":
            return dict(_ml.RF_PARAMS)
        if key == "svm":
            return dict(_ml.SVM_PARAMS)
        return dict(_segment.METHODS[key].params)

    def _on_method_changed(self, *_):
        """Rebuild the parameter form of the selected method."""
        key = self.current_method()
        if key is None:
            return
        info = _segment.METHODS[key]
        if self.session.method != key:
            self.session.set_method(key)
        self.method_help.setText(info.description)
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self._param_widgets = {}
        values = self._defaults(key)
        values.update(self.session.params(key))
        specs = _ml.PARAM_SPECS.get(key, {})
        for name, value in values.items():
            lo, hi, tip = specs.get(name, (None, None, ""))
            if isinstance(value, bool):
                wdg = QCheckBox()
                wdg.setChecked(value)
                wdg.toggled.connect(self._on_param_changed)
            elif isinstance(value, int):
                wdg = QSpinBox()
                wdg.setRange(int(lo if lo is not None else _INT_RANGE[0]),
                             int(hi if hi is not None else _INT_RANGE[1]))
                wdg.setValue(value)
                wdg.valueChanged.connect(self._on_param_changed)
            else:
                wdg = QDoubleSpinBox()
                wdg.setDecimals(4)
                wdg.setRange(float(lo if lo is not None else _FLOAT_RANGE[0]),
                              float(hi if hi is not None else _FLOAT_RANGE[1]))
                wdg.setValue(float(value))
                wdg.valueChanged.connect(self._on_param_changed)
            if name == "pca_dims" and info.family == "statistical":
                wdg.setMinimum(1)
                tip = tip or "Principal components kept before the covariance estimate."
            wdg.setToolTip(tip)
            self.params_form.addRow(name, wdg)
            self._param_widgets[name] = wdg
        self.params_box.setVisible(bool(values))
        self._update_estimate()

    def param_values(self) -> dict:
        """Current values of the parameter form."""
        out = {}
        for name, wdg in self._param_widgets.items():
            out[name] = wdg.isChecked() if isinstance(wdg, QCheckBox) else wdg.value()
        return out

    def _on_param_changed(self, *_):
        """Store the form's parameters in the session and refresh the estimate."""
        self.session.set_params(self.current_method(), self.param_values())
        self._update_estimate()

    def _update_estimate(self):
        """Estimated fit + full-image time of the selected method."""
        try:
            sec = self.session.estimate_seconds(self.current_method())
            self.estimate_label.setText(f"Estimated time: ~{_fmt_seconds(sec)} (fit + full image)")
        except Exception as exc:                        # noqa: BLE001 - shown to the user
            self.estimate_label.setText(f"Estimated time: - ({exc})")

    def run(self):
        """Run the selected method (needs >= 2 classes with ready spectra)."""
        if not self.session.can_run():
            self.status.setText("Need at least 2 classes with ready spectra (set the Class column).")
            return
        self.session.set_params(self.current_method(), self.param_values())
        self._last_job = JOB_RUN
        self.progress.setValue(0)
        self.status.setText("")
        self.session.run()

    def _on_confirm_needed(self, seconds: float, detail: str):
        """A fitted run would be slow: ask before classifying the whole image."""
        self.estimate_label.setText(f"Estimated time (measured after fit): ~{seconds:.0f} s")
        answer = QMessageBox.question(
            self, "Slow classification",
            f"{detail}.\n\nClassify the whole image anyway?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            self.status.setText(f"Classifying (about {seconds:.0f} s)\u2026")
            self.session.continue_run()
        else:
            self.session.discard_pending()
            self.progress.setValue(0)
            self.status.setText("Run cancelled before classifying (too slow); try a faster "
                                "method or fewer training pixels / PCA components.")

    # ── reject slider ─────────────────────────────────────

    def _configure_reject(self):
        """Slider meaning: multiplier (angle / distance) or minimum confidence (probability)."""
        self.reject_slider.blockSignals(True)
        if self.session.is_probability:
            self.reject_label.setText("Min. confidence")
            self.reject_slider.setRange(0, 100)
            self.reject_slider.setValue(int(round(self.session.min_confidence * 100)))
            self.reject_slider.setToolTip("Pixels whose best-class probability is below this "
                                          "are unclassified.")
        else:
            self.reject_label.setText("Threshold x")
            self.reject_slider.setRange(int(SCALE_RANGE[0] * 100), int(SCALE_RANGE[1] * 100))
            self.reject_slider.setValue(int(round(self.session.threshold_scale * 100)))
            self.reject_slider.setToolTip("Multiplier of each class's default threshold "
                                          "(1.0 = 1.5 x P99 of its own ROI pixels).")
        self.reject_slider.blockSignals(False)
        self._show_reject_value()

    def _slider_value(self) -> float:
        """Slider position as a confidence or multiplier."""
        return self.reject_slider.value() / 100.0

    def _show_reject_value(self):
        """Text next to the slider."""
        v = self._slider_value()
        self.reject_value.setText(f"p >= {v:.2f}" if self.session.is_probability else f"{v:.2f} x")

    def _on_reject_moved(self, *_):
        """Debounced re-threshold."""
        self._show_reject_value()
        self._reject_timer.start()

    def apply_reject(self):
        """Apply the slider to the stored maps now (no recompute)."""
        self._reject_timer.stop()
        if self.session.is_probability:
            self.session.set_min_confidence(self._slider_value())
        else:
            self.session.set_threshold_scale(self._slider_value())

    # ── refresh ───────────────────────────────────────────

    def refresh(self):
        """Update buttons, badges and the result table from the session."""
        s = self.session
        if getattr(s, "_closed", False):
            return
        running = s.job_pending(JOB_RUN)
        self.btn_run.setEnabled(not running and s.can_run())
        self.btn_cancel.setEnabled(running)
        self.btn_clear.setEnabled(s.has_result or running)
        self.chk_overlay.blockSignals(True)
        self.chk_overlay.setChecked(s.overlay_visible)
        self.chk_overlay.blockSignals(False)
        sep_running = s.job_pending(JOB_SEPARABILITY)
        self.btn_sep.setEnabled(not sep_running)
        self.btn_sep_cancel.setEnabled(sep_running)
        cmp_running = s.job_pending(JOB_COMPARE)
        self.btn_compare.setEnabled(not cmp_running)
        self.btn_compare_cancel.setEnabled(cmp_running)
        if not running:
            self.progress.setValue(1000 if s.has_result else 0)
        self._fill_result_table()

    def _fill_result_table(self):
        """Per-class pixels and % of the thresholded result."""
        summary = self.session.summary()
        t = self.result_table
        if summary is None:
            t.setRowCount(0)
            self.result_label.setText("No result")
            self.reject_slider.setEnabled(False)
            return
        self.reject_slider.setEnabled(True)
        total = max(summary["total"], 1)
        text = f"{summary['label']}: {len(summary['names'])} classes"
        if summary["stale"]:
            text += f" <span style='color:{_STALE_COLOR}'><b>(stale - ROIs changed, re-run)</b></span>"
        if self.session.warnings:
            text += "<br><span style='color:%s'>%s</span>" % (
                _WARN_COLOR, "<br>".join(str(x) for x in self.session.warnings))
        self.result_label.setText(text)
        rows = list(zip(summary["colors"], summary["names"], summary["counts"]))
        rows.append((None, "unclassified", summary["n_unclassified"]))
        t.setRowCount(len(rows))
        for r, (color, name, n) in enumerate(rows):
            sw = _item("")
            if color is not None:
                sw.setBackground(QBrush(QColor(color)))
            t.setItem(r, 0, sw)
            t.setItem(r, 1, _item(name, Qt.AlignLeft | Qt.AlignVCenter))
            t.setItem(r, 2, _item(f"{n:,}"))
            t.setItem(r, 3, _item(f"{100.0 * n / total:.2f}"))

    def is_stale_shown(self) -> bool:
        """True when the stale badge is shown."""
        return "stale" in self.result_label.text()

    def _on_result(self):
        """A run finished."""
        self._configure_reject()
        r = self.session.result
        self.status.setText(f"Classified in {_fmt_seconds(r.elapsed_s)}" if r is not None else "")
        self.refresh()

    def _on_progress(self, fraction, text):
        """Route the session progress to the bar of the job that reports it."""
        if text == "separability":
            job = JOB_SEPARABILITY
        elif text == "comparing methods":
            job = JOB_COMPARE
        elif text in ("fitting", "classifying"):
            job = JOB_RUN
        else:
            job = self._last_job
        bar = {JOB_RUN: self.progress, JOB_SEPARABILITY: self.sep_progress,
               JOB_COMPARE: self.compare_progress}[job]
        bar.setValue(int(round(fraction * 1000)))
        bar.setFormat(f"{text} %p%")

    def _on_failed(self, message):
        """A job failed."""
        self.status.setText(f"<span style='color:{_WARN_COLOR}'>Failed: {message}</span>")
        self.refresh()

    # ── Separability tab ──────────────────────────────────

    def _build_separability(self) -> QWidget:
        """Angle / JM / F-curve / confusion / warnings."""
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        self.btn_sep = QPushButton(qta.icon("mdi.chart-scatter-plot", color=PLOT_FOREGROUND),
                                   "Compute separability")
        self.btn_sep.clicked.connect(self.compute_separability)
        self.btn_sep_cancel = QPushButton("Cancel")
        self.btn_sep_cancel.clicked.connect(lambda: self.session.cancel(JOB_SEPARABILITY))
        self.sep_progress = QProgressBar()
        self.sep_progress.setRange(0, 1000)
        row.addWidget(self.btn_sep)
        row.addWidget(self.btn_sep_cancel)
        row.addWidget(self.sep_progress, 1)
        lay.addLayout(row)
        tabs = QTabWidget()
        self.angle_table = _table()
        self.jm_table = _table()
        tabs.addTab(self.angle_table, "Angle (deg)")
        tabs.addTab(self.jm_table, "JM (0-2)")
        self.f_plot = pg.PlotWidget(background=PLOT_BACKGROUND)
        self.f_plot.setLabel("bottom", "Wavelength (nm)")
        self.f_plot.setLabel("left", "ANOVA F")
        self.f_curve = self.f_plot.plot([], [], pen=pg.mkPen(ACCENT, width=2))
        tabs.addTab(self.f_plot, "F per band")
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(0, 0, 0, 0)
        self.confusion_mode = QLabel("")
        self.confusion_mode.setWordWrap(True)
        self.confusion_table = _table()
        cl.addWidget(self.confusion_mode)
        cl.addWidget(self.confusion_table, 1)
        tabs.addTab(cw, "ROI consistency")
        self.sep_tabs = tabs
        lay.addWidget(tabs, 1)
        lay.addWidget(QLabel("Warnings"))
        self.sep_warnings = QListWidget()
        self.sep_warnings.setMaximumHeight(140)
        lay.addWidget(self.sep_warnings)
        return w

    def compute_separability(self):
        """Run the separability analysis (confusion with the selected method)."""
        if not self.session.can_run():
            self.status.setText("Need at least 2 classes with ready spectra.")
            return
        key = self.current_method()
        self._last_job = JOB_SEPARABILITY
        self.sep_progress.setValue(0)
        self.session.compute_separability(key, self.session.params(key) or None)

    def _matrix(self, table, names, m, lo, hi, fmt, invert=False):
        """Fill a K x K table, cells shaded by value between ``lo`` and ``hi``."""
        k = len(names)
        table.setRowCount(k)
        table.setColumnCount(k)
        table.setHorizontalHeaderLabels(names)
        table.setVerticalHeaderLabels(names)
        table.verticalHeader().setVisible(True)
        span = (hi - lo) or 1.0
        for i in range(k):
            for j in range(k):
                v = float(m[i][j])
                it = _item(fmt.format(v) if np.isfinite(v) else "-", Qt.AlignCenter)
                if i != j:
                    f = (v - lo) / span
                    it.setBackground(QBrush(_shade(1.0 - f if invert else f)))
                table.setItem(i, j, it)

    def show_separability(self, out):
        """Fill the Separability tab from a ``segment.separability`` dict."""
        self._separability = out
        names = list(out.get("class_names", []))
        ang = np.asarray(out["angle_deg"], dtype=float)
        off = ang[~np.eye(len(names), dtype=bool)] if len(names) > 1 else np.zeros(1)
        self._matrix(self.angle_table, names, ang, 0.0, float(np.nanmax(off)) or 1.0, "{:.2f}")
        self._matrix(self.jm_table, names, out["jm"], 0.0, 2.0, "{:.3f}")
        f = np.asarray(out["f_curve"], dtype=float)
        wl = self.viewer.wavelengths
        x = np.asarray(wl, dtype=float) if wl is not None and len(wl) == f.size else np.arange(f.size)
        self.f_plot.setLabel("bottom", "Wavelength (nm)" if wl is not None and len(wl) == f.size else "Band")
        good = np.isfinite(f)
        self.f_curve.setData(x[good], f[good])
        conf = np.asarray(out["confusion"])
        k = len(names)
        t = self.confusion_table
        t.setRowCount(k)
        t.setColumnCount(k)
        t.setHorizontalHeaderLabels(names)
        t.setVerticalHeaderLabels(names)
        t.verticalHeader().setVisible(True)
        for i in range(k):
            tot = max(int(conf[i].sum()), 1)
            for j in range(k):
                it = _item(str(int(conf[i, j])), Qt.AlignCenter)
                it.setBackground(QBrush(_shade(conf[i, j] / tot if i == j else 0.0)))
                if i != j and conf[i, j]:
                    it.setForeground(QBrush(QColor(_WARN_COLOR)))
                t.setItem(i, j, it)
        note = out.get("confusion_note", "")
        self.confusion_mode.setText(f"Mode: {out.get('confusion_mode', '-')} (optimistic: same image, "
                                    f"rows = true class, columns = predicted). {note}")
        self.sep_warnings.clear()
        for msg in list(out.get("build_warnings", [])) + list(out.get("warnings", [])):
            self.sep_warnings.addItem(str(msg))
        if not self.sep_warnings.count():
            self.sep_warnings.addItem("No warnings")
        self.sep_progress.setValue(1000)
        self.refresh()

    # ── Compare tab ───────────────────────────────────────

    def _build_compare(self) -> QWidget:
        """Method check list, run, results table."""
        w = QWidget()
        lay = QVBoxLayout(w)
        self.compare_list = QListWidget()
        for key, info in _segment.METHODS.items():
            it = QListWidgetItem(info.label)
            it.setData(Qt.UserRole, key)
            if info.available:
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked)
            else:
                it.setFlags(Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Unchecked)
                it.setToolTip("Needs scikit-learn (not installed).")
            self.compare_list.addItem(it)
        self.compare_list.setMaximumHeight(200)
        lay.addWidget(self.compare_list)
        row = QHBoxLayout()
        self.btn_compare = QPushButton(qta.icon("mdi.compare-horizontal", color=PLOT_FOREGROUND),
                                       "Run compare")
        self.btn_compare.clicked.connect(self.run_compare)
        self.btn_compare_cancel = QPushButton("Cancel")
        self.btn_compare_cancel.clicked.connect(lambda: self.session.cancel(JOB_COMPARE))
        self.compare_progress = QProgressBar()
        self.compare_progress.setRange(0, 1000)
        row.addWidget(self.btn_compare)
        row.addWidget(self.btn_compare_cancel)
        row.addWidget(self.compare_progress, 1)
        lay.addLayout(row)
        self.compare_table = _table(("Method", "Accuracy", "Macro-F1", "Fit", "Est. full image",
                                     "Mode"))
        lay.addWidget(self.compare_table, 1)
        note = QLabel("Accuracy on held-out ROIs of the same image (leave-one-ROI-out, else "
                      "split-half): optimistic, use it to rank methods.")
        note.setWordWrap(True)
        lay.addWidget(note)
        self.btn_use = QPushButton(qta.icon("mdi.check", color=PLOT_FOREGROUND), "Use this method")
        self.btn_use.clicked.connect(self.use_selected_method)
        lay.addWidget(self.btn_use)
        return w

    def checked_methods(self) -> list:
        """Method keys checked in the Compare list."""
        out = []
        for i in range(self.compare_list.count()):
            it = self.compare_list.item(i)
            if it.checkState() == Qt.Checked and it.flags() & Qt.ItemIsEnabled:
                out.append(it.data(Qt.UserRole))
        return out

    def run_compare(self):
        """Compare the checked methods in the worker."""
        keys = self.checked_methods()
        if not keys:
            self.status.setText("Check at least one method to compare.")
            return
        if not self.session.can_run():
            self.status.setText("Need at least 2 classes with ready spectra.")
            return
        self._last_job = JOB_COMPARE
        self.compare_progress.setValue(0)
        self.session.run_compare(keys)

    def best_row(self):
        """Index of the best compare row (macro-F1, then accuracy), or None."""
        best, best_key = None, None
        for i, r in enumerate(self._compare_rows):
            if r.get("macro_f1") is None:
                continue
            key = (r["macro_f1"], r.get("accuracy") or 0.0)
            if best_key is None or key > best_key:
                best, best_key = i, key
        return best

    def show_compare(self, rows):
        """Fill the Compare table and highlight the best row."""
        self._compare_rows = list(rows)
        t = self.compare_table
        t.setRowCount(len(rows))
        best = self.best_row()

        def num(v, fmt):
            return "-" if v is None else fmt.format(v)

        for i, r in enumerate(rows):
            cells = [r.get("label", r.get("method")), num(r.get("accuracy"), "{:.3f}"),
                     num(r.get("macro_f1"), "{:.3f}"), _fmt_seconds(r.get("fit_s")),
                     _fmt_seconds(r.get("est_full_image_s")),
                     r.get("mode") if r.get("error") is None else f"error: {r['error']}"]
            for c, text in enumerate(cells):
                it = _item(text, Qt.AlignLeft | Qt.AlignVCenter if c in (0, 5) else
                           Qt.AlignRight | Qt.AlignVCenter)
                if i == best:
                    it.setBackground(QBrush(QColor(ACCENT).darker(160)))
                    font = it.font()
                    font.setBold(True)
                    it.setFont(font)
                t.setItem(i, c, it)
        if best is not None:
            t.selectRow(best)
        self.compare_progress.setValue(1000)
        self.refresh()

    def use_selected_method(self) -> bool:
        """Select the chosen (else best) compared method in the Classify tab."""
        rows = self.compare_table.selectionModel().selectedRows() if self._compare_rows else []
        i = rows[0].row() if rows else self.best_row()
        if i is None or i >= len(self._compare_rows):
            return False
        return self.select_method(self._compare_rows[i]["method"])

    # ── Qt ────────────────────────────────────────────────

    def showEvent(self, event):
        """Refresh the estimate (class count may have changed) when shown."""
        super().showEvent(event)
        self._update_estimate()
        self.refresh()
