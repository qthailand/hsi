"""Qt panel for FWHM / spectral-resolution measurement of a calibration-lamp frame.

Workflow - *select, then calculate*:

1. *Grab from Scan* (or *Load Image*): the wavelength profile and the smile model stored for
   the frame's camera setting are loaded automatically and the frame is smile corrected.
2. A **preview spectrum** is shown: the mean of all rows when a smile model is applied, else
   the mean of the 16 rows around the reference row (averaging every row of an uncorrected
   frame would widen the curved lines). The peaks to measure are **selected automatically**:
   the calibration lines of the wavelength profile inside the usable columns, else peaks
   found with :func:`scipy.signal.find_peaks` on the baseline-corrected preview.
3. The user edits the selection: a click on the spectrum adds a peak (snapped to the local
   maximum, or selects a peak already there), a peak line can be dragged, peaks selected in
   the list can be removed; *Auto detect* / *Use calibration lines* replace the selection.
   The line **profile** (Gaussian / Lorentzian) and the **baseline** remover (ModPoly,
   IModPoly, ZhangFit, AsLS, arPLS and their parameters) are chosen; the preview shows the
   baseline live.
4. *Calculate FWHM* measures every selected peak in every block of slit rows in a worker
   thread (*Cancel* stops it); table, plots, the fitted components on the preview and Excel
   export follow. Any later change of the selection or of a setting clears the result.
5. *Save FWHM for this setting* stores the median FWHM (nm) over the measured lines for the
   frame's camera setting (``CalibrationStore.save_resolution``); the acquisition engine can
   use it as the wavelength bin width (``resolution_source = "fwhm"``).

Host contract (the widget never touches the application directly):

* Inputs: ``set_calibration_source(store, setting_provider)``, ``set_grab_available``,
  ``begin_grab``, ``accept_snapshot``, ``snapshot_failed``, ``set_frame``.
* Outputs: ``grabRequested()``, ``analysisComplete(object)`` (a ``ResolutionResult``),
  ``resolutionStored(str)`` (path of the FWHM file just written for the camera setting).

``store`` is a :class:`modules.contracts.CalibrationStore`; ``setting_provider()`` returns the
current :class:`modules.contracts.CameraSetting` of the camera (or ``None``).

Data conventions:
    Frames are 2-D ``(slit rows, spectral columns)``, oriented and dark-corrected, before
    smile correction (it is applied here). Peak positions are spectral columns (px) at the
    frame's hardware bin; the spectrum plot's x axis is ``m·px + c`` in nm when a wavelength
    profile is loaded, else px.

Threading:
    GUI thread only; the measurement runs in a ``QThread`` owned by the widget. *Cancel*, a
    new frame or closing the panel stops it (the fit polls a cancel flag), and
    ``closeEvent`` waits for it to stop.

Example:
    Stand-alone: ``python -m modules.spectral_resolution``. Scripted (needs a lamp frame and
    a running ``QApplication``, so it is shown as code, not as a doctest)::

        w = SpectralResolutionWidget()
        w.set_frame(frame, camera_bin=2)            # peaks are pre-selected
        w.add_peak(412)
        w.set_fit_profile("lorentzian")
        w.set_baseline("arpls", lam=1e5)
        w.analyze()                                 # result in w.result / analysisComplete
"""
from __future__ import annotations

import logging
import math
import os
import re
import time
from functools import partial

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QItemSelectionModel, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence, QValidator
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                             QProgressBar, QPushButton, QScrollArea, QShortcut, QSizePolicy, QSpinBox, QSplitter,
                             QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from modules.image import png_name, save_png

from .baseline import BASELINE_LABELS, BASELINE_METHODS, DEFAULT_PARAMS, PARAM_HELP
from .core import (DEFAULT_WINDOW_FWHM, PROFILES, MeasurementCancelled, fit_spectrum, measure_frame,
                   remove_spectrum_baseline, row_blocks, usable_columns)

logger = logging.getLogger(__name__)

__all__ = ["SpectralResolutionWidget"]

PREVIEW_ROWS = 16        # rows around the reference row previewed without a smile model
SNAP_PX = 3              # a click / dragged line snaps to the maximum within this many px
NEAR_PX = 2.0            # a click this close to a selected peak selects it instead of adding one
DUPLICATE_PX = 1.5       # two peaks closer than this are the same peak
KNOWN_PX = 1.5           # a peak this close to a calibration point takes its known wavelength
EDGE_PX = 2              # peaks must lie this far inside the usable columns (as measure_spectrum)

_PROFILE_LABELS = {"gaussian": "Gaussian", "lorentzian": "Lorentzian"}

# Parameter controls per baseline method: (name, label, kind, low, high, log step).
_PARAM_UI = {
    "none": (),
    "modpoly": (("degree", "Polynomial degree", "int", 0, 15, None),
                ("max_iter", "Max refits", "int", 0, 1000, None),
                ("tol", "Tolerance", "log", 1e-8, 1.0, 1.0),
                ("use_original", "Clip original spectrum", "bool", None, None, None)),
    "imodpoly": (("degree", "Polynomial degree", "int", 0, 15, None),
                 ("max_iter", "Max fits", "int", 1, 1000, None),
                 ("tol", "Tolerance", "log", 1e-8, 1.0, 1.0)),
    "zhangfit": (("lam", "λ (smoothness)", "log", 1e-2, 1e12, 0.5),
                 ("diff_order", "porder", "int", 1, 3, None),
                 ("max_iter", "Iterations (max)", "int", 1, 100, None),
                 ("tol", "Tolerance", "log", 1e-8, 1.0, 1.0)),
    "asls": (("lam", "λ (smoothness)", "log", 1e-2, 1e12, 0.5),
             ("p", "p (asymmetry)", "log", 1e-5, 0.5, 0.5),
             ("diff_order", "Difference order", "int", 1, 3, None),
             ("max_iter", "Max iterations", "int", 0, 500, None),
             ("tol", "Tolerance", "log", 1e-8, 1.0, 1.0)),
    "arpls": (("lam", "λ (smoothness)", "log", 1e-2, 1e12, 0.5),
              ("diff_order", "Difference order", "int", 1, 3, None),
              ("max_iter", "Max iterations", "int", 0, 500, None),
              ("tol", "Tolerance", "log", 1e-8, 1.0, 1.0)),
}


def _sci(value) -> str:
    """Compact number with 3 significant digits: ``0.01``, ``250``, ``1e5``, ``3.16e-7``, ``5e-4``.

    Plain notation from 0.001 to 9990, else ``<mantissa>e<exponent>`` (no ``+``, no padding).
    """
    value = float(value)
    if value == 0 or not math.isfinite(value):
        return f"{value:g}"
    mantissa, exponent = f"{value:.2e}".split("e")              # always has an exponent part
    exponent = int(exponent)
    mantissa = mantissa.rstrip("0").rstrip(".")
    if -3 <= exponent <= 3:
        return f"{float(mantissa + 'e' + str(exponent)):.10f}".rstrip("0").rstrip(".")
    return f"{mantissa}e{exponent}"


class _LogSpinBox(QDoubleSpinBox):
    """Spin box over ``log10(value)`` that shows and accepts the value itself (``1e5``, ``0.01``).

    The arrows step by ``step`` decades; any positive number in range can be typed. The value
    used (:meth:`real_value`) is exactly the one shown (3 significant digits, see
    :func:`_sci`), and re-reading the shown text (Enter) keeps the value unchanged.
    """

    _PARTIAL = re.compile(r"[0-9]*\.?[0-9]*([eE][-+]?[0-9]*)?")

    def __init__(self, low, high, value, step=0.5, parent=None):
        """Args: low, high: positive range; value: start value; step: decades per arrow step."""
        super().__init__(parent)
        self.setDecimals(10)                     # precision of the exponent, not of the display
        self.setRange(math.log10(low), math.log10(high))
        self.setSingleStep(step)
        self.set_real(value)

    def real_value(self) -> float:
        """The value (not its logarithm) exactly as displayed."""
        return float(self.textFromValue(self.value()))

    def set_real(self, value):
        """Show ``value`` (> 0; clipped to the range)."""
        self.setValue(math.log10(float(value)))

    def textFromValue(self, value):  # noqa: N802 (Qt override)
        """Display ``10 ** value``."""
        return _sci(10.0 ** value)

    def valueFromText(self, text):  # noqa: N802 (Qt override)
        """``log10`` of the typed number (the current value if it cannot be read or is unchanged)."""
        if text.strip() == self.textFromValue(self.value()):
            return self.value()                  # Enter on the shown text: keep the exact value
        try:
            number = float(text.strip())
        except ValueError:
            return self.value()
        return math.log10(number) if number > 0 else self.value()

    def validate(self, text, pos):
        """Accept positive numbers in range, including scientific notation being typed."""
        stripped = text.strip()
        try:
            number = float(stripped)
        except ValueError:
            state = QValidator.Intermediate if self._PARTIAL.fullmatch(stripped) else QValidator.Invalid
            return state, text, pos
        if number > 0 and self.minimum() - 1e-9 <= math.log10(number) <= self.maximum() + 1e-9:
            return QValidator.Acceptable, text, pos
        return QValidator.Intermediate, text, pos

    def fixup(self, text):
        """Fall back to the current value."""
        return self.textFromValue(self.value())


class _PeakList(QListWidget):
    """Peak list that asks for little height (it still expands into free space)."""

    def sizeHint(self):  # noqa: N802 (Qt override)
        """Room for about five entries."""
        return QSize(220, 110)


class _MeasureWorker(QThread):
    """Run :func:`measure_frame` off the GUI thread, then fit the preview spectrum once.

    :meth:`request_cancel` (any thread) makes the running fit stop at its next model
    evaluation (:class:`~modules.spectral_resolution.core.MeasurementCancelled`); the worker
    then emits ``cancelled()`` instead of ``done``.
    """

    progress = pyqtSignal(int, str)
    done = pyqtSignal(object, object)        # ResolutionResult, SpectrumFit of the preview (or None)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, frame, kwargs, preview=None):
        """Args: frame: corrected 2-D frame; kwargs: of :func:`measure_frame`;
        preview: ``(spectrum, clipped, valid_columns)`` to fit once afterwards, or ``None``."""
        super().__init__()
        self.frame, self.kwargs, self.preview = frame, kwargs, preview
        self._cancel = False

    def request_cancel(self):
        """Ask the measurement to stop as soon as possible (returns immediately)."""
        self._cancel = True

    def cancel_requested(self) -> bool:
        """True once :meth:`request_cancel` was called (polled by the fit)."""
        return self._cancel

    def run(self):
        """Measure and emit ``done(result, preview_fit)``, ``cancelled()`` or ``failed(message)``."""
        try:
            result = measure_frame(self.frame, progress=lambda f, m: self.progress.emit(int(100 * f), m),
                                   cancel=self.cancel_requested, **self.kwargs)
        except MeasurementCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("FWHM measurement failed")
            self.failed.emit(str(exc))
            return
        fit = None
        if self.preview is not None:
            spectrum, clipped, columns = self.preview
            keys = ("pixels", "m", "c", "wavelengths", "snr_min", "window_fwhm", "profile", "baseline",
                    "baseline_params")
            try:
                fit = fit_spectrum(spectrum, clipped=clipped, valid_columns=columns, cancel=self.cancel_requested,
                                   **{k: self.kwargs[k] for k in keys if k in self.kwargs})
            except MeasurementCancelled:
                self.cancelled.emit()
                return
            except Exception:  # noqa: BLE001 - the overlay is optional
                logger.exception("FWHM preview fit failed")
        self.done.emit(result, fit)


class SpectralResolutionWidget(QWidget):
    """FWHM / spectral-resolution panel: select peaks on the spectrum, then calculate.

    Constructible with no arguments. Besides the host contract (module docstring) it offers
    a small scripting API used by the UI itself: :meth:`add_peak`, :meth:`click_at`,
    :meth:`set_peaks`, :meth:`remove_selected_peaks`, :meth:`clear_peaks`,
    :meth:`auto_detect_peaks`, :meth:`use_calibration_lines`, :meth:`set_fit_profile`,
    :meth:`set_baseline`, :meth:`analyze`, :meth:`cancel`, :meth:`export_excel` and
    :meth:`store_resolution`.

    Attributes:
        frame: Raw frame as grabbed/loaded. ``corrected``: after smile correction (float32).
        profile: Active :class:`~modules.contracts.WavelengthCalibration` or ``None``.
        smile: Active smile model or ``None``.
        preview: Raw preview spectrum (float64, one value per column); ``preview_rows`` its
            ``(first, last)`` rows; ``preview_baseline`` / ``preview_corrected`` the current
            baseline and ``preview - baseline``; ``columns`` the usable ``(first, last)`` columns.
        result: Last :class:`~modules.spectral_resolution.core.ResolutionResult` (``None``
            when nothing matches the current selection and settings).
        preview_fit: :class:`~modules.spectral_resolution.core.SpectrumFit` of the preview made
            with the result's settings (drawn over the spectrum).
    """

    grabRequested = pyqtSignal()
    analysisComplete = pyqtSignal(object)
    resolutionStored = pyqtSignal(str)

    def __init__(self, parent=None):
        """Build controls and plots. Args: parent: optional parent widget."""
        super().__init__(parent)
        self.frame = None            # raw frame as grabbed/loaded
        self.corrected = None        # frame after smile correction (float32)
        self.camera_bin = None
        self.setting = None          # CameraSetting of the frame (None if unknown)
        self.profile = None          # WavelengthCalibration
        self.profile_source = ""
        self.smile = None            # SmileCalibration
        self.smile_source = ""
        self.result = None
        self.preview = None
        self.preview_rows = None
        self.preview_baseline = None
        self.preview_corrected = None
        self.preview_fit = None
        self.columns = None
        self._peaks = []             # selected peak centres (px), ascending
        self._selection_source = ""  # "calibration" | "detected" | "manual" | ""
        self._store = None
        self._setting_provider = None
        self._grab_request_id = None
        self._worker = None
        self._started = 0.0
        self._generation = 0         # bumped by every change of selection/settings
        self._run_generation = None
        self._result_info = None
        self._peak_lines = []
        self._image_lines = []
        self._fit_items = []
        self._pending_select = None  # peak to select after a deferred refresh (drag)
        self._result_key = None      # settings the current result / running calculation was made with
        self._rows_saved = None      # user's rows per block while it is forced to 1 (no smile model)
        self._forcing_rows = False   # True while rows per block is changed programmatically
        self._notes = None           # status notes collected while a new frame is set up
        self._frame_notes = []       # ... and the ones of the last frame
        self._quiet = True           # True while controls are built/set programmatically
        try:
            self._build_ui()
        finally:
            self._quiet = False
        self._update_plan()

    # ── UI construction ────────────────────────────────────
    def _build_ui(self):
        """Create every control and plot."""
        root = QHBoxLayout(self)
        controls = QWidget()
        left = QVBoxLayout(controls)
        left.setContentsMargins(4, 4, 4, 4)

        box = QGroupBox("1 · Image")
        form = QFormLayout(box)
        row = QHBoxLayout()
        self.btn_grab = QPushButton("Grab from Scan")
        self.btn_grab.setEnabled(False)
        self.btn_grab.clicked.connect(lambda: self.grabRequested.emit())
        row.addWidget(self.btn_grab)
        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)
        row.addWidget(self.btn_load)
        self.btn_export_png = QPushButton("Export PNG…")
        self.btn_export_png.setEnabled(False)
        self.btn_export_png.setToolTip("Save the current image (as grabbed) as an 8/16-bit PNG, values unscaled")
        self.btn_export_png.clicked.connect(self._on_export_png)
        row.addWidget(self.btn_export_png)
        form.addRow(row)
        self.lbl_setting = QLabel("No image")
        self.lbl_setting.setWordWrap(True)
        form.addRow("Setting", self.lbl_setting)
        left.addWidget(box)

        box = QGroupBox("2 · Calibrations for this setting")
        self.calibration_box = box
        grid = QGridLayout(box)                          # description on its own full-width row: wraps fully
        self.lbl_wavelength = QLabel("—")
        self.lbl_smile = QLabel("—")
        for row, (title, label, tip, slot) in enumerate((
                ("Wavelength", self.lbl_wavelength, "Use another wavelength profile (.json) instead of the stored one",
                 self.load_wavelength_file),
                ("Smile", self.lbl_smile, "Use another smile model (.npz) instead of the stored one",
                 self.load_smile_file))):
            label.setWordWrap(True)
            btn = QPushButton("Load…")
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _=False, f=slot: f())
            grid.addWidget(QLabel(f"<b>{title}</b>"), 2 * row, 0)
            grid.addWidget(btn, 2 * row, 1, Qt.AlignRight)
            grid.addWidget(label, 2 * row + 1, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        left.addWidget(box)

        box = QGroupBox("3 · Fit")
        self.fit_box = box
        form = QFormLayout(box)
        self.line_profile = QComboBox()
        for name in PROFILES:
            self.line_profile.addItem(_PROFILE_LABELS.get(name, name.title()), name)
        self.line_profile.setToolTip("Shape of the instrument's line profile; the wrong one biases the width "
                                     "(a Lorentzian fitted to a Gaussian line comes out narrower)")
        self.line_profile.currentIndexChanged.connect(self._on_profile_changed)
        form.addRow("Profile", self.line_profile)
        self.baseline_method = QComboBox()
        for name in BASELINE_METHODS:
            self.baseline_method.addItem(BASELINE_LABELS[name], name)
        self.baseline_method.setToolTip("Smooth background removed from every spectrum before fitting (estimated "
                                        "on the usable columns only); the fit keeps its own linear background")
        self.baseline_method.currentIndexChanged.connect(self._on_baseline_method_changed)
        form.addRow("Baseline", self.baseline_method)
        self._baseline_inputs = {}
        self._baseline_pages = {}
        params_area = QWidget()
        params_layout = QVBoxLayout(params_area)
        params_layout.setContentsMargins(0, 0, 0, 0)
        for method in BASELINE_METHODS:
            page = self._baseline_page(method)
            params_layout.addWidget(page)
            page.setVisible(method == "none")
        form.addRow(params_area)
        self.window_fwhm = QDoubleSpinBox()
        self.window_fwhm.setRange(1.0, 10.0)
        self.window_fwhm.setSingleStep(0.5)
        self.window_fwhm.setSuffix(" × FWHM")
        self.window_fwhm.setValue(1.0)                   # minimum = "auto"
        self.window_fwhm.setToolTip("Fit half window per line; neighbouring lines inside it are fitted together. "
                                    "auto: 3 × FWHM for a Gaussian, 5 × for a Lorentzian (heavier wings)")
        self.window_fwhm.valueChanged.connect(self._settings_changed)
        form.addRow("Fit window", self.window_fwhm)
        self.rows_per_block = QSpinBox()
        self.rows_per_block.setRange(1, 4096)
        self.rows_per_block.setValue(16)
        self.rows_per_block.setToolTip("Slit rows averaged per spectrum (only after smile correction; 1 without it)")
        self.rows_per_block.valueChanged.connect(self._on_rows_changed)
        form.addRow("Rows per block", self.rows_per_block)
        self.saturation = QDoubleSpinBox()
        self.saturation.setRange(0, 1e9)
        self.saturation.setDecimals(0)
        self.saturation.setSpecialValueText("auto")
        self.saturation.setToolTip("Counts at which the sensor clips (auto: estimated from the image)")
        self.saturation.valueChanged.connect(self._settings_changed)
        form.addRow("Saturation", self.saturation)
        self.snr_min = QDoubleSpinBox()
        self.snr_min.setRange(1, 1000)
        self.snr_min.setValue(10)
        self.snr_min.valueChanged.connect(self._settings_changed)
        form.addRow("Min SNR", self.snr_min)
        left.addWidget(box)
        self._on_profile_changed()

        self.btn_analyze = QPushButton("Calculate FWHM")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn_analyze.clicked.connect(self.analyze)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setToolTip("Stop the running calculation (nothing is kept)")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        row = QHBoxLayout()
        row.addWidget(self.btn_analyze, 1)
        row.addWidget(self.btn_cancel, 0)
        left.addLayout(row)
        self.plan = QLabel("")
        self.plan.setWordWrap(True)
        left.addWidget(self.plan)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        left.addWidget(self.progress)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        left.addWidget(self.summary)
        self.btn_export = QPushButton("📊 Export Excel")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export)
        left.addWidget(self.btn_export)
        self.btn_store = QPushButton("💾 Save FWHM for this setting")
        self.btn_store.setEnabled(False)
        self.btn_store.setToolTip("Store the median FWHM (nm) over the lines for this camera setting; "
                                  "scans use it as the bin width when resolution_source is \"fwhm\"")
        self.btn_store.clicked.connect(self.store_resolution)
        left.addWidget(self.btn_store)
        self.status = QLabel("Grab a lamp frame from Scan or load an image")
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        left.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(420)
        root.addWidget(scroll, 0)

        # spectrum with the peak selection | selected peaks
        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        self.spectrum_plot = pg.PlotWidget(title="Select peaks to fit")
        self.spectrum_plot.setMinimumSize(200, 160)
        for side in ("left", "bottom"):
            self.spectrum_plot.getAxis(side).enableAutoSIPrefix(False)
        self.spectrum_plot.setLabel("left", "Counts")
        self._legend = self.spectrum_plot.addLegend(offset=(-5, 28), brush=pg.mkBrush(0, 0, 0, 170))
        self._curve_raw = pg.PlotDataItem(pen=pg.mkPen((150, 150, 150), width=1))
        self._curve_baseline = pg.PlotDataItem(pen=pg.mkPen((255, 165, 0), width=1.5, style=Qt.DashLine))
        self._curve_corrected = pg.PlotDataItem(pen=pg.mkPen((0, 200, 255), width=1))
        for item in (self._curve_raw, self._curve_baseline, self._curve_corrected):
            self.spectrum_plot.addItem(item)
        self.spectrum_plot.scene().sigMouseClicked.connect(self._on_scene_clicked)
        top_layout.addWidget(self.spectrum_plot, 1)
        side = QScrollArea()                             # scrolls instead of squeezing when the window is short
        side.setWidget(self._peaks_panel())
        side.setWidgetResizable(True)
        side.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        side.setFrameShape(QFrame.NoFrame)
        side.setMinimumWidth(235)
        side.setMaximumWidth(320)
        top_layout.addWidget(side, 0)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Peak", "λ nm", "Pixel", "FWHM px", "FWHM nm", "R", "Valid"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.image_plot = pg.PlotWidget(title="Smile-corrected frame")
        self.image = pg.ImageItem(axisOrder="row-major")
        self.image_plot.addItem(self.image)
        self.fwhm_plot = pg.PlotWidget(title="FWHM per line (median, 16–84 %)")
        self.slit_plot = pg.PlotWidget(title="FWHM along the slit")
        self.slit_plot.addLegend(offset=(5, 5))
        for plot in (self.fwhm_plot, self.slit_plot):
            for side in ("left", "bottom"):
                plot.getAxis(side).enableAutoSIPrefix(False)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.table, "Results")
        self.tabs.addTab(self.fwhm_plot, "FWHM per line")
        self.tabs.addTab(self.slit_plot, "Along the slit")
        self.tabs.addTab(self.image_plot, "Frame")
        self.tabs.setMinimumHeight(140)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

    def _peaks_panel(self):
        """The "Selected peak centers" list with its buttons and the peak-finding controls."""
        panel = QWidget()
        self.peaks_panel = panel
        panel.setMinimumWidth(190)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 0, 0, 0)
        title = QLabel("<b>Selected peak centers</b>")
        layout.addWidget(title)
        self.peak_list = _PeakList()
        self.peak_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.peak_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.peak_list.setMinimumHeight(80)
        self.peak_list.itemSelectionChanged.connect(self._restyle_peak_lines)
        shortcut = QShortcut(QKeySequence(QKeySequence.Delete), self.peak_list)
        shortcut.setContext(Qt.WidgetShortcut)
        shortcut.activated.connect(self.remove_selected_peaks)
        layout.addWidget(self.peak_list, 1)
        row = QHBoxLayout()
        self.btn_remove = QPushButton("Remove selected")
        self.btn_remove.clicked.connect(self.remove_selected_peaks)
        row.addWidget(self.btn_remove)
        self.btn_clear = QPushButton("Clear selected peaks")
        self.btn_clear.setToolTip("Remove every peak from the selection")
        self.btn_clear.clicked.connect(self.clear_peaks)
        row.addWidget(self.btn_clear)
        layout.addLayout(row)
        hint = QLabel("Click: add a peak (snaps to the maximum) or select one. Drag a line to move it.")
        hint.setWordWrap(True)
        hint.setToolTip(f"A click within {NEAR_PX:g} px of a selected peak selects it; otherwise a peak is added at the "
                        f"maximum of the corrected preview within ±{SNAP_PX} px. A dragged line snaps the same way on "
                        "release. Delete removes the peaks selected in the list.")
        layout.addWidget(hint)
        box = QGroupBox("Find peaks")
        form = QFormLayout(box)
        self.min_prominence = QDoubleSpinBox()
        self.min_prominence.setRange(1, 1e5)
        self.min_prominence.setDecimals(1)
        self.min_prominence.setValue(20)
        self.min_prominence.setSuffix(" × noise")
        self.min_prominence.setToolTip("Minimum peak prominence of Auto detect, in units of the preview's noise "
                                       "(median absolute first difference / 0.6745)")
        form.addRow("Min prominence", self.min_prominence)
        self.min_distance = QSpinBox()
        self.min_distance.setRange(1, 500)
        self.min_distance.setValue(3)
        self.min_distance.setSuffix(" px")
        self.min_distance.setToolTip("Minimum distance between detected peaks")
        form.addRow("Min distance", self.min_distance)
        self.btn_detect = QPushButton("Auto detect")
        self.btn_detect.setToolTip("Replace the selection with the peaks found in the baseline-corrected preview")
        self.btn_detect.clicked.connect(self.auto_detect_peaks)
        form.addRow(self.btn_detect)
        self.btn_calibration = QPushButton("Use calibration lines")
        self.btn_calibration.setToolTip("Replace the selection with the wavelength profile's calibration lines")
        self.btn_calibration.setEnabled(False)
        self.btn_calibration.clicked.connect(self.use_calibration_lines)
        form.addRow(self.btn_calibration)
        layout.addWidget(box)
        self.lbl_preview = QLabel("")
        self.lbl_preview.setWordWrap(True)
        layout.addWidget(self.lbl_preview)
        return panel

    def _baseline_page(self, method):
        """Parameter controls of one baseline method (only the selected method's page is visible)."""
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(12, 0, 0, 0)
        inputs = {}
        defaults, helps = DEFAULT_PARAMS.get(method, {}), PARAM_HELP.get(method, {})
        for name, label, kind, low, high, step in _PARAM_UI[method]:
            default = defaults[name]
            if kind == "int":
                control = QSpinBox()
                control.setRange(int(low), int(high))
                control.setValue(int(default))
                control.valueChanged.connect(self._on_baseline_params_changed)
            elif kind == "log":
                control = _LogSpinBox(low, high, default, step)
                control.valueChanged.connect(self._on_baseline_params_changed)
            else:
                control = QCheckBox()
                control.setChecked(bool(default))
                control.toggled.connect(self._on_baseline_params_changed)
            tip = helps.get(name, "")
            if kind == "log":
                tip += " Arrows step by ½ decade; type a number such as 1e5." if step == 0.5 else \
                    " Arrows step by one decade; type a number such as 1e-3."
            control.setToolTip(tip.strip())
            form.addRow(label, control)
            inputs[name] = control
        if not inputs:
            note = QLabel("No baseline removal: only the fit's own linear background.")
            note.setWordWrap(True)
            form.addRow(note)
        self._baseline_inputs[method] = inputs
        self._baseline_pages[method] = page
        return page

    # ── host contract ──────────────────────────────────────
    def set_calibration_source(self, store, setting_provider):
        """Where stored calibrations live and how to get the camera's current setting.

        Args:
            store: :class:`modules.contracts.CalibrationStore` (or ``None``).
            setting_provider: Callable returning the current :class:`~modules.contracts.CameraSetting` or ``None``.
        """
        self._store, self._setting_provider = store, setting_provider
        self._update_store_button()

    def set_grab_available(self, available):
        """Enable *Grab from Scan* when a host can deliver frames."""
        self.btn_grab.setEnabled(bool(available))

    def begin_grab(self, request_id):
        """Mark a snapshot request as pending (only that id is accepted)."""
        self._grab_request_id = request_id
        self.btn_grab.setEnabled(False)
        self.status.setText("Waiting for camera frame…")

    def accept_snapshot(self, packet):
        """Receive the requested frame (``request_id``, ``image``, ``camera_bin``)."""
        if packet.request_id != self._grab_request_id:
            return
        self._grab_request_id = None
        self.btn_grab.setEnabled(True)
        self.set_frame(packet.image, packet.camera_bin, setting=self._current_setting(packet.image.shape, packet.camera_bin))
        self._frame_status("Grabbed frame (dark-corrected, oriented)")

    def snapshot_failed(self, request_id, message):
        """Report a failed snapshot request and re-enable Grab."""
        if request_id is not None and request_id != self._grab_request_id:
            return
        self._grab_request_id = None
        self.btn_grab.setEnabled(True)
        self.status.setText(message)

    def _current_setting(self, shape, camera_bin=None):
        """Camera setting from the provider, adjusted to ``shape``/``camera_bin``; ``None`` without provider."""
        try:
            setting = self._setting_provider() if self._setting_provider else None
        except Exception:  # noqa: BLE001
            setting = None
        return setting.with_frame(shape, camera_bin) if setting is not None else None

    # ── frame and calibrations ─────────────────────────────
    def set_frame(self, frame, camera_bin=1, setting=None):
        """Show a lamp frame, auto-load the calibrations stored for ``setting`` and pre-select peaks.

        Args:
            frame: 2-D ``(slit rows, spectral columns)``.
            camera_bin: Hardware bin of ``frame``.
            setting: Its :class:`~modules.contracts.CameraSetting` (``None`` = unknown: load files manually).

        Raises:
            ValueError: ``frame`` is not 2-D.
        """
        a = np.asarray(frame)
        if a.ndim != 2:
            raise ValueError("Image must be 2-D")
        self._invalidate(None)                            # also stops a calculation still running
        self.frame, self.camera_bin, self.setting = a, int(camera_bin), setting
        self.btn_export_png.setEnabled(a.dtype in (np.uint8, np.uint16))
        self.lbl_setting.setText(setting.label() if setting else f"bin {camera_bin} · {a.shape[1]}×{a.shape[0]} px (camera unknown)")
        self.profile = self.smile = None
        self.profile_source = self.smile_source = ""
        self._selection_source = ""                       # a new frame is always auto-selected
        self._notes = []
        try:
            self._auto_load()
            self.btn_calibration.setEnabled(self.profile is not None)
            self._update_corrected()
        finally:
            self._frame_notes, self._notes = self._notes, None
        self._frame_status("")

    def _say(self, text):
        """Show ``text`` in the status line (collected while :meth:`set_frame` sets up a frame)."""
        if self._notes is not None:
            self._notes.append(text)
        else:
            self.status.setText(text)

    def _frame_status(self, origin):
        """Status after a new frame: ``origin``, whether a smile model was applied, and the frame's notes."""
        parts = [origin] if origin else []
        smile = ("smile model applied" if self.smile is not None else
                 "no smile model for this setting" if self.setting is not None else "no smile model (use Load…)")
        parts.append(f"{smile}; {len(self._peaks)} peak{'s' if len(self._peaks) != 1 else ''} selected")
        text = ". ".join(["; ".join(parts)] + list(self._frame_notes))
        self.status.setText(text[:1].upper() + text[1:])

    def _auto_load(self):
        """Load the wavelength profile and smile model stored for ``self.setting`` (if any)."""
        if self._store is None or self.setting is None:
            self.lbl_wavelength.setText("✗ no camera setting: use Load…")
            self.lbl_smile.setText("✗ no camera setting: use Load…")
            return
        try:
            stored = self._store.load_wavelength(self.setting)
            if stored is None:
                self.lbl_wavelength.setText(f"✗ none stored for bin {self.setting.camera_bin}: Apply one in Wavelength Calibration")
            elif not stored.ok:
                self.lbl_wavelength.setText("✗ stored profile does not match: " + "; ".join(stored.problems))
            else:
                self.profile, self.profile_source = stored.profile, str(stored.path)
        except Exception as exc:  # noqa: BLE001
            self.lbl_wavelength.setText(f"✗ cannot read stored profile: {exc}")
        path = self._store.smile_path(self.setting)
        if path.exists():
            try:
                self._set_smile(path)
            except Exception as exc:  # noqa: BLE001
                self.lbl_smile.setText(f"✗ {os.path.basename(path)}: {exc}")
        else:
            self.lbl_smile.setText(f"✗ none stored for bin {self.setting.camera_bin}: Apply one in Smile Analysis")
        self._show_calibrations()

    def _set_smile(self, path):
        """Load a smile model and check it matches the frame size."""
        from modules.smile import load_calibration
        cal = load_calibration(str(path))
        if tuple(cal.image_shape) != tuple(self.frame.shape):
            raise ValueError(f"model is for {cal.image_shape[1]}×{cal.image_shape[0]}, frame is "
                             f"{self.frame.shape[1]}×{self.frame.shape[0]}")
        cal.set_backend("cpu")
        self.smile, self.smile_source = cal, str(path)

    def _show_calibrations(self):
        """Describe the active wavelength profile and smile model."""
        p = self.profile
        if p is not None:
            self.lbl_wavelength.setText(f"✓ {os.path.basename(self.profile_source)} · {len(p.pixels)} lines · "
                                        f"m={p.m:.5g} nm/px · RMS {p.rms_nm:.3g} nm · row {p.selected_row}")
        if self.smile is not None:
            text = f"✓ {os.path.basename(self.smile_source)} · y_ref {self.smile.y_ref} · RMS after {self.smile.rms_after:.3g} px"
            if p is not None and self.smile.y_ref is not None and int(p.selected_row) != int(self.smile.y_ref):
                text += f"\n⚠ wavelength row {p.selected_row} ≠ smile reference row {self.smile.y_ref}"
            self.lbl_smile.setText(text)
        self.btn_calibration.setEnabled(p is not None)

    def load_wavelength_file(self, path=None):
        """Use a wavelength profile file for this frame (must match its bin and size).

        Refused without a frame (as :meth:`load_smile_file`): every new frame loads the
        calibrations stored for its own camera setting, so a profile loaded earlier would be
        dropped by the next frame anyway.
        """
        if self.frame is None:
            self.status.setText("Load or grab a frame first, then load its wavelength profile")
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Wavelength profile", "", "JSON (*.json)")
            if not path:
                return
        from modules.contracts import WavelengthCalibration
        try:
            p = WavelengthCalibration.load(path)
            if int(p.camera_bin) != int(self.camera_bin):
                raise ValueError(f"profile bin {p.camera_bin} ≠ frame bin {self.camera_bin}")
            if p.frame_shape and tuple(p.frame_shape) != tuple(self.frame.shape):
                raise ValueError("profile image size differs from the frame")
            self.profile, self.profile_source = p, str(path)
            self._show_calibrations()
            self._invalidate("Wavelength profile changed: press Calculate FWHM to measure again")
            self._update_preview()                       # the reference row may have changed
            self.spectrum_plot.enableAutoRange()         # x axis now in nm
            if self._selection_source != "manual":
                self._auto_select()
            else:
                self._refresh_peaks(keep_selection=True)
            self._update_ready()
        except Exception as exc:  # noqa: BLE001
            self.error(exc)

    def load_smile_file(self, path=None):
        """Use a smile model file for this frame (must match its size)."""
        if self.frame is None:
            self.status.setText("Load or grab a frame first, then load its smile model")
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Smile model", "", "Smile model (*.npz)")
            if not path:
                return
        try:
            self._set_smile(path)
            self._show_calibrations()
            self._invalidate("Smile model changed: press Calculate FWHM to measure again")
            self._update_corrected()
        except Exception as exc:  # noqa: BLE001
            self.error(exc)

    def _update_corrected(self):
        """Apply the smile model (if any), refresh the image, the preview and the auto-selection."""
        if self.frame is None:
            return
        f = np.ascontiguousarray(self.frame, dtype=np.float32)
        self.corrected = self.smile.apply_native(f) if self.smile is not None else f
        self.image.setImage(self.corrected, autoLevels=True)
        if self.smile is None and self.rows_per_block.value() != 1 and self._rows_saved is None:
            self._rows_saved = self.rows_per_block.value()
            self._set_rows(1)
            self._say(f"No smile model: rows are measured one by one (averaging curved lines would widen them; "
                      f"{self._rows_saved} rows per block return with a smile model)")
        elif self.smile is not None and self._rows_saved is not None:
            rows, self._rows_saved = self._rows_saved, None
            if self.rows_per_block.value() == 1:
                self._set_rows(rows)
                self._say(f"Smile model applied: {rows} rows per block again")
        self._update_preview()
        self.spectrum_plot.enableAutoRange()
        if self._selection_source != "manual" or not self._peaks:
            self._auto_select()
        else:
            self._set_peaks(self._peaks, "manual")      # drops peaks the new border made unusable
        self._update_ready()

    def _update_ready(self):
        """Enable *Calculate* when a frame is loaded, peaks are selected and nothing runs."""
        self.btn_analyze.setEnabled(self.frame is not None and self._worker is None and bool(self._peaks))
        self._update_plan()

    def load_image(self):
        """Ask for a ``.png``/``.tif``/``.npy`` lamp image (assumed taken at the camera's current setting)."""
        path, _ = QFileDialog.getOpenFileName(self, "Load lamp image", "", "Images (*.png *.tif *.tiff *.npy)")
        if not path:
            return
        try:
            import cv2
            a = np.load(path, allow_pickle=False) if path.endswith(".npy") else cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if a is None:
                raise ValueError("cannot read image")
            setting = self._current_setting(a.shape)
            self.set_frame(a, setting.camera_bin if setting else 1, setting=setting)
            self._frame_status(f"Loaded {os.path.basename(path)}"
                               + ("; assumed taken at the camera's current setting" if setting else ""))
        except Exception as exc:  # noqa: BLE001
            self.error(exc)

    # ── preview spectrum ───────────────────────────────────
    def _reference_row(self):
        """Reference row: the wavelength profile's row, else the shared default (centre) row."""
        from modules.contracts import default_reference_row
        rows = self.corrected.shape[0]
        if self.profile is not None and 0 <= int(self.profile.selected_row) < rows:
            return int(self.profile.selected_row)
        return max(0, default_reference_row(rows))

    def _update_preview(self):
        """Average the preview rows of the corrected frame and recompute its baseline."""
        if self.corrected is None:
            return
        rows = self.corrected.shape[0]
        if self.smile is not None:
            r0, r1 = 0, rows - 1
            text = f"Preview: mean of all {rows} rows (smile corrected)."
            tip = "The smile model straightens the lines, so every row can be averaged."
        else:
            ref = self._reference_row()
            n = min(PREVIEW_ROWS, rows)
            r0 = int(np.clip(ref - n // 2, 0, rows - n))
            r1 = r0 + n - 1
            text = f"Preview: mean of rows {r0}–{r1} around the reference row {ref} (no smile model)."
            tip = ("Without a smile model the lines are curved: averaging every row would widen them, so only "
                   f"the {n} rows around the reference row are averaged for the preview.")
        self.preview_rows = (r0, r1)
        self.preview = self.corrected[r0:r1 + 1].astype(float).mean(axis=0)
        first, last = usable_columns(self.corrected)                 # data in every row
        model = self._model_columns()
        if model is not None and max(first, model[0]) < min(last, model[1]):
            first, last = max(first, model[0]), min(last, model[1])
        self.columns = (int(first), int(last))
        self.lbl_preview.setText(text)
        self.lbl_preview.setToolTip(tip)
        self._update_preview_baseline()

    def _update_preview_baseline(self):
        """Baseline of the preview with the current method/parameters; redraw the spectrum."""
        if self.preview is None:
            return
        method, params = self.baseline_settings()
        try:
            self.preview_corrected, self.preview_baseline = remove_spectrum_baseline(
                self.preview, method, params, self.columns)
        except ValueError as exc:
            self.preview_corrected, self.preview_baseline = self.preview.copy(), np.zeros(self.preview.size)
            self._say(f"Baseline {BASELINE_LABELS[method]} failed on the preview: {exc}")
        self._draw_spectrum()

    def _model_columns(self):
        """``(first, last)`` columns the smile model maps from real pixels in every row, or ``None``.

        From the model's ``margin`` (which also leaves out the interpolation kernel's reach
        into the zero border); ``None`` without a smile model or margin.
        """
        if self.smile is None or self.corrected is None:
            return None
        try:
            left, right = (int(v) for v in self.smile.margin)
        except (AttributeError, TypeError, ValueError):
            return None
        last = self.corrected.shape[1] - 1 - right
        return (left, last) if 0 <= left < last else None

    def px_to_view(self, px):
        """Spectrum-plot x coordinate (nm with a profile: ``m·px + c``; else px) of pixel(s) ``px``."""
        px = np.asarray(px, dtype=float)
        return self.profile.m * px + self.profile.c if self.profile is not None else px

    def view_to_px(self, x):
        """Pixel of a spectrum-plot x coordinate (``(λ - c) / m`` with a profile)."""
        x = np.asarray(x, dtype=float)
        return (x - self.profile.c) / self.profile.m if self.profile is not None else x

    def _draw_spectrum(self):
        """Raw preview, baseline (dashed) and corrected spectrum."""
        self._legend.clear()
        if self.preview is None:
            for item in (self._curve_raw, self._curve_baseline, self._curve_corrected):
                item.setData([], [])
            return
        x = self.px_to_view(np.arange(self.preview.size, dtype=float))
        method = self.baseline_settings()[0]
        first, last = self.columns
        if method == "none":
            self._curve_raw.setData([], [])
            self._curve_baseline.setData([], [])
            self._curve_corrected.setData(x, self.preview)
            self._legend.addItem(self._curve_corrected, "preview spectrum")
        else:
            self._curve_raw.setData(x, self.preview)
            self._curve_baseline.setData(x[first:last + 1], self.preview_baseline[first:last + 1])
            self._curve_corrected.setData(x, self.preview_corrected)
            self._legend.addItem(self._curve_raw, "raw preview")
            self._legend.addItem(self._curve_baseline, f"baseline ({BASELINE_LABELS[method]})")
            self._legend.addItem(self._curve_corrected, "baseline-corrected")
        self.spectrum_plot.setLabel("bottom", "Wavelength (nm)" if self.profile is not None else "Pixel")
        self._draw_fit()

    # ── peak selection ─────────────────────────────────────
    @property
    def peaks(self):
        """Selected peak centres (px, ascending = P1, P2, ...); a copy."""
        return list(self._peaks)

    def peak_wavelength(self, px):
        """``(wavelength_nm, source)`` of a peak at ``px``.

        The profile's known wavelength when ``px`` is within ``KNOWN_PX`` of one of its
        calibration points (source ``"calibration line"``), else ``m·px + c`` (``"m·px + c"``);
        ``(nan, "pixel only")`` without a profile.
        """
        p = self.profile
        if p is None:
            return float("nan"), "pixel only"
        if len(p.pixels):
            distance = np.abs(np.asarray(p.pixels, dtype=float) - float(px))
            j = int(np.argmin(distance))
            if distance[j] <= KNOWN_PX:
                return float(p.wavelengths[j]), "calibration line"
        return float(p.m * float(px) + p.c), "m·px + c"

    def peak_label(self, index):
        """List text of peak ``index``: ``"P1 • 546.07 nm (px 412.3)"`` (``"P1 • px 412.3"`` without a profile)."""
        px = self._peaks[index]
        nm, _ = self.peak_wavelength(px)
        if np.isfinite(nm):
            return f"P{index + 1} • {nm:.2f} nm (px {px:.1f})"
        return f"P{index + 1} • px {px:.1f}"

    def _usable(self, px):
        """True when a peak at ``px`` can be measured (inside the usable columns, ``EDGE_PX`` from the edge)."""
        if self.columns is None or not np.isfinite(px):
            return False
        return self.columns[0] + EDGE_PX <= px <= self.columns[1] - EDGE_PX

    def _refine(self, i):
        """Sub-pixel position of the maximum at column ``i`` of the corrected preview.

        Vertex of the parabola through the logarithms of ``i - 1, i, i + 1`` above the local
        floor (exact for a Gaussian line, ≈0.03 px low for a Lorentzian), else of the plain
        parabola; ``i`` itself at the border or on a flat top.
        """
        y = self.preview_corrected
        first, last = self.columns
        if not first < i < last:
            return float(i)
        floor = float(np.min(y[max(first, i - 2 * SNAP_PX): min(last, i + 2 * SNAP_PX) + 1]))
        a, b, c = (float(v) - floor for v in y[i - 1:i + 2])
        if min(a, b, c) > 0:
            a, b, c = math.log(a), math.log(b), math.log(c)
        curvature = a - 2.0 * b + c
        if curvature < 0:
            return float(i + np.clip(0.5 * (a - c) / curvature, -0.5, 0.5))
        return float(i)

    def snap(self, px):
        """Position of the maximum of the corrected preview within ``SNAP_PX`` of ``px`` (sub-pixel)."""
        if self.preview_corrected is None:
            return float(px)
        first, last = self.columns
        i0 = int(round(float(px)))
        lo, hi = max(first, i0 - SNAP_PX), min(last, i0 + SNAP_PX)
        if hi < lo:
            return float(px)
        i = lo + int(np.argmax(self.preview_corrected[lo:hi + 1]))
        return self._refine(i)

    def _nearest(self, px, tolerance, exclude=None):
        """Index of the selected peak closest to ``px`` if within ``tolerance`` px, else ``None``."""
        best, best_d = None, float("inf")
        for i, p in enumerate(self._peaks):
            d = abs(p - px)
            if i != exclude and d <= tolerance and d < best_d:
                best, best_d = i, d
        return best

    def calibration_peaks(self):
        """The wavelength profile's calibration points inside the usable columns (px, ascending)."""
        if self.profile is None or self.columns is None:
            return []
        return sorted(float(p) for p in self.profile.pixels if self._usable(float(p)))

    def detect_peaks(self):
        """Peaks of the baseline-corrected preview (:func:`scipy.signal.find_peaks`), px ascending.

        Prominence >= *Min prominence* × noise (median absolute first difference / 0.6745),
        distance >= *Min distance*; only inside the usable columns; sub-pixel vertices.
        """
        if self.preview_corrected is None:
            return []
        from scipy.signal import find_peaks
        first, last = self.columns
        segment = self.preview_corrected[first:last + 1]
        if segment.size < 3:
            return []
        noise = float(np.median(np.abs(np.diff(segment)))) / 0.6745 + 1e-9
        found, _ = find_peaks(segment, prominence=self.min_prominence.value() * noise,
                              distance=max(1, int(self.min_distance.value())))
        return [self._refine(first + int(i)) for i in found if self._usable(first + int(i))]

    def _auto_select(self):
        """Pre-select the calibration lines (with a profile) or the detected peaks."""
        if self.profile is not None:
            peaks, source = self.calibration_peaks(), "calibration"
        else:
            peaks, source = self.detect_peaks(), "detected"
        self._set_peaks(peaks, source)
        if not peaks:
            self._say("No peaks found: click the spectrum to add peaks, or lower Min prominence and press Auto detect")

    def set_peaks(self, pixels):
        """Replace the selection with ``pixels`` (px; not snapped; unusable or duplicate positions dropped)."""
        self._set_peaks([float(p) for p in pixels], "manual")

    def _set_peaks(self, pixels, source):
        """Store a new selection (sorted, de-duplicated, usable only) and refresh everything."""
        kept = []
        for p in sorted(float(p) for p in pixels if self._usable(float(p))):
            if not kept or p - kept[-1] > DUPLICATE_PX:
                kept.append(p)
        self._peaks = kept
        self._selection_source = source
        self._selection_changed()

    def add_peak(self, px, snap=True):
        """Add a peak at ``px`` (snapped to the local maximum unless ``snap=False``).

        A peak (after snapping) within ``DUPLICATE_PX`` of a selected one is not added twice:
        that one is selected in the list instead.

        Returns:
            Index of the added (or existing) peak, or ``None`` outside the usable columns or
            without a frame or while a calculation runs.
        """
        if self.preview is None or self._worker is not None:
            return None
        px = float(px)
        if not self._usable(px):
            self.status.setText(f"px {px:.1f} is outside the usable columns {self.columns[0]}–{self.columns[1]}"
                                if self.columns else "Load or grab a frame first")
            return None
        position = self.snap(px) if snap else px
        if not self._usable(position):
            position = px
        existing = self._nearest(position, DUPLICATE_PX)
        if existing is not None:
            self.select_peaks([existing])
            self.status.setText(f"P{existing + 1} is already selected")
            return existing
        self._peaks = sorted(self._peaks + [position])
        self._selection_source = "manual"
        index = self._peaks.index(position)
        cleared = self._selection_changed()
        self.select_peaks([index])
        self.status.setText(f"Added {self.peak_label(index)}"
                            + (" (renumbered by wavelength)" if index < len(self._peaks) - 1 else "")
                            + ("; result cleared: press Calculate FWHM to measure again" if cleared else ""))
        return index

    def click_at(self, x_view):
        """What a left click on the spectrum at plot x ``x_view`` does (nm with a profile, else px).

        Within ``NEAR_PX`` of a selected peak: selects it. Otherwise adds a snapped peak.

        Returns:
            Index of the selected/added peak, or ``None``.
        """
        if self.preview is None or self._worker is not None:
            return None
        px = float(self.view_to_px(x_view))
        near = self._nearest(px, NEAR_PX)
        if near is not None:
            self.select_peaks([near])
            return near
        return self.add_peak(px)

    def selected_indices(self):
        """Indices of the peaks selected in the list (ascending)."""
        return sorted(self.peak_list.row(item) for item in self.peak_list.selectedItems())

    def select_peaks(self, indices):
        """Select peaks ``indices`` in the list (and highlight their lines)."""
        self.peak_list.clearSelection()
        for n, i in enumerate(indices):
            if 0 <= i < self.peak_list.count():
                command = QItemSelectionModel.ClearAndSelect if n == 0 else QItemSelectionModel.Select
                self.peak_list.setCurrentRow(i, command)
                self.peak_list.scrollToItem(self.peak_list.item(i))

    def remove_selected_peaks(self):
        """Remove the peaks selected in the list."""
        if self._worker is not None:
            return
        drop = set(self.selected_indices())
        if not drop:
            self.status.setText("Select peaks in the list to remove them")
            return
        self._peaks = [p for i, p in enumerate(self._peaks) if i not in drop]
        self._selection_source = "manual"
        self._selection_changed()

    def clear_peaks(self):
        """Remove every peak from the selection."""
        if self._worker is not None:
            return
        self._peaks = []
        self._selection_source = "manual"
        self._selection_changed()

    def auto_detect_peaks(self):
        """Replace the selection with :meth:`detect_peaks`; returns the number of peaks found."""
        if self.preview is None or self._worker is not None:
            return 0
        peaks = self.detect_peaks()
        self._set_peaks(peaks, "detected")
        self.status.setText(f"Auto detect: {len(self._peaks)} peaks (prominence ≥ {self.min_prominence.value():g} × "
                            f"noise, distance ≥ {self.min_distance.value()} px)")
        return len(self._peaks)

    def use_calibration_lines(self):
        """Replace the selection with :meth:`calibration_peaks`; returns the number of peaks."""
        if self.preview is None or self._worker is not None or self.profile is None:
            return 0
        self._set_peaks(self.calibration_peaks(), "calibration")
        self.status.setText(f"{len(self._peaks)} calibration lines inside the usable columns")
        return len(self._peaks)

    def _selection_changed(self):
        """The selection changed: refresh the list and lines and drop a result that no longer matches.

        Returns:
            True when a result was cleared.
        """
        self._refresh_peaks()
        return self._settings_changed()

    def _refresh_peaks(self, keep_selection=False):
        """Rebuild the list and the peak lines from ``self._peaks``.

        Args:
            keep_selection: Keep the list selection (by index; only when the peaks did not change).
        """
        keep = set(self.selected_indices()) if keep_selection else set()
        self.peak_list.blockSignals(True)
        self.peak_list.clear()
        for i in range(len(self._peaks)):
            item = QListWidgetItem(self.peak_label(i))
            nm, source = self.peak_wavelength(self._peaks[i])
            item.setToolTip(f"px {self._peaks[i]:.3f} · " + (f"{nm:.4f} nm ({source})" if np.isfinite(nm) else source))
            self.peak_list.addItem(item)
            item.setSelected(i in keep)
        self.peak_list.blockSignals(False)
        if self.columns is not None:
            bounds = sorted(float(v) for v in self.px_to_view([self.columns[0] + EDGE_PX, self.columns[1] - EDGE_PX]))
        else:
            bounds = [None, None]
        # Reuse the line objects (line i is always peak Pi+1): an edit costs O(changed lines)
        # instead of rebuilding every labelled line.
        n = len(self._peaks)
        while len(self._peak_lines) > n:
            self.spectrum_plot.removeItem(self._peak_lines.pop())
            self.image_plot.removeItem(self._image_lines.pop())
        while len(self._peak_lines) < n:
            i = len(self._peak_lines)
            line = pg.InfiniteLine(pos=0.0, angle=90, label=f"P{i + 1}",
                                   labelOpts={"position": 0.97, "color": (255, 220, 120), "fill": (0, 0, 0, 140),
                                              "movable": False})
            line.setZValue(10)
            line.sigPositionChangeFinished.connect(partial(self._on_line_moved, i))
            self.spectrum_plot.addItem(line)
            self._peak_lines.append(line)
            mark = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen((0, 255, 255), width=1, style=Qt.DashLine))
            self.image_plot.addItem(mark)
            self._image_lines.append(mark)
        for px, line, mark in zip(self._peaks, self._peak_lines, self._image_lines):
            line.setBounds(bounds)
            line.setValue(float(self.px_to_view(px)))
            line.setMovable(self._worker is None)
            mark.setValue(float(px))
        self._restyle_peak_lines()
        self._update_ready()

    def _restyle_peak_lines(self):
        """Dashed peak lines; the ones selected in the list are thicker and yellow."""
        selected = set(self.selected_indices())
        for i, line in enumerate(self._peak_lines):
            if i in selected:
                pen = pg.mkPen((255, 230, 0), width=2.5, style=Qt.DashLine)
            else:
                pen = pg.mkPen((255, 120, 200), width=1.2, style=Qt.DashLine)
            line.setPen(pen)
            line.setHoverPen(pg.mkPen((255, 255, 255), width=2.5, style=Qt.DashLine))

    def _on_line_moved(self, index, line):
        """A peak line was dragged: snap it to the maximum near its new position (or put it back)."""
        self._pending_select = None
        if self._worker is not None or index >= len(self._peaks):
            QTimer.singleShot(0, self._deferred_refresh)
            return
        px = float(self.view_to_px(line.value()))
        position = self.snap(px) if self._usable(px) else None
        if position is None or not self._usable(position):
            self.status.setText("Peak moved outside the usable columns: put back")
        elif self._nearest(position, DUPLICATE_PX, exclude=index) is not None:
            self.status.setText("Another selected peak is already there: put back")
        elif abs(position - self._peaks[index]) > 1e-9:
            peaks = list(self._peaks)
            peaks[index] = position
            self._peaks = sorted(peaks)
            self._selection_source = "manual"
            self._pending_select = self._peaks.index(position)
            self._settings_changed()
        QTimer.singleShot(0, self._deferred_refresh)       # rebuild after the drag event has finished

    def _deferred_refresh(self):
        """Rebuild the peak lines after a drag; select the moved peak (else keep the list selection).

        A bound method, so PyQt drops the queued call if the widget is destroyed first.
        """
        select, self._pending_select = self._pending_select, None
        self._refresh_peaks(keep_selection=select is None)
        if select is not None:
            self.select_peaks([select])

    def _on_scene_clicked(self, event):
        """Left click inside the spectrum's view box: :meth:`click_at` its x coordinate.

        Ignored when an item already handled the click (e.g. a legend sample toggling its
        curve) or when it lands on the legend: pyqtgraph emits ``sigMouseClicked`` for those too.
        """
        if event.button() != Qt.LeftButton or event.double():
            return
        accepted = getattr(event, "isAccepted", None)
        if accepted is not None and accepted():
            return
        pos = event.scenePos()
        if self._legend.isVisible() and self._legend.sceneBoundingRect().contains(pos):
            return
        box = self.spectrum_plot.getPlotItem().getViewBox()
        if not box.sceneBoundingRect().contains(pos):
            return
        self.click_at(box.mapSceneToView(pos).x())

    # ── fit settings ───────────────────────────────────────
    def fit_profile(self):
        """Selected line profile, one of :data:`~modules.spectral_resolution.core.PROFILES`."""
        return self.line_profile.currentData()

    def set_fit_profile(self, name):
        """Select the line profile ``name`` (case-insensitive).

        Raises:
            ValueError: ``name`` is not in :data:`~modules.spectral_resolution.core.PROFILES`.
        """
        index = self.line_profile.findData(str(name).lower())
        if index < 0:
            raise ValueError(f"profile must be one of {', '.join(PROFILES)}, not {name!r}")
        self.line_profile.setCurrentIndex(index)

    def window_value(self):
        """Fit half window (× FWHM), or ``None`` for the profile's default (spin box at "auto")."""
        value = self.window_fwhm.value()
        return None if value <= self.window_fwhm.minimum() else float(value)

    def baseline_settings(self):
        """``(method, params)`` of the baseline controls (every parameter of the method)."""
        method = self.baseline_method.currentData() or "none"
        params = {}
        for name, control in self._baseline_inputs.get(method, {}).items():
            if isinstance(control, _LogSpinBox):
                params[name] = control.real_value()
            elif isinstance(control, QSpinBox):
                params[name] = int(control.value())
            else:
                params[name] = bool(control.isChecked())
        return method, params

    def set_baseline(self, method, **params):
        """Select a baseline method and set some of its parameters (others keep their controls' values).

        Raises:
            ValueError: unknown method or parameter.
        """
        key = str(method).strip().lower()
        if key not in BASELINE_METHODS:
            raise ValueError(f"unknown baseline method {method!r}; valid methods: {', '.join(BASELINE_METHODS)}")
        inputs = self._baseline_inputs[key]
        unknown = sorted(set(params) - set(inputs))
        if unknown:
            raise ValueError(f"unknown parameter(s) {', '.join(unknown)} for {key}; valid: {', '.join(inputs) or 'none'}")
        self._quiet = True
        try:
            for name, value in params.items():
                control = inputs[name]
                if isinstance(control, _LogSpinBox):
                    control.set_real(value)
                elif isinstance(control, QSpinBox):
                    control.setValue(int(value))
                else:
                    control.setChecked(bool(value))
            self.baseline_method.setCurrentIndex(self.baseline_method.findData(key))
        finally:
            self._quiet = False
        self._on_baseline_method_changed()

    def _on_profile_changed(self, *_):
        """New line profile: show its default window and drop the old result."""
        name = self.fit_profile() or "gaussian"
        self.window_fwhm.setSpecialValueText(f"auto ({DEFAULT_WINDOW_FWHM.get(name, 3.0):g} × FWHM)")
        self._settings_changed()

    def _on_baseline_method_changed(self, *_):
        """Show only the selected method's parameters; update the preview baseline."""
        if self._quiet:
            return
        method = self.baseline_settings()[0]
        for name, page in self._baseline_pages.items():
            page.setVisible(name == method)
        self._update_preview_baseline()
        self._settings_changed()

    def _on_baseline_params_changed(self, *_):
        """A baseline parameter changed: update the preview baseline and drop the old result."""
        if self._quiet:
            return
        self._update_preview_baseline()
        self._settings_changed()

    def _set_rows(self, rows):
        """Set *Rows per block* programmatically (not remembered as the user's choice)."""
        self._forcing_rows = True
        try:
            self.rows_per_block.setValue(int(rows))
        finally:
            self._forcing_rows = False

    def _on_rows_changed(self, *_):
        """*Rows per block* changed: by the user, it is no longer restored with a smile model."""
        if not self._forcing_rows:
            self._rows_saved = None
        self._settings_changed()

    def _settings_key(self):
        """Everything the result depends on that the controls and the selection set."""
        method, params = self.baseline_settings()
        return (tuple(self._peaks), self.fit_profile(), self.window_value(), int(self.rows_per_block.value()),
                float(self.saturation.value()), float(self.snr_min.value()), method,
                tuple(sorted(params.items())))

    def _settings_changed(self, *_):
        """Selection or a setting changed: an old result no longer matches it (True if one was cleared).

        A control that re-emits its unchanged value (Enter in a spin box) keeps the result.
        """
        if self._quiet:
            return False
        if self._result_key is not None and self._settings_key() == self._result_key:
            self._update_plan()
            return False
        cleared = self._invalidate("Selection or settings changed: press Calculate FWHM to measure again")
        self._update_plan()
        return cleared

    def _invalidate(self, message):
        """Forget the result (table, summary, plots, export) and ignore a calculation still running.

        Returns:
            True when there was a result to clear (then ``message``, if any, is shown).
        """
        self._generation += 1
        self._result_key = None
        if self._worker is not None:
            self._worker.request_cancel()                # its result would be discarded anyway
        had_result = self.result is not None or self.preview_fit is not None
        self.result = None
        self.preview_fit = None
        self._result_info = None
        self.btn_export.setEnabled(False)
        self._update_store_button()
        self.table.setRowCount(0)
        self.summary.setText("")
        self.fwhm_plot.clear()
        self.slit_plot.clear()
        self._draw_fit()
        if had_result and message:
            self.status.setText(message)
        return had_result

    def _update_plan(self):
        """Describe what *Calculate FWHM* will do."""
        if self.frame is None or self.corrected is None:
            self.plan.setText("")
            return
        if not self._peaks:
            self.plan.setText("No peaks selected: click the spectrum to add peaks, or press Auto detect.")
            return
        blocks = len(row_blocks(self.corrected.shape[0], self.rows_per_block.value()))
        method = self.baseline_settings()[0]
        baseline = "no baseline" if method == "none" else f"baseline {BASELINE_LABELS[method]}"
        self.plan.setText(f"{len(self._peaks)} peaks × {blocks} row blocks · "
                          f"{_PROFILE_LABELS.get(self.fit_profile(), self.fit_profile())} · {baseline}. "
                          "Each block gets its own fitted centres and widths.")

    def _set_busy(self, busy):
        """Lock the selection and fit controls while a calculation runs."""
        for widget in (self.peaks_panel, self.fit_box, self.calibration_box):
            widget.setEnabled(not busy)
        for line in self._peak_lines:
            line.setMovable(not busy)

    # ── measurement ────────────────────────────────────────
    def analysis_kwargs(self):
        """Keyword arguments for :func:`measure_frame` from the selection, the controls and calibrations."""
        from modules.smile.core import auto_sat_level
        pixels = list(self._peaks)
        p = self.profile
        wavelengths = [self.peak_wavelength(px)[0] for px in pixels] if p is not None else None
        saturation = self.saturation.value() or float(auto_sat_level(self.frame))
        method, params = self.baseline_settings()
        return dict(pixels=pixels, wavelengths=wavelengths, rows_per_block=self.rows_per_block.value(),
                    m=p.m if p else None, c=p.c if p else None, saturation=saturation,
                    snr_min=self.snr_min.value(), window_fwhm=self.window_value(), profile=self.fit_profile(),
                    baseline=method, baseline_params=params, valid_columns=self._model_columns(),
                    settings=self.info())

    def info(self) -> dict:
        """Camera setting, calibration sources, preview and selected peaks (stored with the result and in the export)."""
        method = self.baseline_settings()[0]
        peaks = []
        for i, px in enumerate(self._peaks):
            nm, source = self.peak_wavelength(px)
            peaks.append({"label": f"P{i + 1}", "pixel": float(px), "wavelength_nm": nm, "source": source})
        return {"camera_setting": self.setting.label() if self.setting else f"bin {self.camera_bin}",
                "frame_rows": int(self.frame.shape[0]) if self.frame is not None else None,
                "frame_columns": int(self.frame.shape[1]) if self.frame is not None else None,
                "wavelength_profile": self.profile_source or "none (pixels only)",
                "smile_model": self.smile_source or "none (not corrected)",
                "smile_corrected": self.smile is not None,
                "baseline_label": BASELINE_LABELS[method],
                "preview_rows": list(self.preview_rows) if self.preview_rows else None,
                "peak_selection": self._selection_source or "none",
                "selected_peaks": peaks}

    def analyze(self):
        """Calculate the FWHM of the selected peaks in a worker thread (no-op while one runs)."""
        if self.frame is None or self._worker is not None:
            return
        if not self._peaks:
            self.status.setText("No peaks selected: click the spectrum to add peaks, or press Auto detect")
            return
        try:
            kwargs = self.analysis_kwargs()
        except Exception as exc:  # noqa: BLE001
            return self.error(exc)
        r0, r1 = self.preview_rows
        rows = self.corrected[r0:r1 + 1]
        clipped = rows.max(axis=0) >= 0.98 * float(kwargs["saturation"])
        preview = (self.preview.copy(), clipped, tuple(self.columns))
        self._run_generation = self._generation
        self._result_key = self._settings_key()
        self._result_info = kwargs["settings"]
        self.btn_analyze.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._set_busy(True)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self._started = time.perf_counter()
        logger.info("FWHM analysis started: %s, %d peaks, %s, baseline %s", self._result_info["camera_setting"],
                    len(kwargs["pixels"]), kwargs["profile"], kwargs["baseline"])
        worker = _MeasureWorker(self.corrected, kwargs, preview)
        worker.progress.connect(self._on_progress)
        worker.done.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        worker.start()

    def cancel(self):
        """Stop the running calculation (no-op when none runs); nothing of it is kept.

        The fit polls the request at every model evaluation, so it stops within a fraction of
        a second; the controls unlock when the worker has finished.
        """
        if self._worker is None:
            return
        self._worker.request_cancel()
        self.btn_cancel.setEnabled(False)
        self.status.setText("Cancelling…")

    def _on_progress(self, percent, message):
        """Show the worker's progress (not after a cancel request)."""
        if self._worker is not None and self._worker.cancel_requested():
            return
        self.progress.setValue(percent)
        self.status.setText(f"{message}  ({time.perf_counter() - self._started:.0f} s)")

    def _on_failed(self, message):
        """The calculation raised: show why."""
        self._result_key = None
        self.status.setText(f"FWHM failed: {message}")

    def _on_cancelled(self):
        """The calculation stopped on request (Cancel, or a change that made it stale)."""
        if self._run_generation != self._generation:
            self.status.setText("Frame, selection or settings changed during the calculation: result discarded")
            return
        self._result_key = None
        self.status.setText(f"FWHM calculation cancelled after {time.perf_counter() - self._started:.1f} s: "
                            "nothing was kept")

    def _on_finished(self):
        """Release the worker, unlock the controls and hide the progress bar."""
        if self._worker is not None:
            self._worker.wait()
            self._worker.deleteLater()
        self._worker = None
        self.btn_cancel.setEnabled(False)
        self.progress.setVisible(False)
        self._set_busy(False)
        self._update_ready()

    def _on_done(self, result, preview_fit=None):
        """Show the result (dropped if the frame, selection or settings changed meanwhile)."""
        if self._run_generation != self._generation:
            self.status.setText("Frame, selection or settings changed during the calculation: result discarded")
            return
        self.result = result
        self.preview_fit = preview_fit
        self._show_result(result)
        self._draw_fit()
        self.btn_export.setEnabled(True)
        self._update_store_button()
        self.status.setText(f"FWHM done in {time.perf_counter() - self._started:.1f} s")
        self.analysisComplete.emit(result)

    def _draw_fit(self):
        """Fitted model and components of the preview spectrum (after a calculation)."""
        for item in self._fit_items:
            self.spectrum_plot.removeItem(item)
            self._legend.removeItem(item)
        self._fit_items = []
        fit = self.preview_fit
        if fit is None or self.preview is None:
            return
        for w, window in enumerate(fit.windows):
            xs = np.linspace(window.first - 0.5, window.last + 0.5, 8 * (window.last - window.first + 1) + 1)
            xv = self.px_to_view(xs)
            background = window.background_at(xs)
            total = pg.PlotDataItem(xv, window.total(xs), pen=pg.mkPen((255, 60, 60), width=1.5))
            self.spectrum_plot.addItem(total)
            self._fit_items.append(total)
            if w == 0:
                self._legend.addItem(total, f"fit ({_PROFILE_LABELS.get(window.profile, window.profile)})")
            for k, line in enumerate(window.lines):
                colour = pg.intColor(line, hues=max(len(self._peaks), 1)) if line >= 0 else (160, 160, 160)
                curve = pg.PlotDataItem(xv, background + window.component(k, xs),
                                        pen=pg.mkPen(colour, width=1, style=Qt.DotLine))
                self.spectrum_plot.addItem(curve)
                self._fit_items.append(curve)

    def _show_result(self, result):
        """Fill the table, summary and plots."""
        fmt = lambda v, spec: "" if not np.isfinite(v) else format(v, spec)  # noqa: E731
        self.table.setRowCount(len(result.lines))
        for i, l in enumerate(result.lines):
            values = [f"P{i + 1}", fmt(l.wavelength_nm, ".2f"), fmt(l.pixel, ".1f"), fmt(l.fwhm_px_median, ".3f"),
                      fmt(l.fwhm_nm_median, ".4f"), fmt(l.resolving_power_median, ".0f"),
                      f"{l.n_valid}/{l.n_blocks}" + ("" if l.main_flag in ("ok", "undersampled") else f" ({l.main_flag})")]
            for j, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setToolTip(f"16–84 %: {fmt(l.fwhm_px_p16, '.3f')}–{fmt(l.fwhm_px_p84, '.3f')} px; flag {l.main_flag}")
                self.table.setItem(i, j, item)
        s = result.summary()
        in_nm = np.isfinite(s["fwhm_nm_median"])
        settings = result.settings
        method = settings.get("baseline", "none")
        text = (f"{_PROFILE_LABELS.get(settings.get('profile'), settings.get('profile'))} · "
                f"{'no baseline' if method == 'none' else 'baseline ' + BASELINE_LABELS.get(method, method)}\n"
                f"{s['n_lines_valid']}/{s['n_lines']} lines · FWHM median {s['fwhm_px_median']:.3f} px "
                f"(min {s['fwhm_px_min']:.3f}, max {s['fwhm_px_max']:.3f})")
        if in_nm:
            text += (f"\nΔλ median {s['fwhm_nm_median']:.4f} nm (min {s['fwhm_nm_min']:.4f}, "
                     f"max {s['fwhm_nm_max']:.4f}) · R median {s['resolving_power_median']:.0f}")
        else:
            text += "\nNo wavelength profile: widths in pixels only"
        if not settings.get("smile_corrected"):
            text += "\n⚠ not smile corrected"
        if np.isfinite(s["fwhm_px_min"]) and s["fwhm_px_min"] < 2:
            text += "\n⚠ lines narrower than 2 px are undersampled"
        self.summary.setText(text)

        key = "nm" if in_nm else "px"
        self.fwhm_plot.clear()
        xs = np.array([l.wavelength_nm if in_nm else l.pixel for l in result.lines], dtype=float)
        med = np.array([getattr(l, f"fwhm_{key}_median") for l in result.lines], dtype=float)
        lo = np.array([getattr(l, f"fwhm_{key}_p16") for l in result.lines], dtype=float)
        hi = np.array([getattr(l, f"fwhm_{key}_p84") for l in result.lines], dtype=float)
        ok = np.isfinite(xs) & np.isfinite(med)
        if ok.any():
            self.fwhm_plot.addItem(pg.ErrorBarItem(x=xs[ok], y=med[ok], top=hi[ok] - med[ok], bottom=med[ok] - lo[ok], beam=0))
            self.fwhm_plot.plot(xs[ok], med[ok], pen=None, symbol="o", symbolBrush="c")
        self.fwhm_plot.setLabel("bottom", "Wavelength (nm)" if in_nm else "Pixel")
        self.fwhm_plot.setLabel("left", f"FWHM ({key})")
        self.slit_plot.clear()
        rows = result.block_rows.mean(axis=1)
        values = result.array(f"fwhm_{key}")
        for j, l in enumerate(result.lines):
            if np.isfinite(values[:, j]).any():
                label = f"P{j + 1} " + (f"{l.wavelength_nm:.1f} nm" if np.isfinite(l.wavelength_nm) else f"px {l.pixel:.0f}")
                self.slit_plot.plot(rows, values[:, j], pen=pg.intColor(j, hues=max(len(result.lines), 1)), name=label)
        self.slit_plot.setLabel("bottom", "Slit row")
        self.slit_plot.setLabel("left", f"FWHM ({key})")

    # ── export ─────────────────────────────────────────────
    def export_excel(self, path) -> str:
        """Write the current result to ``path`` (``.xlsx``); returns the path.

        Raises:
            RuntimeError: nothing has been measured with the current selection and settings.
        """
        if self.result is None:
            raise RuntimeError("Calculate FWHM first")
        from .excel_report import write_resolution_excel
        return write_resolution_excel(path, self.result, self._result_info or self.info())

    def _update_store_button(self):
        """Saving needs a result in nm, a known camera setting and a store."""
        ok = (self.result is not None and self.setting is not None and self._store is not None
              and np.isfinite(self.result.summary()["fwhm_nm_median"]))
        self.btn_store.setEnabled(bool(ok))

    def store_resolution(self):
        """Store the median FWHM (nm) of the current result for the frame's camera setting.

        The frame's setting must equal the camera's current one. An existing file is copied to
        ``history/`` first; ``resolutionStored`` is emitted with the path.

        Returns:
            The path written, or ``None`` when it was refused (reason in the status line).
        """
        if self.result is None or self.setting is None or self._store is None:
            return None
        summary = self.result.summary()
        if not np.isfinite(summary["fwhm_nm_median"]):
            self.status.setText("Not saved: no FWHM in nm (load a wavelength calibration first)")
            return None
        current = self._current_setting(self.frame.shape, self.camera_bin)
        if current is not None:
            problems = current.mismatch(self.setting)
            if problems:
                self.status.setText("Not saved: the camera is now at a different setting ("
                                    + "; ".join(problems) + "). Grab a new frame.")
                return None
        settings = self.result.settings
        clean = lambda v: float(v) if np.isfinite(v) else None  # noqa: E731
        lines = [{"wavelength_nm": clean(l.wavelength_nm), "pixel": clean(l.pixel),
                  "fwhm_nm_median": clean(l.fwhm_nm_median), "fwhm_px_median": clean(l.fwhm_px_median),
                  "n_valid": int(l.n_valid), "n_blocks": int(l.n_blocks), "flag": l.main_flag}
                 for l in self.result.lines]
        details = {"statistic": "median of the lines' median FWHM",
                   "fwhm_nm_min": clean(summary["fwhm_nm_min"]), "fwhm_nm_max": clean(summary["fwhm_nm_max"]),
                   "fwhm_px_median": clean(summary["fwhm_px_median"]),
                   "n_lines_valid": summary["n_lines_valid"], "n_lines": summary["n_lines"],
                   "profile": settings.get("profile"), "baseline": settings.get("baseline"),
                   "smile_corrected": bool((self._result_info or {}).get("smile_corrected")),
                   "wavelength_profile": (self._result_info or {}).get("wavelength_profile"),
                   "lines": lines}
        try:
            path = self._store.save_resolution(summary["fwhm_nm_median"], self.setting, details)
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Save failed: {exc}")
            return None
        logger.info("FWHM %.4f nm stored for %s: %s", summary["fwhm_nm_median"], self.setting.label(), path)
        self.status.setText(f"FWHM {summary['fwhm_nm_median']:.3f} nm saved for {self.setting.label()}: {path}")
        self.resolutionStored.emit(str(path))
        return str(path)

    def _on_export_png(self):
        """Ask for a file name and write the frame as grabbed/loaded as an 8/16-bit PNG."""
        if self.frame is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export image", png_name("fwhm", self.camera_bin), "PNG (*.png)")
        if not path:
            return
        try:
            save_png(path, self.frame)
            self.status.setText(f"Image exported: {path} ({self.frame.shape[1]}x{self.frame.shape[0]} px, {self.frame.dtype})")
        except Exception as exc:  # noqa: BLE001
            self.error(exc)

    def _on_export(self):
        """Ask for a file name and export."""
        path, _ = QFileDialog.getSaveFileName(self, "Export FWHM", "fwhm_report.xlsx", "Excel workbook (*.xlsx)")
        if not path:
            return
        try:
            self.status.setText(f"Exported: {self.export_excel(path)}")
        except Exception as exc:  # noqa: BLE001
            self.error(exc)

    def error(self, exc):
        """Show ``exc`` in a warning box."""
        QMessageBox.warning(self, "FWHM / Spectral Resolution", str(exc))

    def closeEvent(self, event):  # noqa: N802 (Qt override)
        """Stop a running measurement and wait for its thread (a fraction of a second) before closing."""
        if self._worker is not None:
            self._worker.request_cancel()
            self._worker.wait()
        super().closeEvent(event)
