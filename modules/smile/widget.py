# smile/widget.py
"""PyQt5 widget for interactive smile correction."""
from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Tuple

import numpy as np

# --- Qt imports (guarded) ---
try:
    from PyQt5.QtCore import QThread, Qt, pyqtSignal
    from PyQt5.QtWidgets import (
        QCheckBox,
        QFileDialog,
        QFrame,
        QLabel,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QSpinBox,
        QTabWidget,
        QHBoxLayout,
        QVBoxLayout,
        QAction,
        QMainWindow,
        QWidget,
    )
    import matplotlib

    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    HAVE_QT = True
except ImportError:
    HAVE_QT = False

    # Stubs so class definitions below do not raise NameError.
    QWidget = QThread = FigureCanvasQTAgg = object

    def pyqtSignal(*args, **kwargs):  # noqa: D103
        return None

    class Figure:  # noqa: D101
        pass

# --- Project imports (no Qt dependency) ---
from modules.image import png_name, save_png

from .core import (
    _INTERP,
    auto_sat_level,
    load_frame,
    robust_sigma,
    save_frame,
)
from .analysis import SmileAnalysis, analyze_smile
from .calibration import (
    SmileCalibration,
    calibration_from_analysis,
    crop_margin,
    load_calibration,
    save_calibration,
)
from .corrector import SmileCorrector

logger = logging.getLogger(__name__)

__all__ = [
    "PlotCanvas", "SmileCorrectionWidget", "SmileMainWindow",
    "launch_gui", "HAVE_QT",
]

_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


class PlotCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas wrapper for Qt layouts.

    Args:
        parent: Parent widget.
        figsize: Figure size ``(width, height)`` in inches.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        figsize: Tuple[float, float] = (8, 5),
    ):
        """Args:
            parent: Unused (kept for call-site compatibility).
            figsize: Figure size ``(width, height)`` in inches.
        """
        self.fig = Figure(figsize=figsize, dpi=100)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def clear(self) -> None:
        """Remove all axes and artists from the figure."""
        self.fig.clear()

    def refresh(self) -> None:
        """Apply tight layout and request a canvas redraw."""
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.draw_idle()


class _AnalyzeWorker(QThread):
    """Run :func:`analyze_smile` in a background thread.

    Signals:
        done: Emitted with the :class:`SmileAnalysis` on success.
        failed: Emitted with the error message string on failure.
    """

    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, frame: np.ndarray, kwargs: dict):
        """Args:
            frame: Calibration image ``(H, W)``; read by the worker thread, do not modify while running.
            kwargs: Keyword arguments forwarded to :func:`analyze_smile`.
        """
        super().__init__()
        self._frame = frame
        self._kwargs = kwargs

    def run(self) -> None:
        """Execute the analysis."""
        try:
            self.done.emit(analyze_smile(self._frame, **self._kwargs))
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            self.failed.emit(str(exc))


class _QualityWorker(QThread):
    """Compute diagnostics off the GUI thread; plotting stays on the GUI thread."""

    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str)        # percent 0-100, message

    #: Share of the total time spent in the per-row fits (the rest is the bootstrap).
    FIT_SHARE = 0.75

    def __init__(self, analysis, config):
        """Args:
            analysis: The :class:`SmileAnalysis` to evaluate.
            config: Wavelength configuration in the *Standards* layout (``m_nm_per_px``,
                ``c_nm``, ``standards``, optional ``resolution_nm``).
        """
        super().__init__()
        self.analysis = analysis
        self.config = config

    def run(self):
        """Compute the quality report and shift uncertainty; emit ``done((report, uncertainty, analysis))`` or ``failed(message)``."""
        from .quality import SpectralReference, analyze_quality, estimate_shift_uncertainty

        started = time.perf_counter()

        def fits(fraction, message):
            self.progress.emit(int(100 * self.FIT_SHARE * fraction), message)

        def bootstrap(fraction, message):
            self.progress.emit(int(100 * (self.FIT_SHARE + (1 - self.FIT_SHARE) * fraction)), message)

        try:
            r = self.analysis
            refs = [SpectralReference(item['wavelength_nm'], item['x_px'],
                                      item.get('half_window_px', 32),
                                      item.get('blended', False))
                    for item in self.config['standards']]
            report = analyze_quality(
                r.frame_orig, r.frame_corr, refs,
                m_nm_per_px=self.config['m_nm_per_px'], c_nm=self.config['c_nm'],
                resolution_nm=self.config.get('resolution_nm', 5.),
                row_step=8, y_ref=int(r.y_ref), saturation=r.sat_level,
                progress=fits,
            )
            uncertainty = estimate_shift_uncertainty(
                r, rows=report.rows[:, None],
                columns=np.array([ref.column_px for ref in refs])[None, :],
                repeats=100, progress=bootstrap,
            )
            logger.info("Spectral quality analysis finished in %.1fs", time.perf_counter() - started)
            self.done.emit((report, uncertainty, self.analysis))
        except Exception as exc:
            logger.exception("Spectral quality analysis failed")
            self.failed.emit(str(exc))


class SmileCorrectionWidget(QWidget):
    """Interactive smile correction widget for PyQt5 applications.

    Provides file loading, analysis, visualisation (4 tabs), and
    calibration save / load.  Can be embedded as a sub-widget or used
    standalone.

    Signals:
        analysisComplete: Emitted with :class:`SmileAnalysis` after
            successful analysis.
        statusChanged: Emitted with a status message string.
        applyRequested: Emitted with the path of the model just stored for the camera
            setting (``modules.contracts.CalibrationStore``); the host loads it into the
            live session.

    Host inputs: ``set_calibration_source(store, setting_provider)`` (stored models and
    wavelength profiles per camera setting), ``set_grab_available``, ``begin_grab``,
    ``accept_snapshot``, ``snapshot_failed``.

    Example::

        widget = SmileCorrectionWidget(parent)
        layout.addWidget(widget)
        widget.analysisComplete.connect(my_handler)
        widget.set_frame(raw_frame)
        widget.analyze()          # async -- wait for signal
    """

    grabRequested = pyqtSignal()
    analysisComplete = pyqtSignal(object)
    applyRequested = pyqtSignal(str)
    statusChanged = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        """Build the controls and plot tabs.

        Args:
            parent: Optional parent widget.

        Raises:
            ImportError: PyQt5 or matplotlib is not installed.
        """
        if not HAVE_QT:
            raise ImportError(
                "PyQt5 and matplotlib are required for the widget."
            )
        super().__init__(parent)
        self._frame: Optional[np.ndarray] = None
        self._source_frame: Optional[np.ndarray] = None   # frame as received (dtype kept) for Export PNG
        self._result: Optional[SmileAnalysis] = None
        self._calibration: Optional[SmileCalibration] = None
        self._corrector: Optional[SmileCorrector] = None
        self._worker: Optional[_AnalyzeWorker] = None
        self._quality_worker = None
        self._wavelength_config = None
        self._wavelength_path = None
        self._quality_report = None
        self._quality_uncertainty = None
        self._store = None                 # CalibrationStore (host)
        self._setting_provider = None      # () -> CameraSetting of the camera now
        self._setting = None               # CameraSetting of the current frame
        self._build_ui()

    # -----------------------------------------------------------------
    #  UI construction
    # -----------------------------------------------------------------

    def _build_ui(self) -> None:
        """Construct the widget layout and controls."""
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # -- control row --
        ctrl = QHBoxLayout()

        self.btn_load = QPushButton("📂 โหลดภาพ")
        self.btn_load.clicked.connect(self._on_load)
        ctrl.addWidget(self.btn_load)
        self._grab_request_id = None
        self.btn_grab = QPushButton("Grab from Scan")
        self.btn_grab.setEnabled(False)
        self.btn_grab.setToolTip("Request one frame after dark/hot-pixel correction and orientation")
        self.btn_grab.clicked.connect(lambda: self.grabRequested.emit())
        ctrl.addWidget(self.btn_grab)
        self.btn_export_png = QPushButton("Export PNG…")
        self.btn_export_png.setEnabled(False)
        self.btn_export_png.setToolTip("Save the current image (as grabbed) as an 8/16-bit PNG, values unscaled")
        self.btn_export_png.clicked.connect(self._on_export_png)
        ctrl.addWidget(self.btn_export_png)


        self.btn_analyze = QPushButton("🔍 วิเคราะห์ Smile")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setStyleSheet("font-weight:bold;")
        self.btn_analyze.clicked.connect(self._on_analyze)
        ctrl.addWidget(self.btn_analyze)

        self.btn_save = QPushButton("💾 บันทึก")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._on_save)
        ctrl.addWidget(self.btn_save)

        self.btn_save_cal = QPushButton("📐 บันทึก Matrix")
        self.btn_save_cal.setEnabled(False)
        self.btn_save_cal.setToolTip(
            "Save map_x for reuse without re-analysis."
        )
        self.btn_save_cal.clicked.connect(self._on_save_cal)
        ctrl.addWidget(self.btn_save_cal)

        self.btn_load_cal = QPushButton("📥 โหลด Matrix")
        self.btn_load_cal.setToolTip(
            "Load a previously saved calibration matrix."
        )
        self.btn_load_cal.clicked.connect(self._on_load_cal)
        ctrl.addWidget(self.btn_load_cal)

        self.btn_export_excel = QPushButton("📊 Export Excel")
        self.btn_export_excel.setEnabled(False)
        self.btn_export_excel.setToolTip(
            "Export the smile result (lines, traces, fit, residuals) and, if run, "
            "the spectral-quality analysis to an .xlsx workbook."
        )
        self.btn_export_excel.clicked.connect(self._on_export_excel)

        self.btn_apply = QPushButton("📌 Apply to camera")
        self.btn_apply.setEnabled(False)
        self.btn_apply.setStyleSheet("font-weight:bold;")
        self.btn_apply.setToolTip(
            "Store this model for the frame's camera setting (camera, bin, orientation, size) "
            "and load it into the live HSI session. The previous model is kept in history/."
        )
        self.btn_apply.clicked.connect(self.apply_to_camera)

        ctrl.addStretch()

        # -- fit parameters --
        ctrl.addWidget(QLabel("deg_y:"))
        self.spin_dy = QSpinBox()
        self.spin_dy.setRange(1, 5)
        self.spin_dy.setValue(2)
        ctrl.addWidget(self.spin_dy)

        ctrl.addWidget(QLabel("deg_x:"))
        self.spin_dx = QSpinBox()
        self.spin_dx.setRange(0, 3)
        self.spin_dx.setValue(1)
        ctrl.addWidget(self.spin_dx)

        ctrl.addWidget(QLabel("sat:"))
        self.spin_sat = QSpinBox()
        self.spin_sat.setRange(0, 65535)
        self.spin_sat.setValue(0)
        self.spin_sat.setSpecialValueText("auto")
        self.spin_sat.setToolTip(
            "Sensor full-scale (e.g. 4095 for 12-bit). "
            "auto = detect from image."
        )
        ctrl.addWidget(self.spin_sat)

        self.chk_crop = QCheckBox("ตัดขอบ (หายจริง)")
        self.chk_crop.setChecked(True)
        ctrl.addWidget(self.chk_crop)

        root.addLayout(ctrl)

        quality_controls = QHBoxLayout()
        self.btn_wavelength = QPushButton("โหลด Wavelength Calibration…")
        self.btn_wavelength.setToolTip("Override the wavelength profile stored for this camera setting")
        self.btn_wavelength.clicked.connect(self._on_load_wavelength)
        quality_controls.addWidget(self.btn_wavelength)
        self.btn_quality = QPushButton("วิเคราะห์ Spectral Quality")
        self.btn_quality.clicked.connect(self.run_quality_analysis)
        self.btn_quality.setEnabled(False)
        quality_controls.addWidget(self.btn_quality)
        self.lbl_quality = QLabel("Wavelength calibration: loaded automatically for the frame's camera setting")
        self.lbl_quality.setWordWrap(True)          # long auto-load messages must not widen the window
        self.lbl_quality.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        quality_controls.addWidget(self.lbl_quality, stretch=1)
        self.quality_progress = QProgressBar()
        self.quality_progress.setRange(0, 100)
        self.quality_progress.setFixedWidth(160)
        self.quality_progress.setVisible(False)
        quality_controls.addWidget(self.quality_progress)
        quality_controls.addWidget(self.btn_export_excel)   # second row: the first one is full
        quality_controls.addWidget(self.btn_apply)
        root.addLayout(quality_controls)

        # -- tabbed plot area --
        self.tabs = QTabWidget()
        self._canvases: List[PlotCanvas] = []
        tab_names = [
            "ภาพ  Before / After",
            "เส้น  Traces",
            "Residual",
            "Shift Map",
            "Wavelength",
            "Line Quality",
            "Uncertainty",
            "Cross-correlation",
        ]
        self._plot_export_names = [
            "Smile_Before_After.png",
            "Smile_Traces.png",
            "Smile_Residual.png",
            "Smile_Shift_Map.png",
            "Smile_Wavelength.png",
            "Smile_Line_Quality.png",
            "Smile_Uncertainty.png",
            "Smile_Cross_Correlation.png",
        ]
        for index, name in enumerate(tab_names):
            page = QWidget(self.tabs)
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(2, 2, 2, 2)
            export_row = QHBoxLayout()
            export_row.addStretch()
            button = QPushButton("ส่งออก PNG")
            button.setToolTip(
                "Export this plot as a presentation-quality PNG (300 dpi)."
            )
            button.clicked.connect(
                lambda _checked=False, i=index: self._on_export_plot(i)
            )
            export_row.addWidget(button)
            page_layout.addLayout(export_row)

            canvas = PlotCanvas(page)
            self._canvases.append(canvas)
            page_layout.addWidget(canvas, stretch=1)
            self.tabs.addTab(page, name)

        (
            self.canvas_img,
            self.canvas_trace,
            self.canvas_resid,
            self.canvas_surf,
        ) = self._canvases[:4]
        root.addWidget(self.tabs, stretch=1)
        self._clear_quality_plots()

        self.lbl_setting = QLabel("Camera setting: —")
        self.lbl_setting.setWordWrap(True)
        self.lbl_setting.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        root.addWidget(self.lbl_setting)

        # -- status bar --
        self.lbl_status = QLabel("พร้อมใช้งาน")
        self.lbl_status.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.lbl_status.setMinimumHeight(24)
        root.addWidget(self.lbl_status)

    # -----------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------

    def set_grab_available(self, available):
        """The host enables capture when a snapshot provider is connected."""
        self.btn_grab.setEnabled(bool(available))

    def set_calibration_source(self, store, setting_provider) -> None:
        """Where calibrations live per camera setting and how to read the camera's current setting.

        Args:
            store: ``modules.contracts.CalibrationStore`` (or ``None``).
            setting_provider: Callable returning the current ``modules.contracts.CameraSetting`` or ``None``.
        """
        self._store, self._setting_provider = store, setting_provider

    def _camera_setting(self):
        """The camera's current setting from the host, or ``None``."""
        try:
            return self._setting_provider() if self._setting_provider else None
        except Exception:  # noqa: BLE001
            return None

    def _auto_load_wavelength(self) -> None:
        """Use the wavelength profile stored for the frame's setting for Spectral Quality."""
        self._wavelength_config = None
        self._wavelength_path = None
        if self._store is None or self._setting is None:
            self.lbl_quality.setText("No camera setting: load a wavelength calibration…")
            return
        try:
            stored = self._store.load_wavelength(self._setting)
        except Exception as exc:  # noqa: BLE001
            self.lbl_quality.setText(f"Stored wavelength calibration unreadable: {exc}")
            return
        if stored is None:
            self.lbl_quality.setText(
                f"No wavelength calibration stored for bin {self._setting.camera_bin}: "
                "Apply one in Wavelength Calibration (or load a file)")
        elif not stored.ok:
            self.lbl_quality.setText("Stored wavelength calibration does not match: " + "; ".join(stored.problems))
        else:
            try:
                self.load_wavelength_calibration(str(stored.path))
                self.lbl_quality.setText(f"✓ {stored.path.name} (stored for this setting)")
            except Exception as exc:  # noqa: BLE001
                self.lbl_quality.setText(f"Stored wavelength calibration invalid: {exc}")

    def _show_setting(self) -> None:
        """Show the frame's camera setting and whether a model is already stored for it."""
        s = self._setting
        if s is None:
            self.lbl_setting.setText("Camera setting: unknown (Apply needs a frame from the connected camera)")
            return
        text = "Camera setting: " + s.label()
        if self._store is not None:
            path = self._store.smile_path(s)
            text += f"  ·  stored model: {path.name}" if path.exists() else "  ·  no model stored yet"
        self.lbl_setting.setText(text)

    def _update_apply_state(self) -> None:
        """Apply needs an analysis, a known camera setting and a store."""
        self.btn_apply.setEnabled(self._result is not None and self._setting is not None and self._store is not None)
        if self._setting is not None:
            self.btn_apply.setText(f"📌 Apply to camera (bin {self._setting.camera_bin})")

    def apply_to_camera(self) -> Optional[str]:
        """Store the current model for the frame's camera setting and ask the host to load it.

        The frame's setting must equal the camera's current one (same camera, bin, orientation
        and frame size). An existing model is copied to ``history/`` first.

        Returns:
            The path written, or ``None`` when it was refused (reason in the status bar).
        """
        if self._result is None or self._setting is None or self._store is None:
            return None
        current = self._camera_setting()
        if current is not None:
            problems = current.mismatch(self._setting)
            if problems:
                self._set_status("Not applied: the camera is now at a different setting (" + "; ".join(problems)
                                 + "). Grab a new frame.")
                return None
        if tuple(self._result.frame_orig.shape) != tuple(self._setting.frame_shape or self._result.frame_orig.shape):
            self._set_status("Not applied: analysed frame size differs from the camera frame")
            return None
        path = self._store.smile_path(self._setting)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            backup = self._store.backup(path)
            if self._corrector is not None:
                self._corrector.save(str(path))
            else:
                save_calibration(str(path), calibration_from_analysis(self._result))
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Apply failed: {exc}")
            return None
        logger.info("Smile model stored for %s: %s%s", self._setting.label(), path,
                    f" (previous kept as {backup})" if backup else "")
        self._set_status(f"Stored for {self._setting.label()}: {path}"
                         + (f" (previous → history/{backup.name})" if backup else ""))
        self._show_setting()
        self.applyRequested.emit(str(path))
        return str(path)

    def begin_grab(self, request_id):
        """Mark a snapshot request as pending and disable the Grab button.

        Args:
            request_id: Opaque id; only a snapshot with the same ``request_id`` is accepted.
        """
        self._grab_request_id = request_id
        self.btn_grab.setEnabled(False)
        self._set_status("Waiting for one camera frame…")

    def accept_snapshot(self, packet):
        """Receive the frame requested via ``grabRequested``.

        Args:
            packet: Object with ``request_id``, ``image`` (2-D, oriented, post-dark; never flipped
                again here) and ``camera_bin``. Ignored if the id does not match the pending request.
        """
        if packet.request_id != self._grab_request_id:
            return
        self._grab_request_id = None
        self.btn_grab.setEnabled(True)
        # Snapshot already has the acquisition orientation; never flip twice.
        current = self._camera_setting()
        if current is None:
            self.set_frame(packet.image)
        else:
            self.set_frame(packet.image, setting=current.with_frame(packet.image.shape, packet.camera_bin))
        self._set_status(f"Grabbed oriented post-dark frame, bin {packet.camera_bin}")

    def snapshot_failed(self, request_id, message):
        """Report a failed snapshot request and re-enable Grab.

        Args:
            request_id: Id of the failed request (None matches any); ignored if it is not the pending one.
            message: Text shown in the status label.
        """
        if request_id is not None and request_id != self._grab_request_id:
            return
        self._grab_request_id = None
        self.btn_grab.setEnabled(True)
        self._set_status(message)

    def set_frame(self, frame: Optional[np.ndarray], setting=None) -> None:
        """Set the calibration image from a NumPy array.

        Clears any previous analysis result to prevent saving stale data, and loads the
        wavelength calibration stored for ``setting`` (used by Spectral Quality).

        Args:
            frame: Grayscale image ``(H, W)``, or ``None`` to clear.
            setting: ``CameraSetting`` the frame was taken with (``None`` = unknown; *Apply* is
                then unavailable).
        """
        self._result = None
        self._corrector = None
        self._quality_report = None
        self.btn_quality.setEnabled(False)
        self._clear_quality_plots()
        self.btn_save.setEnabled(False)
        self.btn_save_cal.setEnabled(False)
        self.btn_export_excel.setEnabled(False)
        self._quality_uncertainty = None

        self._source_frame = np.asarray(frame) if frame is not None else None
        self.btn_export_png.setEnabled(
            self._source_frame is not None and self._source_frame.dtype in (np.uint8, np.uint16))
        self._frame = (
            frame.astype(np.float32) if frame is not None else None
        )
        self._setting = setting if frame is not None else None
        self._show_setting()
        self._auto_load_wavelength()
        self._update_apply_state()
        self.btn_analyze.setEnabled(
            self._frame is not None and self._worker is None
        )
        self._show_preview(self._frame)
        if self._frame is not None:
            h, w = self._frame.shape
            self._set_status(
                f"Size {w}x{h}  max={self._frame.max():.0f}  "
                f"sat_level(auto)={auto_sat_level(self._frame):.0f}"
            )

    def load_file(self, path: str) -> None:
        """Load a calibration image from disk.

        Args:
            path: Image file path.
        """
        frame = load_frame(path)
        current = self._camera_setting()
        self.set_frame(frame, setting=current.with_frame(frame.shape) if current else None)
        h, w = self._frame.shape
        self._set_status(
            f"{os.path.basename(path)}  {w}x{h}  "
            f"max={self._frame.max():.0f}  "
            f"sat_level(auto)={auto_sat_level(self._frame):.0f}"
        )

    def analyze(self) -> None:
        """Trigger async analysis on the current frame.

        Connect to ``analysisComplete`` to receive the result.
        """
        self._on_analyze()

    def get_analysis(self) -> Optional[SmileAnalysis]:
        """Return the most recent analysis result, or ``None``."""
        return self._result

    def get_corrected(self) -> Optional[np.ndarray]:
        """Return the corrected frame, or ``None`` if not yet analysed."""
        return self._result.frame_corr if self._result else None

    def get_corrector(self) -> Optional[SmileCorrector]:
        """Return the fitted :class:`SmileCorrector`, or ``None``."""
        return self._corrector

    def load_wavelength_calibration(self, path: str) -> None:
        """Load wavelength coefficients and references without changing Smile fit."""
        import json
        from pathlib import Path

        from .quality import normalize_wavelength_config

        config = normalize_wavelength_config(json.loads(Path(path).read_text(encoding="utf-8")))
        if not config.get("standards") or not np.isfinite(
            [config["m_nm_per_px"], config["c_nm"]]
        ).all() or config["m_nm_per_px"] == 0:
            raise ValueError("Invalid wavelength calibration JSON.")
        for item in config["standards"]:
            if not np.isfinite([item["wavelength_nm"], item["x_px"]]).all():
                raise ValueError("Invalid wavelength reference.")
        self._wavelength_config = config
        self._wavelength_path = str(path)
        self._quality_report = None
        self._quality_uncertainty = None
        self._clear_quality_plots()
        self.lbl_quality.setText(os.path.basename(path))
        self.btn_quality.setEnabled(self._result is not None and self._quality_worker is None)

    def get_quality_report(self):
        """Return the latest GUI quality report, or None while unavailable."""
        return self._quality_report

    def run_quality_analysis(self) -> None:
        """Start diagnostics in a worker; requires Smile result and wavelength JSON."""
        if self._quality_worker is not None or self._result is None or self._wavelength_config is None:
            return
        expected_row = self._wavelength_config.get("reference_row_y")
        if expected_row is not None and expected_row != self._result.y_ref:
            self.lbl_quality.setText(
                f"Reference row mismatch: calibration was made on row {expected_row}, Smile reference "
                f"row is {self._result.y_ref}. Set 'Spectrum row' in Wavelength Calibration to "
                f"{self._result.y_ref} and recalibrate.")
            return
        self.btn_quality.setEnabled(False)
        self.btn_wavelength.setEnabled(False)
        self.lbl_quality.setText("Analyzing quality: every 8 rows; bootstrap 100 repeats…")
        self.quality_progress.setValue(0)
        self.quality_progress.setVisible(True)
        self._quality_started = time.perf_counter()
        logger.info("Spectral quality analysis started (%d reference lines)",
                    len(self._wavelength_config.get("standards", [])))
        worker = _QualityWorker(self._result, dict(self._wavelength_config))
        worker.progress.connect(self._on_quality_progress)
        worker.done.connect(self._on_quality_done)
        worker.failed.connect(self._on_quality_failed)
        worker.finished.connect(self._on_quality_finished)
        self._quality_worker = worker
        worker.start()

    def _on_load_wavelength(self):
        """Ask for a wavelength JSON and load it; errors are shown in the quality label."""
        path, _ = QFileDialog.getOpenFileName(self, "Wavelength calibration", "", "JSON (*.json)")
        if path:
            try:
                self.load_wavelength_calibration(path)
            except Exception as exc:
                self.lbl_quality.setText(f"Calibration error: {exc}")

    def _on_quality_progress(self, percent, message):
        """Show the worker's progress: bar value plus the step and elapsed time in the label."""
        self.quality_progress.setValue(int(percent))
        elapsed = time.perf_counter() - getattr(self, "_quality_started", time.perf_counter())
        self.lbl_quality.setText(f"{message}  ({elapsed:.0f} s)")

    def _on_quality_failed(self, message):
        """Show the failure message from the quality worker."""
        self.lbl_quality.setText(f"Quality analysis failed: {message}")

    def _on_quality_finished(self):
        """Clean up the finished quality worker and re-enable the buttons."""
        self.quality_progress.setVisible(False)
        if self._quality_worker is not None:
            self._quality_worker.wait()
            self._quality_worker.deleteLater()
        self._quality_worker = None
        self.btn_wavelength.setEnabled(True)
        self.btn_quality.setEnabled(self._result is not None and self._wavelength_config is not None)

    def _clear_quality_plots(self):
        """Replace the quality tabs with the "load wavelength calibration" placeholder."""
        for canvas in self._canvases[4:]:
            canvas.clear()
            ax = canvas.fig.add_subplot(111)
            ax.text(.5, .5, "Load wavelength calibration and run Spectral Quality",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            canvas.refresh()

    def _on_quality_done(self, payload):
        """GUI-thread slot: plot the quality report unless the frame changed while it was computed."""
        report, uncertainty, source = payload
        if source is not self._result:
            self.lbl_quality.setText("Frame changed; run quality analysis again")
            return
        self._quality_report = report
        self._quality_uncertainty = uncertainty
        keys = ["wavelength_error_nm", "fwhm_nm", "correlation_shift_px"]
        for index, key, label in [(4, keys[0], "Wavelength error (nm)"),
                                  (7, keys[2], "Relative shift (px)")]:
            canvas = self._canvases[index]
            canvas.clear()
            ax = canvas.fig.add_subplot(111)
            self._draw_quality_lines(ax, report, key)
            ax.set(xlabel="Slit row", ylabel=label, title="Solid: after; dashed: before (in-sample)")
            ax.axhline(0, color="black", lw=.6)
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=7)
            canvas.refresh()
        canvas = self._canvases[5]
        canvas.clear()
        for index, key, label in [(1, "fwhm_nm", "FWHM (nm)"),
                                  (2, "peak_height", "Peak height (counts)"),
                                  (3, "area_counts_px", "Area (counts px)")]:
            ax = canvas.fig.add_subplot(3, 1, index)
            self._draw_quality_lines(ax, report, key)
            ax.set(xlabel="Slit row", ylabel=label)
            ax.grid(alpha=.2)
            if index == 1:
                ax.set_title("Solid: after; dashed: before; Gaussian + linear background")
                if ax.lines:
                    ax.legend(fontsize=6, ncol=2)
        canvas.refresh()
        canvas = self._canvases[6]
        canvas.clear()
        ax = canvas.fig.add_subplot(111)
        for j, wavelength in enumerate(report.wavelengths_nm):
            if report.metadata["excluded_blends"][j]:
                continue
            supported = uncertainty["supported"][:, j]
            std = np.where(supported, uncertainty["std_px"][:, j], np.nan)
            ax.plot(report.rows, std, label=f"{wavelength:g} nm")
        ax.set(xlabel="Slit row", ylabel="Shift standard deviation (px)",
               title="Conditional row bootstrap; gaps = outside calibration support")
        if ax.lines:
            ax.legend(fontsize=7)
        ax.grid(alpha=.2)
        canvas.refresh()
        stats = report.summary["paired_wavelength_after_nm"]
        self.lbl_quality.setText(
            f"In-sample RMSE {stats['rmse']:.3f} nm; "
            f"{stats['count']} paired fits; blends omitted; row step 8"
        )
        self.tabs.setCurrentIndex(4)

    @staticmethod
    def _draw_quality_lines(ax, report, key):
        """Plot ``report.after[key]`` (solid) and ``report.before[key]`` (dashed) per row for each non-blended line."""
        for j, wavelength in enumerate(report.wavelengths_nm):
            if report.metadata["excluded_blends"][j]:
                continue
            if not np.isfinite(report.after[key][:, j]).any():
                continue
            line, = ax.plot(report.rows, report.after[key][:, j], label=f"{wavelength:g} nm")
            ax.plot(report.rows, report.before[key][:, j], "--", color=line.get_color(), alpha=.5)

    def closeEvent(self, event):
        """Keep running workers alive until completion before destroying the widget."""
        for worker in [self._worker, self._quality_worker]:
            if worker is not None and worker.isRunning():
                worker.wait()
        super().closeEvent(event)

    def analyze_quality(self, references, *, m_nm_per_px, c_nm, **kwargs):
        """Return spectral diagnostics for the current uncropped analysis.

        Call after ``analysisComplete``. This synchronous diagnostic can be
        expensive; use a host worker for full-height measurements. Reports are
        in-sample because this widget's current frame also fits the correction.
        """
        from .quality import analyze_quality

        if self._result is None:
            raise ValueError("Analyze a frame before requesting quality diagnostics.")
        kwargs.pop("independent_frame", None)
        kwargs.setdefault("y_ref", int(self._result.y_ref))
        return analyze_quality(
            self._result.frame_orig, self._result.frame_corr, references,
            m_nm_per_px=m_nm_per_px, c_nm=c_nm,
            independent_frame=False, **kwargs,
        )

    def set_params(
        self,
        deg_y: Optional[int] = None,
        deg_x: Optional[int] = None,
        crop: Optional[bool] = None,
        sat_level: Optional[int] = None,
    ) -> None:
        """Update fit parameters programmatically.

        Args:
            deg_y: Polynomial degree in the spatial direction.
            deg_x: Polynomial degree in the spectral direction.
            crop: Whether to crop invalid borders on save.
            sat_level: Sensor full-scale value (0 = auto-detect).
        """
        if deg_y is not None:
            self.spin_dy.setValue(deg_y)
        if deg_x is not None:
            self.spin_dx.setValue(deg_x)
        if crop is not None:
            self.chk_crop.setChecked(crop)
        if sat_level is not None:
            self.spin_sat.setValue(int(sat_level))

    def apply_calibration(
        self, frame: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply a previously loaded calibration matrix to a frame.

        Args:
            frame: Target image.  Falls back to ``self._frame`` if ``None``.

        Returns:
            Corrected and cropped image.

        Raises:
            ValueError: If no calibration is loaded or no frame available.
        """
        if self._calibration is None:
            raise ValueError("No calibration matrix loaded.")
        src = frame if frame is not None else self._frame
        if src is None:
            raise ValueError("No frame available.")
        corrected = self._calibration.apply_cropped(src)
        self._set_status(
            f"Applied: {self._calibration.image_shape[1]}x"
            f"{self._calibration.image_shape[0]} -> "
            f"{corrected.shape[1]}x{corrected.shape[0]}"
        )
        return corrected

    # -----------------------------------------------------------------
    #  Internal helpers
    # -----------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        """Update the status label and emit ``statusChanged``."""
        self.lbl_status.setText(msg)
        self.statusChanged.emit(msg)

    def _show_preview(self, frame: Optional[np.ndarray]) -> None:
        """Draw the initial preview in the image tab."""
        self.canvas_img.clear()
        ax = self.canvas_img.fig.add_subplot(111)
        if frame is not None:
            vmin, vmax = np.percentile(frame, [1, 99.5])
            ax.imshow(
                frame, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax
            )
            ax.set_title("Original -- waiting for analysis")
        self.canvas_img.refresh()

    def _on_load(self) -> None:
        """Handle the 'Load Image' button click."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "เลือกภาพ calibration",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All (*.*)",
        )
        if path:
            try:
                self.load_file(path)
            except Exception as exc:
                self._set_status(f"Error: {exc}")

    def _on_analyze(self) -> None:
        """Handle the 'Analyze' button click (async via QThread)."""
        if self._frame is None or self._worker is not None:
            return
        kwargs = dict(
            sat_level=(self.spin_sat.value() or None),
            deg_y=self.spin_dy.value(),
            deg_x=self.spin_dx.value(),
            verbose=True,
        )
        self._set_status("Analyzing ...")
        self.btn_analyze.setEnabled(False)
        worker = _AnalyzeWorker(self._frame.copy(), kwargs)
        worker.done.connect(self._on_analysis_done)
        worker.failed.connect(self._on_analysis_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_worker_finished(self) -> None:
        """Clean up after the analysis thread completes."""
        if self._worker is not None:
            self._worker.wait()
        self._worker = None
        self.btn_analyze.setEnabled(self._frame is not None)

    def _on_analysis_failed(self, msg: str) -> None:
        """Handle analysis failure (safe: queued to main thread)."""
        self._set_status(f"Error: {msg}")

    def _on_analysis_done(self, r: SmileAnalysis) -> None:
        """Handle successful analysis result."""
        self._result = r

        # Reuse the completed analysis through the corrector's public factory.
        self._corrector = SmileCorrector.from_analysis(r)

        cv_str = (
            f"{r.cv_rms:.3f}" if np.isfinite(r.cv_rms) else "n/a"
        )
        self._set_status(
            f"Lines {len(r.used_lines)}/{len(r.all_lines)}  |  "
            f"RMS {r.rms_before:.3f}->{r.rms_after:.3f} px "
            f"(CV {cv_str})  |  "
            f"margin L={r.margin[0]} R={r.margin[1]} px "
            f"({r.margin[0] + r.margin[1]} px lost)"
        )
        self.btn_save.setEnabled(True)
        self.btn_save_cal.setEnabled(True)
        self.btn_export_excel.setEnabled(True)
        self._update_apply_state()
        self._update_all_plots(r)
        self.btn_quality.setEnabled(
            self._wavelength_config is not None and self._quality_worker is None
        )
        self._quality_report = None
        self._quality_uncertainty = None
        self._clear_quality_plots()
        self.analysisComplete.emit(r)

    def _on_export_png(self) -> None:
        """Handle 'Export PNG…': save the image as received, at its own bit depth."""
        if self._source_frame is None:
            return
        camera_bin = self._setting.camera_bin if self._setting is not None else None
        path, _ = QFileDialog.getSaveFileName(
            self, "Export image", png_name("smile", camera_bin), "PNG (*.png)")
        if not path:
            return
        try:
            save_png(path, self._source_frame)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Export failed: {exc}")
            return
        h, w = self._source_frame.shape
        self._set_status(f"Exported {path} ({w}x{h} px, {self._source_frame.dtype})")

    def _on_save(self) -> None:
        """Handle the 'Save Image' button click."""
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "บันทึกภาพที่แก้ smile แล้ว",
            "Corrected.png",
            "PNG (*.png);;TIFF (*.tif);;All (*.*)",
        )
        if not path:
            return
        img = self._result.frame_corr
        if self.chk_crop.isChecked():
            img = crop_margin(img, self._result.margin)
        bits = 8 if self._result.sat_level <= 255 else 16
        save_frame(path, img, bits=bits)
        self._set_status(
            f"Saved {path} ({img.shape[1]}x{img.shape[0]} px, {bits}-bit)"
        )

    def export_excel(self, path: str) -> str:
        """Write the current smile result (and quality report, if any) to an ``.xlsx`` file.

        Args:
            path: Output path; ``.xlsx`` is appended if missing.

        Returns:
            The path written.

        Raises:
            RuntimeError: no smile analysis has been run.
        """
        if self._result is None:
            raise RuntimeError("Run the smile analysis first")
        from .excel_report import write_smile_excel

        metadata = {"crop_margin": bool(self.chk_crop.isChecked())}
        if self._wavelength_path:
            metadata["wavelength_calibration_file"] = self._wavelength_path
        return write_smile_excel(path, self._result, quality=self._quality_report,
                                 uncertainty=self._quality_uncertainty, metadata=metadata)

    def _on_export_excel(self) -> None:
        """Ask for a file name and export the workbook; the result is shown in the status label."""
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "ส่งออกผล Smile เป็น Excel", "smile_report.xlsx", "Excel workbook (*.xlsx)")
        if not path:
            return
        try:
            written = self.export_excel(path)
            extra = " + spectral quality" if self._quality_report is not None else ""
            self._set_status(f"Exported Excel{extra}: {written}")
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Excel export failed: {exc}")

    def _on_export_plot(self, index: int) -> None:
        """Save one tab's current Matplotlib figure as a 300-dpi PNG."""
        if not 0 <= index < len(self._canvases):
            self._set_status(f"Invalid plot tab index: {index}")
            return
        default_name = self._plot_export_names[index]
        path, _ = QFileDialog.getSaveFileName(
            self,
            "ส่งออกกราฟเป็น PNG",
            default_name,
            "PNG image (*.png)",
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            self._canvases[index].fig.savefig(
                path,
                format="png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            self._set_status(f"Exported plot PNG: {path} (300 dpi)")
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Plot export failed: {exc}")

    def _on_save_cal(self) -> None:
        """Handle the 'Save Matrix' button click."""
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "บันทึก Smile Calibration Matrix",
            "smile_calibration.npz",
            "Smile Calibration (*.npz);;All (*.*)",
        )
        if not path:
            return
        if not path.endswith(".npz"):
            path += ".npz"
        if self._corrector is not None:
            self._corrector.save(path)
        else:
            cal = calibration_from_analysis(self._result)
            save_calibration(path, cal)
        self._set_status(f"Saved calibration: {path}")

    def _on_load_cal(self) -> None:
        """Handle the 'Load Matrix' button click."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "โหลด Smile Calibration Matrix",
            "",
            "Smile Calibration (*.npz);;All (*.*)",
        )
        if not path:
            return
        try:
            self._calibration = load_calibration(path)
            self._corrector = SmileCorrector.load(path)
            self._set_status(
                self._calibration.summary().replace("\n", " | ")
            )
        except Exception as exc:
            self._set_status(f"Load failed: {exc}")

    # -----------------------------------------------------------------
    #  Plotting
    # -----------------------------------------------------------------

    def _update_all_plots(self, r: SmileAnalysis) -> None:
        """Redraw all four analysis tabs."""
        self._plot_image(r)
        self._plot_traces(r)
        self._plot_residual(r)
        self._plot_surface(r)

    def _plot_image(self, r: SmileAnalysis) -> None:
        """Render the Before / After image comparison tab."""
        self.canvas_img.clear()
        fig = self.canvas_img.fig

        vmin, vmax = np.percentile(r.frame_orig, [1, 99.5])
        n_cols = r.frame_orig.shape[1]

        ax1 = fig.add_subplot(211)
        ax1.imshow(
            r.frame_orig, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax
        )
        ax1.set_title(f"Before  ({n_cols} cols)", fontsize=10)
        ax1.set_ylabel("Row")

        ax2 = fig.add_subplot(212)
        img = r.frame_corr
        left, right = r.margin
        if self.chk_crop.isChecked() and (left + right > 0):
            img = crop_margin(img, (left, right))
            title = (
                f"After  ({img.shape[1]} cols -- cropped "
                f"L={left} R={right}, {left + right} px lost)"
            )
        else:
            title = (
                f"After  ({img.shape[1]} cols -- not cropped, "
                f"border zeros = no data)"
            )
        ax2.imshow(img, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
        ax2.set_title(title, fontsize=10)
        ax2.set_xlabel("Column (spectral)")
        ax2.set_ylabel("Row")
        self.canvas_img.refresh()

    def _plot_traces(self, r: SmileAnalysis) -> None:
        """Render the Traces tab (before / after comparison)."""
        self.canvas_trace.clear()
        fig = self.canvas_trace.fig
        rows = np.arange(r.frame_orig.shape[0])

        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        for i, ln in enumerate(r.used_lines):
            color = _COLORS[i % len(_COLORS)]
            label = f"col {ln.column:.0f}"
            ref = r.refs[ln.column]

            # Before: offset from per-line reference.
            c = r.traces_orig.get(ln.column)
            if c is not None:
                v = np.isfinite(c)
                ax1.plot(
                    c[v] - ref, rows[v],
                    ".", ms=0.5, color=color, label=label,
                )

            # After: use the corrected trace's own reference.
            c2 = r.traces_corr.get(ln.column)
            if c2 is not None:
                v2 = np.isfinite(c2)
                if v2.any():
                    ref2 = float(np.median(c2[v2]))
                    ax2.plot(
                        c2[v2] - ref2, rows[v2],
                        ".", ms=0.5, color=color, label=label,
                    )

        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax1.set_ylabel("Row")
        ax1.set_title(
            f"Before -- RMS {r.rms_before:.4f} px", fontsize=10
        )
        if len(r.used_lines) <= 12:
            ax1.legend(fontsize=5, loc="upper right", ncol=2)

        ax2.set_xlabel("Shift from per-line reference (px)")
        ax2.set_ylabel("Row")
        ax2.set_title(
            f"After -- RMS {r.rms_after:.4f} px  (target: near 0)",
            fontsize=10,
        )
        ax2.axvline(0, color="black", lw=0.5)
        self.canvas_trace.refresh()

    def _plot_residual(self, r: SmileAnalysis) -> None:
        """Render the Residual tab (scatter + histogram)."""
        self.canvas_resid.clear()
        fig = self.canvas_resid.fig

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        sig = robust_sigma(r.residuals)

        for i, ln in enumerate(r.used_lines):
            color = _COLORS[i % len(_COLORS)]
            mask = r.meas_id == i
            if np.any(mask):
                ax1.scatter(
                    r.meas_y[mask], r.residuals[mask],
                    s=0.5, alpha=0.3, color=color,
                )

        ax1.axhline(0, color="black", lw=0.5)
        ax1.axhline(
            +2 * sig,
            color="red", lw=0.5, ls="--",
            label=f"+/-2 sigma = {2 * sig:.4f}",
        )
        ax1.axhline(-2 * sig, color="red", lw=0.5, ls="--")
        ax1.set_xlabel("Row (spatial)")
        ax1.set_ylabel("Residual (px)")
        cv_str = (
            f"{r.cv_rms:.5f}" if np.isfinite(r.cv_rms) else "n/a"
        )
        ax1.set_title(
            f"Residual vs Row  (RMS={r.fit_info['rms']:.5f}, "
            f"CV={cv_str}, n_used={r.fit_info['n_used']})",
            fontsize=9,
        )
        ax1.legend(fontsize=7)

        ax2.hist(
            r.residuals, bins=50,
            edgecolor="black", lw=0.4,
            color="steelblue", alpha=0.8,
        )
        ax2.axvline(0, color="black", lw=0.5)
        ax2.set_xlabel("Residual (px)")
        ax2.set_ylabel("Count")
        ax2.set_title("Distribution", fontsize=9)
        self.canvas_resid.refresh()

    def _plot_surface(self, r: SmileAnalysis) -> None:
        """Render the Shift Map tab (2-D displacement surface)."""
        self.canvas_surf.clear()
        ax = self.canvas_surf.fig.add_subplot(111)

        n_rows, n_cols = r.frame_orig.shape
        yy = np.linspace(0, n_rows - 1, 100)
        xx = np.linspace(0, n_cols - 1, 100)
        XX, YY = np.meshgrid(xx, yy)
        shift_px = np.rint(r.surface(YY, XX)).astype(np.int32)
        levels = np.arange(
            int(shift_px.min()), int(shift_px.max()) + 1, dtype=np.int32
        )
        boundaries = np.arange(
            levels[0] - 0.5, levels[-1] + 1.5, 1.0
        )

        # Treat each rounded pixel shift as a category. Tab20 provides clear,
        # distinct colors for the small integer range typical of smile maps.
        from matplotlib.colors import BoundaryNorm, ListedColormap

        base_cmap = matplotlib.colormaps["tab20"]
        cmap = ListedColormap(
            [base_cmap(i % base_cmap.N) for i in range(levels.size)],
            name="tab20_integer_shift",
        )
        norm = BoundaryNorm(boundaries, ncolors=cmap.N)
        im = ax.pcolormesh(
            xx, yy, shift_px, cmap=cmap, norm=norm, shading="auto"
        )
        self.canvas_surf.fig.colorbar(
            im, ax=ax, ticks=levels, label="Shift (px) — integer"
        )

        ax.scatter(r.meas_x, r.meas_y, s=0.3, c="black", alpha=0.15)
        ax.set_xlabel("Column (spectral)")
        ax.set_ylabel("Row (spatial)")
        ax.set_title(
            f"Smile Shift Surface  "
            f"[deg_y={r.surface.deg_y}, deg_x={r.surface.deg_x}]  "
            f"(clamped outside line range)",
            fontsize=10,
        )
        self.canvas_surf.refresh()


if HAVE_QT:
    class SmileMainWindow(QMainWindow):
        """Standalone main window with menus around the embeddable widget.

        Applications embedding Smile should use :class:`SmileCorrectionWidget`
        directly. This window is a convenience shell for standalone use.
        """

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self.setWindowTitle("Smile Correction")
            self.resize(1150, 780)
            self.smile_widget = SmileCorrectionWidget(self)
            self.setCentralWidget(self.smile_widget)
            self._build_menus()

        def _build_menus(self) -> None:
            file_menu = self.menuBar().addMenu("&File")
            open_action = QAction("&Open image…", self)
            open_action.setShortcut("Ctrl+O")
            open_action.triggered.connect(self._open_image)
            file_menu.addAction(open_action)

            save_action = QAction("Save corrected image…", self)
            save_action.setShortcut("Ctrl+S")
            save_action.triggered.connect(self.smile_widget._on_save)
            file_menu.addAction(save_action)

            file_menu.addSeparator()
            exit_action = QAction("E&xit", self)
            exit_action.setShortcut("Ctrl+Q")
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)

            process_menu = self.menuBar().addMenu("&Process")
            analyze_action = QAction("Analyze Smile", self)
            analyze_action.setShortcut("F5")
            analyze_action.triggered.connect(self.smile_widget.analyze)
            process_menu.addAction(analyze_action)

            calibration_menu = self.menuBar().addMenu("&Calibration")
            load_cal_action = QAction("Load matrix…", self)
            load_cal_action.triggered.connect(self.smile_widget._on_load_cal)
            calibration_menu.addAction(load_cal_action)
            save_cal_action = QAction("Save matrix…", self)
            save_cal_action.triggered.connect(self.smile_widget._on_save_cal)
            calibration_menu.addAction(save_cal_action)

            help_menu = self.menuBar().addMenu("&Help")
            about_action = QAction("About Smile Correction", self)
            about_action.triggered.connect(self._show_about)
            help_menu.addAction(about_action)

        def _open_image(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open calibration image",
                "",
                "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All (*.*)",
            )
            if path:
                try:
                    self.smile_widget.load_file(path)
                except Exception as exc:  # noqa: BLE001
                    self.smile_widget._set_status(f"Error: {exc}")

        def _show_about(self) -> None:
            from PyQt5.QtWidgets import QMessageBox

            QMessageBox.about(
                self,
                "About Smile Correction",
                "Smile Correction\n"
                "Standalone spectrometer smile analysis and correction.\n"
                "The SmileCorrectionWidget can also be embedded in other "
                "PyQt5 applications.",
            )


    def launch_gui(argv: Optional[List[str]] = None) -> int:
        """Launch the standalone window, reusing an existing QApplication.

        If no application exists, this function creates one and starts its
        event loop. When called from a host PyQt5 app it only shows and returns
        the window, leaving event-loop ownership with the host.
        """
        import sys
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance()
        owns_app = app is None
        if owns_app:
            app = QApplication(list(argv) if argv is not None else sys.argv)
        window = SmileMainWindow()
        args = list(argv) if argv is not None else sys.argv[1:]
        if args and os.path.isfile(args[0]):
            window.smile_widget.load_file(args[0])
        window.show()
        window._smile_app = app
        if owns_app:
            return int(app.exec_())
        return 0
else:
    class SmileMainWindow:  # noqa: D101
        def __init__(self, *args, **kwargs):
            raise ImportError("PyQt5 and matplotlib are required for the GUI.")

    def launch_gui(argv: Optional[List[str]] = None) -> int:  # noqa: D103
        raise ImportError("PyQt5 and matplotlib are required for the GUI.")
