# windows/scan_window.py
from __future__ import annotations

import math
import time

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QGroupBox,
    QSplitter,
    QShortcut,
    QScrollArea,
    QSizePolicy,
    QFileDialog,
    QMessageBox,
)

from config import AppConfig
from modules.live_view.mapping import DisplayMapper
from modules.live_view.image_canvas import (
    FOCUS_ROI_DEFAULT_HEIGHT,
    FOCUS_ROI_MIN_HEIGHT,
)
from modules.camera.widget import CameraControls
from modules.live_view.spectrum import LiveLineSpectrumWidget
from modules.cube_preview.widget import RealtimeCubePreviewWidget
from modules.dark.widget import DarkFramePanel
from modules.stage.config import FREE_SCAN_MAX_S, FREE_SCAN_MIN_S
from modules.stage.speed import LINEAR_MAX_UM_S, ROTATION_MAX_DPS, ROTATION_SPEED_RESOLUTION_DPS
from modules.acquisition.widget import CorrectionStatusDisplay
from modules.live_view.display_controls import DisplayControls
from widgets.sharpness_panel import SharpnessPanel
from modules.live_view.histogram import HistogramPlot
from windows.image_viewer import ImageViewer
from modules.spectrogram_viewer.window import SpectrogramViewer

#: Provisional limits of the wavelength range spin boxes until the camera reports its axis.
WAVELENGTH_LIMITS_NM = (0.0, 9999.9)
#: Title of the cube preview box; a scan appends its settings (see ScanWindow.set_scanning).
CUBE_PREVIEW_TITLE = "Realtime Cube Preview"
#: A scan's measured line rate below this fraction of the camera's frame rate is warned about
#: (ScanWindow.status_line_rate_warning), once the scan has at least this many lines (a short
#: scan's rate is too coarse: one frame of jitter is 1 / lines of it).
LINE_RATE_WARN_FRACTION = 0.95
LINE_RATE_WARN_MIN_FRAMES = 50


def _format_duration(seconds: float) -> str:
    """``12.3 s`` below a minute, ``m:ss`` below an hour, else ``h:mm:ss``."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


#: The free-scan countdown when no free scan is running.
COUNTDOWN_IDLE = "—"
#: How often the free-scan countdown is checked while it runs (ms): half its 0.1 s step. A
#: 100 ms timer starts in step with the tenths, so a tick a few ms late shows one tenth twice
#: and skips the next; at 50 ms every tenth shows. The label and title are redrawn only when
#: their text changes.
COUNTDOWN_INTERVAL_MS = 50


def _format_countdown(seconds: float) -> str:
    """Time left, rounded UP so it reads ``0.0 s`` only when the time is over: ``7.3 s`` below
    a minute, ``m:ss`` below an hour, else ``h:mm:ss``."""
    seconds = max(0.0, float(seconds))
    tenths = math.ceil(seconds * 10 - 1e-9)
    if tenths < 600:
        return f"{tenths / 10:.1f} s"
    hours, rest = divmod(math.ceil(seconds - 1e-9), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class _NoWheelUnlessFocusedSpinBox(QDoubleSpinBox):
    """A spin box that ignores the mouse wheel unless it has focus.

    Scrolling the settings panel over it then scrolls the panel instead of changing a
    value that decides what is recorded.
    """

    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)          # the wheel does not give focus

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class ScanWindow(QWidget):
    snapshot_ready = pyqtSignal(object)
    snapshot_failed = pyqtSignal(int, str)
    cube_ready = pyqtSignal(object, object)
    calibration_requested = pyqtSignal(object, object)
    #: *Save as Radiometric Reference*: (cube, wavelengths) of the scan just finished.
    radiometric_reference_requested = pyqtSignal(object, object)
    #: Save on the unsaved-result prompt: open (cube, wavelengths) with the export dialog.
    save_result_requested = pyqtSignal(object, object)
    #: Calibrate (cube, wavelengths) with the camera's matching radiometric reference: sent after
    #: a scan while *Auto radiometric calibration* is ticked, and when it is ticked afterwards.
    auto_calibration_requested = pyqtSignal(object, object)
    #: The user finished a wavelength range edit: Enter, or focus leaving the Start/End pair.
    wavelength_range_committed = pyqtSignal()

    def request_snapshot(self, request_id):
        controller = getattr(self, 'controller', None)
        worker = getattr(controller, 'preview_thread', None)
        if worker is None or not worker.isRunning():
            self.snapshot_failed.emit(request_id, 'Camera acquisition is not running')
        elif controller.dark_lib.is_collecting:
            self.snapshot_failed.emit(request_id, 'Finish dark capture before grabbing a frame')
        else:
            worker.request_snapshot(request_id)

    def request_processed_frame(self):
        """Ask the camera for the next preprocessed frame (before binning); it arrives in ``save_processed_frame``."""
        controller = getattr(self, 'controller', None)
        worker = getattr(controller, 'preview_thread', None)
        if worker is None or not worker.isRunning():
            QMessageBox.warning(self, 'Save Frame', 'Camera acquisition is not running')
        elif controller.dark_lib.is_collecting:
            QMessageBox.warning(self, 'Save Frame', 'Finish dark capture before saving a frame')
        else:
            worker.request_processed_frame()

    def save_processed_frame(self, packet):
        """Ask for a file name and write ``packet.image`` as an 8/16-bit PNG (values unscaled)."""
        from modules.image import png_name, save_png
        tags = (('dark',) if packet.dark_applied else ()) + (('smile',) if packet.stage == 'post_smile' else ())
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save preprocessed frame', png_name('scan', packet.camera_bin, tags), 'PNG (*.png)')
        if not path:
            return
        try:
            save_png(path, packet.image)
        except Exception as exc:
            QMessageBox.warning(self, 'Save Frame', str(exc))

    def __init__(self, config: AppConfig):
        super().__init__()
        self.camera_controls = CameraControls()
        self.camera_controls.set_limits(1, 300, 1, 1000)

        self.camera_layout = QVBoxLayout()
        self.camera_layout.addWidget(self.camera_controls)

        self.config = config
        self.viewer = ImageViewer()
        
        # ── Dark frame ───────────────────────────────────────────
        self.dark_panel = DarkFramePanel()
        self.correction_status = CorrectionStatusDisplay(self)
        self.correction_status.dark.setFixedWidth(180)
        self.correction_status.smile.setFixedWidth(180)
        self.correction_status.wavelength.setFixedWidth(180)

        # ── Stage — rotation ─────────────────────────────────────
        # The rotation stage runs multiples of 0.01 deg/s (half steps of 0.02 deg), so the
        # value entered is the value it runs.
        self.degree_per_second_spin = QDoubleSpinBox()
        self.degree_per_second_spin.setDecimals(2)
        self.degree_per_second_spin.setRange(ROTATION_SPEED_RESOLUTION_DPS, ROTATION_MAX_DPS)
        self.degree_per_second_spin.setValue(float(getattr(config, "rotation_speed_dps", 1.0)))
        self.degree_per_second_spin.setSuffix(" deg/s")
        self.degree_per_second_spin.setToolTip(
            f"The stage runs multiples of {ROTATION_SPEED_RESOLUTION_DPS:g} deg/s, "
            f"up to {ROTATION_MAX_DPS:g} deg/s")

        self.degree_spin = QDoubleSpinBox()
        self.degree_spin.setDecimals(3)
        self.degree_spin.setRange(0.001, 360000.0)
        self.degree_spin.setValue(float(getattr(config, "rotation_degree", 10.0)))
        self.degree_spin.setSuffix(" deg")

        rotation_layout = QFormLayout()
        rotation_layout.addRow("Rotation Speed", self.degree_per_second_spin)
        rotation_layout.addRow("Rotation Degree", self.degree_spin)

        self.rotation_group = QGroupBox("Rotation Stage")
        self.rotation_group.setLayout(rotation_layout)

        # ── Stage — linear ───────────────────────────────────────
        self.distance_mm_spin = QDoubleSpinBox()
        self.distance_mm_spin.setDecimals(3)
        self.distance_mm_spin.setRange(0.001, 100000.0)
        self.distance_mm_spin.setValue(float(getattr(config, "linear_distance_mm", 10.0)))
        self.distance_mm_spin.setSuffix(" mm")

        self.speed_um_s_spin = QDoubleSpinBox()
        self.speed_um_s_spin.setDecimals(3)
        self.speed_um_s_spin.setRange(0.001, LINEAR_MAX_UM_S)
        self.speed_um_s_spin.setValue(float(getattr(config, "linear_speed_um_s", 1000.0)))
        self.speed_um_s_spin.setSuffix(" um/s")
        self.speed_um_s_spin.setToolTip(
            f"Up to {LINEAR_MAX_UM_S:g} um/s. The stage runs the nearest microstep speed "
            "(1 step = 20 um; 0.078 um/s at 1/256 microsteps); the speed it runs is shown in "
            "the cube preview title if different and stored in the scan metadata")

        linear_layout = QFormLayout()
        linear_layout.addRow("Linear Distance", self.distance_mm_spin)
        linear_layout.addRow("Linear Speed", self.speed_um_s_spin)

        self.linear_group = QGroupBox("Linear Stage")
        self.linear_group.setLayout(linear_layout)

        # ── Free scan (no stage) ─────────────────────────────────
        # stage_type "free": nothing moves; the camera records for this time while the
        # scene passes the slit by itself (conveyor, vehicle, hand-held sweep).
        self.free_duration_spin = QDoubleSpinBox()
        self.free_duration_spin.setDecimals(2)
        self.free_duration_spin.setRange(FREE_SCAN_MIN_S, FREE_SCAN_MAX_S)
        self.free_duration_spin.setSingleStep(1.0)
        self.free_duration_spin.setValue(float(getattr(config, "free_duration_s", 10.0)))
        self.free_duration_spin.setSuffix(" s")
        self.free_duration_spin.setToolTip(
            "No stage: the camera records for this many seconds from Start "
            "(Stop ends it early; the frames so far become the cube)")

        # Countdown of the running free scan (only the time box is locked while it runs, so
        # the countdown stays readable when the folded settings pane is dragged open); the
        # cube preview title and the status line also show the time left.
        self.free_countdown_label = QLabel(COUNTDOWN_IDLE)
        self.free_countdown_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.free_countdown_label.setToolTip("Time left in the running free scan")
        self._countdown_timer = QTimer(self)
        # Precise: a coarse timer drifts (a 100 ms one ticks every ~107-109 ms on Windows).
        self._countdown_timer.setTimerType(Qt.PreciseTimer)
        self._countdown_timer.setInterval(COUNTDOWN_INTERVAL_MS)
        self._countdown_timer.timeout.connect(self._update_countdown)

        free_layout = QFormLayout()
        free_layout.addRow("Scan Time", self.free_duration_spin)
        free_layout.addRow("Remaining", self.free_countdown_label)

        self.free_group = QGroupBox("Free Scan (no stage)")
        self.free_group.setLayout(free_layout)

        # ── Wavelength range (bins kept for preview and recording) ──
        self.wavelength_start_spin = _NoWheelUnlessFocusedSpinBox()
        self.wavelength_end_spin = _NoWheelUnlessFocusedSpinBox()
        for spin in (self.wavelength_start_spin, self.wavelength_end_spin):
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setRange(*WAVELENGTH_LIMITS_NM)
            spin.setSuffix(" nm")
        self.wavelength_end_spin.setValue(WAVELENGTH_LIMITS_NM[1])

        wavelength_layout = QFormLayout()
        wavelength_layout.addRow("Start", self.wavelength_start_spin)
        wavelength_layout.addRow("End", self.wavelength_end_spin)

        self.wavelength_range_group = QGroupBox("Wavelength Range")
        self.wavelength_range_group.setCheckable(True)
        self.wavelength_range_group.setChecked(False)
        self.wavelength_range_group.setLayout(wavelength_layout)
        self.wavelength_range_group.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.wavelength_range_group.setToolTip(
            "Record only the wavelengths between Start and End (unchecked: the full calibrated range).\n"
            "Applied from the next frame; locked during a scan.")
        # Usable once the camera has reported its calibrated axis (set_wavelength_range_axis).
        self._wavelength_axis_known = False
        self._recorded_bands = None         # e.g. "17 bands 500.9–599.8 nm", from the engine's axis info
        self._wavelength_range_allowed = True
        self.wavelength_range_group.setEnabled(False)
        self.wavelength_start_spin.editingFinished.connect(
            lambda: self._on_wavelength_spin_finished(self.wavelength_end_spin))
        self.wavelength_end_spin.editingFinished.connect(
            lambda: self._on_wavelength_spin_finished(self.wavelength_start_spin))

        # ── Display controls ─────────────────────────────────────
        self.display_controls = DisplayControls(bit_depth=16)

        # ── Focus ROI ────────────────────────────────────────────
        self.focus_roi_height_spin = QSpinBox()
        self.focus_roi_height_spin.setRange(FOCUS_ROI_MIN_HEIGHT, 100000)
        self.focus_roi_height_spin.setValue(FOCUS_ROI_DEFAULT_HEIGHT)
        self.focus_roi_height_spin.setSuffix(" px")

        self.focus_roi_full_btn = QPushButton("Full Height")

        self.save_frame_btn = QPushButton("Save Frame PNG…")
        self.save_frame_btn.setToolTip(
            "Save the next frame after the active preprocessing (dark/hot-pixel, orientation, "
            "Smile) and before wavelength binning, as an 8/16-bit PNG")

        focus_roi_layout = QHBoxLayout()
        focus_roi_layout.addWidget(QLabel("Focus ROI Height"))
        focus_roi_layout.addWidget(self.focus_roi_height_spin)
        focus_roi_layout.addWidget(self.focus_roi_full_btn)

        # ── Sharpness sidebar ────────────────────────────────────
        self.sharpness_panel = SharpnessPanel()

        # ── Live spectrum ────────────────────────────────────────
        self.live_line_spectrum = LiveLineSpectrumWidget()

        # ── Status dict ──────────────────────────────────────────
        self._status: dict[str, str | int | float] = {
            "state": "PREVIEW",
            "fps": 0.0,
            "frame": 0,
            "recorded": 0,
            "y": 0,
            "sharpness": 0.0,
            "temp": -999.0,
            "camera_fps": None,
            "line_rate": None,           # line_rate_info() of the last scan (set_line_rate)
            "calibration_warning": "",   # set for a scan started with calibration warnings
        }

        # ── LEFT PANEL ───────────────────────────────────────────
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        left_panel.setFixedWidth(350)

        self.viewer.setFixedWidth(300)
        self.viewer.setFixedHeight(240)
        self.sharpness_panel.setFixedHeight(240)
        image_row = QHBoxLayout()
        image_row.setSpacing(0)
        image_row.setContentsMargins(0, 0, 0, 0)
        image_row.addWidget(self.viewer, stretch=1)
        image_row.addWidget(self.sharpness_panel)
        left_layout.addLayout(image_row)

        left_layout.addWidget(self.display_controls)
        left_layout.addLayout(focus_roi_layout)
        left_layout.addWidget(self.save_frame_btn)

        spectrum_label = QLabel("Live Spectrum (horizontal selector)")
        spectrum_label.setStyleSheet("font-size: 11px; color: gray;")
        left_layout.addWidget(spectrum_label)
        left_layout.addWidget(self.live_line_spectrum)

        self.histogram = HistogramPlot(bit_depth=16)
        left_layout.addWidget(self.histogram)
        left_layout.addStretch()

        # ── RIGHT PANEL ──────────────────────────────────────────
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(4, 4, 4, 4)
        controls_layout.setSpacing(6)

        controls_body = QHBoxLayout()
        controls_body.setSpacing(10)
        settings_column = QVBoxLayout()
        settings_column.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.addLayout(self.camera_layout)
        top_row.addWidget(self.dark_panel, stretch=1)
        settings_column.addLayout(top_row)

        stage_row = QHBoxLayout()
        stage_row.addWidget(self.rotation_group)
        stage_row.addWidget(self.linear_group)
        stage_row.addWidget(self.free_group)
        stage_row.addWidget(self.wavelength_range_group)
        settings_column.addLayout(stage_row)

        status_column = QVBoxLayout()
        status_column.setSpacing(6)
        status_column.addWidget(self.correction_status.dark, stretch=1)
        status_column.addWidget(self.correction_status.smile, stretch=1)
        status_column.addWidget(self.correction_status.wavelength, stretch=1)
        controls_body.addLayout(settings_column, stretch=1)
        controls_body.addLayout(status_column)
        controls_layout.addLayout(controls_body)

        controls_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(controls_widget)
        scroll_area.setMinimumWidth(420)
        # Fixed like the left panel's width, so the splitter handle snaps it shut (dragged past
        # half its height) and open again instead of shrinking it step by step.
        scroll_area.setFixedHeight(300)
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._pending_cube = None
        self._pending_wavelengths = None
        self._pending_spectrogram = None
        self._pending_spectrogram_metadata = None
        self._pending_calibrated = None     # float32 reflectance of the pending cube (auto calibration)
        self._pending_calibration_header = None   # its export-header metadata
        self._pending_factors = None        # factors of a scan calibrated frame by frame (counts = R / F)
        self._calibrating_cube = None       # the pending cube while its calibration runs
        self._calibration_token = 0         # counts calibration requests (stale results refused)
        self._result_at_risk = False        # the pending cube was never exported nor knowingly discarded
        self._close_confirmed = False       # confirm_close() already asked for the close under way
        self._cal_win = None  # ★ keep reference
        self._spectro_viewer = None


        # ── Spectrogram Viewer button ────────────────────────
        self.open_spectrogram_btn = QPushButton("Spectrogram Viewer")
        self.open_spectrogram_btn.setEnabled(False)              
        self.open_spectrogram_btn.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32; color: white;
                border-radius: 6px; padding: 6px 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: #43a047; }
            QPushButton:pressed { background-color: #1b5e20; }
            QPushButton:disabled { background-color: #1a2a40; color: #555; }
        """)


        self.cube_preview = RealtimeCubePreviewWidget(
            rgb_target_wavelengths=self.config.rgb_target_wavelengths,
        )

        # ── Cube Viewer + Calibration buttons ────────────────────
        self.open_cube_viewer_btn = QPushButton("Open Cube Viewer")
        self.open_cube_viewer_btn.setEnabled(False)

        self.open_calibration_btn = QPushButton("Open Calibration")
        self.open_calibration_btn.setEnabled(False)
        self.open_calibration_btn.setStyleSheet("""
            QPushButton {
                background-color: #1565c0; color: white;
                border-radius: 6px; padding: 6px 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1e88e5; }
            QPushButton:pressed { background-color: #0d47a1; }
            QPushButton:disabled { background-color: #1a2a40; color: #555; }
        """)

        cube_preview_group = QGroupBox(CUBE_PREVIEW_TITLE)
        self.cube_preview_group = cube_preview_group
        cpg_layout = QVBoxLayout()
        cpg_layout.setContentsMargins(4, 4, 4, 4)
        cpg_layout.setSpacing(4)
        cpg_layout.addWidget(self.cube_preview)

        # reflectance of the last scan (automatic radiometric calibration): shown while there is one
        self.open_reflectance_btn = QPushButton("Open Reflectance Viewer")
        self.open_reflectance_btn.setToolTip(
            "The last scan calibrated with the camera's active radiometric reference. "
            "Open Cube Viewer still shows the raw counts.")
        self.open_reflectance_btn.setStyleSheet("""
            QPushButton {
                background-color: #00897b; color: white;
                border-radius: 6px; padding: 6px 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: #26a69a; }
            QPushButton:pressed { background-color: #00695c; }
            QPushButton:disabled { background-color: #1a2a40; color: #555; }
        """)
        self.open_reflectance_btn.setVisible(False)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.open_cube_viewer_btn)
        btn_row.addWidget(self.open_reflectance_btn)
        btn_row.addWidget(self.open_calibration_btn)
        btn_row.addWidget(self.open_spectrogram_btn)
        cpg_layout.addLayout(btn_row)

        # ── radiometric reference: save this scan as one, and the automatic calibration state ──
        self.save_reference_btn = QPushButton("Save as Radiometric Reference")
        self.save_reference_btn.setEnabled(False)
        self.save_reference_btn.setToolTip(
            "Scan of a Spectralon (or another reference target)? Store it as the radiometric "
            "reference of this camera setting; later scans with the same setting are calibrated "
            "automatically.")
        self.save_reference_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef6c00; color: white;
                border-radius: 6px; padding: 6px 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: #fb8c00; }
            QPushButton:pressed { background-color: #e65100; }
            QPushButton:disabled { background-color: #1a2a40; color: #555; }
        """)
        self.radiometric_status_label = QLabel("Auto radiometric: —")
        self.radiometric_status_label.setWordWrap(True)
        self.set_radiometric_status("off", "Auto radiometric: —")
        self.auto_radiometric_box = QCheckBox("Auto radiometric calibration")
        self.auto_radiometric_box.setChecked(True)
        self.auto_radiometric_box.setToolTip(
            "Ticked when Start is pressed: every frame of the scan is calibrated while it is recorded "
            "with the camera's active radiometric reference, when the camera setting matches it. "
            "The live image and spectrum stay in counts; the counts of the scan can always be had "
            "back exactly (Open Cube Viewer, Calibration, Save as Radiometric Reference).\n"
            "Locked during a scan. Untick before scanning a new reference target to record counts; "
            "tick after a scan to calibrate that scan, untick to drop its reflectance.")
        self.auto_radiometric_box.toggled.connect(self._on_auto_radiometric_toggled)
        ref_row = QHBoxLayout()
        ref_row.addWidget(self.save_reference_btn)
        ref_row.addWidget(self.auto_radiometric_box)
        ref_row.addWidget(self.radiometric_status_label, 1)
        cpg_layout.addLayout(ref_row)
        # what automatic calibration did with the last scan (hidden when there is nothing to say)
        self.scan_calibration_label = QLabel("")
        self.scan_calibration_label.setWordWrap(True)
        self.scan_calibration_label.setVisible(False)
        cpg_layout.addWidget(self.scan_calibration_label)

        cube_preview_group.setLayout(cpg_layout)

        # Settings above, cube preview below: the handle moves, and a scan folds the
        # settings away (see set_scanning).
        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(scroll_area)
        right_panel.addWidget(cube_preview_group)
        right_panel.setCollapsible(0, True)
        right_panel.setCollapsible(1, False)
        right_panel.setStretchFactor(0, 0)
        right_panel.setStretchFactor(1, 1)
        self.settings_splitter = right_panel
        self._settings_sizes_before_scan = None
        self._scan_clock = None             # (monotonic start, expected s) while the stage moves
        self._scanning = False
        self._stage_speed = None            # speed the stage reported it runs, this scan

        # ── SPLITTER ─────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        self._update_stage_controls(self.stage_type())

        # F5 / Esc press Start / Stop Scan (nothing while that button is disabled), only with
        # the focus inside this window, so another module window never starts the stage.
        self.start_scan_shortcut = QShortcut(QKeySequence(Qt.Key_F5), self)
        self.stop_scan_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        for shortcut, button in ((self.start_scan_shortcut, self.viewer.start_btn),
                                 (self.stop_scan_shortcut, self.viewer.stop_btn)):
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(button.click)
        self.viewer.start_btn.setToolTip("Start Scan (F5)")
        self.viewer.stop_btn.setToolTip("Stop Scan (Esc)")

        self.open_cube_viewer_btn.clicked.connect(self._on_open_cube_viewer)
        self.open_reflectance_btn.clicked.connect(self._on_open_reflectance)
        self.open_calibration_btn.clicked.connect(self._on_open_calibration)
        self.save_reference_btn.clicked.connect(self._on_save_reference)
        self.open_spectrogram_btn.clicked.connect(self._on_open_spectrogram)
        self.save_frame_btn.clicked.connect(self.request_processed_frame)
        self.controller = None

    # ── status bar interface ─────────────────────────────────────

    def status_info(self) -> str:
        s = self._status
        cam_fps = "" if s["camera_fps"] is None else f" (cam {s['camera_fps']:.1f})"
        scan_time = ""
        if self._scan_clock is not None:
            started, expected = self._scan_clock
            elapsed = time.monotonic() - started
            if self.stage_type() == "free":     # an exact time, not a stage move estimate
                scan_time = (f"Scan {_format_duration(elapsed)} / {_format_duration(expected)}"
                             f" ({_format_countdown(expected - elapsed)} left)  ·  ")
            else:
                scan_time = (f"Scan {_format_duration(elapsed)}"
                             f" / ~{_format_duration(expected)}  ·  ")
        warning = s.get("calibration_warning") or ""
        line_rate = s.get("line_rate")
        line = "" if not line_rate else f"Line rate {line_rate['line_rate_hz']:.1f} Hz  ·  "
        return (                     # the scan time before the (long) warnings, so it stays in view
            f"{s['state']}  ·  "
            + f"{scan_time}"
            + (f"{warning}  ·  " if warning else "")
            + f"FPS {s['fps']:.1f}"
            f"{cam_fps}  ·  "
            f"Frame {s['frame']:,}  ·  "
            f"Recorded {s['recorded']:,}  ·  "
            + line                   # the last scan's measured line rate (set_line_rate)
            + f"Y {s['y']}  ·  "
            f"Sharpness {s['sharpness']:.1f}"
        )

    def status_line_rate_warning(self) -> str:
        """Why the measured line rate of the last scan needs attention ("" when it does not, or
        is unknown): frames missing inside the recording, or a line rate below
        ``LINE_RATE_WARN_FRACTION`` of the camera's own frame rate (lines then cover more of the
        scene than the frame rate says: the image is squeezed along the scan)."""
        info = self._status.get("line_rate")
        if not info:
            return ""
        parts = []
        camera_hz = info.get("camera_hz")
        if (camera_hz and info.get("frames", 0) >= LINE_RATE_WARN_MIN_FRAMES
                and info["line_rate_hz"] < LINE_RATE_WARN_FRACTION * camera_hz):
            parts.append(f"Line rate {100.0 * info['line_rate_hz'] / camera_hz:.0f}% of camera "
                         f"({info['line_rate_hz']:.1f} / {camera_hz:.1f} Hz)")
        missing = info.get("frames_missing") or 0
        if missing > 0:
            parts.append(f"{missing:,} frames missing")
        return "  ·  ".join(parts)

    def status_temperature(self):
        """Sensor temperature (°C) for the status bar's thermometer, or ``None`` when unknown."""
        temp = self._status["temp"]
        return temp if temp > -900 else None

    # ── cube ready ───────────────────────────────────────────────

    def set_cube_ready(self, cube, wavelengths, calibration=None, note=None):
        """The scan's cube is ready.

        Args:
            cube: ``(lines, samples, bands)`` in counts, or its reflectance when the frames were
                calibrated while recorded (``calibration`` given).
            wavelengths: Band centres (nm).
            calibration: For a reflectance ``cube``: ``{"header": export-header fields,
                "factors": (lines, bands) factors, "text": line for the scan}``; the counts are
                rebuilt from it when asked for (:meth:`counts_cube`).
            note: For a counts ``cube``: ``(state, text)`` saying why it is not calibrated (e.g. not
                enough memory for its reflectance); it is shown and no calibration after the scan
                is tried. ``None``: calibrated after the scan when *Auto radiometric calibration*
                is ticked and a reference fits.
        """
        self.clear_scan_calibration()               # belonged to the cube being replaced
        if calibration is not None:
            self._pending_cube = None               # counts only when asked for (exact: R / F)
            self._pending_calibrated = cube
            self._pending_calibration_header = dict(calibration.get("header") or {})
            self._pending_factors = calibration.get("factors")
        else:
            self._pending_cube = cube
        self._pending_wavelengths = wavelengths
        self._result_at_risk = True                 # in RAM only until it is exported

        shape_str = f"{cube.shape[0]}x{cube.shape[1]}x{cube.shape[2]}"

        self.open_cube_viewer_btn.setEnabled(True)
        self.open_cube_viewer_btn.setText(f"Open Cube Viewer  ({shape_str})")

        self.open_calibration_btn.setEnabled(True)
        self.open_calibration_btn.setText(f"Calibration  ({shape_str})")
        self.save_reference_btn.setEnabled(True)
        if calibration is not None:
            self.open_reflectance_btn.setText(f"Open Reflectance Viewer  ({shape_str})")
            self.open_reflectance_btn.setVisible(True)
            self.set_scan_calibration("applied", calibration.get("text") or "Last scan: calibrated")
        elif note is not None:
            self.set_scan_calibration(*note)
        else:
            self.request_auto_calibration()

    def counts_cube(self):
        """The last scan in counts (Open Cube Viewer, Calibration, Save as Radiometric Reference).

        A scan calibrated while recorded holds only its reflectance: its counts are rebuilt from
        it the first time they are asked for (``counts_from_reflectance``: exact, since the factors
        are finite and positive) and kept from then on.
        """
        if (self._pending_cube is None and self._pending_calibrated is not None
                and self._pending_factors is not None):
            from modules.radiometric.auto import counts_from_reflectance
            self._pending_cube = counts_from_reflectance(self._pending_calibrated, self._pending_factors)
        return self._pending_cube

    def _on_open_cube_viewer(self):
        cube = self.counts_cube()
        if cube is None:
            return
        self.cube_ready.emit(cube, self._pending_wavelengths)

    def has_unsaved_result(self) -> bool:
        """True while the last scan's cube exists only in RAM (never exported, not discarded)."""
        return (self._pending_cube is not None or self._pending_calibrated is not None) and self._result_at_risk

    def mark_result_saved(self, cube):
        """``cube`` was exported; if it is the pending result (its raw counts or its reflectance),
        losing it needs no question."""
        if cube is not None and (cube is self._pending_cube or cube is self._pending_calibrated):
            self._result_at_risk = False

    # ── automatic radiometric calibration of the last scan ──────────

    def request_auto_calibration(self) -> bool:
        """Ask for the pending cube's calibration (``auto_calibration_requested``) when *Auto
        radiometric calibration* is ticked and the cube has no reflectance yet, nor one being made.
        Returns True when asked."""
        cube = self._pending_cube
        if (cube is None or not self.auto_radiometric_box.isChecked()
                or self._pending_calibrated is not None or self._calibrating_cube is cube):
            return False
        self._calibrating_cube = cube
        self._calibration_token += 1                # a result of an earlier request is refused
        self.auto_calibration_requested.emit(cube, self._pending_wavelengths)
        return True

    def calibration_token(self) -> int:
        """Identifies the current calibration request: a result carrying another token (asked
        before the reflectance was dropped, or for another reference) is refused."""
        return self._calibration_token

    def recalibrate(self) -> bool:
        """Calibrate the pending scan again from its counts (e.g. with a reference just made
        active): its current reflectance is dropped first. Returns True when asked."""
        if self._pending_cube is None and self._pending_calibrated is None:
            return False
        self.counts_cube()                          # a scan held only as reflectance: counts first
        self.clear_scan_calibration()
        return self.request_auto_calibration()

    def _on_auto_radiometric_toggled(self, checked):
        if checked:
            self.request_auto_calibration()
        else:
            # this scan goes back to counts: its reflectance (made or being made) is dropped,
            # after its counts are rebuilt when it was calibrated while recorded
            self.counts_cube()
            self.clear_scan_calibration()

    def clear_scan_calibration(self):
        """Forget the reflectance of the pending scan (new scan, or auto calibration unticked);
        a scan held only as reflectance must have its counts made first (:meth:`counts_cube`)."""
        self._pending_calibrated = None
        self._pending_calibration_header = None
        self._pending_factors = None
        self._calibrating_cube = None
        self._calibration_token += 1                # results still on their way are refused
        self.open_reflectance_btn.setVisible(False)
        self.open_reflectance_btn.setText("Open Reflectance Viewer")
        self.set_scan_calibration("off", "")

    def set_scan_calibration(self, state: str, text: str):
        """Show what automatic calibration does with the last scan: ``state`` ``"applied"``
        (green), ``"working"`` (blue), ``"not_applied"`` (amber), ``"failed"`` (red), or ``"off"``
        (the line is hidden)."""
        color = {"applied": "#43a047", "working": "#42a5f5", "not_applied": "#fb8c00",
                 "failed": "#e53935"}.get(state)
        if state in ("not_applied", "failed", "off"):
            self._calibrating_cube = None
        self.scan_calibration_label.setText(text if color else "")
        self.scan_calibration_label.setStyleSheet(f"color: {color or '#7a8699'};")
        self.scan_calibration_label.setVisible(bool(color and text))

    def _current_request(self, cube, token) -> bool:
        return (cube is not None and cube is self._pending_cube and self._calibrating_cube is cube
                and (token is None or token == self._calibration_token))

    def set_calibrated_cube(self, cube, calibrated, header: dict, text: str, token=None) -> bool:
        """The reflectance ``calibrated`` of ``cube`` is ready, with its export-header metadata.

        Refused (False) when ``cube`` is no longer the pending scan (a newer scan replaced it),
        auto calibration was unticked meanwhile, or ``token`` (:meth:`calibration_token` when it
        was asked) is not the current request's: the result belongs to nothing shown any more.
        """
        if not self._current_request(cube, token):
            return False
        self._calibrating_cube = None
        self._pending_calibrated = calibrated
        self._pending_calibration_header = dict(header or {})
        shape = "x".join(str(n) for n in calibrated.shape)
        self.open_reflectance_btn.setText(f"Open Reflectance Viewer  ({shape})")
        self.open_reflectance_btn.setVisible(True)
        self.set_scan_calibration("applied", text)
        return True

    def calibration_failed(self, cube, text: str, token=None) -> bool:
        """Calibrating ``cube`` failed: say so, if it is still the current request. Returns True when shown."""
        if not self._current_request(cube, token):
            return False
        self.set_scan_calibration("failed", text)
        return True

    def calibrated_cube(self):
        """The reflectance of the pending scan, or ``None``."""
        return self._pending_calibrated

    def calibration_header(self, cube):
        """Export-header metadata of ``cube`` if it is the pending scan's reflectance, else ``None``."""
        if cube is not None and cube is self._pending_calibrated:
            return dict(self._pending_calibration_header or {})
        return None

    def _on_open_reflectance(self):
        if self._pending_calibrated is None:
            return
        self.cube_ready.emit(self._pending_calibrated, self._pending_wavelengths)

    def confirm_discard_result(self, action: str) -> bool:
        """Ask before ``action`` drops a never-exported result: Save / Discard / Cancel.

        Returns True to go on (nothing unsaved, or Discard).  Save opens the cube with its
        export dialog (``save_result_requested``) and returns False like Cancel: the action
        waits until the user has exported and tries again.
        """
        if not self.has_unsaved_result():
            return True
        held = self._pending_cube if self._pending_cube is not None else self._pending_calibrated
        shape = "x".join(str(n) for n in held.shape)
        what = (f"cube ({shape}) and its raw spectrogram have" if self._pending_spectrogram is not None
                else f"cube ({shape}) has")
        calibrated = self._pending_calibrated
        which = ("the export dialog of its reflectance cube (Open Cube Viewer exports the raw counts)"
                 if calibrated is not None else "the export dialog")
        answer = QMessageBox.warning(
            self, "Unsaved Scan",
            f"The last scan's {what} not been saved.\n"
            f"{action} discards it.\n\n"
            f"Save opens {which}; try again once the export has finished.",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if answer == QMessageBox.Discard:
            self._result_at_risk = False
            return True
        if answer == QMessageBox.Save:
            self.save_result_requested.emit(calibrated if calibrated is not None else self._pending_cube,
                                            self._pending_wavelengths)
        return False

    def confirm_close(self, action: str = "Closing the scan window") -> bool:
        """Ask what happens to the scan data before the window closes; True to close.

        A running scan is stopped first (Yes): its frames so far become the cube, which is
        then covered by :meth:`confirm_discard_result`.  The next ``closeEvent`` does not ask again.
        """
        controller = self.controller
        # After controller.cleanup() the camera is stopped: nothing can be built any more.
        if (controller is not None and controller.state == controller.SCANNING
                and controller.preview_thread is not None):
            answer = QMessageBox.question(
                self, "Scan Running",
                f"A scan is running. Stop it?\n{action} needs the scan stopped; the frames "
                "recorded so far are kept as a cube you can save.",
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
            if answer != QMessageBox.Yes:
                return False
            controller.stop_scan_manually()
        self._close_confirmed = self.confirm_discard_result(action)
        return self._close_confirmed

    # ── spectrogram ─────────────────────────────────────────────

    def set_spectrogram_ready(self, spectrogram, metadata=None):
        """Called after scan completes with keep_spectrogram=True.

        ``metadata`` (bin, exposure, gain, raw_inverted, m/c, ...) is stored with the data
        when the spectrogram is saved to HDF5.
        """
        self._pending_spectrogram = spectrogram
        self._pending_spectrogram_metadata = metadata
        n, h, w = spectrogram.shape
        self.open_spectrogram_btn.setEnabled(True)
        self.open_spectrogram_btn.setText(
            f"Spectrogram Viewer  ({n}×{h}×{w})"
        )

    def spectrogram_data(self):
        """Public getter — for MainWindow menu access."""
        return self._pending_spectrogram

    def _on_open_spectrogram(self):
        if self._pending_spectrogram is None:                  
            return
        self._spectro_viewer = SpectrogramViewer()
        self._spectro_viewer.load(self._pending_spectrogram, self._pending_spectrogram_metadata)
        self._spectro_viewer.show()

    # ── calibration ──────────────────────────────────────────────

    def _on_open_calibration(self):
        cube = self.counts_cube()
        if cube is None:
            return
        self.calibration_requested.emit(cube, self._pending_wavelengths)

    def _on_save_reference(self):
        cube = self.counts_cube()                  # a reference is fitted on counts, never reflectance
        if cube is None:
            return
        self.radiometric_reference_requested.emit(cube, self._pending_wavelengths)

    def set_radiometric_status(self, state: str, text: str):
        """Show the automatic radiometric calibration state: ``state`` ``"on"`` (green),
        ``"expired"`` (amber) or anything else (grey), with ``text``."""
        color = {"on": "#43a047", "expired": "#fb8c00"}.get(state, "#7a8699")
        self.radiometric_status_label.setText(text)
        self.radiometric_status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    # ── stage controls ───────────────────────────────────────────

    def _update_stage_controls(self, stage_type: str):
        stage_type = str(stage_type).lower()
        self.rotation_group.setVisible(stage_type == "rotation")
        self.linear_group.setVisible(stage_type == "linear")
        self.free_group.setVisible(stage_type == "free")

    def stage_type(self) -> str:
        return str(self.config.stage_type).lower()

    def rotation_speed_dps(self) -> float:
        return float(self.degree_per_second_spin.value())

    def rotation_degree(self) -> float:
        return float(self.degree_spin.value())

    def linear_distance_mm(self) -> float:
        return float(self.distance_mm_spin.value())

    def linear_speed_um_s(self) -> float:
        return float(self.speed_um_s_spin.value())

    def free_duration_s(self) -> float:
        """Recording time of a free scan (stage type ``"free"``, no stage), in seconds."""
        return float(self.free_duration_spin.value())

    # ── bin ──────────────────────────────────────────────────────

    def set_bin_options(self, bins: tuple[int, ...], default_bin: int):
        self.camera_controls.set_bin_options(bins, default_bin)


    def selected_bin(self):
        return self.camera_controls.selected_bin()

    def set_bin_enabled(self, enabled):
        self.camera_controls.set_bin_enabled(enabled)

    # ── scanning ─────────────────────────────────────────────────

    def set_scanning(self, active: bool):
        """Lock the camera and stage settings during a scan and fold them away.

        Starting folds the settings pane so the cube preview gets its height; it can still
        be dragged open to read the (locked) values, and the preview title shows them in one
        line (:meth:`scan_settings_summary`). Ending restores the pane height from before the
        scan and the plain title. The bin, wavelength range and dark panel are locked separately.

        Args:
            active: True when a scan starts, False when it ends (or fails).
        """
        self.camera_controls.set_settings_enabled(not active)
        self.rotation_group.setEnabled(not active)
        self.linear_group.setEnabled(not active)
        self.free_duration_spin.setEnabled(not active)   # the box stays lit for its countdown
        # chosen at Start for the whole scan (frames are calibrated while recorded)
        self.auto_radiometric_box.setEnabled(not active)
        splitter = self.settings_splitter
        if active and self._settings_sizes_before_scan is None:
            sizes = splitter.sizes()
            self._settings_sizes_before_scan = sizes
            splitter.setSizes([0, sum(sizes)])
        elif not active and self._settings_sizes_before_scan is not None:
            splitter.setSizes(self._settings_sizes_before_scan)
            self._settings_sizes_before_scan = None
        self._scanning = bool(active)
        if not active:
            self._scan_clock = None
            self._stage_speed = None
            self._countdown_timer.stop()
        self._update_scan_title()
        self._update_countdown()

    def _update_scan_title(self):
        self.cube_preview_group.setTitle(
            f"{CUBE_PREVIEW_TITLE}  —  {self.scan_settings_summary()}" if self._scanning else CUBE_PREVIEW_TITLE)

    def scan_settings_summary(self) -> str:
        """The settings a scan uses in one line: bin, exposure, gain, recorded bands, stage move
        (or the scan time of a free scan, and its time left once it records).

        Once the stage reported a speed different from the entered one
        (:meth:`start_scan_clock`), the move adds ``(stage runs ...)``.
        """
        controls = self.camera_controls
        parts = [f"Bin {controls.selected_bin()}", f"{controls.exposure_ms():.3f} ms", f"Gain {controls.gain()}"]
        if self._recorded_bands is not None:
            parts.append(self._recorded_bands)
        if self.stage_type() == "free":
            remaining = self.free_scan_remaining()
            parts.append(f"{self.free_duration_s():.2f} s free scan (no stage)"
                         + ("" if remaining is None else f", {_format_countdown(remaining)} left"))
        elif self.stage_type() == "linear":
            speed = self.linear_speed_um_s()
            parts.append(f"{self.linear_distance_mm():.3f} mm @ {speed:g} µm/s"
                         f"{self._stage_runs(speed, '{:.3f} µm/s')}")
        else:
            speed = self.rotation_speed_dps()
            parts.append(f"{self.rotation_degree():.3f}° @ {speed:.3f}°/s"
                         f"{self._stage_runs(speed, '{:.3f}°/s')}")
        return "  ·  ".join(parts)

    def _stage_runs(self, entered: float, form: str) -> str:
        """`` (stage runs <speed>)`` when the stage reported a speed other than ``entered``."""
        speed = self._stage_speed
        if speed is None or math.isclose(speed, entered, rel_tol=1e-9, abs_tol=1e-9):
            return ""
        return f" (stage runs {form.format(speed)})"

    def expected_scan_seconds(self, speed=None) -> float:
        """Duration of the scan move in s (stage acceleration ignored); a free scan's time.

        Args:
            speed: Speed the stage runs (µm/s linear, deg/s rotation), or ``None`` for the
                entered speed (not used by a free scan).
        """
        if self.stage_type() == "free":
            return self.free_duration_s()
        if self.stage_type() == "linear":
            speed = self.linear_speed_um_s() if speed is None else speed
            return self.linear_distance_mm() / (speed / 1000.0)
        speed = self.rotation_speed_dps() if speed is None else speed
        return self.rotation_degree() / speed

    def start_scan_clock(self, speed=None):
        """Show ``Scan elapsed / ~expected`` in the status line from now (the stage move start)
        until the scan ends (:meth:`set_scanning` ``False``); a free scan shows
        ``Scan elapsed / time (left)`` and counts down in its box.

        Args:
            speed: Speed the stage reported it runs, read back from its controller (µm/s
                linear, deg/s rotation), or ``None`` for the entered speed. The expected time
                uses it, and the preview title shows it when it differs from the entered one.
        """
        self._stage_speed = speed
        self._scan_clock = (time.monotonic(), self.expected_scan_seconds(speed))
        if self._scanning:
            self._update_scan_title()
        if self.stage_type() == "free":
            self._countdown_timer.start()
            self._update_countdown()

    def free_scan_remaining(self):
        """Seconds left in the running free scan (0 once its time is up), or ``None`` when no
        free scan is recording (not scanning, a stage scan, or not started yet)."""
        if self._scan_clock is None or self.stage_type() != "free":
            return None
        started, total = self._scan_clock
        return max(0.0, total - (time.monotonic() - started))

    def _update_countdown(self):
        """Redraw the Free Scan box's countdown: the time left while recording, the whole scan
        time between Start and the recording start, ``—`` when no free scan runs."""
        remaining = self.free_scan_remaining()
        if remaining is not None:
            text = _format_countdown(remaining)
        elif self._scanning and self.stage_type() == "free":
            text = _format_countdown(self.free_duration_s())
        else:
            text = COUNTDOWN_IDLE
        if text == self.free_countdown_label.text():
            return                          # most ticks: the same tenth, nothing to redraw
        self.free_countdown_label.setText(text)
        if remaining is not None and self._scanning:
            self._update_scan_title()       # the preview title shows the time left too
    # ── wavelength range ─────────────────────────────────────────

    def requested_wavelength_range(self):
        """``(start, end)`` nm as entered, or ``None`` when the range is off (full range)."""
        if not self.wavelength_range_group.isChecked():
            return None
        return (round(self.wavelength_start_spin.value(), 1), round(self.wavelength_end_spin.value(), 1))

    def wavelength_range_edits(self):
        """The signals emitted when the user commits a range edit (connect them all)."""
        return (self.wavelength_range_group.toggled, self.wavelength_range_committed)

    def _on_wavelength_spin_finished(self, partner):
        """Commit, unless focus just moved to the other spin box (Start then End is one edit)."""
        if QApplication.focusWidget() is not partner:
            self.wavelength_range_committed.emit()

    def set_wavelength_range(self, wavelength_range):
        """Show ``(start, end)`` nm (checked) or ``None`` (unchecked) without emitting edits."""
        widgets = (self.wavelength_range_group, self.wavelength_start_spin, self.wavelength_end_spin)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.wavelength_range_group.setChecked(wavelength_range is not None)
            if wavelength_range is not None:
                lo, hi = (float(v) for v in wavelength_range)
                for spin, value in ((self.wavelength_start_spin, lo), (self.wavelength_end_spin, hi)):
                    spin.setRange(min(spin.minimum(), value), max(spin.maximum(), value))
                    spin.setValue(value)
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def set_wavelength_range_axis(self, info: dict):
        """Limits and tooltip from the engine's ``wavelength_range_applied`` info.

        The limits are the calibrated span rounded outward to 0.1 nm (widened to keep the
        values shown); with the range off the spin boxes show that span, so checking the
        box starts from the full range. A spin box being edited is left alone. The first
        call enables the control.
        """
        lo = math.floor(float(info["calibrated_nm"][0]) * 10.0) / 10.0
        hi = math.ceil(float(info["calibrated_nm"][1]) * 10.0) / 10.0
        widgets = (self.wavelength_start_spin, self.wavelength_end_spin)
        for spin in widgets:
            spin.blockSignals(True)
        try:
            off = not self.wavelength_range_group.isChecked()
            for spin, edge in zip(widgets, (lo, hi)):
                if off:
                    spin.setRange(lo, hi)
                    spin.setValue(edge)
                elif not spin.hasFocus():
                    spin.setRange(min(lo, spin.value()), max(hi, spin.value()))
        finally:
            for spin in widgets:
                spin.blockSignals(False)
        self._wavelength_axis_known = True
        self.wavelength_range_group.setEnabled(self._wavelength_range_allowed)
        first, last = info["centres_nm"]
        if info.get("bands") is None:
            kept = f"all {info['full_bands']} bands, {first:.1f}–{last:.1f} nm"
        else:
            kept = f"{info['bands'][1] - info['bands'][0]} of {info['full_bands']} bands, {first:.1f}–{last:.1f} nm"
        n_bands = info['full_bands'] if info.get("bands") is None else info['bands'][1] - info['bands'][0]
        self._recorded_bands = f"{n_bands} bands {first:.1f}–{last:.1f} nm"
        self.wavelength_range_group.setToolTip(
            f"Recording {kept} (band centres; step {info['step_nm']:.2f} nm)\n"
            f"Calibrated range {info['calibrated_nm'][0]:.1f}–{info['calibrated_nm'][1]:.1f} nm.\n"
            "Unchecked: the full calibrated range. Applied from the next frame; locked during a scan.")

    def set_wavelength_range_enabled(self, enabled: bool):
        """Allow or block range edits (blocked during a scan, like the bin, and until the
        camera has reported its calibrated axis)."""
        self._wavelength_range_allowed = bool(enabled)
        self.wavelength_range_group.setEnabled(self._wavelength_range_allowed and self._wavelength_axis_known)

    # ── focus ROI ────────────────────────────────────────────────

    def focus_roi_height(self) -> int:
        return int(self.focus_roi_height_spin.value())

    def set_focus_roi_height_value(self, value: int):
        self.focus_roi_height_spin.blockSignals(True)
        self.focus_roi_height_spin.setValue(int(value))
        self.focus_roi_height_spin.blockSignals(False)

    # ── status setters ───────────────────────────────────────────

    def set_status(self, text: str):
        self._status["state"] = text
        self.sharpness_panel.set_scanning(text == "SCANNING")

    def set_calibration_warning(self, text: str):
        """Calibration warnings of the running scan, shown after the state ("" for none)."""
        self._status["calibration_warning"] = text

    def set_fps(self, fps: float):
        self._status["fps"] = fps

    def set_camera_frame_rate(self, hz: float):
        """Camera AcquisitionFrameRate (Hz) shown next to the measured FPS."""
        self._status["camera_fps"] = float(hz)

    def set_frame_index(self, index: int):
        self._status["frame"] = index

    def set_recorded_count(self, n: int):
        self._status["recorded"] = n

    def set_line_rate(self, info):
        """Measured line rate of the scan that just ended (the acquisition thread's
        ``line_rate_info()``), shown after the recorded count; ``None`` / ``{}`` clears it
        (a new scan, or a scan too short to measure)."""
        self._status["line_rate"] = dict(info) if info and info.get("line_rate_hz") else None

    def set_selected_y_label(self, y: int):
        self._status["y"] = y

    def set_sharpness_score(self, score: float):
        self._status["sharpness"] = score
        self.sharpness_panel.set_value(score)

    def reset_sharpness_scale(self):
        """Forget the sharpness peak (the scores of a new gain/exposure/bin are not comparable)."""
        self.sharpness_panel.reset()

    def set_temperature(self, temp_c: float):
        self._status["temp"] = temp_c
        self.camera_controls.show_sensor_temperature(temp_c)
        self.dark_panel.set_current_temperature(temp_c)

    def stage_spins(self):
        """The five scan spin boxes (rotation speed/degree, linear distance/speed, free scan
        time), in the order of ``STAGE_SETTING_KEYS``."""
        return (self.degree_per_second_spin, self.degree_spin,
                self.distance_mm_spin, self.speed_um_s_spin, self.free_duration_spin)

    # ── lifecycle ────────────────────────────────────────────────

    def closeEvent(self, event):
        confirmed, self._close_confirmed = self._close_confirmed, False
        if not confirmed and not self.confirm_close():
            event.ignore()
            return
        if self.controller is not None and self.controller.cleanup() is False:
            event.ignore()
            return
        self.cube_preview.cleanup()
        if self._cal_win is not None:
            self._cal_win.close()
        if self._spectro_viewer is not None:                    
            self._spectro_viewer.close()
        super().closeEvent(event)