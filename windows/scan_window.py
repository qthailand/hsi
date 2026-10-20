# windows/scan_window.py
from __future__ import annotations

import math
import time

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
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


def _format_duration(seconds: float) -> str:
    """``12.3 s`` below a minute, else ``m:ss``."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


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

        # ── Stage — rotation ─────────────────────────────────────
        self.degree_per_second_spin = QDoubleSpinBox()
        self.degree_per_second_spin.setDecimals(3)
        self.degree_per_second_spin.setRange(0.001, 100.0)
        self.degree_per_second_spin.setValue(float(getattr(config, "rotation_speed_dps", 1.0)))
        self.degree_per_second_spin.setSuffix(" deg/s")

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
        self.speed_um_s_spin.setRange(0.001, 1000000.0)
        self.speed_um_s_spin.setValue(float(getattr(config, "linear_speed_um_s", 1000.0)))
        self.speed_um_s_spin.setSuffix(" um/s")

        linear_layout = QFormLayout()
        linear_layout.addRow("Linear Distance", self.distance_mm_spin)
        linear_layout.addRow("Linear Speed", self.speed_um_s_spin)

        self.linear_group = QGroupBox("Linear Stage")
        self.linear_group.setLayout(linear_layout)

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
        stage_row.addWidget(self.wavelength_range_group)
        settings_column.addLayout(stage_row)

        status_column = QVBoxLayout()
        status_column.setSpacing(6)
        status_column.addWidget(self.correction_status.dark, stretch=1)
        status_column.addWidget(self.correction_status.smile, stretch=1)
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

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.open_cube_viewer_btn)
        btn_row.addWidget(self.open_calibration_btn)
        btn_row.addWidget(self.open_spectrogram_btn)
        cpg_layout.addLayout(btn_row)

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
        self.open_calibration_btn.clicked.connect(self._on_open_calibration)
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
            scan_time = (f"Scan {_format_duration(time.monotonic() - started)}"
                         f" / ~{_format_duration(expected)}  ·  ")
        temp_str = ""
        if s["temp"] > -900:
            temp_str = f"  ·  \U0001f321 {s['temp']:.1f}\u00b0C"
        return (
            f"{s['state']}  ·  "
            f"{scan_time}"
            f"FPS {s['fps']:.1f}"
            f"{cam_fps}  ·  "
            f"Frame {s['frame']:,}  ·  "
            f"Recorded {s['recorded']:,}  ·  "
            f"Y {s['y']}  ·  "
            f"Sharpness {s['sharpness']:.1f}"
            f"{temp_str}"
        )

    # ── cube ready ───────────────────────────────────────────────

    def set_cube_ready(self, cube, wavelengths):
        self._pending_cube = cube
        self._pending_wavelengths = wavelengths

        shape_str = f"{cube.shape[0]}x{cube.shape[1]}x{cube.shape[2]}"

        self.open_cube_viewer_btn.setEnabled(True)
        self.open_cube_viewer_btn.setText(f"Open Cube Viewer  ({shape_str})")

        self.open_calibration_btn.setEnabled(True)
        self.open_calibration_btn.setText(f"Calibration  ({shape_str})")

    def _on_open_cube_viewer(self):
        if self._pending_cube is None:
            return
        self.cube_ready.emit(self._pending_cube, self._pending_wavelengths)

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
        if self._pending_cube is None:
            return
        self.calibration_requested.emit(self._pending_cube, self._pending_wavelengths)

    # ── stage controls ───────────────────────────────────────────

    def _update_stage_controls(self, stage_type: str):
        stage_type = str(stage_type).lower()
        self.rotation_group.setVisible(stage_type == "rotation")
        self.linear_group.setVisible(stage_type == "linear")

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
        splitter = self.settings_splitter
        if active and self._settings_sizes_before_scan is None:
            sizes = splitter.sizes()
            self._settings_sizes_before_scan = sizes
            splitter.setSizes([0, sum(sizes)])
        elif not active and self._settings_sizes_before_scan is not None:
            splitter.setSizes(self._settings_sizes_before_scan)
            self._settings_sizes_before_scan = None
        if not active:
            self._scan_clock = None
        self.cube_preview_group.setTitle(
            f"{CUBE_PREVIEW_TITLE}  —  {self.scan_settings_summary()}" if active else CUBE_PREVIEW_TITLE)

    def scan_settings_summary(self) -> str:
        """The settings a scan uses in one line: bin, exposure, gain, recorded bands, stage move."""
        controls = self.camera_controls
        parts = [f"Bin {controls.selected_bin()}", f"{controls.exposure_ms():.3f} ms", f"Gain {controls.gain()}"]
        if self._recorded_bands is not None:
            parts.append(self._recorded_bands)
        if self.stage_type() == "linear":
            parts.append(f"{self.linear_distance_mm():.3f} mm @ {self.linear_speed_um_s():g} µm/s")
        else:
            parts.append(f"{self.rotation_degree():.3f}° @ {self.rotation_speed_dps():.3f}°/s")
        return "  ·  ".join(parts)

    def expected_scan_seconds(self) -> float:
        """Duration of the scan move from the stage settings, in s (stage acceleration ignored)."""
        if self.stage_type() == "linear":
            return self.linear_distance_mm() / (self.linear_speed_um_s() / 1000.0)
        return self.rotation_degree() / self.rotation_speed_dps()

    def start_scan_clock(self):
        """Show ``Scan elapsed / ~expected`` in the status line from now (the stage move start)
        until the scan ends (:meth:`set_scanning` ``False``)."""
        self._scan_clock = (time.monotonic(), self.expected_scan_seconds())

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

    def set_fps(self, fps: float):
        self._status["fps"] = fps

    def set_camera_frame_rate(self, hz: float):
        """Camera AcquisitionFrameRate (Hz) shown next to the measured FPS."""
        self._status["camera_fps"] = float(hz)

    def set_frame_index(self, index: int):
        self._status["frame"] = index

    def set_recorded_count(self, n: int):
        self._status["recorded"] = n

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
        """The four stage scan spin boxes (rotation speed/degree, linear distance/speed)."""
        return (self.degree_per_second_spin, self.degree_spin,
                self.distance_mm_spin, self.speed_um_s_spin)

    # ── lifecycle ────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.controller is not None and self.controller.cleanup() is False:
            event.ignore()
            return
        self.cube_preview.cleanup()
        if self._cal_win is not None:
            self._cal_win.close()
        if self._spectro_viewer is not None:                    
            self._spectro_viewer.close()
        super().closeEvent(event)