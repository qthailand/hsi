# windows/scan_window.py
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
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


class ScanWindow(QWidget):
    snapshot_ready = pyqtSignal(object)
    snapshot_failed = pyqtSignal(int, str)
    cube_ready = pyqtSignal(object, object)
    calibration_requested = pyqtSignal(object, object)

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
        scroll_area.setMaximumHeight(300)
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

        cube_preview_group = QGroupBox("Realtime Cube Preview")
        cpg_layout = QVBoxLayout()
        cpg_layout.setContentsMargins(4, 4, 4, 4)
        cpg_layout.setSpacing(4)
        cpg_layout.addWidget(self.cube_preview)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.open_cube_viewer_btn)
        btn_row.addWidget(self.open_calibration_btn)
        cpg_layout.addLayout(btn_row)

        spectro_row = QHBoxLayout()                              
        spectro_row.addWidget(self.open_spectrogram_btn)         
        cpg_layout.addLayout(spectro_row)      

        cube_preview_group.setLayout(cpg_layout)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        right_layout.addWidget(scroll_area)
        right_layout.addWidget(cube_preview_group, stretch=1)

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
        self.open_cube_viewer_btn.clicked.connect(self._on_open_cube_viewer)
        self.open_calibration_btn.clicked.connect(self._on_open_calibration)
        self.open_spectrogram_btn.clicked.connect(self._on_open_spectrogram)
        self.save_frame_btn.clicked.connect(self.request_processed_frame)
        self.controller = None

    # ── status bar interface ─────────────────────────────────────

    def status_info(self) -> str:
        s = self._status
        cam_fps = "" if s["camera_fps"] is None else f" (cam {s['camera_fps']:.1f})"
        temp_str = ""
        if s["temp"] > -900:
            temp_str = f"  ·  \U0001f321 {s['temp']:.1f}\u00b0C"
        return (
            f"{s['state']}  ·  "
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