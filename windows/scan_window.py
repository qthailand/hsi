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
)

from config import AppConfig
from utils.display import DisplayMapper
from widgets.live_image_canvas import (
    FOCUS_ROI_DEFAULT_HEIGHT,
    FOCUS_ROI_MIN_HEIGHT,
)
from widgets.camera_controls import CameraControls
from widgets.spectrum import LiveLineSpectrumWidget
from widgets.cube_preview import RealtimeCubePreviewWidget
from widgets.dark_frame_panel import DarkFramePanel
from widgets.display_controls import DisplayControls
from widgets.sharpness_panel import SharpnessPanel
from widgets.histogram_plot import HistogramPlot
from windows.image_viewer import ImageViewer
from windows.spectrogram_viewer import SpectrogramViewer
from modules.bsq_calibration import CalibrationWindow


class ScanWindow(QWidget):
    cube_ready = pyqtSignal(object, object)

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

        # ── Stage — rotation ─────────────────────────────────────
        self.degree_per_second_spin = QDoubleSpinBox()
        self.degree_per_second_spin.setDecimals(3)
        self.degree_per_second_spin.setRange(0.001, 100.0)
        self.degree_per_second_spin.setValue(1.0)
        self.degree_per_second_spin.setSuffix(" deg/s")

        self.degree_spin = QDoubleSpinBox()
        self.degree_spin.setDecimals(3)
        self.degree_spin.setRange(0.001, 360000.0)
        self.degree_spin.setValue(10.0)
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
        self.distance_mm_spin.setValue(10.0)
        self.distance_mm_spin.setSuffix(" mm")

        self.speed_um_s_spin = QDoubleSpinBox()
        self.speed_um_s_spin.setDecimals(3)
        self.speed_um_s_spin.setRange(0.001, 1000000.0)
        self.speed_um_s_spin.setValue(1000.0)
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

        top_row = QHBoxLayout()
        top_row.addLayout(self.camera_layout)
        top_row.addWidget(self.dark_panel)
        controls_layout.addLayout(top_row)

        stage_row = QHBoxLayout()
        stage_row.addWidget(self.rotation_group)
        stage_row.addWidget(self.linear_group)
        controls_layout.addLayout(stage_row)

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
        self.controller = None

    # ── status bar interface ─────────────────────────────────────

    def status_info(self) -> str:
        s = self._status
        temp_str = ""
        if s["temp"] > -900:
            temp_str = f"  ·  \U0001f321 {s['temp']:.1f}\u00b0C"
        return (
            f"{s['state']}  ·  "
            f"FPS {s['fps']:.1f}  ·  "
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

    def set_spectrogram_ready(self, spectrogram):
        """Called after scan completes with keep_spectrogram=True."""
        self._pending_spectrogram = spectrogram
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
        self._spectro_viewer.load(self._pending_spectrogram)
        self._spectro_viewer.show()

    # ── calibration ──────────────────────────────────────────────

    def _on_open_calibration(self):
        if self._pending_cube is None:
            return

        self._cal_win = CalibrationWindow()
        self._cal_win.load_from_memory(
            cube_array=self._pending_cube,
            wavelengths=self._pending_wavelengths,
            name="ScanResult"
        )
        self._cal_win.show()

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

    def set_frame_index(self, index: int):
        self._status["frame"] = index

    def set_recorded_count(self, n: int):
        self._status["recorded"] = n

    def set_selected_y_label(self, y: int):
        self._status["y"] = y

    def set_sharpness_score(self, score: float):
        self._status["sharpness"] = score
        self.sharpness_panel.set_value(score)

    def set_temperature(self, temp_c: float):
        self._status["temp"] = temp_c

    # ── lifecycle ────────────────────────────────────────────────

    def closeEvent(self, event):
        self.cube_preview.cleanup()
        if self._cal_win is not None:
            self._cal_win.close()
        if self._spectro_viewer is not None:                    
            self._spectro_viewer.close()
        if self.controller is not None:
            self.controller.cleanup()
        super().closeEvent(event)