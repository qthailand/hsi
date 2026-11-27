# windows/scan_window.py
from __future__ import annotations

import json
import math
import time

import qtawesome as qta
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QProgressBar,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QFormLayout,
    QGridLayout,
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
    QMenu,
)

from config import AppConfig
from modules.ui_kit.theme import PLOT_FOREGROUND
from modules.live_view.mapping import DisplayMapper
from modules.live_view.image_canvas import (
    FOCUS_ROI_DEFAULT_HEIGHT,
    FOCUS_ROI_MIN_HEIGHT,
)
from modules.camera.widget import CameraControls
from modules.live_view.spectrum import LiveLineSpectrumWidget
from modules.cube_preview.widget import RealtimeCubePreviewWidget
from modules.dark.state import DarkFrameState
from modules.stage.config import FREE_SCAN_MAX_S, FREE_SCAN_MIN_S
from modules.stage.speed import LINEAR_MAX_UM_S, ROTATION_MAX_DPS, ROTATION_SPEED_RESOLUTION_DPS
from modules.acquisition.widget import CorrectionStatusDisplay
from modules.status_card import StatusCard
from modules.live_view.display_controls import DisplayControls
from windows.image_viewer import ImageViewer
from windows.scan_inspector import (ClippedTile, ReadyList, SharpnessTile, card_frame,
                                    inspector_stylesheet, live_column_stylesheet, section_label)
from modules.spectrogram_viewer.window import SpectrogramViewer

#: Provisional limits of the wavelength range spin boxes until the camera reports its axis.
WAVELENGTH_LIMITS_NM = (0.0, 9999.9)
#: Title of the cube preview box; a scan appends its settings (see ScanWindow.set_scanning).
CUBE_PREVIEW_TITLE = "Realtime Cube Preview"

#: Save frame menu, in pipeline order: (stage of ``FrameStages``, label, tooltip). The two
#: geometry labels are rewritten for the model in use (see ScanWindow._update_save_frame_menu).
SAVE_FRAME_STAGES = (
    ("raw", "Raw frame",
     "Camera counts as read from the sensor; only the slit axis is flipped to the display orientation"),
    ("dark", "Dark + hot-pixel corrected",
     "After dark subtraction and hot-pixel repair, before the smile correction"),
    ("geometry", "Smile corrected, full sensor",
     "After the smile (and keystone) remap. The edges the remap fills with zeros are kept, "
     "so every pixel lines up with the sensor and the other stages"),
    ("valid", "Smile corrected, valid area",
     "After the smile (and keystone) remap, cropped to the rows and spectral columns the cube "
     "keeps (no zero-filled edges)"),
    ("binned", "Binned bands (as recorded)",
     "One line of the cube: binned to the wavelength bands of the scan, keystone rows cropped. "
     "The band wavelengths are stored in the PNG"),
)
#: Font Awesome 5 (solid) icon of each Save frame menu entry; ``all`` = every stage at once.
SAVE_FRAME_ICONS = {"raw": "fa5s.camera", "dark": "fa5s.adjust", "geometry": "fa5s.vector-square",
                    "valid": "fa5s.crop-alt", "binned": "fa5s.layer-group", "all": "fa5s.copy"}
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


#: Smallest width (px) of the settings column (right): three status cards and the camera form fit.
INSPECTOR_MIN_WIDTH = 360
#: Height (px) of the Start / Stop scan buttons pinned under the settings.
SCAN_BUTTON_HEIGHT = 44
#: The Corrections rows of the settings column, in order: (status card key, name shown).
CORRECTION_NAMES = (("wavelength", "Wavelength"), ("dark", "Dark + hot pixel"),
                    ("smile", "Smile + keystone"), ("flatfield", "Flat-field"),
                    ("radiometric", "Radiometric (reflectance)"))
#: How often (ms) the settings column re-reads Ready to scan? and the scan progress.
INSPECTOR_REFRESH_MS = 250
#: Width (px) of the live column (left) and the height of its image.
LIVE_COLUMN_WIDTH = 340
LIVE_IMAGE_HEIGHT = 300
#: A live frame with more saturated pixels than this (%) is listed in Ready to scan?.
CLIPPED_WARN_PERCENT = 0.1


class ScanWindow(QWidget):
    snapshot_ready = pyqtSignal(object)
    snapshot_failed = pyqtSignal(int, str)
    cube_ready = pyqtSignal(object, object)
    calibration_requested = pyqtSignal(object, object)
    #: *Save as Radiometric Reference*: (cube, wavelengths) of the scan just finished.
    radiometric_reference_requested = pyqtSignal(object, object)
    #: *Save as Flat-field Profile*: (cube, wavelengths) of the scan just finished.
    flatfield_profile_requested = pyqtSignal(object, object)
    #: A status card was clicked: ``"smile"``, ``"wavelength"``, ``"flatfield"`` or ``"radiometric"``
    #: (the host opens that module's profile view).
    status_card_clicked = pyqtSignal(str)
    #: Save on the unsaved-result prompt: open (cube, wavelengths) with the export dialog.
    save_result_requested = pyqtSignal(object, object)
    #: Calibrate (cube, wavelengths) with the camera's matching radiometric reference: sent after
    #: a scan while *Auto radiometric calibration* is ticked, and when it is ticked afterwards.
    auto_calibration_requested = pyqtSignal(object, object)
    #: The user finished a wavelength range edit: Enter, or focus leaving the Start/End pair.
    wavelength_range_committed = pyqtSignal()
    #: Smile card switched: ``(smile, keystone)`` - smile off = no remap; keystone off = smile only.
    geometry_requested = pyqtSignal(bool, bool)

    def request_snapshot(self, request_id):
        controller = getattr(self, 'controller', None)
        worker = getattr(controller, 'preview_thread', None)
        if worker is None or not worker.isRunning():
            self.snapshot_failed.emit(request_id, 'Camera acquisition is not running')
        elif controller.dark_lib.is_collecting:
            self.snapshot_failed.emit(request_id, 'Finish dark capture before grabbing a frame')
        else:
            worker.request_snapshot(request_id)

    def request_frame_stages(self, stages):
        """Ask the camera for these processing stages of its next frame (``SAVE_FRAME_STAGES``
        keys); they arrive together in :meth:`save_frame_stages`."""
        controller = getattr(self, 'controller', None)
        worker = getattr(controller, 'preview_thread', None)
        if worker is None or not worker.isRunning():
            QMessageBox.warning(self, 'Save frame', 'Camera acquisition is not running')
        elif controller.dark_lib.is_collecting:
            QMessageBox.warning(self, 'Save frame', 'Finish dark capture before saving a frame')
        else:
            worker.request_frame_stages(tuple(stages))

    def save_frame_stages(self, packet):
        """Ask for a file name and write each image of a ``FrameStages`` as an 8/16-bit PNG.

        Values are unscaled. One stage: the chosen file. Several (all stages of one frame): the
        chosen name plus ``_<stage>`` per file. Each PNG carries a ``hsi`` text chunk (JSON) that
        says which stage it is, which corrections were applied and what a crop left out.
        """
        from modules.image import png_name, save_png
        images = dict(packet.images)
        if not images:
            return
        tags = {key: self._frame_stage_tag(key, packet) for key in images}
        if len(images) == 1:
            (key,) = images
            path, _ = QFileDialog.getSaveFileName(
                self, 'Save frame', png_name('scan', packet.camera_bin, (tags[key],)), 'PNG (*.png)')
            if not path:
                return
            targets = {key: path}
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, f'Save {len(images)} stages of one frame (one PNG each)',
                png_name('scan', packet.camera_bin), 'PNG (*.png)')
            if not path:
                return
            stem = path[:-4] if path.lower().endswith('.png') else path
            targets = {key: f"{stem}_{tags[key]}.png" for key in images}
        captured = time.strftime('%Y-%m-%dT%H:%M:%S')
        try:
            for key, target in targets.items():
                info = self._frame_stage_info(key, images[key], packet, captured)
                save_png(target, images[key], text={'Software': 'HSI System', 'hsi': json.dumps(info)})
        except Exception as exc:
            QMessageBox.warning(self, 'Save frame', str(exc))

    @staticmethod
    def _frame_stage_tag(key, packet) -> str:
        """File-name word of a stage: what was applied, e.g. ``dark``, ``smile-keystone_valid``."""
        geometry = ('smile-keystone' if packet.keystone_applied else 'smile') if packet.smile_applied else 'nosmile'
        return {'raw': 'raw',
                'dark': 'dark' if packet.dark_applied else 'nodark',
                'geometry': geometry,
                'valid': f'{geometry}_valid',
                'binned': 'binned'}.get(key, key)

    @staticmethod
    def _frame_stage_info(key, image, packet, captured) -> dict:
        """The ``hsi`` metadata of one saved stage (JSON-ready)."""
        after_dark = key != 'raw'
        after_geometry = key in ('geometry', 'valid', 'binned')
        info = {
            'stage': key, 'captured': captured, 'camera_bin': int(packet.camera_bin),
            'shape': [int(n) for n in image.shape], 'dtype': str(image.dtype),
            'slit_flipped': bool(packet.raw_inverted),
            'dark_applied': bool(packet.dark_applied and after_dark),
            'smile_applied': bool(packet.smile_applied and after_geometry),
            'keystone_applied': bool(packet.keystone_applied and after_geometry),
        }
        if key in ('valid', 'binned'):
            info['rows_left_out'] = [int(v) for v in packet.valid_rows]      # (top, bottom)
        if key == 'valid':
            info['columns_left_out'] = [int(v) for v in packet.valid_columns]   # (left, right)
        lam = packet.wavelengths_nm
        if key == 'binned' and lam is not None and len(lam) == image.shape[1]:
            info['wavelengths_nm'] = [round(float(v), 3) for v in lam]
        return info

    def _build_save_frame_menu(self):
        """The Save frame drop-down: one entry per processing stage, then all of them at once."""
        menu = QMenu(self.save_frame_btn)
        menu.setToolTipsVisible(True)
        self._save_frame_actions = {}
        for key, label, tip in SAVE_FRAME_STAGES:
            action = menu.addAction(qta.icon(SAVE_FRAME_ICONS[key], color=PLOT_FOREGROUND), label)
            action.setToolTip(tip)
            action.triggered.connect(lambda _=False, key=key: self.request_frame_stages((key,)))
            self._save_frame_actions[key] = action
        menu.addSeparator()
        every = menu.addAction(qta.icon(SAVE_FRAME_ICONS['all'], color=PLOT_FOREGROUND),
                               "All stages of one frame\u2026")
        every.setToolTip("Every stage above from the same frame, one PNG each, sharing one file name")
        every.triggered.connect(lambda: self.request_frame_stages(self._save_frame_stage_keys()))
        self._save_frame_actions['all'] = every
        menu.aboutToShow.connect(self._update_save_frame_menu)
        self.save_frame_btn.setMenu(menu)

    def _geometry_in_use(self) -> tuple:
        """``(smile, keystone)``: the geometric correction the next frame gets (status cards)."""
        geo = getattr(self, '_geometry', {})
        smile = bool(geo.get('model')) and bool(geo.get('smile', True))
        return smile, smile and bool(geo.get('keystone_available')) and bool(geo.get('keystone', True))

    def _save_frame_stage_keys(self) -> tuple:
        """The stages worth saving now: the geometry ones only while a smile model is applied."""
        smile, _ = self._geometry_in_use()
        return tuple(key for key, _, _ in SAVE_FRAME_STAGES if smile or key not in ('geometry', 'valid'))

    def _update_save_frame_menu(self):
        smile, keystone = self._geometry_in_use()
        name = 'Smile + keystone corrected' if keystone else 'Smile corrected'
        for key, where in (('geometry', 'full sensor'), ('valid', 'valid area')):
            action = self._save_frame_actions[key]
            action.setText(f"{name}, {where}" if smile else f"Smile corrected, {where} (no model applied)")
            action.setEnabled(smile)

    def __init__(self, config: AppConfig):
        super().__init__()
        self.camera_controls = CameraControls()
        self.camera_controls.set_limits(1, 300, 1, 1000)

        self.camera_layout = QVBoxLayout()
        self.camera_layout.addWidget(self.camera_controls)

        self.config = config
        self.viewer = ImageViewer()
        
        # ── Dark frame ───────────────────────────────────────────
        # the dark's state and requests (no box of its own: the Dark row, its peek and Ready to scan? show it)
        self.dark_panel = DarkFrameState(self)
        self.correction_status = CorrectionStatusDisplay(self)
        # flat-field profile and radiometric reference of this camera (set by the radiometric
        # integration). Five cards: small squares in a grid, the detail in each tooltip
        self.flatfield_card = StatusCard('Flat-field', self)
        self.flatfield_card.set_content('Waiting', 'Waiting for camera')
        self.radiometric_card = StatusCard('Radiometric', self)
        self.radiometric_card.set_content('Waiting', 'Waiting for camera')
        # Wavelength first (always on: the band grid needs it), then the ones that switch on / off
        self.status_cards = (self.correction_status.wavelength, self.correction_status.dark,
                             self.correction_status.smile, self.flatfield_card, self.radiometric_card)
        self._geometry = dict(model=False, keystone_available=False, smile=True, keystone=True)
        self.correction_status.dark.value_clicked.connect(self._on_dark_card_switch)
        self.correction_status.smile.value_clicked.connect(self._on_smile_card_switch)
        self.radiometric_card.value_clicked.connect(self._on_radiometric_card_switch)
        self._flatfield_on = True                 # Flat-field card switch (used at the next Start)
        self._flatfield_last = None
        self.flatfield_card.value_clicked.connect(self._on_flatfield_card_switch)
        self.correction_status.dark.set_togglable(True, "Click to switch dark + hot-pixel correction on / off")
        self.radiometric_card.set_togglable(True, "Click to switch automatic radiometric calibration on / off")
        # a click on a card opens that module's profile; Smile and Wavelength always have a
        # window to open, Flat-field and Radiometric only while there is a profile to show
        self._card_keys = {"smile": self.correction_status.smile, "wavelength": self.correction_status.wavelength,
                           "dark": self.correction_status.dark,
                           "flatfield": self.flatfield_card, "radiometric": self.radiometric_card}
        for key, card in self._card_keys.items():
            card.clicked.connect(lambda key=key: self.status_card_clicked.emit(key))
        self.correction_status.smile.set_clickable(True, "View ▸")   # the loaded model, else the Smile page
        self.correction_status.wavelength.set_clickable(True, "Open ▸")
        # the Dark row always opens its peek: the dark's details, Capture and Reload (no dark: just those)
        self._dark_viewable = False
        self._dark_warning = ""
        self.correction_status.dark.set_clickable(True, "")

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

        self.save_frame_btn = QPushButton(qta.icon("fa5s.save", color=PLOT_FOREGROUND),
                                          "Save frame (PNG)")      # opens the stage menu
        self.save_frame_btn.setToolTip(
            "Save the next camera frame as an 8/16-bit PNG (counts, unscaled) at the processing "
            "stage you choose, or all stages of that one frame")
        self._build_save_frame_menu()

        focus_roi_layout = QHBoxLayout()
        focus_roi_layout.addWidget(QLabel("Focus ROI Height"))
        focus_roi_layout.addWidget(self.focus_roi_height_spin)
        focus_roi_layout.addWidget(self.focus_roi_full_btn)

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

        # ── LEFT COLUMN: what the camera sees, to focus and expose ──
        left_panel = QWidget()
        left_panel.setObjectName("liveColumn")
        left_panel.setAttribute(Qt.WA_StyledBackground, True)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)
        left_panel.setFixedWidth(LIVE_COLUMN_WIDTH)

        # Live card: title, frame rate and display mode above the image; focus ROI and Save below
        live_card = card_frame()
        live_layout = QVBoxLayout(live_card)
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_layout.setSpacing(0)
        head = QHBoxLayout()
        head.setContentsMargins(12, 8, 8, 8)
        head.setSpacing(8)
        live_title = QLabel("Live")
        live_title.setStyleSheet("color: #e6e9ee; font-weight: 600; font-size: 13px;")
        self.live_fps_label = QLabel("")
        self.live_fps_label.setStyleSheet("color: #7d8794; font-size: 11px; font-family: Consolas, monospace;")
        head.addWidget(live_title)
        head.addWidget(self.live_fps_label)
        head.addStretch(1)
        for label in self.display_controls.findChildren(QLabel):
            if label.text() == "Display:":
                label.hide()                  # the combo says it (RAW / Enhance)
        head.addWidget(self.display_controls)
        live_layout.addLayout(head)
        self.viewer.setFixedHeight(LIVE_IMAGE_HEIGHT)        # the spectrum below takes the rest
        live_layout.addWidget(self.viewer)
        foot = QHBoxLayout()
        foot.setContentsMargins(8, 6, 8, 6)
        foot.setSpacing(6)
        roi_icon = QLabel()
        roi_icon.setPixmap(qta.icon("mdi.select", color="#E3B341").pixmap(16, 16))
        roi_icon.setToolTip("Focus ROI height (rows around the selector used for the sharpness)")
        self.focus_roi_height_spin.setToolTip(roi_icon.toolTip())
        self.focus_roi_height_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.focus_roi_height_spin.setFixedWidth(84)
        self.focus_roi_full_btn.setText("")
        self.focus_roi_full_btn.setIcon(qta.icon("fa5s.arrows-alt-v", color="#aeb7c4"))
        self.focus_roi_full_btn.setToolTip("Full height: the focus ROI over every row")
        self.save_frame_btn.setText("")
        self.save_frame_btn.setIcon(qta.icon("fa5s.save", color="#aeb7c4"))
        self.save_frame_btn.setToolTip("Save frame (PNG) ▾\n" + self.save_frame_btn.toolTip())
        foot.addWidget(roi_icon)
        foot.addWidget(self.focus_roi_height_spin)
        foot.addWidget(self.focus_roi_full_btn)
        foot.addStretch(1)
        foot.addWidget(self.save_frame_btn)
        live_layout.addLayout(foot)
        left_layout.addWidget(live_card)

        # Sharpness and Clipped tiles
        self.sharpness_tile = SharpnessTile()
        self.sharpness_tile.reset_requested.connect(self.reset_sharpness_scale)
        self.clipped_tile = ClippedTile(CLIPPED_WARN_PERCENT)
        self.histogram = self.clipped_tile       # the exposure histogram the controller feeds
        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        tiles.addWidget(self.sharpness_tile, 1)
        tiles.addWidget(self.clipped_tile, 1)
        left_layout.addLayout(tiles)

        spectrum_card = card_frame()
        spectrum_layout = QVBoxLayout(spectrum_card)
        spectrum_layout.setContentsMargins(10, 8, 10, 8)
        spectrum_layout.setSpacing(4)
        spectrum_label = QLabel("Spectrum · selected row")
        spectrum_label.setToolTip("Live spectrum along the horizontal selector of the image")
        spectrum_label.setStyleSheet("color: #9aa4b2; font-size: 11px;")
        spectrum_layout.addWidget(spectrum_label)
        spectrum_layout.addWidget(self.live_line_spectrum, 1)
        left_layout.addWidget(spectrum_card, 1)

        left_panel.setStyleSheet(live_column_stylesheet())

        # ── RIGHT COLUMN: the scan settings in the order they are set, Start / Stop pinned below ──
        controls_widget = QWidget()
        controls_widget.setObjectName("inspectorBody")
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(12, 10, 12, 10)
        controls_layout.setSpacing(4)

        # Ready to scan?: only what needs attention (refreshed by _refresh_inspector)
        self.ready_list = ReadyList()
        self._ready_extra = {}              # host items, e.g. the camera's pixel pitch (set_ready_issue)
        controls_layout.addWidget(self.ready_list)

        controls_layout.addLayout(self.camera_layout)
        for group in (self.rotation_group, self.linear_group, self.free_group):   # one shows (stage.json)
            controls_layout.addWidget(group)
        self.wavelength_range_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        controls_layout.addWidget(self.wavelength_range_group)

        # the corrections of the next scan: each status card drawn as a row (row = open its view,
        # switch = on / off)
        self.corrections_label = section_label("Corrections")
        self.corrections_label.setContentsMargins(0, 14, 0, 4)
        controls_layout.addWidget(self.corrections_label)
        self.correction_rows = {}
        for key, name in CORRECTION_NAMES:
            card = self._card_keys[key]
            card.set_row(name)
            self.correction_rows[key] = card
            controls_layout.addWidget(card)
        controls_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(controls_widget)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        self.settings_scroll = scroll_area

        # Start / Stop: always on screen at the bottom of the column, one of them at a time. They are
        # the live view's Start / Stop (its signals, F5 / Esc, set_scanning_ui); the image has none.
        self.start_scan_btn = QPushButton(qta.icon("fa5s.play", color="white"), "  Start scan")
        self.start_scan_btn.setToolTip("Start Scan (F5)")
        self.stop_scan_btn = QPushButton(qta.icon("fa5s.stop", color="white"), "  Stop scan")
        self.stop_scan_btn.setToolTip("Stop Scan (Esc)")
        for button, colour, hover in ((self.start_scan_btn, "#1a9a55", "#22c06b"),
                                      (self.stop_scan_btn, "#c62828", "#e53935")):
            button.setMinimumHeight(SCAN_BUTTON_HEIGHT)
            button.setCursor(Qt.PointingHandCursor)
            button.setStyleSheet(
                f"QPushButton {{ background-color: {colour}; color: white; border: none; border-radius: 8px;"
                f" font-size: 15px; font-weight: bold; }}"
                f"QPushButton:hover {{ background-color: {hover}; }}"
                "QPushButton:disabled { background-color: #2b2e33; color: #6b7280; }")
        self.viewer.use_scan_buttons(self.start_scan_btn, self.stop_scan_btn)
        self.scan_action_hint = QLabel("")
        self.scan_action_hint.setAlignment(Qt.AlignCenter)
        self.scan_action_hint.setStyleSheet("font-size: 11px; color: #9e9e9e;")
        # how far the running scan is (stage move or free-scan time), above Stop
        self.scan_progress = QProgressBar()
        self.scan_progress.setRange(0, 1000)
        self.scan_progress.setTextVisible(False)
        self.scan_progress.setFixedHeight(4)
        self.scan_progress.hide()
        action_bar = QWidget()
        action_bar.setObjectName("scanActionBar")
        action_bar.setAttribute(Qt.WA_StyledBackground, True)
        action_layout = QVBoxLayout(action_bar)
        action_layout.setContentsMargins(12, 10, 12, 8)
        action_layout.setSpacing(6)
        action_layout.addWidget(self.scan_progress)
        action_layout.addWidget(self.start_scan_btn)
        action_layout.addWidget(self.stop_scan_btn)
        action_layout.addWidget(self.scan_action_hint)
        self.viewer.scanning_ui_changed.connect(self._show_scan_action)

        inspector = QWidget()
        inspector.setObjectName("inspector")
        inspector.setAttribute(Qt.WA_StyledBackground, True)
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.setSpacing(0)
        inspector_layout.addWidget(scroll_area, 1)
        inspector_layout.addWidget(action_bar)
        inspector.setMinimumWidth(INSPECTOR_MIN_WIDTH)
        inspector.setStyleSheet(inspector_stylesheet())
        for spin in controls_widget.findChildren(QAbstractSpinBox):
            spin.setButtonSymbols(QAbstractSpinBox.NoButtons)     # the unit sits in the field; no arrows
        # the wavelength range shows its Start / End only while it is on
        self.wavelength_range_group.toggled.connect(self._show_wavelength_fields)
        self._show_wavelength_fields(self.wavelength_range_group.isChecked())
        self.inspector = inspector
        self._inspector_timer = QTimer(self)
        self._inspector_timer.setInterval(INSPECTOR_REFRESH_MS)
        self._inspector_timer.timeout.connect(self._refresh_inspector)
        self._inspector_timer.start()

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
        self.save_flatfield_btn = QPushButton("Save as Flat-field Profile")
        self.save_flatfield_btn.setEnabled(False)
        self.save_flatfield_btn.setToolTip(
            "Scan of a target that is uniform along the whole slit (a white strip; its reflectance "
            "need not be known)? Store how the light and the optics fall off along the slit. A "
            "radiometric reference made afterwards uses it, so a small Spectralon is enough.")
        self.save_flatfield_btn.setStyleSheet(
            self.save_reference_btn.styleSheet().replace("#ef6c00", "#00838f")
            .replace("#fb8c00", "#0097a7").replace("#e65100", "#006064"))
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
        ref_row.addWidget(self.save_flatfield_btn)
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

        self._scan_clock = None             # (monotonic start, expected s) while the stage moves
        self._scanning = False
        self.dark_panel.dark_toggled.connect(lambda _on: self._update_card_switches())
        self._update_card_switches()             # the switch faces start in step with the settings
        self._stage_speed = None            # speed the stage reported it runs, this scan

        # ── THREE COLUMNS: live view | realtime cube preview (the main view) | settings ──
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(cube_preview_group)
        splitter.addWidget(inspector)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        self.columns_splitter = splitter
        self._show_scan_action(False)

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
        self.save_flatfield_btn.clicked.connect(self._on_save_flatfield)
        self.open_spectrogram_btn.clicked.connect(self._on_open_spectrogram)
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
        self.save_flatfield_btn.setEnabled(True)
        if calibration is not None:
            self.open_reflectance_btn.setText(f"Open {self._calibrated_word(calibration.get('header'))} Viewer  ({shape_str})")
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
        last = getattr(self, "_radiometric_last", None)
        if last is not None:                                  # the card follows the switch
            self.set_radiometric_status(*last)
        if getattr(self, "_flatfield_last", None) is not None:   # Ready / Applied follows radiometric
            self.set_flatfield_status(*self._flatfield_last)
        if hasattr(self, "flatfield_card"):
            self._update_card_switches()
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

    @staticmethod
    def _calibrated_word(header) -> str:
        """``"Reflectance"``, or ``"Flat-fielded"`` for counts evened by the flat-field alone."""
        header = header or {}
        return "Flat-fielded" if header.get("flatfield_profile") and not header.get("reflectance_reference") \
            else "Reflectance"

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
        self.open_reflectance_btn.setText(f"Open {self._calibrated_word(header)} Viewer  ({shape})")
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

    def set_flatfield_status(self, state: str, text: str, tooltip: str = ""):
        """Show the flat-field state on its status card.

        Args:
            state: ``"applied"`` (green *Applied*: the active radiometric reference holds the
                profile), ``"on"`` (cyan *Ready*: a profile exists for this bin, no active reference
                uses it yet) or anything else (orange *Off*: no profile). The card is clickable
                (view the profile) for the first two.
            text: Card detail, e.g. ``bin 4 - 505/512 rows - response 0.46-1.17``; a leading
                ``Flat-field: `` is dropped (the card has that title).
            tooltip: Further lines for the card's tooltip.
        """
        self._flatfield_last = (state, text, tooltip)
        value, color = {"applied": ("Applied", "green"), "on": ("Ready", "cyan")}.get(state, ("Off", "orange"))
        detail = text[len("Flat-field: "):] if text.startswith("Flat-field: ") else text
        if not self.flatfield_enabled():
            value, color, detail = "Off", "orange", "Switched off - " + detail
        elif state == "on" and not self.auto_radiometric_box.isChecked():
            value, color = "Applied", "green"     # the next scan is flat-fielded on its own
        self.flatfield_card.set_content(value, detail, color)
        has_profile = state in ("applied", "on")
        self.flatfield_card.set_clickable(has_profile)
        self.flatfield_card.setToolTip(f"FLAT-FIELD: {value}\n{detail}" + (f"\n\n{tooltip}" if tooltip else "")
                                       + ("\n\nClick to view the profile." if has_profile else ""))

    def _on_save_flatfield(self):
        cube = self.counts_cube()                  # a flat-field profile is fitted on counts
        if cube is None:
            return
        self.flatfield_profile_requested.emit(cube, self._pending_wavelengths)

    def set_radiometric_status(self, state: str, text: str):
        """Show the automatic radiometric calibration state: ``state`` ``"on"`` (green),
        ``"expired"`` (amber) or anything else (grey), with ``text``."""
        self._radiometric_last = (state, text)
        color = {"on": "#43a047", "expired": "#fb8c00"}.get(state, "#7a8699")
        card = getattr(self, "radiometric_card", None)       # absent while the window is being built
        if card is not None:
            value, card_color = {"on": ("Applied", "green"), "expired": ("Expired", "red")}.get(
                state, ("Off", "orange"))
            detail = text[len("Auto radiometric: "):] if text.startswith("Auto radiometric: ") else text
            box = getattr(self, "auto_radiometric_box", None)
            if box is not None and not box.isChecked():       # switched off on the card / box
                value, card_color, detail = "Off", "orange", "Switched off - " + detail
            card.set_content(value, detail, card_color)
            card.set_clickable(state in ("on", "expired"))   # a reference to show
            card.setToolTip(card.toolTip() + ("\n\nClick to view the reference." if card.is_clickable() else ""))
        self.radiometric_status_label.setText(text)
        self.radiometric_status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    # ── correction switches on the status cards ───────────────────
    def _update_card_switches(self):
        """Card values are switches except during a scan (corrections are fixed for a scan)."""
        idle = not getattr(self, "_scanning", False)
        self.correction_status.dark.set_switch_state(self.dark_panel.is_dark_enabled())
        self.radiometric_card.set_switch_state(self.auto_radiometric_box.isChecked())
        self.flatfield_card.set_switch_state(self.flatfield_enabled())
        geo_now = getattr(self, "_geometry", {})
        if not geo_now.get("smile", True):
            self.correction_status.smile.set_switch_state(False)
        elif geo_now.get("keystone_available"):
            self.correction_status.smile.set_switch_state(True, "S+K" if geo_now.get("keystone", True) else "S")
        else:
            self.correction_status.smile.set_switch_state(True)
        self.correction_status.dark.set_togglable(idle, "Click to switch dark + hot-pixel correction on / off")
        self.radiometric_card.set_togglable(idle, "Click to switch automatic radiometric calibration on / off")
        self.flatfield_card.set_togglable(idle, "Click to switch the flat-field on / off (next scan): on its own it "
                                                "evens the counts along the slit; with radiometric calibration it is "
                                                "part of the reflectance")
        geo = getattr(self, "_geometry", {})
        tip = ("Click: Smile + keystone -> Smile only -> Off" if geo.get("keystone_available")
               else "Click to switch the smile correction on / off")
        self.correction_status.smile.set_togglable(idle and bool(geo.get("model")), tip)

    def update_geometry(self, status):
        """Take the engine's geometry switches / model from a correction status dict (``'geometry'``)."""
        geo = (status or {}).get("geometry")
        if not geo:
            return
        self._geometry = dict(geo)
        self._update_card_switches()

    def set_dark_viewable(self, on: bool, warning: str = ""):
        """A dark is loaded (``on``): its peek gets ``View ▸`` (the dark viewer). ``warning`` (e.g. a
        light leak found in it) goes into the card's tooltip and *Ready to scan?*. The Dark card is
        clickable either way: its peek holds Capture and Reload."""
        self._dark_viewable = bool(on)
        self._dark_warning = str(warning or "") if on else ""
        card = self.correction_status.dark
        card.set_clickable(True, "View ▸" if on else "")
        tip = "Click for the dark reference: view, capture, reload" if on else "Click to capture or reload a dark"
        if warning:
            tip = f"⚠ {warning}\n\n{tip}"
        card.setToolTip(tip)

    def dark_viewable(self) -> bool:
        """Whether a dark is loaded that the dark viewer can show (:meth:`set_dark_viewable`)."""
        return self._dark_viewable

    def peek_actions(self, key: str) -> tuple:
        """Buttons of status card ``key``'s peek, ``((action, label), ...)``: for the Dark, Capture
        and Reload (Cancel while capturing; none during a scan or while the panel is locked)."""
        if key != "dark" or self._scanning:
            return ()
        panel = self.dark_panel
        if panel.is_capturing():
            return (("cancel", "Cancel capture"),)
        if not panel.is_interactive():
            return ()
        return (("capture", "Capture"), ("reload", "Reload"))

    def run_peek_action(self, key: str, action: str) -> None:
        """Do what a peek button asked (:meth:`peek_actions`): the dark panel's own requests."""
        if key != "dark":
            return
        panel = self.dark_panel
        request = {"capture": panel.request_capture, "reload": panel.request_reload,
                   "cancel": panel.request_cancel}.get(action)
        if request is not None:
            request()

    def dark_setting_text(self) -> str:
        """The camera setting a dark is stored for, e.g. ``6 ms · gain 1 · bin 2``."""
        controls = self.camera_controls
        return f"{controls.exposure_ms():g} ms · gain {controls.gain():g} · bin {self.selected_bin()}"

    def _on_dark_card_switch(self):
        if self._scanning:
            return
        self.dark_panel.set_dark_enabled(not self.dark_panel.is_dark_enabled())
        self._update_card_switches()

    def flatfield_enabled(self) -> bool:
        """Flat-field card switch: the next scan is flat-fielded (on its own, or within reflectance)."""
        return bool(getattr(self, "_flatfield_on", True))

    def _on_flatfield_card_switch(self):
        if self._scanning:
            return
        self._flatfield_on = not self.flatfield_enabled()
        if self._flatfield_last is not None:
            self.set_flatfield_status(*self._flatfield_last)
        self._update_card_switches()

    def _on_radiometric_card_switch(self):
        if self._scanning or not self.auto_radiometric_box.isEnabled():
            return
        self.auto_radiometric_box.setChecked(not self.auto_radiometric_box.isChecked())

    def _on_smile_card_switch(self):
        """Cycle the geometry: smile + keystone -> smile only -> off (smile on / off for a smile-only model)."""
        geo = self._geometry
        if self._scanning or not geo.get("model"):
            return
        smile, keystone = bool(geo.get("smile", True)), bool(geo.get("keystone", True))
        if geo.get("keystone_available"):
            if smile and keystone:
                smile, keystone = True, False
            elif smile:
                smile, keystone = False, False
            else:
                smile, keystone = True, True
        else:
            smile, keystone = (not smile), True
        geo.update(smile=smile, keystone=keystone)
        self._update_card_switches()
        self.geometry_requested.emit(smile, keystone)

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

    def status_card(self, key: str):
        """The status card for ``key`` (``'wavelength'``, ``'dark'``, ``'smile'``, ``'flatfield'``,
        ``'radiometric'``), e.g. to place its profile peek; ``None`` for an unknown key."""
        return self._card_keys.get(key)

    def set_bin_enabled(self, enabled):
        self.camera_controls.set_bin_enabled(enabled)

    # ── scanning ─────────────────────────────────────────────────

    def _show_wavelength_fields(self, on: bool):
        """Show the Start / End rows of the wavelength range only while the range is on."""
        form = self.wavelength_range_group.layout()
        for spin in (self.wavelength_start_spin, self.wavelength_end_spin):
            spin.setVisible(bool(on))
            label = form.labelForField(spin) if form is not None else None
            if label is not None:
                label.setVisible(bool(on))

    def _show_scan_action(self, scanning: bool):
        """Show Start (idle) or Stop (scanning) under the settings, enabled like the live view's
        own buttons (:meth:`ImageViewer.set_scanning_ui`)."""
        self.start_scan_btn.setVisible(not scanning)
        self.stop_scan_btn.setVisible(scanning)
        self.scan_progress.setVisible(scanning)
        self.scan_action_hint.setText("Esc stops the scan" if scanning else "F5 starts the scan")
        self._refresh_scan_progress()

    # ── settings column: Ready to scan? and the scan progress ─────

    def set_ready_issue(self, key: str, text: str = "", action: str = "", callback=None,
                        severity: str = "warning"):
        """Add (or with an empty ``text`` remove) an item of *Ready to scan?* that the host
        knows about, e.g. a camera without a pixel pitch.

        Args:
            key: Identifies the item (a second call replaces it).
            text: What needs attention; ``""`` removes the item.
            action: Caption of its fix button (``""``: none).
            callback: Called by that button.
            severity: ``"error"``, ``"warning"`` or ``"info"``.
        """
        if text:
            self._ready_extra[key] = (severity, str(text), str(action), callback)
        else:
            self._ready_extra.pop(key, None)
        self._refresh_inspector()

    def ready_items(self):
        """``(items, passed)`` of *Ready to scan?*: what needs attention as ``(severity, text,
        action, callback)``, and the names of the checks that passed. Only states the window
        already shows elsewhere (dark panel, histogram, status cards, status line)."""
        items, passed = [], []
        panel = self.dark_panel
        state = panel.state()
        if state == panel.CAPTURING:              # the frame count is on the Dark row
            items.append(("info", "Capturing the dark frame…", "Cancel", panel.request_cancel))
        elif state != panel.LOADED:
            items.append(("error", "No dark frame", "Capture",
                          None if self._scanning else panel.request_capture))
        elif not panel.is_dark_enabled():
            items.append(("warning", "Dark subtraction is off", "Turn on",
                          None if self._scanning else self._turn_dark_on))
        elif self._dark_warning:
            items.append(("warning", "Light leak in the dark", "View",
                          lambda: self.status_card_clicked.emit("dark")))
        else:
            passed.append("Dark frame")
        # raw clipping counted by the engine (confirmed), near full scale = the exposure warning
        percent = self.histogram.saturated_percent()
        near = self.histogram.near_percent()
        if percent is not None and percent > CLIPPED_WARN_PERCENT:
            items.append(("warning", f"{percent:.2f} % clipped", "", None))
        elif near is not None and near > CLIPPED_WARN_PERCENT:
            items.append(("warning", f"{near:.2f} % near full scale", "", None))
        elif percent is not None:
            passed.append("No clipped pixels")
        elif self.histogram.check_status() == "not_checked":
            items.append(("info", "Clipping not checked", "", None))
        if self._geometry.get("model"):
            passed.append("Smile model")
        else:
            items.append(("info", "No smile model", "Open",
                          lambda: self.status_card_clicked.emit("smile")))
        if self.correction_status.wavelength.color_name() == "green":
            passed.append("Wavelength calibration")
        warning = str(self._status.get("calibration_warning") or "").strip()
        if warning:
            items.append(("warning", warning.splitlines()[0], "", None))
        line_rate = self.status_line_rate_warning()
        if line_rate:
            items.append(("info", line_rate, "", None))
        items.extend(self._ready_extra.values())
        return items, passed

    def _turn_dark_on(self):
        """*Turn on* of the Ready list: dark subtraction on, never in the middle of a cube."""
        if not self._scanning:
            self.dark_panel.set_dark_enabled(True)

    def _refresh_inspector(self):
        items, passed = self.ready_items()
        self.ready_list.set_items(items, passed)
        self._refresh_scan_progress()
        # a dark capture in progress: the Dark row counts its frames (one indicator, in one place)
        done, total = self.dark_panel.capture_progress()
        self.correction_status.dark.set_row_note(
            f"{done} / {total}" if self.dark_panel.is_capturing() and total else None)

    def _refresh_scan_progress(self):
        """Progress bar and time left on Stop while the stage moves (or a free scan records)."""
        clock = self._scan_clock if self._scanning else None
        if clock is None or clock[1] <= 0:
            self.scan_progress.setValue(0)
            self.stop_scan_btn.setText("  Stop scan")
            return
        started, total = clock
        elapsed = max(0.0, time.monotonic() - started)
        self.scan_progress.setValue(int(1000 * min(1.0, elapsed / total)))
        self.stop_scan_btn.setText(f"  Stop scan   {_format_countdown(total - elapsed)} left")

    def set_scanning(self, active: bool):
        """Lock the camera and stage settings during a scan.

        The settings stay in their column, locked, so their values can be read while the cube
        preview fills; the preview title also shows them in one line
        (:meth:`scan_settings_summary`). The bin, wavelength range and dark panel are locked
        separately.

        Args:
            active: True when a scan starts, False when it ends (or fails).
        """
        self.camera_controls.set_settings_enabled(not active)
        self.rotation_group.setEnabled(not active)
        self.linear_group.setEnabled(not active)
        self.free_duration_spin.setEnabled(not active)   # the box stays lit for its countdown
        # chosen at Start for the whole scan (frames are calibrated while recorded)
        self.auto_radiometric_box.setEnabled(not active)
        if active and not self._scanning:
            self.clipped_tile.reset_peak()      # the max then says whether this scan clipped on the way
        self._scanning = bool(active)
        self._update_card_switches()
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
        self._show_wavelength_fields(wavelength_range is not None)

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
        self.sharpness_tile.set_scanning(text == "SCANNING")

    def set_calibration_warning(self, text: str):
        """Calibration warnings of the running scan, shown after the state ("" for none)."""
        self._status["calibration_warning"] = text

    def set_fps(self, fps: float):
        self._status["fps"] = fps
        self.live_fps_label.setText(f"{fps:.1f} fps" if fps else "")

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
        self.sharpness_tile.set_value(score)

    def reset_sharpness_scale(self):
        """Forget the sharpness peak (the scores of a new gain/exposure/bin are not comparable)."""
        self.sharpness_tile.reset()

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

    def showEvent(self, event):
        """Qt handler: refresh the inspector at once, then every :data:`INSPECTOR_REFRESH_MS`."""
        super().showEvent(event)
        timer = getattr(self, "_inspector_timer", None)
        if timer is not None and not timer.isActive():
            self._refresh_inspector()
            timer.start()

    def hideEvent(self, event):
        """Qt handler: hidden or minimised, nobody reads the inspector: its timer pauses."""
        super().hideEvent(event)
        timer = getattr(self, "_inspector_timer", None)
        if timer is not None:
            timer.stop()

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
