# controllers/scan_controller.py
"""Scan controller — Orin-compatible, binning in worker thread.

Pipeline:
    Worker:  backend.read_frame() → orient → LambdaBin → emit(30fps)
    GUI:     display + spectrum + histogram + record (append only)

Dark capture:
    Worker:  backend.read_frame_raw() → emit(raw, no bin) — dark_capture_mode
    GUI:     add to dark library at raw resolution, with the last sensor temperature

Temperature:
    The sensor temperature is polled every ``_temp_interval`` s on the GUI thread. In preview
    the dark nearest that temperature is selected (and switched when another file becomes
    nearer); during a scan the dark stays fixed and the readings go into the scan metadata.
    Cameras with a settable target (TEC) get the configured target at start.

Settings:
    With ``persist_settings`` (set only by ``main.py``), the user's edits of gain, exposure,
    bin, target temperature and the stage scan values are written back to
    ``camera.json``/``scan.json`` (debounced) so the next start uses them. Start-up values
    are never written: the camera may have clamped them (e.g. a fake camera in tests).
    A wavelength range the user set (and the worker applied) is written to this camera
    backend's entry of ``processing.json`` ``wavelength_range_nm``.

Wavelength range:
    The worker bins (and records) only the bins inside the range; changes are applied at a
    frame boundary in preview, refused during a scan (the control is locked), and Start Scan
    waits for a change still being applied (the user presses Start again).
"""

from __future__ import annotations

import logging
import os
import time
import gc

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox

from modules.acquisition import ScanSession, build_raw_metadata
from modules.dark import DarkFrameLibrary, select_dark, valid_temperature
from modules.stage import StageThread

from config import AppConfig
from modules.contracts import CalibrationStore, CameraSetting
from modules.spectral_processing.wavelength import spectrum_from_corrected_row
from modules.cube_io import create_pushbroom_cube_from_corrected
from modules.smile import SmileCorrector
from modules.reference_roi import select_rgb, compose_preview

HISTOGRAM_UPDATE_INTERVAL_SEC = 0.25
SETTINGS_SAVE_DELAY_MS = 1000       # write camera/stage settings once edits pause this long
#: scan.json keys of ``ScanWindow.stage_spins()``, in the same order.
STAGE_SETTING_KEYS = ("rotation_speed_dps", "rotation_degree", "linear_distance_mm", "linear_speed_um_s")
#: processing.json key of the per-backend wavelength range.
RANGE_SETTING_KEY = "wavelength_range_nm"

logger = logging.getLogger(__name__)


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}


class ScanController:
    """Glue between the scan window, the camera/stage and the GUI-free scan session.

    The controller owns *no* acquisition or scan logic of its own: frames are processed by
    ``modules.acquisition`` (:class:`CameraPreviewThread`), the scan state machine and
    recording hand-over live in :class:`ScanSession`, dark selection in
    :func:`modules.dark.select_dark`. What remains here is wiring: widgets <-> worker <->
    stage, plus the live-view throttling that only makes sense with a GUI.
    """

    PREVIEW = ScanSession.PREVIEW
    SCANNING = ScanSession.SCANNING
    ERROR = ScanSession.ERROR

    # Defaults for controllers built without __init__ (tests drive single slots).
    _settings_timer = None
    _dirty_settings = frozenset()       # setting keys edited since the last save
    _scan_temps = ()
    last_scan_header: dict = {}
    dark_reference = None
    _range_requested = None             # range last sent to the worker (or applied), (lo, hi) nm or None
    _range_applied = None               # range the worker reported in use
    _range_dirty = False                # a user edit is waiting for the worker's answer
    _range_saved = None                 # user range the worker applied, written by _save_settings
    _range_bands = ()                   # kept bins (k0, k1) or None, for the sharpness reset; () = none seen

    def __init__(self, win, config: AppConfig, persist_settings: bool = False):
        """Open the camera, wire the window and start the live preview.

        Args:
            win: The :class:`ScanWindow`.
            config: Application settings.
            persist_settings: Save the user's camera/stage edits to ``config/*.json`` (see
                ``_save_settings``). Off by default so tests and tools never write the user's files.
        """
        self.win = win
        self.viewer = win.viewer
        self.config = config
        self.session = ScanSession()
        self._camera_bin: int = 1

        self.backend = None
        self.preview_thread = None
        self.disable_camera = _env_enabled("HSI_DISABLE_CAMERA", False)
        self.disable_preview_thread = _env_enabled("HSI_DISABLE_PREVIEW_THREAD", False)

        # ── dark frame ───────────────────────────────────────────
        self.dark_lib = DarkFrameLibrary(
            folder=self.config.dark_frame_folder,
            n_frames=self.config.dark_frame_n_frames,
        )
        self.dark_reference: dict | None = None
        self._dark_path: str | None = None      # file of the active dark (to notice a nearer one)

        # ── camera backend factory (config.camera_backend: poa | genicam | ximea) ──
        from modules.camera import create_backend
        self.backend = create_backend(self.config)

        self.backend.open()

        # ── preview thread ───────────────────────────────────────
        from modules.acquisition import CameraPreviewThread

        self.preview_thread = CameraPreviewThread(self.backend, config=config,
                                                  profile=bool(getattr(config, "profile_timing", True)))
        self.preview_thread.saturation_ceiling_ready.connect(self._on_saturation_ceiling)
        self.preview_thread.frame_ready.connect(self.on_preview_frame)
        self.preview_thread.processed_frame_ready.connect(self.win.save_processed_frame)
        self.preview_thread.bin_applied.connect(self._on_bin_applied)
        self.preview_thread.bin_failed.connect(self._on_bin_failed)
        self.preview_thread.params_applied.connect(self._on_params_applied)
        self.preview_thread.params_failed.connect(self._on_params_failed)
        self.preview_thread.target_temperature_applied.connect(self._on_target_temperature_applied)
        self.preview_thread.target_temperature_failed.connect(self._on_target_temperature_failed)
        self.preview_thread.frame_rate_ready.connect(self.win.set_camera_frame_rate)
        self.preview_thread.sharpness_ready.connect(self._on_sharpness_ready)
        self.preview_thread.error.connect(self.on_preview_error)
        self.preview_thread.correction_status.connect(self.win.correction_status.update_status)
        self.preview_thread.saturation_ready.connect(self._on_saturation)
        self.preview_thread.set_raw_inverted(self.config.raw_inverted)
        # Wavelength calibrations stored per camera setting replace config m/c at start and on bin changes.
        store = getattr(self.config, '_config_store', None)
        self._wavelength_store = (
            CalibrationStore(store.wavelength_folder(self.config.camera_backend), self.config.smile_map_folder)
            if store is not None else None)
        if self._wavelength_store is not None:
            self.preview_thread.set_wavelength_lookup(self._stored_wavelength_profile)
            # FWHM stored per setting sets the bin width when resolution_source is "fwhm".
            self.preview_thread.set_resolution_lookup(self._stored_fwhm_nm)
        # The configured range applies from the first frame (nothing pending, nothing saved).
        self.preview_thread.wavelength_range_applied.connect(self._on_wavelength_range_applied)
        self.preview_thread.wavelength_range_failed.connect(self._on_wavelength_range_failed)
        start_range = (getattr(self.config, RANGE_SETTING_KEY, None) or {}).get(self._range_backend())
        self.preview_thread.set_wavelength_range(start_range)
        self._range_requested = self._range_applied = self.preview_thread.wavelength_range
        self.win.set_wavelength_range(self._range_requested)

        self.stage: StageThread | None = None

        # ── reference ROI (visual selection bridge to MainWindow) ──
        self._reference_profile = None
        self._reference_overlay_key = None
        self._last_spatial_height = 0
        self.reference_preview_ready = None   # callable(request_id, image, context)
        self.reference_preview_failed = None  # callable(request_id, message)

        self.preview_frame_index = 0

        self.last_display_update = 0.0
        self.last_param_update = 0.0
        self.last_spectrum_update = 0.0
        self.last_histogram_update = 0.0
        self.saturation_ceiling: np.ndarray | None = None   # display-frame shape, see AcquisitionSink
        self.current_wavelengths: np.ndarray | None = None
        self.enable_live_image = _env_enabled("HSI_ENABLE_LIVE_IMAGE", True)
        self.enable_live_spectrum = _env_enabled("HSI_ENABLE_LIVE_SPECTRUM", True)
        self.enable_histogram = _env_enabled("HSI_ENABLE_HISTOGRAM", True)
        self.enable_cube_preview = _env_enabled("HSI_ENABLE_CUBE_PREVIEW", True)

        for env_name, attr in (
            ("HSI_DISABLE_LIVE_IMAGE", "enable_live_image"),
            ("HSI_DISABLE_LIVE_SPECTRUM", "enable_live_spectrum"),
            ("HSI_DISABLE_HISTOGRAM", "enable_histogram"),
            ("HSI_DISABLE_CUBE_PREVIEW", "enable_cube_preview"),
        ):
            if _env_enabled(env_name, False):
                setattr(self, attr, False)

        self.profile_frame_ms = float(os.getenv("HSI_PROFILE_FRAME_MS", "0") or 0)

        self.display_mapper = self.win.display_controls.current_mapper()

        # ★ temperature cache
        self._last_temp: float = -999.0
        self._last_temp_update: float = 0.0
        self._temp_interval: float = 3.0
        self._scan_temps: list[float] = []         # readings during the current scan
        self.last_scan_temperatures: dict = {}     # temperature metadata of the last cube
        self.last_scan_header: dict = {}           # everything the last cube's export header records

        # Settings the user changes are saved once edits pause (see _save_settings).
        if persist_settings:
            self._settings_timer = QTimer(self.win)
            self._settings_timer.setSingleShot(True)
            self._settings_timer.setInterval(SETTINGS_SAVE_DELAY_MS)
            self._settings_timer.timeout.connect(self._save_settings)

        self._apply_backend_limits()
        # The spin boxes start at their minimum; start from the configured defaults
        # so the camera, dark lookup and UI all agree.
        self.win.camera_controls.set_values(
            self.config.default_gain, self.config.default_exposure_ms)
        logger.info("Start settings: gain %s, exposure %s ms (shown %s ms within the camera's range)",
                    self.config.default_gain, self.config.default_exposure_ms,
                    self.win.camera_controls.exposure_ms())
        self._apply_bin_options_and_initial_bin()
        profile = getattr(self.config, '_wavelength_profile', None)
        if profile is not None and profile.frame_shape:
            info = self.backend.info
            if tuple(profile.frame_shape) != (info.height, info.width):
                self.backend.close()
                raise ValueError("Default wavelength calibration geometry differs from camera; select a compatible profile/config")
        self._try_set_camera_bit_depth()
        self._setup_cooling()
        self._connect_ui()
        self._set_state(self.PREVIEW)

        self._poll_temperature(time.time())   # the first dark lookup uses the real temperature
        self.apply_params()  # also looks up the stored dark for these params
        self.preview_thread.start()

    # ══════════════════════════════════════════════════════════════
    #  Smile correction
    # ══════════════════════════════════════════════════════════════

    def apply_wavelength_calibration(self, profile):
        """Queue a preview-session calibration change at a worker frame boundary."""
        profile.validate()
        if self.state != self.PREVIEW or self.dark_lib.is_collecting:
            raise RuntimeError("Apply wavelength calibration only during normal preview")
        if profile.camera_bin != int(self.backend.current_bin):
            raise ValueError("Profile hardware bin differs from current camera bin")
        if profile.frame_shape:
            info = self.backend.info
            if tuple(profile.frame_shape) != (info.height, info.width):
                raise ValueError("Profile image shape differs from camera geometry")
        self.preview_thread.request_wavelength(profile)

    def _stored_wavelength_profile(self):
        """Wavelength profile stored for the camera's current setting (called on the worker thread).

        Raises:
            LookupError: No file for this bin, or it was measured at another setting
                (camera, index, orientation or frame size).
        """
        setting = self.camera_setting()
        stored = self._wavelength_store.load_wavelength(setting)
        if stored is None:
            raise LookupError(f"no {self._wavelength_store.wavelength_path(setting)}")
        if not stored.ok:
            raise LookupError(f"{stored.path} is for another setting: {'; '.join(stored.problems)}")
        return stored.profile

    def _stored_fwhm_nm(self) -> float:
        """Median FWHM (nm) stored for the camera's current setting (called on the worker thread).

        Raises:
            LookupError: No file for this bin, or it was measured at another setting.
        """
        setting = self.camera_setting()
        stored = self._wavelength_store.load_resolution(setting)
        if stored is None:
            raise LookupError(f"no {self._wavelength_store.resolution_path(setting)}")
        if not stored.ok:
            raise LookupError(f"{stored.path} is for another setting: {'; '.join(stored.problems)}")
        return stored.fwhm_nm

    def reload_spectral_resolution(self) -> None:
        """Re-read the stored FWHM (after a new one was saved); applied once no scan is running."""
        self.preview_thread.request_resolution_reload()

    def load_smile_model(self, path: str) -> None:
        if self.state == self.SCANNING or self.dark_lib.is_collecting:
            raise RuntimeError("Stop scan/dark capture before changing smile model")
        self.preview_thread.request_smile_model(path)

    def calibrate_smile(self, frame=None, **params) -> bool:
        """Fit an explicitly supplied oriented, unbinned lamp frame."""
        if frame is None:
            raise ValueError("Supply an oriented sensor lamp frame; binned preview cannot calibrate smile")
        if self.state == self.SCANNING or self.dark_lib.is_collecting:
            raise RuntimeError("Stop scan/dark capture before calibration")
        model = SmileCorrector(**params).fit(frame)
        from pathlib import Path
        folder = Path(self.config.smile_map_folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"smile_model_bin{self.backend.current_bin}.npz"
        model.save(str(path))
        self.load_smile_model(str(path))
        return True

    # ══════════════════════════════════════════════════════════════
    #  Reference ROI
    # ══════════════════════════════════════════════════════════════

    def camera_setting(self):
        """The camera configuration calibrations must match (``modules.contracts.CameraSetting``)."""
        info = self.backend.info
        return CameraSetting(str(self.config.camera_backend).lower(), int(self.config.camera_index),
                             int(self.backend.current_bin), bool(self.config.raw_inverted),
                             (int(info.height), int(info.width)))

    def reference_roi_context(self, spatial_height: int) -> dict:
        """Camera configuration a reference ROI profile is only valid for."""
        return {
            'camera_backend': self.config.camera_backend,
            'camera_id': self.config.camera_id,
            'camera_index': self.config.camera_index,
            'camera_bin': int(self.backend.current_bin),
            'raw_inverted': self.config.raw_inverted,
            'spatial_height': int(spatial_height),
        }

    def set_reference_roi_profile(self, profile) -> None:
        """Set (or clear with None) the active profile; overlay refreshes on next frame."""
        self._reference_profile = profile
        self._reference_overlay_key = ('unset',)
        self.viewer.image.set_reference_roi(None)

    def _refresh_reference_overlay(self, spatial_height: int) -> None:
        profile = self._reference_profile
        band = None
        if profile is not None and profile.matches(self.reference_roi_context(spatial_height)):
            band = (profile.y_start, profile.y_end, int(spatial_height))
        if band == self._reference_overlay_key:
            return
        self._reference_overlay_key = band
        self.viewer.image.set_reference_roi(band)

    def start_reference_preview(self, request_id: int) -> None:
        """Run a normal scan whose result is a spatial-y x scan-x RGB preview.

        No cube is published; the image is delivered through
        ``reference_preview_ready`` when the scan move ends.
        """
        if self.stage is not None:
            raise RuntimeError("Stage is still stopping or returning home")
        if self.state != self.PREVIEW or self.dark_lib.is_collecting:
            raise RuntimeError("Reference preview needs normal preview (no scan/dark capture)")
        self.session.start_reference_preview(request_id)
        self.start_scan()
        if self.state != self.SCANNING:
            self.session.cancel_reference_preview()
            raise RuntimeError("Scan could not be started")

    # ══════════════════════════════════════════════════════════════
    #  Properties
    # ══════════════════════════════════════════════════════════════

    @property
    def state(self) -> str:
        """``PREVIEW``, ``SCANNING`` or ``ERROR`` (held by the session)."""
        return self.session.state

    @state.setter
    def state(self, value: str):
        self.session.state = value

    @property
    def recording(self) -> bool:
        """True while the current scan is recording frames."""
        return self.session.recording

    @recording.setter
    def recording(self, value: bool):
        self.session.recording = bool(value)

    @property
    def images(self) -> list:
        """Binned frames of the last finished scan."""
        return self.session.images

    @property
    def camera_bin(self) -> int:
        """Hardware bin the camera is actually running at."""
        return int(self.backend.current_bin)

    @property
    def temperature(self) -> float:
        return self._last_temp

    # ══════════════════════════════════════════════════════════════
    #  UI wiring
    # ══════════════════════════════════════════════════════════════

    def _connect_ui(self):
        self.viewer.start_scan_requested.connect(self.start_scan)
        self.viewer.stop_scan_requested.connect(self.stop_scan_manually)
        self.viewer.line_y_changed.connect(self.on_line_y_changed)

        self.win.camera_controls.params_changed.connect(self.apply_params)
        self.win.camera_controls.bin_changed.connect(self.apply_bin)
        self.win.camera_controls.target_temperature_changed.connect(self.set_target_temperature)
        # Only the settings the user edits are saved, each on its own: start-up values (possibly
        # clamped to the camera) are not, and another open copy of the app cannot overwrite them.
        controls = self.win.camera_controls
        controls.gain_spin.valueChanged.connect(lambda *_: self._schedule_settings_save("default_gain"))
        controls.exp_spin.valueChanged.connect(lambda *_: self._schedule_settings_save("default_exposure_ms"))
        for spin, key in zip(self.win.stage_spins(), STAGE_SETTING_KEYS):
            spin.valueChanged.connect(lambda *_, key=key: self._schedule_settings_save(key))
        for edited in self.win.wavelength_range_edits():
            edited.connect(self._on_wavelength_range_edited)

        self.win.focus_roi_height_spin.valueChanged.connect(
            self.on_focus_roi_height_changed
        )
        self.win.focus_roi_full_btn.clicked.connect(
            self.set_focus_roi_full_height
        )

        self.win.dark_panel.capture_requested.connect(self.start_dark_capture)
        self.win.dark_panel.cancel_requested.connect(self.cancel_dark_capture)
        self.win.dark_panel.reload_requested.connect(
            self._load_dark_reference_for_current_params
        )
        self.win.dark_panel.dark_toggled.connect(self.toggle_dark_subtraction)

        self.win.display_controls.mapper_changed.connect(
            self._on_display_mapper_changed
        )

    def _try_set_camera_bit_depth(self):
        try:
            bd = getattr(self.backend, "bit_depth", None)
            if bd is None:
                bd = self.backend.info.bit_depth
            if bd is not None:
                try:
                    sensor_bits = int(self.backend.sensor_bit_depth())
                except Exception:
                    sensor_bits = None
                self.win.display_controls.set_bit_depth(int(bd), sensor_bits)
                self._on_display_mapper_changed(self.win.display_controls.current_mapper())
                self.win.histogram.set_bit_depth(int(bd))
        except Exception as e:
            logger.debug("Could not query camera bit depth: %s", e)

    def _on_display_mapper_changed(self, mapper):
        self.display_mapper = mapper
        self.viewer.image.set_display_mapper(mapper)

    def _apply_backend_limits(self):
        try:
            gmin, gmax = self.backend.gain_range
            emin_us, emax_us = self.backend.exposure_range_us
            self.win.camera_controls.set_limits(            
                int(gmin), int(gmax),
                emin_us / 1000, emax_us / 1000
            )
        except Exception as e:
            logger.warning("Could not apply backend limits: %s", e)

    def _apply_bin_options_and_initial_bin(self):
        try:
            bins = tuple(int(b) for b in self.backend.get_bins())
            if bins and self.config.default_bin in bins:
                initial_bin = self.config.default_bin
            elif bins:
                initial_bin = bins[0]
            else:
                bins = (1,)
                initial_bin = 1
            self.win.set_bin_options(bins, initial_bin)
            self.backend.set_bin(initial_bin)
            self._camera_bin = int(initial_bin)
        except Exception as e:
            logger.warning("Could not apply/query camera bins: %s", e)
            self.win.set_bin_options((1,), 1)
            self._camera_bin = 1

    def _set_state(self, state: str):
        self.session.state = state
        self.win.set_status(state)

        if state == self.PREVIEW:
            self.viewer.set_scanning_ui(False)
            self.win.set_scanning(False)
            self.win.set_bin_enabled(True)
            self.win.set_wavelength_range_enabled(True)
            self.win.dark_panel.set_interactive(True)

        elif state == self.SCANNING:
            self.viewer.set_scanning_ui(True)
            self.win.set_scanning(True)          # camera/stage settings locked and folded away
            self.win.set_bin_enabled(False)
            self.win.set_wavelength_range_enabled(False)
            self.win.dark_panel.set_interactive(False)
            if self.enable_cube_preview:
                self.win.cube_preview.reset()
                self.win.cube_preview.setVisible(True)
            else:
                self.win.cube_preview.setVisible(False)
            self.win.open_cube_viewer_btn.setEnabled(False)
            self.win.open_cube_viewer_btn.setText("Open Cube Viewer")
            self.win.open_calibration_btn.setEnabled(False)
            self.win.open_calibration_btn.setText("Open Calibration")
            self.win._pending_cube = None
            self.win._pending_wavelengths = None
            self.win._pending_spectrogram = None
            self.win._pending_spectrogram_metadata = None
            self.win.open_spectrogram_btn.setEnabled(False)
            self.win.open_spectrogram_btn.setText("Spectrogram Viewer")

        elif state == self.ERROR:
            self.viewer.set_scanning_ui(False)
            self.win.set_scanning(False)
            self.win.set_bin_enabled(True)
            self.win.set_wavelength_range_enabled(True)
            self.win.dark_panel.set_interactive(True)

    # ══════════════════════════════════════════════════════════════
    #  Focus ROI
    # ══════════════════════════════════════════════════════════════

    def on_line_y_changed(self, y: int):
        self.win.set_selected_y_label(int(y))

    def on_focus_roi_height_changed(self, value: int):
        self.viewer.set_focus_roi_height(int(value))

    def set_focus_roi_full_height(self):
        self.viewer.full_focus_roi_height()
        y1, y2 = self.viewer.focus_roi_range()
        self.win.set_focus_roi_height_value(y2 - y1)

    # ══════════════════════════════════════════════════════════════
    #  Dark frame
    # ══════════════════════════════════════════════════════════════

    # _current_dark_params
    def _current_dark_params(self) -> tuple[int, float, float]:
        return (
            self.win.selected_bin(),
            float(self.win.camera_controls.exposure_ms()),  
            float(self.win.camera_controls.gain()),        
        )

    def _load_dark_reference_for_current_params(self):
        self._dark_path = None
        try:
            bin_value, exposure_ms, gain = self._current_dark_params()
            info = self.backend.info
            selection = select_dark(self.dark_lib, bin_value, exposure_ms, gain,
                                    (info.height, info.width), temperature=self._last_temp)
            result = selection.result
            if selection.kind == "missing":
                self.dark_reference = None
                self.backend.set_dark_reference(None, None)
                self.win.dark_panel.set_idle()
                return

            dark_shape = selection.dark_shape
            if selection.kind == "mismatch":
                self.dark_reference = None
                self.backend.set_dark_reference(None, None)
                self.win.dark_panel.set_error(
                    f"Dark {dark_shape[1]}x{dark_shape[0]} ≠ camera {info.width}x{info.height}")
                logger.warning(
                    "Stored dark shape %s does not match camera frame %sx%s (bin=%d); "
                    "capture a dark for this camera", dark_shape, info.height, info.width, bin_value)
                return

            self.dark_reference = result
            self._dark_path = os.path.join(self.dark_lib.folder, result["filename"])
            n_hot = selection.n_hot
            self.win.dark_panel.set_loaded(
                bin_value=bin_value,
                exposure_ms=exposure_ms,
                gain=gain,
                n_frames=result["n_frames"],
                n_hot=n_hot,
                temperature_c=selection.dark_temperature,
                current_temperature_c=self._last_temp,
            )
            self.backend.set_dark_reference(
                result["mean"], result["hot_pixel_map"],
            )
            logger.info(
                "Dark loaded: %s bin=%d exp=%.1f gain=%.1f "
                "frames=%d shape=%s hot=%d dark temp=%s camera temp=%s",
                result["filename"], bin_value, exposure_ms, gain,
                result["n_frames"], result["mean"].shape, n_hot,
                selection.dark_temperature, selection.temperature,
            )
        except Exception as e:
            self.dark_reference = None
            self._dark_path = None
            self.backend.set_dark_reference(None, None)
            self.win.dark_panel.set_error("Load failed")
            logger.exception("Failed to load dark frame: %s", e)

    def _follow_dark_temperature(self):
        """In preview, switch to another stored dark once it is nearer the current temperature."""
        if (self.state != self.PREVIEW or self.dark_lib.is_collecting
                or self.dark_reference is None or self._dark_path is None):
            return
        try:
            bin_value, exposure_ms, gain = self._current_dark_params()
            nearest = self.dark_lib.find(bin_value, exposure_ms, gain, self._last_temp)
        except Exception as e:
            logger.debug("Dark lookup by temperature failed: %s", e)
            return
        if nearest is not None and os.path.normcase(nearest) != os.path.normcase(self._dark_path):
            logger.info("Sensor at %.1f C: switching dark %s -> %s", self._last_temp,
                        os.path.basename(self._dark_path), os.path.basename(nearest))
            self._load_dark_reference_for_current_params()

    # ══════════════════════════════════════════════════════════════
    #  Temperature / cooling
    # ══════════════════════════════════════════════════════════════

    def _poll_temperature(self, now: float):
        """Read the sensor temperature and cooler state every ``_temp_interval`` s (GUI thread)."""
        if now - self._last_temp_update < self._temp_interval:
            return
        self._last_temp_update = now
        try:
            state = self.backend.cooling_state()
        except Exception:
            state = None
        self.win.camera_controls.set_cooling_state(state)
        try:
            self._last_temp = self.backend.get_temperature()
        except Exception:
            return
        if self._last_temp > -900:
            self.win.set_temperature(self._last_temp)
        if self.state == self.SCANNING:
            temp = valid_temperature(self._last_temp)
            if temp is not None:
                self._scan_temps.append(temp)
        else:
            self._follow_dark_temperature()

    def _setup_cooling(self):
        """Show the temperature row as a target (TEC cameras) and apply the configured target."""
        controls = self.win.camera_controls
        try:
            supported = bool(self.backend.cooling_supported())
            value_range = self.backend.target_temperature_range() if supported else None
        except Exception as e:
            logger.warning("Could not query camera cooling: %s", e)
            supported, value_range = False, None
        controls.set_cooling_supported(supported, value_range)
        if not supported:
            return
        target = getattr(self.config, "target_temperature_c", None)
        if target is None:
            try:
                current = self.backend.get_target_temperature()
            except Exception:
                current = None
            if current is not None:
                controls.set_target_temperature(current)
            return
        controls.set_target_temperature(target)
        self.preview_thread.request_target_temperature(controls.target_temperature())

    def set_target_temperature(self, temp_c: float):
        """Switch the cooler on and regulate to ``temp_c`` °C (applied by the worker); remembered."""
        self.preview_thread.request_target_temperature(float(temp_c))
        self._schedule_settings_save("target_temperature_c")

    def _on_target_temperature_applied(self, temp_c: float):
        self.win.set_status(f"Cooler on, target {temp_c:.1f} °C")

    def _on_target_temperature_failed(self, message: str):
        self.win.set_status(f"Target temperature failed: {message}")

    def _scan_temperature_metadata(self) -> dict:
        """Sensor temperatures during the scan, the dark's temperature and the cooler target (°C)."""
        meta = {}
        temps = self._scan_temps
        if temps:
            meta.update(sensor_temperature_c_start=temps[0], sensor_temperature_c_end=temps[-1],
                        sensor_temperature_c_min=min(temps), sensor_temperature_c_max=max(temps))
        dark_temp = (self.dark_reference or {}).get("temperature")
        if dark_temp is not None:
            meta["dark_temperature_c"] = float(dark_temp)
        controls = getattr(self.win, "camera_controls", None)
        target = controls.target_temperature() if controls is not None else None
        if target is not None:
            meta["target_temperature_c"] = float(target)
        return meta

    #: Raw-metadata keys of a scan that also go into its cube's export header.
    _HEADER_SCAN_KEYS = ("started_at", "exposure_ms", "gain", "camera_bin", "stage_type",
                         "speed_um_s", "speed_um_s_achieved", "degree_per_second",
                         "degree_per_second_achieved")

    def _scan_header_metadata(self) -> dict:
        """Export-header metadata of the scan that just ended (see ``HypercubeWriter``):
        start time, exposure, gain, bin, stage speeds, temperatures, the corrections the
        frames went through and the frame rate. Missing values are left out."""
        raw = getattr(self, '_raw_metadata', None) or {}
        meta = {key: raw[key] for key in self._HEADER_SCAN_KEYS if raw.get(key) is not None}
        meta.update(self.last_scan_temperatures)
        try:
            meta.update(self.preview_thread.correction_info())
        except Exception:
            logger.debug("Correction state unavailable for the export header", exc_info=True)
        return meta

    # ══════════════════════════════════════════════════════════════
    #  Saved settings
    # ══════════════════════════════════════════════════════════════

    def _schedule_settings_save(self, key: str):
        """Mark one setting (a camera.json/scan.json key) as edited; it is written once edits pause."""
        if self._settings_timer is None:
            return
        self._dirty_settings = self._dirty_settings | {key}
        self._settings_timer.start()

    def _current_setting(self, key: str):
        """The value of one saved setting as the window shows it now."""
        controls = self.win.camera_controls
        if key == "default_gain":
            return int(controls.gain())
        if key == "default_exposure_ms":
            return float(controls.exposure_ms())
        if key == "default_bin":
            return int(self.backend.current_bin)
        if key == "target_temperature_c":
            return controls.target_temperature()
        return float(dict(zip(STAGE_SETTING_KEYS, self.win.stage_spins()))[key].value())

    def _save_settings(self):
        """Write only the settings edited since the last save (camera.json / scan.json).

        Writing just those keys means another copy of the app that is open on the same
        config cannot put back its own values of settings the user did not touch there.
        """
        store = getattr(self.config, "_config_store", None)
        dirty, self._dirty_settings = self._dirty_settings, frozenset()
        if store is None or self.backend is None or not dirty:
            return
        save_range = RANGE_SETTING_KEY in dirty
        values = {key: self._current_setting(key) for key in dirty - {RANGE_SETTING_KEY}}
        values = {key: value for key, value in values.items() if value is not None}
        camera = {k: v for k, v in values.items() if k not in STAGE_SETTING_KEYS}
        scan = {k: v for k, v in values.items() if k in STAGE_SETTING_KEYS}
        backend, saved = self._range_backend(), self._range_saved
        try:
            if camera:
                store.update_section("camera", camera)
            if scan:
                store.update_section("scan", scan)
            if save_range:
                # Only this camera's entry: the file keeps the other backends' ranges.
                store.update_section_entry("processing", RANGE_SETTING_KEY, backend,
                                           None if saved is None else [float(saved[0]), float(saved[1])])
        except Exception as e:
            logger.exception("Could not save camera/stage settings: %s", e)
            return
        if save_range:
            ranges = dict(getattr(self.config, RANGE_SETTING_KEY, None) or {})
            ranges.pop(backend, None)
            if saved is not None:
                ranges[backend] = tuple(saved)
            self.config.wavelength_range_nm = ranges
            values[RANGE_SETTING_KEY] = {backend: saved}
        logger.info("Settings saved for the next start: %s", values)
        for key, value in values.items():
            if key != RANGE_SETTING_KEY:
                setattr(self.config, key, value)

    def _refresh_dark_reference(self, collecting_message: str):
        """Look up the stored dark for the current bin/exposure/gain automatically.

        Falls back to clearing the reference while a dark capture is running.
        """
        if self.dark_lib.is_collecting:
            self.dark_reference = None
            self.backend.set_dark_reference(None, None)
            self.win.dark_panel.set_error(collecting_message)
            return
        self._load_dark_reference_for_current_params()

    def start_dark_capture(self):
        if self.state == self.SCANNING:
            return
        try:
            bin_value, exposure_ms, gain = self._current_dark_params()
            self.dark_lib.start_capture(
                bin=bin_value, exposure=exposure_ms, gain=gain,
            )
            self.dark_reference = None
            self.backend.set_dark_reference(None, None)
            self.preview_thread.set_dark_capture_mode(True)
            self.win.dark_panel.set_capturing(
                collected=0,
                total=self.dark_lib.n_frames,
                bin_value=bin_value,
                exposure_ms=exposure_ms,
                gain=gain,
            )
        except Exception as e:
            logger.exception("Failed to start dark capture: %s", e)
            self.win.dark_panel.set_error("Capture failed to start")

    def cancel_dark_capture(self):
        self.dark_lib.cancel()
        self.preview_thread.set_dark_capture_mode(False)
        self.win.dark_panel.set_idle()

    def _update_dark_capture(self, raw_frame: np.ndarray):
        if not self.dark_lib.is_collecting:
            return
        try:
            done = self.dark_lib.add_frame(raw_frame, temperature=self._last_temp)
            collected, target = self.dark_lib.progress
            self.win.dark_panel.update_progress(collected, target)
            if done:
                self._load_dark_reference_for_current_params()
                self.preview_thread.set_dark_capture_mode(False)
        except Exception as e:
            logger.exception("Dark capture failed: %s", e)
            self.dark_lib.cancel()
            self.preview_thread.set_dark_capture_mode(False)
            self.win.dark_panel.set_error("Capture failed")

    def toggle_dark_subtraction(self, enabled: bool):
        self.backend.set_dark_enabled(enabled)

    # ══════════════════════════════════════════════════════════════
    #  Camera params / bin
    # ══════════════════════════════════════════════════════════════

    def apply_params(self):
        now = time.time()
        if now - self.last_param_update < 0.2:
            return
        self.last_param_update = now
        self.preview_thread.request_params(
            self.win.camera_controls.gain(),                 
            self.win.camera_controls.exposure_ms(),
        )
        self._refresh_dark_reference("Camera params changed — dark cleared")

    def _on_bin_applied(self, bin_value: int):
        self._camera_bin = int(bin_value)
        self.win.camera_controls.set_selected_bin(bin_value)
        self.win.reset_sharpness_scale()
        self._refresh_dark_reference("Bin changed — dark cleared")
        self._schedule_settings_save("default_bin")      # bins change only on the user's request

    def _on_params_applied(self):
        # Sent by the worker before any sharpness score of the new frames.
        self.win.reset_sharpness_scale()

    def _on_bin_failed(self, requested: int, message: str):
        """The camera refused the bin: it keeps running on the previous one."""
        actual = int(self.backend.current_bin)
        logger.error("Bin %d not applied (still bin %d): %s", requested, actual, message)
        self._camera_bin = actual
        self.win.camera_controls.set_selected_bin(actual)
        self.win.set_status(f"Bin {requested} failed: {message}")
        self._refresh_dark_reference("Bin unchanged — dark cleared")

    def _on_params_failed(self, message: str):
        logger.error("Camera parameters not applied: %s", message)
        self.win.set_status(f"Camera parameters failed: {message}")

    def apply_bin(self, *_):
        if self.state == self.SCANNING:
            return
        try:
            new_bin = self.win.selected_bin()
            self.preview_thread.request_bin(new_bin)
            # The frame size changes only once the worker has applied the bin:
            # the matching dark is looked up in _on_bin_applied.
            self.dark_reference = None
            self.backend.set_dark_reference(None, None)
            self.win.dark_panel.set_error("Switching bin…")
        except Exception as e:
            logger.exception("Failed to request bin change: %s", e)

    # ══════════════════════════════════════════════════════════════
    #  Wavelength range
    # ══════════════════════════════════════════════════════════════

    def _range_backend(self) -> str:
        """Key of this camera in the per-backend ``wavelength_range_nm`` setting."""
        return str(self.config.camera_backend).lower()

    def _on_wavelength_range_edited(self, *_):
        """The user committed a range edit (toggle, Enter or focus leaving Start/End): queue it."""
        if self.preview_thread is None:
            return                              # window closing: cleanup already stopped the worker
        requested = self.win.requested_wavelength_range()
        if requested == self._range_requested:
            return                              # focus left without a change
        if self.state == self.SCANNING:
            self.win.set_wavelength_range(self._range_requested)
            return
        if requested is not None and not requested[0] < requested[1]:
            self.win.set_wavelength_range(self._range_requested)
            self.win.set_status("Wavelength range: Start must be below End")
            return
        self.preview_thread.request_wavelength_range(requested)
        self._range_requested = requested
        self._range_dirty = True

    def _on_wavelength_range_applied(self, info: dict):
        """The worker reports the bins in use (after a range change or any binner fit)."""
        self._range_applied = info["range_nm"]
        bands = info["bands"]                   # None without a range: full-grid refits change nothing here
        if bands != self._range_bands:
            if self._range_bands != ():
                self.win.reset_sharpness_scale()     # the score is a mean over the binned frame
            self._range_bands = bands
        if self._range_dirty:
            if info["range_nm"] == self._range_requested:
                self._range_dirty = False
                self._range_saved = info["range_nm"]
                self._schedule_settings_save(RANGE_SETTING_KEY)
            # else: an older range; the user's edit is still queued
        else:
            self._range_requested = info["range_nm"]
            if self.win.requested_wavelength_range() != info["range_nm"]:
                self.win.set_wavelength_range(info["range_nm"])
        self.win.set_wavelength_range_axis(info)     # after the check state: unchecked shows the full span

    def _on_wavelength_range_failed(self, requested, message: str):
        """A range was refused, or cleared because it no longer fits the camera setting."""
        self.win.set_status(f"Wavelength range not applied: {message}")
        if self._range_dirty and requested == self._range_requested:
            self._range_dirty = False
            self._range_requested = self._range_applied
            self.win.set_wavelength_range(self._range_applied)

    def _wavelength_range_metadata(self) -> dict:
        """Range provenance of the bins in use, for the scan metadata (empty without a range)."""
        reader = getattr(self.preview_thread, "wavelength_range_metadata", None)
        return reader() if reader is not None else {}

    # ══════════════════════════════════════════════════════════════
    #  Preview frame handling
    # ══════════════════════════════════════════════════════════════

    def on_preview_frame(
        self, frame_data: np.ndarray, fps: float, timestamp: float,
    ):
        if self.preview_thread is None:
            return

        now = time.time()
        self.preview_frame_index += 1

        if frame_data is None:
            return

        # Before the dark branch: dark capture records this temperature with its frames.
        self._poll_temperature(now)

        # ★ DARK 1st — raw frame เข้า dark_lib ตรง
        if self.dark_lib.is_collecting:
            self._update_dark_capture(frame_data)
            return

        # ── normal path ──

        h, w = frame_data.shape

        if self.win.focus_roi_height_spin.maximum() != h:
            self.win.focus_roi_height_spin.setMaximum(h)
            if self.win.focus_roi_height() > h:
                self.win.set_focus_roi_height_value(h)
                self.viewer.set_focus_roi_height(h)

        self._last_spatial_height = h
        self._refresh_reference_overlay(h)

        wl = self.preview_thread.binner_lambda_out
        if wl is not None and len(wl) == w:
            self.current_wavelengths = wl
        elif self.current_wavelengths is None or len(self.current_wavelengths) != w:
            self.current_wavelengths = np.arange(w, dtype=np.float32)

        if self.preview_frame_index == 1:
            logger.info(
                "Preview frame shape=%s dtype=%s min=%s max=%s",
                frame_data.shape, frame_data.dtype,
                frame_data.min(), frame_data.max(),
            )

        if self.enable_live_image and now - self.last_display_update > 0.03:
            self.viewer.set_image(frame_data)
            self.last_display_update = now

        self.win.set_fps(fps)
        self.win.set_frame_index(self.preview_frame_index)

        self._update_live_spectrum(frame_data, now)
        self._update_live_histogram(frame_data, now)

        if self.recording:
            # Cube frames are recorded per camera frame in the worker thread;
            # this (throttled) path only drives the live count and preview.
            self.win.set_recorded_count(self.preview_thread.recorded_count())

        if self.profile_frame_ms > 0:
            elapsed_ms = (time.monotonic() - timestamp) * 1000.0
            if elapsed_ms >= self.profile_frame_ms:
                logger.warning(
                    "Slow GUI frame: %.1f ms state=%s recording=%s "
                    "frames=%d fps=%.1f shape=%s",
                    elapsed_ms, self.state, self.recording,
                    len(self.session.images), fps, frame_data.shape,
                )

    def _on_saturation_ceiling(self, ceiling):
        self.saturation_ceiling = ceiling

    def _ceiling_for(self, frame_data: np.ndarray):
        ceiling = self.saturation_ceiling
        if ceiling is None or ceiling.shape != frame_data.shape[:2]:
            return None
        return ceiling

    def _update_live_spectrum(self, frame_data: np.ndarray, now: float):
        if frame_data is None or frame_data.ndim != 2:
            return

        spatial_h, _ = frame_data.shape
        selected_y = max(0, min(spatial_h - 1, self.viewer.selected_y()))
        self.viewer.set_selected_y(selected_y)

        if self.enable_live_spectrum and now - self.last_spectrum_update > 0.03:
            spectrum = spectrum_from_corrected_row(frame_data, selected_y)
            ceiling = self._ceiling_for(frame_data)
            self.win.live_line_spectrum.update_spectrum(
                spectrum, wavelengths=self.current_wavelengths,
                ceiling=None if ceiling is None else ceiling[selected_y],
            )
            self.last_spectrum_update = now

        self.win.set_selected_y_label(selected_y)

    def _on_sharpness_ready(self, score: float):
        self.win.set_sharpness_score(score)

    def _update_live_histogram(self, frame_data: np.ndarray, now: float):
        if (
            frame_data is None
            or frame_data.ndim != 2
            or not self.enable_histogram
            or now - self.last_histogram_update < HISTOGRAM_UPDATE_INTERVAL_SEC
        ):
            return
        self.last_histogram_update = now
        try:
            self.win.histogram.update_histogram(frame_data, ceiling=self._ceiling_for(frame_data))
        except Exception as e:
            logger.warning("Histogram update failed: %s", e)

    def on_preview_error(self, msg: str):
        logger.error("Preview error: %s", msg)
        self._set_state(self.ERROR)

    # ══════════════════════════════════════════════════════════════
    #  Scan control
    # ══════════════════════════════════════════════════════════════

    def start_scan(self):
        """Begin a scan sequence.

        Lifecycle:
            1. Guard — reject if dark capture is in progress
            2. Guard — reject unless currently in PREVIEW state, and while a
            wavelength range or bin change is still being applied (status message)
            3. Clear previous recording buffers (binned frames, saturation,
            full-res spectrograms)
            4. Arm preview thread for scan mode — full-res spectrogram capture
            is gated by the user checkbox (keep_spectrogram)
            5. Transition to SCANNING and start the motion stage

        The recording flag stays *False* until ``on_stage_started`` fires;
        this prevents the first frames from being captured before the stage
        has actually begun moving.
        """
        if self.dark_lib.is_collecting:
            self.win.dark_panel.set_error("Finish dark capture first")
            return
        if self.state != self.PREVIEW:
            return
        if self.stage is not None:
            logger.warning("Stage is still stopping or returning home")
            return
        if getattr(self.preview_thread, "wavelength_range_pending", False):
            # Started now, the scan would record the previous range (changes are refused while scanning).
            self.win.set_status("Wavelength range is being applied - press Start again")
            return
        if getattr(self.preview_thread, "bin_pending", False):
            # Started now, the bin would be applied mid-scan (two frame widths in one cube).
            self.win.set_status("Bin change is being applied - press Start again")
            return

        # ── reset buffers ─────────────────────────────────────
        self._raw_metadata = self._collect_raw_metadata()
        start_temp = valid_temperature(self._last_temp)
        self._scan_temps = [] if start_temp is None else [start_temp]
        self.session.begin()
        gc.collect()
        self.win.set_recorded_count(0)
        self._set_recording(False)

        # ── arm preview thread ────────────────────────────────
        self.preview_thread.set_scanning(True)
        self.preview_thread.set_keep_spectrogram(
            self.win.camera_controls.keep_spectrogram()
        )

        # ── state transition + stage launch ───────────────────
        self._set_state(self.SCANNING)
        self._create_and_start_stage()  

    def _collect_raw_metadata(self) -> dict:
        """Provenance of the scan that is starting (stored with the raw spectrogram)."""
        scan = {"stage_type": self.win.stage_type()}
        if scan["stage_type"] == "rotation":
            scan["degree"] = self.win.rotation_degree()
            scan["degree_per_second"] = self.win.rotation_speed_dps()
        else:
            scan["distance_mm"] = self.win.linear_distance_mm()
            scan["speed_um_s"] = self.win.linear_speed_um_s()
        try:
            return build_raw_metadata(backend=self.backend, config=self.config, scan=scan,
                                      binning=self.preview_thread.binning_info())
        except Exception:
            logger.exception("Could not collect scan metadata")
            return {}

    def _create_and_start_stage(self):
        try:
            stage_type = self.win.stage_type()
            self.stage = StageThread(stage_type=stage_type)
            self.stage.trim_extra_ms = float(getattr(self.config, "scan_trim_extra_ms", 0.0))
            self.stage.device = str(getattr(self.config, "stage_device", "") or "")

            if stage_type == "rotation":
                self.stage.degree_per_second = self.win.rotation_speed_dps()
                self.stage.degree = self.win.rotation_degree()
            elif stage_type == "linear":
                self.stage.distance_mm = self.win.linear_distance_mm()
                self.stage.speed_um_s = self.win.linear_speed_um_s()
            else:
                raise ValueError(f"Unsupported stage_type: {stage_type}")

            if hasattr(self.stage, "stagebefore"):
                self.stage.stagebefore.connect(self.on_stage_started)
            if hasattr(self.stage, "stageafter"):
                self.stage.stageafter.connect(self.on_stage_finished)

            self.stage.error.connect(self.on_stage_error)
            self.stage.returned_home.connect(self.on_stage_returned_home)
            self.stage.finished.connect(self.on_stage_thread_finished)

            if not hasattr(self.stage, "stagebefore"):
                self._set_recording(True)
                self.win.start_scan_clock()

            self.stage.start()

        except Exception as e:
            logger.exception("Failed to start stage: %s", e)
            self.on_stage_error(str(e))
            if self.stage is not None and not self.stage.isRunning():
                self.stage.deleteLater()
                self.stage = None

    def _set_recording(self, active: bool, start_ns=None):
        """Arm/disarm per-frame recording (GUI flag + worker buffer).

        ``start_ns``: host time before which the worker records no frame (the start of
        the stage's constant-speed window), or ``None``.
        """
        self.recording = bool(active)
        if self.preview_thread is not None:
            self.preview_thread.set_recording(self.recording, start_ns=start_ns)
            if self.recording:
                # The live preview reads the recording's own list while it grows.
                self._show_in_cube_preview(self.preview_thread.live_recorded_frames())

    def _disarm_stage_recording(self):
        self._set_recording(False)
        if self.preview_thread is not None:
            self.preview_thread.take_recorded_frames()  # discard
            self.preview_thread.take_recorded_raw()
        if self.preview_thread is not None:
            self.preview_thread.set_scanning(False)
            self.preview_thread.set_keep_spectrogram(False)

    def on_stage_started(self):
        if self.state == self.SCANNING:
            # The stage noted its move start before signalling, so frames taken while
            # it accelerates are not recorded at all (nor shown in the live preview).
            start_ns = self.stage.steady_window()[0] if self.stage is not None else None
            self._set_recording(True, start_ns)
            # status line: elapsed / expected move time, at the speed the stage runs
            self.win.start_scan_clock(self._note_achieved_speed())

    def _note_achieved_speed(self):
        """Store the scan speed the stage runs (read back from the controller once it set the
        scan speed) in the scan metadata next to the requested one, and return it (µm/s for
        the linear stage, deg/s for rotation), or ``None`` if the stage did not report it."""
        if self.stage is None:
            return None
        key = "speed_um_s_achieved" if self.stage.stage_type == "linear" else "degree_per_second_achieved"
        achieved = getattr(self.stage, key, None)
        if achieved is not None and isinstance(getattr(self, '_raw_metadata', None), dict):
            self._raw_metadata[key] = achieved
        return achieved

    def on_stage_finished(self):
        # v1 behavior: build the cube as soon as the scan move ends (endpoint or
        # manual stop). Return-home continues in the stage thread; start_scan
        # stays blocked until it has finished.
        self.finish_scan()
        if self.stage is not None and self.stage.isRunning():
            self.win.set_status("Cube ready — stage returning home")

    def on_stage_returned_home(self):
        self.win.set_status("Stage returned home")

    def on_stage_error(self, message):
        logger.error("Stage failure: %s", message)
        if self.state == self.SCANNING:
            # Hardware failed before the scan move ended: motion is unknown,
            # so the partial recording is not published as a cube.
            self._disarm_stage_recording()
            self._set_state(self.PREVIEW)
            self.win.set_status(f"Stage error — scan discarded: {message}")
            request_id = self.session.fail()
            if request_id is not None and self.reference_preview_failed is not None:
                self.reference_preview_failed(request_id, message)
            QMessageBox.critical(self.win, "Stage Error", f"Scan discarded:\n{message}")
        else:
            # Failure after the cube was built (e.g. return-home): data stays.
            self.win.set_status(f"Stage error (scan data kept): {message}")
            QMessageBox.critical(self.win, "Stage Error", f"Scan data kept:\n{message}")

    def on_stage_thread_finished(self):
        stage = self.stage
        if stage is None:
            return
        logger.info("Stage thread finished: completed=%s failed=%s cancelled=%s",
                    stage.completed, stage.failed, stage.cancelled)
        if self.state == self.SCANNING:
            # Thread ended without stageafter (e.g. stopped while homing).
            if stage.failed:
                self.on_stage_error("Stage failed; partial scan was not published")
            else:
                self.finish_scan()
        self.stage = None
        stage.deleteLater()

    def stop_scan_manually(self):
        """v1 behavior: build the cube from the frames recorded so far, then
        let the stage return home in the background."""
        if self.state == self.SCANNING:
            if self.stage is not None:
                self.stage.request_stop()
            self.finish_scan()
            if self.stage is not None and self.stage.isRunning():
                self.win.set_status("Scan stopped — stage returning home")

    def _show_in_cube_preview(self, frames, shift=0):
        """Point the real-time cube preview at ``frames`` (read in place, never copied)."""
        if not self.enable_cube_preview:
            return
        wl = self.current_wavelengths
        self.win.cube_preview.set_frame_source(frames, wl.copy() if wl is not None else None, shift)

    def finish_scan(self):
        """Finalise a scan sequence and build the pushbroom cube.

        Lifecycle:
            1. Guard — only valid while in SCANNING state
            2. Disarm preview thread (stop spectrogram capture + recording)
            3. Transition back to PREVIEW
            4. Build the 3-D data cube from collected binned frames
            5. Attach wavelength axis (falls back to sequential indices
            if the LambdaBinner has not been fitted)
            6. Push saturation overlay and log any saturated frames

        The full-res spectrogram buffer (``_spectrograms``) is intentionally
        *not* cleared here so that downstream consumers can read it after
        the cube is ready.  It is cleared at the beginning of the next
        scan via ``start_scan``.
        """
        if self.state != self.SCANNING:
            return

        # ── disarm preview thread ─────────────────────────────
        range_metadata = self._wavelength_range_metadata()     # the bins this scan recorded
        self.preview_thread.set_scanning(False)
        self.preview_thread.set_keep_spectrogram(False)
        # Keep only frames taken while the stage moved at constant speed (camera
        # timestamps, GenICam); a manual stop has no end time yet, so only the start
        # is trimmed then.
        window = self.stage.steady_window() if self.stage is not None else (None, None)
        images = self.preview_thread.take_recorded_frames(
            *window, reference_ns=getattr(self.stage, 'motion_start_ns', None))
        raw = self.preview_thread.take_recorded_raw()
        self.win.set_recorded_count(len(images))     # the frames kept for the cube (after the trim)

        # ── temperatures of this scan (cube header / spectrogram metadata) ──
        try:
            end_temp = valid_temperature(self.backend.get_temperature())
        except Exception:
            end_temp = None
        if end_temp is not None:
            self._scan_temps = [*self._scan_temps, end_temp]
        self.last_scan_temperatures = self._scan_temperature_metadata()
        if isinstance(getattr(self, '_raw_metadata', None), dict):
            self._raw_metadata.update(self.last_scan_temperatures)
            self._raw_metadata.update(range_metadata)
        self.last_scan_header = self._scan_header_metadata()

        # ── state transition ──────────────────────────────────
        self._set_state(self.PREVIEW)
        if self.session.reference_preview_id is not None:
            # One column per recorded frame (the frames a cube would get), not per displayed frame.
            for frame in images:
                self.session.add_reference_column(select_rgb(frame, self.current_wavelengths))
        outcome = self.session.finish(images, raw)

        if outcome.kind in ("reference", "reference_failed"):
            # Visual-selection scan: publish a preview image, never a cube.
            self._publish_reference_outcome(outcome)
            return

        # The preview now shows the cube's own (trimmed) list; points and bands stay.
        recorder = getattr(self.preview_thread, 'recorder', None)
        self._show_in_cube_preview(images, recorder.last_trim[0] if recorder is not None else 0)

        if outcome.kind == "empty":
            logger.warning("No frames recorded")
            return

        try:
            # ── build data cube ───────────────────────────────
            cube = create_pushbroom_cube_from_corrected(self.session.images)

            # ── wavelength axis ───────────────────────────────
            wavelengths = self.preview_thread.binner_lambda_out
            if wavelengths is None and self.current_wavelengths is not None:
                wavelengths = self.current_wavelengths.copy()

            self.win.set_cube_ready(cube, wavelengths)

            if self.session.spectrograms:
                raw_stack = np.stack(self.session.spectrograms, axis=0)
                self.session.spectrograms = []  # the stacked copy is the only holder now
                self.win.set_spectrogram_ready(raw_stack, getattr(self, '_raw_metadata', None))
            # ── saturation report ─────────────────────────────
            self._push_sat_overlay()
            n_sat = sum(1 for s in self.session.sat_per_frame if s.any())
            if n_sat > 0:
                logger.warning(
                    "Saturation: %d/%d frames affected",
                    n_sat, len(self.session.sat_per_frame),
                )

        except Exception as e:
            logger.exception("Failed to create cube: %s", e)
            self._set_state(self.ERROR)

    def _publish_reference_outcome(self, outcome):
        """Deliver a visual-selection scan result to the host callbacks."""
        if outcome.kind == "reference_failed":
            if self.reference_preview_failed is not None:
                self.reference_preview_failed(outcome.request_id, outcome.error)
            return
        try:
            image = compose_preview(outcome.reference_columns)
            context = self.reference_roi_context(outcome.reference_columns[0].shape[0])
        except Exception as e:
            if self.reference_preview_failed is not None:
                self.reference_preview_failed(outcome.request_id, str(e))
            return
        if self.reference_preview_ready is not None:
            self.reference_preview_ready(outcome.request_id, image, context)

    # ══════════════════════════════════════════════════════════════
    #  Saturation
    # ══════════════════════════════════════════════════════════════

    def _on_saturation(self, sat_line: np.ndarray):
        if not self.recording:
            return
        self.session.sat_per_frame.append(sat_line)
        self._push_sat_overlay()

    def _push_sat_overlay(self):
        if not self.session.sat_per_frame:
            return
        sat_map = np.stack(self.session.sat_per_frame, axis=0)
        self.win.cube_preview.set_saturation_mask(sat_map)

    # ══════════════════════════════════════════════════════════════
    #  Cleanup
    # ══════════════════════════════════════════════════════════════

    def cleanup(self):
        logger.info("ScanController cleanup started")
        self.session.recording = False

        if self._settings_timer is not None and self._settings_timer.isActive():   # edits in the last second
            self._settings_timer.stop()
            self._save_settings()

        try:
            if self.dark_lib.is_collecting:
                self.dark_lib.cancel()
                self.preview_thread.set_dark_capture_mode(False)
        except Exception as e:
            logger.warning("Dark cancel error: %s", e)

        try:
            if self.stage is not None and self.stage.isRunning():
                # Shutdown: abort without return-home so the window can close.
                self.stage.request_stop(return_home=False)
                if not self.stage.wait(3000):
                    logger.error("Stage still running; window close deferred")
                    return False
                self.stage = None
        except Exception as e:
            logger.warning("Stage cleanup error: %s", e)

        try:
            if self.preview_thread is not None:
                if not self.preview_thread.stop(timeout_ms=2000):
                    # Worker still owns the camera; its finally block closes it.
                    logger.error("Preview thread still running; window close deferred")
                    return False
                self.preview_thread = None
        except Exception as e:
            logger.warning("Preview thread cleanup error: %s", e)

        try:
            if self.backend is not None:
                self.backend.close()
                self.backend = None
        except Exception as e:
            logger.warning("Backend close error: %s", e)

        logger.info("ScanController cleanup done")
        return True