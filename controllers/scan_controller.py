# controllers/scan_controller.py
"""Scan controller — Orin-compatible, binning in worker thread.

Pipeline:
    Worker:  backend.read_frame() → orient → LambdaBin → emit(30fps)
    GUI:     display + spectrum + histogram + record (append only)

Dark capture:
    Worker:  backend.read_frame_raw() → emit(raw, no bin) — dark_capture_mode
    GUI:     add to dark library at raw resolution
"""

from __future__ import annotations

import logging
import os
import time
import gc

import numpy as np

from QTNarit import StageThread
from workers import DarkFrameLibrary

from config import AppConfig
from utils.wavelength import spectrum_from_corrected_row
from utils.cube import create_pushbroom_cube_from_corrected
from workers.smile_calibrator import SmileCalibrator
from workers.smile_corrector import SmileCorrector

MAX_RECORD_FRAMES = 10000
HISTOGRAM_UPDATE_INTERVAL_SEC = 0.25

logger = logging.getLogger(__name__)


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}


class ScanController:
    PREVIEW = "PREVIEW"
    SCANNING = "SCANNING"
    ERROR = "ERROR"

    def __init__(self, win, config: AppConfig):
        self.win = win
        self.viewer = win.viewer
        self.config = config
        self._sat_per_frame: list[np.ndarray] = []
        self._spectrograms: list[np.ndarray] = []
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

        # ── smile correction ─────────────────────────────────────
        self._smile_corrector = SmileCorrector(
            folder=self.config.smile_map_folder
        )
        self._smile_calibrator = SmileCalibrator(
            output_folder=self.config.smile_map_folder
        )

        # ── camera backend factory ───────────────────────────────
        backend_type = str(
            getattr(self.config, 'camera_backend', 'poa')
        ).lower()

        if backend_type == "genicam":
            from workers.backends.genicam import GenICamBackend
            self.backend = GenICamBackend(
                camera_index=getattr(self.config, 'camera_index', 0),
                gain=self.config.default_gain,
                exposure_ms=self.config.default_exposure_ms,
                cti_path=getattr(self.config, 'cti_path', None),
                default_bin=self.config.default_bin,
            )
        else:
            from workers.backends.poa_camera import POACameraBackend
            self.backend = POACameraBackend(
                gain=self.config.default_gain,
                exposure_ms=self.config.default_exposure_ms,
            )

        self.backend.open()

        # ── preview thread ───────────────────────────────────────
        from workers.camera_preview import CameraPreviewThread

        self.preview_thread = CameraPreviewThread(self.backend, config=config)
        self.preview_thread.frame_ready.connect(self.on_preview_frame)
        self.preview_thread.spectrogram_ready.connect(self._on_spectrogram)
        self.preview_thread.sharpness_ready.connect(self._on_sharpness_ready)
        self.preview_thread.error.connect(self.on_preview_error)
        self.preview_thread.saturation_ready.connect(self._on_saturation)
        self.preview_thread.set_raw_inverted(self.config.raw_inverted)

        self.stage: StageThread | None = None

        self.state = self.PREVIEW
        self.recording = False
        self.images: list[np.ndarray] = []

        self.preview_frame_index = 0

        self.last_display_update = 0.0
        self.last_param_update = 0.0
        self.last_spectrum_update = 0.0
        self.last_histogram_update = 0.0
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

        self._apply_backend_limits()
        self._apply_bin_options_and_initial_bin()
        self._try_set_camera_bit_depth()
        self._connect_ui()
        self._set_state(self.PREVIEW)

        self._load_dark_reference_for_current_params()
        self._load_smile_maps()
        self.apply_params()
        self.preview_thread.start()

    # ══════════════════════════════════════════════════════════════
    #  Smile correction
    # ══════════════════════════════════════════════════════════════

    def _load_smile_maps(self) -> None:
        if self._smile_corrector.set_bin(self._camera_bin):
            logger.info(
                "Smile maps loaded for bin%d from %s",
                self._camera_bin, self.config.smile_map_folder,
            )
        else:
            logger.info("No smile maps for bin%d", self._camera_bin)

    def calibrate_smile(self, frame: np.ndarray | None = None) -> bool:
        if frame is None:
            frame = self._preview_display_frame
        if frame is None:
            logger.warning("Smile calibration: no frame available")
            return False

        success = self._smile_calibrator.calibrate(
            frame, bin_value=self._camera_bin
        )
        if success:
            self._smile_corrector.set_bin(self._camera_bin)
            logger.info("Smile calibration complete for bin%d", self._camera_bin)
        else:
            logger.error("Smile calibration failed for bin%d", self._camera_bin)
        return success

    # ══════════════════════════════════════════════════════════════
    #  Properties
    # ══════════════════════════════════════════════════════════════

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
            if bd is not None:
                self.win.display_controls.set_bit_depth(int(bd))
                self.display_mapper = self.win.display_controls.current_mapper()
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
            self._load_smile_maps()
        except Exception as e:
            logger.warning("Could not apply/query camera bins: %s", e)
            self.win.set_bin_options((1,), 1)
            self._camera_bin = 1

    def _set_state(self, state: str):
        self.state = state
        self.win.set_status(state)

        if state == self.PREVIEW:
            self.viewer.set_scanning_ui(False)
            self.win.set_bin_enabled(True)
            self.win.dark_panel.set_interactive(True)

        elif state == self.SCANNING:
            self.viewer.set_scanning_ui(True)
            self.win.set_bin_enabled(False)
            self.win.dark_panel.set_interactive(False)
            if self.enable_cube_preview:
                self.win.cube_preview.reset()
                self.win.cube_preview.setVisible(True)
            else:
                self.win.cube_preview.setVisible(False)
            self.win.open_cube_viewer_btn.setEnabled(False)
            self.win.open_cube_viewer_btn.setText("Open Cube Viewer")
            self.win._pending_cube = None
            self.win._pending_wavelengths = None
            self.win._pending_spectrogram = None
            self.win.open_spectrogram_btn.setEnabled(False)
            self.win.open_spectrogram_btn.setText("Spectrogram Viewer")

        elif state == self.ERROR:
            self.viewer.set_scanning_ui(False)
            self.win.set_bin_enabled(True)
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
        try:
            bin_value, exposure_ms, gain = self._current_dark_params()
            result = self.dark_lib.load(
                bin=bin_value, exposure=exposure_ms, gain=gain,
            )
            if result is None:
                self.dark_reference = None
                self.backend.set_dark_reference(None, None)
                self.win.dark_panel.set_idle()
                return

            self.dark_reference = result
            n_hot = int(np.count_nonzero(result["hot_pixel_map"]))
            self.win.dark_panel.set_loaded(
                bin_value=bin_value,
                exposure_ms=exposure_ms,
                gain=gain,
                n_frames=result["n_frames"],
                n_hot=n_hot,
            )
            self.backend.set_dark_reference(
                result["mean"], result["hot_pixel_map"],
            )
            logger.info(
                "Dark loaded: bin=%d exp=%.1f gain=%.1f "
                "frames=%d shape=%s hot=%d",
                bin_value, exposure_ms, gain,
                result["n_frames"], result["mean"].shape, n_hot,
            )
        except Exception as e:
            self.dark_reference = None
            self.backend.set_dark_reference(None, None)
            self.win.dark_panel.set_error("Load failed")
            logger.exception("Failed to load dark frame: %s", e)

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
            done = self.dark_lib.add_frame(raw_frame)
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
        self.dark_reference = None
        self.backend.set_dark_reference(None, None)
        self.win.dark_panel.set_error("Camera params changed — reload dark")

    def apply_bin(self, *_):
        if self.state == self.SCANNING:
            return
        try:
            new_bin = self.win.selected_bin()
            self.preview_thread.request_bin(new_bin)
            self.dark_reference = None
            self.backend.set_dark_reference(None, None)
            self.win.dark_panel.set_error("Bin changed — reload dark")
            self._camera_bin = int(new_bin)
        except Exception as e:
            logger.exception("Failed to request bin change: %s", e)

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

        # ★ DARK 1st — raw frame เข้า dark_lib ตรง
        if self.dark_lib.is_collecting:
            self._update_dark_capture(frame_data)
            return

        # ── normal path ──
        frame_data = self._smile_corrector.apply(frame_data)

        h, w = frame_data.shape

        if self.win.focus_roi_height_spin.maximum() != h:
            self.win.focus_roi_height_spin.setMaximum(h)
            if self.win.focus_roi_height() > h:
                self.win.set_focus_roi_height_value(h)
                self.viewer.set_focus_roi_height(h)

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

        if now - self._last_temp_update >= self._temp_interval:
            self._last_temp_update = now
            try:
                self._last_temp = self.backend.get_temperature()
                if self._last_temp > -900:
                    self.win.set_temperature(self._last_temp)
            except Exception:
                pass

        self._update_live_spectrum(frame_data, now)
        self._update_live_histogram(frame_data, now)

        if self.recording and len(self.images) < MAX_RECORD_FRAMES:
            self.images.append(frame_data)
            self.win.set_recorded_count(len(self.images))
            bands = frame_data.shape[1]
            if self.enable_cube_preview and self.current_wavelengths is not None:
                self.win.cube_preview.push_frame(
                    frame_data, self.current_wavelengths.copy(), bands,
                )

        if self.profile_frame_ms > 0:
            elapsed_ms = (time.perf_counter() - timestamp) * 1000.0
            if elapsed_ms >= self.profile_frame_ms:
                logger.warning(
                    "Slow GUI frame: %.1f ms state=%s recording=%s "
                    "frames=%d fps=%.1f shape=%s",
                    elapsed_ms, self.state, self.recording,
                    len(self.images), fps, frame_data.shape,
                )

    def _update_live_spectrum(self, frame_data: np.ndarray, now: float):
        if frame_data is None or frame_data.ndim != 2:
            return

        spatial_h, _ = frame_data.shape
        selected_y = max(0, min(spatial_h - 1, self.viewer.selected_y()))
        self.viewer.set_selected_y(selected_y)

        if self.enable_live_spectrum and now - self.last_spectrum_update > 0.03:
            spectrum = spectrum_from_corrected_row(frame_data, selected_y)
            self.win.live_line_spectrum.update_spectrum(
                spectrum, wavelengths=self.current_wavelengths,
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
            self.win.histogram.update_histogram(frame_data)
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
            2. Guard — reject unless currently in PREVIEW state
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

        # ── reset buffers ─────────────────────────────────────
        self.images.clear()
        self._sat_per_frame = []
        self._spectrograms = []
        gc.collect()
        self.win.set_recorded_count(0)
        self.recording = False

        # ── arm preview thread ────────────────────────────────
        self.preview_thread.set_scanning(True)
        self.preview_thread.set_keep_spectrogram(
            self.win.camera_controls.keep_spectrogram()
        )

        # ── state transition + stage launch ───────────────────
        self._set_state(self.SCANNING)
        self._create_and_start_stage()  

    def _create_and_start_stage(self):
        try:
            stage_type = self.win.stage_type()
            self.stage = StageThread(stage_type=stage_type)

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

            self.stage.finished.connect(self.on_stage_thread_finished)

            if not hasattr(self.stage, "stagebefore"):
                self.recording = True

            self.stage.start()

        except Exception as e:
            logger.exception("Failed to start stage: %s", e)
            self.recording = False
            self._set_state(self.ERROR)

    def on_stage_started(self):
        self.recording = True

    def on_stage_finished(self):
        self.finish_scan()

    def on_stage_thread_finished(self):
        logger.info("Stage thread finished")

    def stop_scan_manually(self):
        if self.state == self.SCANNING:
            if self.stage is not None:
                self.stage.request_stop()
            self.finish_scan()

    def _on_spectrogram(self, full_frame):
        if self.recording:
            self._spectrograms.append(full_frame)

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
        self.preview_thread.set_scanning(False)
        self.preview_thread.set_keep_spectrogram(False)
        self.recording = False

        # ── state transition ──────────────────────────────────
        self._set_state(self.PREVIEW)

        if not self.images:
            logger.warning("No frames recorded")
            return

        try:
            # ── build data cube ───────────────────────────────
            cube = create_pushbroom_cube_from_corrected(self.images)

            # ── wavelength axis ───────────────────────────────
            wavelengths = self.preview_thread.binner_lambda_out
            if wavelengths is None and self.current_wavelengths is not None:
                wavelengths = self.current_wavelengths.copy()

            self.win.set_cube_ready(cube, wavelengths)

            if self._spectrograms:
                self.win.set_spectrogram_ready(
                    np.stack(self._spectrograms, axis=0)
                )
            # ── saturation report ─────────────────────────────
            self._push_sat_overlay()
            n_sat = sum(1 for s in self._sat_per_frame if s.any())
            if n_sat > 0:
                logger.warning(
                    "Saturation: %d/%d frames affected",
                    n_sat, len(self._sat_per_frame),
                )

        except Exception as e:
            logger.exception("Failed to create cube: %s", e)
            self._set_state(self.ERROR)

    # ══════════════════════════════════════════════════════════════
    #  Saturation
    # ══════════════════════════════════════════════════════════════

    def _on_saturation(self, sat_line: np.ndarray):
        if not self.recording:
            return
        self._sat_per_frame.append(sat_line)
        self._push_sat_overlay()

    def _push_sat_overlay(self):
        if not self._sat_per_frame:
            return
        sat_map = np.stack(self._sat_per_frame, axis=0)
        self.win.cube_preview.set_saturation_mask(sat_map)

    # ══════════════════════════════════════════════════════════════
    #  Cleanup
    # ══════════════════════════════════════════════════════════════

    def cleanup(self):
        logger.info("ScanController cleanup started")
        self.recording = False

        try:
            if self.dark_lib.is_collecting:
                self.dark_lib.cancel()
                self.preview_thread.set_dark_capture_mode(False)
        except Exception as e:
            logger.warning("Dark cancel error: %s", e)

        try:
            if self.stage is not None and self.stage.isRunning():
                if not self.stage.wait(3000):
                    self.stage.terminate()
                    self.stage.wait(500)
                self.stage = None
        except Exception as e:
            logger.warning("Stage cleanup error: %s", e)

        try:
            if self.preview_thread is not None:
                self.preview_thread.stop(timeout_ms=2000)
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