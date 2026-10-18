"""Qt-free acquisition engine: fetch -> preprocess -> spectral bin -> record -> deliver.

Pipeline per frame (worker thread)::

    backend.read_frame_raw()                      (raw sensor frame)
      -> dark subtraction, hot-pixel repair      \\
      -> orientation (raw_inverted)               |  modules.preprocessing
      -> Smile correction (if a model is loaded) /
      -> spectral binning (LambdaBinner)          (modules.spectral_processing)
      -> record every frame while armed           (ScanRecorder)
      -> sharpness every 0.2 s, display frame at <= 30 fps  -> sink

During dark capture the frame skips corrections and binning and is delivered raw.

Data conventions:
    Frames are ``(slit, spectral)`` ``uint16``. Binned frames are
    ``(slit, wavelength_bin)``. Exposure is in milliseconds.

Threading:
    :meth:`AcquisitionEngine.run` blocks on one worker thread, which is the only thread
    that touches the camera. All ``request_*`` and ``set_*`` methods may be called from
    any other thread (typically the GUI); they only queue a change that the worker
    applies at a frame boundary. Results are delivered to an :class:`AcquisitionSink`
    **from the worker thread**; Qt applications use
    :class:`modules.acquisition.CameraPreviewThread`, which turns them into queued
    signals (display frames into a newest-only mailbox the GUI pulls from).

Example:
    >>> from modules.acquisition import AcquisitionEngine, AcquisitionSink
    >>> class Print(AcquisitionSink):
    ...     def frame(self, frame, fps, timestamp):
    ...         print(frame.shape, round(fps))
    >>> # engine = AcquisitionEngine(camera, config, sink=Print())      # doctest: +SKIP
    >>> # threading.Thread(target=engine.run_until_stopped).start()     # doctest: +SKIP
    >>> # ...; engine.request_stop()                                    # doctest: +SKIP
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from modules.dark import DarkProcessor
from modules.frame_exchange.snapshot import FrameSnapshot
from modules.preprocessing import PreprocessingRuntime
from modules.sharpness import ImageSharpness
from modules.smile.runtime import SmileRuntime
from modules.spectral_processing import LambdaBinner

from .profiler import StageProfiler
from .recorder import ScanRecorder

logger = logging.getLogger(__name__)

MAX_GUI_FPS = 30
MIN_FRAME_INTERVAL = 1.0 / MAX_GUI_FPS
SHARPNESS_INTERVAL = 0.2
#: Narrowest bin derived from a stored FWHM, in spectral pixels (closer to one pixel the
#: bins become uneven and some end up empty).
MIN_FWHM_STEP_PX = 2.0


class AcquisitionSink:
    """Receiver of engine results. Override what you need; every method is a no-op by default.

    All methods are called on the acquisition worker thread and must return quickly.
    """

    def frame(self, frame: np.ndarray, fps: float, timestamp: float) -> None:
        """A display frame (<= 30 per second): binned, or raw during dark capture."""

    def saturation_ceiling(self, ceiling) -> None:
        """The value a clipped sensor pixel shows in the display frames, or ``None`` if unknown.

        Same shape as the display frames: a full-scale sensor frame run through the current
        dark subtraction, orientation, Smile and spectral binning (so it is ``full scale -
        dark`` where dark is applied). Sent before the first frame it applies to, and again
        only when it changes.
        """

    def cube_frames(self, frames: list) -> None:
        """A batch of every frame recorded since the previous batch (by reference)."""

    def correction_status(self, status: dict) -> None:
        """``{'dark': (state, detail), 'smile': (state, detail)}``; sent only when it changes."""

    def sharpness(self, score: float) -> None:
        """Focus score of the latest frame (about 5 Hz)."""

    def snapshot(self, packet) -> None:
        """A :class:`~modules.frame_exchange.FrameSnapshot` answering ``request_snapshot``."""

    def processed_frame(self, packet) -> None:
        """A :class:`~modules.frame_exchange.FrameSnapshot` answering ``request_processed_frame``.

        ``image`` is the preprocessed frame before spectral binning; ``stage`` is ``'post_smile'``
        when a Smile calibration was applied, else ``'post_dark_hot_pixel_orientation'``.
        """

    def error(self, message: str) -> None:
        """A failure; fatal ones end the loop."""

    def bin_applied(self, bin_value: int) -> None:
        """The camera now runs at this hardware bin."""

    def bin_failed(self, requested: int, message: str) -> None:
        """The camera refused ``requested``; the previous bin was restored."""

    def params_applied(self) -> None:
        """Gain/exposure were applied; the frames (and sharpness scores) after this use them."""

    def params_failed(self, message: str) -> None:
        """Gain/exposure could not be applied."""

    def target_temperature_applied(self, temp_c: float) -> None:
        """The cooler is on and regulating to ``temp_c`` °C."""

    def target_temperature_failed(self, message: str) -> None:
        """The target temperature could not be set."""

    def frame_rate(self, hz: float) -> None:
        """The camera's ``AcquisitionFrameRate`` (Hz), sent at start and after every bin/exposure change."""

    def wavelength_applied(self, profile) -> None:
        """A wavelength calibration took effect from the next frame."""

    def wavelength_failed(self, profile, message: str) -> None:
        """A wavelength calibration was rejected."""


class AcquisitionEngine:
    """Run one camera through the processing pipeline and report to a sink.

    Args:
        backend: A started-on-demand camera backend (see :mod:`modules.camera`); the
            engine opens it, streams it and closes it when the loop ends.
        config: Object with ``smile_map_folder``, ``preprocessing_backend`` (optional),
            ``m``, ``c`` (nm/pixel at bin 1, nm) and ``resolution`` (nm per output bin).
            A wavelength calibration (applied, or found by :meth:`set_wavelength_lookup`)
            updates ``config.m`` and ``config.c``. With ``resolution_source = "fwhm"`` the
            bin width comes from :meth:`set_resolution_lookup` instead (see :meth:`binning_info`).
        sink: Where results go; defaults to a no-op sink.
        profile: Log running per-stage timings (see :class:`StageProfiler`).
    """

    def __init__(self, backend, config=None, sink: AcquisitionSink | None = None, profile: bool = True):
        self.backend = backend
        self.config = config
        self.sink = sink if sink is not None else AcquisitionSink()
        self._alive = False

        # Scan recording: every binned frame is stored by the worker, independent of the
        # display throttle (see ScanRecorder).
        self.recorder = ScanRecorder()
        self._keep_spectrogram: bool = False
        self._scanning: bool = False

        self._stop_event = threading.Event()
        if hasattr(backend, 'set_stop_event'):
            backend.set_stop_event(self._stop_event)

        self._lock = threading.Lock()
        self._pending_params: tuple[int, float] | None = None
        self._pending_target_temperature: float | None = None
        self._pending_bin: int | None = None
        self._pending_wavelength = None
        self._pending_snapshot = None
        self._pending_processed = False
        self._pending_smile_model = None
        self._last_correction_status = None
        self._wavelength_lookup = None
        self._resolution_lookup = None
        self._fwhm_nm = None                      # FWHM stored for the current setting (nm), or None
        self._pending_resolution_reload = False
        self._smile = SmileRuntime(config.smile_map_folder)
        self._preprocessor = PreprocessingRuntime(getattr(config, 'preprocessing_backend', 'auto'))
        self._shared_preprocessing = getattr(getattr(type(backend), 'read_frame', None), '__qualname__', '') == 'AbstractCameraBackend.read_frame'
        self._passthrough_dark = DarkProcessor()
        self._passthrough_dark.set_dark_enabled(False)

        self._last_frame_time: float | None = None
        self._intervals: list[float] = []
        self._last_emit_time: float = 0.0

        self._raw_inverted: bool = False

        # spectral binning
        self._binner = None
        self._binner_fitted = False
        self._binner_lock = threading.Lock()

        self._dark_capture_mode: bool = False

        # saturation ceiling: key of the last one sent, and the objects whose ids are in it
        # (held so a freed reference cannot hand its id to a new one)
        self._ceiling_key = ()
        self._ceiling_refs = None

        self._emitted = 0
        self._skipped = 0

        self._camera_hz: float | None = None      # last AcquisitionFrameRate read from the camera
        self._clock_offset_ns: int | None = None  # camera ns - host perf_counter_ns, latched on arming
        self._clock_sync_pending = False
        self._last_frame_id: int | None = None
        self._frame_shape: tuple | None = None    # (H, W) of the last frame, for the profile header
        self._profiler = StageProfiler(enabled=profile, log_every=90,
                                       label=self._profile_label, shown_stage="emit")

    # ── lifecycle ─────────────────────────────────────────

    def _prepare_start(self):
        """Reset counters and arm the loop; call before :meth:`run` starts on its thread."""
        self._stop_event.clear()
        self._alive = True
        self._last_frame_time = None
        self._intervals.clear()
        self._last_emit_time = 0.0
        self._emitted = 0
        self._skipped = 0

    def request_stop(self):
        """Ask the loop to end; it closes the camera on its way out. Any thread."""
        self._alive = False
        self._stop_event.set()

    def run_until_stopped(self):
        """Prepare and run the loop on the calling thread (headless use); returns after :meth:`request_stop`."""
        self._prepare_start()
        self.run()

    # ── external setters (any thread) ─────────────────────

    def set_keep_spectrogram(self, enabled: bool):
        """Also record the raw sensor frame of every recorded frame."""
        self._keep_spectrogram = bool(enabled)

    def set_recording(self, active: bool):
        """Arm/disarm per-frame recording. Arming starts a fresh buffer.

        Arming also asks the worker to latch the camera clock, so the scan can later be
        trimmed to the stage motion window (:meth:`take_recorded_frames`).
        """
        if active:
            with self._lock:
                self._clock_sync_pending = True
        self.recorder.set_recording(active)

    def recorded_count(self) -> int:
        """Frames recorded so far."""
        return self.recorder.count()

    def take_recorded_frames(self, motion_start_ns=None, motion_end_ns=None) -> list[np.ndarray]:
        """Disarm recording and hand over (and empty) the recorded binned frames.

        Args:
            motion_start_ns, motion_end_ns: Host ``time.perf_counter_ns()`` when the stage
                started / stopped moving (``None`` = no bound).  With a camera that stamps
                its frames (GenICam) and a clock latched when recording was armed, frames
                taken before the start or after the end are dropped, with their raw
                frames.  Otherwise every recorded frame is returned, as before.

        Returns:
            list: The binned frames, in order.
        """
        window = None
        offset = self._clock_offset_ns
        if offset is not None and (motion_start_ns is not None or motion_end_ns is not None):
            window = (None if motion_start_ns is None else motion_start_ns + offset,
                      None if motion_end_ns is None else motion_end_ns + offset)
        frames = self.recorder.take_frames(window)
        before, after = self.recorder.last_trim
        if before or after:
            logger.info("Scan trimmed to stage motion: %d frames before start and %d after stop dropped, %d kept",
                        before, after, len(frames))
        return frames

    def take_recorded_raw(self) -> list[np.ndarray]:
        """Hand over (and empty) the raw sensor frames kept by Keep Spectrogram."""
        return self.recorder.take_raw()

    def set_scanning(self, active: bool):
        """Tell the engine a scan is running (blocks wavelength changes)."""
        self._scanning = bool(active)

    def set_raw_inverted(self, inverted: bool):
        """Flip the slit axis of every frame (sensor mounted upside down)."""
        self._raw_inverted = inverted

    def set_dark_capture_mode(self, active: bool):
        """While true, frames bypass corrections/binning and are delivered raw."""
        with self._binner_lock:
            self._dark_capture_mode = bool(active)

    def request_params(self, gain: int, exposure_ms: float):
        """Queue a gain/exposure change (applied between frames; the stream restarts)."""
        with self._lock:
            self._pending_params = (int(gain), float(exposure_ms))

    def request_target_temperature(self, temp_c: float):
        """Queue a cooler target (°C); applied between frames without restarting the stream.

        The outcome arrives as ``target_temperature_applied``/``target_temperature_failed``.
        """
        with self._lock:
            self._pending_target_temperature = float(temp_c)

    def request_bin(self, bin_value: int):
        """Queue a hardware-bin change; the outcome arrives as ``bin_applied``/``bin_failed``."""
        with self._lock:
            self._pending_bin = int(bin_value)

    def request_wavelength(self, profile):
        """Queue a wavelength calibration profile.

        Raises:
            ValueError: if the profile is invalid.
            RuntimeError: if another calibration is already pending.
        """
        profile.validate()
        with self._lock:
            if self._pending_wavelength is not None:
                raise RuntimeError("A wavelength calibration change is already pending")
            self._pending_wavelength = profile

    def set_wavelength_lookup(self, lookup):
        """Find the stored wavelength calibration of each camera setting with ``lookup()``.

        Call before the loop starts. ``lookup()`` runs on the worker thread when streaming
        starts and after every bin change, once the camera is at the new bin. It returns the
        ``WavelengthCalibration`` of the current setting, or raises ``LookupError`` saying
        why there is none; ``config.m``/``config.c`` are then kept (with a warning).
        ``None`` disables the lookup.
        """
        self._wavelength_lookup = lookup

    def set_resolution_lookup(self, lookup):
        """Find the FWHM stored for each camera setting with ``lookup()``.

        Call before the loop starts. Used only when ``config.resolution_source`` is ``"fwhm"``;
        ``lookup()`` then runs on the worker thread when streaming starts, after every bin
        change and after :meth:`request_resolution_reload`. It returns the median FWHM in nm,
        or raises ``LookupError`` saying why there is none; ``config.resolution`` is then the
        bin width (with a warning). ``None`` disables the lookup.
        """
        self._resolution_lookup = lookup

    def request_resolution_reload(self):
        """Re-read the stored FWHM (e.g. after a new one was saved); applied when no scan runs."""
        with self._lock:
            self._pending_resolution_reload = True

    def binning_info(self) -> dict:
        """Bin width the binner uses at the current setting, and where it comes from.

        Returns:
            dict: ``resolution_nm`` (bin width, nm), ``resolution_source`` (``"config"`` or
            ``"fwhm"``) and, for ``"fwhm"``, ``fwhm_nm`` and ``fwhm_step_factor``.
        """
        mm = float(self.config.m) * int(getattr(self.backend, 'current_bin', 1) or 1)
        return self._bin_step(mm)

    def _bin_step(self, mm: float) -> dict:
        """:meth:`binning_info` for a slope of ``mm`` nm per recorded pixel."""
        fwhm = self._fwhm_nm
        if getattr(self.config, 'resolution_source', 'config') == 'fwhm' and fwhm is not None:
            factor = float(getattr(self.config, 'fwhm_step_factor', 1.0))
            step = max(factor * fwhm, MIN_FWHM_STEP_PX * abs(mm))
            return {"resolution_nm": step, "resolution_source": "fwhm",
                    "fwhm_nm": fwhm, "fwhm_step_factor": factor}
        return {"resolution_nm": float(self.config.resolution), "resolution_source": "config"}

    def request_snapshot(self, request_id):
        """Ask for one copy of the next oriented, pre-Smile frame (delivered via ``sink.snapshot``)."""
        with self._lock:
            self._pending_snapshot = int(request_id)

    def request_processed_frame(self):
        """Ask for one copy of the next preprocessed frame before binning (delivered via ``sink.processed_frame``)."""
        with self._lock:
            self._pending_processed = True

    def request_smile_model(self, path):
        """Queue loading a Smile model file for the current bin."""
        with self._lock:
            self._pending_smile_model = str(path)

    # ── pending changes (worker thread) ───────────────────

    def _apply_pending_wavelength(self):
        with self._lock:
            profile = self._pending_wavelength
            self._pending_wavelength = None
        if profile is None:
            return
        if self._scanning or self._dark_capture_mode:
            self.sink.wavelength_failed(profile, "Stop scan/dark capture before applying wavelength calibration")
            return
        if profile.camera_bin != int(self.backend.current_bin):
            self.sink.wavelength_failed(profile, "Wavelength profile bin differs from current camera bin")
            return
        self.config.m = profile.base_m
        self.config.c = profile.c
        with self._binner_lock:
            self._binner = None
            self._binner_fitted = False
        self.sink.wavelength_applied(profile)

    def _snapshot_requested(self) -> bool:
        with self._lock:
            return self._pending_snapshot is not None

    def _publish_snapshot_if_requested(self, frame):
        with self._lock:
            request_id = self._pending_snapshot
            self._pending_snapshot = None
        if request_id is None:
            return
        image = frame.copy()
        image.setflags(write=False)
        dark = getattr(self.backend, '_dark_mean', None)
        enabled = bool(getattr(self.backend, '_dark_enabled', False))
        packet = FrameSnapshot(request_id, image, time.monotonic(),
            int(self.backend.current_bin), self._raw_inverted, enabled,
            enabled and dark is not None and dark.shape == frame.shape)
        self.sink.snapshot(packet)

    def _processed_requested(self) -> bool:
        with self._lock:
            return self._pending_processed

    def _publish_processed_if_requested(self, frame, smile_applied):
        with self._lock:
            requested = self._pending_processed
            self._pending_processed = False
        if not requested:
            return
        image = frame.copy()
        image.setflags(write=False)
        dark = getattr(self.backend, '_dark_mean', None)
        enabled = bool(getattr(self.backend, '_dark_enabled', False))
        self.sink.processed_frame(FrameSnapshot(0, image, time.monotonic(),
            int(self.backend.current_bin), self._raw_inverted, enabled,
            enabled and dark is not None and dark.shape == frame.shape,
            'post_smile' if smile_applied else 'post_dark_hot_pixel_orientation'))

    def _read_camera_frame(self, profiler, dark_capture):
        """Raw dark capture bypasses correction; normal frames keep backend policy."""
        if dark_capture:
            with profiler.time("fetch"):
                frame = self.backend.read_frame_raw(timeout_ms=100)
            profiler._record("dark", 0.0)
            profiler._record("hot_pixel", 0.0)
            return frame
        if self._shared_preprocessing:
            # The camera may convert straight into the GPU backend's zero-copy input buffer;
            # that frame is overwritten by the next read (the recorder copies what it keeps).
            with profiler.time('fetch'):
                frame = self.backend.read_frame_raw_into(self._preprocessor.input_buffer, timeout_ms=100)
            # Backends that time their read (GenICam) split fetch into waiting and converting.
            timing = getattr(self.backend, 'last_fetch_timing', None)
            if frame is not None and timing is not None:
                profiler._record('cam_wait', timing[0])
                profiler._record('convert', timing[1])
            return frame
        read_profiled = getattr(self.backend, 'read_frame_profiled', None)
        if read_profiled is not None:
            return read_profiled(profiler, timeout_ms=100)
        with profiler.time("fetch"):
            return self.backend.read_frame(timeout_ms=100)

    def _report_corrections(self, frame, dark_capture=False, smile_error=False):
        """Observe the last processed frame; build payload only on a change."""
        reference = getattr(self.backend, '_dark_mean', None)
        enabled = bool(getattr(self.backend, '_dark_enabled', False))
        model = self._smile.calibration
        bin_value = int(self.backend.current_bin)
        matches = reference is not None and reference.shape == frame.shape
        key = (dark_capture, enabled, id(reference), matches, id(model),
               self._smile.path, bin_value, smile_error)
        if key == self._last_correction_status:
            return
        self._last_correction_status = key
        if dark_capture:
            dark = ('Capturing', 'Collecting raw reference')
            smile = ('Waiting', 'Bypassed during dark capture')
        else:
            if not enabled:
                dark = ('Off', 'Dark subtraction disabled')
            elif reference is None:
                dark = ('Missing', f'No reference · bin {bin_value}')
            elif not matches:
                dark = ('Missing', 'Reference shape mismatch')
            else:
                dark = ('Applied', f'Reference matched · bin {bin_value}')
            if smile_error:
                smile = ('Missing', 'Model could not process frame')
            elif model is None:
                smile = ('Missing', f'No model · bin {bin_value}')
            else:
                smile = ('Applied', Path(self._smile.path).name)
        self.sink.correction_status({'dark': dark, 'smile': smile})

    def _select_smile_bin(self):
        try:
            self._smile.set_bin(self.backend.current_bin)
        except Exception as exc:
            self.sink.error(f"Smile model load failed: {exc}")

    def _select_resolution(self):
        """Use the FWHM stored for the current camera setting, if ``resolution_source`` asks for it."""
        fwhm = None
        lookup = self._resolution_lookup
        if getattr(self.config, 'resolution_source', 'config') == 'fwhm' and lookup is not None:
            bin_value = int(self.backend.current_bin)
            try:
                fwhm = float(lookup())
                if not np.isfinite(fwhm) or fwhm <= 0:
                    raise LookupError(f"stored FWHM {fwhm} is not a positive number")
            except Exception as exc:
                fwhm = None
                logger.warning("Bin %d: no usable FWHM (%s); bin width = config resolution %.4g nm",
                               bin_value, exc, float(self.config.resolution))
            else:
                logger.info("Bin %d: FWHM %.4g nm (stored for this setting)", bin_value, fwhm)
        self._fwhm_nm = fwhm
        with self._binner_lock:
            self._binner = None
            self._binner_fitted = False

    def _apply_pending_resolution(self):
        with self._lock:
            pending = self._pending_resolution_reload
            if not pending or self._scanning or self._dark_capture_mode:
                return                          # a scan keeps its bin width; reload afterwards
            self._pending_resolution_reload = False
        self._select_resolution()

    def _select_wavelength_bin(self):
        """Use the stored wavelength calibration of the current camera setting, if there is one."""
        lookup = self._wavelength_lookup
        if lookup is None:
            return
        bin_value = int(self.backend.current_bin)
        try:
            profile = lookup()
            if int(profile.camera_bin) != bin_value:
                raise LookupError(f"stored profile is for bin {profile.camera_bin}")
        except Exception as exc:
            logger.warning("Bin %d: no usable wavelength calibration (%s); keeping m=%.8g nm/px (bin 1), c=%.8g nm",
                           bin_value, exc, float(self.config.m), float(self.config.c))
            return
        self.config.m = profile.base_m
        self.config.c = profile.c
        with self._binner_lock:
            self._binner = None
            self._binner_fitted = False
        logger.info("Bin %d: wavelength calibration m=%.8g nm/px (bin 1), c=%.8g nm",
                    bin_value, profile.base_m, profile.c)

    def _apply_pending_smile(self):
        with self._lock:
            path = self._pending_smile_model
            self._pending_smile_model = None
        if path is not None:
            try:
                self._smile.load(path, self.backend.current_bin)
                logger.info("Smile model selected: %s", path)
            except Exception as exc:
                self.sink.error(f"Smile model load failed: {exc}")

    def _apply_pending_params_if_needed(self):
        with self._lock:
            params = self._pending_params
            self._pending_params = None
        if params is None:
            return
        gain, exposure_ms = params
        logger.info("Apply params: gain=%s exp=%.3f ms", gain, exposure_ms)
        self.backend.stop_stream()
        failed = False
        try:
            self.backend.set_params(gain=gain, exposure_ms=exposure_ms)
        except Exception as exc:
            failed = True
            logger.error("Apply params failed: %s", exc)
            self.sink.params_failed(str(exc))
        self.backend.start_stream()
        if not failed:
            self.sink.params_applied()
        self._report_frame_rate("exposure/gain change")

    def _apply_pending_target_temperature(self):
        with self._lock:
            temp_c, self._pending_target_temperature = self._pending_target_temperature, None
        if temp_c is None:
            return
        try:
            self.backend.set_target_temperature(temp_c)
        except Exception as exc:
            logger.error("Target temperature %.1f C not applied: %s", temp_c, exc)
            self.sink.target_temperature_failed(str(exc))
            return
        logger.info("Target temperature %.1f C applied (cooler on)", temp_c)
        self.sink.target_temperature_applied(temp_c)

    def _apply_pending_bin_if_needed(self):
        with self._lock:
            bin_value = self._pending_bin
            self._pending_bin = None
        if bin_value is None:
            return
        logger.info("Apply bin: %s", bin_value)
        previous = int(self.backend.current_bin)
        self.backend.stop_stream()
        failure = None
        try:
            self.backend.set_bin(bin_value)
        except Exception as exc:
            failure = exc
            logger.error("Apply bin %s failed: %s", bin_value, exc)
            try:
                self.backend.set_bin(previous)
            except Exception as restore_exc:
                logger.error("Restoring bin %s failed: %s", previous, restore_exc)
        self._select_smile_bin()
        try:
            self.backend.start_stream()
        except Exception as exc:
            if failure is not None:
                raise                           # already back on the previous bin: nothing left to try
            # The camera accepted the bin but will not stream at it: return to the previous bin.
            failure = exc
            logger.error("Camera does not stream at bin %s (%s); returning to bin %s", bin_value, exc, previous)
            self.backend.stop_stream()
            self.backend.set_bin(previous)
            self._select_smile_bin()
            self.backend.start_stream()         # a failure here is fatal, as before
        self._select_wavelength_bin()
        self._select_resolution()
        with self._binner_lock:
            self._binner_fitted = False
            self._binner = None
        if failure is None:
            self.sink.bin_applied(int(self.backend.current_bin))
        else:
            self.sink.bin_failed(int(bin_value), str(failure))
        self._report_frame_rate("bin change")

    def _apply_pending_clock_sync(self):
        """Latch the camera clock once after recording is armed (acquisition thread)."""
        with self._lock:
            pending, self._clock_sync_pending = self._clock_sync_pending, False
        if not pending:
            return
        try:
            self._clock_offset_ns = self.backend.camera_clock_offset_ns()
        except Exception as exc:
            logger.warning("Camera clock latch failed: %s", exc)
            self._clock_offset_ns = None
        if self._clock_offset_ns is None:
            logger.info("Camera clock not latched: the scan is not trimmed to the stage motion")

    def _note_frame_info(self, profiler):
        """Count frames the camera sent but the loop never received; return the frame's camera time."""
        info = getattr(self.backend, 'last_frame_info', None)
        if not info:
            return None
        stamp, frame_id = info
        if isinstance(frame_id, int):
            last, self._last_frame_id = self._last_frame_id, frame_id
            # IDs restart with every AcquisitionStart and wrap: only count forward gaps.
            profiler.add_count('lost', frame_id - last - 1 if last is not None and frame_id > last else 0)
        return stamp

    def _profile_label(self) -> str:
        """Header of the profile log line: backend, camera bin, frame W x H and the camera's frame rate."""
        parts = [f"backend={self._preprocessor.backend_name}",
                 f"bin={getattr(self.backend, 'current_bin', '?')}"]
        if self._frame_shape is not None:
            parts.append(f"{self._frame_shape[1]}x{self._frame_shape[0]}")
        if self._camera_hz is not None:
            parts.append(f"cam={self._camera_hz:.1f}Hz")
        return " ".join(parts)

    def _report_frame_rate(self, reason: str):
        """Log the camera's AcquisitionFrameRate and send it to the sink (skipped if unknown)."""
        try:
            hz = self.backend.frame_rate()
        except Exception as exc:
            logger.debug("Frame rate not readable: %s", exc)
            return
        if hz is None:
            return
        logger.info("AcquisitionFrameRate after %s: %.3f Hz (bin %s, exposure %s ms)",
                    reason, hz, getattr(self.backend, "current_bin", "?"),
                    getattr(self.backend, "exposure_ms", "?"))
        self._camera_hz = float(hz)
        self.sink.frame_rate(float(hz))

    # ── fps ───────────────────────────────────────────────

    def _compute_fps(self) -> float:
        now = time.monotonic()
        if self._last_frame_time is None:
            self._last_frame_time = now
            return 0.0
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt <= 0:
            return 0.0
        self._intervals.append(dt)
        if len(self._intervals) > 30:
            self._intervals.pop(0)
        return len(self._intervals) / sum(self._intervals)

    # ── spectral binning ──────────────────────────────────

    def _ensure_binner(self, width: int) -> LambdaBinner:
        """Return the fitted LambdaBinner, fitting it for ``width`` spectral pixels if needed."""
        with self._binner_lock:
            binner = self._binner
            fitted = self._binner_fitted

        if not fitted:
            camera_bin = getattr(self.backend, 'current_bin', 1)
            mm = float(self.config.m) * camera_bin
            cc = float(self.config.c)
            info = self._bin_step(mm)
            step = info["resolution_nm"]

            binner = LambdaBinner(m=mm, c=cc, step=step)
            binner.fit(int(width))

            with self._binner_lock:
                self._binner = binner
                self._binner_fitted = True

            logger.info(
                "LambdaBinner fitted: %d pixels -> %d bins "
                "(m=%.4f, bin=%d, step=%.2f [%s], lam %.1f-%.1f nm)",
                width, binner.n_bins,
                mm, camera_bin, step, info["resolution_source"],
                binner.lambda_out[0], binner.lambda_out[-1],
            )
        return binner

    def _bin_frame(self, frame: np.ndarray) -> np.ndarray:
        """LambdaBinning. frame = (h, v) uint16"""
        binner = self._ensure_binner(frame.shape[1])
        binned = binner.transform(frame.reshape(1, *frame.shape))
        return binned[0]

    # ── saturation ceiling ────────────────────────────────

    def _saturation_level(self, frame) -> int:
        try:
            return int(self.backend.saturation_level())
        except Exception:
            return int(np.iinfo(frame.dtype).max) if np.issubdtype(frame.dtype, np.integer) else 65535

    def _update_saturation_ceiling(self, frame, correction, dark_capture):
        """Send ``sink.saturation_ceiling`` when the level or any correction setting changed."""
        level = self._saturation_level(frame)
        if dark_capture:
            key, refs = ('raw', frame.shape, frame.dtype.str, level), None
        elif not self._shared_preprocessing:
            key, refs = (None,), None                # the backend corrects frames itself: unknown
        else:
            with self._binner_lock:
                binner = self._binner
            reference = getattr(correction, 'dark_reference', correction)
            dark = getattr(reference, '_dark_mean', None)
            hot = getattr(reference, '_hot_coords', None)
            calibration = self._smile.calibration
            refs = (dark, hot, calibration, binner)
            key = ('display', frame.shape, frame.dtype.str, level,
                   bool(getattr(correction, '_dark_enabled', False)), self._raw_inverted,
                   *(id(obj) for obj in refs))
        if key == self._ceiling_key:
            return
        ceiling = None
        try:
            full = np.full(frame.shape, level, dtype=frame.dtype)
            if key[0] == 'raw':
                ceiling = full
            elif key[0] == 'display':
                ceiling, _, _ = self._preprocessor.process_binned(
                    full, correction, self._ensure_binner(frame.shape[1]),
                    calibration=self._smile.calibration, inverted=self._raw_inverted)
        except Exception as exc:
            logger.warning("Saturation ceiling unavailable: %s", exc)
        self._ceiling_key = key
        self._ceiling_refs = refs
        self.sink.saturation_ceiling(ceiling)

    @property
    def binner_lambda_out(self) -> np.ndarray | None:
        """Centre wavelengths (nm) of the output bins, or ``None`` before the first frame."""
        with self._binner_lock:
            if self._binner is not None:
                return self._binner.lambda_out.copy()
        return None

    @property
    def binner_n_bins(self) -> int:
        """Number of output wavelength bins (0 before the first frame)."""
        with self._binner_lock:
            if self._binner is not None:
                return self._binner.n_bins
        return 0

    # ── main loop ─────────────────────────────────────────

    def _flush_preview_batch(self):
        batch = self.recorder.pop_batch()
        if batch:
            self.sink.cube_frames(batch)

    def run(self):
        """The acquisition loop; blocks until :meth:`request_stop` and always closes the camera.

        Never raises: failures are reported through ``sink.error``. A fatal error in the
        preprocessing of a frame ends the loop; read timeouts are retried.
        """
        logger.info("Acquisition started")

        sharpness_estimator = ImageSharpness()
        last_sharpness_time = 0.0

        frame_idx = 0
        prof = self._profiler

        try:
            self.backend.open()
            self.backend.start_stream()
        except Exception as e:
            logger.error("start failed: %s", e)
            self.sink.error(str(e))
            return
        self._report_frame_rate("start")

        self._select_smile_bin()
        self._select_wavelength_bin()
        self._select_resolution()

        try:
            while self._alive:
                if self._stop_event.is_set():
                    break

                self._apply_pending_params_if_needed()
                self._apply_pending_target_temperature()
                self._apply_pending_bin_if_needed()
                self._apply_pending_smile()
                self._apply_pending_wavelength()
                self._apply_pending_resolution()
                self._apply_pending_clock_sync()
                self._flush_preview_batch()

                with self._binner_lock:
                    dark_capture = self._dark_capture_mode

                # Time raw acquisition separately from dark/hot-pixel kernels.
                try:
                    frame = self._read_camera_frame(prof, dark_capture)
                except Exception as e:
                    logger.warning("read_frame error: %s", e)
                    self.sink.error(str(e))
                    if self._stop_event.is_set():
                        break
                    continue

                if frame is None:
                    continue
                self._frame_shape = frame.shape
                stamp = self._note_frame_info(prof) if not dark_capture else None

                if dark_capture:
                    self._report_corrections(frame, dark_capture=True)
                    self._update_saturation_ceiling(frame, None, dark_capture=True)
                    emit_frame = frame
                else:
                    binner = self._ensure_binner(frame.shape[1])
                    try:
                        correction = self.backend if self._shared_preprocessing else self._passthrough_dark
                        calibration = self._smile.calibration
                        # On a GPU backend correction and binning stay on the device; the
                        # unbinned image is downloaded only for a processed-frame request.
                        emit_frame, corrected, grabbed = self._preprocessor.process_binned(
                            frame, correction, binner, calibration=calibration,
                            inverted=self._raw_inverted, snapshot=self._snapshot_requested(),
                            full=self._processed_requested(), profiler=prof)
                        if grabbed is not None:
                            self._publish_snapshot_if_requested(grabbed)
                        if corrected is not None:
                            self._publish_processed_if_requested(corrected, calibration is not None)
                    except Exception as exc:
                        self._report_corrections(frame, smile_error=True)
                        self.sink.error(str(exc))
                        break
                    self._report_corrections(frame)
                    self._update_saturation_ceiling(frame, correction, dark_capture=False)
                    # transform() allocates a new array per call, so it is safe to keep
                    self.recorder.offer(emit_frame, frame, self._keep_spectrogram, stamp)

                # sharpness
                now = time.monotonic()
                if now - last_sharpness_time > SHARPNESS_INTERVAL:
                    try:
                        with prof.time("sharpness"):
                            sharpness_estimator.LoadImage(emit_frame)
                            score = float(sharpness_estimator.SharpnessScore(axis=0))
                        self.sink.sharpness(score)
                        last_sharpness_time = now
                    except Exception:
                        pass

                # fps
                fps = self._compute_fps()

                # throttle
                if now - self._last_emit_time < MIN_FRAME_INTERVAL:
                    self._skipped += 1
                    prof.tick()
                    continue
                self._last_emit_time = now

                # emit
                with prof.time("emit"):
                    self.sink.frame(emit_frame, fps, now)
                self._emitted += 1
                frame_idx += 1
                prof.tick()

        except Exception as exc:
            logger.exception("Acquisition loop failed")
            self.sink.error(str(exc))
        finally:
            try:
                self.backend.stop_stream()
                self.backend.close()
            except Exception as e:
                logger.warning("cleanup error: %s", e)

        logger.info(
            "Acquisition stopped (emitted=%d, skipped=%d)",
            self._emitted, self._skipped,
        )
