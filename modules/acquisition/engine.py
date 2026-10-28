"""Qt-free acquisition engine: fetch -> preprocess -> spectral bin -> record -> deliver.

Pipeline per frame (worker thread)::

    backend.read_frame_raw()                      (raw sensor frame)
      -> dark subtraction, hot-pixel repair      \\
      -> orientation (raw_inverted)               |  modules.preprocessing
      -> Smile correction (if a model is loaded) /
      -> spectral binning (LambdaBinner)          (modules.spectral_processing)
         (with a wavelength range: only the bins inside it, a subset of the full grid)
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
#: Fewest bins a wavelength range may keep; a narrower range is refused.
MIN_RANGE_BANDS = 3
#: "No range request" (``None`` already means "clear the range").
_NO_CHANGE = object()


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

    def wavelength_range_applied(self, info: dict) -> None:
        """The wavelength bins in use, sent after every binner fit and every range change.

        ``info`` keys: ``range_nm`` (the applied ``(lo, hi)`` nm, or ``None`` for the full
        grid), ``bands`` (``(k0, k1)`` of the full grid, or ``None``), ``full_bands``,
        ``centres_nm`` (first and last bin centre in use), ``calibrated_nm`` (wavelengths
        of the first and last sensor pixel) and ``step_nm``.
        """

    def wavelength_range_failed(self, requested, message: str) -> None:
        """A wavelength range (``(lo, hi)`` nm, or ``None``) was refused, or the applied one
        no longer fits after a bin/calibration change and was cleared. The next
        ``wavelength_range_applied`` (if any) shows the range in use."""


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

    # Wavelength range (class defaults, so engines built with __new__ in tests work too).
    _range_nm = None                # applied (lo, hi) nm, or None: the full bin grid
    _pending_range = _NO_CHANGE     # request queued by request_wavelength_range
    _range_candidate = _NO_CHANGE   # request taken by the worker, until it is resolved
    _binner_full = None             # the full-grid binner the current one was cut from
    _bin_in_flight = False          # bin request taken by the worker, until its outcome is sent

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
        self._armed_ns: int | None = None         # host perf_counter_ns when recording was armed
        self._record_start_ns: int | None = None  # host time before which frames are not recorded
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

    def set_recording(self, active: bool, start_ns=None):
        """Arm/disarm per-frame recording. Arming starts a fresh buffer.

        Arming also asks the worker to latch the camera clock, so the scan can later be
        trimmed to the stage motion window (:meth:`take_recorded_frames`).

        Args:
            active: Arm (True) or disarm.
            start_ns: Host ``time.perf_counter_ns()`` before which frames are not recorded
                at all (the start of the constant-speed window), once the clock is latched;
                ``None`` records every frame until :meth:`take_recorded_frames` trims.
        """
        if active:
            with self._lock:
                if not self.recorder.enabled:
                    self._armed_ns = time.perf_counter_ns()
                self._record_start_ns = start_ns
        self.recorder.set_recording(active)
        if active:
            # After arming, which clears the recorder's bound, so the latch can set it.
            with self._lock:
                self._clock_sync_pending = True

    def recorded_count(self) -> int:
        """Frames recorded so far."""
        return self.recorder.count()

    def live_recorded_frames(self) -> list:
        """The list the current recording appends to, for a live cube preview (by reference).

        Ask after :meth:`set_recording` (arming starts a new list); do not modify it.
        """
        return self.recorder.live_frames()

    def take_recorded_frames(self, motion_start_ns=None, motion_end_ns=None,
                             reference_ns=None) -> list[np.ndarray]:
        """Disarm recording and hand over (and empty) the recorded binned frames.

        Logs one ``Scan trim:`` line every time: why nothing was trimmed, or the frame
        times, the kept window and the counts dropped.

        Args:
            motion_start_ns, motion_end_ns: Host ``time.perf_counter_ns()`` when the stage
                started / stopped moving (``None`` = no bound).  With a camera that stamps
                its frames (GenICam) and a clock latched when recording was armed, frames
                taken before the start or after the end are dropped, with their raw
                frames.  Otherwise every recorded frame is returned, as before.
            reference_ns: Host time the logged times are relative to (the start of the
                stage move); defaults to ``motion_start_ns``.

        Returns:
            list: The binned frames, in order.
        """
        window = None
        offset = self._clock_offset_ns
        if offset is not None and (motion_start_ns is not None or motion_end_ns is not None):
            window = (None if motion_start_ns is None else motion_start_ns + offset,
                      None if motion_end_ns is None else motion_end_ns + offset)
        frames = self.recorder.take_frames(window)
        self._log_trim(frames, offset, motion_start_ns, motion_end_ns,
                       motion_start_ns if reference_ns is None else reference_ns)
        return frames

    def _log_trim(self, frames, offset, start_ns, end_ns, reference_ns):
        """The ``Scan trim:`` summary of :meth:`take_recorded_frames` (times in ms from ``reference_ns``)."""
        first, last, missing, total = self.recorder.last_stamps
        before, after = self.recorder.last_trim

        def ms(host_ns):
            """Host time as signed ms from ``reference_ns`` ('?' if unknown)."""
            if host_ns is None or reference_ns is None:
                return '?'
            return f'{(host_ns - reference_ns) / 1e6:+.0f}'

        def camera_ms(stamp):
            """Camera time, converted to host time, as :func:`ms`."""
            return '?' if stamp is None or offset is None else ms(stamp - offset)

        if offset is None:
            reason = 'not trimmed: camera clock not latched'
        elif start_ns is None and end_ns is None:
            reason = 'not trimmed: no stage motion times'
        elif missing:
            reason = f'not trimmed: {missing} frames without camera time'
        else:
            reason = (f'dropped {before + self.recorder.last_live_dropped} before '
                      f'({self.recorder.last_live_dropped} while recording), {after} after')
        logger.info('Scan trim: %s; %d recorded, %d kept; armed %s ms, frames %s..%s ms, '
                    'keep %s..%s ms (from stage move start)',
                    reason, total, len(frames), ms(self._armed_ns), camera_ms(first), camera_ms(last),
                    ms(start_ns), ms(end_ns))

    def take_recorded_raw(self) -> list[np.ndarray]:
        """Hand over (and empty) the raw sensor frames kept by Keep Spectrogram."""
        return self.recorder.take_raw()

    def set_scanning(self, active: bool):
        """Tell the engine a scan is running (blocks wavelength calibration and range changes)."""
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

    @property
    def bin_pending(self) -> bool:
        """True while a requested bin is queued or being applied (until ``bin_applied``/``bin_failed`` is sent)."""
        with self._lock:
            return self._pending_bin is not None or self._bin_in_flight

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

    @staticmethod
    def _checked_range(wavelength_range):
        """``None``, or ``(lo, hi)`` as floats; raises ``ValueError`` unless finite with lo < hi."""
        if wavelength_range is None:
            return None
        lo, hi = (float(v) for v in wavelength_range)
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"Wavelength range must be two finite nm values with start < end, not {wavelength_range!r}")
        return lo, hi

    def set_wavelength_range(self, wavelength_range):
        """Set the start-up wavelength range; call before the loop starts.

        Args:
            wavelength_range: ``(lo, hi)`` nm, or ``None`` for the full calibrated range.
                The first binner keeps only the bins inside it (see :meth:`request_wavelength_range`).

        Raises:
            ValueError: if the range is not two finite values with lo < hi.
        """
        self._range_nm = self._checked_range(wavelength_range)

    def request_wavelength_range(self, wavelength_range):
        """Queue a wavelength range; applied at a frame boundary, refused while scanning.

        Only the bins whose every spectral pixel lies inside the range are binned,
        recorded and shown, as a subset of the full bin grid (so the kept bins equal the
        same bins without a range, bit for bit). The range is kept in nm and resolved again
        after every bin, calibration or bin-width change. A later request replaces one not
        yet applied. The outcome arrives as ``wavelength_range_applied`` /
        ``wavelength_range_failed`` (also when fewer than :data:`MIN_RANGE_BANDS` bins fit).

        Args:
            wavelength_range: ``(lo, hi)`` nm, or ``None`` for the full calibrated range.

        Raises:
            ValueError: if the range is not two finite values with lo < hi.
        """
        wavelength_range = self._checked_range(wavelength_range)
        with self._lock:
            self._pending_range = wavelength_range

    @property
    def wavelength_range(self):
        """The applied wavelength range ``(lo, hi)`` nm, or ``None`` (full grid)."""
        with self._binner_lock:
            return self._range_nm

    @property
    def wavelength_range_pending(self) -> bool:
        """True while a requested wavelength range has not been applied or refused yet."""
        with self._lock:
            return self._pending_range is not _NO_CHANGE or self._range_candidate is not _NO_CHANGE

    def wavelength_range_metadata(self) -> dict:
        """Scan provenance of the range: empty unless the binner keeps a subset of the grid.

        Returns:
            dict: ``wavelength_range_requested_nm`` (``[lo, hi]``) and
            ``wavelength_range_bands`` (``[k0, k1, n_full]``: the kept bins of the full grid
            ``LambdaBinner(m * bin, c, step).fit(frame_width)``).
        """
        with self._binner_lock:
            binner, range_nm = self._binner, self._range_nm
        source = getattr(binner, 'source_bins', None)
        if range_nm is None or source is None:
            return {}
        return {"wavelength_range_requested_nm": [float(range_nm[0]), float(range_nm[1])],
                "wavelength_range_bands": [int(v) for v in source]}

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

    def correction_info(self) -> dict:
        """Corrections applied to the last processed frame, and the camera frame rate.

        Read at the end of a scan for the cube's export header (any thread; plain reads).

        Returns:
            dict: ``dark_corrected``, ``hot_pixel_corrected``, ``smile_corrected`` (bools),
            ``hot_pixel_count`` and ``smile_model`` (file name) when they apply, and
            ``frame_rate_hz``.  The correction keys are left out when no frame was processed
            yet or the frame was a dark-capture frame (unknown).
        """
        info = {} if self._camera_hz is None else {"frame_rate_hz": self._camera_hz}
        status = self._last_correction_status
        if status is None or status[0]:
            return info
        _, enabled, _, matches, model_id, smile_path, _, smile_error = status
        hot = getattr(self.backend, '_hot_coords', None)
        info["dark_corrected"] = bool(enabled and matches)
        info["hot_pixel_corrected"] = bool(enabled and hot is not None)
        if info["hot_pixel_corrected"]:
            info["hot_pixel_count"] = int(len(hot))
        info["smile_corrected"] = bool(model_id != id(None) and not smile_error)
        if info["smile_corrected"] and smile_path:
            info["smile_model"] = Path(smile_path).name
        return info

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

    def _apply_pending_wavelength_range(self):
        """Take a queued range request: refuse it while scanning, else apply it (worker thread).

        With a fitted binner the range is cut from its full grid now (keeping the binner
        object when the kept bins do not change); otherwise the next fit resolves it.
        """
        if self._pending_range is _NO_CHANGE:
            return                                  # the per-frame path takes no lock
        with self._lock:
            requested = self._range_candidate = self._pending_range
            self._pending_range = _NO_CHANGE
        if requested is _NO_CHANGE:
            return
        if self._scanning:
            self._range_candidate = _NO_CHANGE
            self.sink.wavelength_range_failed(requested, "Stop the scan before changing the wavelength range")
            return
        with self._binner_lock:
            full = self._binner_full if self._binner_fitted else None
            current = self._binner
        if full is not None:
            self._publish_range(full, current)

    def _range_binner(self, full, wavelength_range, current=None):
        """The binner for ``wavelength_range`` on ``full``'s grid, or ``None`` if too few bins fit.

        Returns ``full`` itself when the range keeps every bin, and ``current`` when it keeps
        the same bins as ``current`` (so GPU tables and caches stay valid).
        """
        k0, k1 = full.range_bins(*wavelength_range)
        if k1 - k0 < MIN_RANGE_BANDS:
            return None
        if k1 - k0 == full.n_bins:
            return full
        if current is not None and getattr(current, 'source_bins', None) == (k0, k1, full.n_bins):
            return current
        return full.subset(k0, k1)

    def _publish_range(self, full, current=None):
        """Resolve the range for the freshly fitted (or current) ``full`` grid and publish it.

        The pending candidate (if any) wins over the applied range; a candidate that keeps
        fewer than :data:`MIN_RANGE_BANDS` bins is refused and the applied range kept; an
        applied range that no longer fits is cleared. Never raises: an unexpected failure
        falls back to the full grid. Sends ``wavelength_range_failed`` (if any) and then
        ``wavelength_range_applied``.

        Returns:
            LambdaBinner: the published binner.
        """
        with self._lock:
            candidate, self._range_candidate = self._range_candidate, _NO_CHANGE
        failures = []
        range_nm = self._range_nm
        binner = full
        try:
            if candidate is not _NO_CHANGE:
                if candidate is None:
                    range_nm = None
                else:
                    chosen = self._range_binner(full, candidate, current)
                    if chosen is None:
                        failures.append((candidate, self._range_refusal(full, candidate)))
                    else:
                        range_nm, binner = candidate, chosen
            if binner is full and range_nm is not None and (candidate is _NO_CHANGE or failures):
                chosen = self._range_binner(full, range_nm, current)
                if chosen is None:
                    failures.append((range_nm, self._range_refusal(full, range_nm) + "; full range in use"))
                    range_nm = None
                else:
                    binner = chosen
        except Exception as exc:                    # never stop the camera over a range
            logger.exception("Wavelength range failed")
            failures.append((range_nm if candidate is _NO_CHANGE else candidate, f"Wavelength range failed: {exc}"))
            range_nm, binner = None, full
        with self._binner_lock:
            self._binner = binner
            self._binner_full = full
            self._binner_fitted = True
            self._range_nm = range_nm
        for requested, message in failures:
            logger.warning("Wavelength range %s refused: %s", requested, message)
            self.sink.wavelength_range_failed(requested, message)
        info = self._range_info(full, binner, range_nm)
        if binner is not full and binner is not current:
            first, last = binner.iL[0], binner.iR[-1] - 1
            logger.info("Wavelength range: requested %.1f-%.1f nm -> bands %d:%d of %d, "
                        "centres %.2f-%.2f nm, pixels %.2f-%.2f nm",
                        range_nm[0], range_nm[1], *info["bands"], info["full_bands"], *info["centres_nm"],
                        full.m * float(first) + full.c, full.m * float(last) + full.c)
        self.sink.wavelength_range_applied(info)
        return binner

    @staticmethod
    def _range_refusal(full, wavelength_range) -> str:
        """Why ``wavelength_range`` cannot be used on ``full``'s grid."""
        k0, k1 = full.range_bins(*wavelength_range)
        lam = full.lambda_out
        return (f"{wavelength_range[0]:g}-{wavelength_range[1]:g} nm keeps {k1 - k0} bands "
                f"(at least {MIN_RANGE_BANDS} needed; calibrated bands {lam[0]:.1f}-{lam[-1]:.1f} nm)")

    @staticmethod
    def _range_info(full, binner, range_nm) -> dict:
        """The ``info`` dict of ``wavelength_range_applied``."""
        source = binner.source_bins if binner is not full else None
        lam = binner.lambda_out
        return {
            "range_nm": range_nm,
            "bands": None if source is None else (int(source[0]), int(source[1])),
            "full_bands": int(full.n_bins),
            "centres_nm": (float(lam[0]), float(lam[-1])),
            "calibrated_nm": (float(full.c), float(full.m * (full.v_ - 1) + full.c)),
            "step_nm": float(full.step),
        }

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
            self._bin_in_flight = bin_value is not None
        if bin_value is None:
            return
        try:
            self._apply_bin(bin_value)
        finally:
            with self._lock:
                self._bin_in_flight = False     # cleared after the outcome is sent (see bin_pending)

    def _apply_bin(self, bin_value: int):
        """Switch the camera to ``bin_value`` (back to the previous bin on failure) and report it."""
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
        elif self._record_start_ns is not None:
            self.recorder.set_min_stamp(self._record_start_ns + self._clock_offset_ns)

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

            logger.info(
                "LambdaBinner fitted: %d pixels -> %d bins "
                "(m=%.4f, bin=%d, step=%.2f [%s], lam %.1f-%.1f nm)",
                width, binner.n_bins,
                mm, camera_bin, step, info["resolution_source"],
                binner.lambda_out[0], binner.lambda_out[-1],
            )
            # Publishes the full binner, or the bins of the wavelength range cut from it.
            binner = self._publish_range(binner)
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
                self._apply_pending_wavelength_range()
                self._apply_pending_clock_sync()

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
