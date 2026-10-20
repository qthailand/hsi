"""Qt adapter: run an :class:`~modules.acquisition.engine.AcquisitionEngine` on a ``QThread``.

Embedding:
    ``CameraPreviewThread(camera, config)`` + ``connect`` the signals + ``start()``.
    The engine's sink callbacks (worker thread) become queued signals, so slots run on
    the thread that owns the receiver (normally the GUI thread).

Display frames are pulled, not queued: the worker only replaces the newest frame in a
one-slot mailbox, and the GUI thread emits ``frame_ready`` for it when it is ready for
one. After each frame the GUI rests at least as long as the frame took to handle and
paint (``MAX_GUI_DUTY``), so a slow GUI shows fewer frames instead of falling behind,
and dialogs (e.g. the GTK file chooser on Linux, which only runs when Qt is idle) stay
responsive.

Threading:
    ``request_*``/``set_*`` are callable from the GUI thread; ``stop()`` waits for the
    worker and never terminates it (killing a thread inside a camera SDK call can
    corrupt the driver). The object must be created on the GUI thread.
"""
from __future__ import annotations

import logging
import math
import threading
import time

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal

from .engine import AcquisitionEngine, AcquisitionSink, MIN_FRAME_INTERVAL

logger = logging.getLogger(__name__)

MAX_GUI_DUTY = 0.5   # share of the GUI thread the live display may use; the rest stays free


class _SignalSink(AcquisitionSink):
    """Translate sink callbacks into the signals of a :class:`CameraPreviewThread`."""

    def __init__(self, thread: "CameraPreviewThread"):
        self._t = thread

    def frame(self, frame, fps, timestamp):
        self._t._offer_display_frame(frame, fps, timestamp)

    def saturation_ceiling(self, ceiling):
        self._t.saturation_ceiling_ready.emit(ceiling)

    def correction_status(self, status):
        self._t.correction_status.emit(status)

    def sharpness(self, score):
        self._t.sharpness_ready.emit(score)

    def snapshot(self, packet):
        self._t.snapshot_ready.emit(packet)

    def processed_frame(self, packet):
        self._t.processed_frame_ready.emit(packet)

    def error(self, message):
        self._t.error.emit(message)

    def bin_applied(self, bin_value):
        self._t.bin_applied.emit(bin_value)

    def bin_failed(self, requested, message):
        self._t.bin_failed.emit(requested, message)

    def params_applied(self):
        self._t.params_applied.emit()

    def params_failed(self, message):
        self._t.params_failed.emit(message)

    def target_temperature_applied(self, temp_c):
        self._t.target_temperature_applied.emit(temp_c)

    def target_temperature_failed(self, message):
        self._t.target_temperature_failed.emit(message)

    def frame_rate(self, hz):
        self._t.frame_rate_ready.emit(hz)

    def wavelength_applied(self, profile):
        self._t.wavelength_applied.emit(profile)

    def wavelength_failed(self, profile, message):
        self._t.wavelength_failed.emit(profile, message)

    def wavelength_range_applied(self, info):
        self._t.wavelength_range_applied.emit(info)

    def wavelength_range_failed(self, requested, message):
        self._t.wavelength_range_failed.emit(requested, message)


class CameraPreviewThread(AcquisitionEngine, QThread):
    """Acquisition worker thread with Qt signals.

    Signals:
        frame_ready(frame, fps, timestamp): Newest display frame (<= 30/s): binned, or
            raw during dark capture. Emitted on the GUI thread when the GUI is ready for
            one; frames that arrive while it is busy or resting are replaced by newer ones
            (never repeated). ``timestamp`` is when the worker produced the frame.
        saturation_ceiling_ready(object): Per-pixel level of a clipped pixel in the display
            frames (``None`` if unknown); emitted before the frames it applies to, on change.
        correction_status(dict): ``{'dark': (state, detail), 'smile': (...)}`` on change.
        sharpness_ready(float): Focus score (about 5 Hz).
        snapshot_ready(FrameSnapshot): Answer to :meth:`request_snapshot`.
        processed_frame_ready(FrameSnapshot): Answer to :meth:`request_processed_frame`
            (preprocessed frame before spectral binning).
        error(str): A failure; fatal ones end the thread.
        bin_applied(int) / bin_failed(int, str): Outcome of :meth:`request_bin`.
        params_applied(): Gain/exposure were applied (queued before any score of the new frames).
        params_failed(str): Gain/exposure could not be applied.
        target_temperature_applied(float) / target_temperature_failed(str): Outcome of
            :meth:`request_target_temperature`.
        wavelength_applied(profile) / wavelength_failed(profile, str): Calibration result.
        wavelength_range_applied(dict): The wavelength bins in use, after every binner fit
            and range change (see ``AcquisitionSink.wavelength_range_applied``).
        wavelength_range_failed(object, str): A range ``(lo, hi)`` (or ``None``) was refused,
            or cleared because it no longer fits.
        saturation_ready(object): Reserved for per-line saturation maps (not emitted yet).

    Args:
        backend: Camera backend (see :mod:`modules.camera`).
        config: ``AppConfig``-like object (see :class:`AcquisitionEngine`).
        profile: Log per-stage timings.
    """

    correction_status = pyqtSignal(object)
    wavelength_failed = pyqtSignal(object, str)
    wavelength_applied = pyqtSignal(object)
    wavelength_range_applied = pyqtSignal(object)
    wavelength_range_failed = pyqtSignal(object, str)
    snapshot_ready = pyqtSignal(object)
    processed_frame_ready = pyqtSignal(object)
    frame_ready = pyqtSignal(object, float, float)
    saturation_ceiling_ready = pyqtSignal(object)
    saturation_ready = pyqtSignal(object)
    sharpness_ready = pyqtSignal(float)
    error = pyqtSignal(str)
    bin_applied = pyqtSignal(int)
    bin_failed = pyqtSignal(int, str)
    params_applied = pyqtSignal()
    params_failed = pyqtSignal(str)
    target_temperature_applied = pyqtSignal(float)
    target_temperature_failed = pyqtSignal(str)
    frame_rate_ready = pyqtSignal(float)
    _display_frame_available = pyqtSignal()   # worker -> GUI wake-up; at most one is queued

    def __init__(self, backend, config=None, profile: bool = True):
        QThread.__init__(self)
        AcquisitionEngine.__init__(self, backend, config, sink=_SignalSink(self), profile=profile)
        self._display_lock = threading.Lock()
        self._display_latest = None            # (frame, fps, timestamp) not delivered yet
        self._display_wake_pending = False     # a wake-up is queued or deferred by the rest
        self._display_resting = False
        self._display_started = 0.0
        # Single-shot timers owned by this object, so none can fire after it is deleted.
        self._display_settle_timer = QTimer(self)    # interval 0: runs after the repaints just requested
        self._display_settle_timer.setSingleShot(True)
        self._display_settle_timer.setInterval(0)
        self._display_settle_timer.timeout.connect(self._rest_after_display)
        self._display_rest_timer = QTimer(self)
        self._display_rest_timer.setSingleShot(True)
        self._display_rest_timer.timeout.connect(self._end_display_rest)
        self._display_frame_available.connect(self._deliver_display_frame, type=Qt.QueuedConnection)

    def start(self, priority=None):
        """Start the worker thread (resets counters first)."""
        self._prepare_start()
        QThread.start(self, priority) if priority is not None else QThread.start(self)

    def stop(self, timeout_ms: int = 2000) -> bool:
        """Request stop and wait; an undelivered display frame is dropped.

        Returns:
            True if the thread has finished; on timeout it keeps shutting down on its own
            (its ``finally`` block closes the camera) and False is returned.
        """
        self.request_stop()
        if self.isRunning() and not self.wait(timeout_ms):
            logger.error("Acquisition thread did not stop within %d ms", timeout_ms)
            return False
        self._display_settle_timer.stop()
        self._display_rest_timer.stop()
        with self._display_lock:
            self._display_latest = None
            self._display_wake_pending = False
        self._display_resting = False
        return True

    # ── display frame mailbox ─────────────────────────────

    def _offer_display_frame(self, frame, fps, timestamp):
        """Worker thread: keep only the newest display frame; wake the GUI unless already due."""
        with self._display_lock:
            self._display_latest = (frame, fps, timestamp)
            wake = not self._display_wake_pending
            self._display_wake_pending = True
        if wake:
            self._display_frame_available.emit()

    def _deliver_display_frame(self):
        """GUI thread: emit ``frame_ready`` for the newest frame, then rest (see ``MAX_GUI_DUTY``)."""
        if self._display_resting:
            return                              # _end_display_rest delivers the newest frame
        with self._display_lock:
            latest, self._display_latest = self._display_latest, None
            self._display_wake_pending = False
        if latest is None:
            return
        self._display_resting = True
        self._display_started = time.perf_counter()
        try:
            self.frame_ready.emit(*latest)
        finally:
            self._display_settle_timer.start()

    def _rest_after_display(self):
        """GUI thread, after the frame was handled and painted: wait before taking the next one."""
        busy = time.perf_counter() - self._display_started
        rest = max(MIN_FRAME_INTERVAL - busy, busy * (1.0 - MAX_GUI_DUTY) / MAX_GUI_DUTY)
        self._display_rest_timer.start(max(0, math.ceil(rest * 1000.0)))

    def _end_display_rest(self):
        self._display_resting = False
        self._deliver_display_frame()
