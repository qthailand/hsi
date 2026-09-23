# workers/camera_preview.py
"""Camera preview thread — process + bin in worker, emit numpy to GUI."""

from __future__ import annotations

import logging
import time
import threading
from collections import defaultdict

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from BiningTools import LambdaBinner

logger = logging.getLogger(__name__)

MAX_GUI_FPS = 30
MIN_FRAME_INTERVAL = 1.0 / MAX_GUI_FPS


class _StageProfiler:
    """Lightweight per-stage timing."""

    class _Span:
        __slots__ = ("_prof", "_name", "_t0")

        def __init__(self, prof: "_StageProfiler", name: str):
            self._prof = prof
            self._name = name

        def __enter__(self):
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            dt_ms = (time.perf_counter() - self._t0) * 1000.0
            self._prof._record(self._name, dt_ms)

    def __init__(self, enabled: bool = True, log_every: int = 90):
        self.enabled = enabled
        self.log_every = log_every
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._frame_count = 0

    def time(self, name: str) -> "_StageProfiler._Span":
        return self._Span(self, name)

    def _record(self, name: str, dt_ms: float):
        if not self.enabled:
            return
        buf = self._samples[name]
        buf.append(dt_ms)
        if len(buf) > self.log_every:
            buf.pop(0)

    def tick(self):
        if not self.enabled:
            return
        self._frame_count += 1
        if self._frame_count % self.log_every == 0:
            self._log_summary()

    def _log_summary(self):
        if not self._samples:
            return
        parts = []
        for name, buf in self._samples.items():
            if not buf:
                continue
            avg = sum(buf) / len(buf)
            parts.append(f"{name}={avg:.2f}ms")
        logger.info("[profile] %s", "  ".join(parts))


class CameraPreviewThread(QThread):
    """Process + spectral binning in worker thread.

    Pipeline per frame:
        backend.read_frame() → orient → bin → sharpness → emit

    During dark capture:
        backend.read_frame_raw() → emit (no orient, no bin)
    """

    frame_ready = pyqtSignal(object, float, float)
    spectrogram_ready = pyqtSignal(object) 
    saturation_ready = pyqtSignal(object)
    sharpness_ready = pyqtSignal(float)
    error = pyqtSignal(str)

    def __init__(self, backend, config=None, profile: bool = True):
        super().__init__()
        self.backend = backend
        self.config = config
        self._alive = False

        self._keep_spectrogram: bool = False             
        self._scanning: bool = False                   

        self._stop_event = threading.Event()
        if hasattr(backend, 'set_stop_event'):
            backend.set_stop_event(self._stop_event)

        self._lock = threading.Lock()
        self._pending_params: tuple[int, float] | None = None
        self._pending_bin: int | None = None

        self._last_frame_time: float | None = None
        self._intervals: list[float] = []
        self._last_emit_time: float = 0.0

        self._raw_inverted: bool = False

        # spectral binning
        self._binner = None
        self._binner_fitted = False
        self._binner_lock = threading.Lock()

        self._dark_capture_mode: bool = False

        self._emitted = 0
        self._skipped = 0

        self._profiler = _StageProfiler(enabled=profile, log_every=90)

    # ── lifecycle ─────────────────────────────────────────

    def start(self, priority=None):
        self._stop_event.clear()
        self._alive = True
        self._last_frame_time = None
        self._intervals.clear()
        self._last_emit_time = 0.0
        self._emitted = 0
        self._skipped = 0
        super().start(priority) if priority is not None else super().start()

    def stop(self, timeout_ms: int = 2000):
        self._alive = False
        self._stop_event.set()
        if self.isRunning():
            if not self.wait(timeout_ms):
                logger.warning("Preview thread timeout — terminating")
                self.terminate()
                self.wait(500)

    # ── external setters (GUI thread) ─────────────────────

    def set_keep_spectrogram(self, enabled: bool):
        self._keep_spectrogram = bool(enabled)

    def set_scanning(self, active: bool):
        self._scanning = bool(active)

    def set_raw_inverted(self, inverted: bool):
        self._raw_inverted = inverted

    def set_dark_capture_mode(self, active: bool):
        with self._binner_lock:
            self._dark_capture_mode = bool(active)

    def request_params(self, gain: int, exposure_ms: float):
        with self._lock:
            self._pending_params = (int(gain), float(exposure_ms))

    def request_bin(self, bin_value: int):
        with self._lock:
            self._pending_bin = int(bin_value)

    # ── apply pending (worker thread) ─────────────────────

    def _apply_pending_params_if_needed(self):
        with self._lock:
            params = self._pending_params
            self._pending_params = None
        if params is None:
            return
        gain, exposure_ms = params
        logger.info("Apply params: gain=%s exp=%.3f ms", gain, exposure_ms)
        self.backend.stop_stream()
        self.backend.set_params(gain=gain, exposure_ms=exposure_ms)
        self.backend.start_stream()

    def _apply_pending_bin_if_needed(self):
        with self._lock:
            bin_value = self._pending_bin
            self._pending_bin = None
        if bin_value is None:
            return
        logger.info("Apply bin: %s", bin_value)
        self.backend.stop_stream()
        self.backend.set_bin(bin_value)
        self.backend.start_stream()
        with self._binner_lock:
            self._binner_fitted = False
            self._binner = None

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

    def _bin_frame(self, frame: np.ndarray) -> np.ndarray:
        """LambdaBinning. frame = (h, v) uint16"""
        with self._binner_lock:
            binner = self._binner
            fitted = self._binner_fitted

        if not fitted:
            camera_bin = getattr(self.backend, 'current_bin', 1)
            mm = float(self.config.m) * camera_bin
            cc = float(self.config.c)
            step = float(self.config.resolution)

            binner = LambdaBinner(m=mm, c=cc, step=step)
            binner.fit(int(frame.shape[1]))

            with self._binner_lock:
                self._binner = binner
                self._binner_fitted = True

            logger.info(
                "LambdaBinner fitted: %d pixels -> %d bins "
                "(m=%.4f, bin=%d, step=%.2f, lam %.1f-%.1f nm)",
                frame.shape[1], binner.n_bins,
                mm, camera_bin, step,
                binner.lambda_out[0], binner.lambda_out[-1],
            )

        binned = binner.transform(frame.reshape(1, *frame.shape))
        return binned[0]

    @property
    def binner_lambda_out(self) -> np.ndarray | None:
        with self._binner_lock:
            if self._binner is not None:
                return self._binner.lambda_out.copy()
        return None

    @property
    def binner_n_bins(self) -> int:
        with self._binner_lock:
            if self._binner is not None:
                return self._binner.n_bins
        return 0

    # ── main loop ─────────────────────────────────────────

    def run(self):
        logger.info("CameraPreviewThread started")

        from utils.image import apply_raw_orientation
        from ImageSharpness import ImageSharpness

        sharpness_estimator = ImageSharpness()
        sharpness_interval = 0.2
        last_sharpness_time = 0.0

        frame_idx = 0
        prof = self._profiler

        try:
            self.backend.open()
            self.backend.start_stream()
        except Exception as e:
            logger.error("start failed: %s", e)
            self.error.emit(str(e))
            return

        try:
            while self._alive:
                if self._stop_event.is_set():
                    break

                self._apply_pending_params_if_needed()
                self._apply_pending_bin_if_needed()

                with self._binner_lock:
                    dark_capture = self._dark_capture_mode

                # ★ base class จัดการ fetch + dark + hot_pixel
                try:
                    with prof.time("fetch"):
                        if dark_capture:
                            frame = self.backend.read_frame_raw(timeout_ms=100)
                        else:
                            frame = self.backend.read_frame(timeout_ms=100)
                except Exception as e:
                    logger.warning("read_frame error: %s", e)
                    self.error.emit(str(e))
                    if self._stop_event.is_set():
                        break
                    continue

                if frame is None:
                    continue

                # orientation
                with prof.time("orient"):
                    corrected = apply_raw_orientation(frame, self._raw_inverted)

                # bin or raw
                if dark_capture:
                    emit_frame = frame
                else:
                    if self._scanning and self._keep_spectrogram:
                        self.spectrogram_ready.emit(corrected.copy())
                    with prof.time("bin"):
                        emit_frame = self._bin_frame(corrected)

                # sharpness
                now = time.monotonic()
                if now - last_sharpness_time > sharpness_interval:
                    try:
                        with prof.time("sharpness"):
                            sharpness_estimator.LoadImage(emit_frame)
                            score = float(sharpness_estimator.SharpnessScore(axis=0))
                        self.sharpness_ready.emit(score)
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
                    self.frame_ready.emit(emit_frame, fps, now)
                self._emitted += 1
                frame_idx += 1
                prof.tick()

        finally:
            try:
                self.backend.stop_stream()
                self.backend.close()
            except Exception as e:
                logger.warning("cleanup error: %s", e)

        logger.info(
            "CameraPreviewThread stopped (emitted=%d, skipped=%d)",
            self._emitted, self._skipped,
        )