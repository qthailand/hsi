# workers/cube_render.py
"""Non-blocking cube render worker for realtime preview.

Thread-safe design:
- push_frame() / reset() are called from GUI/main thread
- _tick() runs on worker thread
- All shared state (_frames, _dirty, _wavelengths, etc.) protected by _lock
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, Qt

logger = logging.getLogger(__name__)

CUBE_PREVIEW_FPS = 15
CUBE_PREVIEW_INTERVAL_MS = int(1000 / CUBE_PREVIEW_FPS)


class CubeRenderWorker(QObject):
    """Worker that lives on a dedicated QThread.

    All public methods are thread-safe (called from GUI/main thread).
    All internal state is protected by self._lock.
    """

    rgb_ready = pyqtSignal(object, int, int)          # (rgb, h, w)
    frames_count_ready = pyqtSignal(int)
    spectrum_ready = pyqtSignal(object, object, int, int)  # (x, y, spectrum, x, y)

    def __init__(self):
        super().__init__()

        # ★ ALL shared state protected by this lock
        self._lock = threading.Lock()

        self._frames: list[np.ndarray] = []
        self._wavelengths: np.ndarray | None = None
        self._rgb_bands: tuple[int, int, int] = (0, 0, 0)
        self._dirty = False
        self._display_mapper = None  # ★ compatibility hook (unused internally)

        # pending spectrum request
        self._spec_request: tuple[int, int] | None = None

        self._timer: QTimer | None = None
        self._thread: QThread | None = None

    # ── lifecycle ─────────────────────────────────────────

    def start(self):
        """Move to a new QThread and start the internal timer."""
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._on_thread_started)
        self._thread.start()

    def _on_thread_started(self):
        self._timer = QTimer(self)
        self._timer.setInterval(CUBE_PREVIEW_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        logger.info("CubeRenderWorker started (%d fps)", CUBE_PREVIEW_FPS)

    def stop(self):
        """Stop worker — safe to call from any thread."""
        if self._timer is not None:
            self._timer.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread = None

    # ── thread-safe public API (called from GUI/main thread) ───

    def push_frame(self, frame: np.ndarray):
        """Append a frame. Called from main thread every camera frame."""
        with self._lock:
            self._frames.append(frame)
            self._dirty = True

    def set_wavelengths(self, wl: np.ndarray | None):
        with self._lock:
            self._wavelengths = wl

    def set_rgb_bands(self, r: int, g: int, b: int):
        with self._lock:
            self._rgb_bands = (r, g, b)
            self._dirty = True

    def request_spectrum(self, x: int, y: int):
        with self._lock:
            self._spec_request = (x, y)

    def set_display_mapper(self, mapper):
        """Compatibility hook — accepts but ignores mapper.

        Called by ScanController._on_display_mapper_changed() to prevent
        AttributeError. Actual display mapping is done in the GUI layer.
        """
        with self._lock:
            self._display_mapper = mapper

    def reset(self):
        """Clear all state. Called from main thread before new scan."""
        with self._lock:
            self._frames.clear()
            self._dirty = False
            self._spec_request = None

    # ── internal tick (runs on worker thread) ──────────────

    def _tick(self):
        """Called every CUBE_PREVIEW_INTERVAL_MS on worker thread."""
        with self._lock:
            frames = self._frames
            n = len(frames)
            dirty = self._dirty
            bands = self._rgb_bands
            spec_req = self._spec_request
            self._spec_request = None
            self._dirty = False

        if n == 0:
            return

        # emit frame count
        self.frames_count_ready.emit(n)

        # ── RGB composite ──
        if dirty:
            self._render_rgb(frames, bands)

        # ── spectrum request ──
        if spec_req is not None:
            self._render_spectrum(frames, spec_req)

    def _render_rgb(self, frames: list[np.ndarray], bands: tuple[int, int, int]):
        """Render RGB composite from current frames.

        Pre-allocates a single (H, N, 3) float32 buffer and fills
        R/G/B columns directly — zero intermediate allocations.
        """
        try:
            if not frames:
                return

            snapshot = list(frames)
            sample = snapshot[0]

            if sample.ndim != 2 or sample.size == 0:
                return

            h, w = sample.shape
            n = len(snapshot)
            max_band = w - 1

            r_band = max(0, min(bands[0], max_band))
            g_band = max(0, min(bands[1], max_band))
            b_band = max(0, min(bands[2], max_band))

            # pre-allocate output — (H, N, 3) float32
            rgb_f = np.empty((h, n, 3), dtype=np.float32)

            for i, f in enumerate(snapshot):
                rgb_f[:, i, 0] = f[:, r_band]
                rgb_f[:, i, 1] = f[:, g_band]
                rgb_f[:, i, 2] = f[:, b_band]

            # min-max normalize แบบ In-place (Zero-allocation)
            vmin = float(rgb_f.min())
            vmax = float(rgb_f.max())

            if vmax > vmin:
                rgb_f -= vmin
                rgb_f /= (vmax - vmin)
            else:
                rgb_f.fill(0.0)

            self.rgb_ready.emit(rgb_f, h, n)

        except Exception as e:
            logger.warning("RGB render error: %s", e)

    def _render_spectrum(
        self,
        frames: list[np.ndarray],
        spec_req: tuple[int, int],
    ):
        """Extract spectrum at (x, y) and emit."""
        try:
            x, y = spec_req
            if not frames:
                return

            sample = frames[0]
            if sample.ndim != 2:
                return

            h, w = sample.shape  # h = spatial, w = spectral bands

            # ★ FIX: x is frame index, check against len(frames) not w
            if not (0 <= y < h and 0 <= x < len(frames)):
                return

            arr = np.stack(frames, axis=0)  # (n, h, w)
            cube = arr.transpose(1, 0, 2)   # (h, n, w)

            spectrum = cube[y, x, :].astype(np.float32)

            if self._wavelengths is not None and len(self._wavelengths) == len(spectrum):
                x_data = self._wavelengths
            else:
                x_data = np.arange(len(spectrum), dtype=np.float32)

            self.spectrum_ready.emit(x_data, spectrum, x, y)

        except Exception as e:
            logger.warning("Spectrum render error: %s", e)