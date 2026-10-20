# workers/cube_render.py
"""Non-blocking cube render worker for the realtime preview.

Data conventions:
    The cube is ``(slit, scan, wavelength)``.  The worker receives it frame by
    frame: each frame is a 2-D ``(slit, wavelength)`` array (one camera
    exposure), so ``cube[y, x, :] == frames[x][y, :]`` with ``x`` the frame
    (scan) index.  The rendered RGB image is ``(slit rows, scan columns, 3)``
    float32 in ``0.0 .. 1.0`` (min-max normalised over everything rendered so
    far).  Wavelengths are in nm.

Threading:
    ``start()`` moves the worker to its own ``QThread``; a ``QTimer`` there
    calls ``_tick`` at 15 fps.  ``push_frame``, ``push_frames``, ``set_frames``,
    ``set_wavelengths``, ``set_rgb_bands``, ``request_spectrum``, ``reset`` and
    ``set_display_mapper`` may be called from the GUI thread at any time (state
    is guarded by a lock).  Results come back through the signals, which Qt
    delivers to the receiver's thread (queued to the GUI thread for widgets).

    A list given to ``set_frames`` is read in place on every tick while another
    thread may still append to it (e.g. the scan recorder): ``len`` and slicing
    of a list are atomic under the GIL, so no copy and no shared lock are needed.

Example:
    >>> import numpy as np
    >>> from PyQt5.QtCore import QCoreApplication                  # doctest: +SKIP
    >>> from modules.cube_preview import CubeRenderWorker          # doctest: +SKIP
    >>> app = QCoreApplication([])                                 # doctest: +SKIP
    >>> worker = CubeRenderWorker()                                # doctest: +SKIP
    >>> worker.rgb_ready.connect(lambda rgb, h, w: print(rgb.shape))  # doctest: +SKIP
    >>> worker.start()                                             # doctest: +SKIP
    >>> worker.push_frame(np.zeros((64, 100), np.uint16))          # doctest: +SKIP
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
    """Worker that renders the growing RGB preview and spectra on a dedicated QThread.

    Create it on the GUI thread, connect the signals, then call :meth:`start`.
    Call :meth:`stop` before discarding it (it joins the thread, up to 2 s).

    Signals:
        rgb_ready(object, int, int): ``(rgb, h, w)``; ``rgb`` is a float32
            ``(h, w, 3)`` array in 0..1 (``h`` slit rows, ``w`` frames rendered so
            far).  Emitted on every tick that has new frames or a band change.
        frames_count_ready(int): number of frames received so far; emitted every
            tick while at least one frame is stored.
        spectrum_ready(object, object, int, int): ``(x_data, spectrum, x, y)`` for a
            pixel requested with :meth:`request_spectrum`; ``x_data`` is the
            wavelength array in nm (or ``arange(bands)`` if none/length mismatch),
            ``spectrum`` float32 ``(bands,)``, ``x`` the frame (scan) index and
            ``y`` the slit row.

    Takes no constructor arguments.
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
        self._epoch = 0  # bumped by reset() so the worker drops its cached image
        # incremental RGB cache (worker thread only)
        self._raw: np.ndarray | None = None     # (h, capacity, 3) float32
        self._filled = 0
        self._raw_bands: tuple[int, int, int] | None = None
        self._rendered_from: list | None = None  # the list the cached columns were rendered from
        self._seen_epoch = 0
        self._vmin = 0.0
        self._vmax = 0.0
        self._display_mapper = None  # ★ compatibility hook (unused internally)

        # pending spectrum request
        self._spec_request: tuple[int, int] | None = None

        self._timer: QTimer | None = None
        self._thread: QThread | None = None

    # ── lifecycle ─────────────────────────────────────────

    def start(self):
        """Move the worker to a new QThread and start its render timer (call from the GUI thread)."""
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
        """Stop the timer and the thread (waits up to 2 s); call before discarding the worker."""
        # The timer lives in the worker thread: ending that thread's event loop stops it
        # (calling QTimer.stop() from this thread would violate Qt's thread affinity).
        if self._thread is None:
            return                      # already stopped (or never started): idempotent
        self._thread.quit()
        self._thread.wait(2000)
        self._thread = None

    # ── thread-safe public API (called from GUI/main thread) ───

    def push_frame(self, frame: np.ndarray):
        """Append one frame to the cube.

        Args:
            frame: 2-D ``(slit, wavelength)`` array.  Stored by reference (no copy) -
                the caller must not modify it afterwards.  Thread-safe.
        """
        with self._lock:
            self._frames.append(frame)
            self._dirty = True

    def push_frames(self, frames):
        """Append many frames at once.

        Args:
            frames: Iterable of 2-D ``(slit, wavelength)`` arrays, stored by reference
                (no copies).  Thread-safe.
        """
        with self._lock:
            self._frames.extend(frames)
            self._dirty = True

    def set_frames(self, frames: list):
        """Render from ``frames`` itself: a list its owner may keep appending to.

        The list is read by reference on every tick (never copied).  Columns already
        rendered from the previous list are kept for the leading frames both lists
        share (same objects), so a list cut at the end costs no re-render; otherwise
        the image is rendered again from frame 0.  Do not ``push_frame(s)`` into a list
        owned by someone else.  Thread-safe.

        Args:
            frames: List of 2-D ``(slit, wavelength)`` arrays.
        """
        with self._lock:
            self._frames = frames
            self._dirty = True

    def set_wavelengths(self, wl: np.ndarray | None):
        """Set the band wavelengths used for spectra.

        Args:
            wl: 1-D array in nm with one entry per spectral band, or ``None``.
        """
        with self._lock:
            self._wavelengths = wl

    def set_rgb_bands(self, r: int, g: int, b: int):
        """Choose the three spectral bands of the RGB composite (triggers a re-render).

        Args:
            r: Band index (column of a frame) for red.
            g: Band index for green.
            b: Band index for blue.  Indices are clamped to the frame width.
        """
        with self._lock:
            self._rgb_bands = (r, g, b)
            self._dirty = True

    def request_spectrum(self, x: int, y: int):
        """Ask for the spectrum of one pixel; answered via ``spectrum_ready`` on the next tick.

        Args:
            x: Frame (scan) index.
            y: Slit row index.
            Out-of-range requests are silently dropped.
        """
        with self._lock:
            self._spec_request = (x, y)

    def set_display_mapper(self, mapper):
        """Compatibility hook: stores ``mapper`` but it is not used for rendering.

        Args:
            mapper: Ignored (display mapping is done in the GUI layer).
        """
        with self._lock:
            self._display_mapper = mapper

    def reset(self):
        """Discard all frames and cached images (call before a new scan; thread-safe)."""
        with self._lock:
            self._frames = []   # fresh list; references held by the worker stay valid
            self._epoch += 1
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
            if self._epoch != self._seen_epoch:
                self._seen_epoch = self._epoch
                self._raw = None
                self._filled = 0
            # Band change or empty cache => refill from frame 0.
            if bands != self._raw_bands or self._raw is None:
                self._filled = 0
            cut = False
            if frames is not self._rendered_from:
                # Another list (set_frames): keep the columns of the frames both share.
                old, shared = self._rendered_from or [], 0
                limit = min(self._filled, n, len(old))
                while shared < limit and old[shared] is frames[shared]:
                    shared += 1
                cut = shared < self._filled
                self._filled = shared
                self._rendered_from = frames
            new_frames = frames[self._filled:n]

        if n == 0:
            return

        # emit frame count
        self.frames_count_ready.emit(n)

        # ── RGB composite (incremental: only new frames are copied) ──
        if dirty or new_frames:
            self._render_rgb(new_frames, n, bands, rescale=cut)

        # ── spectrum request ──
        if spec_req is not None:
            self._render_spectrum(frames[:n], spec_req)

    def _render_rgb(self, new_frames: list[np.ndarray], n: int,
                    bands: tuple[int, int, int], rescale: bool = False):
        """Update the cached (H, N, 3) float32 image with new frames and emit.

        Already rendered columns are not recomputed unless the band
        selection changed (refill) or the scan was reset.  ``rescale``: columns were
        cut off, so the min/max normalisation is taken again from those left.
        """
        try:
            if not new_frames and self._raw is None:
                return
            if rescale and self._filled:
                kept = self._raw[:, :self._filled]
                self._vmin, self._vmax = float(kept.min()), float(kept.max())
            if new_frames:
                sample = new_frames[0]
                if sample.ndim != 2 or sample.size == 0:
                    return
                h, w = sample.shape
                if self._raw is not None and self._raw.shape[0] != h:
                    self._raw = None
                    self._filled = 0
                    return  # frame geometry changed mid-scan; next tick refills
                if self._raw is None or self._raw.shape[1] < n:
                    grown = np.empty((h, n + n // 2 + 256, 3), dtype=np.float32)
                    if self._raw is not None and self._filled:
                        grown[:, :self._filled] = self._raw[:, :self._filled]
                    self._raw = grown
                self._raw_bands = bands
                raw = self._raw
                start = self._filled
                r_band, g_band, b_band = (max(0, min(b, w - 1)) for b in bands)
                for i, f in enumerate(new_frames):
                    col = start + i
                    raw[:, col, 0] = f[:, r_band]
                    raw[:, col, 1] = f[:, g_band]
                    raw[:, col, 2] = f[:, b_band]
                stop = start + len(new_frames)
                block = raw[:, start:stop]
                bmin, bmax = float(block.min()), float(block.max())
                if start == 0:
                    self._vmin, self._vmax = bmin, bmax
                else:
                    self._vmin = min(self._vmin, bmin)
                    self._vmax = max(self._vmax, bmax)
                self._filled = stop

            view = self._raw[:, :self._filled]
            vmin, vmax = self._vmin, self._vmax
            if vmax > vmin:
                rgb_f = (view - vmin) / (vmax - vmin)
            else:
                rgb_f = np.zeros_like(view)

            self.rgb_ready.emit(rgb_f, self._raw.shape[0], self._filled)

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

            # cube[y, x, :] == frames[x][y, :]; read one row, never stack the scan.
            spectrum = np.asarray(frames[x][y, :]).astype(np.float32)

            if self._wavelengths is not None and len(self._wavelengths) == len(spectrum):
                x_data = self._wavelengths
            else:
                x_data = np.arange(len(spectrum), dtype=np.float32)

            self.spectrum_ready.emit(x_data, spectrum, x, y)

        except Exception as e:
            logger.warning("Spectrum render error: %s", e)