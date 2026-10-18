"""Thread-safe per-frame scan recording buffers.

The acquisition worker appends **every** processed frame while recording is armed,
independent of how often the GUI displays frames; the GUI thread arms/disarms and
collects the result. Frames are stored by reference (no copy) except raw sensor frames,
which are copied because a camera backend may reuse its acquisition buffer.

Data conventions:
    Binned frames are ``(slit, wavelength_bin)`` ``uint16``; raw frames are
    ``(slit, spectral_pixel)`` ``uint16`` straight from the sensor. Index *i* of
    ``take_raw()`` is the sensor frame that produced index *i* of ``take_frames()``.

Threading:
    All methods take one internal lock and may be called from any thread.
    :meth:`ScanRecorder.offer` and :meth:`ScanRecorder.pop_batch` are meant for the
    worker, the others for the GUI.
"""
from __future__ import annotations

import threading
import time

import numpy as np

PREVIEW_BATCH_INTERVAL = 1.0 / 30  # recorded frames reach a live preview in batches this often


def _inside(stamp, window):
    """-1 before ``window``, 1 after it, 0 inside (a ``None`` bound is open)."""
    start, end = window
    if start is not None and stamp < start:
        return -1
    if end is not None and stamp > end:
        return 1
    return 0


class ScanRecorder:
    """Collect the frames of one scan, plus an optional batch feed for live previews."""

    def __init__(self):
        self._lock = threading.Lock()
        self._enabled = False
        self._frames: list = []
        self._raw: list = []
        self._stamps: list = []         # camera time (ns or None) of each binned frame
        self._raw_stamps: list = []     # ... and of each raw frame
        self._batch: list = []
        self._last_batch_time = 0.0
        self.last_trim = (0, 0)         # (frames dropped before, after) by the last take_frames

    @property
    def enabled(self) -> bool:
        """True while frames are being recorded."""
        return self._enabled

    def set_recording(self, active: bool) -> None:
        """Arm or disarm recording; arming (False -> True) starts empty buffers."""
        with self._lock:
            if active and not self._enabled:
                self._frames = []
                self._raw = []
                self._stamps = []
                self._raw_stamps = []
                self._batch = []
            self._enabled = bool(active)

    def count(self) -> int:
        """Number of frames recorded so far."""
        with self._lock:
            return len(self._frames)

    def take_frames(self, window=None) -> list:
        """Disarm recording and hand over (and empty) the recorded binned frames.

        Args:
            window: Optional ``(start_ns, end_ns)`` in camera time; either bound may be
                ``None``.  Frames stamped outside it (taken before the stage moved, or
                after it stopped) are dropped, together with their raw frames.  Nothing
                is dropped when any frame has no camera time.

        Returns:
            list: The kept binned frames, in order (``last_trim`` gives the counts dropped).
        """
        with self._lock:
            self._enabled = False
            frames, stamps, self._frames, self._stamps = self._frames, self._stamps, [], []
            self._batch = []
            self.last_trim = (0, 0)
            if window is None or any(s is None for s in stamps):
                return frames
            keep = [_inside(s, window) for s in stamps]
            self.last_trim = (sum(1 for k in keep if k < 0), sum(1 for k in keep if k > 0))
            if all(s is not None for s in self._raw_stamps):
                self._raw = [r for r, s in zip(self._raw, self._raw_stamps) if _inside(s, window) == 0]
            self._raw_stamps = []
            return [f for f, k in zip(frames, keep) if k == 0]

    def take_raw(self) -> list:
        """Hand over (and empty) the raw sensor frames kept alongside the binned ones."""
        with self._lock:
            raw, self._raw, self._raw_stamps = self._raw, [], []
            return raw

    def offer(self, binned: np.ndarray, raw_frame: np.ndarray, keep_raw: bool, stamp=None) -> bool:
        """Record one processed frame if armed (worker thread).

        Args:
            binned: The spectrally binned frame; stored by reference, so it must be a
                fresh array per call (``LambdaBinner.transform`` allocates one).
            raw_frame: The sensor frame it came from; copied when ``keep_raw``.
            keep_raw: Also keep the raw frame (the "Keep Spectrogram" option).
            stamp: Camera time of the frame in ns (``None`` if unknown), used by
                ``take_frames(window)``.

        Returns:
            True if the frame was recorded.
        """
        with self._lock:
            if not self._enabled:
                return False
            self._frames.append(binned)
            self._stamps.append(stamp)
            if keep_raw:
                self._raw.append(raw_frame.copy())
                self._raw_stamps.append(stamp)
            self._batch.append(binned)
            return True

    def pop_batch(self, now: float | None = None):
        """Return the frames recorded since the last batch if the interval elapsed, else ``None``.

        The batch holds references only; it is what a live cube preview consumes.
        """
        if not self._batch:
            return None
        now = time.monotonic() if now is None else now
        if now - self._last_batch_time < PREVIEW_BATCH_INTERVAL:
            return None
        with self._lock:
            batch, self._batch = self._batch, []
        self._last_batch_time = now
        return batch or None
