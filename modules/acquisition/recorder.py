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
    :meth:`ScanRecorder.offer` is meant for the worker, the others for the GUI.  The list
    returned by :meth:`ScanRecorder.live_frames` is appended to by the worker without
    the reader holding the lock (``len`` and slicing of a list are atomic under the GIL).
"""
from __future__ import annotations

import threading

import numpy as np


def _inside(stamp, window):
    """-1 before ``window``, 1 after it, 0 inside (a ``None`` bound is open)."""
    start, end = window
    if start is not None and stamp < start:
        return -1
    if end is not None and stamp > end:
        return 1
    return 0


class ScanRecorder:
    """Collect the frames of one scan; a live preview reads the same list (:meth:`live_frames`)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._enabled = False
        self._frames: list = []
        self._raw: list = []
        self._stamps: list = []         # camera time (ns or None) of each binned frame
        self._raw_stamps: list = []     # ... and of each raw frame
        self._min_stamp = None          # frames stamped earlier are not recorded (set_min_stamp)
        self._live_dropped = 0          # ... and how many were refused so far
        self.last_trim = (0, 0)         # (frames dropped before, after) by the last take_frames
        self.last_live_dropped = 0      # frames refused by offer before the last take_frames
        # (first stamp, last stamp, frames without a stamp, frames) of the last take_frames
        self.last_stamps = (None, None, 0, 0)

    @property
    def enabled(self) -> bool:
        """True while frames are being recorded."""
        return self._enabled

    def set_recording(self, active: bool) -> None:
        """Arm or disarm recording; arming (False -> True) starts empty buffers and clears
        the :meth:`set_min_stamp` bound."""
        with self._lock:
            if active and not self._enabled:
                self._frames = []
                self._raw = []
                self._stamps = []
                self._raw_stamps = []
                self._min_stamp = None
                self._live_dropped = 0
            self._enabled = bool(active)

    def set_min_stamp(self, stamp) -> None:
        """Stop recording frames stamped before ``stamp`` (camera ns; ``None`` = no bound).

        Frames already recorded stay (``take_frames`` trims them); a frame without a
        camera time is always recorded.
        """
        with self._lock:
            self._min_stamp = stamp

    def live_frames(self) -> list:
        """The list the frames of the current recording are appended to (by reference).

        A live preview reads it while it grows; :meth:`take_frames` returns this same
        list object when nothing is trimmed.  Readers must not modify it.  Arming
        starts a new list, so ask again after :meth:`set_recording`.
        """
        with self._lock:
            return self._frames

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
            list: The kept binned frames, in order: the :meth:`live_frames` list itself
            when nothing is dropped, else a new list (``last_trim`` gives the counts
            dropped, ``last_stamps`` the first/last camera time of all recorded frames).
        """
        with self._lock:
            self._enabled = False
            frames, stamps, self._frames, self._stamps = self._frames, self._stamps, [], []
            self.last_trim = (0, 0)
            self.last_live_dropped, self._live_dropped = self._live_dropped, 0
            known = [s for s in stamps if s is not None]
            self.last_stamps = (known[0] if known else None, known[-1] if known else None,
                                len(stamps) - len(known), len(stamps))
            if window is None or any(s is None for s in stamps):
                return frames
            keep = [_inside(s, window) for s in stamps]
            self.last_trim = (sum(1 for k in keep if k < 0), sum(1 for k in keep if k > 0))
            if self.last_trim == (0, 0):
                return frames                   # the live list itself (a preview may show it)
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
                ``take_frames(window)`` and the :meth:`set_min_stamp` bound.

        Returns:
            True if the frame was recorded (not when disarmed, or stamped before the
            :meth:`set_min_stamp` bound).
        """
        with self._lock:
            if not self._enabled:
                return False
            if self._min_stamp is not None and stamp is not None and stamp < self._min_stamp:
                self._live_dropped += 1
                return False
            self._frames.append(binned)
            self._stamps.append(stamp)
            if keep_raw:
                self._raw.append(raw_frame.copy())
                self._raw_stamps.append(stamp)
            return True
