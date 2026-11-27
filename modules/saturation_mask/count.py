"""Basic saturation detection: clipped and near-saturated raw pixels counted in one pass.

Every acquired frame is read once, before dark subtraction, smile or binning, and two numbers
come out: ``clipped`` (``raw >= level``) and ``near`` (``raw >= near_ratio * level``, clipped
included). No mask is built, nothing is allocated per frame and the frame is not modified; the
full mask (:class:`~.core.SaturationPropagator`) reuses the counts as its fast path.

Implementation: a Numba kernel over the rows (``prange``), each row compared against both
thresholds in the same loop. :class:`RawCounter` runs it with a fixed number of threads chosen
by frame size (:data:`THREADS_BY_PIXELS`): ``numba.set_num_threads`` is local to the calling
thread and is restored after every call, so the count never changes the thread count the
lambda binning of the same worker runs with. One thread runs a plain loop (no thread pool).

Thresholds are integers for integer frames: ``near`` uses ``ceil(near_ratio * level)``, which
selects exactly the pixels ``raw >= near_ratio * level`` does (:func:`.core.detect_raw`).

Threading:
    :func:`count_saturated` is reentrant (``nogil``); a :class:`RawCounter` belongs to one
    thread (the acquisition worker).

Example:
    >>> import numpy as np
    >>> from modules.saturation_mask.count import count_saturated
    >>> count_saturated(np.array([[65520, 64300, 10]], np.uint16), 65520)
    (1, 2)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numba
import numpy as np
from numba import njit, prange

from .core import NEAR_RATIO

#: Thread count of the count by raw frame size: ``(max pixels, threads)``, first match. From the
#: acquisition-loop line rate (POA bins 1-3 on a 24-thread CPU, 2026-11-27): bin 3 (0.20 MP) is
#: fastest in one thread (a pool wake-up costs more than the count), bin 2 (0.44 MP) with 4, bin 1
#: (1.78 MP) with 8 (16 / 24 gain < 0.5 % line rate for 3-5 ms more CPU per frame).
THREADS_BY_PIXELS: Tuple[Tuple[float, int], ...] = ((300_000, 1), (1_000_000, 4), (math.inf, 8))
#: Dtypes compiled by :meth:`RawCounter.warm_up` (the raw formats the backends deliver).
WARM_DTYPES = (np.uint16, np.uint8)


@njit(cache=True, nogil=True)
def _count_single(image, level, near_level):
    rows, cols = image.shape
    clipped = 0
    near = 0
    for r in range(rows):
        for j in range(cols):
            v = image[r, j]
            clipped += v >= level
            near += v >= near_level
    return clipped, near


@njit(cache=True, nogil=True, parallel=True)
def _count_parallel(image, level, near_level):
    rows, cols = image.shape
    clipped = 0
    near = 0
    for r in prange(rows):
        c = 0
        n = 0
        for j in range(cols):
            v = image[r, j]
            c += v >= level
            n += v >= near_level
        clipped += c
        near += n
    return clipped, near


def near_threshold(level, near_ratio: float = NEAR_RATIO, integer: bool = True):
    """The near-saturation threshold: ``ceil(near_ratio * level)`` for integer frames (the
    same pixels as ``raw >= near_ratio * level``), else the float itself."""
    value = float(near_ratio) * float(level)
    return int(math.ceil(value)) if integer else value


def threads_for(pixels: int, table=THREADS_BY_PIXELS) -> int:
    """Threads for a frame of ``pixels`` pixels (:data:`THREADS_BY_PIXELS`), at most Numba's pool."""
    for limit, threads in table:
        if pixels <= limit:
            return max(1, min(int(threads), numba.config.NUMBA_NUM_THREADS))
    return 1


def count_saturated(raw, level, near_ratio: float = NEAR_RATIO, threads: Optional[int] = None) -> Tuple[int, int]:
    """``(clipped, near)`` pixel counts of a raw frame, in one pass.

    Args:
        raw: 2-D raw frame; not modified.
        level: Full scale (counts), > 0.
        near_ratio: Share of ``level`` that counts as near saturation, in ``(0, 1]``.
        threads: Numba threads (1: a plain loop); ``None``: :func:`threads_for` the frame size.

    Returns:
        tuple: ``(clipped, near)`` with ``near >= clipped``.

    Raises:
        ValueError: ``raw`` is not 2-D, ``level`` is not positive or ``near_ratio`` out of range.
    """
    raw = np.asarray(raw)
    if raw.ndim != 2:
        raise ValueError(f"raw frame must be 2-D, got shape {raw.shape}")
    level = int(level)
    if level <= 0:
        raise ValueError(f"saturation level must be positive, got {level}")
    if not 0.0 < float(near_ratio) <= 1.0:
        raise ValueError(f"near_ratio must be in (0, 1], got {near_ratio}")
    if raw.size == 0:
        return 0, 0
    if raw.dtype.kind not in "ui":                   # float / bool frames: NumPy, exact thresholds
        return (int(np.count_nonzero(raw >= level)),
                int(np.count_nonzero(raw >= near_threshold(level, near_ratio, integer=False))))
    info = np.iinfo(raw.dtype)
    if level > info.max:                             # the frames cannot reach it
        return 0, int(np.count_nonzero(raw >= near_threshold(level, near_ratio)))
    near_level = near_threshold(level, near_ratio)
    threads = threads_for(raw.size) if threads is None else max(1, int(threads))
    if threads == 1:
        clipped, near = _count_single(raw, level, near_level)
        return int(clipped), int(near)
    threads = min(threads, numba.config.NUMBA_NUM_THREADS)
    previous = numba.get_num_threads()               # thread-local: the binner's setting stays
    numba.set_num_threads(threads)
    try:
        clipped, near = _count_parallel(raw, level, near_level)
    finally:
        numba.set_num_threads(previous)
    return int(clipped), int(near)


class RawCounter:
    """:func:`count_saturated` with a fixed thread choice, warmed up before acquisition.

    Args:
        near_ratio: Share of the level that counts as near saturation.
        threads: Fixed thread count, or ``None`` for :data:`THREADS_BY_PIXELS`.
    """

    def __init__(self, near_ratio: float = NEAR_RATIO, threads: Optional[int] = None):
        self.near_ratio = float(near_ratio)
        self.threads = threads
        self.warm = False

    def count(self, raw, level) -> Tuple[int, int]:
        """``(clipped, near)`` of ``raw`` against ``level``."""
        return count_saturated(raw, level, self.near_ratio, self.threads)

    def warm_up(self, dtypes=WARM_DTYPES) -> None:
        """Compile (or load from Numba's cache) both kernels for ``dtypes`` and start the thread
        pool, so the first acquired frame does not pay for it. Call on the worker thread."""
        wanted = {self.threads} if self.threads else {threads for _, threads in THREADS_BY_PIXELS}
        pool = sorted({min(max(1, int(t)), numba.config.NUMBA_NUM_THREADS) for t in wanted} - {1})
        for dtype in dtypes:
            frame = np.zeros((4, 8), dtype=dtype)
            _count_single(frame, 1, 1)
            previous = numba.get_num_threads()
            try:
                for threads in pool or [min(2, numba.config.NUMBA_NUM_THREADS)]:
                    numba.set_num_threads(threads)      # each thread count's arena started too
                    _count_parallel(frame, 1, 1)
            finally:
                numba.set_num_threads(previous)
        self.warm = True


__all__ = ["THREADS_BY_PIXELS", "count_saturated", "near_threshold", "threads_for", "RawCounter"]
