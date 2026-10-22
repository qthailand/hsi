"""Exact ROI statistics and pixel-exact ROI rasterisation for the cube viewer (numpy + stdlib only).

Data conventions:
    The cube is ``(H, W, B)`` = ``(slit, scan, wavelength)`` of any real numeric dtype - an
    ndarray, a ``numpy.memmap`` or a non-contiguous view; it is only read, never copied
    whole, and stays owned by the caller.  Statistics are float64 in the cube's own units
    (counts for integer cubes), one value per band in the cube's band order.

    Float *image-corner* coordinates: pixel ``(c, r)`` (column = scan, row = slit) covers
    ``[c, c + 1) x [r, r + 1)`` and its centre is ``(c + 0.5, r + 0.5)``.  A pixel belongs to an
    ROI when its centre does:

    * ``snap_rect`` - closed rule, a centre on the rectangle's edge is inside:
      ``c0 = ceil(min(x) - 0.5)``, ``c1 = floor(max(x) - 0.5) + 1`` (rows alike), clipped to
      the image, half-open.  ``xmin = 3.5`` therefore includes column 3 (``round()`` would
      not), and a drag that encloses no centre gives an empty box.
    * ``polygon_mask`` - even-odd rule at pixel centres, bit-identical to
      ``matplotlib.path.Path(vertices).contains_points(centres)`` - including centres that lie
      exactly on an edge or a vertex - but computed by a vectorised scanline (a few ms) and
      only inside the polygon's clipped bounding box.

Statistics (``roi_stats``):
    Per band: valid-pixel count, mean, population std (ddof=0), min, max and the number of
    values ``>= saturation`` (counted, not excluded).

    * ``'exact-int'`` - integer cubes of at most 16 bits.  The sums ``S1 = sum(x)`` and
      ``S2 = sum(x**2)`` are exact integers: each chunk is summed in float64, where every
      partial sum is an integer below 2**53 (the chunk height is capped to guarantee it),
      and the chunk sums are accumulated in uint64 (``S1`` of a signed cube in int64).
      This gives the same sums as ``np.add.reduce(dtype=uint64)`` over uint32 squares,
      2.5x faster (22 vs 56 ms for a 400 x 400 x 185 box).  The finish uses Python
      integers: ``mean = S1 / N`` and ``std = sqrt((N * S2 - S1**2) / N**2)`` are both
      *correctly rounded* (0 ulp from the exact value).  ``math.sqrt(float(var))`` would be
      1 ulp off through double rounding (measured: 15 of 185 bands of a random 400 x 400
      box, 13 % of 200 000 random ratios), so the square root is rounded exactly.  An ignore value ``v`` is removed exactly per band (``S1 -= v k``,
      ``S2 -= v**2 k``, ``N -= k``).  ``sum1`` / ``sum2`` are kept, so ddof=1 or exact
      pooling of several ROIs can be derived later.
    * ``'float64-two-pass'`` - float cubes and wider integers.  Each bbox row ``r`` gets a
      two-pass mean ``m_r`` (rounded to float64), the sum of squared deviations ``Q_r`` and
      the residual ``e_r = sum(x - m_r)`` (not 0, because ``m_r`` is rounded).  The rows are
      merged about the rounded ROI mean ``mu`` with the full decomposition
      ``M2 = sum(Q_r) + sum(n_r (m_r - mu)**2) + 2 sum((m_r - mu) e_r) - E**2 / n``, where
      ``E = sum(n_r (m_r - mu) + e_r)`` is the residual of ``mu`` itself; the sums are
      ``math.fsum``.  The std is then within 1 ulp of the exact value in every case
      measured against exact rational references (rects and lassos; float32, float64 and
      int32 data with offsets up to 1e15) - also where the error of a plain merge grows
      with (offset / spread)**2: without the two terms ``2e9 + {0..3}`` was 3e-9 relative
      off, and ``numpy.std`` itself is about 4e-9 off for ``1e12 + N(0, 1)``.  The per-row work
      does not depend on the chunking, so any ``chunk_bytes`` gives bit-identical results.
      NaN, +-inf and the ignore value are skipped per band and are not counted in
      ``n_valid``.

    A band without valid pixels has NaN mean / std / min / max (the caller marks the ROI
    failed).  ``point_stats`` is the single-pixel record (``mean == cube[iy, ix, :]`` as
    float64, std 0).

Memory:
    The cube - ``(H, W, B)`` ndarray, ``numpy.memmap`` or a non-contiguous view such as the
    transposed memmap of a BSQ / BIL file - is never copied whole: ``roi_stats`` reads the
    ROI's bounding box in row chunks of about ``chunk_bytes`` (default 1 MiB) and needs a
    few MB of scratch space.  Rows of the box that hold no ROI pixel are not read at all.

Measured cost (i9-12900K, numpy 2.1, in-RAM 1024 x 1762 x 185 uint16 cube, 637 MiB; peak
extra memory from tracemalloc):
    * ``roi_stats`` 100 x 100 rect: 3.2-3.4 ms, 5 MiB.  400 x 400 rect: 32-33 ms, 5 MiB.
      20 % lasso (361k px): 80-81 ms, 4 MiB; whole image: 350-360 ms, 3 MiB.  The same
      ROIs through a BSQ-layout view: 400 x 400 49-51 ms, lasso 114-115 ms.  A float32
      cube (two-pass path), min of 7 runs in one session, without -> with the two merge
      correction terms (+10-30 %): 100 x 100 7.0-7.8 -> 9.3-10.2 ms, 400 x 400 74-87 ->
      85-99 ms, a 628k-px wavy-ellipse lasso 337-383 -> 398-419 ms.  The exact-int path
      is unchanged.
    * ``polygon_mask`` for that 300-vertex lasso: 1.0 ms, 1 MiB.  A pathological
      5000-vertex full-height zigzag (5 M edge-row crossings): 0.74 s, 13 MiB, in batches
      of about 10 ms (65,536 edge-row crossings) between which ``cancel`` is polled.
    * ``point_stats``: 26 us.

Threading:
    All functions are pure numpy (they release the GIL in the heavy loops) and safe to call
    from a worker thread.  ``cancel()`` is polled once per chunk by ``roi_stats`` and once
    per batch of crossings by ``polygon_mask``; both then raise ``RoiCancelled``.

Example:
    >>> import numpy as np
    >>> cube = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
    >>> snap_rect(0.5, 0.6, 2.5, 2.4, 4, 5)
    (0, 1, 3, 2)
    >>> roi_stats(cube, (0, 1, 3, 2)).mean.tolist()
    [18.0, 19.0, 20.0]
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

__all__ = ("RoiCancelled", "RoiStats", "snap_rect", "polygon_mask", "point_stats", "roi_stats")

# |vertex| bound for polygon_mask: keeps every float step of the scanline exact (|v| << 2**52).
_COORD_LIMIT = 1e9
# float64 holds every integer below 2**53 exactly: the exact-int chunk sums stay below it.
_FLOAT_EXACT = 1 << 53
# A count array denser than 1/_SPARSE_DIV is summed per column; sparser ones via bincount.
_SPARSE_DIV = 16
# uint8 column sums are done in blocks of this many rows (no uint8 overflow).
_U8_BLOCK = 255
# polygon_mask handles at most about this many (edge, row) crossings per batch (bounded memory).
_PAIR_BATCH = 1 << 16


class RoiCancelled(Exception):
    """Raised by :func:`roi_stats` and :func:`polygon_mask` when their ``cancel()`` callback returns True."""


@dataclass(frozen=True, eq=False)
class RoiStats:
    """Per-band statistics of one ROI (immutable; the arrays are read-only).

    Attributes:
        n_px: Pixels in the ROI (before any NaN / ignore-value exclusion).
        n_valid: ``(B,)`` int64 - pixels that entered the statistics of each band.
        mean, std, min, max: ``(B,)`` float64; ``std`` is the population std (ddof=0).
            NaN for a band without valid pixels.
        n_saturated: ``(B,)`` int64 - valid values ``>= saturation`` (0 when no level was
            given).  Saturated values are counted, not excluded.
        sum1, sum2: ``(B,)`` exact ``sum(x)`` and ``sum(x**2)`` over the valid values for
            integer cubes of at most 16 bits, else None.  ``sum2`` is uint64; ``sum1`` is
            uint64 for unsigned cubes and int64 for signed ones.
        method: ``'point'``, ``'exact-int'`` or ``'float64-two-pass'``.

    Compared by identity (``eq=False``): the fields are arrays.
    """

    n_px: int
    n_valid: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    n_saturated: np.ndarray
    sum1: Optional[np.ndarray]
    sum2: Optional[np.ndarray]
    method: str

    def __post_init__(self):
        """Coerce the fields to their documented dtypes, check the shapes, freeze the arrays."""
        object.__setattr__(self, "n_px", operator.index(self.n_px))
        fields = [("n_valid", np.int64), ("mean", np.float64), ("std", np.float64),
                  ("min", np.float64), ("max", np.float64), ("n_saturated", np.int64),
                  ("sum1", None), ("sum2", np.uint64)]
        shape = None
        for name, dtype in fields:
            value = getattr(self, name)
            if value is None and name in ("sum1", "sum2"):
                continue
            arr = np.array(value, dtype=dtype)          # always a private copy
            if arr.ndim != 1 or (shape is not None and arr.shape != shape):
                raise ValueError(f"RoiStats.{name}: expected a 1-D array of the band count, "
                                 f"got shape {arr.shape}")
            shape = arr.shape
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        if self.method not in ("point", "exact-int", "float64-two-pass"):
            raise ValueError(f"RoiStats.method: unknown method {self.method!r}")


# --------------------------------------------------------------------------- geometry


def _dims(height, width):
    """Return the image size as two non-negative ints (ValueError otherwise)."""
    h, w = operator.index(height), operator.index(width)
    if h < 0 or w < 0:
        raise ValueError(f"image size must be non-negative, got {h} x {w}")
    return h, w


def _snap_span(lo: float, hi: float, n: int):
    """Half-open index span of the pixel centres in the closed interval ``[lo, hi]``, clipped to ``[0, n]``."""
    a = math.ceil(lo - 0.5)
    b = math.floor(hi - 0.5) + 1
    return min(max(a, 0), n), min(max(b, 0), n)


def snap_rect(x0, y0, x1, y1, height, width):
    """Snap a dragged rectangle to the pixels whose centres it encloses (closed rule).

    Args:
        x0, y0, x1, y1: Two opposite corners in float image-corner coordinates, any order.
        height, width: Image size in pixels.

    Returns:
        ``(c0, r0, c1, r1)`` ints, half-open and clipped to the image:
        ``c0 = ceil(min(x) - 0.5)``, ``c1 = floor(max(x) - 0.5) + 1`` and the same for
        rows.  The box is empty when ``c1 <= c0`` or ``r1 <= r0`` (for example a sub-pixel
        drag between two centres, or a drag outside the image).

    Raises:
        ValueError: A coordinate is not finite, or the size is negative.
    """
    h, w = _dims(height, width)
    xs = (float(x0), float(x1))
    ys = (float(y0), float(y1))
    if not all(math.isfinite(v) for v in xs + ys):
        raise ValueError("snap_rect: coordinates must be finite")
    c0, c1 = _snap_span(min(xs), max(xs), w)
    r0, r1 = _snap_span(min(ys), max(ys), h)
    return c0, r0, c1, r1


def polygon_mask(vertices, height, width, *, cancel: Optional[Callable[[], bool]] = None):
    """Rasterise a closed polygon (lasso) with the even-odd rule at pixel centres.

    The result equals ``matplotlib.path.Path(vertices).contains_points`` evaluated at
    every pixel centre, bit for bit (matplotlib's crossing test - ``vy >= ty`` half-open
    rows, ``>=`` / ``>`` column ties - is reproduced with the same float64 operations),
    so centres exactly on an edge or a vertex get the same answer.  The ring closes
    itself (last vertex back to the first); self-intersecting shapes such as a figure-8
    follow the even-odd rule.  (Pixels outside the snapped bounding box are outside by
    construction; matplotlib could only disagree there through rounding when a vertex
    lies within about 1e-16 of the edge length from a pixel centre.)

    Args:
        vertices: ``(N, 2)`` float ``(x, y)`` image-corner coordinates; vertices may lie
            outside the image.  Fewer than 3 vertices give an empty mask (as matplotlib).
        height, width: Image size in pixels.
        cancel: Optional callable polled before every batch of about 65,536 edge-row
            crossings (about 10 ms of work) and before the final pass; when it returns
            True the rasterisation stops with ``RoiCancelled``.

    Returns:
        ``(c0, r0, mask)``: ``mask`` is bool, local to the polygon's bounding box snapped
        like :func:`snap_rect` and clipped to the image, so pixel ``(c0 + j, r0 + i)`` is
        inside when ``mask[i, j]``; it may have all-False border rows / columns.  An empty
        box gives a mask of size 0.

    Raises:
        ValueError: Wrong shape, a non-finite vertex, or ``|vertex| > 1e9``.
        RoiCancelled: ``cancel()`` returned True.
    """
    h_img, w_img = _dims(height, width)
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError(f"polygon_mask: vertices must have shape (N, 2), got {v.shape}")
    if v.size:
        if not np.isfinite(v).all():
            raise ValueError("polygon_mask: vertices must be finite")
        if np.abs(v).max() > _COORD_LIMIT:
            raise ValueError(f"polygon_mask: |vertex| must be <= {_COORD_LIMIT:g}")
    if v.shape[0] < 3:
        return 0, 0, np.zeros((0, 0), dtype=bool)
    xa, ya = v[:, 0], v[:, 1]
    c0, r0, c1, r1 = snap_rect(xa.min(), ya.min(), xa.max(), ya.max(), h_img, w_img)
    h, w = r1 - r0, c1 - c0
    if h <= 0 or w <= 0:
        return c0, r0, np.zeros((max(h, 0), max(w, 0)), dtype=bool)

    # Edge e runs from vertex e (x0, y0) to vertex e+1 (x1, y1); the last one closes the ring.
    xb, yb = np.roll(xa, -1), np.roll(ya, -1)
    lo, hi = np.minimum(ya, yb), np.maximum(ya, yb)
    # An edge crosses the row-centre line ty = r + 0.5 when exactly one end has y >= ty, i.e.
    # lo < ty <= hi.  Wherever the result is not clipped away (y > -0.5), y - 0.5 is exact or
    # rounds without crossing an integer, so floor() gives the exact rows.
    first = np.clip(np.floor(lo - 0.5).astype(np.int64) + 1, r0, r1)
    stop = np.clip(np.floor(hi - 0.5).astype(np.int64) + 1, r0, r1)
    counts = stop - first
    sel = np.flatnonzero(counts > 0)
    mask_par = np.zeros((h, w + 1), dtype=np.uint8)
    # One work item per (edge, row) pair, in batches of about _PAIR_BATCH pairs: scratch memory
    # stays bounded (a 5000-vertex full-height zigzag has 5 M pairs).
    ends = np.cumsum(counts[sel])
    start = 0
    while start < sel.size:
        if cancel is not None and cancel():
            raise RoiCancelled()
        done = int(ends[start - 1]) if start else 0
        stop_e = max(start + 1, int(np.searchsorted(ends, done + _PAIR_BATCH, side="right")))
        es = sel[start:stop_e]
        cnt = counts[es]
        e = np.repeat(es, cnt)
        offsets = np.cumsum(cnt) - cnt
        rows = np.repeat(first[es] - offsets, cnt) + np.arange(e.size, dtype=np.int64)
        k = _crossing_prefix(xa[e], ya[e], xb[e], yb[e], rows + 0.5, c0, w)
        # Pair (edge, row) toggles the local columns [0, k) of that row; record the parity at k.
        np.bitwise_xor.at(mask_par, (rows - r0, k), 1)
        start = stop_e
    if cancel is not None and cancel():
        raise RoiCancelled()
    row_par = np.bitwise_xor.reduce(mask_par, axis=1)                # parity of all crossings
    inside = np.bitwise_xor.accumulate(mask_par[:, :w], axis=1)      # parity of k <= c
    inside ^= row_par[:, None]                                       # -> parity of k > c
    return c0, r0, inside.view(bool)


def _crossing_prefix(vx0, vy0, vx1, vy1, ty, c0, w):
    """Local column count ``k`` (0..w) such that the pair toggles exactly the columns ``[0, k)``.

    matplotlib's test for the point ``(tx, ty)`` against the edge ``(vx0, vy0) -> (vx1, vy1)``
    is ``((vy1 - ty) * (vx0 - vx1) >= (vx1 - tx) * (vy0 - vy1)) == (vy1 >= ty)``.  Every
    float step in it is monotone in ``tx``, so the toggled columns form a prefix; the
    crossing abscissa gives a first guess that is then corrected with the exact test.
    """
    yflag1 = vy1 >= ty
    dx = vx0 - vx1
    dy = vy0 - vy1                       # never 0: the edge crosses the row
    lhs = (vy1 - ty) * dx
    xi = vx1 + (ty - vy1) * dx / dy      # crossing abscissa (approximate)
    k = np.clip(np.floor(xi - 0.5) + 1.0 - c0, 0, w).astype(np.int64)

    def toggles(i, col):
        """Exact matplotlib test of pairs ``i`` at local columns ``col``."""
        tx = (col + c0).astype(np.float64) + 0.5
        return (lhs[i] >= (vx1[i] - tx) * dy[i]) == yflag1[i]

    todo = np.arange(k.size)
    # Usually 1-2 rounds (the guess is within a column); a monotone test needs at most w + 1.
    for _ in range(w + 2):
        if not todo.size:
            return k
        kk = k[todo]
        up = kk < w
        up[up] = toggles(todo[up], kk[up])                 # column k still toggled -> k + 1
        down = kk > 0
        down[down] = ~toggles(todo[down], kk[down] - 1)    # column k - 1 not toggled -> k - 1
        k[todo] += up.astype(np.int64) - down.astype(np.int64)
        todo = todo[up | down]
    raise RuntimeError("polygon_mask: the crossing search did not converge")  # pragma: no cover


# --------------------------------------------------------------------------- statistics helpers


def _check_cube(cube):
    """Return ``(H, W, B)`` of a 3-D numeric cube (TypeError / ValueError otherwise)."""
    shape = getattr(cube, "shape", None)
    dtype = getattr(cube, "dtype", None)
    if shape is None or dtype is None or len(shape) != 3:
        raise ValueError(f"cube must be a 3-D array (H, W, B), got shape {shape}")
    if np.dtype(dtype).kind not in "biuf":
        raise TypeError(f"cube dtype {dtype} is not a real numeric type")
    return tuple(int(s) for s in shape)


def _exact_capable(dtype) -> bool:
    """True for integer dtypes of at most 16 bits (the exact-int path)."""
    dtype = np.dtype(dtype)
    return dtype.kind in "iu" and dtype.itemsize <= 2


def _int_value(value, dtype):
    """``value`` as an int of ``dtype``, or None when no element of ``dtype`` can equal it."""
    if value is None:
        return None
    v = float(value)
    if not math.isfinite(v) or v != math.floor(v):
        return None
    info = np.iinfo(dtype)
    iv = int(value) if isinstance(value, (int, np.integer)) else int(v)
    return iv if info.min <= iv <= info.max else None


def _native_value(value, dtype):
    """``value`` cast to the cube's float / wide-int ``dtype`` for equality tests, or None."""
    if value is None:
        return None
    dtype = np.dtype(dtype)
    if dtype.kind in "iu":
        iv = _int_value(value, dtype)
        return None if iv is None else dtype.type(iv)
    if dtype.kind == "b":
        return None if value not in (0, 1) else np.bool_(value)
    v = float(value)
    if math.isnan(v):
        return None                      # NaN is skipped anyway
    with np.errstate(over="ignore"):
        native = dtype.type(v)
    return native if np.isfinite(native) or math.isinf(v) else None


def _count_true(flags, n_bands):
    """Per-column count of a 2-D bool array ``(n, B)`` (int64), sparse- and dense-fast."""
    total = int(np.count_nonzero(flags))
    if total == 0:
        return None
    if total * _SPARSE_DIV < flags.size:
        return np.bincount(np.flatnonzero(flags) % n_bands, minlength=n_bands).astype(np.int64)
    as_u8 = flags.view(np.uint8)
    out = np.zeros(n_bands, dtype=np.int64)
    for a in range(0, as_u8.shape[0], _U8_BLOCK):
        out += np.add.reduce(as_u8[a:a + _U8_BLOCK], axis=0, dtype=np.uint8)
    return out


def _odd_mantissa(f: float) -> bool:
    """True when the last mantissa bit of the positive normal float ``f`` is 1."""
    m, _ = math.frexp(f)
    return int(m * (1 << 53)) & 1 == 1


def _cmp_sqrt_mid(num: int, den: int, a: float, b: float) -> int:
    """Sign of ``sqrt(num / den) - (a + b) / 2`` computed exactly (a, b >= 0)."""
    pa, qa = a.as_integer_ratio()
    pb, qb = b.as_integer_ratio()
    p = pa * qb + pb * qa                # (a + b) / 2 = p / q exactly
    q = 2 * qa * qb
    lhs, rhs = num * q * q, p * p * den
    return (lhs > rhs) - (lhs < rhs)


def _sqrt_ratio(num: int, den: int) -> float:
    """Correctly rounded ``sqrt(num / den)`` for ints ``num >= 0``, ``den > 0`` (ties to even)."""
    if num <= 0:
        return 0.0
    f = math.sqrt(num / den)             # within 1 ulp; walk to the float whose interval holds it
    for _ in range(64):
        lo = math.nextafter(f, 0.0)
        c = _cmp_sqrt_mid(num, den, lo, f)
        if c < 0 or (c == 0 and _odd_mantissa(f)):
            f = lo
            continue
        hi = math.nextafter(f, math.inf)
        c = _cmp_sqrt_mid(num, den, f, hi)
        if c > 0 or (c == 0 and _odd_mantissa(f)):
            f = hi
            continue
        break
    return f


def _freeze_empty(n_px, n_bands, exact, signed):
    """Stats of an ROI without pixels: zero counts and NaN values."""
    nan = np.full(n_bands, np.nan)
    zeros = np.zeros(n_bands, dtype=np.int64)
    sum1 = np.zeros(n_bands, dtype=np.int64 if signed else np.uint64) if exact else None
    sum2 = np.zeros(n_bands, dtype=np.uint64) if exact else None
    return RoiStats(n_px, zeros, nan, nan, nan, nan, zeros, sum1, sum2,
                    "exact-int" if exact else "float64-two-pass")


# --------------------------------------------------------------------------- public statistics


def point_stats(cube, ix, iy, *, saturation=None, ignore_value=None) -> RoiStats:
    """Single-pixel record: ``mean == cube[iy, ix, :].astype(float64)``, std 0, ``n_px`` 1.

    Args:
        cube: ``(H, W, B)`` array (ndarray, memmap or view).
        ix, iy: Column (scan) and row (slit) of the pixel.
        saturation: Optional level; values ``>=`` it count in ``n_saturated``.
        ignore_value: Optional ENVI 'data ignore value': a band holding it counts as no
            data (``n_valid`` 0, NaN mean/std/min/max, sums 0), as in :func:`roi_stats`.

    Returns:
        ``RoiStats`` with method ``'point'``; ``min == max == mean``.  A non-finite value
        (float cubes) gets ``n_valid`` 0 and std NaN in that band, while ``mean`` keeps the
        raw value.  ``sum1`` / ``sum2`` are set for integer cubes of at most 16 bits.

    Raises:
        IndexError: The pixel is outside the image.
    """
    h, w, n_bands = _check_cube(cube)
    ix, iy = operator.index(ix), operator.index(iy)
    if not (0 <= ix < w and 0 <= iy < h):
        raise IndexError(f"pixel ({ix}, {iy}) is outside the {w} x {h} image")
    raw = np.asarray(cube[iy, ix, :])
    mean = raw.astype(np.float64)
    finite = np.isfinite(mean)
    ignored = np.zeros(n_bands, dtype=bool)
    if ignore_value is not None:
        ignored = finite & (mean == float(ignore_value))
        finite = finite & ~ignored
        mean = np.where(ignored, np.nan, mean)
    sat = np.zeros(n_bands, dtype=np.int64)
    if saturation is not None:
        sat = (finite & (mean >= float(saturation))).astype(np.int64)
    sum1 = sum2 = None
    if _exact_capable(raw.dtype):
        sum1 = raw.astype(np.int64 if raw.dtype.kind == "i" else np.uint64)
        wide = raw.astype(np.int64)
        sum2 = (wide * wide).astype(np.uint64)
        sum1[ignored] = 0
        sum2[ignored] = 0
    return RoiStats(1, finite.astype(np.int64), mean, np.where(finite, 0.0, np.nan),
                    mean, mean, sat, sum1, sum2, "point")


def roi_stats(cube, bbox, local_mask=None, *, ignore_value=None, saturation=None,
              cancel: Optional[Callable[[], bool]] = None, chunk_bytes: int = 1 << 20) -> RoiStats:
    """Mean, std (ddof=0), min, max, valid and saturated counts of an ROI, per band.

    Args:
        cube: ``(H, W, B)`` array - ndarray, ``numpy.memmap`` or a non-contiguous view;
            only the bounding box is read, in row chunks.
        bbox: ``(c0, r0, c1, r1)`` half-open pixel box inside the image (``snap_rect`` /
            ``polygon_mask`` output).
        local_mask: Optional bool ``(r1 - r0, c1 - c0)`` mask (``polygon_mask``); None
            means the whole box.
        ignore_value: A value excluded per band (ENVI 'data ignore value'), or None.  For
            integer cubes it only matters when it is an integer the dtype can hold.
        saturation: Values ``>=`` this level are counted in ``n_saturated`` (not excluded).
        cancel: Optional callable polled once per chunk; when it returns True the
            computation stops with ``RoiCancelled``.
        chunk_bytes: Approximate bytes of cube data per chunk (whole box rows; at least one
            row per chunk).  The result does not depend on it.

    Returns:
        ``RoiStats``: method ``'exact-int'`` for integer cubes of at most 16 bits (exact
        sums, correctly rounded mean and std), otherwise ``'float64-two-pass'`` (NaN and
        +-inf skipped per band).  A band without valid pixels gets NaN statistics.

    Raises:
        ValueError: Bad cube shape, bbox or mask shape.
        TypeError: Non-numeric cube.
        RoiCancelled: ``cancel()`` returned True.
    """
    h_img, w_img, n_bands = _check_cube(cube)
    c0, r0, c1, r1 = (operator.index(v) for v in bbox)
    if not (0 <= c0 <= c1 <= w_img and 0 <= r0 <= r1 <= h_img):
        raise ValueError(f"bbox {(c0, r0, c1, r1)} is not inside the {w_img} x {h_img} image")
    h, w = r1 - r0, c1 - c0
    mask = None
    if local_mask is not None:
        mask = np.asarray(local_mask, dtype=bool)
        if mask.shape != (h, w):
            raise ValueError(f"local_mask shape {mask.shape} != bbox shape {(h, w)}")
        row_counts = np.count_nonzero(mask, axis=1).astype(np.int64)
    else:
        row_counts = np.full(h, w, dtype=np.int64)
    n_px = int(row_counts.sum())
    dtype = np.dtype(cube.dtype)
    exact = False
    if _exact_capable(dtype):
        info = np.iinfo(dtype)
        max_sq = max(-int(info.min), int(info.max)) ** 2
        # S2 must fit uint64 and one bbox row of float64 partial sums must stay exact.
        exact = n_px * max_sq < (1 << 64) and max(w, 1) * max_sq <= _FLOAT_EXACT
    if n_px == 0:
        return _freeze_empty(0, n_bands, exact, dtype.kind == "i")
    rows_per = max(1, int(chunk_bytes) // max(1, w * n_bands * dtype.itemsize))
    if exact:
        rows_per = min(rows_per, max(1, _FLOAT_EXACT // (max(w, 1) * max_sq)))
    chunks = _ChunkReader(cube, c0, r0, w, mask, row_counts, rows_per, cancel)
    if exact:
        return _exact_int_stats(chunks, n_px, n_bands, dtype, ignore_value, saturation)
    return _float_stats(chunks, n_px, n_bands, dtype, ignore_value, saturation)


class _ChunkReader:
    """Iterate ``(row_counts, x)`` over row chunks of the bbox; ``x`` is ``(n, B)`` C-contiguous.

    Chunks without ROI pixels are skipped without reading the cube; ``cancel`` is polled
    before every chunk.  ``max_px`` is the largest pixel count of any chunk.
    """

    def __init__(self, cube, c0, r0, w, mask, row_counts, rows_per, cancel):
        """Store the geometry; nothing is read until iteration."""
        self.cube, self.c0, self.r0, self.w = cube, c0, r0, w
        self.mask, self.row_counts, self.rows_per, self.cancel = mask, row_counts, rows_per, cancel
        h = row_counts.size
        bounds = np.arange(0, h + rows_per, rows_per).clip(max=h)
        cums = np.concatenate(([0], np.cumsum(row_counts)))
        self.max_px = int(np.max(cums[bounds[1:]] - cums[bounds[:-1]])) if h else 0

    def __iter__(self):
        """Yield ``(counts_of_rows_in_chunk, gathered_pixels)`` per non-empty chunk.

        ``gathered_pixels`` is a view of one buffer reused for every chunk (no per-chunk
        allocation): it is only valid until the next chunk is requested.
        """
        h = self.row_counts.size
        n_bands = self.cube.shape[2]
        xbuf = np.empty((max(self.max_px, 1), n_bands), dtype=self.cube.dtype)
        strides = getattr(self.cube, "strides", None)
        bands_contiguous = strides is not None and strides[2] == self.cube.dtype.itemsize
        for a in range(0, h, self.rows_per):
            if self.cancel is not None and self.cancel():
                raise RoiCancelled()
            b = min(h, a + self.rows_per)
            counts = self.row_counts[a:b]
            if not counts.any():
                continue
            block = self.cube[self.r0 + a:self.r0 + b, self.c0:self.c0 + self.w]
            x = xbuf[:int(counts.sum())]
            if self.mask is None:
                np.copyto(x.reshape(b - a, self.w, n_bands), block)
            elif not bands_contiguous:
                # BSQ / BIL views: numpy's boolean gather beats a per-row compress (measured).
                x = np.ascontiguousarray(block[self.mask[a:b]])
            else:
                p = 0
                for i, c in enumerate(counts.tolist()):
                    if c:
                        np.compress(self.mask[a + i], block[i], axis=0, out=x[p:p + c])
                        p += c
            yield counts, x


def _exact_int_stats(chunks, n_px, n_bands, dtype, ignore_value, saturation):
    """Exact-int path: integer S1 / S2 per band, Python-int finish (correctly rounded)."""
    signed = dtype.kind == "i"
    info = np.iinfo(dtype)
    ignore = _int_value(ignore_value, dtype)
    sat_thr = None
    if saturation is not None:
        thr = math.ceil(float(saturation))
        if thr <= info.max:
            sat_thr = dtype.type(max(thr, int(info.min)))
    s1 = np.zeros(n_bands, dtype=np.int64 if signed else np.uint64)
    s2 = np.zeros(n_bands, dtype=np.uint64)
    n_ign = np.zeros(n_bands, dtype=np.int64)
    n_sat = np.zeros(n_bands, dtype=np.int64)
    gmin = np.full(n_bands, info.max, dtype=dtype)
    gmax = np.full(n_bands, info.min, dtype=dtype)
    buf = np.empty((max(chunks.max_px, 1), n_bands), dtype=np.float64)
    for _, x in chunks:
        f = buf[:x.shape[0]]
        np.copyto(f, x)                  # exact: |x| <= 32768; every partial sum < 2**53
        s1 += f.sum(axis=0).astype(s1.dtype)
        s2 += np.einsum("ij,ij->j", f, f).astype(np.uint64)
        cmin, cmax = x.min(axis=0), x.max(axis=0)
        eq = None
        if ignore is not None and ((cmin <= ignore) & (ignore <= cmax)).any():
            eq = x == dtype.type(ignore)
            counted = _count_true(eq, n_bands)
            if counted is None:
                eq = None
            else:
                n_ign += counted
                cmin = np.where(eq, info.max, x).min(axis=0).astype(dtype)
                cmax = np.where(eq, info.min, x).max(axis=0).astype(dtype)
        np.minimum(gmin, cmin, out=gmin)
        np.maximum(gmax, cmax, out=gmax)
        if sat_thr is not None and (cmax >= sat_thr).any():
            hit = x >= sat_thr
            if eq is not None:
                hit &= ~eq
            counted = _count_true(hit, n_bands)
            if counted is not None:
                n_sat += counted

    mean = np.full(n_bands, np.nan)
    std = np.full(n_bands, np.nan)
    sum1 = s1.tolist()
    sum2 = s2.tolist()
    n_valid = n_px - n_ign
    if ignore is not None:
        for b in np.flatnonzero(n_ign).tolist():
            k = int(n_ign[b])
            sum1[b] -= ignore * k
            sum2[b] -= ignore * ignore * k
    for b, nb in enumerate(n_valid.tolist()):
        if nb > 0:
            mean[b] = sum1[b] / nb                       # int / int: correctly rounded
            std[b] = _sqrt_ratio(nb * sum2[b] - sum1[b] * sum1[b], nb * nb)
    has = n_valid > 0
    vmin = np.where(has, gmin.astype(np.float64), np.nan)
    vmax = np.where(has, gmax.astype(np.float64), np.nan)
    return RoiStats(n_px, n_valid, mean, std, vmin, vmax, n_sat,
                    np.array(sum1, dtype=s1.dtype), np.array(sum2, dtype=np.uint64), "exact-int")


def _row_deviations(f, mean_r, cnt, starts, out):
    """``out[p] = f[p] - mean_r[row of p]`` for consecutive rows of ``cnt`` pixels, in place.

    Equal row lengths (any rect, a one-row chunk) use one broadcast subtraction, ragged
    lasso rows one subtraction per row; each element is the same single rounded
    subtraction either way, so the result does not depend on the chunking.
    """
    n_bands = f.shape[1]
    if (cnt == cnt[0]).all():
        k = int(cnt[0])
        np.subtract(f.reshape(-1, k, n_bands), mean_r[:, None, :], out=out.reshape(-1, k, n_bands))
        return
    for a, c, m in zip(starts.tolist(), cnt.tolist(), mean_r):
        np.subtract(f[a:a + c], m, out=out[a:a + c])


def _float_stats(chunks, n_px, n_bands, dtype, ignore_value, saturation):
    """Float64 path: per-row two-pass, rows merged with the full decomposition; NaN / inf / ignore skipped.

    Per row ``r`` (valid values only): ``n_r``, ``s_r = sum(x)``, the rounded mean
    ``m_r = s_r / n_r``, ``Q_r = sum((x - m_r)**2)`` and ``e_r = sum(x - m_r)``.  Then
    ``sum((x - mu)**2) = sum(Q_r) + sum(n_r (m_r - mu)**2) + 2 sum((m_r - mu) e_r)`` holds
    exactly for any ``m_r`` (``e_r`` is not 0 for a rounded ``m_r``), and subtracting
    ``E**2 / n`` with ``E = sum(n_r (m_r - mu) + e_r) = sum(x - mu)`` turns it into the sum
    about the exact mean although ``mu`` is rounded.  Without the cross term the std was up
    to 3e-9 relative off on offset data (``2e9 + {0..3}``), without the ``E`` correction
    1.5e8 ulp (``1e12 + N(0, 1)``); with both it was within 1 ulp in every measured case.
    """
    ignore = _native_value(ignore_value, dtype)
    sat = None if saturation is None else float(saturation)
    n_rows_total = int(np.count_nonzero(chunks.row_counts))
    row_n = np.zeros((n_rows_total, n_bands), dtype=np.int64)
    row_s = np.zeros((n_rows_total, n_bands))
    row_m = np.zeros((n_rows_total, n_bands))
    row_q = np.zeros((n_rows_total, n_bands))
    row_e = np.zeros((n_rows_total, n_bands))
    gmin = np.full(n_bands, np.inf)
    gmax = np.full(n_bands, -np.inf)
    n_sat = np.zeros(n_bands, dtype=np.int64)
    size = max(chunks.max_px, 1)
    buf = np.empty((size, n_bands), dtype=np.float64)      # values, reused per chunk
    dev = np.empty((size, n_bands), dtype=np.float64)      # deviations, reused per chunk
    at = 0
    for counts, x in chunks:
        cnt = counts[counts > 0]
        n_r = cnt.size
        starts = np.concatenate(([0], np.cumsum(cnt)[:-1]))
        f = buf[:x.shape[0]]
        d = dev[:x.shape[0]]
        np.copyto(f, x)
        s = np.add.reduceat(f, starts, axis=0)
        bad = None
        if not np.isfinite(s).all():
            bad = ~np.isfinite(f)
        if ignore is not None:
            eq = x == ignore
            if eq.any():
                bad = eq if bad is None else (bad | eq)
        if bad is None:                  # every value valid: the common case
            nr = np.broadcast_to(cnt[:, None], s.shape)
            mean_r = s / cnt[:, None]
            _row_deviations(f, mean_r, cnt, starts, d)
            # every value is finite here: min / max of the narrower raw values are the same
            # numbers (the float64 conversion is monotone) and read fewer bytes
            vals_lo, vals_hi = x, x
            hit = None if sat is None else f >= sat
        else:
            good = ~bad
            f[bad] = 0.0
            s = np.add.reduceat(f, starts, axis=0)
            nr = np.add.reduceat(good, starts, axis=0, dtype=np.int64)
            mean_r = np.divide(s, nr, out=np.zeros_like(s), where=nr > 0)
            _row_deviations(f, mean_r, cnt, starts, d)
            d[bad] = 0.0
            vals_lo = np.where(good, f, np.inf)
            vals_hi = np.where(good, f, -np.inf)
            hit = None if sat is None else (good & (f >= sat))
        e = np.add.reduceat(d, starts, axis=0)       # residual of the rounded row mean
        np.multiply(d, d, out=d)
        q = np.add.reduceat(d, starts, axis=0)
        row_n[at:at + n_r] = nr
        row_s[at:at + n_r] = s
        row_m[at:at + n_r] = mean_r
        row_q[at:at + n_r] = q
        row_e[at:at + n_r] = e
        at += n_r
        np.minimum(gmin, vals_lo.min(axis=0), out=gmin)
        np.maximum(gmax, vals_hi.max(axis=0), out=gmax)
        if hit is not None:
            counted = _count_true(hit, n_bands)
            if counted is not None:
                n_sat += counted

    n_valid = row_n.sum(axis=0)
    mean = np.full(n_bands, np.nan)
    std = np.full(n_bands, np.nan)
    valid = np.flatnonzero(n_valid).tolist()
    if valid:
        # per-band columns as Python lists in one call each (fsum needs them); the
        # element-wise terms are computed for all bands at once
        sums = row_s.T.tolist()
        mu = np.zeros(n_bands)
        for b in valid:
            mu[b] = math.fsum(sums[b]) / int(n_valid[b])
        dm = row_m - mu                                   # m_r - mu
        n_dm = row_n * dm
        q_cols, b_cols = row_q.T.tolist(), (n_dm * dm).T.tolist()
        c_cols = (2.0 * dm * row_e).T.tolist()
        r_cols, e_cols = n_dm.T.tolist(), row_e.T.tolist()
        for b in valid:
            nb = int(n_valid[b])
            m2 = math.fsum(q_cols[b] + b_cols[b] + c_cols[b])
            resid = math.fsum(r_cols[b] + e_cols[b])     # sum(x - mu): mu is rounded
            m2 -= resid * resid / nb
            mean[b] = mu[b]
            std[b] = math.sqrt(max(m2, 0.0) / nb)
    has = n_valid > 0
    return RoiStats(n_px, n_valid, mean, std, np.where(has, gmin, np.nan),
                    np.where(has, gmax, np.nan), n_sat, None, None, "float64-two-pass")


def roi_median(cube, bbox, local_mask=None, *, ignore_value=None,
               cancel: Optional[Callable[[], bool]] = None, chunk_bytes: int = 1 << 22) -> np.ndarray:
    """Per-band median of an ROI's pixels (NaN / ignore value excluded), as float64.

    Unlike the mean and std this needs every pixel: the ROI's pixels are gathered row
    chunk by row chunk (``n_px x B`` values in the cube's dtype; float32 when NaN or an
    ignore value has to be masked), then ``numpy.median`` / ``nanmedian`` per band.

    Args:
        cube: ``(H, W, B)`` array, memmap or view.
        bbox: ``(c0, r0, c1, r1)`` half-open pixel box (as for :func:`roi_stats`).
        local_mask: Optional bool ``(r1 - r0, c1 - c0)`` mask; None = the whole box.
        ignore_value: Optional ENVI 'data ignore value' excluded per band.
        cancel: Optional callable polled per chunk; True stops with ``RoiCancelled``.
        chunk_bytes: Approximate cube bytes read per chunk.

    Returns:
        numpy.ndarray: ``(B,)`` float64 medians; NaN for a band without a valid pixel.

    Raises:
        ValueError: Empty box or mask shape mismatch.
        RoiCancelled: ``cancel()`` returned True.
    """
    h, w, n_bands = _check_cube(cube)
    c0, r0, c1, r1 = (int(v) for v in bbox)
    if not (0 <= c0 < c1 <= w and 0 <= r0 < r1 <= h):
        raise ValueError(f"bbox {bbox} is empty or outside the {w} x {h} image")
    if local_mask is not None and np.shape(local_mask) != (r1 - r0, c1 - c0):
        raise ValueError(f"mask shape {np.shape(local_mask)} does not match bbox {bbox}")
    row_bytes = max(1, (c1 - c0) * n_bands * np.dtype(cube.dtype).itemsize)
    rows_per_chunk = max(1, int(chunk_bytes // row_bytes))
    parts = []
    for top in range(r0, r1, rows_per_chunk):
        if cancel is not None and cancel():
            raise RoiCancelled("median cancelled")
        bottom = min(r1, top + rows_per_chunk)
        block = np.asarray(cube[top:bottom, c0:c1, :])
        if local_mask is not None:
            block = block[np.asarray(local_mask[top - r0:bottom - r0], dtype=bool)]
        parts.append(block.reshape(-1, n_bands))
    pixels = np.concatenate(parts, axis=0) if parts else np.empty((0, n_bands), cube.dtype)
    if pixels.shape[0] == 0:
        return np.full(n_bands, np.nan)
    needs_mask = pixels.dtype.kind == "f" or ignore_value is not None
    if not needs_mask:
        return np.median(pixels, axis=0).astype(np.float64)
    values = pixels.astype(np.float32 if pixels.dtype.itemsize <= 4 else np.float64)
    values[~np.isfinite(values)] = np.nan
    if ignore_value is not None:
        values[pixels == ignore_value] = np.nan
    with np.errstate(all="ignore"):
        valid = np.isfinite(values).any(axis=0)
        out = np.full(n_bands, np.nan)
        if valid.any():
            out[valid] = np.nanmedian(values[:, valid], axis=0)
    return out.astype(np.float64)
