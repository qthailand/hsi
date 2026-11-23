"""Band-fusion preview of a cube: neighbouring bands stacked onto a centre band to show keystone by eye (Qt-free).

Keystone moves a scene feature along the slit from band to band. Summing the bands next to a
chosen centre band onto it (grayscale) makes that movement visible: where the bands line up an
edge stays exactly as sharp as in the centre band; where they wander, the summed edge fills in
- it widens, ramps or doubles - more and more as bands are added. The two sides are cumulative:
``n_left`` sums the ``n_left`` bands just below the centre band (shorter wavelengths),
``n_right`` the ``n_right`` bands just above it.

Every band is divided by its own level (99th percentile of the lit rows) first, so a bright and
a dim band count the same, and the sum is shown divided by the band count in the centre band's
display range: adding bands never changes the brightness, only the edges.

Data conventions:
    ``cube`` is ``(slit, scan, wavelength)`` (ENVI BIP order; a ``numpy.memmap`` is fine).
    Images returned are ``(slit, scan)`` = rows down the slit, scan columns across.

Threading:
    :class:`BandFusion` keeps a band cache and is not thread-safe: use one instance from one
    thread at a time (the Keystone Check window runs it on its preview worker, one job at a time).

Example:
    >>> import numpy as np
    >>> cube = np.ones((50, 20, 12), np.float32)
    >>> cube[25:, :, :] = 3.0
    >>> f = BandFusion(cube)
    >>> parts = f.compose(6, 3, 3)
    >>> img = fusion_image(parts, "sum")
    >>> img.shape, bool(np.allclose(img, fusion_image(f.compose(6, 0, 0), "sum")))   # straight edge: unchanged
    ((50, 20), True)
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

#: Display modes of :func:`fusion_image`.
MODES = ("sum", "difference")


@dataclass
class FusionParts:
    """Normalised pieces of one fusion (all ``(slit, scan)`` float32, centre level ~ 1).

    Attributes:
        centre: The centre band.
        left: Mean of the ``n_left`` bands below the centre band, or None when ``n_left == 0``.
        right: Mean of the ``n_right`` bands above the centre band, or None when ``n_right == 0``.
        centre_band, n_left, n_right: The request (band indices / counts, clipped to the cube).
    """
    centre: np.ndarray
    left: Optional[np.ndarray]
    right: Optional[np.ndarray]
    centre_band: int
    n_left: int
    n_right: int

    @property
    def side_left(self) -> np.ndarray:
        """``left``, or the centre band when no band is fused on that side."""
        return self.centre if self.left is None else self.left

    @property
    def side_right(self) -> np.ndarray:
        """``right``, or the centre band when no band is fused on that side."""
        return self.centre if self.right is None else self.right

    def mean(self) -> np.ndarray:
        """Sum of all fused bands (both sides and the centre) divided by their count (cached)."""
        cached = self.__dict__.get("_mean")
        if cached is None:
            tot = self.centre.astype(np.float32).copy()
            if self.left is not None:
                tot += self.n_left * self.left
            if self.right is not None:
                tot += self.n_right * self.right
            cached = self.__dict__["_mean"] = tot / (1 + self.n_left + self.n_right)
        return cached


class BandFusion:
    """Normalised, cached band images of a cube and their cumulative fusion around a centre band.

    Args:
        cube: ``(slit, scan, wavelength)`` cube (memmap welcome).
        row_margin: ``(top, bottom)`` rows left out of the band levels (keystone margin).
        cache_mb: Band cache budget (float16 images). When the whole cube fits, the first request
            reads it in one pass of contiguous row chunks; otherwise bands are read in blocks.
        block: Bands per read when the cube does not fit the cache.
    """

    def __init__(self, cube, row_margin: Tuple[int, int] = (0, 0), cache_mb: float = 1500.0, block: int = 32):
        if getattr(cube, "ndim", 0) != 3:
            raise ValueError("Use a 3-D cube (slit, scan, wavelength)")
        self.cube = cube
        self.shape = tuple(int(v) for v in cube.shape)
        L, S, B = self.shape
        self._margin = tuple(max(0, int(v)) for v in row_margin)
        per_band = L * S * 2
        self._max_bands = max(8, int(cache_mb * 2 ** 20 // max(per_band, 1)))
        self._fits = self._max_bands >= B
        self._block = max(1, int(block))
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._levels: Optional[np.ndarray] = None

    @property
    def bands(self) -> int:
        """Number of bands of the cube."""
        return self.shape[2]

    @property
    def max_side(self) -> int:
        """Most bands one side of a fusion may take (all of them when the cube fits the cache)."""
        return self.bands if self._fits else max(0, (self._max_bands - 2 * self._block - 1) // 2)

    # ── band levels / cache ────────────────────────────────────
    def levels(self) -> np.ndarray:
        """Per-band level (99th percentile of the lit rows, from a sub-sampled read), floored at 2 % of the largest."""
        if self._levels is None:
            L, S, B = self.shape
            top, bottom = self._margin
            r0, r1 = min(top, L - 1), max(min(top, L - 1) + 1, L - bottom)
            rs = max(1, (r1 - r0) // 200)
            cs = max(1, S // 200)
            sub = np.nan_to_num(np.asarray(self.cube[r0:r1:rs, ::cs, :], np.float32)).reshape(-1, B)
            lev = np.percentile(sub, 99, axis=0)
            floor = 0.02 * float(lev.max()) if lev.max() > 0 else 1.0
            self._levels = np.maximum(lev, floor).astype(np.float32)
        return self._levels

    def _read(self, a: int, b: int, progress: Callable[[int], None]) -> None:
        """Read bands ``a:b`` in contiguous row chunks into the cache."""
        L, S, B = self.shape
        lev = self.levels()
        out = np.empty((b - a, L, S), np.float16)
        step = max(1, int(2 ** 26 // max(S * B * 4, 1)))        # ~64 MB of file per chunk
        for r0 in range(0, L, step):
            r1 = min(L, r0 + step)
            chunk = np.nan_to_num(np.asarray(self.cube[r0:r1, :, a:b], np.float32))   # (rows, S, b-a)
            chunk /= lev[None, None, a:b]
            out[:, r0:r1, :] = np.moveaxis(chunk, 2, 0)
            progress(int(100 * r1 / L))
        for i, k in enumerate(range(a, b)):
            self._cache[k] = out[i]
            self._cache.move_to_end(k)
        while len(self._cache) > self._max_bands:
            self._cache.popitem(last=False)

    def _ensure(self, ks: np.ndarray, progress: Callable[[int], None]) -> None:
        missing = [int(k) for k in ks if int(k) not in self._cache]
        if not missing:
            for k in ks:
                self._cache.move_to_end(int(k))
            return
        if self._fits:
            self._read(0, self.bands, progress)
            return
        blocks = sorted({k // self._block for k in missing})
        for j, blk in enumerate(blocks):
            a, b = blk * self._block, min(self.bands, (blk + 1) * self._block)
            self._read(a, b, lambda p, j=j: progress(int((100 * j + p) / len(blocks))))
        for k in ks:                       # keep the requested bands newest
            if int(k) in self._cache:
                self._cache.move_to_end(int(k))

    def band(self, k: int) -> np.ndarray:
        """Normalised band *k* (float32 ``(slit, scan)``)."""
        self._ensure(np.array([int(k)]), lambda p: None)
        return self._cache[int(k)].astype(np.float32)

    def _side_mean(self, ks: np.ndarray) -> Optional[np.ndarray]:
        if ks.size == 0:
            return None
        tot = np.zeros(self.shape[:2], np.float32)
        for k in ks:
            tot += self._cache[int(k)]
        return tot / ks.size

    # ── fusion ─────────────────────────────────────────────────
    def compose(self, centre: int, n_left: int, n_right: int,
                progress: Optional[Callable[[int], None]] = None) -> FusionParts:
        """Fuse ``n_left`` bands below and ``n_right`` above band *centre* (cumulative, clipped to the cube)."""
        B = self.bands
        c = int(np.clip(centre, 0, B - 1))
        nl = int(np.clip(n_left, 0, c))
        nr = int(np.clip(n_right, 0, B - 1 - c))
        left = np.arange(c - nl, c)
        right = np.arange(c + 1, c + 1 + nr)
        need = np.r_[left, c, right]
        if not self._fits and need.size + 2 * self._block > self._max_bands:
            raise ValueError(f"{need.size} bands do not fit the preview cache ({self._max_bands} bands)")
        self._ensure(need, progress or (lambda p: None))
        return FusionParts(centre=self._cache[c].astype(np.float32), left=self._side_mean(left),
                           right=self._side_mean(right), centre_band=c, n_left=nl, n_right=nr)


def _stretch(ref: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> Tuple[float, float]:
    """Display range from the percentiles of *ref* (sub-sampled)."""
    sub = ref[:: max(1, ref.shape[0] // 400), :: max(1, ref.shape[1] // 400)]
    lo, hi = np.percentile(sub, [lo_pct, hi_pct])
    if not hi > lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def fusion_image(parts: FusionParts, mode: str = "sum") -> np.ndarray:
    """Display image of a fusion, ``(slit, scan)`` float32.

    Modes:
        ``sum``: grayscale sum of every fused band (divided by the count) in the centre band's
            display range, 0..1. Lined-up bands leave the edges as sharp as the centre band;
            misregistered ones fill the edges in (wider / ramped / doubled).
        ``difference``: right side minus left side (the centre band stands in for an empty side),
            in units of the centre's contrast; signed, about 0 where the bands line up.
    """
    lo, hi = _stretch(parts.centre)
    if mode == "difference":
        return ((parts.side_right - parts.side_left) / (hi - lo)).astype(np.float32)
    if mode != "sum":
        raise ValueError(f"mode must be one of {MODES}")
    return np.clip((parts.mean() - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def row_profiles(parts: FusionParts, cols: Tuple[int, int]) -> dict:
    """Mean along the scan columns ``cols[0]:cols[1]`` of the centre band and of the fused sum (per row)."""
    a, b = int(max(0, cols[0])), int(min(parts.centre.shape[1], max(cols[1], cols[0] + 1)))
    return {"centre": parts.centre[:, a:b].mean(axis=1), "fused": parts.mean()[:, a:b].mean(axis=1)}
