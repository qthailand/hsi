"""Raw saturation masks, carried through the correction pipeline to the recorded bands (CPU reference).

A raw pixel that clipped is found before any correction (``raw >= level``) and followed through
the same stages the intensities go through, with a logical OR instead of a weighted sum:

1. **Raw detection** (:func:`detect_raw`): ``clipped`` = ``raw >= level`` (the camera's full
   scale, ``backend.saturation_level()``), ``near`` = ``raw >= near_ratio * level`` (clipped
   included). Sensor orientation.
2. **Hot-pixel repair** (:func:`hot_pixel_mask`): with the correction on, a hot pixel takes the
   OR of the valid neighbours its repair averages (``_hot_ys/_hot_xs/_hot_valid`` of the same
   dark reference); its own raw value no longer counts. Off: unchanged.
3. **Geometry** (:class:`GeometryMask`): orientation and the smile (+ keystone) remap. An output
   pixel is affected when any source tap with a non-zero weight reads a masked pixel (OpenCV's
   own 1/32 fixed-point tables, :func:`tap_support`); ``'nearest'`` follows the model's own
   ``apply_native`` (whose fixed-point path differs from :func:`prepare_maps`).
4. **Lambda binning** (:func:`bin_masks`): a band is affected when any corrected pixel in its
   ``iL:iR`` columns is (the binner's own boundaries, overlaps included; a subset binner gives
   the subset's bands). The smile border is left out by the binner subset, not cropped here.
5. **Keystone rows**: ``row_crop`` ``(top, bottom)`` as the intensity frames are cropped.

Fractions:
    ``raw_clipped_fraction`` - share of the raw frame's pixels that clipped (frame level: after
    a remap a raw pixel has no footprint inside one band). ``affected_fraction`` - per slit row
    and band, the share of the band's ``w`` corrected pixels that are affected (0..1).

The intensities are never touched: everything here reads the calibration, the dark reference
and the binner the intensity pipeline uses, and writes new boolean arrays.
"""
from __future__ import annotations

import functools
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from numba import njit, prange

#: Share of the full scale at and above which a raw pixel is near saturation.
NEAR_RATIO = 0.98
#: How a band is judged: any source pixel with a non-zero weight that clipped.
POLICY = "any contributing tap"
#: Taps of each interpolation, and the first tap's offset from the fixed-point base pixel.
TAPS = {"nearest": (1, 0), "linear": (2, 0), "cubic": (4, 1), "lanczos": (8, 3)}
_FLAGS = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC,
          "lanczos": cv2.INTER_LANCZOS4}
_TAB = 32                                  # OpenCV INTER_TAB_SIZE: fractions in 1/32 pixel


# ── 1. raw detection ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class RawSaturation:
    """Raw-frame masks of one frame (sensor orientation, read-only).

    Attributes:
        clipped: ``bool`` ``raw >= level``.
        near: ``bool`` ``raw >= near_ratio * level`` (``clipped`` included).
        level: Full scale used (counts).
        near_ratio: Share of ``level`` that counts as near saturation.
        raw_clipped_fraction, raw_near_fraction: Share of the frame's pixels in each mask.
    """

    clipped: np.ndarray
    near: np.ndarray
    level: int
    near_ratio: float
    raw_clipped_fraction: float
    raw_near_fraction: float

    def any(self) -> bool:
        """Whether any pixel is near saturation (or clipped); ``False``: the fast path."""
        return self.raw_near_fraction > 0


def detect_raw(raw, level, near_ratio: float = NEAR_RATIO, counts=None) -> RawSaturation:
    """Clipped and near-saturated pixels of a raw sensor frame.

    Args:
        raw: 2-D raw frame (any numeric dtype); not modified.
        level: The camera's full scale (``backend.saturation_level()``), > 0.
        near_ratio: Share of ``level`` that counts as near saturation, in ``(0, 1]``.
        counts: ``(clipped, near)`` of this frame against the same ``level`` and ``near_ratio``
            when already counted (:func:`.count.count_saturated`, the basic detection): no
            frame is read for a frame with nothing near saturation, and the masks of the others
            are built without counting them again.

    Returns:
        RawSaturation: read-only masks; with nothing near saturation both masks are the same
        all-False array (one comparison over the frame).

    Raises:
        ValueError: ``raw`` is not 2-D, ``level`` is not positive or ``near_ratio`` out of range.
    """
    raw = np.asarray(raw)
    if raw.ndim != 2:
        raise ValueError(f"raw frame must be 2-D, got shape {raw.shape}")
    level = int(level)
    near_ratio = float(near_ratio)
    if level <= 0:
        raise ValueError(f"saturation level must be positive, got {level}")
    if not 0.0 < near_ratio <= 1.0:
        raise ValueError(f"near_ratio must be in (0, 1], got {near_ratio}")
    threshold = near_ratio * level
    if counts is not None:
        n_clipped, n_near = (int(v) for v in counts)
        if not 0 <= n_clipped <= n_near <= raw.size:
            raise ValueError(f"counts {counts} do not fit a frame of {raw.size} pixels")
    if raw.size == 0 or (n_near == 0 if counts is not None else raw.max() < threshold):
        empty = _zeros(raw.shape, np.bool_)             # fast path: no mask built
        return RawSaturation(empty, empty, level, near_ratio, 0.0, 0.0)
    near = raw >= threshold
    if counts is None:
        n_near = int(np.count_nonzero(near))
    if counts is not None and n_clipped == n_near:
        clipped = near                                  # every near pixel clipped: one mask
    else:
        clipped = raw >= level
        if counts is None:
            n_clipped = int(np.count_nonzero(clipped))
        if n_clipped == n_near:
            clipped = near
    clipped.setflags(write=False)
    near.setflags(write=False)
    size = float(raw.size)
    return RawSaturation(clipped, near, level, near_ratio, n_clipped / size, n_near / size)


# ── 2. hot-pixel repair ──────────────────────────────────────────────────

def hot_pixel_mask(mask, hot, enabled: bool) -> np.ndarray:
    """``mask`` after the hot-pixel repair the intensity pipeline applies.

    Args:
        mask: 2-D ``bool`` raw mask (sensor orientation); not modified.
        hot: The dark reference the frame is corrected with (``_hot_coords``, ``_hot_ys``,
            ``_hot_xs``, ``_hot_valid`` of ``modules.dark``), or ``None``.
        enabled: Whether dark / hot correction is applied to the frame.

    Returns:
        numpy.ndarray: a new ``bool`` array; with the repair on, each hot pixel is the OR of
        its valid neighbours (none valid: ``False``, the repair reads no pixel).
    """
    out = np.array(mask, dtype=bool, copy=True)
    coords = getattr(hot, "_hot_coords", None) if hot is not None else None
    if not enabled or coords is None or len(coords) == 0:
        return out
    neighbours = np.asarray(mask, dtype=bool)[hot._hot_ys, hot._hot_xs] & hot._hot_valid
    out[coords[:, 0], coords[:, 1]] = neighbours.any(axis=1)
    return out


# ── 3. geometry ──────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def tap_weights(interp: str) -> np.ndarray:
    """OpenCV's own 1-D interpolation weights: ``float32 (32, taps)``, one row per 1/32 fraction.

    Measured with ``cv2.remap`` on unit impulses, like
    :func:`modules.preprocessing.maps.horizontal_table`, but with every tap of the kernel
    (Lanczos4: 8). Read-only, cached.
    """
    taps, origin = TAPS[interp]
    if taps == 1:
        weights = np.ones((_TAB, 1), dtype=np.float32)
    else:
        x = (origin + np.arange(_TAB, dtype=np.float32) / _TAB).reshape(1, _TAB)
        y = np.zeros_like(x)
        weights = np.stack([cv2.remap(np.eye(taps, dtype=np.float32)[k:k + 1], x, y, _FLAGS[interp],
                                      borderMode=cv2.BORDER_CONSTANT)[0] for k in range(taps)], axis=1)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    weights.setflags(write=False)
    return weights


@functools.lru_cache(maxsize=None)
def tap_support(interp: str) -> np.ndarray:
    """``bool (32, taps)``: the taps of each 1/32 fraction whose weight is exactly non-zero."""
    support = tap_weights(interp) != 0
    support.setflags(write=False)
    return support


class GeometryMask:
    """Orientation + smile (+ keystone) remap of masks, for one model and orientation.

    Built once per ``(calibration, inverted, shape)`` (the maps and the source tables are cached);
    :meth:`apply` then runs per frame.

    Args:
        calibration: The smile model the intensities are remapped with (``interp``,
            ``image_shape``, and ``map_x``/``_get_map_y()`` or, for ``'nearest'``,
            ``apply_native``), or ``None`` (orientation only).
        inverted: Whether the slit axis is flipped (as the intensity pipeline's ``inverted``).
        shape: ``(rows, columns)`` of the raw frames.

    Raises:
        ValueError: The model's ``image_shape`` is not ``shape``.
        KeyError: Unknown interpolation.
    """

    #: Use the per-tap reference loop instead of the dilation form (tests compare the two).
    reference_loop = False

    def __init__(self, calibration, inverted: bool, shape: Tuple[int, int]):
        self.calibration = calibration
        self.inverted = bool(inverted)
        self.shape = (int(shape[0]), int(shape[1]))
        self.interp = None if calibration is None else str(calibration.interp)
        if calibration is not None and tuple(int(v) for v in calibration.image_shape) != self.shape:
            raise ValueError(f"Smile model {tuple(calibration.image_shape)} does not match frames {self.shape}")
        self._source = None              # nearest: CV_16SC2 map of the pixel each output copies
        self._fixed = self._fractions = None
        if calibration is None:
            return
        if self.interp == "nearest":
            self._source = self._nearest_sources()
            return
        if self.interp not in TAPS:
            raise KeyError(self.interp)
        from modules.preprocessing.maps import prepare_maps
        _x, _y, fixed, fractions, _flag = prepare_maps(calibration, self.inverted)
        self._fixed = np.ascontiguousarray(fixed[..., 0], dtype=np.int32), \
            np.ascontiguousarray(fixed[..., 1], dtype=np.int32)
        fractions = np.zeros(self.shape, np.uint16) if fractions is None else np.asarray(fractions)
        self._fractions = (np.asarray(fractions & (_TAB - 1), dtype=np.intp),
                           np.asarray((fractions >> 5) & (_TAB - 1), dtype=np.intp))
        # Dilation form: every tap table here is "all taps" for a non-zero fraction and "the base
        # pixel only" for fraction 0, so the OR over the taps is a dilation (K wide along each axis
        # whose fraction is non-zero) read at the base pixel. Checked, else the reference loop.
        taps, origin = TAPS[self.interp]
        support = tap_support(self.interp)
        base_only = np.zeros(taps, dtype=bool)
        base_only[origin] = True
        self._separable = bool(np.array_equal(support[0], base_only) and support[1:].all())
        pad = taps + 1
        self._pad = pad
        xy = np.stack([self._fixed[0] + pad, self._fixed[1] + pad], axis=-1)
        limit = np.iinfo(np.int16)
        self._base_map = np.clip(xy, limit.min, limit.max).astype(np.int16)   # beyond: border 0
        fx, fy = self._fractions
        self._moves_x, self._moves_y = fx != 0, fy != 0
        self._any_y = bool(self._moves_y.any())

    def _nearest_sources(self):
        """The pixel every output copies (a ``CV_16SC2`` map, -1 outside), read through the
        model's ``apply_native`` on row / column index images."""
        h, w = self.shape
        if h >= 32767 or w >= 32767:
            raise ValueError("frames too large for the nearest source map")
        rows, cols = np.indices((h, w), dtype=np.uint16)
        # ``apply_native`` takes the oriented frame; 0 marks the border fill (sources are 1-based)
        got_r = np.asarray(self.calibration.apply_native(np.ascontiguousarray(rows + 1)), dtype=np.int32) - 1
        got_c = np.asarray(self.calibration.apply_native(np.ascontiguousarray(cols + 1)), dtype=np.int32) - 1
        inside = (got_r >= 0) & (got_c >= 0)
        return np.stack([np.where(inside, got_c, -1), np.where(inside, got_r, -1)], axis=-1).astype(np.int16)

    def apply(self, mask) -> np.ndarray:
        """The corrected-frame mask: True where an output pixel reads a masked raw pixel.

        Args:
            mask: ``bool`` ``shape`` mask in sensor orientation (after :func:`hot_pixel_mask`);
                not modified.

        Returns:
            numpy.ndarray: new ``bool`` array of ``shape`` (the corrected frame's layout).
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.shape:
            raise ValueError(f"mask {mask.shape} does not match {self.shape}")
        if self.calibration is None:
            return np.array(mask[::-1] if self.inverted else mask, copy=True, order="C")
        if not mask.any():
            return np.zeros(self.shape, dtype=bool)
        if self._source is not None:
            oriented = np.ascontiguousarray(mask[::-1] if self.inverted else mask, dtype=np.uint8)
            return cv2.remap(oriented, self._source, None, cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        if self.reference_loop or not self._separable:
            return self._remap_or(mask)
        return self._remap_dilated(mask)

    def _remap_dilated(self, mask):
        """The OR over the taps as dilations read at each output's base pixel (see ``__init__``)."""
        h, w = self.shape
        taps, origin = TAPS[self.interp]
        pad = self._pad
        padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
        padded[pad:pad + h, pad:pad + w] = mask

        def read(image):
            """``image`` (padded) at every output's base pixel; outside: 0."""
            return cv2.remap(image, self._base_map, None, cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)

        def dilate(kw, kh):
            """OR over a ``kh`` x ``kw`` tap window starting ``origin`` before each pixel."""
            return cv2.dilate(padded, np.ones((kh, kw), np.uint8),
                              anchor=(origin if kw > 1 else 0, origin if kh > 1 else 0),
                              borderType=cv2.BORDER_CONSTANT, borderValue=0)

        out = np.where(self._moves_x, read(dilate(taps, 1)), read(padded))
        if self._any_y:
            both = read(dilate(taps, taps))
            column = read(dilate(1, taps))
            out = np.where(self._moves_y, np.where(self._moves_x, both, column), out)
        return out

    def _remap_or(self, mask):
        """Reference: OR over the source taps with a non-zero weight, tap by tap (the flip is in
        the maps, as for the intensities)."""
        h, w = self.shape
        taps, origin = TAPS[self.interp]
        x0, y0 = self._fixed
        fx, fy = self._fractions
        # candidates: outputs whose whole tap window holds a masked pixel (a dilation), checked exactly
        padded = np.zeros((h + 2 * taps, w + 2 * taps), dtype=np.uint8)
        padded[taps:taps + h, taps:taps + w] = mask
        window = cv2.dilate(padded, np.ones((taps, taps), np.uint8), anchor=(origin, origin),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0)
        by = np.clip(y0 + taps, 0, h + 2 * taps - 1)
        bx = np.clip(x0 + taps, 0, w + 2 * taps - 1)
        candidate = window[by, bx].astype(bool)
        out = np.zeros(self.shape, dtype=bool)
        if not candidate.any():
            return out
        iy, ix = np.nonzero(candidate)
        cy, cx = y0[iy, ix], x0[iy, ix]
        sy_frac, sx_frac = fy[iy, ix], fx[iy, ix]
        support = tap_support(self.interp)
        hit = np.zeros(iy.size, dtype=bool)
        for j in range(taps):
            row_ok = support[sy_frac, j]
            if not row_ok.any():
                continue
            sy = cy - origin + j
            row_ok &= (sy >= 0) & (sy < h)
            for k in range(taps):
                ok = row_ok & support[sx_frac, k]
                if not ok.any():
                    continue
                sx = cx - origin + k
                ok &= (sx >= 0) & (sx < w)
                idx = np.nonzero(ok & ~hit)[0]
                if idx.size:
                    hit[idx] |= mask[sy[idx], sx[idx]]
        out[iy[hit], ix[hit]] = True
        return out


# ── 4. lambda binning ────────────────────────────────────────────────────

def bin_masks(mask, binner) -> Tuple[np.ndarray, np.ndarray]:
    """Bands of a corrected-frame mask, with the binner's own boundaries.

    Args:
        mask: ``bool (rows, binner.v_)`` corrected-frame mask; not modified.
        binner: Fitted ``LambdaBinner`` (or a subset of one): ``iL``, ``iR``, ``w``, ``v_``.

    Returns:
        tuple: ``(any, affected_fraction)`` - ``bool (rows, n_bins)`` (a band is affected when
        any of its ``iL:iR`` pixels is) and ``float32 (rows, n_bins)`` (that count / ``w``).

    Raises:
        ValueError: The binner is fitted for another width.
    """
    mask = np.asarray(mask, dtype=bool)
    if int(binner.v_) != mask.shape[1]:
        raise ValueError(f"Binner fitted for {binner.v_} spectral pixels, mask has {mask.shape[1]}")
    il = np.ascontiguousarray(binner.iL, dtype=np.int32)
    ir = np.ascontiguousarray(binner.iR, dtype=np.int32)
    width = np.ascontiguousarray(binner.w, dtype=np.int32)
    return _bin_any_count(np.ascontiguousarray(mask.view(np.uint8)), il, ir, width)


@njit(parallel=True, cache=True)
def _bin_any_count(mask, il, ir, width):
    """Per row and bin: any masked pixel in ``il:ir``, and their share of ``width`` (the
    binner's loop, with a count instead of a rounded mean)."""
    rows, cols = mask.shape
    n_bins = il.shape[0]
    hit = np.zeros((rows, n_bins), dtype=np.bool_)
    fraction = np.zeros((rows, n_bins), dtype=np.float32)
    for r in prange(rows):
        if mask[r].max() == 0:
            continue
        for k in range(n_bins):
            count = 0
            for j in range(il[k], ir[k]):
                count += mask[r, j]
            if count:
                hit[r, k] = True
                fraction[r, k] = count / width[k]
    return hit, fraction


def crop_rows(values, row_crop: Tuple[int, int]):
    """``values`` without ``row_crop`` ``(top, bottom)`` slit rows (the keystone margin the
    intensity frames are cropped by, ``keystone_row_margin``); unchanged for ``(0, 0)``."""
    top, bottom = (max(0, int(v)) for v in row_crop)
    if top + bottom <= 0:
        return values
    return values[top:values.shape[0] - bottom]


# ── the whole frame ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class FrameSaturation:
    """Saturation of one recorded frame, in the layout of its binned intensities.

    Attributes:
        clipped_mask: ``bool (rows, bands)`` bands any clipped raw pixel contributes to.
        near_saturation_mask: ``bool (rows, bands)`` the same for near saturation (clipped included).
        affected_fraction: ``float32 (rows, bands)`` share of each band's corrected pixels
            affected by clipping (0..1).
        near_affected_fraction: The same for near saturation.
        raw_clipped_fraction, raw_near_fraction: Shares of the raw frame (frame level).
        level, near_ratio: Thresholds used. policy: :data:`POLICY`.
    """

    clipped_mask: np.ndarray
    near_saturation_mask: np.ndarray
    affected_fraction: np.ndarray
    near_affected_fraction: np.ndarray
    raw_clipped_fraction: float
    raw_near_fraction: float
    level: int
    near_ratio: float
    policy: str = POLICY

    def any_clipped(self) -> bool:
        """Whether any band of the frame is affected by a clipped raw pixel."""
        return self.raw_clipped_fraction > 0 and bool(self.clipped_mask.any())


class _NoProfiler:
    def time(self, _name):
        return nullcontext()


@functools.lru_cache(maxsize=8)
def _zeros(shape, dtype):
    out = np.zeros(shape, dtype=dtype)
    out.setflags(write=False)
    return out


class SaturationPropagator:
    """Raw frame -> :class:`FrameSaturation`, with the stages of the intensity pipeline.

    Keeps one :class:`GeometryMask` (rebuilt when the model, the orientation or the frame shape
    changes). One instance per processing thread.

    Profiling spans (``profiler.time(name)``): ``sat_detect``, ``sat_hot``, ``sat_remap``,
    ``sat_bin``; frames with nothing near saturation stop after ``sat_detect``.
    """

    def __init__(self, near_ratio: float = NEAR_RATIO):
        self.near_ratio = float(near_ratio)
        self._geometry: Optional[GeometryMask] = None
        self._geometry_key = None

    def geometry(self, calibration, inverted: bool, shape) -> GeometryMask:
        """The cached :class:`GeometryMask` for these settings (built on first use)."""
        key = (id(calibration), bool(inverted), tuple(shape))
        if key != self._geometry_key or self._geometry is None or self._geometry.calibration is not calibration:
            self._geometry = GeometryMask(calibration, inverted, shape)
            self._geometry_key = key
        return self._geometry

    def corrected(self, raw_saturation: RawSaturation, hot, enabled, calibration, inverted, profiler=None):
        """``(clipped, near)`` in the corrected (unbinned, uncropped) frame's layout."""
        profiler = profiler or _NoProfiler()
        shape = raw_saturation.near.shape
        geometry = self.geometry(calibration, inverted, shape)

        def follow(mask):
            """One raw mask through the hot-pixel repair and the geometry."""
            with profiler.time("sat_hot"):
                repaired = hot_pixel_mask(mask, hot, enabled)
            with profiler.time("sat_remap"):
                return geometry.apply(repaired)

        near = follow(raw_saturation.near)
        if raw_saturation.clipped is raw_saturation.near:   # every near pixel clipped
            clipped = near
        elif raw_saturation.raw_clipped_fraction > 0:
            clipped = follow(raw_saturation.clipped)
        else:                                       # near saturation only: nothing clipped
            clipped = np.zeros(near.shape, dtype=bool)
        return clipped, near

    def frame(self, raw, level, hot, enabled, calibration, inverted, binner,
              row_crop: Tuple[int, int] = (0, 0), profiler=None, counts=None) -> FrameSaturation:
        """Saturation of one raw frame in the layout of its recorded, binned line.

        Args:
            raw: The raw sensor frame (before dark subtraction); not modified.
            level: The camera's full scale (``backend.saturation_level()``).
            hot, enabled, calibration, inverted: As the intensity pipeline gets them for this
                frame (the same dark-reference snapshot, the same model).
            binner: The binner the frame is binned with (a subset: its bands only).
            row_crop: ``(top, bottom)`` keystone rows the binned frames lose.
            profiler: Optional object with ``time(name)``.
            counts: ``(clipped, near)`` of ``raw`` already counted against ``level`` (the basic
                detection): passed to :func:`detect_raw`, which then does not read the frame
                again when nothing is near saturation.

        Returns:
            FrameSaturation: masks and fractions of shape ``(rows - top - bottom, n_bins)``.
        """
        profiler = profiler or _NoProfiler()
        with profiler.time("sat_detect"):
            raw_saturation = detect_raw(raw, level, self.near_ratio, counts=counts)
        rows = raw_saturation.near.shape[0]
        top, bottom = (max(0, int(v)) for v in row_crop)
        if top + bottom >= rows:
            top = bottom = 0
        shape = (rows - top - bottom, len(binner.iL))
        if not raw_saturation.any():
            if int(binner.v_) != raw_saturation.near.shape[1]:
                raise ValueError(f"Binner fitted for {binner.v_} spectral pixels, frame has "
                                 f"{raw_saturation.near.shape[1]}")
            empty, zero = _zeros(shape, np.bool_), _zeros(shape, np.float32)
            return FrameSaturation(empty, empty, zero, zero, 0.0, 0.0, raw_saturation.level, self.near_ratio)
        clipped, near = self.corrected(raw_saturation, hot, enabled, calibration, inverted, profiler)
        with profiler.time("sat_bin"):
            near_any, near_fraction = bin_masks(near, binner)
            if clipped is near:
                clip_any, clip_fraction = near_any, near_fraction
            elif raw_saturation.raw_clipped_fraction > 0:
                clip_any, clip_fraction = bin_masks(clipped, binner)
            else:
                clip_any, clip_fraction = _zeros(near_any.shape, np.bool_), _zeros(near_any.shape, np.float32)
        crop = (top, bottom)
        return FrameSaturation(crop_rows(clip_any, crop), crop_rows(near_any, crop),
                               crop_rows(clip_fraction, crop), crop_rows(near_fraction, crop),
                               raw_saturation.raw_clipped_fraction, raw_saturation.raw_near_fraction,
                               raw_saturation.level, self.near_ratio)


# ── bit packing (for the recording; the sidecar format is Phase 3) ──────────

def pack_mask(mask) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """``(packed uint8, shape)`` of a boolean mask: ``np.packbits`` of the flat C-order mask,
    ``bitorder='little'`` (bit ``i % 8`` of byte ``i // 8`` is element ``i``)."""
    mask = np.asarray(mask, dtype=bool)
    return np.packbits(mask.ravel(), bitorder="little"), tuple(mask.shape)


def unpack_mask(packed, shape) -> np.ndarray:
    """The boolean mask :func:`pack_mask` packed."""
    count = int(np.prod(shape)) if len(shape) else 1
    return np.unpackbits(np.asarray(packed, dtype=np.uint8), count=count, bitorder="little").astype(bool).reshape(shape)
