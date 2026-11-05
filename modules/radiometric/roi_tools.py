"""Choosing the reference-target pixels of a scan (Qt-free): three ways to pick, one clean-up, QC.

The boolean ROI ``(lines, samples)`` these functions build is what the row-wise factor fit
(:func:`modules.radiometric.core.compute_rowwise_factors_array`) averages, so every pixel in it
should be evenly lit, unsaturated reference target.

Ways to pick the target:

* **manual shapes** - :func:`rect_mask`, :func:`ellipse_mask`, :func:`polygon_mask` (a freehand
  path is simplified first with :func:`simplify_polyline`);
* **shape detection** - :func:`detect_shapes` finds bright compact blobs, classifies each as a
  rectangle or an ellipse and builds the shape inset from its edges; :func:`largest_shape`
  picks the largest one;
* **spectral pick** - :class:`SpectralIndex` ``.pick(row, col)``: every pixel connected to the
  clicked one whose spectrum is within a spectral angle of the clicked spectrum. The spectral
  angle ignores brightness, so the panel is followed to the slit ends despite vignetting.

Then :func:`clean_mask` (every mode): saturated pixels out, the edge inset, and a row gate that
drops pixels whose brightness differs from the ROI median of the same slit row (shadows, bevel,
dust: same spectrum, less light, so the spectral angle cannot see them). :func:`assess_roi`
turns the result into QC items (row coverage, signal level, uniformity, homogeneity).

Data conventions:
    The cube is ``(lines, samples, bands)`` (the scan's ``(slit, scan, wavelength)``) in
    dark-subtracted counts; wavelengths in nm. Shape coordinates are matplotlib image
    coordinates: pixel (row ``r``, column ``c``) has its centre at ``(x=c, y=r)``, and a pixel
    belongs to a shape when its centre does. Angles are in degrees.

Threading:
    Pure NumPy / SciPy / OpenCV on the calling thread; no shared state. The cube-wide steps
    (:func:`saturation_map`, :func:`brightness_map`, :class:`SpectralIndex`) take about a second
    for a 552 x 3000 x 300 cube and work a few slit rows at a time.

Example:
    >>> import numpy as np
    >>> from modules.radiometric.roi_tools import brightness_map, clean_mask, rect_mask
    >>> cube = np.full((40, 60, 8), 1000.0, np.float32)
    >>> roi = rect_mask((40, 60), 10, 5, 50, 35)          # x0, y0, x1, y1
    >>> result = clean_mask(roi, brightness_map(cube, np.linspace(450, 900, 8)), inset_px=3)
    >>> int(roi.sum()), int(result.mask.sum())
    (1271, 875)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from matplotlib.path import Path as _MplPath
from scipy import ndimage as ndi

__all__ = ["rect_mask", "ellipse_mask", "polygon_mask", "simplify_polyline", "saturation_map",
           "brightness_map", "rgb_preview", "ShapeCandidate", "detect_shapes", "largest_shape",
           "SpectralIndex", "SpectralPick", "robust_threshold", "spectral_outliers", "CleanupResult", "clean_mask", "QcItem", "RoiQuality",
           "assess_roi", "SATURATION_FRACTION", "DEFAULT_INSET_PX", "DEFAULT_ROW_TOLERANCE",
           "DEFAULT_SAM_THRESHOLD_DEG", "SIMILARITY_RANGE_NM"]

#: A pixel is saturated when a band reaches this fraction of its saturation level.
SATURATION_FRACTION = 0.95
#: Edge inset (px) of every ROI: bevel, optical blur, keystone and one frame of motion smear.
DEFAULT_INSET_PX = 3
#: Row gate: pixels whose brightness differs more than this from their row's ROI median go.
DEFAULT_ROW_TOLERANCE = 0.05
#: Spectral angle (degrees) used when the automatic threshold cannot be estimated.
DEFAULT_SAM_THRESHOLD_DEG = 2.0
#: Bands used for brightness and spectral similarity (low-SNR ends are left out).
SIMILARITY_RANGE_NM = (430.0, 950.0)

_CROSS = ndi.generate_binary_structure(2, 1)        # 4-connectivity
_ROWS_PER_CHUNK = 32


def _disk(radius: int) -> np.ndarray:
    r = max(int(radius), 0)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r + r       # a round disk, not the 4-connected diamond


def _band_selection(wavelengths, band_range=SIMILARITY_RANGE_NM) -> np.ndarray:
    """Indices of the bands inside ``band_range``; all bands when fewer than 3 are inside."""
    wl = np.asarray(wavelengths, dtype=np.float64).ravel()
    idx = np.where(np.isfinite(wl) & (wl >= band_range[0]) & (wl <= band_range[1]))[0]
    return idx if idx.size >= 3 else np.arange(wl.size)


# ── manual shapes ──────────────────────────────────────────────────────────

def rect_mask(shape, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Pixels whose centre lies in the axis-aligned rectangle (corners in any order, inclusive)."""
    h, w = int(shape[0]), int(shape[1])
    xa, xb = sorted((float(x0), float(x1)))
    ya, yb = sorted((float(y0), float(y1)))
    mask = np.zeros((h, w), dtype=bool)
    c0, c1 = max(0, math.ceil(xa)), min(w - 1, math.floor(xb))
    r0, r1 = max(0, math.ceil(ya)), min(h - 1, math.floor(yb))
    if c0 <= c1 and r0 <= r1:
        mask[r0:r1 + 1, c0:c1 + 1] = True
    return mask


def ellipse_mask(shape, cx: float, cy: float, rx: float, ry: float, angle_deg: float = 0.0) -> np.ndarray:
    """Pixels whose centre lies in the ellipse with semi-axes ``rx`` (along x before rotation) and
    ``ry``, rotated by ``angle_deg`` (counter-clockwise in image coordinates, i.e. x towards y)."""
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=bool)
    rx, ry = abs(float(rx)), abs(float(ry))
    if rx <= 0 or ry <= 0:
        return mask
    reach = max(rx, ry)
    c0, c1 = max(0, math.floor(cx - reach)), min(w - 1, math.ceil(cx + reach))
    r0, r1 = max(0, math.floor(cy - reach)), min(h - 1, math.ceil(cy + reach))
    if c0 > c1 or r0 > r1:
        return mask
    yy, xx = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    t = math.radians(angle_deg)
    cos_t, sin_t = math.cos(t), math.sin(t)
    cos_t, sin_t = (0.0 if abs(cos_t) < 1e-12 else cos_t), (0.0 if abs(sin_t) < 1e-12 else sin_t)
    dx, dy = xx - cx, yy - cy
    u = dx * cos_t + dy * sin_t
    v = -dx * sin_t + dy * cos_t
    mask[r0:r1 + 1, c0:c1 + 1] = (u / rx) ** 2 + (v / ry) ** 2 <= 1.0
    return mask


def polygon_mask(shape, vertices: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Pixels whose centre lies inside the closed polygon ``[(x, y), ...]`` (even-odd rule)."""
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=bool)
    pts = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3 or not np.all(np.isfinite(pts)):
        return mask
    c0, c1 = max(0, math.floor(pts[:, 0].min())), min(w - 1, math.ceil(pts[:, 0].max()))
    r0, r1 = max(0, math.floor(pts[:, 1].min())), min(h - 1, math.ceil(pts[:, 1].max()))
    if c0 > c1 or r0 > r1:
        return mask
    yy, xx = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    inside = _MplPath(pts).contains_points(np.column_stack([xx.ravel(), yy.ravel()]))
    mask[r0:r1 + 1, c0:c1 + 1] = inside.reshape(xx.shape)
    return mask


def simplify_polyline(points: Sequence[Tuple[float, float]], tolerance: float = 0.5) -> List[Tuple[float, float]]:
    """Douglas-Peucker simplification of a freehand path (``tolerance`` in pixels).

    Returns the kept ``(x, y)`` points (the path is not closed by repeating the first point).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) <= 3:
        return [tuple(map(float, p)) for p in pts]
    approx = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), float(tolerance), True)
    out = approx.reshape(-1, 2)
    if len(out) < 3:
        out = pts[np.linspace(0, len(pts) - 1, 3).astype(int)]
    return [(float(x), float(y)) for x, y in out]


# ── cube-wide maps ─────────────────────────────────────────────────────────

def saturation_map(cube, ceiling=None, full_scale=None, fraction: float = SATURATION_FRACTION,
                   dilate_px: int = 1) -> Tuple[np.ndarray, str]:
    """Pixels where any band reaches ``fraction`` of its saturation level, grown by ``dilate_px``.

    Args:
        cube: ``(lines, samples, bands)`` counts.
        ceiling: Optional ``(lines, bands)`` value a saturated sensor pixel has after the scan's
            processing (dark subtraction, Smile, binning: the scan controller's
            ``saturation_ceiling``); the most exact level. Rows/bands with a level <= 0 are ignored.
        full_scale: Otherwise the frames' full scale (``camera_bit_info``).
        fraction: Fraction of the level counted as saturated.
        dilate_px: Neighbours added around each saturated pixel (blooming).

    Returns:
        ``(mask, source)`` with ``source`` ``"ceiling"``, ``"full scale"``, ``"data type"`` or
        ``"unknown"`` (then the mask is empty: the level is not known).
    """
    h, w, b = cube.shape
    level = None
    if ceiling is not None and np.shape(ceiling) == (h, b):
        level = np.asarray(ceiling, dtype=np.float32) * np.float32(fraction)
        level = np.where(np.isfinite(level) & (level > 0), level, np.inf)[:, None, :]
        source = "ceiling"
    elif full_scale:
        level = np.float32(float(full_scale) * fraction)
        source = "full scale"
    elif np.issubdtype(np.asarray(cube[:1, :1, :1]).dtype, np.integer):
        level = np.float32(np.iinfo(np.asarray(cube[:1, :1, :1]).dtype).max * fraction)
        source = "data type"
    else:
        return np.zeros((h, w), dtype=bool), "unknown"
    mask = np.zeros((h, w), dtype=bool)
    for r0 in range(0, h, _ROWS_PER_CHUNK):
        part = np.asarray(cube[r0:r0 + _ROWS_PER_CHUNK], dtype=np.float32)
        lv = level[r0:r0 + _ROWS_PER_CHUNK] if isinstance(level, np.ndarray) and level.ndim == 3 else level
        mask[r0:r0 + _ROWS_PER_CHUNK] = (part >= lv).any(axis=2)
    if dilate_px > 0 and mask.any():
        mask = ndi.binary_dilation(mask, structure=_disk(dilate_px))
    return mask, source


def brightness_map(cube, wavelengths, band_range=SIMILARITY_RANGE_NM) -> np.ndarray:
    """Mean counts over the bands in ``band_range`` per pixel, float32 ``(lines, samples)``
    (NaN where a band is NaN)."""
    idx = _band_selection(wavelengths, band_range)
    h, w = cube.shape[:2]
    out = np.empty((h, w), dtype=np.float32)
    for r0 in range(0, h, _ROWS_PER_CHUNK):
        part = np.asarray(cube[r0:r0 + _ROWS_PER_CHUNK][:, :, idx], dtype=np.float32)
        out[r0:r0 + _ROWS_PER_CHUNK] = part.mean(axis=2)
    return out


def rgb_preview(cube, wavelengths, targets=(640.0, 550.0, 460.0)) -> np.ndarray:
    """Display image ``(lines, samples, 3)`` in ``[0, 1]``: the bands nearest ``targets`` (red,
    green, blue), each stretched between its 1st and 99.5th percentile; grey brightness when the
    cube does not span the targets."""
    wl = np.asarray(wavelengths, dtype=np.float64).ravel()
    if wl.size and np.nanmin(wl) <= min(targets) + 20 and np.nanmax(wl) >= max(targets) - 20:
        planes = [np.asarray(cube[:, :, int(np.nanargmin(np.abs(wl - t)))], dtype=np.float32) for t in targets]
    else:
        grey = brightness_map(cube, wl if wl.size else np.arange(cube.shape[2]))
        planes = [grey, grey, grey]
    out = []
    for p in planes:
        finite = p[np.isfinite(p)]
        lo, hi = (np.percentile(finite, [1, 99.5]) if finite.size else (0.0, 1.0))
        out.append(np.clip((np.nan_to_num(p, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1))
    return np.dstack(out).astype(np.float32)


# ── shape detection ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ShapeCandidate:
    """One bright compact blob found by :func:`detect_shapes`.

    Attributes:
        kind: ``"rectangle"``, ``"ellipse"`` or ``"irregular"``.
        area_px: Pixels of the blob (holes filled).
        score: How well the shape fits (1 = perfectly), for ``kind``.
        center: ``(x, y)`` of the fitted shape.
        size: ``(width, height)`` of the fitted shape before the inset, pixels.
        angle_deg: Rotation of the fitted shape (OpenCV convention).
        axis_aligned: The shape's sides/axes are along the image axes (within 1 degree).
        outline: ``(N, 2)`` ``(x, y)`` outline of the shape inset by ``inset_px`` (empty for irregular).
        mask: Pixels inside the inset outline that belong to the blob (screws and labels left out).
        touches_border: The blob reaches the image border (the panel may be cut off).
        mean_brightness: Mean brightness of the blob.
    """

    kind: str
    area_px: int
    score: float
    center: Tuple[float, float]
    size: Tuple[float, float]
    angle_deg: float
    axis_aligned: bool
    outline: np.ndarray
    mask: np.ndarray = field(repr=False)
    touches_border: bool = False
    mean_brightness: float = float("nan")

    def label(self) -> str:
        """Short description for a list, e.g. ``rectangle 40861 px (0.99)``."""
        return f"{self.kind} {self.area_px} px ({self.score:.2f})" + (", at border" if self.touches_border else "")


def _otsu(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean0 = m0 / w0
        mean1 = (m0[-1] - m0) / w1
        between = w0 * w1 * (mean0 - mean1) ** 2
    between = np.nan_to_num(between, nan=-1.0)
    return float(centers[int(np.argmax(between))])


def _rect_outline(center, size, angle, inset) -> np.ndarray:
    w, h = max(size[0] - 2 * inset, 0.0), max(size[1] - 2 * inset, 0.0)
    return cv2.boxPoints((tuple(center), (w, h), angle)).astype(np.float64)


def _ellipse_outline(center, size, angle, inset, n=96) -> np.ndarray:
    rx, ry = max(size[0] / 2 - inset, 0.0), max(size[1] / 2 - inset, 0.0)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a = math.radians(angle)
    x = rx * np.cos(t) * math.cos(a) - ry * np.sin(t) * math.sin(a) + center[0]
    y = rx * np.cos(t) * math.sin(a) + ry * np.sin(t) * math.cos(a) + center[1]
    return np.column_stack([x, y])


def detect_shapes(brightness, saturated=None, *, min_area_px: int = 500, opening_px: int = 2,
                  inset_px: int = DEFAULT_INSET_PX, rect_min_score: float = 0.90,
                  ellipse_min_score: float = 0.92) -> List[ShapeCandidate]:
    """Bright compact blobs of ``brightness``, largest first.

    Hysteresis: weak = above the Otsu level of log brightness (background vs bright objects),
    strong = above half the 99th-percentile brightness. A blob is kept when it reaches the
    strong level somewhere, so the darker slit ends of a panel stay connected to its bright
    centre while dim objects are dropped; no per-row normalisation (a row without the panel
    would be stretched to "bright"). Saturated pixels count as bright
    for the shape (a hot spot must not break it; :func:`clean_mask` removes them later). Holes
    are filled for the shape fit only.

    Args:
        brightness: ``(lines, samples)`` (:func:`brightness_map`).
        saturated: Optional saturated-pixel mask.
        min_area_px: Smaller blobs are ignored.
        opening_px: Opening radius that cuts thin bridges to neighbouring bright objects.
        inset_px: Inset of the outline and mask from the fitted shape's edges.
        rect_min_score, ellipse_min_score: Fit quality needed for each kind.
    """
    b = np.asarray(brightness, dtype=np.float64)
    finite = np.isfinite(b) & (b > 0)
    if not finite.any():
        return []
    logb = np.full(b.shape, np.nan)
    logb[finite] = np.log(b[finite])
    low = _otsu(logb[finite])
    unsat = finite if saturated is None else finite & ~np.asarray(saturated, dtype=bool)
    peak = float(np.percentile(b[unsat], 99)) if unsat.any() else float(np.max(b[finite]))
    below = finite & (logb < low)
    above = finite & (logb >= low)
    if below.any() and above.any() and float(np.mean(b[above])) < 2.0 * float(np.mean(b[below])) and peak > 0:
        # no real background (the classes differ by less than 2x: Otsu split the target's own
        # vignetting, e.g. a panel filling the frame): everything above a quarter of the peak
        low = math.log(0.25 * peak)
    high = max(low, math.log(0.5 * peak)) if peak > 0 else low
    weak = finite & (logb >= low)
    strong = finite & (logb >= high)
    if saturated is not None:
        strong |= saturated
        weak |= saturated
    lab, n = ndi.label(weak, structure=_CROSS)
    if n == 0:
        return []
    keep = np.unique(lab[strong])
    blobs = np.isin(lab, keep[keep > 0])
    if opening_px > 0:
        blobs = ndi.binary_opening(blobs, structure=_disk(opening_px))
    filled = ndi.binary_fill_holes(blobs)
    lab2, n2 = ndi.label(filled, structure=_CROSS)
    h, w = b.shape
    found = []
    for i in range(1, n2 + 1):
        comp = lab2 == i
        area = int(comp.sum())
        if area < min_area_px:
            continue
        contours, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = max(contours, key=len).reshape(-1, 2).astype(np.float32)
        (rcx, rcy), (rw, rh), rang = cv2.minAreaRect(pts)
        # the fit runs through boundary pixel centres: the shape is one pixel larger each way
        rect_score = 1.0 - abs(1.0 - area / max((rw + 1) * (rh + 1), 1.0))
        ell = None
        ell_score = -1.0
        if len(pts) >= 5:
            (ecx, ecy), (ew, eh), eang = cv2.fitEllipse(pts)
            ell = ((ecx, ecy), (ew + 1, eh + 1), eang)
            ell_score = 1.0 - abs(1.0 - area / max(math.pi / 4 * (ew + 1) * (eh + 1), 1.0))
        rect_ok, ell_ok = rect_score >= rect_min_score, ell is not None and ell_score >= ellipse_min_score
        if rect_ok and (not ell_ok or rect_score >= ell_score):
            kind, score = "rectangle", rect_score
            center, size, angle = (rcx, rcy), (rw + 1, rh + 1), rang
            outline = _rect_outline(center, size, angle, inset_px)
        elif ell_ok:
            kind, score = "ellipse", ell_score
            center, size, angle = ell
            outline = _ellipse_outline(center, size, angle, inset_px)
        else:
            kind, score = "irregular", max(rect_score, ell_score)
            center, size, angle = (rcx, rcy), (rw + 1, rh + 1), rang
            outline = np.zeros((0, 2))
        aligned = min(angle % 90.0, 90.0 - angle % 90.0) <= 1.0
        mask = (polygon_mask((h, w), outline) & blobs & comp) if len(outline) else np.zeros((h, w), bool)
        border = bool(comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any())
        found.append(ShapeCandidate(kind, area, float(score), (float(center[0]), float(center[1])),
                                    (float(size[0]), float(size[1])), float(angle), bool(aligned), outline,
                                    mask, border, float(np.nanmean(b[comp]))))
    return sorted(found, key=lambda c: c.area_px, reverse=True)


def largest_shape(candidates: Sequence[ShapeCandidate]) -> Optional[ShapeCandidate]:
    """The largest rectangle or ellipse among ``candidates`` (``None`` if there is none)."""
    shapes = [c for c in candidates if c.kind in ("rectangle", "ellipse")]
    return max(shapes, key=lambda c: c.area_px) if shapes else None


# ── spectral pick ──────────────────────────────────────────────────────────

def robust_threshold(angles, sigmas: float = 8.0, floor_deg: float = 0.75, ceiling_deg: float = 5.0) -> Optional[float]:
    """Spectral-angle limit for one material from its own angles (degrees):
    ``clip(median + sigmas x 1.4826 MAD, floor_deg, ceiling_deg)``.

    Noise angles of one material cluster tightly around their median (a Spectralon scan:
    0.34 +- 0.024 deg), while a different material sits well apart (a white paper label on the
    panel: about 1.1 deg and more). Eight robust sigmas keep all of the material; the floor
    leaves room for a small spectral-shape drift along the slit (Smile, wavelength-dependent
    vignetting). ``None`` when fewer than 9 finite angles are given.
    """
    a = np.asarray(angles, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size < 9:
        return None
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return float(np.clip(med + sigmas * 1.4826 * mad, floor_deg, ceiling_deg))


def _bin_groups(wavelengths, bin_nm: float = 10.0) -> List[np.ndarray]:
    """Band indices of each ``bin_nm`` bin inside :data:`SIMILARITY_RANGE_NM`."""
    wl = np.asarray(wavelengths, dtype=np.float64).ravel()
    idx = _band_selection(wl)
    if not idx.size:
        return []
    bins = np.floor((wl[idx] - wl[idx].min()) / float(bin_nm)).astype(int)
    return [idx[bins == k] for k in np.unique(bins)]


def _row_unit(y: int, cols: np.ndarray, index=None, cube=None, groups=None) -> np.ndarray:
    """Binned unit spectra ``(n, K)`` of row ``y``'s pixels ``cols`` (NaN rows for zero spectra)."""
    if index is not None:
        return index.unit[y, cols]
    part = np.asarray(cube[y, cols, :], dtype=np.float32)
    out = np.empty((cols.size, len(groups)), dtype=np.float32)
    for k, g in enumerate(groups):
        out[:, k] = part[:, g].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def _row_medians(mask, index=None, cube=None, wavelengths=None, per_row: int = 512):
    """``(rows, (n_rows, K) median unit spectrum of each row's ROI pixels)`` - at most ``per_row``
    evenly spread pixels per row; rows without a finite spectrum are left out."""
    groups = None if index is not None else _bin_groups(wavelengths)
    rows, meds = [], []
    for y in np.where(mask.any(axis=1))[0]:
        cols = np.nonzero(mask[y])[0]
        if cols.size > per_row:
            cols = cols[np.linspace(0, cols.size - 1, per_row).astype(int)]
        u = _row_unit(y, cols, index, cube, groups)
        u = u[np.all(np.isfinite(u), axis=1)]
        if len(u):
            rows.append(y)
            meds.append(np.median(u, axis=0))
    return np.asarray(rows, dtype=int), (np.asarray(meds, dtype=np.float64) if meds else np.zeros((0, 0)))


def spectral_outliers(mask, *, index=None, cube=None, wavelengths=None,
                      threshold_deg: Optional[float] = None, smooth_rows: int = 31) -> Tuple[np.ndarray, Optional[float]]:
    """Pixels of ``mask`` whose spectrum departs from the ROI's own spectrum of their slit row.

    The reference of each row is the median unit spectrum of that row's ROI pixels, smoothed
    along the slit (so a minority of other material - a label, tape, the holder edge - does
    not move it, while a slow Smile drift is followed). The limit is ``threshold_deg`` or
    :func:`robust_threshold` of the ROI's own angles. Works one slit row at a time (no copy of
    the ROI's spectra).

    Args:
        mask: ROI ``(lines, samples)``.
        index: A :class:`SpectralIndex` of the cube (fast), or ``cube`` + ``wavelengths``.
        threshold_deg: Fixed limit; ``None`` = automatic.
        smooth_rows: Running-median window along the slit.

    Returns:
        ``(outlier mask, limit used or None when the ROI is too small)``.
    """
    m = np.asarray(mask, dtype=bool)
    out = np.zeros(m.shape, dtype=bool)
    if m.sum() < 9:
        return out, None
    rows, meds = _row_medians(m, index, cube, wavelengths)
    if not rows.size:
        out[m] = True
        return out, None
    refs = _references_from_medians(meds, rows, m.shape[0], smooth_rows)
    groups = None if index is not None else _bin_groups(wavelengths)
    angles = {}
    for y in np.where(m.any(axis=1))[0]:
        cols = np.nonzero(m[y])[0]
        u = _row_unit(y, cols, index, cube, groups)
        with np.errstate(invalid="ignore"):
            angles[y] = (cols, np.degrees(np.arccos(np.clip(u @ refs[y], -1.0, 1.0))))
    if threshold_deg is not None:
        limit = float(threshold_deg)
    else:
        every = np.concatenate([a for _, a in angles.values()])
        limit = robust_threshold(every[:: max(1, every.size // 200000)])
    if limit is None:
        return out, None
    for y, (cols, a) in angles.items():
        bad = ~np.isfinite(a) | (a > limit)
        out[y, cols[bad]] = True
    return out, limit


def _row_references(unit_rows: List[np.ndarray], rows: np.ndarray, n_rows: int, smooth_rows: int = 31) -> np.ndarray:
    """Per slit row reference unit spectrum ``(n_rows, K)``: the median spectrum of each row in
    ``rows`` (``unit_rows[i]`` = that row's ``(n, K)`` unit spectra), smoothed along the slit by a
    running median, held constant beyond the first / last row."""
    med = np.array([np.median(u, axis=0) for u in unit_rows], dtype=np.float64)
    return _references_from_medians(med, rows, n_rows, smooth_rows)


def _references_from_medians(med: np.ndarray, rows: np.ndarray, n_rows: int, smooth_rows: int = 31,
                             hold_rows: Optional[int] = None) -> np.ndarray:
    """:func:`_row_references` from per-row median spectra already computed. With ``hold_rows``,
    rows farther than that from every reference row get NaN (no reference) instead of the nearest."""
    med = np.array(med, dtype=np.float64)
    if len(rows) >= 3:
        med = ndi.median_filter(med, size=(min(smooth_rows, len(rows) | 1), 1), mode="nearest")
    med /= np.maximum(np.linalg.norm(med, axis=1, keepdims=True), 1e-12)
    every = np.arange(n_rows)
    position = np.clip(np.searchsorted(rows, every), 0, len(rows) - 1)
    previous = np.clip(position - 1, 0, len(rows) - 1)
    nearer = np.where(np.abs(rows[previous] - every) < np.abs(rows[position] - every), previous, position)
    refs = med[nearer].astype(np.float32)
    if hold_rows is not None:
        refs[np.abs(rows[nearer] - every) > int(hold_rows)] = np.nan
    return refs


@dataclass(frozen=True)
class SpectralPick:
    """Result of :meth:`SpectralIndex.pick`.

    Attributes:
        mask: The connected region of similar spectra (``False`` everywhere when the seed was invalid).
        seed: ``(row, col)`` clicked.
        threshold_deg: Spectral angle used.
        auto: Whether ``threshold_deg`` was estimated from the seed's neighbourhood.
        angles: Spectral angle of every pixel to the seed spectrum, degrees (NaN = invalid pixel).
    """

    mask: np.ndarray = field(repr=False)
    seed: Tuple[int, int]
    threshold_deg: float
    auto: bool
    angles: np.ndarray = field(repr=False)


class SpectralIndex:
    """Unit spectra of a cube for spectral-angle picking (built once, then each click is fast).

    Bands inside :data:`SIMILARITY_RANGE_NM` are summed into ``bin_nm`` bins (more signal per
    value, less Smile influence); bins darker than 2 % of the brightest are dropped.

    Args:
        cube: ``(lines, samples, bands)`` counts.
        wavelengths: Band centres, nm.
        saturated: Optional saturated-pixel mask: such pixels never join a region.
        bin_nm: Width of the similarity bins, nm.
    """

    def __init__(self, cube, wavelengths, saturated=None, bin_nm: float = 10.0):
        groups = _bin_groups(wavelengths, bin_nm)
        h, w = cube.shape[:2]
        x = np.empty((h, w, len(groups)), dtype=np.float32)
        for r0 in range(0, h, _ROWS_PER_CHUNK):
            part = np.asarray(cube[r0:r0 + _ROWS_PER_CHUNK], dtype=np.float32)
            for k, g in enumerate(groups):
                x[r0:r0 + _ROWS_PER_CHUNK, :, k] = part[:, :, g].sum(axis=2)
        n_bands = sum(len(g) for g in groups)
        #: :func:`brightness_map` of the cube (mean counts over the similarity bands), from the same sums.
        self.brightness = (x.sum(axis=2) / max(n_bands, 1)).astype(np.float32)
        sample = x.reshape(-1, x.shape[2])[::max(1, (h * w) // 20000)]
        med = np.nanmedian(sample, axis=0) if sample.size else np.zeros(0)
        keep = np.isfinite(med) & (med >= 0.02 * (np.nanmax(med) if med.size else 0))
        if keep.sum() >= 3 and not keep.all():
            x = np.ascontiguousarray(x[:, :, keep])
        norm = np.sqrt(np.einsum("ijk,ijk->ij", x, x))
        with np.errstate(divide="ignore", invalid="ignore"):
            x /= norm[:, :, None]
        self.unit = x
        self.valid = np.isfinite(norm) & (norm > 0) & np.all(np.isfinite(x), axis=2)
        if saturated is not None:
            self.valid &= ~np.asarray(saturated, dtype=bool)
        self.shape = (h, w)

    def seed_spectrum(self, row: int, col: int, half: int = 2) -> Optional[np.ndarray]:
        """Median unit spectrum of the valid pixels in a ``(2 half + 1)`` square around the seed."""
        h, w = self.shape
        r0, r1 = max(0, row - half), min(h, row + half + 1)
        c0, c1 = max(0, col - half), min(w, col + half + 1)
        sel = self.valid[r0:r1, c0:c1]
        if not sel.any():
            return None
        s = np.median(self.unit[r0:r1, c0:c1][sel], axis=0)
        n = float(np.linalg.norm(s))
        return (s / n).astype(np.float32) if n > 0 else None

    def angle_map(self, spectrum: np.ndarray) -> np.ndarray:
        """Spectral angle (degrees, float32) of every pixel to ``spectrum``; NaN for invalid pixels."""
        h, w = self.shape
        out = np.empty((h, w), dtype=np.float32)
        s = np.asarray(spectrum, dtype=np.float32)
        for r0 in range(0, h, _ROWS_PER_CHUNK):
            cos = np.clip(self.unit[r0:r0 + _ROWS_PER_CHUNK] @ s, -1.0, 1.0)
            out[r0:r0 + _ROWS_PER_CHUNK] = np.degrees(np.arccos(cos))
        out[~self.valid] = np.nan
        return out

    @staticmethod
    def auto_threshold(angles: np.ndarray, row: int, col: int, half: int = 7) -> Optional[float]:
        """Threshold from the angles in a ``(2 half + 1)`` square around the seed: :func:`robust_threshold`."""
        h, w = angles.shape
        win = angles[max(0, row - half):min(h, row + half + 1), max(0, col - half):min(w, col + half + 1)]
        return robust_threshold(win)

    def _region(self, angles, thr, row, col, hole_px):
        """Connected region (4-connectivity, one-pixel bridges cut) of ``angles <= thr`` at the seed."""
        h, w = self.shape
        cand = self.valid & (angles <= thr)
        cand = ndi.binary_opening(cand, structure=_CROSS)          # cut one-pixel bridges
        lab, _ = ndi.label(cand, structure=_CROSS)
        target = lab[row, col]
        if target == 0:                                             # seed on a speck: nearest region
            r0, r1, c0, c1 = max(0, row - 3), min(h, row + 4), max(0, col - 3), min(w, col + 4)
            near = lab[r0:r1, c0:c1]
            if near.any():
                rr, cc = np.nonzero(near)
                k = int(np.argmin((rr + r0 - row) ** 2 + (cc + c0 - col) ** 2))
                target = near[rr[k], cc[k]]
        if target == 0:
            return np.zeros((h, w), dtype=bool)
        region = lab == target
        holes = ndi.binary_fill_holes(region) & ~region
        if holes.any() and hole_px > 0:
            hl, hn = ndi.label(holes, structure=_CROSS)
            sizes = np.bincount(hl.ravel())
            small = np.isin(hl, np.where(sizes <= hole_px)[0][1:] if hn else [])
            region |= small & self.valid
        return region

    def row_angle_map(self, region: np.ndarray, smooth_rows: int = 31, min_row_px: int = 20) -> np.ndarray:
        """Angle of every pixel to the reference spectrum of its own slit row, taken from
        ``region``: the median of each row with at least ``min_row_px`` region pixels (a row
        holding one stray pixel of a neighbouring material must not become a reference),
        smoothed along the slit, and held at most ``smooth_rows`` rows beyond the region's ends
        (NaN further out: the region grows step by step, it cannot jump into a far material)."""
        counts = region.sum(axis=1)
        rows = np.where(counts >= min(int(min_row_px), max(1, int(counts.max()))))[0]
        refs = _row_references([self.unit[y, region[y]] for y in rows], rows, self.shape[0], smooth_rows)
        far = np.ones(self.shape[0], dtype=bool)
        lo, hi = int(rows.min()), int(rows.max())
        far[max(0, lo - smooth_rows):hi + smooth_rows + 1] = False
        refs[far] = np.nan
        with np.errstate(invalid="ignore"):
            cos = np.clip(np.einsum("ijk,ik->ij", self.unit, refs), -1.0, 1.0)
        out = np.degrees(np.arccos(cos)).astype(np.float32)
        out[~self.valid] = np.nan
        return out

    def pick(self, row: int, col: int, threshold_deg: Optional[float] = None, hole_px: int = 25,
             adaptive: bool = True) -> SpectralPick:
        """The 4-connected region of pixels within ``threshold_deg`` of the seed spectrum.

        Pass 1 compares every pixel with the seed spectrum (fixed: no running mean that could drift
        into a neighbouring material). With ``adaptive``, up to two more passes compare each pixel
        with the smoothed median spectrum of *its own slit row* in the previous region, so a slow
        spectral-shape change along the slit (Smile, wavelength-dependent vignetting) does not cut
        the panel ends while the threshold stays tight. Holes up to ``hole_px`` pixels (noise
        speckle) are filled; larger ones (screws, labels, saturated spots) stay out.

        Args:
            row, col: Seed pixel.
            threshold_deg: Spectral angle limit; ``None`` = :meth:`auto_threshold`
                (:data:`DEFAULT_SAM_THRESHOLD_DEG` when it cannot be estimated).
            hole_px: Largest hole filled.
            adaptive: Run the row-adaptive passes.
        """
        h, w = self.shape
        row, col = int(row), int(col)
        empty = np.zeros((h, w), dtype=bool)
        if not (0 <= row < h and 0 <= col < w):
            return SpectralPick(empty, (row, col), float(threshold_deg or DEFAULT_SAM_THRESHOLD_DEG),
                                threshold_deg is None, np.full((h, w), np.nan, np.float32))
        spectrum = self.seed_spectrum(row, col)
        if spectrum is None:
            return SpectralPick(empty, (row, col), float(threshold_deg or DEFAULT_SAM_THRESHOLD_DEG),
                                threshold_deg is None, np.full((h, w), np.nan, np.float32))
        angles = self.angle_map(spectrum)
        auto = threshold_deg is None
        thr = self.auto_threshold(angles, row, col) if auto else float(threshold_deg)
        if thr is None:
            thr = DEFAULT_SAM_THRESHOLD_DEG
        region = self._region(angles, thr, row, col, hole_px)
        if adaptive and region.any():
            for _ in range(4):
                row_angles = self.row_angle_map(region)
                grown = self._region(row_angles, thr, row, col, hole_px)
                if not grown.any():
                    break
                angles = row_angles
                if np.array_equal(grown, region):
                    break
                region = grown
        return SpectralPick(region, (row, col), float(thr), auto, angles)


# ── common clean-up ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CleanupResult:
    """Result of :func:`clean_mask`: the effective ROI and what each step removed (pixels)."""

    mask: np.ndarray = field(repr=False)
    drawn_px: int
    saturated_px: int
    inset_removed_px: int
    gate_removed_px: int
    spectral_removed_px: int = 0
    spectral_threshold_deg: Optional[float] = None
    #: Slit rows dropped as a separate band (holder, bevel, label row) by the row segmentation.
    rows_left_out: Tuple[int, ...] = ()

    @property
    def final_px(self) -> int:
        """Pixels left."""
        return int(self.mask.sum())


#: Relative jump of the row median brightness between neighbouring slit rows that is a step
#: (a holder band, a bevel row). Above the steepest realistic vignetting slope (about 2 % per row).
STEP_TOLERANCE = 0.06
#: Rows with fewer ROI pixels are not used to find spectral steps (their median spectrum is noisy).
MIN_SPECTRAL_ROW_PX = 20
#: Interior bands of at most this many rows whose two sides continue each other are a defect
#: line of the slit (dust, a scratch): kept, they need their own factors.
_THIN_BAND_ROWS = 5


def _band_rows(rows: np.ndarray, levels: np.ndarray, counts: np.ndarray, spectra: Optional[np.ndarray] = None,
               step: float = STEP_TOLERANCE, seed_rows=()) -> List[int]:
    """Positions (indices into ``rows``) of the ROI rows that are a separate band, not target.

    Rows are compared only with their neighbours on the slit; a gap of more than one slit row
    without ROI pixels starts an independent block (two targets picked on purpose stay).
    Within a block a boundary is a step when the row median brightness ``levels`` jumps by more
    than ``step`` both directly and across one row (so one noisy row is no step and a one-row
    transition cannot hide one), or - with ``spectra`` (per-row median unit spectra) - when the
    row spectrum jumps by more than the row medians' own noise allows (:func:`robust_threshold`,
    floor 0.3 deg).

    * Spectral steps separate materials: only the material holding a seed row (``seed_rows``),
      else the one with the most pixels, stays.
    * Within it, brightness steps separate bands. A thin interior band whose two sides continue
      each other (allowing for the local vignetting slope) is a slit defect line and is kept; then
      end bands smaller than half the largest band are dropped (a holder frame or bevel rows at
      the ROI's top or bottom), one after the other. Other interior bands are kept.
    """
    n = len(rows)
    drop: List[int] = []
    if n < 2:
        return drop
    lv = np.asarray(levels, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    seeds = set(int(r) for r in seed_rows)
    unit = None
    spectral_limit = None
    if spectra is not None and len(spectra) == n:
        unit = np.asarray(spectra, dtype=np.float64)
        unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
        ok = (counts[1:] >= MIN_SPECTRAL_ROW_PX) & (counts[:-1] >= MIN_SPECTRAL_ROW_PX) & (np.diff(rows) <= 2)
        if ok.sum() >= 9:
            consecutive = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", unit[1:], unit[:-1]), -1, 1)))
            spectral_limit = robust_threshold(consecutive[ok], sigmas=8.0, floor_deg=0.3, ceiling_deg=5.0)

    def rel(i, j):
        return abs(lv[i] / lv[j] - 1.0) if lv[j] > 0 else np.inf

    def angle(i, j):
        return float(np.degrees(np.arccos(np.clip(unit[i] @ unit[j], -1.0, 1.0))))

    def spectral_ok(*idx):
        return all(counts[i] >= MIN_SPECTRAL_ROW_PX for i in idx)

    starts = [0] + [i for i in range(1, n) if rows[i] - rows[i - 1] > 2]
    for a, z in zip(starts, starts[1:] + [n]):
        spectral_cuts = []
        if spectral_limit is not None:
            # between neighbouring well-filled rows, so a thin bridge of a few pixels cannot hide a
            # change of material; the thin rows in between go with the material that follows
            big = [i for i in range(a, z) if spectral_ok(i)]
            for t in range(1, len(big)):
                i, j = big[t - 1], big[t]
                far_l, far_r = big[max(t - 2, 0)], big[min(t + 1, len(big) - 1)]
                if angle(j, i) > spectral_limit and angle(far_r, far_l) > spectral_limit:
                    spectral_cuts.append(i + 1)
        level_cuts = []
        for k in range(a + 1, z):
            far_l, far_r = max(k - 2, a), min(k + 1, z - 1)
            if k not in spectral_cuts and rel(k, k - 1) > step and rel(far_r, far_l) > step:
                level_cuts.append(k)
        bounds = [a, *spectral_cuts, z]
        materials = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        with_seed = [m for m in materials if any(int(rows[i]) in seeds for i in range(*m))]
        keep = with_seed or [max(materials, key=lambda m: int(counts[m[0]:m[1]].sum()))]
        for m in materials:
            if m not in keep:
                drop.extend(range(*m))
        for s, e in keep:
            cuts = [k for k in level_cuts if s < k < e]
            segs = [list(p) for p in zip([s, *cuts], [*cuts, e])]
            merged = True
            while merged and len(segs) >= 3:                       # thin interior defect lines
                merged = False
                for i in range(1, len(segs) - 1):
                    left, mid, right = segs[i - 1], segs[i], segs[i + 1]
                    width = mid[1] - mid[0]
                    if width > _THIN_BAND_ROWS:
                        continue
                    tail = lv[max(left[0], left[1] - 6):left[1]]
                    slope = float(np.median(np.abs(np.diff(tail)) / np.maximum(tail[:-1], 1e-12))) if tail.size > 1 else 0.0
                    if rel(right[0], left[1] - 1) <= step + slope * (width + 1):
                        segs[i - 1:i + 2] = [[left[0], right[1]]]
                        merged = True
                        break
            pixels = lambda seg: int(counts[seg[0]:seg[1]].sum())
            has_seed = lambda seg: any(int(rows[i]) in seeds for i in range(seg[0], seg[1]))
            for end in (0, -1):
                while len(segs) > 1:
                    largest = max(pixels(g) for g in segs)
                    edge = segs[end]
                    if has_seed(edge) or pixels(edge) >= 0.5 * largest:
                        break
                    drop.extend(range(edge[0], edge[1]))
                    segs.pop(end)
    # blocks across a gap: targets of the same material picked on purpose stay, another material
    # (e.g. reached through a thin bridge the inset then removed) goes
    blocks = list(zip(starts, starts[1:] + [n]))
    if spectral_limit is not None and len(blocks) > 1:
        kept = set(range(n)) - set(drop)
        spectra_of = {}
        for b in blocks:
            idx = [i for i in range(*b) if i in kept and spectral_ok(i)]
            if idx:
                s = np.median(unit[idx], axis=0)
                spectra_of[b] = s / max(float(np.linalg.norm(s)), 1e-12)
        if len(spectra_of) > 1:
            seeded = [b for b in spectra_of if any(int(rows[i]) in seeds for i in range(*b))]
            main = seeded[0] if seeded else max(spectra_of, key=lambda b: int(counts[b[0]:b[1]].sum()))
            for b, s in spectra_of.items():
                if b != main and b not in seeded and \
                        np.degrees(np.arccos(np.clip(s @ spectra_of[main], -1, 1))) > spectral_limit:
                    drop.extend(range(*b))
    return sorted(set(drop))


def clean_mask(mask, brightness, saturated=None, *, inset_px: int = DEFAULT_INSET_PX,
               row_tolerance: Optional[float] = DEFAULT_ROW_TOLERANCE, step_tolerance: Optional[float] = None,
               index: Optional["SpectralIndex"] = None, cube=None, wavelengths=None,
               spectral: bool = True, spectral_threshold_deg: Optional[float] = None,
               seed_rows=()) -> CleanupResult:
    """The last step of every ROI mode: what the factor fit may use of the drawn ``mask``.

    1. saturated and non-finite pixels out (the factor fit itself only drops NaN);
    2. inset: erosion by a disk of ``inset_px``. The image border does not count as an edge: a
       panel longer than the slit reaches the first/last slit row, and those rows are panel;
    3. spectral gate (with ``index`` or ``cube`` + ``wavelengths``, unless ``spectral`` is off):
       :func:`spectral_outliers` - other material inside the ROI (a label, tape, the holder edge)
       that is as bright as the target, so no brightness rule can see it;
    4. row gate: in each slit row, pixels whose brightness differs more than ``row_tolerance``
       from the median of that row's ROI pixels go (shadow, bevel, dust, hot spot). Per row,
       because the slit-direction falloff is what the row-wise factors are there to correct;
    5. row bands (:func:`_band_rows`): a holder band or bevel row that fills whole rows - which
       step 4 cannot see, because it is the row's own median - is a step between neighbouring
       rows; vignetting, even a steep falloff at the slit ends, is continuous. Another material
       (a spectral step) is dropped wherever it is; brightness bands only at the ROI's ends, so a
       dust line or scratch across the slit keeps its rows (they need their own factors), and a
       gap between two picked targets is never a step. Rows dropped here are in ``rows_left_out``.

    Args:
        mask: Drawn ROI ``(lines, samples)``.
        brightness: :func:`brightness_map` of the cube.
        saturated: Optional saturated-pixel mask (:func:`saturation_map`).
        inset_px: Edge inset, pixels; ``0`` = none.
        row_tolerance: Relative tolerance of the row gate; ``None`` = no row gate and no row bands.
        step_tolerance: Relative brightness jump between neighbouring rows that is a step; default
            :data:`STEP_TOLERANCE` (6 %), independent of the row gate.
        seed_rows: Slit rows of spectral-pick seeds: their material is the one kept.
        index, cube, wavelengths: Spectra for the spectral gate (an index is faster).
        spectral: Run the spectral gate when spectra are given.
        spectral_threshold_deg: Its fixed limit; ``None`` = automatic (:func:`robust_threshold`).
    """
    drawn = np.asarray(mask, dtype=bool)
    b = np.asarray(brightness, dtype=np.float32)
    m = drawn & np.isfinite(b)
    sat_px = 0
    if saturated is not None:
        sat = np.asarray(saturated, dtype=bool)
        sat_px = int((m & sat).sum())
        m &= ~sat
    before = int(m.sum())
    if inset_px > 0 and before:
        m = ndi.binary_erosion(m, structure=_disk(inset_px), border_value=1)
    inset_removed = before - int(m.sum())
    spectral_removed, spectral_limit = 0, None
    if spectral and m.any() and (index is not None or (cube is not None and wavelengths is not None)):
        outliers, spectral_limit = spectral_outliers(m, index=index, cube=cube, wavelengths=wavelengths,
                                                     threshold_deg=spectral_threshold_deg)
        spectral_removed = int((m & outliers).sum())
        m &= ~outliers
    gate_removed = 0
    if row_tolerance is not None and m.any():
        for y in np.where(m.any(axis=1))[0]:
            row = m[y]
            med = float(np.median(b[y, row]))
            if med <= 0:
                gate_removed += int(row.sum())
                m[y] = False
                continue
            keep = np.abs(b[y] / med - 1.0) <= float(row_tolerance)
            gate_removed += int((row & ~keep).sum())
            m[y] = row & keep
    left_out: List[int] = []
    rows = np.where(m.any(axis=1))[0]
    if row_tolerance is not None and rows.size >= 2:
        levels = np.array([np.median(b[y, m[y]]) for y in rows])
        counts = m.sum(axis=1)[rows]
        spectra = None
        if spectral_limit is not None:
            spec_rows, meds = _row_medians(m, index, cube, wavelengths, per_row=256)
            if np.array_equal(spec_rows, rows):
                spectra = meds
        step = float(step_tolerance) if step_tolerance is not None else STEP_TOLERANCE
        for i in _band_rows(rows, levels, counts, spectra, step, seed_rows):
            y = int(rows[i])
            gate_removed += int(m[y].sum())
            m[y] = False
            left_out.append(y)
    return CleanupResult(m, int(drawn.sum()), sat_px, inset_removed, gate_removed, spectral_removed,
                         spectral_limit, tuple(sorted(left_out)))


# ── quality checks ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QcItem:
    """One quality check: ``level`` ``"ok"``, ``"warn"`` or ``"error"`` (blocks saving), and its text."""

    level: str
    text: str


@dataclass(frozen=True)
class RoiQuality:
    """QC of an ROI (:func:`assess_roi`): the items, and the numbers behind them (JSON-friendly)."""

    items: Tuple[QcItem, ...]
    numbers: dict

    @property
    def ok(self) -> bool:
        """True when no item is an error."""
        return not any(item.level == "error" for item in self.items)

    def errors(self) -> List[str]:
        """Texts of the error items."""
        return [item.text for item in self.items if item.level == "error"]


def assess_roi(cleanup: CleanupResult, cube, wavelengths, brightness, *, saturation_source: str = "unknown",
               ceiling=None, full_scale=None, min_px_per_row: int = 20, max_samples: int = 20000) -> RoiQuality:
    """Quality checks of a cleaned ROI.

    Errors (saving refused): no pixel left; more than half of the drawn pixels saturated.
    Warnings: saturation level unknown; more than 15 % of the drawn pixels removed by the spectral
    gate, row gate and row profile; slit-row coverage below 90 % (rows outside the ROI get
    the factors of the nearest ROI rows); fewer than ``min_px_per_row`` pixels in a covered row or
    fewer than 20 scan frames; peak signal below 30 % or above 90 % of the saturation level;
    median row CV of brightness above 2 %; 99th-percentile spectral angle to the ROI median
    spectrum above 1.5 degrees (another material or mixed pixels inside).

    Args:
        cleanup: :func:`clean_mask` result.
        cube, wavelengths, brightness: The scan and its :func:`brightness_map`.
        saturation_source: ``source`` of :func:`saturation_map`.
        ceiling, full_scale: The saturation level used (for the signal check), as for
            :func:`saturation_map`.
        min_px_per_row: Fewest pixels a covered slit row should have.
        max_samples: ROI pixels sampled for the spectral checks.
    """
    m = cleanup.mask
    h, w = m.shape
    items: List[QcItem] = []
    numbers = {"drawn_px": cleanup.drawn_px, "saturated_px": cleanup.saturated_px,
               "inset_removed_px": cleanup.inset_removed_px, "gate_removed_px": cleanup.gate_removed_px,
               "spectral_removed_px": cleanup.spectral_removed_px,
               "spectral_threshold_deg": cleanup.spectral_threshold_deg,
               "final_px": cleanup.final_px, "lines": h, "saturation_source": saturation_source}
    if cleanup.drawn_px == 0:
        return RoiQuality((QcItem("error", "No ROI drawn yet"),), numbers)
    sat_fraction = cleanup.saturated_px / max(cleanup.drawn_px, 1)
    if sat_fraction > 0.5:
        items.append(QcItem("error", f"{sat_fraction:.0%} of the ROI is saturated: re-scan with a shorter exposure"))
    elif cleanup.saturated_px:
        items.append(QcItem("warn", f"{cleanup.saturated_px} saturated pixels excluded"))
    if saturation_source == "unknown":
        items.append(QcItem("warn", "Saturation level unknown: saturated pixels cannot be excluded"))
    if cleanup.rows_left_out:
        rows_out = cleanup.rows_left_out
        numbers["rows_left_out"] = len(rows_out)
        items.append(QcItem("warn", f"{len(rows_out)} slit rows ({rows_out[0]}-{rows_out[-1]}) left out as a "
                                    "separate band (holder, bevel or label rows): they get the factors of the "
                                    "nearest ROI row"))
    gated = (cleanup.gate_removed_px + cleanup.spectral_removed_px) / max(cleanup.drawn_px, 1)
    numbers["gated_fraction"] = round(float(gated), 4)
    if gated > 0.15:
        items.append(QcItem("warn", f"{gated:.0%} of the ROI left out as unevenly lit or other material "
                                    "(red): check the panel for shadow, dirt or a label"))
    if cleanup.final_px == 0:
        items.append(QcItem("error", "No usable pixels left after the clean-up (inset, row gate)"))
        return RoiQuality(tuple(items), numbers)

    rows = np.where(m.any(axis=1))[0]
    per_row = m.sum(axis=1)[rows]
    cols = np.where(m.any(axis=0))[0]
    coverage = rows.size / h
    numbers.update(rows_covered=int(rows.size), span=[int(rows.min()), int(rows.max())],
                   min_px_per_row=int(per_row.min()), samples_covered=int(cols.size),
                   row_coverage=round(float(coverage), 4))
    text = f"ROI covers {rows.size}/{h} slit rows ({coverage:.0%}), rows {rows.min()}-{rows.max()}"
    if coverage < 0.9:
        items.append(QcItem("warn", text + ": rows outside get the factors of the nearest ROI rows"))
    else:
        items.append(QcItem("ok", text))
    if per_row.min() < min_px_per_row:
        items.append(QcItem("warn", f"Some slit rows have only {per_row.min()} ROI pixels (< {min_px_per_row}): noisy factors"))
    if cols.size < 20:
        items.append(QcItem("warn", f"Only {cols.size} scan frames in the ROI: scan the panel longer"))

    rr, cc = np.nonzero(m)
    if rr.size > max_samples:
        pick = np.linspace(0, rr.size - 1, max_samples).astype(int)
        rr, cc = rr[pick], cc[pick]
    spectra = np.asarray(cube[rr, cc, :], dtype=np.float64)
    mean_spec = np.nanmean(spectra, axis=0)
    peak_band = int(np.nanargmax(mean_spec)) if np.isfinite(mean_spec).any() else 0
    level = None
    if ceiling is not None and np.shape(ceiling) == (h, cube.shape[2]):
        level = float(np.nanmedian(np.asarray(ceiling, dtype=np.float64)[rows, peak_band]))
    elif full_scale:
        level = float(full_scale)
    if level and level > 0:
        p99 = float(np.nanpercentile(spectra[:, peak_band], 99)) / level
        numbers["peak_signal_fraction"] = round(p99, 4)
        if p99 < 0.30:
            items.append(QcItem("warn", f"Peak signal only {p99:.0%} of saturation: low SNR, raise the exposure"))
        elif p99 > 0.90:
            items.append(QcItem("warn", f"Peak signal {p99:.0%} of saturation: close to non-linearity"))
        else:
            items.append(QcItem("ok", f"Peak signal {p99:.0%} of saturation"))

    b = np.asarray(brightness, dtype=np.float64)
    cvs = []
    for y in rows:
        vals = b[y, m[y]]
        if vals.size >= 3 and np.mean(vals) > 0:
            cvs.append(np.std(vals) / np.mean(vals))
    if cvs:
        cv = float(np.median(cvs))
        numbers["row_cv"] = round(cv, 5)
        if cv > 0.02:
            items.append(QcItem("warn", f"Uneven along the scan: row CV {cv:.1%} (dust, shadow or tilt)"))
        else:
            items.append(QcItem("ok", f"Uniform along the scan: row CV {cv:.1%}"))

    sel = _band_selection(wavelengths)
    sub = spectra[:, sel]
    good = np.all(np.isfinite(sub), axis=1)
    if good.sum() >= 10:
        sub = sub[good]
        ref = np.median(sub, axis=0)
        norms = np.linalg.norm(sub, axis=1) * np.linalg.norm(ref)
        with np.errstate(divide="ignore", invalid="ignore"):
            ang = np.degrees(np.arccos(np.clip(sub @ ref / norms, -1, 1)))
        ang = ang[np.isfinite(ang)]
        if ang.size:
            p99a = float(np.percentile(ang, 99))
            numbers["angle_p99_deg"] = round(p99a, 3)
            if p99a > 1.5:
                items.append(QcItem("warn", f"Spectra not homogeneous: 99 % within {p99a:.1f} deg "
                                            "(another material or edge pixels inside?)"))
            else:
                items.append(QcItem("ok", f"Spectrally homogeneous: 99 % within {p99a:.2f} deg"))
    return RoiQuality(tuple(items), numbers)
