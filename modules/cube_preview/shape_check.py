"""Circle / square target check on the preview image, for stage-speed calibration.

Scan one or more round or square targets: when the stage speed matches the frame rate the
targets keep their shape; a wrong speed stretches them along the scan axis only.
``detect_shapes`` finds every target in the RGB preview and reports how round / square each
one is and its scan/slit stretch ``aspect`` (``> 1`` = too many frames over the target =
stage too slow, so multiply the speed by ``aspect``); ``summarize_aspects`` combines the
targets into one speed factor.  ``detect_shape`` returns only the best scoring target.

Data conventions:
    Input is a ``(slit rows, scan columns, 3)`` RGB image of any dtype (or a 2-D grey
    image); coordinates are image ``(x, y)`` = ``(scan column, slit row)`` with pixel
    centres at integers.  Prefer the image before any display percentile stretch: clipping
    fattens small targets evenly and pulls their W/H towards 1 (-0.6 % at r = 6 px).

Algorithm:
    1. Working grey: the luminance, R, G, B or the first principal component of the
       colours, whichever splits best into two classes (Otsu's eta), so targets that differ
       from the background only in colour are found too.
    2. Line defects: a slit row (dead / hot pixel) or scan column (dropped frame) is replaced
       by interpolation when it sits at another level than both neighbours along most of
       its length, or stays flat where the lines on both sides cross the same targets (a
       dead pixel at the background level, which would otherwise cut those targets).
    3. Background: a separable model, row level + column level (medians of background
       pixels), subtracted.  It removes vignetting along the slit, fixed-pattern stripes and
       illumination drift / flicker along the scan.  It is fitted from three starting sets
       of background pixels (all pixels; the Otsu class holding the border; pixels reached
       from the border without crossing an edge), the one leaving the narrowest spread
       around zero wins, then it is fitted again with the targets masked out.
    4. Candidates: the flattened image, smoothed (sigma 1 px), thresholded in both
       polarities at ``max(NOISE_K`` noise sigmas, ``MIN_CONTRAST_FRACTION`` of the strongest
       contrast), holes filled.  No size relative to the other targets is required, so
       small and dim targets are kept.  Targets joined only by their blurred edges are
       separated at the region's mid level; a region still not convex is split between the
       peaks of its distance transform by a power diagram (for touching or overlapping
       discs this is exactly the line through their contact / intersection points).
    5. Per target, in a window around it: target and background levels (medians of its
       core and of a ring around it), then the region above their mid level on the window
       upsampled (linear) to ~``UPSAMPLE_TARGET_PX`` across (on the smoothed image when the
       target is noisy, ``SMOOTH_BELOW_SNR``).  The mid level is the edge of a blurred step,
       also under (symmetric) motion smear along the scan.  Area, centre and second moments
       come from that pixel mask: sub-pixel, and without the ~0.5 px inward bias of a
       contour polygon through boundary pixel centres.  A disc split off a touching
       neighbour is measured by an ellipse fitted to its free edge.
    6. Kind, tested with the measured stretch undone (so a badly stretched target is still
       recognised): circle = round (circularity), not rectangular (fill of the min-area
       rectangle) and close to its moment-equivalent ellipse; square = rectangular, sides at
       most ``SQUARE_MAX_SIDE_RATIO`` apart.  Both must be convex.  Targets touching the
       image border are dropped, and so is a target inside a larger one of the same kind
       (the hole of a ring, a specular spot).  Squares under ~15 px a side blur into discs
       and cannot be told apart from them.

    * circularity = 4*pi*A / P**2 (1 = circle, pi/4 = square), capped at 1; the perimeter
      is taken on a lightly smoothed contour so pixel staircase does not lower the score.
      It is insensitive to a small stretch (a 1.1:1 ellipse still gives 0.994), so use
      ``aspect`` to tune the speed.
    * squareness = (short / long side of the min-area rectangle) * (A / rectangle area)
      (1 = square; a circle gives pi/4).
    * aspect = sqrt(mu20 / mu02) of the region (scan width / slit height).  For any shape
      with 90-degree symmetry (circle, square at any angle) it is exactly the stretch.

Threading:
    Pure functions, no shared state; call from any thread.

Example:
    >>> import numpy as np, cv2
    >>> img = np.zeros((120, 200, 3), np.uint8)
    >>> _ = cv2.circle(img, (50, 60), 30, (255, 255, 255), -1)
    >>> _ = cv2.ellipse(img, (140, 60), (33, 30), 0, 0, 360, (255, 255, 255), -1)
    >>> [round(r.aspect, 2) for r in detect_shapes(img, "circle")]
    [1.0, 1.1]
    >>> r = detect_shape(img, "circle")
    >>> round(r.score, 2) >= 0.97 and abs(r.aspect - 1) < 0.02
    True
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SHAPES = ("circle", "square")
MIN_AREA_PX = 16               # smaller regions are dust / noise (a disc of r ~ 2.3 px)
NOISE_K = 5.0                  # candidate threshold in noise sigmas of the smoothed image ...
MIN_CONTRAST_FRACTION = 0.1    # ... and at least this fraction of the strongest contrast
MIN_CONTRAST_SIGMA = 4.0       # a target's own contrast must exceed this many noise sigmas
SMOOTH_BELOW_SNR = 10.0        # below this contrast / raw noise a target is measured on the smoothed image
EDGE_NOISE_K = 5.0             # gradient above this many times its median is a target edge
LINE_DEFECT_K = 6.0            # a row / column this many typical deviations off its neighbours is a defect
MAX_DEFECT_FRACTION = 0.1      # more "defects" than this = scene texture, repair nothing
FLAT_LINE_RATIO = 0.25         # a row varying < 1/4 as much as the rows on both sides is dead ...
FLAT_LINE_MIN_SIGMA = 8.0      # ... when those rows vary by more than this many noise sigmas
FLAT_LINE_MIN_CORR = 0.9       # ... and look alike (correlation), i.e. it cuts through the same targets
SEPARATE_MIN_KEPT = 0.6        # mid-level parts must keep this much of the region (else a dim one vanished)
SPLIT_SOLIDITY = 0.93          # a region less convex than this may be touching targets
SPLIT_MIN_AREA_PX = 150        # pixelised small discs are not convex enough to judge
SADDLE_RATIO = 0.7             # distance-transform peaks joined by a ridge this high are one target
UPSAMPLE_TARGET_PX = 160       # measure each target on a window upsampled to about this size ...
MAX_UPSAMPLE = 8               # ... by at most this factor
CIRCLE_MIN_CIRCULARITY = 0.85  # de-stretched target at least this round ...
CIRCLE_MAX_FILL = 0.87         # ... and filling less of its min-area rectangle (disc pi/4, square 1)
SQUARE_MIN_FILL = 0.88         # rectangular: fills its min-area rectangle (rounded corners ~0.95)
SQUARE_MAX_SIDE_RATIO = 2.0    # longer / shorter side; thin bars are not targets
MIN_SOLIDITY = 0.9             # area / convex-hull area of a target
MAX_ELLIPSE_RMS = 0.04         # a disc's edge stays within 4 % of its (fitted / moment) ellipse ...
ELLIPSE_RMS_PX = 0.3           # ... or 0.3 px for small discs (pixel-scale roughness)
OUTLIER_MIN_REL = 0.015        # a target 1.5 % off the median W/H is clearly not the same stretch ...
OUTLIER_MAD_K = 3.5            # ... when it is also this many robust sigmas off (noisy sets keep all)
_SMOOTH_SIGMA_PX = 1.5
_MAD_TO_SIGMA = 1.4826
_MEDIAN_SAMPLES = 400          # values per row / column for background and defect medians


@dataclass(frozen=True)
class ShapeResult:
    """One detected target.

    Attributes:
        kind: ``"circle"`` or ``"square"`` (what was asked for).
        score: circularity (circle) or squareness (square), 1.0 = perfect.
        circularity: 4*pi*A / P**2 of the target.
        squareness: (short / long side) * fill of the min-area rectangle.
        aspect: scan width / slit height from second moments (1.0 = no stretch).
        width, height: axis-aligned bounding box in pixels (scan, slit).
        center: ``(x, y)`` centroid in image pixels.
        area: region area in pixels.
        outline: ``(N, 2)`` float polygon of the moment-equivalent ellipse (circle) or the
            min-area rectangle (square), image coordinates.
        touching: the target touched another one and was split off it (circles: measured
            by an ellipse fit to the free edge; less certain than a free-standing target).
    """

    kind: str
    score: float
    circularity: float
    squareness: float
    aspect: float
    width: int
    height: int
    center: tuple[float, float]
    area: float
    outline: np.ndarray
    touching: bool = False


@dataclass(frozen=True)
class _Region:
    mask: np.ndarray          # bool crop of the image whose top-left pixel is (x0, y0)
    y0: int
    x0: int
    sign: float               # +1 bright target, -1 dark target
    touching: bool


# ── public API ──────────────────────────────────────────────────────────────


def detect_shapes(rgb8: np.ndarray, kind: str) -> list[ShapeResult]:
    """Find every circle-like or square-like target in an RGB preview image.

    Args:
        rgb8: ``(H, W, 3)`` uint8 RGB image (or ``(H, W)`` grey; other dtypes are accepted).
        kind: ``"circle"`` or ``"square"``.

    Returns:
        The targets of that kind in reading order (top to bottom, then left to right);
        empty when there are none.

    Raises:
        ValueError: ``kind`` is not one of ``SHAPES``.
    """
    if kind not in SHAPES:
        raise ValueError(f"kind must be one of {SHAPES}, got {kind!r}")
    img = np.asarray(rgb8)
    if img.ndim != 2 and (img.ndim != 3 or img.shape[2] < 3):
        return []
    if img.shape[0] < 3 or img.shape[1] < 3:
        return []
    gray = _working_grey(img)
    if not np.isfinite(gray).all() or float(gray.max() - gray.min()) <= 0:
        return []
    gray = _repair_lines(_repair_lines(gray).T).T
    flat, smooth, sigma, sigma_smooth = _flatten(gray)
    regions, thr = _candidate_regions(smooth, sigma_smooth)
    if not regions:
        return []
    occupied = np.zeros(flat.shape, bool)
    for reg in regions:
        h, w = reg.mask.shape
        occupied[reg.y0:reg.y0 + h, reg.x0:reg.x0 + w] |= reg.mask
    measured = []
    for reg in regions:
        m = _measure_region(flat, smooth, reg, occupied, sigma, sigma_smooth, thr)
        if m is not None:
            measured.append(m)
    results = [r for r, cls in measured if cls == kind]
    return _reading_order(_drop_nested(results))


def detect_shape(rgb8: np.ndarray, kind: str) -> ShapeResult | None:
    """Find the most circle-like or square-like target in an RGB preview image.

    Args:
        rgb8: ``(H, W, 3)`` uint8 RGB image (or ``(H, W)`` grey).
        kind: ``"circle"`` or ``"square"``.

    Returns:
        The best scoring :class:`ShapeResult` of :func:`detect_shapes`, or ``None``.

    Raises:
        ValueError: ``kind`` is not one of ``SHAPES``.
    """
    return max(detect_shapes(rgb8, kind), key=lambda r: r.score, default=None)


@dataclass(frozen=True)
class AspectSummary:
    """W/H over several targets (the stage-speed factor of a multi-target check).

    Attributes:
        count: Number of targets used.
        aspect: Mean W/H of the used targets (new speed = speed x aspect).
        std: Sample standard deviation of the used W/H (0.0 for one target).
        min, max: Range of the used W/H.
        outliers: Indices (into the results) left out: W/H far from the median of all.
    """

    count: int
    aspect: float
    std: float
    min: float
    max: float
    outliers: tuple[int, ...]


def summarize_aspects(results) -> AspectSummary | None:
    """Combine the W/H of several targets, leaving out the ones far from the rest.

    A target is left out (with three or more targets) when its W/H differs from the median
    by more than ``OUTLIER_MIN_REL`` and by more than ``OUTLIER_MAD_K`` robust standard
    deviations (1.4826 x median absolute deviation), unless that would drop half of them.

    Args:
        results: Sequence of :class:`ShapeResult` (anything with an ``aspect``).

    Returns:
        The :class:`AspectSummary`, or ``None`` for no results.
    """
    aspects = np.array([r.aspect for r in results], dtype=np.float64)
    if aspects.size == 0:
        return None
    outliers: tuple[int, ...] = ()
    if aspects.size >= 3:
        rel = aspects / np.median(aspects) - 1.0
        sigma = _MAD_TO_SIGMA * float(np.median(np.abs(rel)))
        bad = np.abs(rel) > max(OUTLIER_MIN_REL, OUTLIER_MAD_K * sigma)
        if 0 < bad.sum() < aspects.size / 2:
            outliers = tuple(int(i) for i in np.flatnonzero(bad))
    used = np.delete(aspects, outliers)
    return AspectSummary(
        count=int(used.size),
        aspect=float(used.mean()),
        std=float(used.std(ddof=1)) if used.size > 1 else 0.0,
        min=float(used.min()),
        max=float(used.max()),
        outliers=outliers,
    )


# ── 1-3: working grey, line defects, background ─────────────────────────────


def _otsu_eta(values: np.ndarray) -> float:
    # Otsu's separability: best between-class variance / total variance (0..1)
    lo, hi = np.percentile(values, (0.5, 99.5))
    if hi <= lo:
        return 0.0
    hist = np.bincount(np.clip((values - lo) / (hi - lo) * 255, 0, 255).astype(np.intp),
                       minlength=256).astype(np.float64)
    p = hist / hist.sum()
    levels = np.arange(256, dtype=np.float64)
    w0 = np.cumsum(p)
    m0 = np.cumsum(p * levels)
    mean = m0[-1]
    total = float(np.sum(p * (levels - mean) ** 2))
    if total <= 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mean * w0 - m0) ** 2 / (w0 * (1 - w0))
    return float(np.nanmax(between[:-1]) / total)


def _working_grey(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    rgb = img[:, :, :3].astype(np.float32)
    options = [rgb @ np.float32([0.299, 0.587, 0.114]), rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]
    step = max(1, int(np.sqrt(rgb.shape[0] * rgb.shape[1] / 40_000)))
    sample = rgb[::step, ::step].reshape(-1, 3)
    if len(sample) > 3:
        evals, evecs = np.linalg.eigh(np.cov(sample.T.astype(np.float64)))
        if evals[-1] > 0:
            options.append(rgb @ evecs[:, -1].astype(np.float32))
    etas = [_otsu_eta(o[::step, ::step].ravel()) for o in options]
    return np.ascontiguousarray(options[int(np.argmax(etas))])


def _repair_lines(g: np.ndarray) -> np.ndarray:
    """Replace rows that are a sensor defect (dead / hot slit pixel) by interpolation.

    A defect row either sits at another level than both neighbours along most of its
    length, or stays flat where the rows on both sides of it cross targets (a dead pixel
    at the background level, which would otherwise cut the targets it crosses).
    """
    h = g.shape[0]
    if h < 5 or g.shape[1] < 5:
        return g
    s = g[:, ::max(1, g.shape[1] // _MEDIAN_SAMPLES)]
    pred = np.empty_like(s)
    pred[1:-1] = 0.5 * (s[:-2] + s[2:])
    pred[0], pred[-1] = s[1], s[-2]
    dev = np.median(np.abs(s - pred), axis=1)
    span = float(np.percentile(s, 99.9) - np.percentile(s, 0.1))
    limit = LINE_DEFECT_K * max(float(np.median(dev)), 1e-3 * span)
    peak = np.maximum(np.r_[dev[1:], 0.0], np.r_[0.0, dev[:-1]])
    offset = (dev > limit) & (dev >= 0.75 * peak)    # the defect itself, not its neighbours
    lo, hi = np.percentile(s, (1, 99), axis=1)
    spread = hi - lo
    noise = _robust_sigma(s[1:] - s[:-1]) / np.sqrt(2.0)
    pad = np.r_[0.0, 0.0, spread, 0.0, 0.0]
    above = np.maximum(pad[1:-3], pad[:-4])          # rows y-1, y-2 (two dead rows side by side)
    below = np.maximum(pad[3:-1], pad[4:])           # rows y+1, y+2
    flat = (spread < FLAT_LINE_RATIO * np.minimum(above, below)) & \
        (np.minimum(above, below) > FLAT_LINE_MIN_SIGMA * max(noise, 1e-3 * span))
    for y in np.flatnonzero(flat):
        # the rows on both sides must show the same structure (a target the dead row cuts
        # through), not two different targets with a background gap between them
        # nearest row on each side, skipping one that is itself flat (two dead rows side by side)
        ya = y - 2 if y >= 2 and spread[y - 1] < FLAT_LINE_RATIO * spread[y - 2] else y - 1
        yb = y + 2 if y <= h - 3 and spread[y + 1] < FLAT_LINE_RATIO * spread[y + 2] else y + 1
        if not (0 <= ya < h and 0 <= yb < h) or np.corrcoef(s[ya], s[yb])[0, 1] < FLAT_LINE_MIN_CORR:
            flat[y] = False
    bad = offset | flat
    if not bad.any() or bad.mean() > MAX_DEFECT_FRACTION:
        return g
    good = np.flatnonzero(~bad)
    rows = np.flatnonzero(bad)
    i = np.clip(np.searchsorted(good, rows), 1, len(good) - 1)
    lo, hi = good[i - 1], good[i]
    t = np.clip((rows - lo) / np.maximum(hi - lo, 1), 0.0, 1.0)[:, None].astype(np.float32)
    out = g.copy()
    out[rows] = (1 - t) * g[lo] + t * g[hi]
    return out


def _masked_median(a: np.ndarray, keep: np.ndarray, axis: int) -> np.ndarray:
    """Median along ``axis`` of the kept values; lines with none are interpolated.

    Uses at most ``_MEDIAN_SAMPLES`` evenly spaced values per line (speed).
    """
    if axis == 0:
        a, keep = a.T, keep.T
    step = max(1, a.shape[1] // _MEDIAN_SAMPLES)
    vals = np.where(keep[:, ::step], a[:, ::step], np.inf)
    vals.sort(axis=1)                          # excluded (inf) values go to the end
    count = keep[:, ::step].sum(axis=1)
    rows = np.arange(vals.shape[0])
    lo = np.clip((count - 1) // 2, 0, None)
    hi = np.clip(count // 2, 0, None)
    med = 0.5 * (vals[rows, lo] + vals[rows, hi])
    ok = count > 0
    if not ok.any():
        return np.zeros(vals.shape[0], np.float32)
    if not ok.all():
        idx = np.arange(med.size)
        med = np.interp(idx, idx[ok], med[ok])
    return med.astype(np.float32)


def _robust_sigma(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    values = values[::max(1, values.size // 100_000)]
    return _MAD_TO_SIGMA * float(np.median(np.abs(values - np.median(values))))


def _otsu_background(g: np.ndarray) -> np.ndarray:
    # the Otsu class holding most of the border pixels is the background
    lo, hi = float(g.min()), float(g.max())
    g8 = ((g - lo) / max(hi - lo, 1e-12) * 255).astype(np.uint8)
    _, high = cv2.threshold(g8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high = high.astype(bool)
    border = np.r_[high[0], high[-1], high[:, 0], high[:, -1]]
    return high if border.mean() > 0.5 else ~high


def _border_background(g: np.ndarray) -> np.ndarray:
    # pixels reachable from the image border without crossing an edge: smooth vignetting and
    # drift are crossed, target edges are not, however much of a row / column targets cover
    s = cv2.GaussianBlur(g, (0, 0), 1.5)
    mag = np.hypot(cv2.Sobel(s, cv2.CV_32F, 1, 0), cv2.Sobel(s, cv2.CV_32F, 0, 1))
    sample = mag[::2, ::2]
    calm = (mag < max(EDGE_NOISE_K * float(np.median(sample)), 0.05 * float(np.percentile(sample, 99.9)))).astype(np.uint8)
    _, labels = cv2.connectedComponents(calm, connectivity=4)
    border = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    keep = np.isin(labels, border[border > 0]).astype(np.uint8)
    keep = cv2.erode(keep, np.ones((3, 3), np.uint8), iterations=2).astype(bool)
    return keep if keep.sum() >= 0.05 * keep.size else np.ones(g.shape, bool)


def _border_mean_abs(a: np.ndarray) -> float:
    return float(np.mean(np.abs(np.r_[a[:2].ravel(), a[-2:].ravel(), a[:, :2].ravel(), a[:, -2:].ravel()])))


def _fit_background(g: np.ndarray, keep: np.ndarray):
    """Row + column background from the kept pixels: ``(flat, smooth, sigma, sigma_smooth)``."""
    rows = _masked_median(g, keep, axis=1)
    cols = _masked_median(g - rows[:, None], keep, axis=0)
    flat = g - rows[:, None] - cols[None, :]
    smooth = cv2.GaussianBlur(flat, (0, 0), 1.0)
    both = keep[1:, :] & keep[:-1, :]
    if both.sum() < 100:
        both = np.ones_like(both)
    sigma = _robust_sigma((flat[1:, :] - flat[:-1, :])[both]) / np.sqrt(2.0)
    sigma_smooth = _robust_sigma(smooth[keep] if keep.sum() >= 100 else smooth.ravel())
    return flat, smooth, sigma, sigma_smooth


def _flatten(g: np.ndarray):
    """Subtract a row + column background; return flat, smoothed flat, raw and smoothed noise.

    Two starting sets of background pixels: all pixels (a median ignores targets covering
    less than half of a row / column, but not dense rows of targets) and the Otsu class
    holding the border (dense targets, but strong vignetting splits the background itself).
    The fit with the narrower spread around zero over all pixels wins (a wrong background
    widens it wherever it is wrong); it is then fitted again with the candidate targets
    (dilated) masked out.
    """
    starts = [np.ones(g.shape, bool), _otsu_background(g), _border_background(g)]
    fits = [_fit_background(g, keep) for keep in starts]
    # plus the mean |residual| on the image border (background there): it separates the
    # fits when all spreads are zero (noise-free images) and adds little to good fits
    spreads = [_robust_sigma(f[1][::2, ::2].ravel()) + 0.1 * _border_mean_abs(f[1]) for f in fits]
    best = int(np.argmin(spreads))
    flat, smooth, sigma, sigma_smooth = fits[best]
    objects = np.abs(smooth) > _threshold(smooth, sigma_smooth)
    objects = cv2.dilate(objects.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if (~objects).sum() >= 0.1 * objects.size:      # else nearly all "target": keep this fit
        flat, smooth, sigma, sigma_smooth = _fit_background(g, ~objects)
    return flat, smooth, sigma, sigma_smooth


def _threshold(smooth: np.ndarray, sigma_smooth: float) -> float:
    sample = np.abs(smooth[::2, ::2])
    strongest = float(np.percentile(sample, 99.9))
    return max(NOISE_K * sigma_smooth, MIN_CONTRAST_FRACTION * strongest, 1e-6)


# ── 4: candidate regions ────────────────────────────────────────────────────


def _fill_holes(mask8: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask8)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


def _candidate_regions(smooth: np.ndarray, sigma_smooth: float) -> tuple[list[_Region], float]:
    thr = _threshold(smooth, sigma_smooth)
    h, w = smooth.shape
    regions = []
    for sign in (1.0, -1.0):
        mask = _fill_holes((sign * smooth > thr).astype(np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < MIN_AREA_PX:
                continue
            if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
                continue                     # touches the border: cut / still being scanned
            comp = labels[y:y + bh, x:x + bw] == i
            for sub in _separate(comp, sign * smooth[y:y + bh, x:x + bw]):
                for part, touching in _split(sub):
                    if part.sum() >= MIN_AREA_PX:
                        regions.append(_Region(part, y, x, sign, touching))
    return regions, thr


def _separate(comp: np.ndarray, v: np.ndarray) -> list[np.ndarray]:
    """Targets joined only by their blurred edges come apart at the region's mid level."""
    level = 0.5 * float(np.percentile(v[comp], 90))
    sub = _fill_holes((comp & (v >= level)).astype(np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    parts = [labels == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA_PX]
    # a dim target would vanish at a bright neighbour's mid level: then keep the region whole
    if len(parts) < 2 or sum(int(p.sum()) for p in parts) < SEPARATE_MIN_KEPT * comp.sum():
        return [comp]
    return parts


def _split(comp: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """Split a region of touching convex targets at its distance-transform peaks."""
    area = int(comp.sum())
    if area < SPLIT_MIN_AREA_PX:
        return [(comp, False)]
    contours, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.contourArea(cv2.convexHull(max(contours, key=cv2.contourArea)))
    if hull <= 0 or area / hull >= SPLIT_SOLIDITY:
        return [(comp, False)]
    padded = np.pad(comp.astype(np.uint8), 1)
    dt = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    top = float(dt.max())
    size = max(3, int(0.3 * top) | 1)
    peaks = (dt >= cv2.dilate(dt, np.ones((size, size), np.uint8))) & (dt >= max(2.0, 0.25 * top))
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-dt[ys, xs])
    seeds: list[tuple[float, float, float]] = []
    for k in order:
        y, x, d = float(ys[k]), float(xs[k]), float(dt[ys[k], xs[k]])
        if not any(_ridge(dt, (y, x), (sy, sx)) >= SADDLE_RATIO * min(d, sd) for sy, sx, sd in seeds):
            seeds.append((y, x, d))
    if len(seeds) < 2:
        return [(comp, False)]
    # power diagram: |p - c|^2 - r^2; for touching / overlapping discs the cell border is
    # the line through their contact / intersection points
    yy, xx = np.nonzero(comp)
    power = np.stack([(yy - sy) ** 2 + (xx - sx) ** 2 - sd ** 2 for sy, sx, sd in seeds])
    owner = np.argmin(power, axis=0)
    parts = []
    for i in range(len(seeds)):
        part = np.zeros_like(comp)
        part[yy[owner == i], xx[owner == i]] = True
        parts.append((part, True))
    return parts


def _ridge(dt: np.ndarray, a, b) -> float:
    # lowest distance-transform value on the segment a-b
    n = int(np.hypot(a[0] - b[0], a[1] - b[1])) + 2
    ys = np.rint(np.linspace(a[0], b[0], n)).astype(np.intp)
    xs = np.rint(np.linspace(a[1], b[1], n)).astype(np.intp)
    return float(dt[ys, xs].min())


# ── 5-6: per-target measurement and kind ────────────────────────────────────


def _measure_region(flat, smooth, reg: _Region, occupied, sigma, sigma_smooth, thr):
    """Measure one candidate; return ``(ShapeResult, kind or None)`` or ``None``."""
    H, W = flat.shape
    mh, mw = reg.mask.shape
    size = max(1.0, float(min(mh, mw)))
    margin = int(max(4, np.ceil(0.3 * size)))
    y0, y1 = max(0, reg.y0 - margin), min(H, reg.y0 + mh + margin)
    x0, x1 = max(0, reg.x0 - margin), min(W, reg.x0 + mw + margin)
    own = np.zeros((y1 - y0, x1 - x0), np.uint8)
    own[reg.y0 - y0:reg.y0 - y0 + mh, reg.x0 - x0:reg.x0 - x0 + mw] = reg.mask
    others = occupied[y0:y1, x0:x1] & ~own.astype(bool)
    v = reg.sign * flat[y0:y1, x0:x1]

    # target level: median of its core (holes of a ring and specular spots left out);
    # background level: median of a ring around it, away from other targets
    core = (own > 0) & (v > thr)
    dt = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 3)
    interior = dt >= max(1.5, 0.3 * float(dt.max()))
    if not interior.any():
        interior = core if core.any() else own.astype(bool)
    ring_in, ring_out = 3, 3 + max(3, int(0.15 * size))
    near_others = cv2.dilate(others.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    ring = (_dilate(own, ring_out) & ~_dilate(own, ring_in)) & ~near_others
    fg = float(np.median(v[interior]))
    bg = float(np.median(v[ring])) if ring.sum() >= 8 else 0.0
    contrast = fg - bg
    if contrast <= 0:
        return None
    if contrast < SMOOTH_BELOW_SNR * sigma:
        # noisy: the raw mid-level edge would jitter, measure on the smoothed image instead
        v = reg.sign * smooth[y0:y1, x0:x1]
        if contrast <= MIN_CONTRAST_SIGMA * sigma_smooth:
            return None
    level = bg + 0.5 * contrast

    # the target at that level, on the window upsampled (linear) for sub-pixel moments
    territory = _dilate(own, ring_in + 1) & ~cv2.dilate(others.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    up = int(np.clip(np.ceil(UPSAMPLE_TARGET_PX / size), 1, MAX_UPSAMPLE))
    if reg.touching:
        up = max(up, 4)
    hu, wu = v.shape[0] * up, v.shape[1] * up
    v_up = cv2.resize(v, (wu, hu), interpolation=cv2.INTER_LINEAR) if up > 1 else v
    terr_up = cv2.resize(territory.astype(np.uint8), (wu, hu), interpolation=cv2.INTER_NEAREST) if up > 1 \
        else territory.astype(np.uint8)
    mask = ((v_up >= level) & (terr_up > 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = _fill_holes((labels == best).astype(np.uint8))
    bx, by, bw, bh, _ = stats[best]
    # cut by the image border (the window reaches it there)
    if (y0 == 0 and by == 0) or (x0 == 0 and bx == 0) or (y1 == H and by + bh >= hu) or (x1 == W and bx + bw >= wu):
        return None

    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] <= 0 or m["mu02"] <= 0 or m["mu20"] <= 0:
        return None
    scale = 1.0 / up
    area = m["m00"] * scale * scale
    if area < MIN_AREA_PX:
        return None
    center = ((m["m10"] / m["m00"] + 0.5) * scale - 0.5 + x0, (m["m01"] / m["m00"] + 0.5) * scale - 0.5 + y0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=len).reshape(-1, 2).astype(np.float64)
    points = (contour + 0.5) * scale - 0.5 + np.array([x0, y0], np.float64)
    cov = np.array([[m["mu20"], m["mu11"]], [m["mu11"], m["mu02"]]]) / m["m00"] * scale * scale

    # a disc cut off a touching neighbour: the ellipse through its free edge
    fitted = _free_edge_ellipse(points, others, x0, y0, scale) if reg.touching else None
    if fitted is not None:
        center, cov = fitted
        area = float(4 * np.pi * np.sqrt(max(np.linalg.det(cov), 0.0)))
        kind = "circle"
    aspect = float(np.sqrt(cov[0, 0] / cov[1, 1]))
    if fitted is None:
        kind = _classify(points, area, aspect, center, cov, scale)

    rect = cv2.minAreaRect(points.astype(np.float32))
    rw, rh = rect[1][0] + scale, rect[1][1] + scale    # contour runs half a (sub)pixel inside
    squareness = (min(rw, rh) / max(rw, rh)) * min(1.0, area / (rw * rh)) if rw > 0 and rh > 0 else 0.0
    circularity = _ellipse_circularity(cov) if fitted is not None else _circularity(points, area, scale)
    if kind == "circle":
        outline, score = _ellipse_outline(center, cov), circularity
    else:
        outline, score = cv2.boxPoints((rect[0], (rw, rh), rect[2])), squareness
    result = ShapeResult(
        kind=kind or "",
        score=float(score),
        circularity=float(circularity),
        squareness=float(squareness),
        aspect=aspect,
        width=int(round(float(points[:, 0].max() - points[:, 0].min()) + scale)),
        height=int(round(float(points[:, 1].max() - points[:, 1].min()) + scale)),
        center=(float(center[0]), float(center[1])),
        area=float(area),
        outline=np.asarray(outline, dtype=np.float64).reshape(-1, 2),
        touching=reg.touching,
    )
    return result, kind


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def _free_edge_ellipse(points, others, x0, y0, scale):
    """Ellipse ``(center, cov)`` through the edge not facing a touching neighbour, or ``None``.

    ``None`` too when that edge is not elliptic (``MAX_ELLIPSE_RMS``): not a disc.
    """
    near = cv2.dilate(others.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    ix = np.clip(np.rint(points[:, 0] - x0).astype(np.intp), 0, near.shape[1] - 1)
    iy = np.clip(np.rint(points[:, 1] - y0).astype(np.intp), 0, near.shape[0] - 1)
    free = points[~near[iy, ix]]
    if len(free) < 0.4 * len(points) or len(free) < 10:
        return None
    (ex, ey), (ea, eb), angle = cv2.fitEllipse(free.astype(np.float32))
    # contour points are (sub)pixel centres just inside the edge: grow by half a step
    a, b = ea / 2 + 0.5 * scale, eb / 2 + 0.5 * scale
    if a <= 0 or b <= 0:
        return None
    t = np.radians(angle)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    u = (free - [ex, ey]) @ rot
    rms = float(np.sqrt(np.mean((np.sqrt((u[:, 0] / a) ** 2 + (u[:, 1] / b) ** 2) - 1) ** 2)))
    if rms > MAX_ELLIPSE_RMS:
        return None
    return (float(ex), float(ey)), rot @ np.diag([a * a / 4, b * b / 4]) @ rot.T


def _classify(points, area, aspect, center, cov, scale):
    """``"circle"``, ``"square"`` or ``None``, judged with the stretch undone."""
    hull = cv2.convexHull(points.astype(np.float32))
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0 or area / (hull_area + scale * cv2.arcLength(hull, True) / 2) < MIN_SOLIDITY:
        return None
    flat_pts = points.copy()
    flat_pts[:, 0] = center[0] + (points[:, 0] - center[0]) / aspect
    flat_area = area / aspect
    circ = _circularity(flat_pts, flat_area, scale)
    fill, ratio = _rect_fill(flat_pts, flat_area, scale)
    fill_raw, ratio_raw = _rect_fill(points, area, scale)
    if fill < CIRCLE_MAX_FILL and fill_raw < CIRCLE_MAX_FILL and circ >= CIRCLE_MIN_CIRCULARITY:
        # the edge must follow the moment-equivalent ellipse (rejects small blurred triangles / blobs)
        unstretch = np.diag([1.0 / aspect, 1.0])
        d = flat_pts - np.asarray(center)
        rho = np.sqrt(np.einsum("ij,jk,ik->i", d, np.linalg.inv(unstretch @ cov @ unstretch), d)) / 2.0
        radius = np.sqrt(flat_area / np.pi)
        if float(np.std(rho)) <= max(MAX_ELLIPSE_RMS, ELLIPSE_RMS_PX / radius):
            return "circle"
        return None
    if max(fill, fill_raw) >= SQUARE_MIN_FILL and min(ratio, ratio_raw) <= SQUARE_MAX_SIDE_RATIO:
        return "square"
    return None


def _rect_fill(points, area, scale):
    rect = cv2.minAreaRect(points.astype(np.float32))
    rw, rh = rect[1][0] + scale, rect[1][1] + scale
    if rw <= 0 or rh <= 0:
        return 0.0, np.inf
    return min(1.0, area / (rw * rh)), max(rw, rh) / min(rw, rh)


def _smoothed_perimeter(points: np.ndarray, sigma: float = _SMOOTH_SIGMA_PX) -> float:
    n = len(points)
    if n < 8:
        return float(cv2.arcLength(points.reshape(-1, 1, 2).astype(np.float32), True))
    radius = min(int(np.ceil(3 * sigma)), n // 2 - 1)
    taps = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (taps / sigma) ** 2)
    kernel /= kernel.sum()
    pts = points.astype(np.float64)
    padded = np.concatenate([pts[-radius:], pts, pts[:radius]])
    smooth = np.column_stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(2)])
    return float(np.sum(np.hypot(*(np.roll(smooth, -1, axis=0) - smooth).T)))


def _circularity(points: np.ndarray, area: float, scale: float) -> float:
    # smoothing over ~1.5 image pixels whatever the upsampling (points are in image pixels)
    perimeter = _smoothed_perimeter(points, _SMOOTH_SIGMA_PX / scale)
    return min(1.0, 4.0 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0


def _ellipse_circularity(cov: np.ndarray) -> float:
    evals = np.clip(np.linalg.eigvalsh(cov), 0, None)
    a, b = 2 * np.sqrt(evals[1]), 2 * np.sqrt(evals[0])
    if a <= 0:
        return 0.0
    h = ((a - b) / (a + b)) ** 2
    perimeter = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))   # Ramanujan
    return float(min(1.0, 4 * np.pi * np.pi * a * b / perimeter ** 2))


def _ellipse_outline(center, cov: np.ndarray, n: int = 120) -> np.ndarray:
    # the ellipse with the target's area and second moments (uniform ellipse: cov = R diag(a^2, b^2) R^T / 4)
    evals, evecs = np.linalg.eigh(cov)
    radii = 2 * np.sqrt(np.clip(evals, 0, None))
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    unit = np.column_stack([radii[0] * np.cos(t), radii[1] * np.sin(t)])
    return unit @ evecs.T + np.asarray(center, np.float64)


def _drop_nested(results: list[ShapeResult]) -> list[ShapeResult]:
    # a target whose centre lies inside a larger one (hole of a ring, specular spot) is not a target
    keep = []
    for r in results:
        inside = any(o is not r and o.area > r.area and
                     cv2.pointPolygonTest(o.outline.astype(np.float32), r.center, False) >= 0
                     for o in results)
        if not inside:
            keep.append(r)
    return keep


def _reading_order(results: list[ShapeResult]) -> list[ShapeResult]:
    # a new row starts when a centre is more than half a typical target height below the row's first
    if not results:
        return []
    by_y = sorted(results, key=lambda r: r.center[1])
    step = 0.5 * float(np.median([r.height for r in results]))
    rows, row_y = [[by_y[0]]], by_y[0].center[1]
    for r in by_y[1:]:
        if r.center[1] - row_y > step:
            rows.append([])
            row_y = r.center[1]
        rows[-1].append(r)
    return [r for row in rows for r in sorted(row, key=lambda r: r.center[0])]
