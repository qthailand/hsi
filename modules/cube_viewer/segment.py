"""Supervised segmentation of the cube from the viewer's ROI classes (numpy only, no Qt, no sklearn).

Pipeline:
    1. :func:`build_classes` pools the ready spectrum records by ``class_name`` into
       :class:`ClassSpec` objects: the exact mean over the union of the class's ROI pixels
       (``roi_stats`` on the union mask over the combined bounding box), and a fixed-seed
       sample of at most ``max_per_class`` of those pixels for covariances, thresholds and
       the consistency checks.  Pixels claimed by two classes are excluded from both.
    2. :func:`fit` builds a model for one method of :data:`METHODS`.
    3. :func:`classify_cube` gives every pixel its best class and that class's score.
    4. :func:`apply_thresholds` rejects pixels whose score is worse than their class's
       threshold (``-1`` = unclassified) from the stored maps, without recomputing.
    :func:`separability` tells how different the classes are (angle matrix, JM distance,
    per-band F, ROI-consistency confusion, warnings).

Data conventions:
    The cube is ``(H, W, B)`` = ``(slit, scan, wavelength)``, any real numeric dtype - an
    ndarray, a ``numpy.memmap`` or a non-contiguous view (the transposed memmap of a BSQ /
    BIL file); it is only read, row chunk by row chunk, and never copied whole.  Pixel
    ``(c, r)`` is ``cube[r, c, :]``.  Records are
    :class:`~modules.cube_viewer.spectra.SpectrumRecord` (or any object with ``rid``,
    ``class_name``, ``kind``, ``geometry``, ``color``, ``state`` and ``stats``).
    Labels are int16 class indices into the class list (``-1`` = unclassified); scores are
    float32 and NaN where a pixel cannot be scored.

Methods (``METHODS``; score kind in brackets, lower is better unless 'probability'):
    * ``sam`` - spectral angle to each class mean [angle_deg].  Ignores brightness, but an
      additive (dark) offset changes the angle.
    * ``scm`` - spectral correlation mapper: the angle ``arccos(r)`` of the Pearson
      correlation with each class mean (SAM on mean-centred spectra) [angle_deg].  Ignores
      brightness and an additive offset; a flat pixel spectrum cannot be scored.
    * ``euclidean`` - distance to each class mean in cube units [distance].
    * ``euclidean_norm`` - distance after dividing every spectrum by its mean level
      (``y = x / mean(x)``) [distance, relative units].  A pixel with mean level <= 0
      cannot be scored.
    * ``sid`` - spectral information divergence ``sum((p - q) (log p - log q))`` of the
      spectra as distributions ``p = max(x, 0) / sum(max(x, 0))``, floored at 1e-9 before
      the log [divergence].
    * ``mahalanobis`` - Mahalanobis distance with one pooled within-class covariance
      (equal-prior linear discriminant, LDA) in a PCA subspace [distance, std units].
    * ``gaussian_ml`` - Gaussian maximum likelihood (QDA, equal priors): one covariance per
      class in a PCA subspace; the label minimises ``d_k**2 + log det(S_k)``, the score is
      ``d_k`` of that class [distance, std units].
    * ``random_forest``, ``svm`` - scikit-learn models [probability]; listed here but
      disabled until ``modules.cube_viewer.ml`` is imported and registers them with
      :func:`register_method` (only possible when scikit-learn is installed).

    The PCA basis is the leading eigenvectors of the class-balanced total covariance
    (mean of the per-class sample covariances plus the covariance of the class means),
    ``pca_dims`` of them (default 20 for ``mahalanobis``, 10 for ``gaussian_ml``).
    Covariances use :func:`ledoit_wolf` shrinkage (the same estimate as
    ``sklearn.covariance.ledoit_wolf``), so 25 pixels per class are enough.

    Every method marks a pixel with any non-finite band or an all-zero spectrum as
    unclassified (label -1, score NaN) - and the method-specific cases above.

Thresholds:
    ``model.default_thresholds()`` is ``THRESHOLD_FACTOR`` (1.5) x the
    ``THRESHOLD_PERCENTILE`` (99th) percentile of the scores of the class's own training
    pixels (its sample) against its own prototype - the angle for SAM, the distance for the
    distance methods.  A class with fewer than ``MIN_THRESHOLD_PX`` (10) scoreable
    training pixels (for example a single point) gets the median threshold of the other
    classes (``inf`` when no class has enough).  ``apply_thresholds`` keeps a pixel when
    ``score <= threshold[label]`` (``>=`` for 'probability'), compared exactly as in
    float64 although the maps are float32; a NaN threshold rejects its whole class.

Separability (``separability``):
    * ``angle_deg`` - K x K spectral angle between the exact class means (symmetric, zero
      diagonal; ``2 atan2(|a - b|, |a + b|)`` of the unit means, exact for small angles).
    * ``jm`` - Jeffries-Matusita distance ``2 (1 - exp(-B))`` (0 = identical, 2 = fully
      separable) with the Bhattacharyya distance B of Gaussian classes in ``JM_DIMS`` (10)
      PCA dimensions with Ledoit-Wolf covariances; NaN for a class with fewer than 3
      sample pixels.
    * ``f_curve`` - one-way ANOVA F per band on the class samples.
    * ``confusion`` - ROI consistency (rows = true class, columns = best class; thresholds
      not applied): leave-one-ROI-out for a class with >= 2 ROIs in its sample, else
      split-half (raster-order top / bottom halves of its sample).  Optimistic: neighbouring
      pixels are correlated - not a model accuracy.
    * ``warnings`` - class warnings (only points, small), pairs whose mean angle is within
      a class's default SAM threshold, and a possible additive (dark) offset estimated from
      how the spectral shape of each class changes with brightness.

Precision (float32 kernels; checked against float64 references in the tests):
    SAM / SCM angle error is about ``2e-7 rad / sin(angle)``: at most 2e-3 degrees for
    angles >= 0.5 degrees.  Distances lose about ``6e-8 |x|**2 / d**2`` relative accuracy
    (cancellation) for pixels very close to a class mean.  Labels can differ from a float64
    computation only where two classes score equal within that rounding.
    The labels and scores of :func:`classify_cube` do not depend on ``chunk_rows`` or on
    the cube's memory layout: every matrix product is done one image row at a time
    (BLAS products of different row counts round differently), on bit-identical float32
    input.

Memory:
    ``classify_cube`` allocates the two result maps (``6 H W`` bytes) and one contiguous
    float32 buffer of ``chunk_rows`` image rows padded to a multiple of 16 pixels (8 rows
    of 1776 x 185: 10.5 MB) plus small per-chunk scratch (SID: one more chunk-sized
    buffer).  ``build_classes`` holds one int32 owner map (and a bool exclusion mask) per
    class over its ROIs' combined bounding box.

Measured cost (i9-12900K, numpy 2.1.3 + MKL, in-RAM synthetic 1024 x 1762 x 185 uint16
cube, best of 3 in one process, K = 2 / 5 / 8 classes of 400 px; peak extra memory from
tracemalloc, including the 10.8 MB of result maps):
    * ``classify_cube``: SAM 196-202 ms, 21 MiB; SCM 276-283 ms, 21 MiB; Euclidean
      192-202 ms, 21 MiB; Euclidean-normalised 237-244 ms, 21 MiB; SID 827-841 ms,
      31-32 MiB; Mahalanobis 248-274 ms, 22 MiB; Gaussian ML 275-369 ms, 23-27 MiB.
      Through a BSQ / BIL view: SAM 317 / 332 ms, Gaussian ML 491 / 772 ms; a warm
      ``.npy`` memmap: SAM 268 ms.  Same-process repeats drift by up to about 1.5x on
      this hybrid-core CPU (Gaussian ML 275 -> 430 ms).
    * ``fit``: 1-3 ms (SID 3-13 ms) for the prototype methods; Mahalanobis / Gaussian ML
      12-18 ms with 400 px per class, 59-65 ms with 5000 px samples.
    * ``build_classes``: 2-6 ms for 2-8 rectangles of 400 px; 226 ms and 20 MiB for a
      711k-px class (300-vertex lasso + rectangle) plus two rectangles with an overlap.
    * ``separability``: 21 / 27 / 47 ms for K = 2 / 5 / 8 (400 px per class); 223 ms for
      3 classes with 5000-px samples.
    * ``apply_thresholds`` 8.5-13 ms, ``class_counts`` 6.6-7.5 ms for the full map.

Threading:
    Pure numpy; safe in a worker thread, the GIL is released in the heavy loops.
    ``cancel()`` is polled once per chunk (``classify_cube``), per ROI / class
    (``build_classes``, ``fit``) and per fold (``separability``); it then raises
    :class:`~modules.cube_viewer.roi_stats.RoiCancelled`.  Models are read-only after
    ``fit`` and may be shared between threads.

Example:
    >>> import numpy as np
    >>> from types import SimpleNamespace as Rec
    >>> cube = np.ones((4, 6, 3), np.uint16) * np.array([10, 20, 30], np.uint16)
    >>> cube[:, 3:] = np.array([30, 20, 10], np.uint16)
    >>> recs = [Rec(rid=1, class_name="a", kind="rect", color="#ff0000", state="ready",
    ...             stats=object(), geometry={"c0": 0, "r0": 0, "c1": 2, "r1": 4}),
    ...         Rec(rid=2, class_name="b", kind="rect", color="#0000ff", state="ready",
    ...             stats=object(), geometry={"c0": 4, "r0": 0, "c1": 6, "r1": 4})]
    >>> classes, notes = build_classes(cube, recs)
    >>> result = classify_cube(cube, fit("sam", classes))
    >>> apply_thresholds(result, result.default_thresholds)[0].tolist()
    [0, 0, 0, 1, 1, 1]
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .roi_stats import RoiCancelled, polygon_mask, roi_stats

__all__ = (
    "UNCLASSIFIED", "THRESHOLD_FACTOR", "THRESHOLD_PERCENTILE", "MIN_THRESHOLD_PX",
    "SMALL_CLASS_PX", "JM_DIMS", "SCORE_KINDS", "FAMILIES",
    "ClassSpec", "MethodInfo", "SegmentResult", "METHODS",
    "build_classes", "register_method", "fit", "classify_cube", "apply_thresholds",
    "class_counts", "separability", "spectral_angle_matrix", "ledoit_wolf",
)

UNCLASSIFIED = -1
THRESHOLD_FACTOR = 1.5
THRESHOLD_PERCENTILE = 99.0
MIN_THRESHOLD_PX = 10
SMALL_CLASS_PX = 185
JM_DIMS = 10
SCORE_KINDS = ("angle_deg", "distance", "divergence", "probability")
FAMILIES = ("prototype", "statistical", "ml")

_SID_FLOOR = 1e-9            # p and q are floored here before the log
_ARGBEST_LOOP_MAX = 8        # classes up to which _argbest loops over columns
_SCM_FLAT = 1e-10            # a spectrum with var <= 1e-10 * mean**2 is flat for SCM
_JM_MIN_PX = 3               # fewer sample pixels -> no JM for that class
_OFFSET_MIN_PX = 20          # offset estimate: sample pixels needed
_OFFSET_MIN_SPREAD = 0.05    # ... and std of the relative brightness
_OFFSET_MIN_LEVEL = 0.02     # warn when |offset| > 2 % of the class's mean level


# --------------------------------------------------------------------------- data classes


@dataclass(eq=False)
class ClassSpec:
    """One training class: the pooled ROI pixels of every record with the same ``class_name``.

    Attributes:
        name: The class name.
        color: ``'#rrggbb'`` of the class's first record in creation order.
        rids: Record ids pooled into the class, in creation order.
        n_px: Pixels used: the union of the records' pixels minus pixels another class
            also claims.
        mean: ``(B,)`` float64 exact mean over those pixels (``roi_stats``: correctly rounded
            for integer cubes of <= 16 bits; NaN values skipped per band).
        sample: ``(n, B)`` float32, ``n <= max_per_class`` pixels drawn without replacement
            with a fixed seed, in raster order; only finite, not all-zero spectra.
        sample_rids: ``(n,)`` int64 record id of each sampled pixel (a pixel inside two ROIs
            of the class belongs to the earlier one).
        only_points: True when every record of the class is a point.
        warnings: Human-readable warnings about this class.

    Compared by identity (``eq=False``): the fields are arrays.
    """

    name: str
    color: str
    rids: tuple
    n_px: int
    mean: np.ndarray
    sample: np.ndarray
    sample_rids: np.ndarray
    only_points: bool
    warnings: list = field(default_factory=list)

    @property
    def n_rois(self) -> int:
        """Number of records (ROIs) pooled into the class."""
        return len(self.rids)


@dataclass(frozen=True)
class MethodInfo:
    """Description of one segmentation method in :data:`METHODS`.

    Attributes:
        key: Registry key (``'sam'``, ...).
        label: Text for the method dropdown.
        family: ``'prototype'`` (class means), ``'statistical'`` (means and covariances) or
            ``'ml'`` (scikit-learn).
        score_kind: ``'angle_deg'``, ``'distance'``, ``'divergence'`` (lower is better) or
            ``'probability'`` (higher is better).
        params: Default parameters (copy before editing); :func:`fit` accepts these keys.
        available: False when the method cannot be fitted (scikit-learn missing or
            ``modules.cube_viewer.ml`` not imported); the dropdown shows it disabled.
        description: One or two sentences for a tooltip (or why it is unavailable).
        fitter: ``fitter(classes, params, *, cancel=None) -> model``; None for the built-in
            numpy methods.
    """

    key: str
    label: str
    family: str
    score_kind: str
    params: dict
    available: bool = True
    description: str = ""
    fitter: Optional[Callable] = field(default=None, repr=False, compare=False)

    @property
    def lower_is_better(self) -> bool:
        """True unless the score is a probability."""
        return self.score_kind != "probability"


@dataclass(eq=False)
class SegmentResult:
    """Output of :func:`classify_cube` (best class and its score for every pixel).

    Attributes:
        method: Method key of the model.
        class_names, colors: Tuples, one entry per class, in label order.
        labels: ``(H, W)`` int16 best class, ``-1`` where the pixel cannot be scored.
        score: ``(H, W)`` float32 score of that class, NaN where unscored.
        default_thresholds: ``(K,)`` float64, ``model.default_thresholds()``.
        elapsed_s: Wall time of the classification.
        score_kind: The method's score kind (see :class:`MethodInfo`).
    """

    method: str
    class_names: tuple
    colors: tuple
    labels: np.ndarray
    score: np.ndarray
    default_thresholds: np.ndarray
    elapsed_s: float
    score_kind: str = "angle_deg"

    @property
    def n_classes(self) -> int:
        """Number of classes K."""
        return len(self.class_names)


# --------------------------------------------------------------------------- helpers


def _cube_shape(cube):
    """``(H, W, B)`` of a 3-D numeric cube (ValueError / TypeError otherwise)."""
    shape = getattr(cube, "shape", None)
    dtype = getattr(cube, "dtype", None)
    if shape is None or dtype is None or len(shape) != 3:
        raise ValueError(f"cube must be a 3-D array (H, W, B), got shape {shape}")
    if np.dtype(dtype).kind not in "biuf":
        raise TypeError(f"cube dtype {dtype} is not a real numeric type")
    return tuple(int(s) for s in shape)


def _check_cancel(cancel) -> None:
    """Raise ``RoiCancelled`` when ``cancel()`` returns True."""
    if cancel is not None and cancel():
        raise RoiCancelled()


def _record_pixels(kind, geometry, height, width, cancel=None):
    """``(bbox, local_mask_or_None, n_px)`` of a record geometry (same rule as ``roi_draw.roi_selection``)."""
    h, w = int(height), int(width)
    if kind == "point":
        x, y = int(geometry["x"]), int(geometry["y"])
        if 0 <= x < w and 0 <= y < h:
            return (x, y, x + 1, y + 1), None, 1
        return (0, 0, 0, 0), None, 0
    if kind == "rect":
        c0, c1 = (min(max(int(geometry[k]), 0), w) for k in ("c0", "c1"))
        r0, r1 = (min(max(int(geometry[k]), 0), h) for k in ("r0", "r1"))
        return (c0, r0, c1, r1), None, max(0, c1 - c0) * max(0, r1 - r0)
    if kind in ("lasso", "polygon"):
        vertices = np.asarray(geometry["vertices"], dtype=np.float64).reshape(-1, 2)
        c0, r0, mask = polygon_mask(vertices, h, w, cancel=cancel)
        return (c0, r0, c0 + mask.shape[1], r0 + mask.shape[0]), mask, int(np.count_nonzero(mask))
    raise ValueError(f"unknown record kind {kind!r}")


def _is_ready(record) -> bool:
    """True for a record whose statistics are available."""
    ready = getattr(record, "is_ready", None)
    if isinstance(ready, bool):
        return ready
    return getattr(record, "state", None) == "ready" and getattr(record, "stats", None) is not None


def _scratch(work: dict, name: str, shape, dtype=np.float32):
    """A C-contiguous scratch array of ``shape`` from ``work`` (grown on demand, reused)."""
    size = math.prod(shape)
    buf = work.get(name)
    if buf is None or buf.dtype != dtype or buf.size < size:
        buf = np.empty(max(size, 1), dtype=dtype)
        work[name] = buf
    return buf[:size].reshape(shape)


def _argbest(G, labels_out, work, *, largest=True):
    """Best column of ``G (..., K)`` (first one on ties) into ``labels_out`` (int16); returns its value.

    Up to ``_ARGBEST_LOOP_MAX`` classes a loop over the columns is used: an 8-row chunk
    (product included) took 0.43 / 0.64 / 0.46 ms for K = 2 / 5 / 8 instead of 0.60 /
    0.73 / 0.67 ms with ``argmax`` + ``max`` over the short last axis, but 1.35 instead of
    1.00 ms for K = 16.  Both give the same result for finite scores.
    """
    shp = G.shape[:-1]
    best = _scratch(work, "best", shp)
    k_all = G.shape[-1]
    if k_all <= _ARGBEST_LOOP_MAX:
        better = _scratch(work, "better", shp, np.bool_)
        compare = np.greater if largest else np.less
        np.copyto(best, G[..., 0])
        labels_out[...] = 0
        for k in range(1, k_all):
            col = G[..., k]
            compare(col, best, out=better)
            np.copyto(best, col, where=better)
            np.copyto(labels_out, k, where=better, casting="unsafe")
        return best
    idx = _scratch(work, "idx", shp, np.intp)
    if largest:
        np.argmax(G, axis=-1, out=idx)
        np.max(G, axis=-1, out=best)
    else:
        np.argmin(G, axis=-1, out=idx)
        np.min(G, axis=-1, out=best)
    np.copyto(labels_out, idx, casting="unsafe")
    return best


def _row_sum(X, work, name):
    """Per-pixel ``sum(x)`` of ``X (..., B)`` (float32 einsum: 2x faster than ``np.sum`` here)."""
    out = _scratch(work, name, X.shape[:-1])
    np.einsum("...b->...", X, out=out)
    return out


def _sq_norm(X, work, name="s2"):
    """Per-pixel ``sum(x**2)`` of ``X (..., B)`` (float32; NaN / inf propagate)."""
    out = _scratch(work, name, X.shape[:-1])
    np.einsum("...b,...b->...", X, X, out=out)
    return out


def _nonzero_finite(s2):
    """Pixels that can be scored: squared norm finite and > 0."""
    return (s2 > 0) & (s2 < np.inf)


def _mark_invalid(valid, labels_out, score_out) -> None:
    """Set label -1 and score NaN where ``valid`` is False."""
    if not valid.all():
        bad = ~valid
        labels_out[bad] = UNCLASSIFIED
        score_out[bad] = np.nan


def _means(classes) -> np.ndarray:
    """``(K, B)`` float64 class means."""
    return np.stack([np.asarray(c.mean, dtype=np.float64) for c in classes])


def _require(ok, classes, what: str) -> None:
    """ValueError naming the classes where ``ok`` is False."""
    ok = np.asarray(ok, dtype=bool)
    if not ok.all():
        names = ", ".join(repr(classes[i].name) for i in np.flatnonzero(~ok))
        raise ValueError(f"class {names} {what}")


def _f32_toward(t, direction):
    """Thresholds as float32, rounded down (``direction < 0``) or up so float32 comparisons are exact."""
    t = np.asarray(t, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        f = t.astype(np.float32)
        f64 = f.astype(np.float64)
        wrong = (f64 > t) if direction < 0 else (f64 < t)
    if wrong.any():
        target = np.float32(-np.inf if direction < 0 else np.inf)
        f[wrong] = np.nextafter(f[wrong], target)
    return f


def _accept(labels, score, thresholds, score_kind):
    """Bool map: classified pixels whose score passes their class's threshold."""
    k = len(thresholds)
    if score_kind == "probability":
        thr = _f32_toward(thresholds, +1)
    else:
        thr = _f32_toward(thresholds, -1)
    table = np.empty(k + 1, dtype=np.float32)
    table[0] = np.nan                       # label -1 -> NaN -> never accepted
    table[1:] = thr
    per_px = np.take(table, labels + np.int16(1))
    if score_kind == "probability":
        return score >= per_px
    return score <= per_px


# --------------------------------------------------------------------------- classes


def build_classes(cube, records, *, max_per_class: int = 5000, seed: int = 0, cancel=None):
    """Pool the ready records into training classes by ``class_name``.

    Args:
        cube: ``(H, W, B)`` array (ndarray, memmap or view); only the ROI pixels are read.
        records: Spectrum records (any order; sorted by ``rid`` = creation order).  Only
            ready ones are used, whatever their visibility - filter before calling if needed.
            They are only read (``rid``, ``name``, ``class_name``, ``kind``, ``geometry``,
            ``color``, ``state``, ``stats``); from a worker thread pass copies taken in the
            GUI thread, so a rename during the run cannot mix old and new classes.
        max_per_class: Size limit of each class's pixel sample.
        seed: Seed of the sample; each class draws with ``default_rng(seed)``, so its sample
            depends only on its own pixels.
        cancel: Optional callable polled per record and per class (``RoiCancelled``).

    Returns:
        ``(classes, warnings)``: :class:`ClassSpec` list in order of first appearance, and
        warnings about skipped records, overlapping classes and dropped classes.

    Raises:
        ValueError / TypeError: Bad cube.
        RoiCancelled: ``cancel()`` returned True.
    """
    h_img, w_img, n_bands = _cube_shape(cube)
    limit = max(1, int(max_per_class))
    warnings: list = []
    ready, not_ready = [], []
    for rec in records:
        (ready if _is_ready(rec) else not_ready).append(rec)
    if not_ready:
        names = ", ".join(str(getattr(r, "name", r.rid)) for r in not_ready[:5])
        more = f" and {len(not_ready) - 5} more" if len(not_ready) > 5 else ""
        warnings.append(f"{len(not_ready)} spectra are not ready (computing or failed) and "
                        f"were left out: {names}{more}.")
    ready.sort(key=lambda r: r.rid)

    groups: dict = {}
    for rec in ready:
        _check_cancel(cancel)
        bbox, mask, n_px = _record_pixels(rec.kind, rec.geometry, h_img, w_img, cancel)
        if n_px == 0:
            warnings.append(f"'{getattr(rec, 'name', rec.rid)}' has no pixel inside the image "
                            f"and was left out.")
            continue
        groups.setdefault(rec.class_name, []).append((rec, bbox, mask))

    # Union of each class's ROIs over their combined bbox; owner = 1-based record index
    # (an earlier record wins a pixel shared by two ROIs of the class).
    unions = []
    for name, items in groups.items():
        c0 = min(b[0] for _, b, _ in items)
        r0 = min(b[1] for _, b, _ in items)
        c1 = max(b[2] for _, b, _ in items)
        r1 = max(b[3] for _, b, _ in items)
        owner = np.zeros((r1 - r0, c1 - c0), dtype=np.int32)
        for i, (_, (bc0, br0, bc1, br1), mask) in enumerate(items, 1):
            sub = owner[br0 - r0:br1 - r0, bc0 - c0:bc1 - c0]
            free = sub == 0
            if mask is not None:
                free &= mask
            sub[free] = i
        unions.append([name, items, (c0, r0, c1, r1), owner, None])

    # Pixels claimed by two classes are excluded from both.
    for a in range(len(unions)):
        for b in range(a + 1, len(unions)):
            (ca0, ra0, ca1, ra1), (cb0, rb0, cb1, rb1) = unions[a][2], unions[b][2]
            x0, y0, x1, y1 = max(ca0, cb0), max(ra0, rb0), min(ca1, cb1), min(ra1, rb1)
            if x1 <= x0 or y1 <= y0:
                continue
            both = ((unions[a][3][y0 - ra0:y1 - ra0, x0 - ca0:x1 - ca0] > 0)
                    & (unions[b][3][y0 - rb0:y1 - rb0, x0 - cb0:x1 - cb0] > 0))
            n_both = int(np.count_nonzero(both))
            if not n_both:
                continue
            for u, (cx, ry) in ((unions[a], (ca0, ra0)), (unions[b], (cb0, rb0))):
                if u[4] is None:
                    u[4] = np.zeros(u[3].shape, dtype=bool)
                u[4][y0 - ry:y1 - ry, x0 - cx:x1 - cx] |= both
            warnings.append(f"Classes '{unions[a][0]}' and '{unions[b][0]}' overlap on {n_both:,} px; "
                            f"those pixels are excluded from both.")

    classes = []
    for name, items, (c0, r0, c1, r1), owner, excluded in unions:
        _check_cancel(cancel)
        keep = owner > 0
        if excluded is not None:
            keep &= ~excluded
        n_px = int(np.count_nonzero(keep))
        if n_px == 0:
            warnings.append(f"Class '{name}' has no pixel left after removing the overlaps "
                            f"and was left out.")
            continue
        stats = roi_stats(cube, (c0, r0, c1, r1), keep, cancel=cancel)
        mean = np.array(stats.mean, dtype=np.float64)
        if not np.isfinite(mean).all():
            warnings.append(f"Class '{name}' has bands without a valid pixel and was left out.")
            continue
        if not np.any(mean != 0):
            warnings.append(f"Class '{name}' has an all-zero mean spectrum and was left out.")
            continue
        local = np.flatnonzero(keep)
        if local.size > limit:
            rng = np.random.default_rng(seed)
            local = local[np.sort(rng.choice(local.size, size=limit, replace=False))]
        bw = c1 - c0
        rows, cols = local // bw + r0, local % bw + c0
        sample = np.asarray(cube[rows, cols], dtype=np.float32).reshape(-1, n_bands)
        rid_of = np.array([rec.rid for rec, _, _ in items], dtype=np.int64)
        sample_rids = rid_of[owner.ravel()[local] - 1]
        usable = np.isfinite(sample).all(axis=1) & np.any(sample != 0, axis=1)
        if not usable.all():
            sample, sample_rids = sample[usable], sample_rids[usable]
        only_points = all(rec.kind == "point" for rec, _, _ in items)
        notes = []
        if only_points:
            notes.append(f"'{name}' is made only of points ({n_px} px): single-pixel "
                         f"references are noisy; draw a rectangle or lasso for this class.")
        elif n_px < max(SMALL_CLASS_PX, n_bands):
            notes.append(f"'{name}' has only {n_px} px (< {max(SMALL_CLASS_PX, n_bands)}): "
                         f"its covariance (Mahalanobis, Gaussian ML, JM) is poorly estimated.")
        classes.append(ClassSpec(
            name=name, color=items[0][0].color, rids=tuple(int(r) for r in rid_of),
            n_px=n_px, mean=mean, sample=np.ascontiguousarray(sample), sample_rids=sample_rids,
            only_points=only_points, warnings=notes))
    return classes, warnings


# --------------------------------------------------------------------------- statistics helpers


def ledoit_wolf(X):
    """Ledoit-Wolf shrunk covariance of ``X (n, p)`` - the estimate of ``sklearn.covariance.ledoit_wolf``.

    The data are centred on their mean; the empirical covariance (``/ n``) is shrunk
    toward ``mu I`` with ``mu = trace / p``.

    Returns:
        ``(cov (p, p) float64, shrinkage float)``.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 1:
        raise ValueError(f"ledoit_wolf: expected a non-empty (n, p) array, got shape {X.shape}")
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    if p == 1:
        return np.atleast_2d((Xc ** 2).mean()), 0.0
    emp = Xc.T @ Xc / n
    mu = np.trace(emp) / p
    X2 = Xc ** 2
    delta_ = float(np.sum(emp ** 2))
    beta = (float(np.sum(X2.T @ X2)) / n - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * np.trace(emp) + p * mu ** 2) / p
    beta = min(beta, delta)
    shrink = 0.0 if beta == 0 else beta / delta
    cov = (1.0 - shrink) * emp
    cov.flat[::p + 1] += shrink * mu
    return cov, float(shrink)


def _cholesky_pd(S):
    """Cholesky factor of a symmetric PSD matrix with a tiny relative ridge (None if degenerate)."""
    p = S.shape[0]
    scale = np.trace(S) / p
    if not np.isfinite(scale) or scale <= 0:
        return None
    try:
        return np.linalg.cholesky(S + (1e-10 * scale) * np.eye(p))
    except np.linalg.LinAlgError:
        return None


def _pca_basis(classes, dims: int):
    """``(P (B, p), mu0 (B,))``: leading eigenvectors of the class-balanced total covariance."""
    means = _means(classes)
    n_bands = means.shape[1]
    p = max(1, min(int(dims), n_bands))
    mu0 = means.mean(axis=0)
    C = np.zeros((n_bands, n_bands))
    for c in classes:
        X = np.asarray(c.sample, dtype=np.float64)
        if X.shape[0] >= 2:
            Xc = X - X.mean(axis=0)
            C += Xc.T @ Xc / X.shape[0]
    D = means - mu0
    C = (C + D.T @ D) / len(classes)
    w, V = np.linalg.eigh(C)
    V = V[:, np.argsort(w)[::-1][:p]]
    flip = V[np.abs(V).argmax(axis=0), np.arange(p)] < 0    # deterministic signs
    V[:, flip] *= -1.0
    return V, mu0


def _pca_dims(params, n_bands) -> int:
    """Validated ``pca_dims`` parameter, clipped to the band count."""
    try:
        dims = int(params["pca_dims"])
    except (TypeError, ValueError):
        raise ValueError(f"pca_dims must be an integer, got {params['pca_dims']!r}") from None
    if dims < 1:
        raise ValueError(f"pca_dims must be >= 1, got {dims}")
    return min(dims, n_bands)


def spectral_angle_matrix(means) -> np.ndarray:
    """``(K, K)`` angles in degrees between mean spectra (symmetric, exact zero diagonal).

    Uses ``2 atan2(|a - b|, |a + b|)`` of the unit vectors, which stays accurate for small
    angles.  A zero vector gives NaN in its row and column.
    """
    M = np.asarray(means, dtype=np.float64)
    k = M.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        U = M / np.linalg.norm(M, axis=1, keepdims=True)
    out = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            ang = 2.0 * math.atan2(float(np.linalg.norm(U[i] - U[j])), float(np.linalg.norm(U[i] + U[j])))
            out[i, j] = out[j, i] = math.degrees(ang)
    bad = ~np.isfinite(U).all(axis=1)
    out[bad, :] = np.nan
    out[:, bad] = np.nan
    return out


def _angle_deg(X, unit):
    """float64 angle (degrees) between rows of ``X`` and the unit vector ``unit``; NaN for zero rows."""
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (X @ unit) / np.linalg.norm(X, axis=1)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


# --------------------------------------------------------------------------- models


class _NumpyModel:
    """Common part of the numpy models (the Model protocol: see :func:`fit`)."""

    overwrites_input = False

    def __init__(self, method, classes, params):
        """Store the classes and parameters; subclasses fit and then call ``_init_thresholds``."""
        info = METHODS[method]
        self.method = method
        self.score_kind = info.score_kind
        self.classes = list(classes)
        self.params = dict(params)
        self.n_bands = int(self.classes[0].mean.size)
        self.warnings: list = []
        self._thresholds = None

    def own_scores(self, k: int, X) -> np.ndarray:
        """float64 scores of the pixels ``X (n, B)`` against class ``k`` (NaN when unscoreable)."""
        raise NotImplementedError

    def _init_thresholds(self) -> None:
        """Default per-class thresholds: 1.5 x P99 of each class's own training scores."""
        k_all = len(self.classes)
        thr = np.full(k_all, np.nan)
        reliable = np.zeros(k_all, dtype=bool)
        for k, c in enumerate(self.classes):
            s = self.own_scores(k, np.asarray(c.sample, dtype=np.float64))
            s = s[np.isfinite(s)]
            if s.size >= MIN_THRESHOLD_PX:
                thr[k] = THRESHOLD_FACTOR * float(np.percentile(s, THRESHOLD_PERCENTILE))
                reliable[k] = True
        if not reliable.all():
            fallback = float(np.median(thr[reliable])) if reliable.any() else math.inf
            for k in np.flatnonzero(~reliable):
                thr[k] = fallback
                why = ("the median of the other classes" if math.isfinite(fallback)
                       else "infinite (no class has enough pixels)")
                self.warnings.append(f"'{self.classes[k].name}' has fewer than {MIN_THRESHOLD_PX} "
                                     f"training pixels; its default threshold is {why}.")
        self._thresholds = thr

    def default_thresholds(self) -> np.ndarray:
        """``(K,)`` float64 default reject thresholds (a copy)."""
        return self._thresholds.copy()

    def predict_chunk(self, X):
        """Best class and score of every pixel of ``X (..., B)``; ``X`` is not modified.

        Returns:
            ``(labels int16 (...), score float32 (...))``; label -1 / score NaN where a pixel
            cannot be scored.
        """
        X = np.asarray(X)
        if X.ndim < 1 or X.shape[-1] != self.n_bands:
            raise ValueError(f"expected pixels with {self.n_bands} bands, got shape {X.shape}")
        if self.overwrites_input or X.dtype != np.float32:
            X = np.array(X, dtype=np.float32)
        labels = np.empty(X.shape[:-1], dtype=np.int16)
        score = np.empty(X.shape[:-1], dtype=np.float32)
        if labels.size:
            with np.errstate(all="ignore"):              # NaN / inf / zero pixels are expected
                self.predict_into(X, labels, score, {})
        return labels, score

    def predict_into(self, X, labels_out, score_out, work: dict) -> None:
        """Fast path of :func:`classify_cube`: write into ``labels_out`` / ``score_out``.

        ``X`` is float32 ``(..., B)`` and may be overwritten (``overwrites_input``);
        ``work`` is a dict of reusable scratch buffers owned by the caller.
        """
        raise NotImplementedError


class _SamModel(_NumpyModel):
    """Spectral angle mapper."""

    def __init__(self, classes, params):
        """Unit class means."""
        super().__init__("sam", classes, params)
        M = _means(self.classes)
        norms = np.linalg.norm(M, axis=1)
        _require(norms > 0, self.classes, "has an all-zero mean spectrum")
        self._unit = M / norms[:, None]
        self._MT = np.ascontiguousarray(self._unit.T, dtype=np.float32)
        self._init_thresholds()

    def own_scores(self, k, X):
        """Angle in degrees to the class mean."""
        return _angle_deg(X, self._unit[k])

    def predict_into(self, X, labels_out, score_out, work):
        """``argmax(x . m_k / |m_k|)``, angle ``arccos(that / |x|)``."""
        shp = X.shape[:-1]
        G = _scratch(work, "G", shp + (self._MT.shape[1],))
        np.matmul(X, self._MT, out=G)
        best = _argbest(G, labels_out, work)
        s2 = _sq_norm(X, work)
        valid = _nonzero_finite(s2)
        with np.errstate(all="ignore"):
            np.sqrt(s2, out=s2)
            np.divide(best, s2, out=best)
            np.clip(best, -1.0, 1.0, out=best)
            np.arccos(best, out=best)
        np.degrees(best, out=score_out)
        _mark_invalid(valid, labels_out, score_out)


class _ScmModel(_NumpyModel):
    """Spectral correlation mapper (Pearson correlation angle)."""

    overwrites_input = True

    def __init__(self, classes, params):
        """Unit mean-centred class means."""
        super().__init__("scm", classes, params)
        M = _means(self.classes)
        Mc = M - M.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(Mc, axis=1)
        _require(norms > 0, self.classes, "has a flat mean spectrum (no correlation is defined)")
        self._unit = Mc / norms[:, None]
        self._MT = np.ascontiguousarray(self._unit.T, dtype=np.float32)
        self._init_thresholds()

    def own_scores(self, k, X):
        """``arccos(r)`` in degrees; NaN for flat pixels."""
        mean = X.mean(axis=1, keepdims=True)
        Xc = X - mean
        v = np.einsum("ij,ij->i", Xc, Xc)
        ang = _angle_deg(Xc, self._unit[k])
        ang[~(v > _SCM_FLAT * X.shape[1] * mean[:, 0] ** 2)] = np.nan
        return ang

    def predict_into(self, X, labels_out, score_out, work):
        """Centre every pixel in place, then SAM against the centred unit means."""
        shp = X.shape[:-1]
        n_bands = X.shape[-1]
        mean = _row_sum(X, work, "mean")
        np.divide(mean, np.float32(n_bands), out=mean)
        np.subtract(X, mean[..., None], out=X)
        v = _sq_norm(X, work, "v")
        tol = _scratch(work, "tol", shp)
        np.multiply(mean, mean, out=tol)
        np.multiply(tol, np.float32(_SCM_FLAT * n_bands), out=tol)
        valid = (v > tol) & (v < np.inf)
        G = _scratch(work, "G", shp + (self._MT.shape[1],))
        np.matmul(X, self._MT, out=G)
        best = _argbest(G, labels_out, work)
        with np.errstate(all="ignore"):
            np.sqrt(v, out=v)
            np.divide(best, v, out=best)
            np.clip(best, -1.0, 1.0, out=best)
            np.arccos(best, out=best)
        np.degrees(best, out=score_out)
        _mark_invalid(valid, labels_out, score_out)


class _EuclideanModel(_NumpyModel):
    """Minimum distance to the class means in cube units."""

    def __init__(self, classes, params):
        """Class means and half their squared norms."""
        super().__init__("euclidean", classes, params)
        M = _means(self.classes)
        self._M = M
        self._MT = np.ascontiguousarray(M.T, dtype=np.float32)
        self._half = (0.5 * np.einsum("kb,kb->k", M, M)).astype(np.float32)
        self._init_thresholds()

    def own_scores(self, k, X):
        """Euclidean distance to the class mean (NaN for all-zero pixels)."""
        d = np.linalg.norm(X - self._M[k], axis=1)
        d[~np.any(X != 0, axis=1)] = np.nan
        return d

    def predict_into(self, X, labels_out, score_out, work):
        """``argmax(x . m - |m|^2 / 2)``; ``d = sqrt(|x|^2 - 2 max)``."""
        shp = X.shape[:-1]
        G = _scratch(work, "G", shp + (self._MT.shape[1],))
        np.matmul(X, self._MT, out=G)
        np.subtract(G, self._half, out=G)
        best = _argbest(G, labels_out, work)
        s2 = _sq_norm(X, work)
        valid = _nonzero_finite(s2)
        with np.errstate(all="ignore"):
            np.multiply(best, np.float32(2.0), out=best)
            np.subtract(s2, best, out=best)
            np.maximum(best, np.float32(0.0), out=best)
            np.sqrt(best, out=score_out)
        _mark_invalid(valid, labels_out, score_out)


class _EuclideanNormModel(_NumpyModel):
    """Minimum distance between brightness-normalised spectra ``x / mean(x)``."""

    def __init__(self, classes, params):
        """Normalised class means and half their squared norms."""
        super().__init__("euclidean_norm", classes, params)
        M = _means(self.classes)
        sums = M.sum(axis=1)
        _require(sums > 0, self.classes, "has a mean spectrum whose mean level is not positive")
        self._Mn = M * (M.shape[1] / sums)[:, None]
        self._MT = np.ascontiguousarray(self._Mn.T, dtype=np.float32)
        self._half = (0.5 * np.einsum("kb,kb->k", self._Mn, self._Mn)).astype(np.float32)
        self._init_thresholds()

    def own_scores(self, k, X):
        """Distance between ``x / mean(x)`` and the normalised class mean; NaN when mean <= 0."""
        s = X.sum(axis=1)
        with np.errstate(all="ignore"):
            d = np.linalg.norm(X * (X.shape[1] / s)[:, None] - self._Mn[k], axis=1)
        d[~(s > 0)] = np.nan
        return d

    def predict_into(self, X, labels_out, score_out, work):
        """With ``f = B / sum(x)``: ``argmax(f x . m - |m|^2 / 2)``, ``d^2 = f^2 |x|^2 - 2 max``."""
        shp = X.shape[:-1]
        f = _row_sum(X, work, "f")
        valid = (f > 0) & (f < np.inf)
        G = _scratch(work, "G", shp + (self._MT.shape[1],))
        np.matmul(X, self._MT, out=G)
        with np.errstate(all="ignore"):
            np.divide(np.float32(X.shape[-1]), f, out=f)
            np.multiply(G, f[..., None], out=G)
        np.subtract(G, self._half, out=G)
        best = _argbest(G, labels_out, work)
        s2 = _sq_norm(X, work)
        with np.errstate(all="ignore"):
            np.multiply(s2, f, out=s2)
            np.multiply(s2, f, out=s2)
            np.multiply(best, np.float32(2.0), out=best)
            np.subtract(s2, best, out=best)
            np.maximum(best, np.float32(0.0), out=best)
            np.sqrt(best, out=score_out)
        _mark_invalid(valid, labels_out, score_out)


class _SidModel(_NumpyModel):
    """Spectral information divergence."""

    overwrites_input = True

    def __init__(self, classes, params):
        """Class distributions ``q``, their logs relative to a reference log-shape ``r``."""
        super().__init__("sid", classes, params)
        M = np.maximum(_means(self.classes), 0.0)
        sums = M.sum(axis=1)
        _require(sums > 0, self.classes, "has no positive value in its mean spectrum")
        Q = np.maximum(M / sums[:, None], _SID_FLOOR)
        logQ = np.log(Q)
        self._Q, self._logQ = Q, logQ
        # Terms are taken relative to r (the mean log-shape) to limit float32 cancellation;
        # sum((p - q)(log p - log q)) is unchanged by subtracting r from both logs.
        r = logQ.mean(axis=0)
        lq = logQ - r
        self._r = r.astype(np.float32)
        self._QT = np.ascontiguousarray(Q.T, dtype=np.float32)
        self._LT = np.ascontiguousarray(lq.T, dtype=np.float32)
        self._c = np.einsum("kb,kb->k", Q, lq).astype(np.float32)
        self._init_thresholds()

    def own_scores(self, k, X):
        """SID of each pixel against the class (direct float64 formula); NaN when unscoreable."""
        Xp = np.maximum(X, 0.0)
        s = Xp.sum(axis=1)
        with np.errstate(all="ignore"):
            P = np.maximum(Xp / s[:, None], _SID_FLOOR)
            d = np.einsum("ij,ij->i", P - self._Q[k], np.log(P) - self._logQ[k])
        d[~((s > 0) & np.isfinite(s))] = np.nan
        return d

    def predict_into(self, X, labels_out, score_out, work):
        """``SID_k = a + c_k - p . (log q_k - r) - (log p - r) . q_k`` with ``a = p . (log p - r)``."""
        shp = X.shape[:-1]
        if not work.get("nonnegative"):
            np.maximum(X, np.float32(0.0), out=X)
        s = _row_sum(X, work, "s")
        valid = (s > 0) & (s < np.inf)
        L = _scratch(work, "L", X.shape)
        a = _scratch(work, "a", shp)
        k_all = self._QT.shape[1]
        G = _scratch(work, "G", shp + (k_all,))
        G2 = _scratch(work, "G2", shp + (k_all,))
        with np.errstate(all="ignore"):
            np.divide(np.float32(1.0), s, out=s)
            np.multiply(X, s[..., None], out=X)
            np.maximum(X, np.float32(_SID_FLOOR), out=X)
            np.log(X, out=L)
            np.subtract(L, self._r, out=L)
            np.einsum("...b,...b->...", X, L, out=a)
            np.matmul(X, self._LT, out=G)
            np.matmul(L, self._QT, out=G2)
            np.add(G, G2, out=G)
            np.subtract(G, a[..., None], out=G)
            np.subtract(G, self._c, out=G)              # G = -SID
        best = _argbest(G, labels_out, work)
        np.negative(best, out=best)
        np.maximum(best, np.float32(0.0), out=score_out)
        _mark_invalid(valid & np.isfinite(score_out), labels_out, score_out)


class _MahalanobisModel(_NumpyModel):
    """Mahalanobis distance with one pooled Ledoit-Wolf covariance (LDA) in a PCA subspace."""

    def __init__(self, classes, params, cancel=None):
        """PCA basis, pooled within-class covariance, whitening."""
        super().__init__("mahalanobis", classes, params)
        p = _pca_dims(self.params, self.n_bands)
        P, mu0 = _pca_basis(self.classes, p)
        p = P.shape[1]
        within = []
        for c in self.classes:
            _check_cancel(cancel)
            Z = (np.asarray(c.sample, dtype=np.float64) - mu0) @ P
            if Z.shape[0] >= 2:
                within.append(Z - Z.mean(axis=0))
        if not within:
            raise ValueError("Mahalanobis / LDA needs at least one class with 2 or more pixels")
        S, self.shrinkage = ledoit_wolf(np.concatenate(within))
        L = _cholesky_pd(S)
        if L is None:
            raise ValueError("Mahalanobis / LDA: the training pixels have no variance")
        Wh = np.linalg.inv(L).T                              # y = z @ Wh is whitened
        self._A = P @ Wh                                     # (B, p): x -> y (before the offset)
        self._off = mu0 @ self._A
        self._mw = ((_means(self.classes) - mu0) @ P) @ Wh   # whitened class means (K, p)
        self.pca_dims = p
        self._A32 = np.ascontiguousarray(self._A, dtype=np.float32)
        self._off32 = self._off.astype(np.float32)
        self._MwT = np.ascontiguousarray(self._mw.T, dtype=np.float32)
        self._half = (0.5 * np.einsum("kp,kp->k", self._mw, self._mw)).astype(np.float32)
        self._init_thresholds()

    def own_scores(self, k, X):
        """Pooled Mahalanobis distance to class ``k`` (NaN for all-zero pixels)."""
        d = np.linalg.norm(X @ self._A - self._off - self._mw[k], axis=1)
        d[~np.any(X != 0, axis=1)] = np.nan
        return d

    def predict_into(self, X, labels_out, score_out, work):
        """Whiten (one product with ``(B, p)``), then minimum distance to the whitened means."""
        shp = X.shape[:-1]
        Y = _scratch(work, "Y", shp + (self._A32.shape[1],))
        np.matmul(X, self._A32, out=Y)
        np.subtract(Y, self._off32, out=Y)
        y2 = _sq_norm(Y, work, "y2")
        G = _scratch(work, "G", shp + (self._MwT.shape[1],))
        np.matmul(Y, self._MwT, out=G)
        np.subtract(G, self._half, out=G)
        best = _argbest(G, labels_out, work)
        s2 = _sq_norm(X, work)
        valid = _nonzero_finite(s2)
        with np.errstate(all="ignore"):
            np.multiply(best, np.float32(2.0), out=best)
            np.subtract(y2, best, out=best)
            np.maximum(best, np.float32(0.0), out=best)
            np.sqrt(best, out=score_out)
        _mark_invalid(valid & np.isfinite(score_out), labels_out, score_out)


class _GaussianMlModel(_NumpyModel):
    """Gaussian maximum likelihood (QDA): one Ledoit-Wolf covariance per class in a PCA subspace."""

    def __init__(self, classes, params, cancel=None):
        """PCA basis and per-class whitening matrices and log-determinants."""
        super().__init__("gaussian_ml", classes, params)
        p = _pca_dims(self.params, self.n_bands)
        P, mu0 = _pca_basis(self.classes, p)
        p = P.shape[1]
        zs = [(np.asarray(c.sample, dtype=np.float64) - mu0) @ P for c in self.classes]
        pooled = None
        within = [Z - Z.mean(axis=0) for Z in zs if Z.shape[0] >= 2]
        if within:
            pooled = _cholesky_pd(ledoit_wolf(np.concatenate(within))[0])
        mz = (_means(self.classes) - mu0) @ P
        whiten, offsets, logdet = [], [], []
        self.shrinkage = []
        for k, (c, Z) in enumerate(zip(self.classes, zs)):
            _check_cancel(cancel)
            L, shrink = None, math.nan
            if Z.shape[0] >= 2:
                S, shrink = ledoit_wolf(Z)
                L = _cholesky_pd(S)
            if L is None:
                if pooled is None:
                    raise ValueError("Gaussian ML needs at least one class with 2 or more "
                                     "pixels that vary")
                L = pooled
                self.warnings.append(f"'{c.name}': too few varying pixels for its own "
                                     f"covariance; the pooled covariance is used.")
            Wk = np.linalg.inv(L).T
            whiten.append(Wk)
            offsets.append(mz[k] @ Wk)
            logdet.append(2.0 * float(np.log(np.diag(L)).sum()))
            self.shrinkage.append(shrink)
        self.pca_dims = p
        self._P, self._mu0P = P, mu0 @ P
        self._W = whiten
        self._b = offsets
        self._logdet = np.array(logdet)
        self._P32 = np.ascontiguousarray(P, dtype=np.float32)
        self._mu0P32 = self._mu0P.astype(np.float32)
        self._Wcat = np.ascontiguousarray(np.hstack(whiten), dtype=np.float32)    # (p, K p)
        self._bcat = np.concatenate(offsets).astype(np.float32)
        self._logdet32 = self._logdet.astype(np.float32)
        self._init_thresholds()

    def own_scores(self, k, X):
        """Mahalanobis distance under class ``k``'s own covariance (NaN for all-zero pixels)."""
        Z = X @ self._P - self._mu0P
        d = np.linalg.norm(Z @ self._W[k] - self._b[k], axis=1)
        d[~np.any(X != 0, axis=1)] = np.nan
        return d

    def predict_into(self, X, labels_out, score_out, work):
        """Project, whiten per class (one product with ``(p, K p)``), minimise ``d^2 + log det``."""
        shp = X.shape[:-1]
        k_all = len(self._W)
        p = self._P32.shape[1]
        Z = _scratch(work, "Z", shp + (p,))
        np.matmul(X, self._P32, out=Z)
        np.subtract(Z, self._mu0P32, out=Z)
        Y = _scratch(work, "Y", shp + (k_all * p,))
        np.matmul(Z, self._Wcat, out=Y)
        np.subtract(Y, self._bcat, out=Y)
        Y4 = Y.reshape(shp + (k_all, p))
        d2 = _scratch(work, "d2", shp + (k_all,))
        np.einsum("...kp,...kp->...k", Y4, Y4, out=d2)
        G = _scratch(work, "G", shp + (k_all,))
        np.add(d2, self._logdet32, out=G)
        _argbest(G, labels_out, work, largest=False)
        idx = labels_out.astype(np.intp)[..., None]
        dbest = np.take_along_axis(d2, idx, axis=-1)[..., 0]
        np.sqrt(dbest, out=score_out)
        s2 = _sq_norm(X, work)
        _mark_invalid(_nonzero_finite(s2) & np.isfinite(score_out), labels_out, score_out)


# --------------------------------------------------------------------------- registry and fit


def _fit_sam(classes, params, *, cancel=None):
    """Fitter of 'sam'."""
    return _SamModel(classes, params)


def _fit_scm(classes, params, *, cancel=None):
    """Fitter of 'scm'."""
    return _ScmModel(classes, params)


def _fit_euclidean(classes, params, *, cancel=None):
    """Fitter of 'euclidean'."""
    return _EuclideanModel(classes, params)


def _fit_euclidean_norm(classes, params, *, cancel=None):
    """Fitter of 'euclidean_norm'."""
    return _EuclideanNormModel(classes, params)


def _fit_sid(classes, params, *, cancel=None):
    """Fitter of 'sid'."""
    return _SidModel(classes, params)


def _fit_mahalanobis(classes, params, *, cancel=None):
    """Fitter of 'mahalanobis'."""
    return _MahalanobisModel(classes, params, cancel)


def _fit_gaussian_ml(classes, params, *, cancel=None):
    """Fitter of 'gaussian_ml'."""
    return _GaussianMlModel(classes, params, cancel)


METHODS: dict = {}


def register_method(info: MethodInfo, fitter: Optional[Callable] = None) -> MethodInfo:
    """Add or replace a method in :data:`METHODS` (``modules.cube_viewer.ml`` uses this).

    Args:
        info: The description; ``family`` and ``score_kind`` must be valid.
        fitter: ``fitter(classes, params, *, cancel=None) -> model``; when given it replaces
            ``info.fitter``.  A model implements the protocol described in :func:`fit`.

    Returns:
        The stored :class:`MethodInfo`.
    """
    if info.family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {info.family!r}")
    if info.score_kind not in SCORE_KINDS:
        raise ValueError(f"score_kind must be one of {SCORE_KINDS}, got {info.score_kind!r}")
    if fitter is not None:
        info = dataclasses.replace(info, fitter=fitter)
    METHODS[info.key] = info
    return info


for _info, _fitter in (
    (MethodInfo("sam", "SAM (spectral angle)", "prototype", "angle_deg", {}, True,
                "Angle between the pixel and each class mean. Ignores brightness; "
                "needs data without a dark offset."), _fit_sam),
    (MethodInfo("scm", "SCM (spectral correlation)", "prototype", "angle_deg", {}, True,
                "SAM on mean-centred spectra (Pearson correlation). Ignores brightness and "
                "an additive offset."), _fit_scm),
    (MethodInfo("euclidean", "Euclidean (raw DN)", "prototype", "distance", {}, True,
                "Distance to each class mean in cube units. Sensitive to brightness "
                "(illumination fall-off)."), _fit_euclidean),
    (MethodInfo("euclidean_norm", "Euclidean (brightness-normalised)", "prototype", "distance",
                {}, True, "Distance after dividing every spectrum by its mean level."),
     _fit_euclidean_norm),
    (MethodInfo("sid", "SID (information divergence)", "prototype", "divergence", {}, True,
                "Spectral information divergence of the spectra as distributions. "
                "About 5x slower than SAM."), _fit_sid),
    (MethodInfo("mahalanobis", "Mahalanobis / LDA (PCA + shrinkage)", "statistical", "distance",
                {"pca_dims": 20}, True,
                "Mahalanobis distance with one pooled covariance (linear discriminant) after "
                "PCA, Ledoit-Wolf shrinkage."), _fit_mahalanobis),
    (MethodInfo("gaussian_ml", "Gaussian ML (QDA, PCA + shrinkage)", "statistical", "distance",
                {"pca_dims": 10}, True,
                "Gaussian maximum likelihood with one covariance per class after PCA, "
                "Ledoit-Wolf shrinkage."), _fit_gaussian_ml),
    (MethodInfo("random_forest", "Random Forest (scikit-learn)", "ml", "probability", {}, False,
                "Needs scikit-learn (registered by modules.cube_viewer.ml)."), None),
    (MethodInfo("svm", "SVM (scikit-learn)", "ml", "probability", {}, False,
                "Needs scikit-learn (registered by modules.cube_viewer.ml)."), None),
):
    register_method(_info, _fitter)
del _info, _fitter


def fit(method_key: str, classes, params=None, *, cancel=None):
    """Fit a segmentation model.

    Args:
        method_key: A key of :data:`METHODS`.
        classes: :class:`ClassSpec` list (>= 1; all with the same band count).
        params: Overrides of the method's default ``params`` (unknown keys are an error).
        cancel: Optional callable polled by slow fits (``RoiCancelled``).

    Returns:
        A model with: ``method`` (key), ``score_kind``, ``classes``, ``warnings`` (list of
        str), ``predict_chunk(X float32 (..., B)) -> (labels int16, score float32)``
        (``X`` not modified; label -1 / score NaN for unscoreable pixels) and
        ``default_thresholds() -> (K,) float64``.  The numpy models also have
        ``predict_into(X, labels_out, score_out, work)``, the in-place fast path used by
        :func:`classify_cube`.

    Raises:
        ValueError: Unknown method or parameter, no classes, mismatched bands, or classes
            the method cannot use (for example an all-zero mean for SAM).
        RuntimeError: The method is listed but not available.
    """
    info = METHODS.get(method_key)
    if info is None:
        raise ValueError(f"unknown method {method_key!r}; known: {sorted(METHODS)}")
    if not info.available or info.fitter is None:
        raise RuntimeError(f"{info.label} is not available: {info.description}")
    classes = list(classes)
    if not classes:
        raise ValueError("fit needs at least one class")
    n_bands = {int(np.asarray(c.mean).size) for c in classes}
    if len(n_bands) != 1:
        raise ValueError(f"classes have different band counts: {sorted(n_bands)}")
    merged = dict(info.params)
    for key, value in dict(params or {}).items():
        if key not in merged:
            raise ValueError(f"{info.label}: unknown parameter {key!r}; "
                             f"expected {sorted(merged) or 'none'}")
        merged[key] = value
    _check_cancel(cancel)
    return info.fitter(classes, merged, cancel=cancel)


def _model_score_kind(model) -> str:
    """Score kind of a model (its attribute, else from :data:`METHODS`)."""
    kind = getattr(model, "score_kind", None)
    if kind in SCORE_KINDS:
        return kind
    info = METHODS.get(getattr(model, "method", None))
    return info.score_kind if info is not None else "probability"


# --------------------------------------------------------------------------- classification


def _row_buffer(rows: int, width: int, n_bands: int):
    """C-contiguous float32 ``(rows, wpad, n_bands)``, ``wpad`` = ``width`` rounded up to 16 pixels.

    With 16 pixels per padding step every image row starts on a 64-byte boundary, so each
    row-wise BLAS call sees the same size and alignment whatever the chunking; the buffer
    stays contiguous, so in-place ufuncs need no hidden copies (a strided view made numpy
    copy the 10 MB chunk and ran 6x slower, measured).
    """
    wpad = -(-max(width, 1) // 16) * 16
    size = rows * wpad * n_bands
    flat = np.zeros(size + 16, dtype=np.float32)
    start = (-(flat.ctypes.data // 4)) % 16
    return flat[start:start + size].reshape(rows, wpad, n_bands)


def classify_cube(cube, model, *, chunk_rows: int = 8, cancel=None, progress=None) -> SegmentResult:
    """Best class and score of every pixel of the cube, in row chunks.

    Args:
        cube: ``(H, W, B)`` array (ndarray, memmap or non-contiguous view); never copied
            whole: ``chunk_rows`` image rows at a time are converted into one reused
            float32 buffer.
        model: A fitted model (:func:`fit`).  Models without ``predict_into`` are called
            with ``predict_chunk`` once per image row.
        chunk_rows: Image rows per chunk (>= 1).  The result does not depend on it.
        cancel: Optional callable polled before every chunk (and every row for models
            without ``predict_into``); True stops with ``RoiCancelled``.
        progress: Optional ``progress(fraction)`` called after every chunk (0 < f <= 1).

    Returns:
        :class:`SegmentResult` (thresholds not applied).

    Raises:
        ValueError: Bad cube or band count different from the model's.
        RoiCancelled: ``cancel()`` returned True.
    """
    h_img, w_img, n_bands = _cube_shape(cube)
    model_bands = int(np.asarray(model.classes[0].mean).size)
    if model_bands != n_bands:
        raise ValueError(f"the model has {model_bands} bands, the cube {n_bands}")
    t0 = time.perf_counter()
    labels = np.empty((h_img, w_img), dtype=np.int16)
    score = np.empty((h_img, w_img), dtype=np.float32)
    rows = max(1, min(int(chunk_rows), max(h_img, 1)))
    buf = _row_buffer(rows, w_img, n_bands)
    padded = buf.shape[1] != w_img
    if padded:
        lab_buf = np.empty(buf.shape[:2], dtype=np.int16)
        score_buf = np.empty(buf.shape[:2], dtype=np.float32)
    fast = callable(getattr(model, "predict_into", None))
    # unsigned data needs no clipping at 0 (SID)
    work: dict = {"nonnegative": np.dtype(cube.dtype).kind in "bu"}
    for r0 in range(0, h_img, rows):
        _check_cancel(cancel)
        r1 = min(h_img, r0 + rows)
        n = r1 - r0
        X = buf[:n]
        np.copyto(X[:, :w_img], cube[r0:r1], casting="unsafe")
        if fast:
            # Each matrix product inside runs once per image row (stacked matmul), so the
            # result does not depend on the chunking.  Padding pixels are zeros (unscored).
            with np.errstate(all="ignore"):              # NaN / inf / zero pixels are expected
                if padded:
                    X[:, w_img:] = 0.0
                    model.predict_into(X, lab_buf[:n], score_buf[:n], work)
                    labels[r0:r1] = lab_buf[:n, :w_img]
                    score[r0:r1] = score_buf[:n, :w_img]
                else:
                    model.predict_into(X, labels[r0:r1], score[r0:r1], work)
        else:
            for i in range(n):
                if i:
                    _check_cancel(cancel)
                lab, sc = model.predict_chunk(X[i, :w_img])
                labels[r0 + i] = lab
                score[r0 + i] = sc
        if progress is not None:
            progress(r1 / h_img)
    return SegmentResult(
        method=str(getattr(model, "method", type(model).__name__)),
        class_names=tuple(c.name for c in model.classes),
        colors=tuple(c.color for c in model.classes),
        labels=labels, score=score,
        default_thresholds=np.asarray(model.default_thresholds(), dtype=np.float64).copy(),
        elapsed_s=time.perf_counter() - t0,
        score_kind=_model_score_kind(model))


def apply_thresholds(result: SegmentResult, thresholds, score_kind: Optional[str] = None) -> np.ndarray:
    """Labels with rejected pixels set to -1, from the stored maps (no recompute).

    A pixel keeps its label when ``score <= thresholds[label]`` (``>=`` for
    ``'probability'``); this is evaluated exactly as in float64 (the float32 maps are
    compared with the thresholds rounded outward to float32).  ``inf`` (``-inf`` /
    ``0`` for probabilities) keeps every scored pixel of the class; NaN rejects all.

    Args:
        result: A :class:`SegmentResult`.
        thresholds: ``(K,)`` per-class thresholds (for example
            ``multiplier * result.default_thresholds``).
        score_kind: Overrides ``result.score_kind``.

    Returns:
        ``(H, W)`` int16, ``-1`` = unclassified.
    """
    kind = score_kind or result.score_kind
    if kind not in SCORE_KINDS:
        raise ValueError(f"score_kind must be one of {SCORE_KINDS}, got {kind!r}")
    thr = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if thr.size != result.n_classes:
        raise ValueError(f"expected {result.n_classes} thresholds, got {thr.size}")
    ok = _accept(result.labels, result.score, thr, kind)
    return np.where(ok, result.labels, np.int16(UNCLASSIFIED)).astype(np.int16, copy=False)


def class_counts(labels, n_classes: int):
    """Pixels per class and unclassified pixels of a label map.

    Returns:
        ``(counts (K,) int64, n_unclassified int)``.

    Raises:
        ValueError: A label outside ``-1 .. K - 1``.
    """
    lab = np.asarray(labels).ravel()
    k = int(n_classes)
    if lab.size and (int(lab.min()) < -1 or int(lab.max()) >= k):
        raise ValueError(f"labels must lie in -1..{k - 1}")
    counts = np.bincount(lab + 1, minlength=k + 1).astype(np.int64)
    return counts[1:], int(counts[0])


# --------------------------------------------------------------------------- separability


def _jm_matrix(classes):
    """``(jm (K, K), dims)``: Jeffries-Matusita distance after PCA-``JM_DIMS``, Ledoit-Wolf covariances."""
    k_all = len(classes)
    P, mu0 = _pca_basis(classes, JM_DIMS)
    mus, covs, logdets = [], [], []
    for c in classes:
        X = np.asarray(c.sample, dtype=np.float64)
        mus.append((np.asarray(c.mean, dtype=np.float64) - mu0) @ P)
        if X.shape[0] < _JM_MIN_PX:
            covs.append(None)
            logdets.append(math.nan)
            continue
        S = ledoit_wolf((X - mu0) @ P)[0]
        L = _cholesky_pd(S)
        if L is None:
            covs.append(None)
            logdets.append(math.nan)
            continue
        covs.append(L @ L.T)
        logdets.append(2.0 * float(np.log(np.diag(L)).sum()))
    jm = np.zeros((k_all, k_all))
    for i in range(k_all):
        for j in range(i + 1, k_all):
            value = math.nan
            if covs[i] is not None and covs[j] is not None:
                S = 0.5 * (covs[i] + covs[j])
                L = np.linalg.cholesky(S)
                d = np.linalg.solve(L, mus[i] - mus[j])
                logdet = 2.0 * float(np.log(np.diag(L)).sum())
                bhat = 0.125 * float(d @ d) + 0.5 * (logdet - 0.5 * (logdets[i] + logdets[j]))
                value = min(2.0, max(0.0, 2.0 * (1.0 - math.exp(-bhat))))
            jm[i, j] = jm[j, i] = value
    return jm, P.shape[1]


def _anova_f(classes) -> np.ndarray:
    """One-way ANOVA F per band on the class samples (NaN where undefined)."""
    n_bands = int(np.asarray(classes[0].mean).size)
    ns, means, ssw = [], [], np.zeros(n_bands)
    for c in classes:
        X = np.asarray(c.sample, dtype=np.float64)
        if X.shape[0] == 0:
            continue
        m = X.mean(axis=0)
        ssw += ((X - m) ** 2).sum(axis=0)
        ns.append(X.shape[0])
        means.append(m)
    k_used, n_tot = len(ns), int(sum(ns))
    if k_used < 2 or n_tot <= k_used:
        return np.full(n_bands, np.nan)
    ns_arr = np.array(ns, dtype=np.float64)
    M = np.stack(means)
    grand = (ns_arr[:, None] * M).sum(axis=0) / n_tot
    ssb = (ns_arr[:, None] * (M - grand) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (ssb / (k_used - 1)) / (ssw / (n_tot - k_used))


def _subset_class(c: ClassSpec, keep) -> ClassSpec:
    """Class ``c`` restricted to the sample pixels ``keep`` (mean = sample mean)."""
    sample = c.sample[keep]
    rids = c.sample_rids[keep]
    return dataclasses.replace(
        c, rids=tuple(int(r) for r in dict.fromkeys(rids.tolist())), n_px=int(sample.shape[0]),
        mean=sample.astype(np.float64).mean(axis=0), sample=sample, sample_rids=rids, warnings=[])


def _roi_consistency(classes, method, params, cancel):
    """Confusion of held-out ROI pixels: leave-one-ROI-out, else split-half per class."""
    k_all = len(classes)
    conf = np.zeros((k_all, k_all), dtype=np.int64)
    rejected = np.zeros(k_all, dtype=np.int64)
    modes = []
    for k, c in enumerate(classes):
        n = c.sample.shape[0]
        rois = list(dict.fromkeys(c.sample_rids.tolist()))
        if len(rois) >= 2:
            modes.append("leave-one-ROI-out")
            folds = [(c.sample_rids != r, c.sample_rids == r) for r in rois]
        elif n >= 2:
            modes.append("split-half")
            first = np.arange(n) < n // 2
            folds = [(~first, first), (first, ~first)]
        else:
            modes.append("too few pixels")
            continue
        for train, test in folds:
            _check_cancel(cancel)
            fold = list(classes)
            fold[k] = _subset_class(c, train)
            model = fit(method, fold, params, cancel=cancel)
            lab, sc = model.predict_chunk(c.sample[test])
            scored = lab >= 0
            conf[k] += np.bincount(lab[scored].astype(np.intp), minlength=k_all)
            ok = _accept(lab, sc, model.default_thresholds(), _model_score_kind(model))
            rejected[k] += int(np.count_nonzero(~ok))
    return conf, rejected, modes


def _offset_estimates(classes) -> np.ndarray:
    """Per class: estimated additive offset (cube units) or NaN when not estimable.

    Each sample pixel is fitted as ``x = a m + b`` (``m`` = class mean).  With a pure gain
    ``s`` and an offset ``c``, ``a = s`` and ``b = c (1 - s)``, so ``c`` is minus the slope of
    ``b`` against ``a``; the slope is corrected for the noise covariance of ``(a, b)``.
    """
    out = np.full(len(classes), np.nan)
    for k, c in enumerate(classes):
        X = np.asarray(c.sample, dtype=np.float64)
        n, n_bands = X.shape
        if n < _OFFSET_MIN_PX or n_bands < 3:
            continue
        m = np.asarray(c.mean, dtype=np.float64)
        mm, m1 = float(m @ m), float(m.sum())
        det = mm * n_bands - m1 * m1
        if det <= 1e-12 * mm * n_bands:
            continue                                       # flat mean: gain and offset mix
        xm, x1 = X @ m, X.sum(axis=1)
        a = (n_bands * xm - m1 * x1) / det
        b = (mm * x1 - m1 * xm) / det
        resid = X - a[:, None] * m - b[:, None]
        sigma2 = float(np.mean(np.einsum("ij,ij->i", resid, resid))) / (n_bands - 2)
        var_a = float(np.var(a)) - n_bands * sigma2 / det
        cov_ab = float(np.mean((a - a.mean()) * (b - b.mean()))) + m1 * sigma2 / det
        if var_a <= 0 or math.sqrt(var_a) < _OFFSET_MIN_SPREAD * abs(float(a.mean())):
            continue
        offset = -cov_ab / var_a
        level = float(m.mean())
        resid_b = (b - b.mean()) + offset * (a - a.mean())
        se = math.sqrt(float(np.mean(resid_b ** 2)) / max(n - 2, 1) / var_a)
        if abs(offset) > _OFFSET_MIN_LEVEL * abs(level) and abs(offset) > 3.0 * se:
            out[k] = offset
    return out


def separability(classes, *, method: str = "sam", params=None, cancel=None) -> dict:
    """How different the classes are ('how different are the legends').

    Args:
        classes: :class:`ClassSpec` list (>= 1).
        method: Method of the ROI-consistency confusion (default SAM).
        params: Its parameters.
        cancel: Optional callable polled per fold (``RoiCancelled``).

    Returns:
        dict with ``angle_deg`` (K, K) mean-to-mean spectral angle; ``jm`` (K, K) JM
        distance 0-2 (NaN for classes with < 3 sample pixels); ``jm_dims``; ``f_curve``
        (B,) ANOVA F; ``confusion`` (K, K) int64 (rows true, columns best class);
        ``confusion_mode`` ('leave-one-ROI-out', 'split-half' or 'mixed: ...');
        ``confusion_modes`` (per class); ``confusion_rejected`` (K,) held-out pixels the
        fold's default thresholds would reject; ``confusion_note``; ``sam_thresholds_deg``
        (K,); ``offset_estimate`` (K,) cube units (NaN = none detected);
        ``class_names``; ``warnings`` list of str.
    """
    classes = list(classes)
    if not classes:
        raise ValueError("separability needs at least one class")
    names = [c.name for c in classes]
    angle = spectral_angle_matrix(_means(classes))
    _check_cancel(cancel)
    jm, jm_dims = _jm_matrix(classes)
    f_curve = _anova_f(classes)
    conf, rejected, modes = _roi_consistency(classes, method, params, cancel)
    sam_thr = _SamModel(classes, {}).default_thresholds()
    offsets = _offset_estimates(classes)

    warnings = [w for c in classes for w in c.warnings]
    k_all = len(classes)
    for i in range(k_all):
        for j in range(i + 1, k_all):
            closer = [n for n, t in ((i, sam_thr[i]), (j, sam_thr[j])) if angle[i, j] < t]
            if closer:
                who = " and ".join(f"'{names[n]}' ({sam_thr[n]:.2f} deg)" for n in closer)
                warnings.append(f"'{names[i]}' and '{names[j]}' are only {angle[i, j]:.2f} deg apart, "
                                f"within the SAM threshold of {who}: pixels of one may be "
                                f"classified as the other.")
    if np.isnan(jm[np.triu_indices(k_all, 1)]).any():
        few = [names[k] for k, c in enumerate(classes) if c.sample.shape[0] < _JM_MIN_PX]
        if few:
            warnings.append(f"No JM distance for {', '.join(repr(n) for n in few)}: fewer than "
                            f"{_JM_MIN_PX} pixels.")
    found = np.flatnonzero(np.isfinite(offsets))
    if found.size:
        parts = ", ".join(f"{offsets[k]:+.0f} in '{names[k]}'" for k in found)
        warnings.append(f"Possible additive offset (dark level not subtracted?): about {parts} "
                        f"(cube units; the spectral shape changes with brightness). SAM, SID and "
                        f"Euclidean (brightness-normalised) assume no offset; SCM ignores it.")
    used = sorted(set(m for m in modes if m != "too few pixels"))
    if len(used) == 1:
        mode = used[0]
    elif not used:
        mode = "none"
    else:
        mode = "mixed: " + "; ".join(
            f"{m} for {', '.join(names[k] for k in range(k_all) if modes[k] == m)}" for m in used)
    return {
        "class_names": names,
        "angle_deg": angle,
        "jm": jm,
        "jm_dims": jm_dims,
        "f_curve": f_curve,
        "confusion": conf,
        "confusion_mode": mode,
        "confusion_modes": modes,
        "confusion_rejected": rejected,
        "confusion_note": ("ROI consistency, not accuracy: held-out pixels come from the same "
                           "images and are correlated with the training pixels (optimistic)."),
        "sam_thresholds_deg": sam_thr,
        "offset_estimate": offsets,
        "warnings": warnings,
    }
