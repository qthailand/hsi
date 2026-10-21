"""Compare segmentation methods on the ROI pixels: held-out accuracy, macro-F1 and timings (numpy).

``compare_methods`` answers 'which method should I use, and what does a full run cost?'
for the Segment dialog's 'Compare methods' table, before anything is run on the image:

* **Accuracy / macro-F1** on held-out ROI pixels.  Folds per class: leave-one-ROI-out when
  the class has >= 2 ROIs in its sample (ROIs beyond ``max_folds`` are grouped
  round-robin into ``max_folds`` folds), otherwise split-half (raster-order first / second
  half of its sample, as ``segment.separability``); a class with a single pixel is always
  trained on and never tested.  Round ``r`` holds out fold ``r`` of every class at once
  and trains one model on the rest, so a method is fitted ``max(folds)`` times (not once
  per ROI).  Every held-out pixel is predicted exactly once; the best-class label is
  scored (thresholds not applied; ``reject_rate`` = share the fold's default thresholds
  would reject).  Neighbouring pixels are correlated, so split-half in particular is
  optimistic.
* **Timings**: ``fit_s`` = mean fit time of the folds (each fit sees at least half the
  pixels, so a fit on all of them takes somewhat longer); ``predict_us_per_px`` =
  ``segment.classify_cube`` on a small cube of tiled sample pixels (about 50,000 px) with
  the image's row width - the real code path - as the slope between a third and all of
  its rows (best of 2 each), so the fixed cost per call does not count per pixel;
  ``est_full_image_s`` = fixed cost + that rate x the image's pixel count
  (classification only, without the fit).

``estimate_seconds`` predicts fit + full classification *before* fitting, from the rates
measured below (``model=`` refines the RF / SVM terms with the fitted forest depth /
support-vector count).

Results are deterministic: fixed folds, the methods' fixed seeds; only the timings vary.

Validation on real data: SWIR reflectance of the pipeline's Cashew nuts S01 (Moist / Dry,
2 ground-truth components each -> leave-one-ROI-out, 5000 px per class) against the
accuracy of the same methods on the held-out image S02 (94,865 GT px): Mahalanobis 0.990
(S02: 0.995), Gaussian ML 0.979 (0.978), RF 0.972 (0.970), SVM 0.969 (0.978), SID 0.887
(0.891), Euclidean-norm 0.881 (0.887), SAM 0.878 (0.884), SCM 0.816 (0.825), Euclidean
0.725 (0.733): every method within 0.01, nearly the same order.  All 9 methods took 2.4 s
there and 7.6-8.2 s for K = 8 synthetic classes of 5000 px (3 ROIs each).

Measured rates (i9-12900K, 8 P + 8 E cores, 34 GB, Windows 11, Python 3.13, numpy 2.1.3
with MKL, scikit-learn 1.6.1, 2026-10-21; ``segment.fit`` + ``segment.classify_cube`` of a
synthetic in-RAM 1024 x 1762 x 185 uint16 cube, 5000 training px per class from 3 ROIs per
class, default parameters, 3 runs; multiply us / px by 1.8 for seconds per full image):

    ==================  =========================  ==========================
    method              classify us / px, K=2/5/8  fit s, K = 2 / 5 / 8
    ==================  =========================  ==========================
    sam                 0.107 / 0.112 / 0.121      0.01 / 0.02-0.03 / 0.04-0.05
    scm                 0.154 / 0.159 / 0.173      0.013 / 0.035 / 0.065
    euclidean           0.110 / 0.112 / 0.114      0.012 / 0.03 / 0.05
    euclidean_norm      0.130 / 0.134 / 0.140      0.015 / 0.04 / 0.07
    sid                 0.457 / 0.464 / 0.505      0.03 / 0.08 / 0.14
    mahalanobis         0.131 / 0.147 / 0.163      0.04 / 0.08 / 0.12-0.14
    gaussian_ml         0.152 / 0.176 / 0.204      0.035 / 0.075 / 0.12-0.14
    random_forest       0.78-0.84 / 0.80-0.90 /    0.36-0.37 / 0.65-0.68 /
    (100 trees)         0.91-1.08 (depth 7/10/20)  1.32-1.57
    svm (RBF, PCA-10)   2.0-3.8 / 3.5-8.0 /        0.09-0.11 / 0.35-0.37 /
                        3.3-11.6 (266/487/437 SV)  0.52-0.83
    svm (RBF, PCA-20)   3.3 / - / 10.6 (492/1706)  0.15 / - / 1.45
    svm (linear)        0.57 / - / 0.81            0.02-0.04 / - / 0.10
    ==================  =========================  ==========================

    Ranges are runs on a quiet machine and runs while other processes were busy (low /
    high end); the multi-threaded SVM varies most (up to 3.5x), the numpy methods by 10-40
    % (SID up to 2x).  A BSQ / BIL
    memory map costs about 0.7 s more per full image (strided reads; ``segment``).
    ``estimate_seconds`` uses fits to this table (``_RATES`` and the ``_RF_*`` /
    ``_SVM_*`` constants): 0.90-1.10 x the measured fit + classification of every row
    except the SVM before fitting, 1.1-1.45 x (on the safe side): its support-vector count
    is guessed as ``0.012 x pca_dims x n_train`` (measured 0.055-0.25 x n_train on the
    synthetic and the real data).

Threading:
    Pure computation for a worker thread.  ``cancel()`` is polled between folds,
    prediction chunks and methods (and inside the fits); it raises
    :class:`~modules.cube_viewer.roi_stats.RoiCancelled`.  ``progress(fraction)`` is called
    after every fold and timing step.

Example:
    >>> rows = compare_methods(classes, ["sam", "mahalanobis", "random_forest"])  # doctest: +SKIP
    >>> [(r["method"], round(r["accuracy"], 3), r["mode"]) for r in rows]        # doctest: +SKIP
"""

from __future__ import annotations

import dataclasses
import math
import time

import numpy as np

from . import ml as _ml
from . import segment as _segment
from .roi_stats import RoiCancelled

__all__ = ("DEFAULT_IMAGE_SHAPE", "MAX_FOLDS", "cv_folds", "compare_methods", "macro_f1",
           "estimate_seconds")

#: (H, W) assumed when the image size is not given: a full camera frame.
DEFAULT_IMAGE_SHAPE = (1024, 1762)
#: Most folds per class (more ROIs are grouped round-robin).
MAX_FOLDS = 5
# Pixels of the timing cube (about 50 k: per-call overheads are below 2 %).
_TIMING_PX = 50_000
# Rows of one prediction call on the held-out pixels.
_PREDICT_ROWS = 8 * 1762
_TOO_FEW = "too few pixels"

# Measured classify cost in us per pixel at B = 185 bands: (a, b) -> a + b * K
# (prototype / statistical: scaled by n_bands / 185), and fit seconds (c, d) -> c + d * K.
_RATES = {
    "sam": ((0.100, 0.0023), (0.0, 0.006)),
    "scm": ((0.147, 0.0032), (0.0, 0.008)),
    "euclidean": ((0.109, 0.0007), (0.0, 0.0065)),
    "euclidean_norm": ((0.127, 0.0017), (0.0, 0.0085)),
    "sid": ((0.440, 0.008), (0.0, 0.017)),
    "mahalanobis": ((0.121, 0.0053), (0.0, 0.016)),
    "gaussian_ml": ((0.135, 0.0087), (0.0, 0.016)),
}
# Random Forest: classify us/px = trees * (_RF_TREE_US + _RF_LEVEL_US * mean depth) + _RF_BASE_US;
# fit s = _RF_FIT_S * trees / 100 * (n_train / 10_000) ** 1.1 (8 threads).  A priori depth =
# log2(n_train) + 2 log2(K) (measured 7.2 / 9.9 / 20 for K = 2 / 5 / 8).
_RF_TREE_US = 0.0052
_RF_LEVEL_US = 0.00015
_RF_BASE_US = 0.15
_RF_FIT_S = 0.3
# SVM (RBF): classify us/px = _SVM_BASE_US + n_support * (_SVM_SV_US + _SVM_SV_DIM_US * d)
# (8 threads, d = features); a priori n_support = _SVM_SV_FRACTION * d * n_train.
# fit s = _SVM_FIT_S * (n_train / 8000) ** 1.3.  Linear: (a, b) -> a + b K us/px, fit
# _SVM_LINEAR_FIT_S * K s.
_SVM_BASE_US = 0.4
_SVM_SV_US = 0.0072
_SVM_SV_DIM_US = 0.00005
_SVM_SV_FRACTION = 0.012
_SVM_FIT_S = 0.55
_SVM_LINEAR_US = (0.49, 0.04)
_SVM_LINEAR_FIT_S = 0.013
_REF_BANDS = 185


def _class_folds(spec, max_folds: int):
    """``(fold (n,) int, n_folds, mode)`` of one class's sample pixels (-1 = never held out)."""
    rids = np.asarray(spec.sample_rids)
    n = int(np.asarray(spec.sample).shape[0])
    present = set(rids.tolist())
    rois = [r for r in spec.rids if r in present]
    rois += [r for r in dict.fromkeys(rids.tolist()) if r not in set(rois)]
    if len(rois) >= 2:
        n_folds = min(len(rois), max_folds)
        position = {r: i for i, r in enumerate(rois)}
        unique, inverse = np.unique(rids, return_inverse=True)
        fold = np.array([position[r] % n_folds for r in unique.tolist()], dtype=np.int64)[inverse]
        mode = ("leave-one-ROI-out" if n_folds == len(rois)
                else f"leave-ROI-group-out ({n_folds} groups)")
        return fold, n_folds, mode
    if n >= 2:
        return (np.arange(n) >= n // 2).astype(np.int64), 2, "split-half"
    return np.full(n, -1, dtype=np.int64), 0, _TOO_FEW


def _overall_mode(names, modes) -> str:
    """One mode string for all classes (as ``segment.separability``'s ``confusion_mode``)."""
    used = sorted(set(m for m in modes if m != _TOO_FEW))
    if not used:
        return "none"
    if len(used) == 1:
        return used[0]
    return "mixed: " + "; ".join(
        f"{m} for {', '.join(n for n, mm in zip(names, modes) if mm == m)}" for m in used)


def cv_folds(classes, *, max_folds: int = MAX_FOLDS):
    """Cross-validation folds of the class samples.

    Returns:
        dict with ``folds`` (list of int64 ``(n_k,)`` fold index per sample pixel, -1 =
        always trained on), ``n_rounds`` (max folds of a class), ``modes`` (per class:
        'leave-one-ROI-out', 'leave-ROI-group-out (k groups)', 'split-half' or
        'too few pixels') and ``mode`` (one string for all).
    """
    if max_folds < 2:
        raise ValueError(f"max_folds must be >= 2, got {max_folds}")
    folds, counts, modes = [], [], []
    for spec in classes:
        fold, n_folds, mode = _class_folds(spec, max_folds)
        folds.append(fold)
        counts.append(n_folds)
        modes.append(mode)
    return {"folds": folds, "n_rounds": max(counts, default=0), "modes": modes,
            "mode": _overall_mode([c.name for c in classes], modes)}


def _subset(spec, keep):
    """``spec`` restricted to the sample pixels ``keep`` (mean = their float64 mean)."""
    sample = np.ascontiguousarray(np.asarray(spec.sample)[keep])
    rids = np.asarray(spec.sample_rids)[keep]
    return dataclasses.replace(
        spec, rids=tuple(int(r) for r in dict.fromkeys(rids.tolist())), n_px=int(sample.shape[0]),
        mean=sample.astype(np.float64).mean(axis=0), sample=sample, sample_rids=rids, warnings=[])


def macro_f1(y_true, y_pred, n_classes: int):
    """``(macro_f1, per_class_f1 (K,))`` over the classes present in ``y_true``.

    ``y_pred`` may hold -1 (unscored): it counts as a miss of the true class.  F1 of a
    class = ``2 TP / (2 TP + FP + FN)``; classes absent from ``y_true`` get NaN and are
    left out of the mean (scikit-learn's ``f1_score(average='macro', labels=present)``).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    f1 = np.full(n_classes, np.nan)
    for k in range(n_classes):
        is_true = y_true == k
        if not is_true.any():
            continue
        is_pred = y_pred == k
        tp = int(np.count_nonzero(is_true & is_pred))
        denom = int(np.count_nonzero(is_true)) + int(np.count_nonzero(is_pred))   # 2TP + FP + FN
        f1[k] = 2.0 * tp / denom if denom else 0.0
    present = ~np.isnan(f1)
    return (float(f1[present].mean()) if present.any() else math.nan), f1


def _check_cancel(cancel) -> None:
    """Raise ``RoiCancelled`` when ``cancel()`` returns True."""
    if cancel is not None and cancel():
        raise RoiCancelled()


def _score_kind(model) -> str:
    """The model's score kind (its attribute, else from ``segment.METHODS``)."""
    kind = getattr(model, "score_kind", None)
    if kind in _segment.SCORE_KINDS:
        return kind
    return _segment.METHODS[model.method].score_kind


def _predict(model, X, cancel):
    """Labels / scores of ``X (n, B)`` in chunks of ``_PREDICT_ROWS``, polling ``cancel``."""
    labels = np.empty(X.shape[0], dtype=np.int16)
    score = np.empty(X.shape[0], dtype=np.float32)
    for r0 in range(0, X.shape[0], _PREDICT_ROWS):
        _check_cancel(cancel)
        labels[r0:r0 + _PREDICT_ROWS], score[r0:r0 + _PREDICT_ROWS] = \
            model.predict_chunk(X[r0:r0 + _PREDICT_ROWS])
    return labels, score


def _timing_cube(classes, width: int) -> np.ndarray:
    """float32 ``(rows, width, B)`` of the class samples tiled in order (about ``_TIMING_PX`` px)."""
    pixels = np.concatenate([np.asarray(c.sample, dtype=np.float32) for c in classes])
    rows = max(1, -(-_TIMING_PX // width))
    return np.resize(pixels, (rows * width, pixels.shape[1])).reshape(rows, width, pixels.shape[1])


def _time_classify(model, timing: np.ndarray, cancel):
    """``(seconds per pixel, fixed seconds per call)`` of ``segment.classify_cube``.

    The slope between the first third and all rows of the timing cube (best of 2 each),
    so the fixed cost of a call (buffers, thread wake-up: about 2.5 ms) does not inflate
    the per-pixel rate of the fast methods (SAM: 0.33 us/px on 8 rows, 0.12 on 64).
    """
    rows, width = timing.shape[:2]
    parts = sorted({max(1, rows // 3), rows})
    best = {}
    for part in parts:
        best[part] = math.inf
        for _ in range(2):
            _check_cancel(cancel)
            t0 = time.perf_counter()
            _segment.classify_cube(timing[:part], model, cancel=cancel)
            best[part] = min(best[part], time.perf_counter() - t0)
    if len(parts) == 2:
        slope = (best[parts[1]] - best[parts[0]]) / ((parts[1] - parts[0]) * width)
        if slope > 0:
            return slope, max(0.0, best[parts[0]] - slope * parts[0] * width)
    return best[rows] / (rows * width), 0.0


def _rejected(model, labels, score) -> np.ndarray:
    """Bool: held-out pixels the model's default thresholds would reject (or unscored)."""
    result = _segment.SegmentResult(
        method=str(model.method), class_names=tuple(c.name for c in model.classes),
        colors=tuple(c.color for c in model.classes), labels=labels[None, :], score=score[None, :],
        default_thresholds=np.asarray(model.default_thresholds(), dtype=np.float64),
        elapsed_s=0.0, score_kind=_score_kind(model))
    return _segment.apply_thresholds(result, result.default_thresholds)[0] < 0


def _empty_row(key: str, mode: str, n_rounds: int) -> dict:
    """A result row with every metric unset."""
    info = _segment.METHODS.get(key)
    return {"method": key, "label": info.label if info is not None else key,
            "accuracy": None, "macro_f1": None, "fit_s": None, "predict_us_per_px": None,
            "est_full_image_s": None, "mode": mode, "n_folds": n_rounds, "n_test_px": 0,
            "reject_rate": None, "per_class_f1": None, "confusion": None, "error": None}


def compare_methods(classes, method_keys, *, params=None, cancel=None, progress=None,
                    image_shape=None, max_folds: int = MAX_FOLDS) -> list:
    """Cross-validated accuracy, macro-F1 and timings of each method on the class samples.

    Args:
        classes: ``segment.ClassSpec`` list (from ``segment.build_classes``).
        method_keys: Keys of ``segment.METHODS`` in table order.
        params: Optional ``{method_key: params dict}`` (defaults otherwise).
        cancel: Optional ``cancel() -> bool``; True stops with ``RoiCancelled``.
        progress: Optional ``progress(fraction)``.
        image_shape: ``(H, W)`` of the image for ``est_full_image_s`` and the timing row
            width; default :data:`DEFAULT_IMAGE_SHAPE`.
        max_folds: Most folds per class.

    Returns:
        One dict per method: ``method``, ``label``, ``accuracy`` and ``macro_f1`` (0-1, of
        the held-out best-class labels), ``fit_s``, ``predict_us_per_px``,
        ``est_full_image_s``, ``mode`` (see :func:`cv_folds`), ``n_folds``, ``n_test_px``,
        ``reject_rate``, ``per_class_f1`` (list, NaN = class not tested), ``confusion``
        (K x K int64, rows = true class, columns = predicted; unscored pixels left out)
        and ``error`` (None, or why the method could not be evaluated - then the metrics
        are None, e.g. scikit-learn missing or a class the method cannot use).

    Raises:
        ValueError: Fewer than 2 classes or no class with 2 or more sample pixels.
        RoiCancelled: ``cancel()`` returned True.
    """
    classes = list(classes)
    keys = list(method_keys)
    if len(classes) < 2:
        raise ValueError(f"compare needs at least 2 classes, got {len(classes)}")
    height, width = (int(v) for v in (image_shape or DEFAULT_IMAGE_SHAPE))
    folds = cv_folds(classes, max_folds=max_folds)
    n_rounds, mode = folds["n_rounds"], folds["mode"]
    if n_rounds == 0:
        raise ValueError("no class has 2 or more pixels to hold out")
    n_classes = len(classes)
    params = dict(params or {})
    steps = max(1, len(keys) * (n_rounds + 1))
    done = 0
    timing = _timing_cube(classes, width)
    rows = []
    for key in keys:
        _check_cancel(cancel)
        row = _empty_row(key, mode, n_rounds)
        rows.append(row)
        info = _segment.METHODS.get(key)
        if info is None or not info.available:
            row["error"] = (f"unknown method {key!r}" if info is None
                            else f"not available: {info.description}")
            done += n_rounds + 1
            if progress is not None:
                progress(done / steps)
            continue
        try:
            y_true, y_pred, rejected, fit_times, model = [], [], [], [], None
            for r in range(n_rounds):
                test = [fold == r for fold in folds["folds"]]
                train = [spec if not t.any() else _subset(spec, ~t)
                         for spec, t in zip(classes, test)]
                X = np.concatenate([np.asarray(c.sample, dtype=np.float32)[t]
                                    for c, t in zip(classes, test)])
                y = np.concatenate([np.full(int(t.sum()), k, dtype=np.int64)
                                    for k, t in enumerate(test)])
                if X.shape[0]:
                    t0 = time.perf_counter()
                    model = _segment.fit(key, train, params.get(key), cancel=cancel)
                    fit_times.append(time.perf_counter() - t0)
                    labels, score = _predict(model, X, cancel)
                    y_true.append(y)
                    y_pred.append(labels.astype(np.int64))
                    rejected.append(_rejected(model, labels, score))
                done += 1
                if progress is not None:
                    progress(done / steps)
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            row["accuracy"] = float(np.mean(y_pred == y_true))
            row["macro_f1"], per_class = macro_f1(y_true, y_pred, n_classes)
            row["per_class_f1"] = per_class.tolist()
            scored = y_pred >= 0
            pairs = y_true[scored] * n_classes + y_pred[scored]
            row["confusion"] = np.bincount(pairs, minlength=n_classes ** 2).reshape(n_classes, n_classes)
            row["n_test_px"] = int(y_true.size)
            row["reject_rate"] = float(np.mean(np.concatenate(rejected)))
            row["fit_s"] = float(np.mean(fit_times))
            per_px, per_call = _time_classify(model, timing, cancel)
            row["predict_us_per_px"] = per_px * 1e6
            row["est_full_image_s"] = per_call + per_px * height * width
        except RoiCancelled:
            raise
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            for name in ("accuracy", "macro_f1", "fit_s", "predict_us_per_px", "est_full_image_s",
                         "reject_rate", "per_class_f1", "confusion"):
                row[name] = None
            row["n_test_px"] = 0
            row["error"] = f"{type(exc).__name__}: {exc}"
        done = min(steps, (len(rows)) * (n_rounds + 1))
        if progress is not None:
            progress(done / steps)
    return rows


def estimate_seconds(method_key: str, n_pixels: int, n_bands: int, n_classes: int, params=None,
                     *, model=None, n_train=None) -> float:
    """Predicted wall time of fit + ``classify_cube`` of ``n_pixels`` pixels, in seconds.

    From the rates measured on the benchmark machine (module docstring): 0.9-1.1 x the
    measured time there; before fitting, the SVM term is the least certain (its
    support-vector count depends on how much the classes overlap) and up to 1.45 x.
    Scale for a slower PC.

    Args:
        method_key: A key of ``segment.METHODS``.
        n_pixels: Pixels to classify (``H * W``).
        n_bands: Bands per pixel.
        n_classes: K.
        params: The method's parameters (defaults otherwise).
        model: Optional fitted model: RF uses its trees' depth, SVM its support-vector
            count, and the fit term is dropped (already done).
        n_train: Training pixels in total (default ``n_classes * 5000``, the sample cap
            of ``build_classes``).

    Raises:
        ValueError: Unknown method or negative sizes.
    """
    if method_key not in _segment.METHODS:
        raise ValueError(f"unknown method {method_key!r}")
    if n_pixels < 0 or n_bands < 1 or n_classes < 1:
        raise ValueError("n_pixels must be >= 0, n_bands and n_classes >= 1")
    k = int(n_classes)
    n_train = int(n_train) if n_train is not None else 5000 * k
    band_scale = n_bands / _REF_BANDS
    fit_s = 0.0
    if method_key in _RATES:
        (a, b), (c, d) = _RATES[method_key]
        us = (a + b * k) * band_scale
        fit_s = (c + d * k) * band_scale
    elif method_key == "random_forest":
        p = _ml.resolve_params("random_forest", params)
        trees = p["n_estimators"]
        if model is not None:
            depth = float(model.mean_depth)
        else:
            depth = math.log2(max(n_train, 2)) + 2.0 * math.log2(max(k, 2))
            if p["max_depth"]:
                depth = min(depth, p["max_depth"])
        us = trees * (_RF_TREE_US + _RF_LEVEL_US * depth) + _RF_BASE_US * band_scale
        fit_s = _RF_FIT_S * trees / 100.0 * (n_train / 10_000) ** 1.1 * math.sqrt(band_scale)
    elif method_key == "svm":
        p = _ml.resolve_params("svm", params)
        dims = min(p["pca_dims"], n_bands) or n_bands
        n_svm = min(n_train, k * p["max_train_per_class"])
        if p["linear_kernel"]:
            us = (_SVM_LINEAR_US[0] + _SVM_LINEAR_US[1] * k) * band_scale
            fit_s = _SVM_LINEAR_FIT_S * k * band_scale
        else:
            n_sv = (model.n_support if model is not None
                    else min(n_svm, _SVM_SV_FRACTION * dims * n_svm))
            us = _SVM_BASE_US * band_scale + n_sv * (_SVM_SV_US + _SVM_SV_DIM_US * dims)
            fit_s = _SVM_FIT_S * (n_svm / 8000.0) ** 1.3
    else:
        raise ValueError(f"no measured rate for method {method_key!r}")
    if model is not None:
        fit_s = 0.0
    return fit_s + us * n_pixels * 1e-6
