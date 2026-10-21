"""Random Forest and SVM pixel classifiers for the cube viewer's segmentation (scikit-learn, optional).

The two learned methods of the Segment dialog, next to the numpy prototype and
statistical methods of :mod:`modules.cube_viewer.segment`:

* ``'random_forest'`` - :class:`RandomForestModel`: scikit-learn ``RandomForestClassifier``
  on the raw bands (default) or on whitened PCA scores.  Score = the forest's probability
  of the winning class (mean over the trees of the leaf class proportions).
* ``'svm'`` - :class:`SvmModel`: scikit-learn ``SVC`` with an RBF kernel on whitened PCA
  scores (default), or ``LinearSVC`` (``linear_kernel``).  Score = the Platt-calibrated
  probability of the winning class (libsvm's ``predict_proba``; for the linear kernel
  ``CalibratedClassifierCV(method='sigmoid')``).

Importing this module registers both in ``segment.METHODS`` (:func:`register`, through
``segment.register_method``): with scikit-learn they are available and ``segment.fit``
returns these models; without it (``HAVE_SKLEARN`` False) they stay listed but disabled
(``available=False``) and the model constructors raise ``RuntimeError``.

The models implement the segment ``Model`` protocol: ``method``, ``score_kind``
(``'probability'``: higher is better), ``classes``, ``params``, ``warnings``,
``predict_chunk(X)``, ``predict_into(X, labels_out, score_out, work)`` (the fast path of
``segment.classify_cube``) and ``default_thresholds()``.

Data conventions:
    Training pixels are the fixed-seed class samples ``ClassSpec.sample`` (float32
    ``(n, B)``, the cube's own units), labelled by the class's index in ``classes``.  A
    pixel is *invalid* when a band is NaN / +-inf, when all its bands are 0, or (with
    ``normalize``) when its band sum is not positive - as in ``segment``: invalid pixels are
    left out of training and get label ``-1`` and score NaN.  ``predict_chunk`` takes
    pixels ``(..., B)`` (converted to float32, never modified) and returns ``labels`` int16
    and ``score`` float32 of shape ``(...)``, the score in [0, 1].
    :meth:`~_SklearnModel.predict_proba` gives all K probabilities.

Features:
    ``normalize`` divides every pixel by its band mean (brightness normalisation: the
    spectral shape is kept, the level is not).  ``pca_dims = k > 0`` then projects on the
    top ``k`` principal axes of the training pixels and divides by their standard
    deviations (whitening; covariance accumulated in float64 in row blocks,
    ``numpy.linalg.eigh``, the largest entry of every axis made positive); axes with an
    eigenvalue below 1e-10 of the largest are dropped.  With ``k = 0`` the RF uses the raw
    bands and the SVM per-band z-scores.  Training and image pixels go through the same
    float32 transform.

Determinism and chunking:
    Fixed seeds (``seed``, default 0) for the RF trees, the SVM training subsample,
    libsvm's Platt cross-validation and liblinear.  The RF is fitted in batches of
    ``2 n_jobs`` trees with ``warm_start`` (``cancel`` is polled between batches);
    scikit-learn draws the tree seeds exactly as in one fit, so the forest is the same.
    Prediction splits every image row into equal pieces of at most 2048 pixels (a split
    that depends only on the row width) and spreads the pieces over threads; RF
    probabilities are summed per pixel in tree order and libsvm is per pixel, so labels
    and scores are bit-identical for any ``n_jobs``, any ``chunk_rows`` of
    ``classify_cube`` and any memory layout of the cube.  ``predict_chunk`` on a
    differently shaped array may differ from ``classify_cube`` in the last float32 bit of
    a PCA feature (BLAS rounding depends on the product shape).

Threading:
    Pure computation, safe in a worker thread; a fitted model is read-only and may be
    shared.  Each ``predict_into`` / ``predict_chunk`` call runs its pieces on up to
    ``n_jobs`` short-lived threads (RF at most 4: more were slower), which release the GIL
    in the tree traversal and in libsvm; the RF fit uses scikit-learn's threads.  The main
    thread stays responsive (timer latencies below).  ``cancel()`` is polled before the fit,
    between RF tree batches and after the SVM solver, and raises
    :class:`~modules.cube_viewer.roi_stats.RoiCancelled`; the libsvm / liblinear solver
    itself cannot be interrupted (it is bounded by ``max_train_per_class``).

Measured cost (i9-12900K, 8 P + 8 E cores, 34 GB, Windows 11, Python 3.13, numpy 2.1.3 MKL,
scikit-learn 1.6.1, 2026-10-21).  Synthetic 1024 x 1762 x 185 uint16 scenes (8 materials, a
0.55-1.0 brightness ramp along the slit, unknown blobs): 5000 training px per class from 3
ROIs per class of scene A; full-image ``classify_cube`` of an in-RAM scene B with other
noise and layout (held out); accuracy on its ROI-class pixels.  K = 2 / 5 / 8 classes:

    =====================  =================  =======================  ======================
    configuration          fit s              classify s (1.8 M px)    held-out accuracy
    =====================  =================  =======================  ======================
    RF defaults            0.36 / 0.67 /      1.4-1.5 / 1.4-1.8 /      0.996 / 0.996 / 0.954
                           1.3-1.6            1.6-2.1
    RF, 8 predict threads  -                  2.4 (K = 8; GIL-bound)   -
    RF, normalize          4.4 (K = 8)        2.4                      1.000
    RF, pca_dims 10        0.87 (K = 8)       1.2                      0.995
    RF, 50 trees           0.73 (K = 8)       1.05                     0.953
    SVM defaults (RBF,     0.09 / 0.36 / 0.54 3.6 / 6.3 / 6.0          1.000 / 0.9997 /
    PCA-10, 1000 px/cls)                                               0.9998
    SVM, pca_dims 20       0.15 / - / 1.45    6.0 / - / 19.2           1.000 / - / 0.9998
    SVM, linear_kernel     0.03 / - / 0.10    1.0 / - / 1.5            1.000 / - / 1.000
    SVM, 2000 px / class   1.2 (K = 8)        7.4                      0.9998
    SVM, softmax of the    0.13 (K = 8)       8.0                      0.9998 (scores
    decision values                                                    0.63-0.88 only)
    =====================  =================  =======================  ======================

    The 0.954 of the RF is two near-duplicate leaf spectra under the brightness ramp
    (``normalize``: 1.000); SAM gets 1.000 on these scenes.  With one class made of 6
    materials (K = 3): RBF SVM 0.9996, RF 0.951, linear SVM 0.75-0.96, SAM 0.88 - why the
    RBF kernel is the default.  Real SWIR reflectance (Cashew nuts S01 -> S02 of the
    pipeline's dataset, Moist / Dry, 5000 px per class, 94,865 GT px): SVM 0.978
    (``pca_dims`` 20: 0.992), RF 0.970 (``normalize``: 0.977), linear SVM 0.974, against
    Mahalanobis 0.995, Gaussian ML 0.978 and SAM 0.884.

    The table shows runs on a quiet machine.  While other processes were busy the same
    runs took up to 1.5x (RF; 1.4x for SAM) and up to 3.5x for the
    multi-threaded SVM: K = 2 / 5 / 8 up to 6.8 / 14.4 / 21 s (where Windows places the 8
    threads on this P + E core CPU matters too); one SVM thread: 19-23 us / px.  16 or 24
    threads were not reliably faster than 8.

    Memory (private bytes above the class samples, after warm-up): prediction 1-13 MB
    per call; fit RF 14 / 57 MB (K = 2 / 8: the float32 training matrix and
    scikit-learn's buffers), SVM 9 / 87 MB (libsvm's kernel cache; numpy part 5 / 17
    MB).  The first float64 matrix product of a process makes MKL reserve about 150 MB
    once, for every numpy user alike.

    A 1 ms timer in the main thread was late by at most 4.2 ms during RF fits and 7.6 ms
    during SVM prediction.  Cancel latency: RF classify <= 2.3 ms, SVM classify <= 50 ms,
    RF fit <= 160 ms (one batch of 16 trees), SVM fit up to the solver time (<= 0.44 s,
    K = 8).

Example:
    >>> from modules.cube_viewer import segment, ml                    # doctest: +SKIP
    >>> model = segment.fit("random_forest", classes, {"n_estimators": 50})  # doctest: +SKIP
    >>> result = segment.classify_cube(cube, model, cancel=cancel)     # doctest: +SKIP
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import segment as _segment
from .roi_stats import RoiCancelled

try:  # optional dependency: without it the methods stay listed but disabled
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC, LinearSVC
    HAVE_SKLEARN = True
except ImportError:  # pragma: no cover - depends on the installation
    CalibratedClassifierCV = RandomForestClassifier = SVC = LinearSVC = None
    HAVE_SKLEARN = False

__all__ = ("HAVE_SKLEARN", "RF_PARAMS", "SVM_PARAMS", "PARAM_SPECS", "RandomForestModel",
           "SvmModel", "fit_model", "register", "resolve_params")

#: Random Forest defaults (measurements in the module docstring).
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 0,
    "min_samples_leaf": 1,
    "pca_dims": 0,
    "normalize": False,
    "n_jobs": 8,
    "seed": 0,
}

#: SVM defaults.
SVM_PARAMS = {
    "linear_kernel": False,
    "C": 10.0,
    "gamma": 0.0,
    "pca_dims": 10,
    "max_train_per_class": 1000,
    "normalize": False,
    "n_jobs": 8,
    "seed": 0,
}

#: ``{method: {param: (minimum, maximum, help text)}}`` for the parameter widgets
#: (``None`` bounds for booleans).
PARAM_SPECS = {
    "random_forest": {
        "n_estimators": (1, 1000, "Number of trees. Prediction time grows in proportion "
                                  "(100 trees: about 1-1.5 s for 1.8 M px)."),
        "max_depth": (0, 200, "Maximum tree depth; 0 = grow until the leaves are pure."),
        "min_samples_leaf": (1, 10000, "Minimum training pixels per leaf; larger = smoother, "
                                       "less over-fitting."),
        "pca_dims": (0, 200, "0 = use the raw bands; k > 0 = use k whitened principal components."),
        "normalize": (None, None, "Divide every spectrum by its mean level first (ignores "
                                  "brightness, keeps the spectral shape)."),
        "n_jobs": (1, 64, "Threads for training (prediction uses at most 4)."),
        "seed": (0, 2 ** 31 - 1, "Random seed; a given seed always gives the same forest."),
    },
    "svm": {
        "linear_kernel": (None, None, "Linear SVM instead of RBF: much faster, but it cannot model "
                                      "a class made of several different materials."),
        "C": (1e-4, 1e6, "Penalty of training errors; larger = tighter fit to the ROIs."),
        "gamma": (0.0, 1e4, "RBF width; 0 = 'scale' (1 / (features x variance))."),
        "pca_dims": (0, 200, "Whitened principal components used as features; 0 = all bands "
                             "(z-scored, slower)."),
        "max_train_per_class": (2, 100000, "Training pixels per class (fixed-seed subsample); "
                                           "time grows faster than linearly."),
        "normalize": (None, None, "Divide every spectrum by its mean level first."),
        "n_jobs": (1, 64, "Threads for prediction."),
        "seed": (0, 2 ** 31 - 1, "Random seed of the subsample and the probability calibration."),
    },
}

_LABELS = {"random_forest": "Random Forest (scikit-learn)", "svm": "SVM (scikit-learn)"}
_DESCRIPTIONS = {
    "random_forest": "Random Forest on the raw bands; score = share of the trees' class "
                     "probability. About 1-2 s per 1.8 M px plus 0.4-1.4 s training.",
    "svm": "Support vector machine, RBF kernel on 10 whitened principal components, trained on "
           "up to 1000 px per class; score = calibrated probability. About 4-12 s per 1.8 M px.",
}
_MISSING = "Needs scikit-learn, which is not installed (pip install scikit-learn)."
_DEFAULTS = {"random_forest": RF_PARAMS, "svm": SVM_PARAMS}

# Rows of one covariance accumulation step (float64 scratch = rows x bands x 8 bytes).
_COV_ROWS = 4096
# Image rows are split into equal pieces of at most this many pixels (the work unit of the
# prediction threads; depends only on the row width, so results do not depend on chunking).
_PIECE_PX = 2048
# RF trees fitted per warm-start batch, per fit thread (cancel is polled between batches).
_TREES_PER_THREAD = 2
# RF prediction threads at most: per-tree calls contend for the GIL; one 14,096 px chunk of
# 100 trees took 25 / 17 / 8.3 / 12 / 17 ms with 1 / 2 / 4 / 6 / 8 threads.
_RF_PREDICT_THREADS = 4
# Principal axes with an eigenvalue below this fraction of the largest are dropped.
_EIG_RTOL = 1e-10
# LinearSVC iteration limit (reaching it becomes a model warning).
_LINEAR_MAX_ITER = 5000


def _require_sklearn() -> None:
    """Raise ``RuntimeError`` when scikit-learn is not installed."""
    if not HAVE_SKLEARN:
        raise RuntimeError(_MISSING)


def _check_cancel(cancel) -> None:
    """Raise ``RoiCancelled`` when ``cancel()`` returns True."""
    if cancel is not None and cancel():
        raise RoiCancelled()


def resolve_params(method: str, params=None) -> dict:
    """Parameters of ``'random_forest'`` / ``'svm'``: defaults updated with ``params``, type-coerced.

    Works without scikit-learn (``compare.estimate_seconds`` uses it).

    Raises:
        ValueError: Unknown parameter, non-integral value for an int, or a value outside
            :data:`PARAM_SPECS`.
    """
    defaults = _DEFAULTS[method]
    merged = dict(defaults)
    for key, value in dict(params or {}).items():
        if key not in defaults:
            raise ValueError(f"{_LABELS[method]}: unknown parameter {key!r}; "
                             f"expected {sorted(defaults)}")
        kind = type(defaults[key])
        try:
            if kind is bool:
                merged[key] = bool(value)
            elif kind is int:
                number = float(value)
                if not number.is_integer():
                    raise ValueError
                merged[key] = int(number)
            else:
                merged[key] = float(value)
                if not math.isfinite(merged[key]):
                    raise ValueError
        except (TypeError, ValueError):
            raise ValueError(f"{_LABELS[method]}: {key} must be a finite "
                             f"{kind.__name__}, got {value!r}") from None
    for key, (low, high, _) in PARAM_SPECS[method].items():
        if low is not None and not low <= merged[key] <= high:
            raise ValueError(f"{_LABELS[method]}: {key} must lie in [{low}, {high}], "
                             f"got {merged[key]}")
    return merged


def _valid_rows(X: np.ndarray, normalize: bool) -> np.ndarray:
    """Bool ``(n,)``: every band finite, not all zero and (``normalize``) a positive band sum."""
    sq = np.einsum("ij,ij->i", X, X)          # NaN / inf propagate; 0 only for all-zero rows
    valid = (sq > 0) & (sq < np.inf)
    if normalize:
        valid &= X.sum(axis=1, dtype=np.float64) > 0
    return valid


def _training_arrays(classes, max_per_class, seed: int, normalize: bool):
    """``(X float32 (n, B), y int64 (n,), counts (K,))`` of the valid training pixels.

    ``max_per_class`` caps every class with a fixed-seed subsample (kept in sample order).

    Raises:
        ValueError: Fewer than 2 classes, mismatched band counts or a class without a
            valid training pixel.
    """
    if len(classes) < 2:
        raise ValueError(f"needs at least 2 classes, got {len(classes)}")
    rng = np.random.default_rng(seed)
    xs, ys, bands = [], [], None
    for index, spec in enumerate(classes):
        sample = np.asarray(spec.sample, dtype=np.float32)
        if sample.ndim != 2 or sample.shape[0] == 0:
            raise ValueError(f"class {spec.name!r} has no training pixels")
        if bands is None:
            bands = sample.shape[1]
        elif sample.shape[1] != bands:
            raise ValueError(f"class {spec.name!r} has {sample.shape[1]} bands, expected {bands}")
        valid = _valid_rows(sample, normalize)
        if not valid.all():
            sample = sample[valid]
        if sample.shape[0] == 0:
            raise ValueError(f"class {spec.name!r} has no valid training pixel")
        if max_per_class is not None and sample.shape[0] > max_per_class:
            sample = sample[np.sort(rng.choice(sample.shape[0], max_per_class, replace=False))]
        xs.append(np.ascontiguousarray(sample))
        ys.append(np.full(sample.shape[0], index, dtype=np.int64))
    y = np.concatenate(ys)
    return np.concatenate(xs), y, np.bincount(y, minlength=len(classes))


def _pieces(n_rows: int, width: int) -> list:
    """``(row, start, stop)`` work pieces: each row in equal parts of at most ``_PIECE_PX``."""
    parts = max(1, -(-width // _PIECE_PX))
    bounds = [i * width // parts for i in range(parts + 1)]
    return [(r, bounds[i], bounds[i + 1]) for r in range(n_rows) for i in range(parts)
            if bounds[i + 1] > bounds[i]]


def _softmax(values: np.ndarray) -> np.ndarray:
    """Row-wise softmax of float64 ``(n, K)`` (shifted by the row maximum: no overflow)."""
    shifted = values - values.max(axis=1, keepdims=True)
    np.exp(shifted, out=shifted)
    shifted /= shifted.sum(axis=1, keepdims=True)
    return shifted


class _Features:
    """Per-pixel feature transform fitted on the training pixels (see the module docstring).

    Args:
        X: Valid training pixels, float32 ``(n, B)``.
        pca_dims: ``k > 0`` = whitened scores of the top ``k`` principal axes.
        normalize: Divide each pixel by its band mean first.
        standardize: With ``k == 0``: per-band z-scores (else the raw bands).
    """

    def __init__(self, X: np.ndarray, *, pca_dims: int, normalize: bool, standardize: bool):
        self.normalize = bool(normalize)
        self.n_bands = int(X.shape[1])
        self.weights = None          # float32 (B, d): PCA whitening
        self.scale = None            # float32 (B,): z-scores
        self.center = None           # float32 (B,): training mean, subtracted first
        self.dropped_dims = 0
        if not (pca_dims > 0 or standardize):
            return
        if self.normalize:
            X = self._normalized(X)
        n, bands = X.shape
        mean = X.mean(axis=0, dtype=np.float64)
        scatter = np.zeros((bands, bands))
        for r0 in range(0, n, _COV_ROWS):
            block = X[r0:r0 + _COV_ROWS].astype(np.float64)
            block -= mean
            scatter += block.T @ block
        cov = scatter / max(n - 1, 1)
        self.center = mean.astype(np.float32)
        if pca_dims > 0:
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1][:min(int(pca_dims), bands)]
            floor = max(float(evals[order[0]]), 0.0) * _EIG_RTOL
            keep = [i for i in order if evals[i] > floor]
            if not keep:
                raise ValueError("the training pixels do not vary (all spectra are identical)")
            self.dropped_dims = len(order) - len(keep)
            vecs = evecs[:, keep]
            biggest = np.abs(vecs).argmax(axis=0)
            vecs = vecs * np.sign(vecs[biggest, np.arange(len(keep))])
            self.weights = np.ascontiguousarray(vecs / np.sqrt(evals[keep]), dtype=np.float32)
        else:
            std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
            std[~(std > 0)] = 1.0
            self.scale = (1.0 / std).astype(np.float32)

    @property
    def n_features(self) -> int:
        """Number of output features."""
        return self.weights.shape[1] if self.weights is not None else self.n_bands

    @staticmethod
    def _normalized(X: np.ndarray) -> np.ndarray:
        """``X * (B / band_sum)`` as a float32 copy (rows with a sum <= 0 become 0)."""
        sums = X.sum(axis=1, dtype=np.float64)
        factor = np.zeros_like(sums)
        np.divide(X.shape[1], sums, out=factor, where=sums > 0)
        return X * factor.astype(np.float32)[:, None]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Features of float32 pixels ``(n, B)``: float32 ``(n, d)`` (``X`` itself for raw bands).

        The training mean is subtracted before the product, so a large common level (raw
        DN) does not cancel in float32: on the synthetic DN scenes the features were within
        2.5e-5 (PCA, unit variance) and 4.4e-7 (z-scores) of the same transform in float64.
        """
        if self.center is None:
            return X
        centred = self._normalized(X) if self.normalize else X.copy()
        centred -= self.center
        if self.weights is not None:
            return centred @ self.weights
        centred *= self.scale
        return centred


class _SklearnModel:
    """Shared part of the scikit-learn models: protocol attributes and the piece-wise prediction."""

    method = ""
    score_kind = "probability"
    overwrites_input = False

    def __init__(self, classes, params):
        _require_sklearn()
        self.classes = list(classes)
        self.params = resolve_params(self.method, params)
        self.warnings: list = []
        self.n_bands = 0
        self.n_train = 0
        self.features: _Features | None = None

    @property
    def n_classes(self) -> int:
        """Number of classes ``K``."""
        return len(self.classes)

    def _threads(self) -> int:
        """Threads used by one prediction call."""
        return int(self.params["n_jobs"])

    def _fit_features(self, X: np.ndarray, *, standardize: bool) -> None:
        """Fit the feature transform and note dropped principal axes."""
        p = self.params
        self.n_bands = int(X.shape[1])
        self.features = _Features(X, pca_dims=p["pca_dims"], normalize=p["normalize"],
                                  standardize=standardize)
        if self.features.dropped_dims:
            self.warnings.append(f"Only {self.features.n_features} of {p['pca_dims']} principal "
                                 f"components vary in the training pixels; the rest were dropped.")

    def default_thresholds(self) -> np.ndarray:
        """Minimum probability per class, float64 ``(K,)``: 0 = every valid pixel keeps its label."""
        return np.zeros(len(self.classes))

    def _proba(self, F: np.ndarray) -> np.ndarray:
        """Class probabilities float64 ``(n, K)`` of the feature rows ``F`` (one thread)."""
        raise NotImplementedError

    def _predict_piece(self, X: np.ndarray, labels: np.ndarray, score: np.ndarray, proba) -> None:
        """Labels / scores (and optionally all probabilities) of the pixels ``X (n, B)``.

        Only valid pixels are predicted.  When they form one run (the usual case: the zero
        padding ``classify_cube`` adds to a row is at its end) they are a view, not a copy.
        """
        valid = _valid_rows(X, self.features.normalize)
        all_valid = bool(valid.all())
        Xv = X
        if not all_valid:
            index = np.flatnonzero(valid)
            if index.size and index[-1] - index[0] + 1 == index.size:
                valid = slice(int(index[0]), int(index[-1]) + 1)
            Xv = X[valid]
            labels[:] = _segment.UNCLASSIFIED
            score[:] = np.nan
            if proba is not None:
                proba[:] = np.nan
        if Xv.shape[0] == 0:
            return
        P = self._proba(np.ascontiguousarray(self.features.transform(Xv), dtype=np.float32))
        best = P.argmax(axis=1)
        top = P[np.arange(best.size), best]
        if all_valid:
            labels[:] = best
            score[:] = top
            if proba is not None:
                proba[:] = P
        else:
            labels[valid] = best
            score[valid] = top
            if proba is not None:
                proba[valid] = P

    def _run(self, X, labels_out, score_out, proba_out=None) -> None:
        """Predict float32 pixels ``X (..., B)`` piece by piece on the prediction threads."""
        X3 = X.reshape((-1,) + X.shape[-2:]) if X.ndim >= 2 else X.reshape(1, 1, -1)
        rows, width = X3.shape[:2]
        labels = np.empty((rows, width), dtype=np.int16)
        score = np.empty((rows, width), dtype=np.float32)
        proba = None if proba_out is None else np.empty((rows, width, len(self.classes)))
        pieces = _pieces(rows, width)
        n_groups = max(1, min(self._threads(), len(pieces)))
        groups = [pieces[i * len(pieces) // n_groups:(i + 1) * len(pieces) // n_groups]
                  for i in range(n_groups)]

        def work(group):
            for r, a, b in group:
                self._predict_piece(X3[r, a:b], labels[r, a:b], score[r, a:b],
                                    None if proba is None else proba[r, a:b])

        if n_groups == 1:
            work(groups[0])
        else:
            with ThreadPoolExecutor(max_workers=n_groups, thread_name_prefix="cube-viewer-ml") as pool:
                for future in [pool.submit(work, group) for group in groups]:
                    future.result()
        labels_out[...] = labels.reshape(labels_out.shape)
        score_out[...] = score.reshape(score_out.shape)
        if proba_out is not None:
            proba_out[...] = proba.reshape(proba_out.shape)

    def _as_pixels(self, X) -> np.ndarray:
        """``X`` as float32 ``(..., B)`` (no copy when it already is), band count checked."""
        X = np.asarray(X)
        if X.ndim < 1 or X.shape[-1] != self.n_bands:
            raise ValueError(f"expected pixels with {self.n_bands} bands, got shape {X.shape}")
        return X.astype(np.float32, copy=False)

    def predict_into(self, X, labels_out, score_out, work=None) -> None:
        """Fast path of ``segment.classify_cube``: write labels / scores of ``X (..., B)``.

        ``X`` is float32 and is not modified (``overwrites_input`` is False); ``work`` is
        accepted for the protocol and not used.
        """
        X = self._as_pixels(X)
        if X.size:
            self._run(X, labels_out, score_out)

    def predict_chunk(self, X):
        """Best class and its probability for pixels ``X (..., B)``; ``X`` is not modified.

        Returns:
            tuple: ``labels`` int16 ``(...)`` (class index, ``-1`` = invalid pixel) and
            ``score`` float32 ``(...)`` = probability of that class (NaN when invalid).
        """
        X = self._as_pixels(X)
        labels = np.empty(X.shape[:-1], dtype=np.int16)
        score = np.empty(X.shape[:-1], dtype=np.float32)
        if labels.size:
            self._run(X, labels, score)
        return labels, score

    def predict_proba(self, X) -> np.ndarray:
        """All class probabilities of pixels ``X (..., B)``: float64 ``(..., K)``, NaN rows = invalid."""
        X = self._as_pixels(X)
        proba = np.empty(X.shape[:-1] + (len(self.classes),))
        if proba.size:
            labels = np.empty(X.shape[:-1], dtype=np.int16)
            score = np.empty(X.shape[:-1], dtype=np.float32)
            self._run(X, labels, score, proba)
        return proba


class RandomForestModel(_SklearnModel):
    """Random Forest pixel classifier implementing the segment ``Model`` protocol.

    Args:
        classes: ``segment.ClassSpec`` list (>= 2): the label order.
        params: Overrides of :data:`RF_PARAMS`.
        cancel: Optional ``cancel() -> bool``, polled between batches of trees.

    Raises:
        RuntimeError: scikit-learn is not installed.
        ValueError: Bad parameters or training data.
        RoiCancelled: ``cancel()`` returned True.
    """

    method = "random_forest"

    def __init__(self, classes, params=None, *, cancel=None):
        super().__init__(classes, params)
        p = self.params
        _check_cancel(cancel)
        X, y, _ = _training_arrays(self.classes, None, p["seed"], p["normalize"])
        self.n_train = int(X.shape[0])
        self._fit_features(X, standardize=False)
        F = np.ascontiguousarray(self.features.transform(X), dtype=np.float32)
        forest = RandomForestClassifier(
            n_estimators=1, max_depth=p["max_depth"] or None,
            min_samples_leaf=p["min_samples_leaf"], max_features="sqrt", bootstrap=True,
            n_jobs=p["n_jobs"], random_state=p["seed"], warm_start=True)
        step = _TREES_PER_THREAD * p["n_jobs"]
        built = 0
        while built < p["n_estimators"]:
            _check_cancel(cancel)
            forest.set_params(n_estimators=min(p["n_estimators"], built + step))
            forest.fit(F, y)
            built = len(forest.estimators_)
        _check_cancel(cancel)
        self.forest = forest
        self._trees = [estimator.tree_ for estimator in forest.estimators_]
        self._tables = []
        for tree in self._trees:
            table = np.array(tree.value[:, 0, :], dtype=np.float64)
            table /= table.sum(axis=1, keepdims=True)
            self._tables.append(table)
        self.n_trees = len(self._trees)
        self.mean_depth = float(np.mean([tree.max_depth for tree in self._trees]))

    def _threads(self) -> int:
        """At most ``_RF_PREDICT_THREADS`` prediction threads."""
        return min(int(self.params["n_jobs"]), _RF_PREDICT_THREADS)

    def _proba(self, F: np.ndarray) -> np.ndarray:
        """Mean leaf class proportions over the trees, summed per pixel in tree order."""
        out = np.zeros((F.shape[0], len(self.classes)))
        for tree, table in zip(self._trees, self._tables):
            out += table.take(tree.apply(F), axis=0)
        out /= len(self._trees)
        return out


class SvmModel(_SklearnModel):
    """Support vector machine pixel classifier implementing the segment ``Model`` protocol.

    RBF kernel: ``sklearn.svm.SVC`` (libsvm, one-vs-one; Platt probabilities from an
    internal 5-fold cross-validation, combined by pairwise coupling).  Linear kernel:
    ``sklearn.svm.LinearSVC`` (one-vs-rest) calibrated by ``CalibratedClassifierCV(
    method='sigmoid', cv=min(3, smallest class), ensemble=False)``; with a 1-pixel class
    the score is a softmax of the decision values instead (``calibrated`` False, with a
    model warning).

    Args:
        classes: ``segment.ClassSpec`` list (>= 2): the label order.
        params: Overrides of :data:`SVM_PARAMS`.
        cancel: Optional ``cancel() -> bool``, polled before and after the solver.

    Raises:
        RuntimeError: scikit-learn is not installed.
        ValueError: Bad parameters or training data.
        RoiCancelled: ``cancel()`` returned True.
    """

    method = "svm"

    def __init__(self, classes, params=None, *, cancel=None):
        super().__init__(classes, params)
        p = self.params
        _check_cancel(cancel)
        X, y, counts = _training_arrays(self.classes, p["max_train_per_class"], p["seed"],
                                        p["normalize"])
        self.n_train = int(X.shape[0])
        self._fit_features(X, standardize=True)
        F = np.ascontiguousarray(self.features.transform(X), dtype=np.float64)
        _check_cancel(cancel)
        self.linear = bool(p["linear_kernel"])
        self.n_support = 0
        if not self.linear:
            gamma = p["gamma"] if p["gamma"] > 0 else "scale"
            self.estimator = SVC(C=p["C"], kernel="rbf", gamma=gamma, probability=True,
                                 decision_function_shape="ovr", cache_size=500,
                                 random_state=p["seed"]).fit(F, y)
            self.calibrated = True
            self.n_support = int(self.estimator.n_support_.sum())
        else:
            base = LinearSVC(C=p["C"], dual="auto", max_iter=_LINEAR_MAX_ITER,
                             random_state=p["seed"])
            self.calibrated = int(counts.min()) >= 2
            if self.calibrated:
                self.estimator = CalibratedClassifierCV(
                    base, method="sigmoid", cv=int(min(3, counts.min())), ensemble=False).fit(F, y)
                fitted = self.estimator.calibrated_classifiers_[0].estimator
            else:
                self.estimator = fitted = base.fit(F, y)
                self.warnings.append("A class has a single training pixel: the SVM scores are a "
                                     "softmax of the decision values, not calibrated probabilities.")
            if int(np.max(fitted.n_iter_)) >= _LINEAR_MAX_ITER:
                self.warnings.append(f"The linear SVM did not converge in {_LINEAR_MAX_ITER} "
                                     f"iterations; try a smaller C.")
        _check_cancel(cancel)

    def _proba(self, F: np.ndarray) -> np.ndarray:
        """Class probabilities (libsvm / liblinear on float64 features)."""
        F64 = np.ascontiguousarray(F, dtype=np.float64)
        if self.calibrated:
            return np.asarray(self.estimator.predict_proba(F64), dtype=np.float64)
        values = np.asarray(self.estimator.decision_function(F64), dtype=np.float64)
        if values.ndim == 1:                          # binary: one signed margin
            values = np.stack([-values, values], axis=1)
        return _softmax(values)


def _fit_random_forest(classes, params, *, cancel=None):
    """Fitter registered for ``'random_forest'``."""
    return RandomForestModel(classes, params, cancel=cancel)


def _fit_svm(classes, params, *, cancel=None):
    """Fitter registered for ``'svm'``."""
    return SvmModel(classes, params, cancel=cancel)


_FITTERS = {"random_forest": _fit_random_forest, "svm": _fit_svm}


def fit_model(method_key: str, classes, params=None, *, cancel=None):
    """Fit ``'random_forest'`` or ``'svm'`` directly (``segment.fit`` does the same via the registry).

    Raises:
        KeyError: Not one of the two keys.
        RuntimeError: scikit-learn is not installed.
    """
    if method_key not in _FITTERS:
        raise KeyError(f"not a scikit-learn method: {method_key!r}")
    return _FITTERS[method_key](classes, params, cancel=cancel)


def register() -> None:
    """(Re-)register both methods in ``segment.METHODS`` with their defaults.

    Available with scikit-learn; otherwise listed with ``available=False`` and no fitter,
    so ``segment.fit`` raises ``RuntimeError``.  Called on import.
    """
    for key in ("random_forest", "svm"):
        info = _segment.MethodInfo(
            key, _LABELS[key], "ml", "probability", dict(_DEFAULTS[key]), HAVE_SKLEARN,
            _DESCRIPTIONS[key] if HAVE_SKLEARN else _MISSING)
        _segment.register_method(info, _FITTERS[key] if HAVE_SKLEARN else None)


register()
