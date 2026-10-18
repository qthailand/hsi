"""Standard baseline (background) removers for 1-D spectra.

Selected by name (case-insensitive) through :func:`estimate_baseline` and
:func:`remove_baseline`:

============  ===================  ==========================================================
name          label                algorithm
============  ===================  ==========================================================
``none``      None                 zero baseline (no correction)
``modpoly``   ModPoly              Lieber & Mahadevan-Jansen, Appl. Spectrosc. 57 (2003) 1363
``imodpoly``  IModPoly             Zhao et al., Appl. Spectrosc. 61 (2007) 1225
``zhangfit``  ZhangFit (airPLS)    Zhang, Chen & Liang, Analyst 135 (2010) 1138
``asls``      AsLS                 Eilers & Boelens, Leiden University report (2005)
``arpls``     arPLS                Baek et al., Analyst 140 (2015) 250
============  ===================  ==========================================================

Every tunable parameter, its default and its valid range are in :data:`DEFAULT_PARAMS` and
:data:`PARAM_HELP` and in the docstring of :func:`estimate_baseline`.

Reference implementations (what "identical" below means is tested in
``tests/test_baseline.py``: same iterations and decisions, differences only from rounding):

* ``modpoly`` - identical to ``BaselineRemoval.ModPoly(degree, repitition=max_iter,
  gradient=tol)`` when ``use_original=True``, and to ``pybaselines.polynomial.modpoly`` for a
  fixed number of iterations (``tol=0``; pybaselines stops on the change of the fit instead).
  Deliberate differences from BaselineRemoval: the default ``use_original=False`` clips the
  *working* spectrum, ``y := min(y, fit)``, as in Lieber's paper (BaselineRemoval clips the
  original one); a working value of exactly 0 that does not change counts as no change
  (BaselineRemoval divides 0/0, gets NaN and silently stops after the first fit).
* ``imodpoly`` - identical to ``BaselineRemoval.IModPoly(degree, repitition=max_iter,
  gradient=tol)``. An exactly polynomial input (DEV = 0 up to rounding) returns that
  polynomial: the peak test allows for rounding and DEV = 0 ends the iteration (the reference
  can drop every point there and divide by zero). ``pybaselines.polynomial.imodpoly`` is a different variant
  (it keeps the removed peak points in DEV) and is not reproduced.
* ``zhangfit`` - the loop of Zhang's reference ``airPLS.py`` (``lambda_=lam, porder=diff_order,
  itermax=max_iter``, its stop rule and end-point weights). Tested identical to
  ``BaselineRemoval.ZhangFit`` for ``diff_order=1`` (BaselineRemoval ignores ``porder`` and
  always uses first differences) and, with the end-point weights switched off, to
  ``pybaselines.whittaker.airpls`` (which omits them) for ``diff_order`` 1-3. As in pybaselines
  the iteration number in the weight exponent is capped at 50 (no effect for
  ``max_iter <= 50``; avoids overflow), and the iteration stops when no residual is negative
  (the reference fails on ``max([])`` there).
* ``asls`` / ``arpls`` - identical to ``pybaselines.whittaker.asls`` / ``arpls``.

Penalized methods (``zhangfit``, ``asls``, ``arpls``) solve ``(W + lam·D'D) z = W y`` in banded
storage with LAPACK's banded Cholesky (:func:`scipy.linalg.solveh_banded`, ~50 µs for 1600
points, the solver pybaselines uses) and fall back to sparse LU
(:func:`scipy.sparse.linalg.spsolve`, CSC) if Cholesky fails, which only extremely stiff
systems do (e.g. ``diff_order=3, lam=1e14`` on 4000 points). No dense ``n x n`` matrix is
ever built. Like any double-precision solver, accuracy is bounded by the conditioning
(~ ``lam·4**diff_order / min(w)``): about 1e-4 relative for ``lam=1e12, diff_order=2`` on 4000
points, far below the noise of a measured spectrum.

Data conventions:
    ``spectrum`` is 1-D (e.g. counts per spectral column), any real dtype, at least a few
    points (method-dependent minimum, see :func:`estimate_baseline`), all values finite.
    Samples are assumed evenly spaced; polynomial methods use the sample index scaled to
    ``[-1, 1]``. The baseline has the spectrum's units and length. Outputs are new
    ``float64`` arrays; the input is never modified.

Threading:
    Pure functions; call from any thread. The only shared state is a read-only cache of
    difference-penalty bands (``functools.lru_cache``, thread-safe).

Example:
    >>> import numpy as np
    >>> from modules.spectral_resolution.baseline import estimate_baseline, remove_baseline
    >>> x = np.arange(400.0)
    >>> background = 200 + 0.002 * (x - 150) ** 2
    >>> noise = np.random.default_rng(0).normal(0, 2, x.size)
    >>> spectrum = background + 1000 * np.exp(-0.5 * ((x - 150) / 3) ** 2) + noise
    >>> corrected, baseline = remove_baseline(spectrum, "arpls", lam=1e5)
    >>> bool(abs(corrected[150] - 1000) < 5), bool(np.abs(baseline - background).max() < 5)
    (True, True)
    >>> polynomial = estimate_baseline(spectrum, "IModPoly", degree=2)   # name: any case
    >>> bool(np.abs(polynomial - background).max() < 5)
    True
"""
from __future__ import annotations

import math
import numbers
from functools import lru_cache
from typing import Callable, Dict, Tuple

import numpy as np
from scipy import sparse
from scipy.linalg import LinAlgError, solveh_banded
from scipy.sparse.linalg import spsolve
from scipy.special import expit

__all__ = ["BASELINE_METHODS", "BASELINE_LABELS", "DEFAULT_PARAMS", "PARAM_HELP",
           "estimate_baseline", "remove_baseline"]

#: Method names accepted by :func:`estimate_baseline` (case-insensitive), in display order.
BASELINE_METHODS = ("none", "modpoly", "imodpoly", "zhangfit", "asls", "arpls")

#: Display label of every method in :data:`BASELINE_METHODS`.
BASELINE_LABELS = {"none": "None", "modpoly": "ModPoly", "imodpoly": "IModPoly",
                   "zhangfit": "ZhangFit (airPLS)", "asls": "AsLS", "arpls": "arPLS"}

#: Every tunable parameter of every method with its default (see :data:`PARAM_HELP`).
#: A reference copy: editing it does not change the defaults :func:`estimate_baseline` uses.
DEFAULT_PARAMS = {
    "none": {},
    "modpoly": {"degree": 2, "max_iter": 100, "tol": 1e-3, "use_original": False},
    "imodpoly": {"degree": 2, "max_iter": 100, "tol": 1e-3},
    "zhangfit": {"lam": 100.0, "diff_order": 1, "max_iter": 15, "tol": 1e-3},
    "asls": {"lam": 1e6, "p": 1e-2, "diff_order": 2, "max_iter": 50, "tol": 1e-3},
    "arpls": {"lam": 1e5, "diff_order": 2, "max_iter": 50, "tol": 1e-3},
}

_LAM_HELP = ("Smoothness lambda (> 0) of the difference penalty; larger = stiffer baseline. "
             "Scales roughly with (feature width in samples) ** (2 * diff_order).")
_DIFF_HELP = "Order of the difference penalty (1, 2 or 3)."

#: One-line description (meaning, valid range, reference name) of every parameter.
PARAM_HELP = {
    "none": {},
    "modpoly": {
        "degree": "Polynomial degree (>= 0). BaselineRemoval 'degree', pybaselines 'poly_order'.",
        "max_iter": ("Maximum number of refits after the first fit (>= 0), i.e. at most "
                     "max_iter + 1 polynomial fits. BaselineRemoval 'repitition' and "
                     "pybaselines 'max_iter' (same meaning)."),
        "tol": ("Stop when sum_i |y_i(new) - y_i(old)| / |y_i(old)| of the clipped working "
                "spectrum falls below tol (>= 0; 0 = always max_iter refits). "
                "BaselineRemoval 'gradient'."),
        "use_original": ("False (Lieber 2003): clip the working spectrum, y := min(y, fit). "
                         "True: clip the original spectrum, y := min(y0, fit), as "
                         "BaselineRemoval.ModPoly does (pybaselines 'use_original')."),
    },
    "imodpoly": {
        "degree": "Polynomial degree (>= 0). BaselineRemoval 'degree'.",
        "max_iter": ("Maximum total number of polynomial fits including the first fit to the "
                     "whole spectrum (>= 1). BaselineRemoval 'repitition' (same meaning)."),
        "tol": ("Stop when |DEV_i - DEV_(i-1)| / DEV_i < tol, DEV = standard deviation of the "
                "fit residual (>= 0). BaselineRemoval 'gradient'."),
    },
    "zhangfit": {
        "lam": _LAM_HELP + " airPLS / BaselineRemoval 'lambda_'.",
        "diff_order": _DIFF_HELP + " airPLS 'porder' (BaselineRemoval always uses 1).",
        "max_iter": ("Maximum number of penalized fits (>= 1). airPLS 'itermax', "
                     "BaselineRemoval 'repitition'."),
        "tol": ("Stop when the summed magnitude of the negative residuals is below "
                "tol * sum|y| (>= 0). Fixed at 0.001 in airPLS / BaselineRemoval."),
    },
    "asls": {
        "lam": _LAM_HELP,
        "p": ("Asymmetry, 0 < p < 1: weight p above the baseline, 1 - p below it. The default "
              "0.01 lets the baseline climb into wide Lorentzian wings (about -2.5 % FWHM at "
              "14 px near the spectrum ends); p = 1e-3 with lam = 1e7 avoids that."),
        "diff_order": _DIFF_HELP,
        "max_iter": ("Maximum number of re-weightings (>= 0), i.e. at most max_iter + 1 fits "
                     "(pybaselines 'max_iter')."),
        "tol": "Stop when ||w_new - w_old|| / ||w_old|| of the weights is below tol (>= 0).",
    },
    "arpls": {
        "lam": _LAM_HELP,
        "diff_order": _DIFF_HELP,
        "max_iter": ("Maximum number of re-weightings (>= 0), i.e. at most max_iter + 1 fits "
                     "(pybaselines 'max_iter')."),
        "tol": "Stop when ||w_new - w_old|| / ||w_old|| of the weights is below tol (>= 0).",
    },
}

# Private copy actually used, so that editing DEFAULT_PARAMS cannot change behaviour.
_DEFAULTS = {method: dict(params) for method, params in DEFAULT_PARAMS.items()}

_EPS = float(np.finfo(float).eps)
_MAX_EXPONENT_ITERATION = 50      # airPLS weight exponent cap (as pybaselines)


# ─────────────────────────────────── public API ───────────────────────────────────

def estimate_baseline(spectrum, method: str = "none", **params) -> np.ndarray:
    """Estimate the smooth background of a 1-D spectrum.

    Args:
        spectrum: 1-D array-like of real numbers (counts), all finite. Not modified.
        method: One of :data:`BASELINE_METHODS`, case-insensitive.
        **params: Method parameters; omitted ones take :data:`DEFAULT_PARAMS`:

            * ``modpoly`` - ``degree=2`` (>= 0), ``max_iter=100`` refits after the first
              fit (>= 0), ``tol=1e-3`` (>= 0, summed relative change of the clipped
              spectrum), ``use_original=False`` (True clips the original spectrum, as
              BaselineRemoval does). Needs ``2 * (degree + 1)`` points.
            * ``imodpoly`` - ``degree=2``, ``max_iter=100`` fits in total (>= 1),
              ``tol=1e-3`` (relative change of DEV). Needs ``2 * (degree + 1)`` points.
            * ``zhangfit`` - ``lam=100`` (> 0), ``diff_order=1`` (1-3), ``max_iter=15`` fits
              (>= 1), ``tol=1e-3`` (negative-residual mass / sum|y|).
            * ``asls`` - ``lam=1e6``, ``p=0.01`` (0 < p < 1), ``diff_order=2``,
              ``max_iter=50`` re-weightings (>= 0), ``tol=1e-3`` (relative weight change).
            * ``arpls`` - ``lam=1e5``, ``diff_order=2``, ``max_iter=50``, ``tol=1e-3``.

            Penalized methods need ``diff_order + 2`` points. Integer parameters accept
            integral floats; ``use_original`` must be a bool.

    Returns:
        numpy.ndarray: New ``float64`` array, same length as ``spectrum``, in its units.
        ``"none"`` returns zeros.

    Raises:
        ValueError: unknown method or parameter name (the message lists the valid ones),
            a parameter out of range, ``spectrum`` not 1-D, not real, empty, containing
            NaN/inf or shorter than the method needs, or a penalized system so
            ill-conditioned that no finite solution exists (reduce ``lam``).
    """
    key, options = _prepare(method, params)
    return _estimate(_as_spectrum(spectrum), key, options)


def remove_baseline(spectrum, method: str = "none", **params) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract the baseline from a spectrum.

    Args:
        spectrum: 1-D array-like of real numbers, all finite. Not modified.
        method: One of :data:`BASELINE_METHODS`, case-insensitive.
        **params: Method parameters, see :func:`estimate_baseline`.

    Returns:
        tuple: ``(corrected, baseline)``, both new ``float64`` arrays of the spectrum's
        length; ``corrected = spectrum - baseline``.

    Raises:
        ValueError: as :func:`estimate_baseline`.
    """
    key, options = _prepare(method, params)
    y = _as_spectrum(spectrum)
    baseline = _estimate(y, key, options)
    return y - baseline, baseline


# ─────────────────────────────── validation helpers ───────────────────────────────

def _integer(low, high=None):
    """Validator for an integer parameter in ``[low, high]`` (integral floats accepted)."""
    def _check(name, value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        if not isinstance(value, numbers.Integral):
            if not (math.isfinite(float(value)) and float(value).is_integer()):
                raise ValueError(f"{name} must be an integer, got {value!r}")
        result = int(value)
        if result < low or (high is not None and result > high):
            span = f">= {low}" if high is None else f"in {low}..{high}"
            raise ValueError(f"{name} must be {span}, got {value!r}")
        return result
    return _check


def _real(low, low_open=False, high=None, high_open=False):
    """Validator for a finite float parameter with optional open/closed bounds."""
    def _check(name, value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a number, got {value!r}")
        result = float(value)
        bad = (not math.isfinite(result) or result < low or (low_open and result == low)
               or (high is not None and (result > high or (high_open and result == high))))
        if bad:
            span = f"{'>' if low_open else '>='} {low:g}"
            if high is not None:
                span += f" and {'<' if high_open else '<='} {high:g}"
            raise ValueError(f"{name} must be finite and {span}, got {value!r}")
        return result
    return _check


def _boolean(name, value):
    """Validator for a bool parameter (``bool`` or ``numpy.bool_`` only)."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be True or False, got {value!r}")
    return bool(value)


_TOL = _real(0.0)
_LAM = _real(0.0, low_open=True)
_DIFF_ORDER = _integer(1, 3)

_RULES: Dict[str, Dict[str, Callable]] = {
    "none": {},
    "modpoly": {"degree": _integer(0), "max_iter": _integer(0), "tol": _TOL,
                "use_original": _boolean},
    "imodpoly": {"degree": _integer(0), "max_iter": _integer(1), "tol": _TOL},
    "zhangfit": {"lam": _LAM, "diff_order": _DIFF_ORDER, "max_iter": _integer(1), "tol": _TOL},
    "asls": {"lam": _LAM, "p": _real(0.0, low_open=True, high=1.0, high_open=True),
             "diff_order": _DIFF_ORDER, "max_iter": _integer(0), "tol": _TOL},
    "arpls": {"lam": _LAM, "diff_order": _DIFF_ORDER, "max_iter": _integer(0), "tol": _TOL},
}


def _prepare(method, params) -> Tuple[str, dict]:
    """Normalise the method name and merge/validate ``params`` with the defaults."""
    key = method.strip().lower() if isinstance(method, str) else None
    if key not in _RULES:
        raise ValueError(f"unknown baseline method {method!r}; valid methods: "
                         f"{', '.join(BASELINE_METHODS)}")
    rules = _RULES[key]
    unknown = sorted(set(params) - set(rules))
    if unknown:
        valid = f"valid parameters: {', '.join(rules)}" if rules else "it takes no parameters"
        raise ValueError(f"unknown parameter(s) {', '.join(unknown)} for baseline method "
                         f"{key!r}; {valid}")
    options = dict(_DEFAULTS[key])
    for name, value in params.items():
        try:
            options[name] = rules[name](name, value)
        except ValueError as exc:
            raise ValueError(f"{key}: {exc}") from None
    return key, options


def _as_spectrum(spectrum) -> np.ndarray:
    """Validated ``float64`` copy of a 1-D real, finite, non-empty spectrum."""
    array = np.asarray(spectrum)
    if array.ndim != 1:
        raise ValueError(f"spectrum must be 1-D, got shape {array.shape}")
    if array.dtype.kind not in "biufO":
        raise ValueError(f"spectrum must contain real numbers, got dtype {array.dtype}")
    try:
        y = np.array(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"spectrum must contain real numbers ({exc})") from None
    if y.size == 0:
        raise ValueError("spectrum is empty; at least 1 point is needed")
    if not np.isfinite(y).all():
        raise ValueError("spectrum contains NaN or infinite values")
    return y


def _min_length(key: str, options: dict) -> int:
    """Fewest samples ``key`` accepts with ``options``."""
    if key in ("modpoly", "imodpoly"):
        # IModPoly keeps >= half of the points after peak removal (Cantelli), so this
        # always leaves at least degree + 1 points for the polynomial.
        return 2 * (options["degree"] + 1)
    if key in ("zhangfit", "asls", "arpls"):
        return options["diff_order"] + 2
    return 1


def _estimate(y: np.ndarray, key: str, options: dict) -> np.ndarray:
    """Run method ``key`` on the validated spectrum ``y``."""
    need = _min_length(key, options)
    if y.size < need:
        raise ValueError(f"{key}: spectrum has {y.size} point(s); this method needs at least "
                         f"{need} with these parameters")
    if key == "none":
        return np.zeros(y.size)
    return _METHODS[key](y, **options)


# ─────────────────────────────── polynomial methods ───────────────────────────────

def _polynomial_basis(n: int, degree: int) -> np.ndarray:
    """Legendre basis ``(n, degree + 1)`` on the sample index scaled to ``[-1, 1]``.

    Spans the same polynomials as raw powers of the index (so least-squares fits are the
    same) but stays well conditioned for any degree and length.
    """
    return np.polynomial.legendre.legvander(np.linspace(-1.0, 1.0, n), degree)


def _summed_relative_change(old: np.ndarray, new: np.ndarray) -> float:
    """``sum_i |new_i - old_i| / |old_i|`` (BaselineRemoval's ModPoly criterion).

    A zero ``old_i`` contributes 0 if unchanged and makes the sum infinite otherwise
    (BaselineRemoval gets NaN for 0/0, which ends its iteration prematurely).
    """
    change = np.abs(new - old)
    scale = np.abs(old)
    if scale.all():
        return float(np.sum(change / scale))
    zero = scale == 0
    if np.any(change[zero] > 0):
        return math.inf
    return float(np.sum(change[~zero] / scale[~zero]))


def _modpoly(y, degree, max_iter, tol, use_original):
    """ModPoly (Lieber & Mahadevan-Jansen 2003): iterated polynomial fit of ``min(y, fit)``.

    Fit a polynomial to the working spectrum, clip the working spectrum (or, with
    ``use_original``, the original one) to the fit, repeat until the summed relative change
    of the clipped spectrum is below ``tol`` or ``max_iter`` refits were made. Returns the
    last fit. Same loop as ``BaselineRemoval.ModPoly`` (``repitition = max_iter``).
    """
    basis = _polynomial_basis(y.size, degree)
    projector = np.linalg.pinv(basis)
    work = y
    for _ in range(max_iter + 1):
        fit = basis @ (projector @ work)
        clipped = np.minimum(y if use_original else work, fit)
        change = _summed_relative_change(work, clipped)
        work = clipped
        if change < tol:
            break
    return fit


def _imodpoly(y, degree, max_iter, tol):
    """IModPoly (Zhao et al. 2007): ModPoly with the residual deviation DEV as threshold.

    1. Fit the whole spectrum; ``DEV`` = standard deviation of the residual.
    2. Remove the peak points ``y > fit + DEV`` for good (first pass only).
    3. Refit the remaining points, ``y := min(y, fit + DEV)``; repeat until
       ``|DEV_i - DEV_(i-1)| / DEV_i < tol`` or ``max_iter`` fits in total.

    The last polynomial is evaluated over the whole spectrum. Same loop as
    ``BaselineRemoval.IModPoly`` (``repitition = max_iter``). The peak test ``y > fit + DEV``
    allows for rounding (``64·eps·max|y|``): for an exactly polynomial spectrum DEV is itself
    rounding noise, and a strict test could remove every point (an empty or underdetermined
    refit). If fewer than ``degree + 1`` points would remain, the first fit is returned.
    """
    basis = _polynomial_basis(y.size, degree)
    coef = np.linalg.pinv(basis) @ y
    fit = basis @ coef
    previous = float(np.std(y - fit))
    if max_iter > 1:
        rounding = 64.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(y))))
        keep = (y - fit) <= previous + rounding
        if np.count_nonzero(keep) <= degree:
            return fit
        sub_basis = basis[keep]
        projector = np.linalg.pinv(sub_basis)
        work = y[keep]
        for _ in range(max_iter - 1):
            coef = projector @ work
            sub_fit = sub_basis @ coef
            deviation = float(np.std(work - sub_fit))
            if deviation == 0.0 or abs(deviation - previous) / deviation < tol:
                break
            work = np.minimum(work, sub_fit + deviation)
            previous = deviation
    return basis @ coef


# ──────────────────────────── penalized least squares ─────────────────────────────

@lru_cache(maxsize=16)
def _difference_penalty(n: int, diff_order: int) -> np.ndarray:
    """Lower bands of ``D'D`` for the ``(n - d, n)`` ``d``-th order difference matrix ``D``.

    Layout of :func:`scipy.linalg.solveh_banded` with ``lower=True``: row ``k`` holds the
    ``k``-th sub-diagonal, ``bands[k, j] = (D'D)[j + k, j]``. Cached and read-only.
    """
    d = diff_order
    coefficients = [(-1) ** (d - j) * math.comb(d, j) for j in range(d + 1)]
    difference = sparse.diags(coefficients, list(range(d + 1)), shape=(n - d, n), format="csr")
    gram = (difference.T @ difference).tocsr()
    bands = np.zeros((d + 1, n))
    for k in range(d + 1):
        bands[k, :n - k] = gram.diagonal(-k)
    bands.setflags(write=False)
    return bands


def _whittaker_solve(penalty: np.ndarray, weights: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve ``(W + P) z = W y``; ``penalty`` = lower bands of ``P = lam·D'D`` (not modified).

    Banded Cholesky first; sparse LU when Cholesky fails or returns non-finite values.
    """
    lhs = penalty.copy()
    lhs[0] += weights
    try:
        z = solveh_banded(lhs, weights * y, lower=True, overwrite_ab=True, overwrite_b=True,
                          check_finite=False)
        if np.isfinite(z).all():
            return z
    except (LinAlgError, ValueError):
        pass
    return _sparse_solve(penalty, weights, y)


def _sparse_solve(penalty: np.ndarray, weights: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sparse-LU fallback of :func:`_whittaker_solve` (CSC matrix, ``spsolve``)."""
    n = y.size
    order = penalty.shape[0] - 1
    diagonals = [penalty[0] + weights]
    offsets = [0]
    for k in range(1, order + 1):
        diagonals += [penalty[k, :n - k], penalty[k, :n - k]]
        offsets += [-k, k]
    system = sparse.diags(diagonals, offsets, shape=(n, n), format="csc")
    z = np.asarray(spsolve(system, weights * y), dtype=np.float64).reshape(n)
    if not np.isfinite(z).all():
        raise ValueError("penalized least-squares system is singular or too ill-conditioned "
                         f"(n={n}, diff_order={order}); reduce lam")
    return z


def _relative_difference(old: np.ndarray, new: np.ndarray) -> float:
    """``||new - old|| / ||old||`` (2-norm, denominator at least machine epsilon)."""
    return float(np.linalg.norm(new - old) / max(float(np.linalg.norm(old)), _EPS))


def _airpls(y, lam, diff_order, max_iter, tol, end_weights=True):
    """airPLS / ZhangFit (Zhang, Chen & Liang 2010), as the reference ``airPLS.py``.

    Iteration ``t = 1, 2, ...``: smooth with weights ``w``; residual ``d = y - z``;
    stop when ``|sum(d[d < 0])| < tol * sum|y|`` or ``t == max_iter``; otherwise
    ``w = 0`` where ``d >= 0``, ``w = exp(t·|d| / |sum(d[d<0])|)`` where ``d < 0`` and, with
    ``end_weights`` (the reference), both end points get ``exp(t·max(d[d<0]) / |sum(d[d<0])|)``
    so they never lose their weight. ``end_weights=False`` is pybaselines' variant.
    """
    penalty = lam * _difference_penalty(y.size, diff_order)
    weights = np.ones(y.size)
    total = float(np.abs(y).sum())
    for iteration in range(1, max_iter + 1):
        z = _whittaker_solve(penalty, weights, y)
        residual = y - z
        negative = residual < 0
        if not negative.any():
            break                                   # z lies below all data: nothing to reweight
        negative_residual = residual[negative]
        mass = abs(float(negative_residual.sum()))
        if mass < tol * total or iteration == max_iter:
            break
        t = min(iteration, _MAX_EXPONENT_ITERATION)
        weights = np.zeros(y.size)
        weights[negative] = np.exp(t * np.abs(negative_residual) / mass)
        if end_weights:
            weights[0] = weights[-1] = math.exp(t * float(negative_residual.max()) / mass)
    return z


def _asls(y, lam, p, diff_order, max_iter, tol):
    """AsLS (Eilers & Boelens 2005), as ``pybaselines.whittaker.asls``.

    Smooth with weights ``w`` (start 1), then ``w = p`` where ``y > z`` and ``1 - p``
    elsewhere; stop when the relative change of ``w`` is below ``tol`` or after
    ``max_iter + 1`` fits. Returns the last fit.
    """
    penalty = lam * _difference_penalty(y.size, diff_order)
    weights = np.ones(y.size)
    for _ in range(max_iter + 1):
        z = _whittaker_solve(penalty, weights, y)
        new_weights = np.where(y > z, p, 1.0 - p)
        if _relative_difference(weights, new_weights) < tol:
            break
        weights = new_weights
    return z


def _arpls(y, lam, diff_order, max_iter, tol):
    """arPLS (Baek et al. 2015), as ``pybaselines.whittaker.arpls``.

    Weights ``w = 1 / (1 + exp(2·(d - (2s - m)) / s))`` from the mean ``m`` and the
    standard deviation ``s`` (``ddof=1``) of the negative residuals ``d < 0``; stop when the
    relative change of ``w`` is below ``tol``, when fewer than two residuals are negative,
    or after ``max_iter + 1`` fits. Returns the last fit.
    """
    penalty = lam * _difference_penalty(y.size, diff_order)
    weights = np.ones(y.size)
    for _ in range(max_iter + 1):
        z = _whittaker_solve(penalty, weights, y)
        residual = y - z
        negative_residual = residual[residual < 0]
        if negative_residual.size < 2:
            break
        mean = float(negative_residual.mean())
        std = float(negative_residual.std(ddof=1)) or _EPS
        new_weights = expit(-(2.0 / std) * (residual - (2.0 * std - mean)))
        if _relative_difference(weights, new_weights) < tol:
            break
        weights = new_weights
    return z


_METHODS: Dict[str, Callable[..., np.ndarray]] = {
    "modpoly": _modpoly, "imodpoly": _imodpoly, "zhangfit": _airpls, "asls": _asls,
    "arpls": _arpls,
}
