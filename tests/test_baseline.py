"""Baseline removers: accuracy on synthetic spectra, API contract, agreement with the references.

Reference agreement (``BaselineRemoval`` 0.1.x and ``pybaselines`` 1.2, skipped when absent) is
asserted to ``1e-9 * max|y|``: both sides run the same iterations and take the same decisions,
so only floating-point rounding differs. Measured: <= 1e-12 against BaselineRemoval (raw-index
Vandermonde + scikit-learn least squares, SuperLU) and bit-identical against pybaselines (the
same LAPACK banded Cholesky). The margin covers other BLAS/LAPACK builds.
"""
import contextlib
import doctest
import inspect
import io
import time
import warnings

import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import LinAlgError
from scipy.sparse.linalg import spsolve

from modules.spectral_resolution import baseline as bl
from modules.spectral_resolution.baseline import (BASELINE_LABELS, BASELINE_METHODS, DEFAULT_PARAMS,
                                                  PARAM_HELP, estimate_baseline, remove_baseline)

N = 1600                     # spectral columns of the camera
X = np.arange(N, dtype=float)
NOISE = 5.0                  # counts (sigma)
PEAKS = ((200, 2500.0, 3.0), (420, 900.0, 6.0), (700, 3000.0, 2.5),
         (760, 1500.0, 4.0), (1100, 2000.0, 8.0), (1400, 1200.0, 3.0))   # centre px, height, sigma px
FAR = np.all([np.abs(X - c) > 6 * s for c, _, s in PEAKS], axis=0)     # samples away from every line
REL = 1e-9                   # reference agreement, relative to max|y| (see module docstring)


def peaks():
    """Gaussian emission lines, heights in counts."""
    return sum(h * np.exp(-0.5 * ((X - c) / s) ** 2) for c, h, s in PEAKS)


def polynomial_background():
    """Quadratic background (what a degree-2 polynomial method can represent exactly)."""
    u = (X - N / 2) / (N / 2)
    return 800 + 300 * u + 250 * u ** 2


def smooth_background():
    """Quadratic plus a slow sine: smooth but not a polynomial."""
    return polynomial_background() + 60 * np.sin(2 * np.pi * 1.3 * X / N)


def synthetic(background, seed=0):
    """Background + lines + Gaussian noise."""
    return background + peaks() + np.random.default_rng(seed).normal(0, NOISE, N)


def assert_close(mine, reference, y):
    """Agreement with a reference implementation, relative to the data scale."""
    error = np.abs(np.asarray(mine) - np.asarray(reference)).max() / np.abs(y).max()
    assert error <= REL, f"max deviation {error:.2e} x max|y| > {REL:g}"


# ─────────────────────────────── accuracy on known spectra ───────────────────────────────

# (method, params, background, unbiased). Polynomial methods get a background of their degree.
# ZhangFit runs with lam=1e5 and second differences: its reference default (lam=100, first
# differences) is very flexible and creeps ~2 % into a 19 px wide line - the same in airPLS.py.
# ModPoly / AsLS / airPLS fit the lower envelope of the noise (bias ~ -2 sigma) by design.
ACCURACY_CASES = [
    ("modpoly", {}, polynomial_background, False),
    ("modpoly", {"use_original": True}, polynomial_background, False),
    ("imodpoly", {}, polynomial_background, True),
    ("zhangfit", {"lam": 1e5, "diff_order": 2}, smooth_background, False),
    ("asls", {}, smooth_background, False),
    ("arpls", {}, smooth_background, True),
]


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("method, params, background, unbiased", ACCURACY_CASES,
                         ids=[f"{c[0]}{'-' + str(c[1]) if c[1] else ''}" for c in ACCURACY_CASES])
def test_known_baseline_is_recovered_and_peak_heights_are_kept(method, params, background, unbiased, seed):
    truth = background()
    y = synthetic(truth, seed)
    corrected, base = remove_baseline(y, method, **params)
    error = base - truth
    assert np.abs(error[FAR]).max() < 4 * NOISE
    if unbiased:
        assert abs(error[FAR].mean()) < NOISE / 2
    for centre, height, _ in PEAKS:
        assert corrected[centre] == pytest.approx(height, rel=0.03), (centre, corrected[centre])


@pytest.mark.parametrize("method", BASELINE_METHODS)
@pytest.mark.parametrize("level", [0.0, 1234.5, -50.0])
def test_constant_spectrum_is_its_own_baseline(method, level):
    y = np.full(300, level)
    corrected, base = remove_baseline(y, method)
    expected = 0.0 if method == "none" else level
    assert np.allclose(base, expected, rtol=0, atol=1e-6 * max(1.0, abs(level)))
    assert np.allclose(corrected, level - expected, rtol=0, atol=1e-6 * max(1.0, abs(level)))


@pytest.mark.parametrize("n", [10, 100, 400, 1600, 1608])
@pytest.mark.parametrize("degree", [0, 1, 2, 3, 5])
def test_imodpoly_returns_an_exactly_polynomial_spectrum(n, degree):
    """DEV is rounding noise there; the strict peak test used to drop every point (baseline 0)."""
    if n < 2 * (degree + 1):
        pytest.skip("too short for this degree")
    t = np.arange(n, dtype=float)
    cases = [(0, np.full(n, level)) for level in (3.7, 1.0, 0.1, -7.3, 1234.5)]
    cases += [(1, 100 + 0.37 * t), (2, 3 + 1e-3 * (t - n / 3) ** 2)]           # (lowest degree, spectrum)
    for lowest, y in cases:
        if degree < lowest:
            continue                                              # not representable at this degree
        with warnings.catch_warnings():
            warnings.simplefilter("error")                        # no empty-slice / divide warnings either
            base = estimate_baseline(y, "imodpoly", degree=degree)
        assert np.abs(base - y).max() < 1e-9 * max(1.0, np.abs(y).max()), (n, degree, y[:2])


def test_none_returns_a_zero_baseline():
    y = synthetic(smooth_background())
    corrected, base = remove_baseline(y, "none")
    assert base.dtype == np.float64 and not base.any()
    assert np.array_equal(corrected, y)


# ───────────────────────────────────── API contract ─────────────────────────────────────

def test_method_tables_are_complete():
    assert BASELINE_METHODS == ("none", "modpoly", "imodpoly", "zhangfit", "asls", "arpls")
    assert BASELINE_LABELS == {"none": "None", "modpoly": "ModPoly", "imodpoly": "IModPoly",
                               "zhangfit": "ZhangFit (airPLS)", "asls": "AsLS", "arpls": "arPLS"}
    assert tuple(DEFAULT_PARAMS) == BASELINE_METHODS and tuple(PARAM_HELP) == BASELINE_METHODS
    assert DEFAULT_PARAMS["none"] == {}
    for method in BASELINE_METHODS:
        assert set(PARAM_HELP[method]) == set(DEFAULT_PARAMS[method])
        assert all(isinstance(text, str) and text for text in PARAM_HELP[method].values())


@pytest.mark.parametrize("method", BASELINE_METHODS[1:])
def test_every_default_parameter_is_accepted_explicitly(method):
    y = synthetic(smooth_background())
    assert np.array_equal(estimate_baseline(y, method, **DEFAULT_PARAMS[method]), estimate_baseline(y, method))


def test_editing_default_params_does_not_change_behaviour(monkeypatch):
    y = synthetic(smooth_background())
    before = estimate_baseline(y, "asls")
    monkeypatch.setitem(DEFAULT_PARAMS["asls"], "lam", 10.0)
    assert np.array_equal(estimate_baseline(y, "asls"), before)


def test_defaults_are_those_of_the_reference_implementations():
    assert DEFAULT_PARAMS["modpoly"] == {"degree": 2, "max_iter": 100, "tol": 1e-3, "use_original": False}
    assert DEFAULT_PARAMS["imodpoly"] == {"degree": 2, "max_iter": 100, "tol": 1e-3}
    assert DEFAULT_PARAMS["zhangfit"] == {"lam": 100.0, "diff_order": 1, "max_iter": 15, "tol": 1e-3}
    assert DEFAULT_PARAMS["asls"] == {"lam": 1e6, "p": 1e-2, "diff_order": 2, "max_iter": 50, "tol": 1e-3}
    assert DEFAULT_PARAMS["arpls"] == {"lam": 1e5, "diff_order": 2, "max_iter": 50, "tol": 1e-3}


def test_defaults_match_the_installed_references():
    reference = pytest.importorskip("BaselineRemoval").BaselineRemoval
    whittaker = pytest.importorskip("pybaselines.whittaker")

    def defaults(function):
        return {k: p.default for k, p in inspect.signature(function).parameters.items()
                if p.default is not inspect.Parameter.empty}

    for name, ours in (("ModPoly", "modpoly"), ("IModPoly", "imodpoly")):
        ref = defaults(getattr(reference, name))
        assert (ref["degree"], ref["repitition"], ref["gradient"]) == (
            DEFAULT_PARAMS[ours]["degree"], DEFAULT_PARAMS[ours]["max_iter"], DEFAULT_PARAMS[ours]["tol"])
    ref = defaults(reference.ZhangFit)
    assert (ref["lambda_"], ref["porder"], ref["repitition"]) == (100, 1, 15)
    for name in ("asls", "arpls"):
        ref = defaults(getattr(whittaker, name))
        assert {k: ref[k] for k in DEFAULT_PARAMS[name]} == DEFAULT_PARAMS[name]


@pytest.mark.parametrize("method", BASELINE_METHODS)
def test_input_is_not_modified_and_output_is_new_float64(method):
    y = synthetic(smooth_background())
    kept = y.copy()
    base = estimate_baseline(y, method)
    corrected, base2 = remove_baseline(y, method)
    assert np.array_equal(y, kept)
    assert base.dtype == np.float64 and base.shape == y.shape and base is not y
    assert np.array_equal(base, base2) and np.array_equal(corrected, y - base2)
    assert not np.shares_memory(corrected, y) and not np.shares_memory(base, y)

    counts = np.clip(np.round(y), 0, None).astype(np.uint16)       # raw camera counts
    counts_kept = counts.copy()
    corrected16, base16 = remove_baseline(counts, method)
    assert np.array_equal(counts, counts_kept)
    assert corrected16.dtype == np.float64 and base16.dtype == np.float64
    assert np.array_equal(base16, estimate_baseline(counts.astype(float), method))
    assert np.array_equal(estimate_baseline(list(counts), method), base16)
    assert np.array_equal(estimate_baseline(counts.astype(np.float32), method), base16)


@pytest.mark.parametrize("name", ["ArPLS", "ASLS", " zhangfit ", "ModPoly", "IMODPOLY", "None"])
def test_method_names_are_case_insensitive(name):
    y = synthetic(smooth_background())
    assert np.array_equal(estimate_baseline(y, name), estimate_baseline(y, name.strip().lower()))


@pytest.mark.parametrize("method", ["foo", "", "airpls", None, 3])
def test_unknown_method_is_rejected_with_the_valid_names(method):
    with pytest.raises(ValueError) as info:
        estimate_baseline(np.ones(50), method)
    assert all(name in str(info.value) for name in BASELINE_METHODS)
    with pytest.raises(ValueError):
        remove_baseline(np.ones(50), method)


@pytest.mark.parametrize("method, wrong", [("asls", "lamda"), ("modpoly", "lam"), ("arpls", "p"),
                                           ("zhangfit", "porder"), ("imodpoly", "use_original")])
def test_unknown_parameter_is_rejected_with_the_valid_names(method, wrong):
    with pytest.raises(ValueError) as info:
        estimate_baseline(np.ones(50), method, **{wrong: 1})
    assert wrong in str(info.value)
    assert all(name in str(info.value) for name in DEFAULT_PARAMS[method])


def test_method_none_takes_no_parameters():
    with pytest.raises(ValueError, match="no parameters"):
        estimate_baseline(np.ones(50), "none", lam=1e5)


@pytest.mark.parametrize("method, params", [
    ("modpoly", {"degree": -1}), ("modpoly", {"degree": 2.5}), ("modpoly", {"degree": True}),
    ("modpoly", {"degree": "2"}), ("modpoly", {"max_iter": -1}), ("modpoly", {"tol": -1e-3}),
    ("modpoly", {"tol": float("nan")}), ("modpoly", {"use_original": 1}), ("modpoly", {"use_original": "yes"}),
    ("imodpoly", {"max_iter": 0}), ("imodpoly", {"degree": None}),
    ("zhangfit", {"max_iter": 0}), ("zhangfit", {"lam": 0}), ("zhangfit", {"diff_order": 0}),
    ("asls", {"lam": -1.0}), ("asls", {"lam": float("inf")}), ("asls", {"p": 0.0}), ("asls", {"p": 1.0}),
    ("asls", {"p": 1.5}), ("asls", {"diff_order": 4}), ("asls", {"max_iter": 2.5}),
    ("arpls", {"diff_order": 1.5}), ("arpls", {"tol": "0.001"}), ("arpls", {"lam": False}),
])
def test_out_of_range_parameters_are_rejected(method, params):
    with pytest.raises(ValueError):
        estimate_baseline(synthetic(smooth_background()), method, **params)


def test_integral_floats_are_accepted_for_integer_parameters():
    y = synthetic(smooth_background())
    assert np.array_equal(estimate_baseline(y, "modpoly", degree=3.0, max_iter=np.int64(20)),
                          estimate_baseline(y, "modpoly", degree=3, max_iter=20))
    assert np.array_equal(estimate_baseline(y, "asls", lam=10 ** 6, diff_order=2.0),
                          estimate_baseline(y, "asls"))


@pytest.mark.parametrize("method", BASELINE_METHODS)
@pytest.mark.parametrize("bad", [np.ones((4, 50)), np.float64(3.0), np.array([]), [1.0, np.nan, 2.0] * 10,
                                 [1.0, np.inf, 2.0] * 10, np.ones(30) + 1j, np.array(["1"] * 30)],
                         ids=["2-D", "0-D", "empty", "nan", "inf", "complex", "strings"])
def test_bad_spectra_are_rejected(method, bad):
    with pytest.raises(ValueError):
        estimate_baseline(bad, method)
    with pytest.raises(ValueError):
        remove_baseline(bad, method)


@pytest.mark.parametrize("method, params, shortest", [
    ("none", {}, 1), ("modpoly", {}, 6), ("modpoly", {"degree": 0}, 2), ("imodpoly", {"degree": 4}, 10),
    ("zhangfit", {}, 3), ("asls", {}, 4), ("arpls", {"diff_order": 3}, 5), ("arpls", {"diff_order": 1}, 3),
])
def test_too_short_spectra_are_rejected(method, params, shortest):
    rng = np.random.default_rng(0)
    base = estimate_baseline(100 + rng.normal(0, 1, shortest), method, **params)
    assert base.shape == (shortest,) and np.isfinite(base).all()
    with pytest.raises(ValueError, match="at least"):
        estimate_baseline(100 + rng.normal(0, 1, shortest - 1) if shortest > 1 else [], method, **params)


def test_module_docstring_example_runs():
    result = doctest.testmod(bl, optionflags=doctest.ELLIPSIS)
    assert result.attempted > 0 and result.failed == 0


# ─────────────────────────── agreement with the references ───────────────────────────

@pytest.fixture(scope="module")
def y_ref():
    """Positive spectrum (BaselineRemoval's ModPoly criterion divides by the data)."""
    return synthetic(smooth_background(), seed=7)


def _baseline_removal():
    return pytest.importorskip("BaselineRemoval").BaselineRemoval


@pytest.mark.parametrize("degree", [1, 2, 3, 5])
@pytest.mark.parametrize("repitition, gradient", [(100, 1e-3), (30, 0.0), (400, 1e-3), (100, 0.2)])
def test_modpoly_matches_baselineremoval(y_ref, degree, repitition, gradient):
    """With n = 1600 the summed criterion hardly ever falls below 1e-3 (all refits run);
    ``gradient=0.2`` stops after 25-59 fits and checks that both stop at the same one."""
    reference = _baseline_removal()(y_ref.copy()).ModPoly(degree, repitition, gradient)    # corrected
    ours = remove_baseline(y_ref, "modpoly", degree=degree, max_iter=repitition, tol=gradient,
                           use_original=True)[0]
    assert_close(ours, reference, y_ref)


def test_modpoly_keeps_iterating_over_exact_zeros(y_ref):
    """BaselineRemoval divides by the data: an unchanged 0 gives 0/0 = NaN and stops it after one
    fit. Here an unchanged 0 is no change, so the result is BaselineRemoval's for a tiny value."""
    y = y_ref.copy()
    y[[0, 900]] = 0.0
    tiny = y.copy()
    tiny[[0, 900]] = 1e-300
    reference = _baseline_removal()(tiny).ModPoly(2, 100, 1e-3)
    assert_close(remove_baseline(y, "modpoly", use_original=True)[0], reference, y)


@pytest.mark.parametrize("degree", [2, 4])
@pytest.mark.parametrize("use_original", [False, True])
def test_modpoly_matches_pybaselines_for_fixed_iterations(y_ref, degree, use_original):
    """pybaselines stops on the change of the fit, BaselineRemoval (and we) on the change of the
    clipped data; with ``tol=0`` both run exactly ``max_iter`` refits."""
    polynomial = pytest.importorskip("pybaselines.polynomial")
    reference, _ = polynomial.modpoly(y_ref, poly_order=degree, tol=0.0, max_iter=60, use_original=use_original)
    ours = estimate_baseline(y_ref, "modpoly", degree=degree, max_iter=60, tol=0.0, use_original=use_original)
    assert_close(ours, reference, y_ref)


def test_modpoly_variants_differ():
    """Lieber's y := min(y, fit) (default) is not BaselineRemoval's y := min(y0, fit): they part
    where a later fit rises above an earlier one (here: a strongly curved background)."""
    x = np.arange(400.0)
    y = 200 + 0.002 * (x - 150) ** 2 + 1000 * np.exp(-0.5 * ((x - 150) / 3) ** 2)
    assert np.abs(estimate_baseline(y, "modpoly") - estimate_baseline(y, "modpoly", use_original=True)).max() > 1


@pytest.mark.parametrize("degree", [1, 2, 3, 5])
@pytest.mark.parametrize("repitition, gradient", [(100, 1e-3), (100, 1e-6), (5, 1e-3), (1, 1e-3)])
def test_imodpoly_matches_baselineremoval(y_ref, degree, repitition, gradient):
    reference = _baseline_removal()(y_ref.copy()).IModPoly(degree, repitition, gradient)
    ours = remove_baseline(y_ref, "imodpoly", degree=degree, max_iter=repitition, tol=gradient)[0]
    assert_close(ours, reference, y_ref)


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")    # np.matrix inside BaselineRemoval
@pytest.mark.parametrize("lam", [10.0, 100.0, 1e4, 1e6])
@pytest.mark.parametrize("repitition", [1, 2, 3, 15, 30])
def test_zhangfit_matches_baselineremoval(y_ref, lam, repitition):
    """BaselineRemoval ignores ``porder`` (always first differences), so only diff_order=1 is compared."""
    with contextlib.redirect_stdout(io.StringIO()):                 # it prints "max iteration reached"
        reference = _baseline_removal()(y_ref.copy()).ZhangFit(lam, 1, repitition)
    ours = remove_baseline(y_ref, "zhangfit", lam=lam, diff_order=1, max_iter=repitition)[0]
    assert_close(ours, reference, y_ref)


@pytest.mark.parametrize("lam", [100.0, 1e5, 1e7])
@pytest.mark.parametrize("diff_order", [1, 2, 3])
def test_airpls_matches_pybaselines_without_the_end_point_weights(y_ref, lam, diff_order):
    """pybaselines' airpls = airPLS.py minus the end-point weights; its ``max_iter`` counts
    re-weightings (one fit fewer than airPLS's ``itermax``)."""
    whittaker = pytest.importorskip("pybaselines.whittaker")
    reference, _ = whittaker.airpls(y_ref, lam=lam, diff_order=diff_order, max_iter=14, tol=1e-3)
    ours = bl._airpls(bl._as_spectrum(y_ref), lam, diff_order, 15, 1e-3, end_weights=False)
    assert_close(ours, reference, y_ref)


ASLS_CASES = [dict(lam=lam, p=p, diff_order=d) for lam in (1e3, 1e6, 1e9) for p in (1e-3, 1e-2, 0.1)
              for d in (1, 2, 3)] + [dict(max_iter=2, tol=0.0), dict(tol=1e-8), dict(max_iter=0)]


@pytest.mark.parametrize("params", ASLS_CASES, ids=[str(p) for p in ASLS_CASES])
def test_asls_matches_pybaselines(y_ref, params):
    whittaker = pytest.importorskip("pybaselines.whittaker")
    reference, _ = whittaker.asls(y_ref, **params)
    assert_close(estimate_baseline(y_ref, "asls", **params), reference, y_ref)


ARPLS_CASES = [dict(lam=lam, diff_order=d) for lam in (1e3, 1e5, 1e7, 1e9) for d in (1, 2, 3)] + [
    dict(max_iter=3), dict(tol=1e-6), dict(max_iter=0)]


@pytest.mark.parametrize("params", ARPLS_CASES, ids=[str(p) for p in ARPLS_CASES])
def test_arpls_matches_pybaselines(y_ref, params):
    whittaker = pytest.importorskip("pybaselines.whittaker")
    reference, _ = whittaker.arpls(y_ref, **params)
    assert_close(estimate_baseline(y_ref, "arpls", **params), reference, y_ref)


# ──────────────────────────────── solver and speed ────────────────────────────────

def _sparse_reference(weights, y, lam, diff_order):
    """``(W + lam·D'D) z = W y`` the textbook way (scipy.sparse, CSC, SuperLU)."""
    difference = sparse.eye(y.size, format="csc")
    for _ in range(diff_order):
        difference = difference[1:] - difference[:-1]
    system = (sparse.diags(weights, 0, format="csc") + lam * (difference.T @ difference)).tocsc()
    return spsolve(system, weights * y)


@pytest.mark.parametrize("lam", [1e2, 1e5, 1e8])
@pytest.mark.parametrize("diff_order", [1, 2, 3])
def test_banded_solver_matches_sparse_lu(lam, diff_order):
    """Agreement is limited by the conditioning (~lam·4^d / min w), not by the method."""
    rng = np.random.default_rng(5)
    n = 4000
    y = 600 + 0.05 * np.arange(n) + 40 * np.sin(np.arange(n) / 300) + rng.normal(0, 5, n)
    weights = rng.uniform(0.01, 1.0, n)
    ours = bl._whittaker_solve(lam * bl._difference_penalty(n, diff_order), weights, y)
    assert np.abs(ours - _sparse_reference(weights, y, lam, diff_order)).max() <= 1e-6 * np.abs(y).max()


def test_cholesky_failure_falls_back_to_sparse_lu(monkeypatch):
    y = synthetic(smooth_background())
    expected = {m: estimate_baseline(y, m) for m in ("zhangfit", "asls", "arpls")}

    def not_positive_definite(*args, **kwargs):
        raise LinAlgError("1-th leading minor not positive definite")

    monkeypatch.setattr(bl, "solveh_banded", not_positive_definite)
    for method, baseline in expected.items():
        assert_close(estimate_baseline(y, method), baseline, y)


@pytest.mark.parametrize("method", ["zhangfit", "asls", "arpls"])
def test_stiff_long_spectra_stay_finite_and_accurate(method):
    """n = 4000, lam = 1e12: still a finite, sensible baseline and no solver warning."""
    n = 4000
    x = np.arange(n, dtype=float)
    line = 500 + 0.05 * x
    y = line + np.random.default_rng(3).normal(0, NOISE, n)
    for centre in range(300, n, 500):
        y += 2000 * np.exp(-0.5 * ((x - centre) / 4) ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stiff = estimate_baseline(y, method, lam=1e12, diff_order=2)
        first = estimate_baseline(y, method, lam=1e12, diff_order=1)
    assert np.isfinite(stiff).all() and np.isfinite(first).all()
    assert np.abs(stiff - line).max() < 2 * NOISE


def test_all_methods_are_fast():
    """One analysis calls this ~70 times per method; each call takes ~0.1-2 ms here."""
    y = synthetic(smooth_background())
    for method in BASELINE_METHODS:                  # warm caches
        estimate_baseline(y, method)
    start = time.perf_counter()
    for _ in range(5):
        for method in BASELINE_METHODS:
            estimate_baseline(y, method)
    assert (time.perf_counter() - start) / 5 < 0.5   # all six methods, n = 1600
