"""Spectral quality diagnostics, independent of Qt and correction fitting.

Coordinates are zero-based: rows follow the slit; columns follow wavelength.
Input pairs must share an uncropped detector grid. Results distinguish absolute
wavelength error from relative line straightness. A validation frame separate
from the fitting frame is required for an independent accuracy claim.

Progress and logging:
    ``analyze_quality`` and ``estimate_shift_uncertainty`` accept ``progress``, a callable
    ``progress(fraction, message)`` (fraction 0..1) called from the computing thread, and log
    start/finish with timings to the ``modules.smile.quality`` logger (per-line detail at DEBUG).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time

import numpy as np

# np.trapezoid exists from NumPy 2.0; NumPy 1.x only has np.trapz (same result).
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz
from scipy.optimize import least_squares, minimize_scalar
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


def _report(progress, fraction, message):
    """Call ``progress(fraction, message)`` if given; a failing callback never stops the analysis."""
    if progress is not None:
        try:
            progress(float(fraction), message)
        except Exception:  # noqa: BLE001
            logger.debug("progress callback failed", exc_info=True)


def _gaussian_residual(p, x, origin, v):
    """Gaussian plus linear background minus data: ``a*exp(-z^2/2) + bg + slope*(x-origin) - v``."""
    return p[0] * np.exp(-0.5 * ((x-p[1])/p[2])**2) + p[3] + p[4]*(x-origin) - v


def _gaussian_jacobian(p, x, origin, v):
    """Analytic Jacobian of :func:`_gaussian_residual` (replaces finite differences; same model)."""
    a, mu, sigma = p[0], p[1], p[2]
    z = (x - mu) / sigma
    e = np.exp(-0.5 * z * z)
    jac = np.empty((x.size, 5))
    jac[:, 0] = e
    jac[:, 1] = a * e * z / sigma
    jac[:, 2] = a * e * z * z / sigma
    jac[:, 3] = 1.0
    jac[:, 4] = x - origin
    return jac

__all__ = ['SpectralReference', 'QualityReport', 'analyze_quality',
           'estimate_shift_uncertainty', 'correlation_shift']


@dataclass(frozen=True)
class SpectralReference:
    """Known emission wavelength and expected column on the corrected grid.

    ``half_window_px`` bounds a local Gaussian-plus-linear-background fit.
    Mark unresolved or contaminated features with ``blended=True``; they are
    excluded from single-line wavelength, FWHM, and area measurements.
    """
    wavelength_nm: float
    column_px: float
    half_window_px: int = 20
    blended: bool = False


@dataclass
class QualityReport:
    """Serializable row-resolved measurements and paired quality summaries.

    Missing/rejected measurements are NaN internally and null in JSON.
    Areas are detector counts times pixels, not calibrated radiance.
    """
    rows: np.ndarray
    wavelengths_nm: np.ndarray
    before: dict
    after: dict
    summary: dict
    metadata: dict

    def save(self, path: str) -> None:
        """Write strict JSON without non-standard NaN/Infinity tokens."""
        def clean(value):
            if isinstance(value, np.ndarray):
                return clean(value.tolist())
            if isinstance(value, dict):
                return {str(k): clean(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            if isinstance(value, (float, np.floating)):
                return float(value) if np.isfinite(value) else None
            if isinstance(value, np.integer):
                return int(value)
            return value
        Path(path).write_text(json.dumps(clean(asdict(self)), indent=2,
                                        allow_nan=False), encoding='utf-8')


def _frame(value):
    """Validate and convert to a float64 2-D frame.

    Raises:
        ValueError: Not 2-D, any dimension < 3, or non-finite values.
    """
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or min(value.shape) < 3 or not np.isfinite(value).all():
        raise ValueError('Expected a finite 2-D frame with dimensions >= 3.')
    return value


def _stats(values):
    """Count, bias, RMSE, 95th-percentile and maximum absolute value of the finite entries
    (NaN for each statistic when there are none).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return dict(count=0, bias=np.nan, rmse=np.nan, p95_abs=np.nan,
                    max_abs=np.nan)
    return dict(count=int(values.size), bias=float(values.mean()),
                rmse=float(np.sqrt(np.mean(values ** 2))),
                p95_abs=float(np.percentile(np.abs(values), 95)),
                max_abs=float(np.max(np.abs(values))))


def _peak(profile, center, halfwin, snr_min, saturation):
    """Fit a local isolated line; reject clipped, weak, or boundary fits."""
    lo = max(0, int(np.floor(center)) - halfwin)
    hi = min(profile.size, int(np.floor(center)) + halfwin + 1)
    x = np.arange(lo, hi, dtype=float)
    v = profile[lo:hi]
    if v.size < 9 or (saturation is not None and np.max(v) >= saturation):
        return None
    origin = x.mean()
    baseline = np.polyfit(x[[0, 1, -2, -1]] - origin, v[[0, 1, -2, -1]], 1)
    signal = v - np.polyval(baseline, x - origin)
    amplitude = float(signal.max())
    noise = np.median(np.abs(np.diff(v) - np.median(np.diff(v)))) / 0.67449 / np.sqrt(2)
    noise = max(float(noise), np.finfo(float).eps * max(1., abs(v).max()))
    if amplitude <= snr_min * noise:
        return None
    peak = float(x[np.argmax(signal)])
    if peak <= x[1] or peak >= x[-2]:
        return None
    fit = least_squares(
        _gaussian_residual,
        [amplitude, peak, max(1., halfwin/4), baseline[1], baseline[0]],
        jac=_gaussian_jacobian, args=(x, origin, v),
        bounds=([0, x[1], 0.35, -np.inf, -np.inf],
                [np.inf, x[-2], halfwin/2, np.inf, np.inf]),
        max_nfev=150,
    )
    a, mu, sigma, bg, slope = fit.x
    if not fit.success or mu-3*sigma < x[0] or mu+3*sigma > x[-1]:
        return None
    residual_rms = float(np.sqrt(np.mean(fit.fun**2)))
    if a / max(noise, residual_rms) < snr_min:
        return None
    area = float(_trapezoid(v-bg-slope*(x-origin), x))
    return mu, sigma*2.354820045, a, area, residual_rms


# Batched Levenberg-Marquardt of _peaks: iteration cap; rows that need more than _LM_SLOW
# iterations, end within 1e-6 of a bound or with sigma below _LM_MIN_SIGMA (near the 0.35 px
# bound scipy's trust region crawls and can run out of max_nfev, i.e. reject the row) are
# refitted by _peak itself.
_LM_MAX_ITER = 60
_LM_SLOW = 30
_LM_MIN_SIGMA = 0.55


def _peaks(profiles, center, halfwin, snr_min, saturation):
    """:func:`_peak` for every row of ``profiles`` at once: ``(n_rows, 5)`` array.

    Columns ``(mu, fwhm_px, amplitude, area, residual_rms)``; a row :func:`_peak` would
    reject is all NaN. The window is the same for every row (it depends only on
    ``center``), so the pre-checks (saturation, 4-point baseline, MAD noise, SNR, peak
    position) are evaluated on the whole ``(rows, window)`` block, and the Gaussian +
    linear-background fits run as one batched, bounded Levenberg-Marquardt (same model,
    analytic Jacobian, start values and bounds as :func:`_peak`). It converges to the same
    minimum as scipy's trust-region fit (differences below scipy's own ``ftol``/``xtol``,
    ~1e-3 of the fit's statistical uncertainty); rows whose solution touches a bound, that
    converge slowly or not at all are handed to :func:`_peak`, so edge cases get exactly
    the scipy result. The acceptance tests are those of :func:`_peak`.
    """
    profiles = np.asarray(profiles, dtype=float)
    n_rows = profiles.shape[0]
    out = np.full((n_rows, 5), np.nan)
    lo = max(0, int(np.floor(center)) - halfwin)
    hi = min(profiles.shape[1], int(np.floor(center)) + halfwin + 1)
    x = np.arange(lo, hi, dtype=float)
    V = profiles[:, lo:hi]
    if x.size < 9 or n_rows == 0:
        return out
    ok = np.ones(n_rows, dtype=bool)
    if saturation is not None:
        ok &= ~(V.max(axis=1) >= saturation)
    origin = x.mean()
    xo = x - origin
    edge = [0, 1, -2, -1]
    slope, intercept = np.polyfit(xo[edge], V[:, edge].T, 1)       # one fit per row (columns)
    signal = V - (slope[:, None] * xo + intercept[:, None])
    amplitude = signal.max(axis=1)
    d = np.diff(V, axis=1)
    noise = np.median(np.abs(d - np.median(d, axis=1)[:, None]), axis=1) / 0.67449 / np.sqrt(2)
    noise = np.maximum(noise, np.finfo(float).eps * np.maximum(1., np.abs(V).max(axis=1)))
    ok &= ~(amplitude <= snr_min * noise)
    peak = x[np.argmax(signal, axis=1)]
    ok &= (peak > x[1]) & (peak < x[-2])
    # Rows with another peak in the window (prominence above snr_min·noise, >= 3 px from the
    # start) can have several minima; scipy's trust region and a Levenberg-Marquardt path from
    # the same start may end in different ones, so those rows go to _peak directly.
    crowded = np.zeros(n_rows, dtype=bool)
    for row in np.flatnonzero(ok):
        found, _ = find_peaks(signal[row], prominence=snr_min * noise[row])
        crowded[row] = np.any(np.abs(x[found] - peak[row]) >= 3)
    scipy_rows = np.flatnonzero(ok & crowded)
    rows = np.flatnonzero(ok & ~crowded)
    if not rows.size:
        return _scipy_rows(out, profiles, scipy_rows, center, halfwin, snr_min, saturation)

    lower = np.array([0., x[1], 0.35, -np.inf, -np.inf])
    upper = np.array([np.inf, x[-2], halfwin / 2, np.inf, np.inf])
    P = np.column_stack([amplitude[rows], peak[rows], np.full(rows.size, max(1., halfwin / 4)),
                         intercept[rows], slope[rows]])
    Vr = V[rows]

    def evaluate(P, data):
        a, mu, sigma = P[:, 0:1], P[:, 1:2], P[:, 2:3]
        z = (x - mu) / sigma
        e = np.exp(-0.5 * z * z)
        r = a * e + P[:, 3:4] + P[:, 4:5] * xo - data               # _gaussian_residual
        J = np.empty(r.shape + (5,))                                 # _gaussian_jacobian
        J[..., 0] = e
        J[..., 1] = a * e * z / sigma
        J[..., 2] = a * e * z * z / sigma
        J[..., 3] = 1.0
        J[..., 4] = xo
        return r, J

    r, J = evaluate(P, Vr)
    cost = 0.5 * np.einsum('nm,nm->n', r, r)
    lam = np.full(rows.size, 1e-3)
    active = np.ones(rows.size, dtype=bool)
    converged = np.zeros(rows.size, dtype=bool)
    iterations = np.zeros(rows.size, dtype=int)
    eye = np.eye(5)
    for _ in range(_LM_MAX_ITER):
        idx = np.flatnonzero(active)
        if not idx.size:
            break
        Ja, ra = J[idx], r[idx]
        H = np.einsum('nmk,nml->nkl', Ja, Ja)
        g = np.einsum('nmk,nm->nk', Ja, ra)
        diag = np.einsum('nkk->nk', H)
        diag = np.maximum(diag, 1e-12 * diag.max(axis=1, keepdims=True) + 1e-300)   # never singular
        damped = H + lam[idx, None, None] * diag[:, :, None] * eye
        try:
            step = np.linalg.solve(damped, -g[..., None])[..., 0]
        except np.linalg.LinAlgError:                                 # left to scipy
            break
        # Stay strictly inside the bounds like scipy's trust-region-reflective fit: shorten the
        # step to 95 % of the way to the nearest bound it would cross (clipping onto a bound
        # can lock a line at zero amplitude, where the centre and width lose their gradient).
        current = P[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            reach = np.where(step > 0, (upper - current) / step,
                             np.where(step < 0, (lower - current) / step, np.inf))
        factor = np.minimum(1.0, 0.95 * reach.min(axis=1))
        full = factor >= 1.0                    # the step was not shortened by a bound
        trial = current + factor[:, None] * step
        r_new, J_new = evaluate(trial, Vr[idx])
        cost_new = 0.5 * np.einsum('nm,nm->n', r_new, r_new)
        iterations[idx] += 1
        better = np.isfinite(cost_new) & (cost_new < cost[idx])
        take = idx[better]
        decrease = cost[take] - cost_new[better]
        relative = np.max(np.abs(trial - current) / (np.abs(current) + 1e-12), axis=1)
        moved = relative[better]
        # Only a FULL Gauss-Newton step (small damping, not shortened by a bound) that no longer
        # lowers the cost, or lowers it by < 1e-14 / moves < 1e-12 relative, means "at the
        # minimum": a step cut short near a bound is tiny for that reason alone.
        taken = trial - current
        predicted = -np.einsum('nk,nk->n', g, taken) - 0.5 * np.einsum('nk,nkl,nl->n', taken, H, taken)
        rejected = idx[~better]
        flat = (relative[~better] <= 1e-10) | (predicted[~better] <= 1e-13 * cost[rejected])
        converged[rejected[(lam[rejected] <= 1e-3) & full[~better] & flat]] = True
        gauss_newton = (lam[take] <= 1e-3) & full[better]
        P[take], r[take], J[take], cost[take] = trial[better], r_new[better], J_new[better], cost_new[better]
        lam[take] = np.maximum(lam[take] / 10., 1e-12)
        lam[idx[~better]] *= 10.
        done = gauss_newton & ((decrease <= 1e-14 * cost[take]) | (moved <= 1e-12))
        converged[take[done]] = True
        stuck = rejected[(lam[rejected] > 1e10) & full[~better]]     # no full step lowers the cost
        converged[stuck] = True
        active &= ~converged

    a, mu, sigma = P[:, 0], P[:, 1], P[:, 2]
    tolerance = 1e-6 * (1 + np.abs(np.where(np.isfinite(lower), lower, 0)))
    near_lower = np.isfinite(lower) & (P - lower <= tolerance)
    tolerance = 1e-6 * (1 + np.abs(np.where(np.isfinite(upper), upper, 0)))
    near_upper = np.isfinite(upper) & (upper - P <= tolerance)
    at_bound = np.any(near_lower | near_upper, axis=1)
    rms = np.sqrt(np.mean(r * r, axis=1))
    accept = converged & ~at_bound & (iterations <= _LM_SLOW) & (sigma >= _LM_MIN_SIGMA) \
        & (mu - 3 * sigma >= x[0]) & (mu + 3 * sigma <= x[-1]) \
        & ~(a / np.maximum(noise[rows], rms) < snr_min)
    area = _trapezoid(Vr - P[:, 3:4] - P[:, 4:5] * xo, x, axis=1)
    out[rows[accept]] = np.column_stack([mu, sigma * 2.354820045, a, area, rms])[accept]
    # Every row the batched fit does not accept cleanly gets the scipy fit itself: a rejection is
    # always scipy's own decision.
    scipy_rows = np.concatenate([scipy_rows, rows[~accept]])
    return _scipy_rows(out, profiles, scipy_rows, center, halfwin, snr_min, saturation)


def _scipy_rows(out, profiles, rows, center, halfwin, snr_min, saturation):
    """Fill ``out[rows]`` with :func:`_peak` (NaN where it rejects) and return ``out``."""
    for row in rows:
        fit = _peak(profiles[row], center, halfwin, snr_min, saturation)
        if fit is not None:
            out[row] = fit
    return out


def correlation_shift(reference, target, max_shift_px=5., min_score=0.8):
    """Return target displacement in pixels and normalized correlation score.

    Linear subpixel sampling evaluates a fixed interior interval, so candidate
    shifts use identical support. Flat or low-correlation profiles return NaN.
    The sign is positive when a feature moves toward larger detector indices.
    """
    ref, dst = np.asarray(reference, float), np.asarray(target, float)
    if ref.ndim != 1 or ref.shape != dst.shape or not np.isfinite([ref, dst]).all():
        raise ValueError('Profiles must be equally sized finite 1-D arrays.')
    if max_shift_px <= 0 or not -1 <= min_score <= 1:
        raise ValueError('Invalid shift bound or minimum correlation.')
    edge = int(np.ceil(max_shift_px)) + 1
    if ref.size <= 2*edge+3:
        raise ValueError('Profile is too short for the requested shift bound.')
    x = np.arange(edge, ref.size-edge)
    a = ref[x] - ref[x].mean()
    def score(shift):
        b = np.interp(x+shift, np.arange(dst.size), dst)
        b -= b.mean()
        denom = np.linalg.norm(a)*np.linalg.norm(b)
        return float(a@b/denom) if denom > 0 else -1.
    grid = np.linspace(-max_shift_px, max_shift_px, 41)
    # All grid shifts at once (same maths as score(); the bounded refinement below uses score()).
    b = np.interp(x[None, :] + grid[:, None], np.arange(dst.size), dst)
    b -= b.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = np.where(denom > 0, (b @ a) / denom, -1.)
    best = int(np.argmax(scores))
    left, right = grid[max(0, best-1)], grid[min(grid.size-1, best+1)]
    fit = minimize_scalar(lambda s: -score(s), bounds=(left, right), method='bounded')
    shift, quality = float(fit.x), score(fit.x)
    boundary = abs(shift) >= max_shift_px-0.01
    return (shift if quality >= min_score and not boundary else np.nan), quality


def normalize_wavelength_config(config):
    """Return a Smile-quality wavelength config from either supported JSON layout.

    Thin alias of :func:`modules.contracts.wavelength.normalize_wavelength_config`,
    kept so existing imports keep working.
    """
    from modules.contracts.wavelength import normalize_wavelength_config as normalize
    return normalize(config)


def analyze_quality(before, after, references, *, m_nm_per_px, c_nm,
                    row_step=1, resolution_nm=5., snr_min=5.,
                    saturation=None, y_ref=None, independent_frame=False, progress=None):
    """Measure absolute wavelength error, FWHM, area, and correlation residual.

    ``m_nm_per_px`` and ``c_nm`` map *corrected-grid* columns to wavelength.
    References must include expected corrected columns; blended neighbors closer
    than ``resolution_nm`` are automatically excluded. ``independent_frame`` is
    a provenance declaration, not a test that detects frame reuse.

    Before fits search around each row's local maximum inside the reference
    window. Excessive curvature outside that window requires a larger window.
    FWHM assumes a Gaussian instrument line shape; fit residuals are also saved.

    ``progress(fraction, message)`` is called after each reference line of each frame.
    """
    before, after = _frame(before), _frame(after)
    if before.shape != after.shape:
        raise ValueError('Use equal, uncropped before/after frame grids.')
    refs = list(references)
    if not refs or row_step < 1 or int(row_step) != row_step or resolution_nm <= 0:
        raise ValueError('Provide references, positive resolution and integer row_step.')
    if not np.isfinite([m_nm_per_px, c_nm]).all() or m_nm_per_px == 0:
        raise ValueError('Invalid wavelength calibration coefficients.')
    for ref in refs:
        if not np.isfinite([ref.wavelength_nm, ref.column_px]).all() or not 0 <= ref.column_px < before.shape[1] or ref.half_window_px < 4:
            raise ValueError('Invalid spectral reference or fitting window.')
    from .analysis import _default_reference_row
    y_ref = _default_reference_row(before.shape[0]) if y_ref is None else int(y_ref)
    if not 0 <= y_ref < before.shape[0]:
        raise ValueError('Reference row is outside the frame.')
    rows = np.arange(0, before.shape[0], int(row_step))
    wavelengths = np.array([r.wavelength_nm for r in refs], dtype=float)
    distance = np.abs(wavelengths[:, None]-wavelengths[None, :])
    np.fill_diagonal(distance, np.inf)
    excluded = np.array([r.blended for r in refs]) | (distance.min(axis=1) < resolution_nm)
    measurements = []
    started = time.perf_counter()
    steps = 2 * len(refs)
    logger.info("Spectral quality: %d lines x %d rows (row step %d), frame %dx%d",
                len(refs), rows.size, int(row_step), before.shape[1], before.shape[0])
    _report(progress, 0.0, "Quality: starting")
    keys = ['center_px', 'wavelength_error_nm', 'fwhm_nm', 'peak_height',
            'area_counts_px', 'fit_rms_counts', 'correlation_shift_px', 'correlation_score']
    for stage, frame in enumerate([before, after]):
        stage_name = ("before", "after")[stage]
        data = {k: np.full((rows.size, len(refs)), np.nan) for k in keys}
        for j, ref in enumerate(refs):
            done = stage * len(refs) + j + 1
            if excluded[j]:
                logger.debug("Quality %s: %g nm skipped (blended)", stage_name, ref.wavelength_nm)
                _report(progress, done / steps, "Quality %s: line %d/%d (%g nm, blended - skipped)"
                        % (stage_name, j + 1, len(refs), ref.wavelength_nm))
                continue
            line_started = time.perf_counter()
            lo = max(0, int(ref.column_px)-ref.half_window_px)
            hi = min(frame.shape[1], int(ref.column_px)+ref.half_window_px+1)
            reference_profile = frame[y_ref, lo:hi]
            # Gaussian fits of every row of this line at once (same window), see _peaks
            fits = _peaks(frame[rows], ref.column_px, ref.half_window_px, snr_min, saturation)
            mu, width = fits[:, 0], fits[:, 1]
            for k, value in zip(keys[:6], [mu, m_nm_per_px*mu+c_nm-ref.wavelength_nm,
                                          abs(m_nm_per_px)*width, fits[:, 2], fits[:, 3], fits[:, 4]]):
                data[k][:, j] = value
            bound = min(5., (hi-lo-5)/4)
            if bound > 0:
                for i, y in enumerate(rows):
                    shift, score = correlation_shift(reference_profile, frame[y, lo:hi], bound)
                    data['correlation_shift_px'][i, j] = shift
                    data['correlation_score'][i, j] = score
            fitted = int(np.isfinite(data['center_px'][:, j]).sum())
            logger.debug("Quality %s: %g nm, %d/%d rows fitted in %.2fs", stage_name,
                         ref.wavelength_nm, fitted, rows.size, time.perf_counter() - line_started)
            _report(progress, done / steps, "Quality %s: line %d/%d (%g nm), %d/%d rows fitted"
                    % (stage_name, j + 1, len(refs), ref.wavelength_nm, fitted, rows.size))
        measurements.append(data)
    pre, post = measurements
    paired = np.isfinite(pre['center_px']) & np.isfinite(post['center_px'])
    summary = {'paired_wavelength_before_nm': _stats(pre['wavelength_error_nm'][paired]),
               'paired_wavelength_after_nm': _stats(post['wavelength_error_nm'][paired]),
               'wavelength_before_nm': _stats(pre['wavelength_error_nm']),
               'wavelength_after_nm': _stats(post['wavelength_error_nm']),
               'correlation_after_px': _stats(post['correlation_shift_px']),
               'valid_fraction_before_by_line': np.mean(np.isfinite(pre['center_px']), axis=0).tolist(),
               'valid_fraction_after_by_line': np.mean(np.isfinite(post['center_px']), axis=0).tolist()}
    for key in ['fwhm_nm', 'peak_height', 'area_counts_px']:
        valid = np.isfinite(pre[key]) & np.isfinite(post[key]) & (pre[key] != 0)
        change = np.full(pre[key].shape, np.nan)
        change[valid] = 100*(post[key][valid]/pre[key][valid]-1)
        summary[key+'_change_percent'] = _stats(change)
    after_stats = summary['paired_wavelength_after_nm']
    logger.info("Spectral quality done in %.1fs: %d paired fits, RMSE after %.4f nm",
                time.perf_counter() - started, after_stats['count'], after_stats['rmse'])
    return QualityReport(rows, wavelengths, pre, post, summary,
                         dict(y_ref=y_ref, independent_frame=bool(independent_frame),
                              excluded_blends=excluded.tolist(), resolution_nm=resolution_nm,
                              m_nm_per_px=m_nm_per_px, c_nm=c_nm,
                              line_model='Gaussian + linear background',
                              uncertainty_scope='not included; use estimate_shift_uncertainty'))


def estimate_shift_uncertainty(analysis, *, rows, columns, repeats=100, seed=0, progress=None):
    """Bootstrap complete slit rows to estimate conditional shift uncertainty.

    Resampling rows preserves within-row correlation across emission lines.
    Results exclude lamp identification errors, wavelength coefficient uncertainty
    and systematic model bias. Outside measured spatial/spectral bounds, the
    fitted surface is clamped; the support mask marks those values unsupported.
    At least three distinct rows and two distinct columns are required.
    ``progress(fraction, message)`` is called about every 5 % of the repeats.
    """
    from .surface import fit_surface_2d
    yy, xx = np.broadcast_arrays(np.asarray(rows, float), np.asarray(columns, float))
    if repeats < 20 or not (np.isfinite(yy).all() and np.isfinite(xx).all()):
        raise ValueError('Use >=20 bootstrap repeats and finite query coordinates.')
    y, x, v = analysis.meas_y, analysis.meas_x, analysis.meas_v
    unique_rows = np.unique(y)
    if unique_rows.size < 3 or np.unique(x).size < 2:
        raise ValueError('Insufficient measurement coverage for bootstrap.')
    groups = [np.flatnonzero(y == row) for row in unique_rows]
    rng = np.random.default_rng(seed)
    samples = []
    started = time.perf_counter()
    logger.info("Shift uncertainty: %d bootstrap repeats over %d rows", repeats, len(groups))
    every = max(1, repeats // 20)
    for k_repeat in range(repeats):
        if k_repeat % every == 0:
            _report(progress, k_repeat / repeats, "Bootstrap %d/%d" % (k_repeat, repeats))
        index = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        if np.unique(y[index]).size <= analysis.surface.deg_y:
            continue
        model, _ = fit_surface_2d(y[index], x[index], v[index],
                                  analysis.surface.deg_y, analysis.surface.deg_x)
        samples.append(model(yy, xx)-model(analysis.y_ref, xx))
    if len(samples) < 20:
        raise ValueError('Too few nondegenerate bootstrap fits.')
    samples = np.asarray(samples)
    _report(progress, 1.0, "Bootstrap %d/%d" % (repeats, repeats))
    logger.info("Shift uncertainty done in %.1fs: %d/%d successful fits",
                time.perf_counter() - started, len(samples), repeats)
    return dict(std_px=np.std(samples, axis=0, ddof=1),
                lower95_px=np.percentile(samples, 2.5, axis=0),
                upper95_px=np.percentile(samples, 97.5, axis=0),
                supported=(yy >= y.min()) & (yy <= y.max()) & (xx >= x.min()) & (xx <= x.max()),
                successful_repeats=len(samples), seed=seed,
                scope='Conditional row-bootstrap; excludes wavelength and systematic uncertainty.')

