# smile/analysis.py
"""Full multi-line smile analysis pipeline."""
from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import (
    Line,
    _traces_rms,
    auto_sat_level,
    detect_lines,
    estimate_noise,
    trace_line,
)
from .surface import Surface2D, cross_validate_lines, fit_surface_2d
from .calibration import correct_smile, crop_margin
from .debug_log import logged_analysis

__all__ = ["SmileAnalysis", "analyze_smile", "review_analysis", "verify_on_frame", "anchor_surface"]

logger = logging.getLogger(__name__)


def anchor_surface(surface: Surface2D, y_ref: float) -> Surface2D:
    """``surface(y, x) - surface(y_ref, x)`` as a :class:`Surface2D`, exactly.

    Only the ``y**0`` coefficients change (``C'[0, j] = C[0, j] - sum_i C[i, j] * Yr**i`` with
    ``Yr`` the normalised, clamped ``y_ref``); the fitted ranges and the clamping are kept, so the
    shift is unchanged inside the measured lines and still held at the edge value outside them.
    (Re-fitting the difference on a coarse grid over the whole frame, as before, bent the surface
    inside the measured span by ~0.1 px RMS / 0.5 px max when the lines did not span the width.)

    Args:
        surface: Fitted displacement surface.
        y_ref: Reference row (px); the result is exactly 0 there for every column.

    Returns:
        A new :class:`Surface2D` with the same degrees, ranges and clamp.
    """
    y0, y1 = surface.y_rng
    yr = 2.0 * (float(y_ref) - y0) / (y1 - y0) - 1.0 if y1 != y0 else 0.0
    if surface.clamp:
        yr = float(np.clip(yr, -1.0, 1.0))
    coef = np.asarray(surface.coef, dtype=float).reshape(surface.deg_y + 1, surface.deg_x + 1).copy()
    coef[0] -= sum(coef[i] * yr ** i for i in range(surface.deg_y + 1))
    return Surface2D(coef.ravel(), surface.deg_y, surface.deg_x, surface.y_rng, surface.x_rng, surface.clamp)


def _default_reference_row(height: int) -> int:
    """Return the zero-based reference row (shared rule from :mod:`modules.contracts`)."""
    from modules.contracts.geometry import default_reference_row
    return default_reference_row(height)


@dataclass
class SmileAnalysis:
    """Complete results of a smile correction analysis.

    Attributes:
        frame_orig: Original calibration image.
        frame_corr: Corrected image (zero-valued borders).
        all_lines: Every detected emission line.
        used_lines: Lines used in the surface fit.
        traces_orig: ``{column -> trace_array}`` for original image.
        traces_corr: ``{column -> trace_array}`` for corrected image.
        refs: ``{column -> sub-pixel reference}`` for each line.
        surface: Fitted 2-D displacement surface.
        fit_info: Surface fitting diagnostics.
        meas_y: Row coordinates of measurement points.
        meas_x: Sub-pixel reference column coordinates (adjusted).
        meas_v: Measured shift values.
        meas_id: Index into ``used_lines`` for each point.
        residuals: ``meas_v - surface(meas_y, meas_x)``.
        margin: ``(left, right)`` border pixels to crop.
        rms_before: Straightness RMS before correction.
        rms_after: Straightness RMS after correction (in-sample).
        cv_rms: Leave-one-line-out cross-validated RMS.
        interp: Interpolation method used.
        sat_level: Effective saturation threshold.
        noise: Estimated per-pixel noise level.
        timings: Seconds spent per stage (``detect``, ``trace``, ``fit``, ``cross_validate``,
            ``remap``, ``retrace``, ``total``), to see where a slow computer spends its time.
    """

    frame_orig: np.ndarray
    frame_corr: np.ndarray
    all_lines: List[Line]
    used_lines: List[Line]
    traces_orig: Dict[float, np.ndarray]
    traces_corr: Dict[float, np.ndarray]
    refs: Dict[float, float]
    surface: Surface2D
    fit_info: dict
    meas_y: np.ndarray
    meas_x: np.ndarray
    meas_v: np.ndarray
    meas_id: np.ndarray
    residuals: np.ndarray
    margin: Tuple[int, int]
    rms_before: float
    rms_after: float
    cv_rms: float
    interp: str
    sat_level: float
    noise: float
    y_ref: float
    timings: Dict[str, float] = field(default_factory=dict)


@logged_analysis("smile")
def analyze_smile(
    frame: np.ndarray,
    sat_level: Optional[float] = None,
    deg_y: int = 2,
    deg_x: int = 1,
    interp: str = "cubic",
    snr_min: float = 5.0,
    n_ref_iter: int = 2,
    y_ref: Optional[float] = None,
    verbose: bool = False,
) -> SmileAnalysis:
    """Run the full multi-line smile analysis pipeline.

    Args:
        frame: Calibration image ``(H, W)``.
        sat_level: Saturation threshold.  Inferred if ``None`` or <= 0.
        deg_y: Polynomial degree in the spatial (row) direction.
        deg_x: Polynomial degree in the spectral (column) direction.
        interp: Interpolation method.
        snr_min: Minimum peak-to-noise ratio for tracing.
        n_ref_iter: Number of reference-offset iterations.
        verbose: Print per-line report to stdout.
        y_ref: Row used as zero-displacement reference. Defaults to the
            geometric image center, with the lower-index center row selected.

    Returns:
        A :class:`SmileAnalysis` with all diagnostics.

    Raises:
        ValueError: If no usable emission lines are found.
    """
    frame = np.asarray(frame, dtype=np.float32)
    n_rows = frame.shape[0]
    if sat_level is None or sat_level <= 0:
        sat_level = auto_sat_level(frame)
    noise = estimate_noise(frame)

    tick = time.perf_counter()
    timings: Dict[str, float] = {}

    def lap(name: str) -> None:
        nonlocal tick
        now = time.perf_counter()
        timings[name] = timings.get(name, 0.0) + (now - tick)
        tick = now

    # Step 1: Detect emission lines.
    lines = detect_lines(frame, sat_level)
    lap("detect")
    if not lines:
        raise ValueError(
            "No emission lines detected. Check image brightness."
        )

    # Step 2: Trace every line through all rows.
    traces_orig: Dict[float, np.ndarray] = {}
    for ln in lines:
        halfwin = max(3, int(round(1.2 * ln.fwhm)))
        traces_orig[ln.column] = trace_line(
            frame, ln.column, halfwin, sat_level, noise, snr_min
        )

    lap("trace")
    # Step 3: Collect measurement points.
    #   Reference = median of the trace (sub-pixel).
    ys, xs, vs, ids, used = [], [], [], [], []
    refs: Dict[float, float] = {}
    for ln in lines:
        if ln.saturated:
            continue
        centers = traces_orig[ln.column]
        rows = np.where(np.isfinite(centers))[0]
        if rows.size < 20 or rows.size / n_rows < 0.02:
            continue
        ref = float(np.median(centers[rows]))
        refs[ln.column] = ref
        ys.append(rows.astype(float))
        xs.append(np.full(rows.size, ref))
        vs.append(centers[rows] - ref)
        ids.append(np.full(rows.size, len(used)))
        used.append(ln)

    if not ys:
        raise ValueError(
            "All lines were filtered out "
            "(saturated, low coverage, or low SNR)."
        )

    meas_y = np.concatenate(ys)
    meas_x = np.concatenate(xs)
    meas_v = np.concatenate(vs)
    meas_id = np.concatenate(ids)

    # Preserve original x coordinates for cross-validation.
    meas_x_orig = meas_x.copy()

    # Validate deg_x against available lines.
    n_ux = np.unique(np.round(meas_x, 4)).size
    deg_x_eff = min(deg_x, n_ux - 1)
    if deg_x_eff < deg_x:
        warnings.warn(
            f"Only {n_ux} unique line positions; "
            f"reducing deg_x from {deg_x} to {deg_x_eff}."
        )

    # Step 4: Iterative reference adjustment + robust surface fit.
    for _ in range(max(0, n_ref_iter)):
        surf_tmp, _ = fit_surface_2d(
            meas_y, meas_x, meas_v, deg_y, deg_x_eff
        )
        res_tmp = meas_v - surf_tmp(meas_y, meas_x)
        for k, ln in enumerate(used):
            mk = meas_id == k
            off = float(np.median(res_tmp[mk]))
            meas_v[mk] -= off
            meas_x[mk] += off
            refs[ln.column] += off

    surface, fit_info = fit_surface_2d(
        meas_y, meas_x, meas_v, deg_y, deg_x_eff
    )
    residuals = meas_v - surface(meas_y, meas_x)
    lap("fit")
    cv_rms = cross_validate_lines(
        meas_y, meas_x_orig, meas_v, meas_id, deg_y, deg_x_eff
    )

    # Anchor the fitted displacement at the requested physical reference row.
    if y_ref is None:
        y_ref = float(_default_reference_row(n_rows))
    if not np.isfinite(y_ref) or not 0 <= y_ref < n_rows:
        raise ValueError(f"y_ref must be within image rows [0, {n_rows - 1}]")
    surface = anchor_surface(surface, y_ref)

    lap("cross_validate")
    # Step 5: Remap (zero-valued borders).
    frame_corr, margin, _, _ = correct_smile(frame, surface, interp)
    lap("remap")

    # Step 6: Re-trace on the corrected image.
    traces_corr: Dict[float, np.ndarray] = {}
    for ln in used:
        halfwin = max(3, int(round(1.2 * ln.fwhm)))
        traces_corr[ln.column] = trace_line(
            frame_corr, refs[ln.column], halfwin, sat_level, noise, snr_min
        )

    lap("retrace")
    timings["total"] = sum(timings.values())
    logger.info("Smile analysis %dx%d: %s", frame.shape[1], frame.shape[0],
                ", ".join("%s %.2f s" % kv for kv in timings.items()))
    # Step 7: Straightness metrics (reuse existing traces).
    rms_before = _traces_rms([traces_orig[ln.column] for ln in used])
    rms_after = _traces_rms([traces_corr[ln.column] for ln in used])

    if verbose:
        print(
            f"Lines used: {len(used)}/{len(lines)}  "
            f"deg={deg_y}x{deg_x_eff}  interp={interp}"
        )
        print(
            f"Straightness RMS: {rms_before:.4f} -> {rms_after:.4f} px "
            f"(in-sample)   LOO-CV: {cv_rms:.4f} px"
        )
        print(f"Border margin: left={margin[0]} right={margin[1]} px")

    return SmileAnalysis(
        frame_orig=frame,
        frame_corr=frame_corr,
        all_lines=lines,
        used_lines=used,
        traces_orig=traces_orig,
        traces_corr=traces_corr,
        refs=refs,
        surface=surface,
        fit_info=fit_info,
        meas_y=meas_y,
        meas_x=meas_x,
        meas_v=meas_v,
        meas_id=meas_id,
        residuals=residuals,
        margin=margin,
        rms_before=rms_before,
        rms_after=rms_after,
        cv_rms=cv_rms,
        interp=interp,
        sat_level=float(sat_level),
        noise=float(noise),
        y_ref=float(y_ref),
        timings=timings,
    )


def review_analysis(
    frame: np.ndarray,
    surface: Surface2D,
    *,
    interp: str = "cubic",
    y_ref: Optional[float] = None,
    sat_level: Optional[float] = None,
    cv_rms: Optional[float] = None,
    snr_min: float = 5.0,
    n_ref_iter: int = 2,
) -> Tuple[SmileAnalysis, bool]:
    """Rebuild the analysis of a stored model from its lamp frame, keeping the stored smile.

    The lamp lines are detected and traced as in :func:`analyze_smile` (same degrees, reference
    row and interpolation as the stored model). When that fit reproduces the stored surface the
    analysis is returned as it is; otherwise the stored surface replaces the new fit: the frame is
    corrected with it, the lines are re-traced, and the residuals are the measured shifts minus the
    stored surface (per-line offset removed). So the plots and numbers always show *the stored
    model* on its own lamp frame, never a silent re-fit.

    Args:
        frame: The lamp frame stored with the model ``(H, W)``.
        surface: The stored smile surface (anchored at ``y_ref``).
        interp: The model's interpolation.
        y_ref: The model's reference row (``None``: the default centre row).
        sat_level: Sensor full scale (``None``: auto).
        cv_rms: Cross-validated RMS stored with the model, used when the stored surface differs
            from the re-fit (its own value cannot be recomputed without re-fitting).
        snr_min: Minimum peak-to-noise ratio for tracing.
        n_ref_iter: Reference-offset iterations of the line tracing.

    Returns:
        ``(analysis, identical)``: *identical* is ``True`` when the re-fit equals the stored
        surface within 0.001 px (the analysis is then exactly what *Analyze* gives).
    """
    import dataclasses

    frame = np.asarray(frame, dtype=np.float32)
    r = analyze_smile(frame, sat_level=sat_level, deg_y=surface.deg_y, deg_x=surface.deg_x, interp=interp,
                      snr_min=snr_min, n_ref_iter=n_ref_iter, y_ref=y_ref)
    h, w = frame.shape
    Y, X = np.meshgrid(np.linspace(0, h - 1, 64), np.linspace(0, w - 1, 64), indexing="ij")
    same_shape = (r.surface.deg_y, r.surface.deg_x) == (surface.deg_y, surface.deg_x)
    if same_shape and float(np.max(np.abs(r.surface(Y, X) - surface(Y, X)))) < 1e-3:
        return r, True
    residuals = r.meas_v - surface(r.meas_y, r.meas_x)
    for k in range(len(r.used_lines)):
        mk = r.meas_id == k
        if np.any(mk):
            residuals[mk] -= float(np.median(residuals[mk]))
    frame_corr, margin, _, _ = correct_smile(frame, surface, interp)
    traces_corr: Dict[float, np.ndarray] = {}
    for ln in r.used_lines:
        halfwin = max(3, int(round(1.2 * ln.fwhm)))
        traces_corr[ln.column] = trace_line(frame_corr, r.refs[ln.column], halfwin, r.sat_level, r.noise, snr_min)
    fit_info = dict(r.fit_info, rms=float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else float("nan"),
                    n_used=int(residuals.size), deg_y=surface.deg_y, deg_x=surface.deg_x,
                    coef=np.asarray(surface.coef, dtype=float).copy(), stored=True)
    out = dataclasses.replace(
        r, surface=surface, frame_corr=frame_corr, margin=margin, traces_corr=traces_corr, residuals=residuals,
        fit_info=fit_info, rms_after=_traces_rms([traces_corr[ln.column] for ln in r.used_lines]),
        cv_rms=float(cv_rms) if cv_rms is not None else float("nan"))
    return out, False


def verify_on_frame(
    analysis: SmileAnalysis, other_frame: np.ndarray
) -> float:
    """Measure straightness RMS after correction on a *different* frame.

    Args:
        analysis: Completed :class:`SmileAnalysis`.
        other_frame: Second calibration image with the same shape.

    Returns:
        Straightness RMS in pixels after applying the correction.

    Raises:
        ValueError: If image shapes do not match.
    """
    other = np.asarray(other_frame, dtype=np.float32)
    if other.shape != analysis.frame_orig.shape:
        raise ValueError(
            f"Shape mismatch: {other.shape} vs {analysis.frame_orig.shape}"
        )
    corr, _, _, _ = correct_smile(other, analysis.surface, analysis.interp)
    noise = estimate_noise(other)
    traces = []
    for ln in analysis.used_lines:
        halfwin = max(3, int(round(1.2 * ln.fwhm)))
        traces.append(
            trace_line(
                corr, analysis.refs[ln.column], halfwin,
                analysis.sat_level, noise,
            )
        )
    return _traces_rms(traces)
