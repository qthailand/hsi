"""Flat-field profile of a pushbroom scan: the relative response of every slit row, per band.

A scan of a target that is uniform along the slit and spans it (a white strip; its reflectance
need not be known) shows how the lamp profile, the optics' vignetting and the sensor change
along the slit. Averaged along the scan and divided by its median over the slit, that is the
flat-field profile ``F(row, band)`` (about 1). ``counts / F`` is then the same for the same
material at every slit row, so a small reference target (a Spectralon disc covering a few rows)
can give the reflectance scale for the whole slit.

Data conventions:
    Cubes are ``(lines, samples, bands)``: slit rows, scan positions, wavelength bands, in
    dark-subtracted counts (any real dtype, array or memmap). ROI masks are boolean
    ``(lines, samples)``. A profile is float32 ``(lines, bands)``, finite and positive, with
    median 1 over the measured rows of each band.

Threading:
    Plain synchronous NumPy with no shared state: any thread. ``progress_callback`` gets ``0``
    and then the number of bands done.

Example:
    >>> import numpy as np
    >>> lines, samples, bands = 6, 5, 3
    >>> falloff = np.linspace(1.0, 0.5, lines)                     # light falls off along the slit
    >>> cube = (1000 * falloff)[:, None, None] * np.ones((lines, samples, bands))
    >>> flat = compute_flatfield(cube, np.ones((lines, samples), bool))
    >>> flat['profile'].shape, bool(np.allclose(apply_flatfield(cube, flat['profile']), 750.0))
    ((6, 3), True)
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = ["compute_flatfield", "apply_flatfield", "check_profile", "DEFAULT_MIN_COVERAGE",
           "DEFAULT_MAX_SATURATED", "DEFAULT_STEP_THRESHOLD"]

#: Smallest share of the slit rows the target must cover (the rest is interpolated or held).
DEFAULT_MIN_COVERAGE = 0.5
#: Largest share of saturated ROI pixels: a clipped target is flatter than the light on it.
DEFAULT_MAX_SATURATED = 0.01
#: Smallest lasting jump of the response along the slit that is reported (two pieces of the
#: target that do not lie alike).
DEFAULT_STEP_THRESHOLD = 0.01


def _neighbour_prediction(positions, values, half):
    """What each sample should be, judged from its neighbours (the sample itself is never used).

    The median of straight lines through pairs of neighbours, read at the sample's position:

    * the ``k``-th neighbour before and the ``k``-th after it, ``k = 1..half`` (interpolation);
    * within ``half`` samples of an end, where few such pairs exist, also pairs of neighbours on
      the longer side (extrapolation), so the first and the last sample are judged too and one
      bad sample at the end cannot outvote the good ones.

    ``positions`` are the samples' slit rows: the lines are exact on a straight slope also where
    rows are missing between samples (a gap in the ROI). Unaffected by an outlier run shorter
    than about ``half / 2`` (a quarter of ``half`` at an end). Fewer than three samples predict
    themselves.
    """
    x = np.asarray(positions, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    half = max(1, int(half))
    out = v.copy()
    if n < 3:
        return out
    for i in range(n):
        before, after = i, n - 1 - i
        pairs = min(half, before, after)
        predictions = []
        if pairs:
            k = np.arange(1, pairs + 1)
            xl, vl, xr, vr = x[i - k], v[i - k], x[i + k], v[i + k]
            predictions.append(vl + (vr - vl) * (x[i] - xl) / (xr - xl))
        if pairs < half:
            side = 1 if after >= before else -1
            reach = min(half, max(before, after))
            if reach >= 2:
                apart = reach // 2
                a = np.arange(1, reach - apart + 1)
                x1, v1 = x[i + side * a], v[i + side * a]
                x2, v2 = x[i + side * (a + apart)], v[i + side * (a + apart)]
                predictions.append(v1 + (v2 - v1) * (x[i] - x1) / (x2 - x1))
        if predictions:
            out[i] = np.median(np.concatenate(predictions))
    return out


def _line_at(x, v, at):
    """A robust straight line through the samples ``(x, v)``, read at ``at``: the median of the
    lines through pairs of samples half the run apart."""
    apart = len(x) // 2
    a = np.arange(len(x) - apart)
    x1, v1, x2, v2 = x[a], v[a], x[a + apart], v[a + apart]
    return float(np.median(v1 + (v2 - v1) * (at - x1) / (x2 - x1)))


def _lasting_steps(positions, values, half, threshold):
    """Places where the response jumps and stays: ``[(row, step), ...]`` in row order.

    Between every two consecutive samples, the line through the ``half`` samples before and the
    line through the ``half`` samples after are read at the middle; ``step`` is their ratio
    minus 1 (``+0.02``: the rows from ``row`` on are 2 % brighter than the rows before
    continue to). A smooth falloff, however steep or curved, gives none: both lines follow it.
    Of the jumps over ``threshold`` within ``half`` samples of each other the largest is kept.
    """
    x = np.asarray(positions, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    half = max(4, int(half))
    need = max(4, half // 2)
    steps = np.zeros(max(n - 1, 0))
    for i in range(n - 1):
        # the runs end where rows are missing: a line across a gap in the ROI would cut the
        # falloff's curve short and look like a jump
        first = int(np.searchsorted(x, x[i] - half + 1))
        last = int(np.searchsorted(x, x[i + 1] + half - 1, side="right"))
        left, right = slice(first, i + 1), slice(i + 1, last)
        if i + 1 - first < need or last - i - 1 < need:
            continue
        at = 0.5 * (x[i] + x[i + 1])
        before = _line_at(x[left], v[left], at)
        if before > 0:
            steps[i] = _line_at(x[right], v[right], at) / before - 1.0
    found = []
    for i in np.argsort(-np.abs(steps)):
        if not abs(steps[i]) > threshold:
            break
        if all(abs(int(i) - j) > half for j, _ in found):
            found.append((int(i), float(steps[i])))
    # the robust lines give the same jump a few samples around it: it is where two consecutive
    # samples differ most from the slope around them
    located = []
    for i, step in found:
        lo, hi = max(0, i - half), min(n - 1, i + half)
        slope = np.median(np.diff(v[lo:hi + 1]) / np.diff(x[lo:hi + 1]))
        near = np.arange(max(0, i - half // 2), min(n - 1, i + half // 2 + 1))
        jump = (v[near + 1] - v[near]) - slope * (x[near + 1] - x[near])
        located.append((int(x[near[int(np.argmax(np.abs(jump)))] + 1]), step))
    return sorted(located)


def compute_flatfield(cube, mask, *, seam_threshold=None, seam_window=31, min_coverage=DEFAULT_MIN_COVERAGE,
                      saturation=None, max_saturated=DEFAULT_MAX_SATURATED,
                      step_threshold=DEFAULT_STEP_THRESHOLD, progress_callback=None):
    """Fit the flat-field profile on a scan of a target that is uniform along the slit.

    Each slit row's response is the mean of its ROI pixels (along the scan). Rows the ROI does not
    cover - a joint between two pieces of the target left out of the ROI, or found by
    ``seam_threshold`` - are linearly interpolated from the rows around them; rows beyond the
    first / last measured row take that row's value (reported as ``extrapolated_rows``: the
    profile is not measured there).

    Args:
        cube: ``(lines, samples, bands)`` dark-subtracted counts.
        mask: Boolean ROI ``(lines, samples)`` on the uniform target.
        seam_threshold: When given (e.g. ``0.03``), a measured row whose broadband response differs
            from what its neighbours predict by more than this fraction is treated as a joint
            (dirt, gap, shadow) and interpolated; a dark first or last row (the ROI reaching the
            target's edge) is dropped the same way and then held. ``None``: off. A lasting step between two
            pieces (another height or tilt) is not a narrow outlier and is not found.
        seam_window: Rows around a row used for that prediction (``seam_window // 2`` on each
            side); joints up to about a quarter of it are found. Keep it short where the
            response curves strongly along the slit.
        min_coverage: Smallest share of the slit rows that must be measured.
        saturation: Counts at or above this are saturated; their share of the ROI is reported.
        max_saturated: Largest share of saturated ROI pixels accepted (with ``saturation``).
        step_threshold: A jump of the broadband response that stays (two pieces of the target at
            another height or tilt, or of another material) larger than this fraction is reported
            in ``steps``. It is not removed: the profile cannot tell it from the light, and
            carries it into every scan. ``None``: not looked for.
        progress_callback: Called with ``0`` and then the number of bands done.

    Returns:
        dict: ``profile`` float32 ``(lines, bands)``; ``measured_rows`` bool ``(lines,)``; ``span``
        (first, last measured row); ``coverage`` (share of rows measured); ``seam_rows`` (list of
        rows dropped by ``seam_threshold``); ``steps`` (list of ``(row, step)``: the response
        jumps by the fraction ``step`` at slit row ``row`` and stays, see ``step_threshold``);
        ``extrapolated_rows`` (rows outside ``span``);
        ``invalid_bands`` (bands without any positive response: profile 1 there);
        ``row_profile`` float64 ``(lines,)`` (broadband response, median 1, for a plot);
        ``saturated_fraction`` (``None`` without ``saturation``).

    Raises:
        ValueError: Wrong shapes, an empty ROI, a saturated target, so many rows looking like
            joints that the target cannot be uniform, or fewer measured rows than ``min_coverage``.
    """
    progress_callback = progress_callback or (lambda value: None)
    if getattr(cube, "ndim", None) != 3:
        raise ValueError(f"cube must be (lines, samples, bands), got shape {getattr(cube, 'shape', None)}")
    lines, samples, bands = cube.shape
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (lines, samples):
        raise ValueError(f"ROI mask {mask.shape} does not match the cube ({lines}, {samples})")
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        raise ValueError("Selected ROI does not cover any pixels.")
    cols = np.where(mask.any(axis=0))[0]
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    box_mask = mask[r0:r1, c0:c1]

    means = np.full((lines, bands), np.nan, dtype=np.float64)
    saturated = total = 0
    progress_callback(0)
    for b in range(bands):
        box = np.asarray(cube[r0:r1, c0:c1, b], dtype=np.float64)
        valid = box_mask & np.isfinite(box)
        count = valid.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            means[r0:r1, b] = np.where(valid, box, 0.0).sum(axis=1) / count
        if saturation is not None:
            saturated += int(np.count_nonzero(valid & (box >= saturation)))
            total += int(valid.sum())
        progress_callback(b + 1)
    means[~(means > 0)] = np.nan                     # no pixel, or no light: not a response
    saturated_fraction = (saturated / total if total else 0.0) if saturation is not None else None
    if saturated_fraction is not None and saturated_fraction > float(max_saturated):
        raise ValueError(f"{saturated_fraction:.1%} of the flat-field ROI is saturated (limit "
                         f"{float(max_saturated):.1%}): scan the target again with a shorter exposure.")

    def broadband(values):
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows / bands
            return np.nanmedian(values / np.nanmedian(values, axis=0), axis=1)

    seam_rows = []
    if seam_threshold is not None:
        relative = broadband(means)
        measured = np.where(np.isfinite(relative))[0]
        half = max(1, int(seam_window) // 2)
        # drop the worst row, judge again from the rows left: a joint's neighbours are then
        # compared with clean rows, not with the joint
        limit = max(1, measured.size // 4)
        while measured.size >= 3:
            predicted = _neighbour_prediction(measured, relative[measured], half)
            deviation = np.abs(relative[measured] / predicted - 1.0)
            worst = int(np.argmax(deviation))
            if deviation[worst] <= float(seam_threshold):
                break
            if len(seam_rows) >= limit:
                raise ValueError(f"More than {limit} slit rows differ from their neighbours by over "
                                 f"{float(seam_threshold):.1%}: the target is not uniform along the slit "
                                 "(saturated, textured, or the ROI leaves it), or the threshold is too tight.")
            seam_rows.append(int(measured[worst]))
            measured = np.delete(measured, worst)
        seam_rows.sort()
        means[seam_rows] = np.nan

    steps = []
    if step_threshold is not None:
        relative = broadband(means)
        measured = np.where(np.isfinite(relative))[0]
        steps = _lasting_steps(measured, relative[measured], max(1, int(seam_window) // 2), float(step_threshold))

    measured_rows = np.isfinite(means).any(axis=1)
    measured = np.where(measured_rows)[0]
    coverage = measured.size / float(lines)
    if measured.size == 0 or coverage < float(min_coverage):
        raise ValueError(f"The ROI measures {measured.size} of {lines} slit rows ({coverage:.0%}); a flat-field "
                         f"target must span the slit (at least {float(min_coverage):.0%}).")

    profile = np.ones((lines, bands), dtype=np.float64)
    all_rows = np.arange(lines)
    invalid_bands = 0
    for b in range(bands):
        good = np.where(np.isfinite(means[:, b]))[0]
        if good.size == 0:
            invalid_bands += 1
            continue
        filled = np.interp(all_rows, good, means[good, b])
        profile[:, b] = filled / np.median(filled[good])

    row_profile = np.median(profile, axis=1)
    first, last = int(measured[0]), int(measured[-1])
    return {"profile": profile.astype(np.float32), "measured_rows": measured_rows, "span": (first, last),
            "coverage": coverage, "seam_rows": seam_rows, "steps": steps, "extrapolated_rows": int(first + lines - 1 - last),
            "invalid_bands": invalid_bands, "row_profile": row_profile,
            "saturated_fraction": saturated_fraction}


def check_profile(profile, lines=None, bands=None):
    """``profile`` as a float32 ``(lines, bands)`` array, refused unless finite, positive and fitting.

    Raises:
        ValueError: Not 2-D, not finite and positive everywhere, or not ``(lines, bands)``.
    """
    profile = np.asarray(profile, dtype=np.float32)
    if profile.ndim != 2:
        raise ValueError(f"a flat-field profile is (lines, bands), got shape {profile.shape}")
    if not np.all(np.isfinite(profile)) or np.any(profile <= 0):
        raise ValueError("a flat-field profile must be finite and positive everywhere")
    if lines is not None and profile.shape[0] != int(lines):
        raise ValueError(f"Flat-field profile for {profile.shape[0]} slit rows does not fit {int(lines)} rows: "
                         "make one at the same camera setting.")
    if bands is not None and profile.shape[1] != int(bands):
        raise ValueError(f"Flat-field profile for {profile.shape[1]} bands does not fit {int(bands)} bands: "
                         "make one at the same camera setting.")
    return profile


def apply_flatfield(cube, profile, rows_per_chunk=64, progress_callback=None):
    """``cube / profile[:, None, :]`` as a new float32 cube (counts, evened out along the slit).

    Args:
        cube: ``(lines, samples, bands)`` dark-subtracted counts.
        profile: Flat-field profile ``(lines, bands)`` (:func:`compute_flatfield`).
        rows_per_chunk: Slit rows divided per step (no float64 copy of the whole cube).
        progress_callback: Optional ``callable(int)``, given the slit rows done after each step.

    Raises:
        ValueError: The profile does not have the cube's slit rows and bands.
    """
    if getattr(cube, "ndim", None) != 3:
        raise ValueError(f"cube must be (lines, samples, bands), got shape {getattr(cube, 'shape', None)}")
    profile = check_profile(profile, cube.shape[0], cube.shape[2])
    out = np.empty(cube.shape, dtype=np.float32)
    step = max(1, int(rows_per_chunk))
    for y in range(0, cube.shape[0], step):
        part = np.asarray(cube[y:y + step], dtype=np.float32)
        np.divide(part, profile[y:y + step, None, :], out=out[y:y + step])
        if progress_callback is not None:
            progress_callback(min(y + step, cube.shape[0]))
    return out
