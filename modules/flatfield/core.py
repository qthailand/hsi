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

__all__ = ["compute_flatfield", "combine_flatfields", "plan_next_scan", "apply_flatfield", "evening_factors", "check_profile",
           "DEFAULT_MIN_COVERAGE", "DEFAULT_MAX_SATURATED", "DEFAULT_STEP_THRESHOLD", "DEFAULT_COMBINE_PAD",
           "DEFAULT_STRAY_SIGMA", "DEFAULT_LAMP_RESIDUAL"]

#: Smallest share of the slit rows the target must cover (the rest is interpolated or held).
DEFAULT_MIN_COVERAGE = 0.5
#: Largest share of saturated ROI pixels: a clipped target is flatter than the light on it.
DEFAULT_MAX_SATURATED = 0.01
#: Smallest lasting jump of the response along the slit that is reported (two pieces of the
#: target that do not lie alike).
DEFAULT_STEP_THRESHOLD = 0.01
#: Rows beside a joint that :func:`combine_flatfields` also leaves out of that scan.
DEFAULT_COMBINE_PAD = 3
#: Robust sigmas a scan's row may stray from the others before :func:`combine_flatfields` drops it.
DEFAULT_STRAY_SIGMA = 4.0
#: A scan whose residual after combining is larger than this (fraction) saw other light.
DEFAULT_LAMP_RESIDUAL = 0.01
#: Smallest robust sigma of the stray test (fraction): noiseless scans would make every tiny
#: difference a stray.
_STRAY_FLOOR = 0.0025


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
            response curves strongly along the slit. It is capped at a quarter of the measured
            rows (at least 3): on a short ROI a window reaching across the slit would read the
            falloff towards its ends as joints. The window used is returned as ``seam_window``.
        min_coverage: Smallest share of the slit rows that must be measured.
        saturation: Counts at or above this are saturated; their share of the ROI is reported.
        max_saturated: Largest share of saturated ROI pixels accepted (with ``saturation``).
        step_threshold: A jump of the broadband response that stays (two pieces of the target at
            another height or tilt, or of another material) larger than this fraction is reported
            in ``steps``. It is not removed: the profile cannot tell it from the light, and
            carries it into every scan. ``None``: not looked for.
        progress_callback: Called with ``0`` and then the number of bands done.

    Returns:
        dict: ``profile`` float32 ``(lines, bands)``; ``measured`` float32 ``(lines, bands)``: the
        same before the gaps are filled (NaN on joint rows and rows the ROI leaves out; what
        :func:`combine_flatfields` works on); ``measured_rows`` bool ``(lines,)``; ``span``
        (first, last measured row); ``coverage`` (share of rows measured); ``seam_rows`` (list of
        rows dropped by ``seam_threshold``); ``steps`` (list of ``(row, step)``: the response
        jumps by the fraction ``step`` at slit row ``row`` and stays, see ``step_threshold``);
        ``extrapolated_rows`` (rows outside ``span``);
        ``invalid_bands`` (bands without any positive response: profile 1 there);
        ``row_profile`` float64 ``(lines,)`` (broadband response, median 1, for a plot);
        ``saturated_fraction`` (``None`` without ``saturation``); ``seam_window`` (the window used,
        after the cap).

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

    window = effective_seam_window(seam_window, int(np.isfinite(broadband(means)).sum()))
    half = max(1, window // 2)
    seam_rows = []
    if seam_threshold is not None:
        relative = broadband(means)
        measured = np.where(np.isfinite(relative))[0]
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
        steps = _lasting_steps(measured, relative[measured], half, float(step_threshold))

    measured_rows = np.isfinite(means).any(axis=1)
    measured = np.where(measured_rows)[0]
    coverage = measured.size / float(lines)
    if measured.size == 0 or coverage < float(min_coverage):
        raise ValueError(f"The ROI measures {measured.size} of {lines} slit rows ({coverage:.0%}); a flat-field "
                         f"target must span the slit (at least {float(min_coverage):.0%}).")

    profile = np.ones((lines, bands), dtype=np.float64)
    known = np.where(measured_rows[:, None], 1.0, np.nan) * np.ones((1, bands))
    all_rows = np.arange(lines)
    invalid_bands = 0
    for b in range(bands):
        good = np.where(np.isfinite(means[:, b]))[0]
        if good.size == 0:
            invalid_bands += 1                      # profile 1; measured 1 on the measured rows
            continue
        filled = np.interp(all_rows, good, means[good, b])
        scale = np.median(filled[good])
        profile[:, b] = filled / scale
        # NaN only on the rows not measured: a measured row where one dark band read <= 0 is not
        # a joint (combine would take it for one), it is that band's noise, filled like the profile
        known[:, b] = np.where(measured_rows, filled / scale, np.nan)

    row_profile = np.median(profile, axis=1)
    first, last = int(measured[0]), int(measured[-1])
    return {"profile": profile.astype(np.float32), "measured": known.astype(np.float32),
            "measured_rows": measured_rows, "span": (first, last),
            "coverage": coverage, "seam_rows": seam_rows, "steps": steps, "extrapolated_rows": int(first + lines - 1 - last),
            "invalid_bands": invalid_bands, "row_profile": row_profile,
            "saturated_fraction": saturated_fraction, "seam_window": window}


def _gaps(known):
    """Runs of unknown rows between the first and the last known row: ``[(first, last), ...]``.
    These are the joints of a scan; unknown rows at the ends are not joints."""
    rows = np.where(known)[0]
    if rows.size == 0:
        return []
    missing = np.where(~known[rows[0]:rows[-1] + 1])[0] + rows[0]
    gaps = []
    for r in missing:
        if gaps and r == gaps[-1][1] + 1:
            gaps[-1][1] = int(r)
        else:
            gaps.append([int(r), int(r)])
    return [tuple(g) for g in gaps]


def _solve_levels(logf, valid, pieces, anchor):
    """Least-squares levels of every piece of every scan, with the response eliminated row by row.

    ``logf`` ``(scans, lines, bands)`` log response (NaN where unknown); ``valid`` ``(scans, lines)``
    rows to use; ``pieces`` per scan, the piece index of every row; ``anchor`` ``(scan, piece)``
    held at level 0. Returns ``(t, offsets)``: ``t`` ``(lines, bands)`` the median of the levelled
    scans (NaN where no scan is valid), ``offsets`` per scan ``(pieces, bands)``.
    """
    scans, lines, bands = logf.shape
    base = np.cumsum([0] + [int(p.max()) + 1 for p in pieces])
    unknowns = int(base[-1])
    normal = np.zeros((unknowns, unknowns))
    right = np.zeros((unknowns, bands))
    for r in range(lines):
        s = np.where(valid[:, r])[0]
        n = s.size
        if n < 2:
            continue
        columns = base[s] + np.array([pieces[i][r] for i in s])
        centred = np.eye(n) - 1.0 / n                     # the row's own response drops out
        normal[np.ix_(columns, columns)] += centred
        right[columns] += centred @ logf[s, r]
    fixed = int(base[anchor[0]] + anchor[1])
    free = np.array([k for k in range(unknowns) if k != fixed and normal[k, k] > 0], dtype=int)
    levels = np.zeros((unknowns, bands))
    if free.size:
        levels[free] = np.linalg.lstsq(normal[np.ix_(free, free)], right[free], rcond=None)[0]
    offsets = [levels[base[i]:base[i + 1]] for i in range(scans)]
    levelled = np.stack([logf[i] - offsets[i][pieces[i]] for i in range(scans)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # rows no scan measured
        t = np.nanmedian(np.where(valid[:, :, None], levelled, np.nan), axis=0)
    return t, offsets


def _fill_rows(values, rows_known):
    """Each band of ``values`` ``(lines, bands)`` with the unknown rows interpolated (held at the ends)."""
    out = np.array(values, dtype=np.float64)
    good = np.where(rows_known)[0]
    every = np.arange(out.shape[0])
    for b in range(out.shape[1]):
        out[:, b] = np.interp(every, good, out[good, b])
    return out


def _fill_bands(logf, known):
    """``logf`` ``(scans, lines, bands)`` with each band's NaN on ``known`` rows interpolated along
    the rows from that band's finite rows (a band with none: 0, the median level); unknown rows stay NaN."""
    out = np.array(logf, dtype=np.float64)
    rows = np.arange(out.shape[1])
    for i in range(out.shape[0]):
        for b in range(out.shape[2]):
            good = np.isfinite(out[i, :, b])
            hole = known[i] & ~good
            if hole.any():
                out[i, hole, b] = np.interp(rows[hole], rows[good], out[i, good, b]) if good.any() else 0.0
    return out


def _broadband(values):
    """Per row, the median over bands of ``values`` ``(..., lines, bands)`` (NaN-aware): one weak
    band cannot pull a row's residual the way a mean over bands lets it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # rows with no finite band
        return np.nanmedian(values, axis=-1)


def combine_flatfields(measured, *, pad=DEFAULT_COMBINE_PAD, stray_sigma=DEFAULT_STRAY_SIGMA,
                       stray_window=61, stray_grow=3, rounds=3, leave_one_out=True,
                       step_threshold=DEFAULT_STEP_THRESHOLD):
    """One flat-field profile from several scans of a target whose joints lie at other slit rows.

    Every scan is the camera's response times the target's: the target's pieces each have their own
    level (another height, tilt or material) and carry their own blemishes, and they move from scan
    to scan, while the response stays on the slit. So, in log and per band::

        log scan_i(row) = t(row) + level_i(piece of row in scan i) + noise

    The joints (unknown rows inside a scan) split each scan into pieces. The levels are solved by
    least squares with ``t`` eliminated row by row, ``t`` is then the median of the levelled scans.
    With three scans or more, rows where one scan strays from the others (a blemish, the curled
    edge of a piece) are left out of that scan and the levels solved again. No shape is imposed on
    ``t``: the camera's own row-to-row pattern stays.

    Args:
        measured: Two or more ``(lines, bands)`` arrays (``compute_flatfield(...)["measured"]``,
            :meth:`FlatFieldProfile.measured`): the response, NaN on rows not measured.
        pad: Rows beside each joint also left out of that scan.
        stray_sigma: A row whose broadband residual (after a ``stray_window`` running mean is
            taken off) is this many robust sigmas off is a stray.
        stray_window: Rows of the running mean, odd.
        stray_grow: Rows added on each side of a stray.
        rounds: Times the strays are looked for and the levels solved again.
        leave_one_out: With three scans or more, also combine without each scan in turn and report
            the largest change (``leave_one_out``): how much one scan decides the result.
        step_threshold: Brightness steps over this (fraction) left in the combined response, on the
            rows two scans or more measure, are reported in ``steps``; ``None``: not looked for.

    Returns:
        dict: ``profile`` float32 ``(lines, bands)`` (median 1 per band, gaps filled);
        ``measured`` float32 the same with NaN where no scan is valid; ``scans_per_row`` int
        ``(lines,)``; ``joints`` per scan ``[(first, last), ...]``; ``strays`` per scan, the rows
        left out beyond its joints; ``levels`` per scan ``[(first row, last row, level), ...]``
        (broadband, relative); ``residual_rms`` per scan (fraction: what is left of it after its
        levels; large when the light was not the same); ``leave_one_out`` (fraction, the 95th
        percentile of the broadband change, or ``None``); ``single_rows`` / ``uncovered_rows``
        (rows with one / no valid scan); ``steps`` ``[(row, step), ...]`` as
        :func:`compute_flatfield` reports them.

    Raises:
        ValueError: Fewer than two scans, shapes that differ, or values that are not positive.
    """
    arrays = [np.asarray(m, dtype=np.float64) for m in measured]
    if len(arrays) < 2:
        raise ValueError("Combining needs two flat-field scans or more")
    shape = arrays[0].shape
    if len(shape) != 2 or any(a.shape != shape for a in arrays):
        raise ValueError(f"Flat-field scans must all be (lines, bands), got {[a.shape for a in arrays]}")
    stack = np.stack(arrays)
    if np.any(stack[np.isfinite(stack)] <= 0):
        raise ValueError("A flat-field scan has a response that is not positive")
    scans, lines, bands = stack.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        logf = np.log(stack)
    # a row is measured when any band is: a band missing on a measured row (a profile saved
    # before the NaN was kept to unmeasured rows) is that band's noise, not a joint; it is
    # filled from the band's neighbouring rows
    known = np.isfinite(logf).any(axis=2)
    logf = _fill_bands(logf, known)
    joints = [_gaps(k) for k in known]
    pieces = [np.searchsorted([b for _, b in j], np.arange(lines), side="left") for j in joints]
    base_valid = known.copy()
    for i, j in enumerate(joints):
        for a, b in j:
            base_valid[i, max(0, a - pad):b + pad + 1] = False
    counts = np.bincount(pieces[0][base_valid[0]], minlength=int(pieces[0].max()) + 1)
    anchor = (0, int(np.argmax(counts)))

    def solve(valid, use=None):
        use = list(range(scans)) if use is None else use
        sub_anchor = (use.index(anchor[0]), anchor[1]) if anchor[0] in use else (0, int(np.argmax(
            np.bincount(pieces[use[0]][valid[use[0]]], minlength=int(pieces[use[0]].max()) + 1))))
        return _solve_levels(logf[use], valid[use], [pieces[i] for i in use], sub_anchor)

    valid = base_valid.copy()
    t, offsets = solve(valid)
    if scans >= 3:
        half = max(1, int(stray_window) // 2)
        kernel = np.ones(2 * half + 1) / (2 * half + 1)
        for _ in range(max(0, int(rounds))):
            resid = np.stack([_broadband(logf[i] - offsets[i][pieces[i]] - t) for i in range(scans)])
            stray = np.zeros_like(valid)
            for i in range(scans):
                r = np.where(valid[i], resid[i], np.nan)
                filled = np.where(np.isfinite(r), r, np.nanmedian(r))
                high = filled - np.convolve(np.pad(filled, half, mode="edge"), kernel, mode="valid")
                spread = max(1.4826 * np.nanmedian(np.abs(high[valid[i]])), _STRAY_FLOOR)
                stray[i] = valid[i] & (np.abs(high) > float(stray_sigma) * spread)
            # a row is judged by the others only where at least two others are there to judge it
            others = base_valid.sum(axis=0)[None, :] - base_valid
            stray &= others >= 2
            grown = stray.copy()
            for s in range(1, int(stray_grow) + 1):
                grown[:, s:] |= stray[:, :-s]
                grown[:, :-s] |= stray[:, s:]
            # cumulative: strays are judged on the rows still valid, so the ones left out in an
            # earlier round stay out (rebuilding from base_valid would put them back every other round)
            new_valid = valid & ~grown
            if np.array_equal(new_valid, valid):
                break
            valid = new_valid
            t, offsets = solve(valid)

    per_row = valid.sum(axis=0)
    covered = per_row > 0
    if not covered.any():
        raise ValueError("No slit row is measured by any of the flat-field scans")
    response = np.exp(_fill_rows(t, covered))
    scale = np.median(response[covered], axis=0)
    profile = response / scale
    combined = np.where(covered[:, None], profile, np.nan)

    residual_rms = []
    for i in range(scans):
        r = _broadband(logf[i] - offsets[i][pieces[i]] - t)[valid[i]]       # broadband, like the strays
        r = r[np.isfinite(r)]
        residual_rms.append(float(np.sqrt(np.mean(r ** 2))) if r.size else float("nan"))
    levels = []
    for i in range(scans):
        starts = [0] + [b + 1 for _, b in joints[i]]
        ends = [a - 1 for a, _ in joints[i]] + [lines - 1]
        broadband = np.expm1(np.median(offsets[i], axis=1))
        levels.append([(int(s), int(e), round(float(v), 5)) for s, e, v in zip(starts, ends, broadband)])
    loo = None
    if leave_one_out and scans >= 3:
        worst = 0.0
        for drop in range(scans):
            part = combine_flatfields([a for i, a in enumerate(arrays) if i != drop], pad=pad,
                                      stray_sigma=stray_sigma, stray_window=stray_window,
                                      stray_grow=stray_grow, rounds=rounds, leave_one_out=False)
            both = np.isfinite(part["measured"][:, 0]) & covered
            if both.sum() < 2:
                continue
            d = np.log(part["measured"][both].astype(np.float64)) - np.log(combined[both])
            d = d - np.median(d, axis=0)
            worst = max(worst, float(np.percentile(np.abs(np.expm1(np.median(d, axis=1))), 95)))
        loo = worst
    strays = [[int(r) for r in np.where(base_valid[i] & ~valid[i])[0]] for i in range(scans)]
    # brightness steps the combination kept (joints that sat at the same rows in every scan):
    # judged, like a single scan's, on the broadband response, over the rows two scans or more agree on
    steps = []
    two = np.where(per_row >= 2)[0]
    if step_threshold is not None and two.size >= 8:
        relative = np.exp(_broadband(np.log(profile)))
        half = max(1, effective_seam_window(31, int(two.size)) // 2)
        steps = _lasting_steps(two, relative[two], half, float(step_threshold))
    return {"profile": profile.astype(np.float32), "measured": combined.astype(np.float32),
            "scans_per_row": per_row.astype(int), "joints": joints, "strays": strays, "levels": levels,
            "residual_rms": residual_rms, "leave_one_out": loo, "steps": steps,
            "single_rows": int((per_row == 1).sum()), "uncovered_rows": int((per_row == 0).sum())}


def plan_next_scan(scans_per_row, moving_rows, *, joints=(), earlier_joints=(), min_shift=None,
                   max_shift=None, spacing=None):
    """Where to move the flat-field target for the next scan, so that it helps most.

    The target's joints, curled edges and blemishes move with it. Two things make a next scan
    useful, in this order:

    1. Its joints land away from every joint seen so far (``joints`` and ``earlier_joints``), by
       ``spacing`` rows if the move allows: scans whose joints bunch up see the same stretch of the
       target at the same rows, and their pieces' levels and blemishes cannot be told apart.
    2. The rows it loses (``moving_rows`` shifted) fall on slit rows other scans already cover well:
       each lost row costs ``max(0, 3 - scans on it)``.

    Of equal merit the smaller move wins (the target stays over the slit more surely).

    Args:
        scans_per_row: ``(lines,)`` how many scans measure each slit row well now
            (``combine_flatfields(...)["scans_per_row"]``, or ``1`` / ``0`` for a single scan).
        moving_rows: Rows the latest scan loses that move with the target: its joints (and the rows
            beside them left out) and its strays.
        joints: The latest scan's joints ``[(first, last), ...]``: they move.
        earlier_joints: Joints of the other scans combined (any nesting of ``(first, last)`` pairs).
        min_shift: Smallest move in rows (default: a fiftieth of the slit, at least 10).
        max_shift: Largest move in rows (default: a sixth of the slit): the target must still span it.
        spacing: Rows between joints that count as apart (default: a twentieth of the slit, at
            least 10): about how far a curled edge of a piece reaches.

    Returns:
        dict: ``shift`` (rows; positive: towards the last slit row; ``0`` when nothing moves with
        the target), ``joints_now`` / ``joints_next`` (joint centre rows), ``thin_rows`` (rows
        measured by fewer than two scans now), ``thin_after`` (fewer than two after a scan at the
        shift), ``cost_now`` / ``cost_next`` (the cost at the same place / at the shift).
    """
    counts = np.asarray(scans_per_row, dtype=np.int64)
    lines = counts.size
    moving = np.unique(np.asarray([int(r) for r in moving_rows if 0 <= int(r) < lines], dtype=np.int64))
    weight = np.maximum(0, 3 - counts).astype(np.float64)
    centres = [int(round((a + b) / 2)) for a, b in joints]
    thin = int(np.count_nonzero(counts < 2))
    if moving.size == 0:
        return {"shift": 0, "joints_now": centres, "joints_next": centres, "thin_rows": thin,
                "thin_after": int(np.count_nonzero(counts + 1 < 2)), "cost_now": 0.0, "cost_next": 0.0}
    low = max(10, lines // 50) if min_shift is None else max(1, int(min_shift))
    high = max(low, lines // 6) if max_shift is None else max(low, int(max_shift))
    apart = max(10, lines // 20) if spacing is None else max(1, int(spacing))
    seen = np.array(centres + [int(round((a + b) / 2)) for a, b in _pairs(earlier_joints)], dtype=np.int64)

    def cost(shift):
        rows = moving + shift
        rows = rows[(rows >= 0) & (rows < lines)]
        return float(weight[rows].sum())

    def distance(shift):
        if not centres or not seen.size:
            return apart
        moved = np.array([c + shift for c in centres if 0 <= c + shift < lines])
        if not moved.size:
            return apart
        return int(min(apart, np.abs(moved[:, None] - seen[None, :]).min()))

    candidates = [s for s in range(-high, high + 1) if abs(s) >= low]
    on_slit = [s for s in candidates if all(0 <= c + s < lines for c in centres)]
    candidates = on_slit or candidates          # advice names rows where the joints go: keep them on the slit
    best = min(candidates, key=lambda s: (-distance(s), round(cost(s), 6), abs(s), s))
    after = counts + 1
    lost = moving + best
    after[lost[(lost >= 0) & (lost < lines)]] -= 1
    return {"shift": int(best), "joints_now": centres,
            "joints_next": [c + best for c in centres if 0 <= c + best < lines],
            "thin_rows": thin, "thin_after": int(np.count_nonzero(after < 2)),
            "cost_now": cost(0), "cost_next": cost(best)}


def _pairs(nested):
    """Every ``(first, last)`` pair in ``nested`` (a pair, a list of pairs, or lists of those)."""
    out = []
    for item in nested or ():
        if len(item) == 2 and all(np.isscalar(v) for v in item):
            out.append((int(item[0]), int(item[1])))
        else:
            out.extend(_pairs(item))
    return out


def effective_seam_window(seam_window, measured_rows):
    """The seam window :func:`compute_flatfield` uses: ``seam_window`` capped at a quarter of the
    measured slit rows, never below 3 (and never above ``seam_window``).

    Args:
        seam_window: Requested window in slit rows.
        measured_rows: Number of slit rows with a response.

    Returns:
        int: The window in slit rows.

    Example:
        >>> effective_seam_window(31, 1104), effective_seam_window(31, 60), effective_seam_window(9, 60)
        (31, 15, 9)
    """
    requested = max(1, int(seam_window))
    return min(requested, max(3, int(measured_rows) // 4))


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


def evening_factors(profile):
    """Factors ``1 / profile`` (float32, read-only) that even counts along the slit when multiplied
    in (frame by frame, as radiometric factors are): the same result as :func:`apply_flatfield`.

    Raises:
        ValueError: The profile is not finite and positive everywhere.
    """
    profile = check_profile(profile)
    factors = np.ascontiguousarray(1.0 / np.asarray(profile, np.float64), dtype=np.float32)
    factors.setflags(write=False)
    return factors


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
