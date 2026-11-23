"""Read-only bar-spacing diagnostic of the Keystone page (never changes the calibration).

The spacing of two neighbouring target bars is compared, column by column, with the spacing of the
same two bars at the reference column (their reference rows ``model.target_Y``), not with an
assumed physical pitch. Only MEASURED bar centres are used (tracer and gap-recovered points, at the
columns where both bars of a pair have one): nothing is read off a fitted curve, which would be
extrapolated beyond the points it was fitted to.

    before = (y_b - y_a)(column) - (Y_b - Y_a)        raw centres
    after  = (Yc_b - Yc_a)(column) - (Y_b - Y_a)      centres mapped to destination rows

After a good correction ``after`` is ~0 everywhere: each bar is a straight row at its reference row.
"""
from __future__ import annotations

import numpy as np


def spacing_variation(lines, model) -> dict:
    """Spacing change of adjacent bars against the reference column, before / after correction (px).

    Args:
        lines: Target bars (``KeystoneAnalysis.target_lines``), in the order of ``model.target_Y``.
        model: The fitted :class:`~modules.smile.keystone.GeometryModel`.

    Returns:
        dict: ``x`` (sampled columns), ``before`` / ``after`` (mean over the pairs per column, NaN
        where no pair is measured), ``before_rms`` / ``after_rms`` (over every pair point),
        ``pairs`` and ``points`` (counts behind them).
    """
    empty = dict(x=np.array([]), before=np.array([]), after=np.array([]),
                 before_rms=float("nan"), after_rms=float("nan"), pairs=0, points=0)
    if not lines or len(lines) < 2:
        return empty
    ref = list(getattr(model, "target_Y", []) or [])
    if len(ref) != len(lines):
        return empty
    bars = []
    for line, yr in zip(lines, ref):
        x = np.asarray(line.x, float); y = np.asarray(line.y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2:
            bars.append(None); continue
        x, y = x[ok], y[ok]
        _X, yc = model.destination(x, y, iterations=12)
        bars.append((float(yr), dict(zip(np.round(x).astype(int).tolist(), zip(y.tolist(), np.asarray(yc, float).tolist())))))
    order = sorted((b for b in bars if b is not None), key=lambda b: b[0])
    cols = sorted({c for _r, pts in order for c in pts})
    if not cols:
        return empty
    pos = {c: i for i, c in enumerate(cols)}
    sb = np.zeros(len(cols)); sa = np.zeros(len(cols)); n = np.zeros(len(cols))
    all_b, all_a, pairs = [], [], 0
    for (ra, pa), (rb, pb) in zip(order[:-1], order[1:]):
        common = sorted(set(pa) & set(pb))
        if not common:
            continue
        pairs += 1
        d_ref = rb - ra
        for c in common:
            b = (pb[c][0] - pa[c][0]) - d_ref
            a = (pb[c][1] - pa[c][1]) - d_ref
            i = pos[c]
            sb[i] += b; sa[i] += a; n[i] += 1
            all_b.append(b); all_a.append(a)
    if not all_b:
        return empty
    with np.errstate(invalid="ignore", divide="ignore"):
        before = np.where(n > 0, sb / n, np.nan)
        after = np.where(n > 0, sa / n, np.nan)
    all_b, all_a = np.asarray(all_b), np.asarray(all_a)
    return dict(x=np.asarray(cols, float), before=before, after=after,
                before_rms=float(np.sqrt(np.mean(all_b ** 2))), after_rms=float(np.sqrt(np.mean(all_a ** 2))),
                pairs=int(pairs), points=int(all_b.size))
