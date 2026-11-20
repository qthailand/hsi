"""Residual keystone of a corrected cube, measured on the horizontal edges of the scene (Qt-free).

After smile + keystone correction every scene feature should sit on the same slit row in every
band. A target with sharp edges across the slit (a checkerboard, a printed page, bar target)
shows what is left: :func:`check_keystone` finds the horizontal edges in a reference band and
measures, band by band, how far each edge has moved along the slit.

Method: the row derivative of each band (light Gaussian smoothing) is squared and averaged over
the scan columns where that edge is strong in the reference band; its centroid within
``half_window`` rows of the edge is the edge row in that band. The centroid of a symmetric blur
stays put, so defocus that changes with wavelength does not read as a shift. Thin features (a
printed line, whose two edges merge when blurred) are flagged ``thin`` and left out of the
statistics: an edge whose gradient changes sign within its window, or two edges closer than
``2 * half_window + 4`` rows.

Data conventions:
    ``cube`` is ``(slit, scan, wavelength)`` (ENVI BIP order; a ``numpy.memmap`` is fine),
    ``wavelengths`` in nm. Shifts are in pixels along the slit, positive = the edge lies lower
    in the frame (larger row) than in the reference band.

Threading:
    Pure functions; any thread. Reading every third band of a 1024 x 730 x 399 cube takes a few
    seconds from disk.

Example:
    >>> import numpy as np
    >>> cube = np.ones((200, 40, 30), np.float32)
    >>> cube[100:, :, :] = 3.0                        # one straight edge at row 100
    >>> r = check_keystone(cube, np.linspace(900, 1700, 30), band_step=1)
    >>> len(r.rows), round(float(np.nanmax(np.abs(r.shifts))), 3)
    (1, 0.0)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

#: Residual (px) below which the correction counts as good (same limit as the keystone re-trace).
PASS_PX = 0.25


@dataclass
class KeystoneCheck:
    """Result of :func:`check_keystone`.

    Attributes:
        rows: Row of each edge in the reference band (px, between two rows: ``k + 0.5``).
        wavelengths: Wavelength of each measured band (nm).
        shifts: ``(edges, bands)`` edge shift vs the reference band (px); NaN where the band is dark.
        thin: ``(edges,)`` True for edges of thin features (left out of the statistics).
        ref_nm: Reference band (nm).
        columns: ``(first, last)`` scan columns used.
        zones: Per wavelength zone: ``dict(lo, hi, median, p95, max, worst_row)`` (px, reliable edges).
        median, p95, max: Over all reliable edges and bands (px).
        worst: ``(row, nm, shift)`` of the largest reliable shift.
    """
    rows: np.ndarray
    wavelengths: np.ndarray
    shifts: np.ndarray
    thin: np.ndarray
    ref_nm: float
    columns: Tuple[int, int]
    zones: List[dict] = field(default_factory=list)
    median: float = float("nan")
    p95: float = float("nan")
    max: float = float("nan")
    worst: Tuple[float, float, float] = (float("nan"), float("nan"), float("nan"))

    @property
    def passed(self) -> bool:
        """``p95 < PASS_PX``: 95 % of the edge / band shifts are under the limit."""
        return bool(np.isfinite(self.p95) and self.p95 < PASS_PX)

    def verdict(self) -> str:
        """One line: PASS / CHECK with the numbers and, when failing, where."""
        if not np.isfinite(self.p95):
            return "No reliable edges measured."
        text = ("PASS" if self.passed else "CHECK") + (
            f" - residual keystone median {self.median:.3f} px, 95 % below {self.p95:.2f} px, "
            f"largest {self.max:.2f} px (row {self.worst[0]:.0f} at {self.worst[1]:.0f} nm)")
        bad = [z for z in self.zones if np.isfinite(z["p95"]) and z["p95"] >= PASS_PX]
        if bad and not self.passed:
            text += "; over the limit at " + ", ".join(f"{z['lo']:.0f}-{z['hi']:.0f} nm" for z in bad)
        return text


def check_keystone(cube, wavelengths, ref_nm: float = 1300.0, band_step: int = 3, half_window: int = 4,
                   row_margin: Tuple[int, int] = (0, 0), zone_nm: float = 100.0,
                   progress: Optional[Callable[[int], None]] = None) -> KeystoneCheck:
    """Measure the residual keystone of a corrected cube on its horizontal edges.

    Args:
        cube: ``(slit, scan, wavelength)`` cube (any dtype; memmap welcome).
        wavelengths: Band wavelengths (nm), one per band.
        ref_nm: Reference band; shifts are relative to it.
        band_step: Measure every ``band_step``-th band (the reference band is always measured).
        half_window: Rows on each side of an edge used for its centroid.
        row_margin: ``(top, bottom)`` rows left out (the keystone margin of the cube).
        zone_nm: Width of the wavelength zones of the summary.
        progress: Optional ``callable(percent)``.

    Returns:
        :class:`KeystoneCheck`.

    Raises:
        ValueError: Not a 3-D cube, wavelengths of another length, or no horizontal edge found.
    """
    if getattr(cube, "ndim", 0) != 3:
        raise ValueError("Use a 3-D cube (slit, scan, wavelength)")
    wl_all = np.asarray(wavelengths, float)
    L, S, B = cube.shape
    if wl_all.size != B:
        raise ValueError(f"{wl_all.size} wavelengths for {B} bands")
    report = progress or (lambda p: None)
    ref_b = int(np.argmin(np.abs(wl_all - ref_nm)))
    bsel = np.unique(np.r_[np.arange(0, B, max(1, int(band_step))), ref_b])
    ref = int(np.flatnonzero(bsel == ref_b)[0])
    w = wl_all[bsel]
    data = np.empty((L, S, bsel.size), np.float32)
    chunk = max(1, bsel.size // 20)
    for i in range(0, bsel.size, chunk):                    # in pieces: progress + bounded temporaries
        data[:, :, i:i + chunk] = np.asarray(cube[:, :, bsel[i:i + chunk]], np.float32)
        report(int(60 * min(1.0, (i + chunk) / bsel.size)))
    data = np.nan_to_num(data)
    top, bottom = (max(0, int(v)) for v in row_margin)

    level = data[:, :, ref].mean(axis=0)
    cols = np.flatnonzero(level > 0.5 * np.percentile(level, 95))
    if cols.size < 10:
        raise ValueError("Too few bright scan columns in the reference band")
    cols = cols[min(5, cols.size // 10): cols.size - min(5, cols.size // 10)]
    gy = np.diff(gaussian_filter1d(data, 0.7, axis=0), axis=0)          # (L-1, S, bands)
    report(75)
    gref = gy[:, :, ref]
    prof = np.mean(np.abs(gref[:, cols]), axis=1)
    lo_row, hi_row = top + half_window + 1, L - bottom - half_window - 3
    prof[:lo_row] = 0
    prof[hi_row:] = 0
    peaks, _ = find_peaks(prof, prominence=0.15 * prof.max() if prof.max() > 0 else 1.0, distance=max(3, half_window + 2))
    if peaks.size == 0:
        raise ValueError("No horizontal edges in the reference band: use a scene with sharp edges across the slit "
                         "(checkerboard, printed page, bar target)")
    # bands too dark to measure (outside the lit spectrum)
    band_level = np.array([np.median(data[:, cols, k]) for k in range(bsel.size)])
    dark = band_level < 0.05 * max(band_level[ref], 1e-12)

    rows, shifts, thin = [], [], []
    yy_off = np.arange(-half_window, half_window + 1)
    for e in peaks:
        g = np.abs(gref[e, cols])
        strong = cols[g > 0.5 * g.max()]
        if strong.size < 10:
            continue
        # a step edge has one gradient sign around it; a thin line (dark or bright) has both
        # (looked for over twice the window: a printed text line ~10 rows tall blurs both edges together)
        sgn = np.sign(gref[e, strong])
        reach = 2 * half_window + 3
        a, b = max(0, e - reach), min(gref.shape[0], e + reach + 1)
        q = (gref[a:b][:, strong] * sgn[None, :]).mean(axis=1)
        thin.append(bool(q.min() < -0.3 * q.max()))
        win = gy[e - half_window:e + half_window + 1][:, strong, :] ** 2        # (2hw+1, n, bands)
        p = win.mean(axis=1)
        p = p - p.min(axis=0, keepdims=True)
        tot = p.sum(axis=0)
        cen = np.where(tot > 0, (p * yy_off[:, None]).sum(axis=0) / np.maximum(tot, 1e-30), np.nan)
        s = cen - cen[ref]
        s[dark] = np.nan
        rows.append(e + 0.5)
        shifts.append(s)
    report(95)
    if not rows:
        raise ValueError("No horizontal edge strong enough across the scan")
    rows = np.asarray(rows)
    shifts = np.asarray(shifts)
    gap = np.full(rows.size, np.inf)
    if rows.size > 1:
        d = np.diff(rows)
        gap[:-1] = np.minimum(gap[:-1], d)
        gap[1:] = np.minimum(gap[1:], d)
    thin = np.asarray(thin, bool) | (gap < 2 * half_window + 4)
    res = KeystoneCheck(rows=rows, wavelengths=w, shifts=shifts, thin=thin, ref_nm=float(wl_all[ref_b]),
                        columns=(int(cols[0]), int(cols[-1])))
    _summarise(res, zone_nm)
    report(100)
    return res


def _summarise(res: KeystoneCheck, zone_nm: float) -> None:
    """Fill the statistics of *res* from its reliable (not thin) edges."""
    ok = ~res.thin
    a = np.abs(res.shifts[ok])
    if not ok.any() or not np.isfinite(a).any():
        return
    res.median = float(np.nanmedian(a))
    res.p95 = float(np.nanpercentile(a, 95))
    res.max = float(np.nanmax(a))
    i, k = np.unravel_index(int(np.nanargmax(a)), a.shape)
    res.worst = (float(res.rows[ok][i]), float(res.wavelengths[k]), float(res.shifts[ok][i, k]))
    w = res.wavelengths
    for lo in np.arange(np.floor(w.min() / zone_nm) * zone_nm, w.max(), zone_nm):
        sel = (w >= lo) & (w < lo + zone_nm)
        z = a[:, sel]
        if not sel.any() or not np.isfinite(z).any():
            continue
        r_i = int(np.nanargmax(np.nanmax(z, axis=1)))
        res.zones.append(dict(lo=float(max(lo, w.min())), hi=float(min(lo + zone_nm, w.max())),
                              median=float(np.nanmedian(z)), p95=float(np.nanpercentile(z, 95)),
                              max=float(np.nanmax(z)), worst_row=float(res.rows[ok][r_i])))
