"""Keystone (along-slit) correction joined with the smile into one 2-D geometry map.

The smile is measured from an emission-lamp frame (lines run along the slit); the keystone from a
*line target* frame: a halogen-lit target whose dark (or bright) bars cross the slit, so each bar
is one slit position and should be a horizontal line on the detector. Both are solved together
and applied as ONE remap (one interpolation):

    source_x = X + dx(X, Y)      smile:     dx(X, y_ref) = 0  (the wavelength axis of the reference row is kept)
    source_y = Y + dy(X, Y)      keystone:  dy(x_ref, Y) = 0  (every band is co-registered onto the slit
                                                                as seen at the reference column)

``(X, Y)`` are destination (corrected) pixels. Both shifts are low-order 2-D polynomials whose
basis enforces the anchors exactly (no ``y**0`` term in dx, no ``(x - x_ref)**0`` term in dy).
They are fitted jointly, alternating: a lamp point measured at raw row ``y`` belongs to the
destination row ``Y = y - dy(X, Y)``, a bar point measured at raw column ``x`` to the destination
column ``X = x - dx(X, Y)``. Outside the measured span the shifts hold the edge value along the
spectrum (as the smile-only model does) and continue linearly along the slit.

Accuracy notes:
    * Line and bar centres are intensity centroids in a window that never reaches a neighbour
      (robust to the broad, double-lobed lamp lines of some spectrographs).
    * Uneven illumination along the slit that changes with wavelength biases a bar centre
      differently per column, which looks like keystone. Pass a *white* frame (same light and
      distance, target removed) to divide it out; without one a slow trend along the slit is
      removed with a wide running mean.
    * The rotation of the target in its plane does not matter (every bar still crosses the slit
      at one point); its focus and working distance do.

Data conventions:
    Frames are 2-D ``(slit rows, spectral columns)``, zero-based pixel centres, the same
    orientation / hardware bin / size as the frames the model will be applied to (dark-subtracted
    when the live pipeline subtracts dark before the remap). Maps are OpenCV destination ->
    source ``float32`` arrays.

Threading:
    Pure NumPy / OpenCV, no shared state: any thread.

Example:
    >>> from modules.smile import SmileCorrector                          # doctest: +SKIP
    >>> c = SmileCorrector().fit(lamp)                                     # doctest: +SKIP
    >>> c.fit_keystone(target, white=white)                                # doctest: +SKIP
    >>> c.save('smile_model_bin2.npz', camera_setting)                     # doctest: +SKIP
"""
from __future__ import annotations

import copy
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks

from .core import _INTERP, _centroid_row, auto_sat_level, estimate_noise, robust_sigma, trace_line

__all__ = [
    "KEYSTONE_SCHEMA", "TargetLine", "GeometrySurface", "GeometryModel", "KeystoneAnalysis",
    "normalize_target", "trace_target_lines", "lamp_points_from_analysis", "fit_geometry",
    "analyze_keystone", "row_margin_from_map", "keystone_report",
]

KEYSTONE_SCHEMA = "hsi-smile-keystone/1"
#: Target bars must span at least this share of the slit (rows).
MIN_SLIT_COVERAGE = 0.5
#: A bar is kept when it is followed over at least this share of the columns.
MIN_BAR_COVERAGE = 0.4
#: A lamp line enters the joint fit when it is traced over this share of the rows (and spans y_ref).
MIN_LAMP_COVERAGE = 0.5
#: Re-traced straightness (px RMS of the median line / bar) above which the result carries a warning.
RETRACE_WARN_PX = 0.25
#: Highest keystone degree along the slit tried by ``dy_ydeg="auto"``.
MAX_DY_YDEG = 4


# ----------------------------------------------------------------------------------------------- basis
class GeometrySurface:
    """``f(X, Y) = sum_ij C[i, j] * yn**ypows[i] * xn**xpows[j]`` with ``xn = (X - x0) / sx``,
    ``yn = (Y - y0) / sy``.

    Evaluation outside the measured box: ``X`` is clamped to ``[xlo, xhi]`` (edge value held along
    the spectrum); ``Y`` outside ``[ylo, yhi]`` continues linearly from the edge (value + slope),
    so a magnification term keeps scaling the rows instead of a cubic blowing up.
    """

    def __init__(self, xpows, ypows, x0, sx, y0, sy, coef=None, box=None):
        self.xpows = [int(v) for v in xpows]
        self.ypows = [int(v) for v in ypows]
        self.x0, self.sx, self.y0, self.sy = float(x0), float(sx), float(y0), float(sy)
        n = (len(self.ypows), len(self.xpows))
        self.C = np.zeros(n) if coef is None else np.asarray(coef, float).reshape(n)
        self.box = None if box is None else [float(v) for v in box]     # [xlo, xhi, ylo, yhi]

    # -- basis
    def _bx(self, X, deriv=False):
        xn = (np.asarray(X, float).ravel() - self.x0) / self.sx
        return self._pow(xn, self.xpows, deriv) / (self.sx if deriv else 1.0)

    def _by(self, Y, deriv=False):
        yn = (np.asarray(Y, float).ravel() - self.y0) / self.sy
        return self._pow(yn, self.ypows, deriv) / (self.sy if deriv else 1.0)

    @staticmethod
    def _pow(t, pows, deriv):
        out = np.empty((t.size, len(pows)))
        for k, p in enumerate(pows):
            if deriv:
                out[:, k] = p * t ** (p - 1) if p > 0 else 0.0
            else:
                out[:, k] = t ** p
        return out

    def design(self, X, Y):
        bx, by = self._bx(X), self._by(Y)
        return (by[:, :, None] * bx[:, None, :]).reshape(len(bx), -1)

    def raw(self, X, Y):
        """Polynomial value at points, no box rule."""
        X, Y = np.broadcast_arrays(np.asarray(X, float), np.asarray(Y, float))
        return np.sum((self._by(Y) @ self.C) * self._bx(X), axis=1).reshape(X.shape)

    def fit(self, X, Y, values, weights=None):
        A = self.design(X, Y)
        v = np.asarray(values, float).ravel()
        if weights is not None:
            w = np.sqrt(np.asarray(weights, float).ravel())
            A, v = A * w[:, None], v * w
        n = A.shape[1]
        AtA = A.T @ A
        coef = np.linalg.solve(AtA + 1e-12 * max(1.0, np.trace(AtA) / n) * np.eye(n), A.T @ v)
        self.C = coef.reshape(self.C.shape)
        return self

    # -- evaluation with the box rule
    def _clamped(self, X, Y):
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        if self.box is None:
            return X, Y, np.zeros_like(Y)
        Xc = np.clip(X, self.box[0], self.box[1])
        Yc = np.clip(Y, self.box[2], self.box[3])
        return Xc, Yc, Y - Yc

    def at(self, X, Y):
        """Value at points with the box rule."""
        X, Y = np.broadcast_arrays(np.asarray(X, float), np.asarray(Y, float))
        Xc, Yc, dY = self._clamped(X.ravel(), Y.ravel())
        bx = self._bx(Xc)
        f = np.sum((self._by(Yc) @ self.C) * bx, axis=1)
        f = f + np.sum((self._by(Yc, True) @ self.C) * bx, axis=1) * dY
        return f.reshape(X.shape)

    def grid(self, xs, ys):
        """Values on the grid ``ys`` (rows) x ``xs`` (columns), box rule, separable."""
        Xc, Yc, dY = self._clamped(np.asarray(xs, float), np.asarray(ys, float))
        bx = self._bx(Xc)
        by = self._by(Yc) + self._by(Yc, True) * np.asarray(dY, float).ravel()[:, None]
        return by @ self.C @ bx.T

    def to_json(self) -> dict:
        return dict(xpows=self.xpows, ypows=self.ypows, x0=self.x0, sx=self.sx, y0=self.y0, sy=self.sy,
                    coef=self.C.tolist(), box=self.box)

    @classmethod
    def from_json(cls, d: dict) -> "GeometrySurface":
        return cls(d["xpows"], d["ypows"], d["x0"], d["sx"], d["y0"], d["sy"], d["coef"], d.get("box"))


@dataclass
class GeometryModel:
    """The joint smile + keystone model of one frame layout.

    Attributes:
        shape: ``(H, W)`` of the frames.
        y_ref: Reference row (``dx = 0``).
        x_ref: Reference column (``dy = 0``).
        dx: Smile surface (columns), anchored at ``y_ref``.
        dy: Keystone surface (rows), anchored at ``x_ref``.
        lamp_X: Destination column of every lamp line used.
        target_Y: Destination row of every target bar used.
    """

    shape: Tuple[int, int]
    y_ref: float
    x_ref: float
    dx: GeometrySurface
    dy: GeometrySurface
    lamp_X: List[float] = field(default_factory=list)
    target_Y: List[float] = field(default_factory=list)

    def maps(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(map_x, map_y)`` float32 ``(H, W)`` destination -> source maps."""
        h, w = self.shape
        xs = np.arange(w, dtype=float); ys = np.arange(h, dtype=float)
        mx = xs[None, :] + self.dx.grid(xs, ys)
        my = ys[:, None] + self.dy.grid(xs, ys)
        return np.ascontiguousarray(mx, np.float32), np.ascontiguousarray(my, np.float32)

    def source(self, X, Y):
        """Raw (source) coordinates of destination points: ``(x, y)``."""
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        return X + self.dx.at(X, Y), Y + self.dy.at(X, Y)

    def destination(self, x, y, iterations: int = 8):
        """Destination coordinates of raw points (fixed-point inverse of :meth:`source`)."""
        x = np.asarray(x, float); y = np.asarray(y, float)
        X, Y = x.copy(), y.copy()
        for _ in range(iterations):
            X = x - self.dx.at(X, Y)
            Y = y - self.dy.at(X, Y)
        return X, Y

    def to_json(self) -> dict:
        return dict(schema=KEYSTONE_SCHEMA, shape=[int(v) for v in self.shape], y_ref=float(self.y_ref),
                    x_ref=float(self.x_ref), dx=self.dx.to_json(), dy=self.dy.to_json(),
                    lamp_X=[float(v) for v in self.lamp_X], target_Y=[float(v) for v in self.target_Y])

    @classmethod
    def from_json(cls, d: dict) -> "GeometryModel":
        if d.get("schema") != KEYSTONE_SCHEMA:
            raise ValueError(f"Not a smile + keystone geometry model ({d.get('schema')!r}).")
        return cls(shape=tuple(int(v) for v in d["shape"]), y_ref=float(d["y_ref"]), x_ref=float(d["x_ref"]),
                   dx=GeometrySurface.from_json(d["dx"]), dy=GeometrySurface.from_json(d["dy"]),
                   lamp_X=list(d.get("lamp_X", [])), target_Y=list(d.get("target_Y", [])))


# ----------------------------------------------------------------------------------------------- target
@dataclass
class TargetLine:
    """One bar of the line target, traced across the spectrum.

    Attributes:
        y_seed: Row of the bar at the seed column (px).
        x: Sampled column centres (px) where the bar was found.
        y: Bar centre (row, px) at each sampled column.
        coef: Robust cubic ``y(xn)``, ``xn = (x - xc) / sx`` (frame centre / half width), increasing powers.
        coverage: Share of the frame width followed.
        halfwin: Centroid half-window (rows) used for this bar (never reaches a neighbour).
        span: Half distance (rows) between the outer dips of a merged pair (0 for a single bar).
    """

    y_seed: float
    x: np.ndarray
    y: np.ndarray
    coef: np.ndarray
    coverage: float
    halfwin: int = 0
    span: float = 0.0

    def y_at(self, x, w: int) -> np.ndarray:
        xc, sx = (w - 1) / 2.0, max(1.0, (w - 1) / 2.0)
        return np.polynomial.polynomial.polyval((np.asarray(x, float) - xc) / sx, self.coef)


def normalize_target(target, white=None, dark=None, trend_rows: Optional[int] = None,
                     min_white_fraction: float = 0.08) -> np.ndarray:
    """Target frame with the illumination divided out (bright background ~ 1).

    Args:
        target: Line-target frame ``(H, W)``.
        white: Same setup with the target removed (optional). Divides out the light, vignetting
            and pixel response, which is the safest way to avoid a false keystone.
        dark: Dark frame subtracted from both (optional).
        trend_rows: Without *white*: window (rows) of the running mean that removes the slow trend
            along the slit; ``None`` uses ~1/8 of the height (at least 31 rows).
        min_white_fraction: Pixels whose white level is below this share of the white's 99th
            percentile (unlit) become 0.

    Returns:
        float32 ``(H, W)``; unlit pixels 0.
    """
    t = np.asarray(target, np.float32)
    if t.ndim != 2:
        raise ValueError("The target frame must be 2-D (slit rows, spectral columns).")
    d = None if dark is None else np.asarray(dark, np.float32)
    if d is not None:
        if d.shape != t.shape:
            raise ValueError(f"Dark {d.shape} differs from the target {t.shape}.")
        t = t - d
    if white is not None:
        wf = np.asarray(white, np.float32)
        if wf.shape != t.shape:
            raise ValueError(f"White {wf.shape} differs from the target {t.shape}.")
        if d is not None:
            wf = wf - d
        lit = wf > min_white_fraction * float(np.percentile(wf, 99))
        out = np.zeros_like(t)
        out[lit] = t[lit] / wf[lit]
        return out
    h = t.shape[0]
    n = int(trend_rows) if trend_rows else max(31, h // 8)
    n += 1 - n % 2
    trend = uniform_filter1d(t, size=n, axis=0, mode="nearest")
    lit = trend > min_white_fraction * float(np.percentile(trend, 99))
    out = np.zeros_like(t)
    out[lit] = t[lit] / trend[lit]
    return out


def _band_columns(img: np.ndarray, band: int) -> np.ndarray:
    """Mean over ``band`` neighbouring columns, centred (same width as *img*)."""
    return uniform_filter1d(img, size=max(1, int(band)), axis=1, mode="nearest")


def _robust_poly(x, y, deg, k=4.0, floor=0.05, iters=4):
    """Polynomial fit with MAD rejection; returns ``(coef, keep)``."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    keep = np.ones(x.size, bool)
    deg = int(min(deg, max(0, x.size - 2)))
    coef = np.polynomial.polynomial.polyfit(x, y, deg)
    for _ in range(iters):
        if keep.sum() <= deg + 2:
            break
        coef = np.polynomial.polynomial.polyfit(x[keep], y[keep], deg)
        r = y - np.polynomial.polynomial.polyval(x, coef)
        sig = max(floor, robust_sigma(r[keep]))
        new = np.abs(r) <= k * sig
        if new.sum() <= deg + 2 or np.array_equal(new, keep):
            break
        keep = new
    coef = np.polynomial.polynomial.polyfit(x[keep], y[keep], deg)
    return coef, keep


def _bar_profile_noise(prof: np.ndarray) -> float:
    return max(robust_sigma(np.diff(prof)) / math.sqrt(2.0), 1e-6)


def _column_noise(img: np.ndarray) -> np.ndarray:
    """Per-column noise of slit profiles (robust sigma of row-to-row differences / sqrt 2).

    Row differences also contain the bar edges, so the median absolute difference over-estimates
    the noise a little where the bars are sharp; it is a conservative detection floor.
    """
    d = np.diff(np.asarray(img, np.float64), axis=0)
    med = np.median(d, axis=0)
    sig = 1.4826 * np.median(np.abs(d - med), axis=0) / math.sqrt(2.0)
    return np.maximum(sig, 1e-6)


def _merge_close(peaks: np.ndarray, near: float = 0.6):
    """Group dips closer than ``near`` x the median spacing into one feature.

    A pair of thin lines that the optics resolve in one part of the spectrum and blur together in
    another would make a centroid locked on one of them jump; the centroid of the whole pair
    stays put. Returns a list of ``(centre, half_span)`` (``half_span`` 0 for a single dip) and
    the list of merged groups (rows) for the report.
    """
    peaks = np.sort(np.asarray(peaks, float))
    spacing = float(np.median(np.diff(peaks))) if len(peaks) > 1 else 0.0
    groups = [[peaks[0]]] if len(peaks) else []
    for p in peaks[1:]:
        if p - groups[-1][-1] < near * spacing:
            groups[-1].append(p)
        else:
            groups.append([p])
    feats = [(0.5 * (g[0] + g[-1]), 0.5 * (g[-1] - g[0])) for g in groups]
    merged = [[float(v) for v in g] for g in groups if len(g) > 1]
    return feats, merged


def _feature_halfwins(feats, base: int) -> List[int]:
    """Centroid half-window per feature: its half span + ``base``, never past half-way to a neighbour's dip."""
    out = []
    for i, (c, s) in enumerate(feats):
        r = s + base
        for j in (i - 1, i + 1):
            if 0 <= j < len(feats):
                cj, sj = feats[j]
                d = abs(cj - c) - sj                    # distance to the neighbour's nearest dip
                r = min(r, 0.5 * (d + s))               # half-way between own edge dip and that one
        out.append(int(max(2, math.floor(r))))
    return out


def _neighbour_halfwins(rows: np.ndarray, halfwin: int) -> List[int]:
    """Centroid half-window per bar: ``halfwin``, but never past half-way to a neighbour."""
    return _feature_halfwins([(float(r), 0.0) for r in rows], halfwin)


def trace_target_lines(target, polarity: str = "dark", white=None, dark=None, band: int = 5, step: Optional[int] = None,
                       snr_min: float = 5.0, sat_level: Optional[float] = None, seed_column: Optional[float] = None,
                       max_gap: Optional[int] = None, min_coverage: float = MIN_BAR_COVERAGE, normalized=None
                       ) -> Tuple[List[TargetLine], dict]:
    """Follow the bars of a line-target frame across the spectrum.

    Each bar is located per sampled column as the intensity centroid of the (inverted, for dark
    bars) slit profile in a window of ~0.45 bar periods, re-centred iteratively, starting from the
    brightest part of the spectrum and walking outward with a jump limit.

    Args:
        target: Line-target frame ``(H, W)`` (raw counts).
        polarity: ``"dark"`` (bars darker than the background) or ``"bright"``.
        white, dark: See :func:`normalize_target`.
        band: Columns averaged per sample (noise).
        step: Column step between samples; ``None`` = *band*.
        snr_min: Minimum bar depth over the profile noise.
        sat_level: Sensor full scale (``None``: inferred). Windows touching clipped pixels are skipped.
        seed_column: Column where bars are found first; ``None``: the brightest column of the middle 60 %.
        max_gap: Consecutive failed samples before a bar is given up in that direction; ``None``:
            about 4 % of the width (crosses a dim stripe of the target or the light).
        min_coverage: Minimum share of the width a bar must be followed over.
        normalized: Pre-computed :func:`normalize_target` output (skips normalisation).

    Returns:
        ``(lines, info)``; lines sorted by row. ``info``: ``seed_column``, ``period``, ``halfwin``,
        ``noise``, ``rejected`` (bars dropped and why).

    Raises:
        ValueError: Fewer than three bars, or bars spanning less than ``MIN_SLIT_COVERAGE`` of the slit.
    """
    raw = np.asarray(target, np.float32)
    if raw.ndim != 2 or min(raw.shape) < 16:
        raise ValueError("Use a 2-D target frame of at least 16 x 16 pixels.")
    h, w = raw.shape
    if polarity not in ("dark", "bright"):
        raise ValueError("polarity must be 'dark' or 'bright'")
    sign = -1.0 if polarity == "dark" else 1.0
    if sat_level is None or sat_level <= 0:
        sat_level = auto_sat_level(raw)
    clipped = raw >= 0.98 * float(sat_level)
    norm = normalize_target(raw, white, dark) if normalized is None else np.asarray(normalized, np.float32)
    lit = norm > 0
    band = max(1, int(band))
    step = max(1, int(step or band))
    img = sign * _band_columns(norm, band)
    clip_cols = _band_columns(clipped.astype(np.float32), band) > 0
    lit_cols = _band_columns(lit.astype(np.float32), band) > 0.99

    level = np.median(np.where(lit, norm, 0.0), axis=0)
    if seed_column is None:
        lo, hi = int(0.2 * w), max(int(0.2 * w) + 1, int(0.8 * w))
        seed = int(lo + np.argmax(gaussian_filter1d(level[lo:hi], 5)))
    else:
        seed = int(np.clip(round(seed_column), 0, w - 1))
    noise_col = _column_noise(img)                      # bar contrast and noise change along the spectrum
    prof = img[:, seed]
    noise = float(noise_col[seed])
    sm = gaussian_filter1d(prof, 1.0)
    peaks, props = find_peaks(sm, prominence=snr_min * noise)
    prom = props["prominences"] if len(peaks) else np.zeros(0)
    if len(peaks) >= 2:
        keep = prom >= 0.25 * float(np.percentile(prom, 90))
        peaks, prom = peaks[keep], prom[keep]
    inside = lit[peaks, seed] if len(peaks) else np.zeros(0, bool)
    peaks = peaks[inside]
    if len(peaks) < 3:
        raise ValueError("Fewer than three target bars found at the seed column. Check the halogen light, the "
                         "bar polarity (dark / bright) and the exposure.")
    feats, merged = _merge_close(peaks)
    centres = np.array([c for c, _ in feats])
    period = float(np.median(np.diff(centres))) if len(centres) > 1 else float(np.median(np.diff(peaks)))
    halfwin = int(max(2, math.floor(0.45 * period)))
    halfwins = _feature_halfwins(feats, halfwin)
    if max_gap is None:
        max_gap = max(6, int(math.ceil(0.04 * w / step)))
    cols = np.arange(seed % step, w, step)
    cols = cols[(cols >= band // 2) & (cols <= w - 1 - band // 2)]
    if seed not in cols:
        cols = np.unique(np.r_[cols, seed])
    rejected = []
    lines: List[TargetLine] = []
    for p, hw in zip(centres, halfwins):
        max_jump = max(1.0, 0.5 * hw)
        pts: Dict[int, float] = {}
        c0, ok = _centroid_row(img[:, seed], float(p), hw, np.inf, snr_min * noise)
        if not ok or abs(c0 - p) > max_jump:
            rejected.append((float(p), "no centroid at the seed column"))
            continue
        pts[seed] = c0
        for direction in (-1, 1):
            chosen = cols[cols < seed][::-1] if direction < 0 else cols[cols > seed]
            yc, gap = c0, 0
            for c in chosen:
                r0, r1 = int(round(yc)) - hw, int(round(yc)) + hw + 1
                bad = r0 < 0 or r1 > h or clip_cols[max(0, r0):r1, c].any() or not lit_cols[max(0, r0):r1, c].all()
                if not bad:
                    yy, ok = _centroid_row(img[:, c], yc, hw, np.inf, snr_min * float(noise_col[c]))
                    bad = not ok or abs(yy - yc) > max_jump
                if bad:
                    gap += 1
                    if gap > max_gap:
                        break
                    continue
                pts[int(c)] = yy
                yc, gap = yy, 0
        xs = np.array(sorted(pts), float)
        ys = np.array([pts[int(c)] for c in xs], float)
        cov = float(np.ptp(xs) / max(1, w - 1)) if xs.size else 0.0
        if xs.size < 8 or cov < min_coverage:
            rejected.append((float(p), "followed over %.0f %% of the width only" % (100 * cov)))
            continue
        xn = (xs - (w - 1) / 2.0) / max(1.0, (w - 1) / 2.0)
        coef, keep = _robust_poly(xn, ys, 3)
        lines.append(TargetLine(y_seed=float(c0), x=xs[keep], y=ys[keep], coef=coef, coverage=cov,
                                halfwin=int(hw), span=float(dict(feats).get(float(p), 0.0))))
    lines.sort(key=lambda t: t.y_seed)
    # drop bars that merged with a neighbour (same track)
    out: List[TargetLine] = []
    for t in lines:
        if out and abs(t.y_seed - out[-1].y_seed) < 2.0:
            rejected.append((t.y_seed, "duplicate of the bar at row %.1f" % out[-1].y_seed))
            continue
        out.append(t)
    if len(out) < 3:
        raise ValueError("Fewer than three target bars could be followed across the spectrum. Check the halogen "
                         "light, the focus and the bar polarity.")
    span = out[-1].y_seed - out[0].y_seed
    if span < MIN_SLIT_COVERAGE * (h - 1):
        raise ValueError("The target bars span only %.0f %% of the slit (minimum %.0f %%). Place the line target "
                         "across the whole slit." % (100 * span / max(1, h - 1), 100 * MIN_SLIT_COVERAGE))
    info = dict(seed_column=seed, period=period, halfwin=halfwin, noise=noise, band=band, step=step,
                polarity=polarity, rejected=rejected, merged=merged)
    return out, info


# ----------------------------------------------------------------------------------------------- lamp points
@dataclass
class _LampLine:
    column: float          # detection column (id)
    y: np.ndarray          # raw rows
    x: np.ndarray          # raw centres
    coef: np.ndarray       # robust quadratic x(yn), yn = (y - y_ref) / sy


def lamp_points_from_analysis(analysis, min_coverage: float = MIN_LAMP_COVERAGE) -> Tuple[List[_LampLine], list]:
    """Lamp-line traces of a :class:`~modules.smile.analysis.SmileAnalysis` usable for the joint fit.

    Saturated lines are already excluded by the smile analysis; here lines traced over less than
    *min_coverage* of the rows, or not spanning the reference row, are left out too (their
    destination column would be extrapolated).

    Returns:
        ``(lines, rejected)``.
    """
    h = analysis.frame_orig.shape[0]
    y_ref = float(analysis.y_ref)
    sy = max(1.0, (h - 1) / 2.0)
    out, rejected = [], []
    for ln in analysis.used_lines:
        c = np.asarray(analysis.traces_orig[ln.column], float)
        rows = np.flatnonzero(np.isfinite(c))
        cov = rows.size / float(h)
        if cov < min_coverage or rows.size < 20 or not (rows.min() <= y_ref <= rows.max()):
            rejected.append((float(ln.column), "traced over %.0f %% of the slit" % (100 * cov)))
            continue
        y = rows.astype(float); x = c[rows]
        coef, keep = _robust_poly((y - y_ref) / sy, x, 2, floor=0.05)
        out.append(_LampLine(column=float(ln.column), y=y[keep], x=x[keep], coef=coef))
    out.sort(key=lambda t: t.coef[0])
    return out, rejected


# ----------------------------------------------------------------------------------------------- joint fit
def _huber_weights(r: np.ndarray, keep: np.ndarray, k: float = 1.5, reject: float = 4.0, floor: float = 0.02):
    sig = max(floor, robust_sigma(r[keep])) if keep.any() else floor
    u = np.abs(r) / (k * sig)
    w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-9))
    w[np.abs(r) > reject * sig] = 0.0
    return w


#: Floor (px) of a feature's scatter in the inverse-variance weights (no single line dominates).
FEATURE_SIGMA_FLOOR = 0.05


def _feature_weights(r: np.ndarray, keep: np.ndarray, idx: np.ndarray, n: int,
                     floor: float = FEATURE_SIGMA_FLOOR) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse-variance weight of every line / bar from its own scatter around the joint model.

    A target bar whose profile is asymmetric (a wide or two-level bar) has a centroid that moves
    with the wavelength-dependent blur: a smooth false keystone of its own. Such a bar scatters
    more around the common geometry and gets less weight, without a hard threshold.

    Returns:
        ``(weights, sigma)`` per feature; weights normalised to median 1 (0 for a feature with no
        kept point).
    """
    sig = np.full(n, np.nan)
    for i in range(n):
        m = keep & (idx == i)
        if m.any():
            sig[i] = _rms(r[m])
    w = np.where(np.isfinite(sig), 1.0 / np.maximum(np.nan_to_num(sig, nan=1.0), floor) ** 2, 0.0)
    pos = w[w > 0]
    if pos.size:
        w = w / float(np.median(pos))
    return w, sig


def _surfaces(shape, y_ref, x_ref, dx_deg, dy_deg):
    h, w = shape
    sx, sy = max(1.0, (w - 1) / 2.0), max(1.0, (h - 1) / 2.0)
    xc, yc = (w - 1) / 2.0, (h - 1) / 2.0
    # dx: powers of (Y - y_ref) start at 1 -> dx(X, y_ref) = 0; any power of X (centred)
    sdx = GeometrySurface(range(0, dx_deg[1] + 1), range(1, dx_deg[0] + 1), xc, sx, y_ref, sy)
    # dy: powers of (X - x_ref) start at 1 -> dy(x_ref, Y) = 0; any power of Y (centred)
    sdy = GeometrySurface(range(1, dy_deg[1] + 1), range(0, dy_deg[0] + 1), x_ref, sx, yc, sy)
    return sdx, sdy


class _Data:
    def __init__(self, lamp: List[_LampLine], target: List[TargetLine], shape, y_ref):
        h, w = shape
        self.h, self.w = h, w
        self.y_ref = float(y_ref)
        self.sy = max(1.0, (h - 1) / 2.0)
        self.lamp, self.target = lamp, target
        self.ly = np.concatenate([t.y for t in lamp]); self.lx = np.concatenate([t.x for t in lamp])
        self.lk = np.concatenate([np.full(t.y.size, i) for i, t in enumerate(lamp)])
        self.tx = np.concatenate([t.x for t in target]); self.ty = np.concatenate([t.y for t in target])
        self.tm = np.concatenate([np.full(t.x.size, i) for i, t in enumerate(target)])

    def lamp_x_at(self, k: int, y: float) -> float:
        return float(np.polynomial.polynomial.polyval((y - self.y_ref) / self.sy, self.lamp[k].coef))

    def bar_y_at(self, m: int, x: float) -> float:
        return float(self.target[m].y_at([x], self.w)[0])


def _solve(data: _Data, sdx: GeometrySurface, sdy: GeometrySurface, x_ref: float, iterations: int = 10):
    y_ref = data.y_ref
    Xk = np.array([data.lamp_x_at(k, y_ref) for k in range(len(data.lamp))])
    Ym = np.array([data.bar_y_at(m, x_ref) for m in range(len(data.target))])
    wl = np.ones(data.lx.size); wt = np.ones(data.tx.size)
    for it in range(iterations):
        Xl = Xk[data.lk]
        Yl = data.ly.copy()
        for _ in range(3):
            Yl = data.ly - sdy.raw(Xl, Yl)
        sdx.fit(Xl, Yl, data.lx - Xl, wl)
        Yt = Ym[data.tm]
        Xt = data.tx.copy()
        for _ in range(3):
            Xt = data.tx - sdx.raw(Xt, Yt)
        sdy.fit(Xt, Yt, data.ty - Yt, wt)
        newX = np.array([data.lamp_x_at(k, y_ref + float(sdy.raw(Xk[k], y_ref))) for k in range(len(data.lamp))])
        newY = np.array([data.bar_y_at(m, x_ref + float(sdx.raw(x_ref, Ym[m]))) for m in range(len(data.target))])
        moved = max(float(np.max(np.abs(newX - Xk))), float(np.max(np.abs(newY - Ym))))
        Xk, Ym = newX, newY
        if it >= 1:
            rl = data.lx - (Xl + sdx.raw(Xl, Yl))
            rt = data.ty - (Yt + sdy.raw(Xt, Yt))
            hl = _huber_weights(rl, wl > 0); ht = _huber_weights(rt, wt > 0)
            fl, _ = _feature_weights(rl, hl > 0, data.lk, len(data.lamp))
            ft, _ = _feature_weights(rt, ht > 0, data.tm, len(data.target))
            wl = hl * fl[data.lk]; wt = ht * ft[data.tm]
        if it >= 3 and moved < 1e-4:
            break
    # final destination coordinates and residuals with the final anchors
    Xl = Xk[data.lk]; Yl = data.ly.copy()
    for _ in range(3):
        Yl = data.ly - sdy.raw(Xl, Yl)
    Yt = Ym[data.tm]; Xt = data.tx.copy()
    for _ in range(3):
        Xt = data.tx - sdx.raw(Xt, Yt)
    rl = data.lx - (Xl + sdx.raw(Xl, Yl))
    rt = data.ty - (Yt + sdy.raw(Xt, Yt))
    fl, sl = _feature_weights(rl, wl > 0, data.lk, len(data.lamp))
    ft, st = _feature_weights(rt, wt > 0, data.tm, len(data.target))
    return dict(Xk=Xk, Ym=Ym, Xl=Xl, Yl=Yl, Xt=Xt, Yt=Yt, rl=rl, rt=rt, wl=wl, wt=wt,
                lamp_weight=fl, lamp_sigma=sl, bar_weight=ft, bar_sigma=st)


def _rms(v) -> float:
    v = np.asarray(v, float)
    return float(np.sqrt(np.mean(v * v))) if v.size else float("nan")


def _loo(data: _Data, sol: dict, make, which: str):
    """Leave-one-line-out RMS (px) of one surface on the solved destination coordinates."""
    if which == "x":
        n, idx, X, Y, v, w = len(data.lamp), data.lk, sol["Xl"], sol["Yl"], data.lx - sol["Xl"], sol["wl"]
    else:
        n, idx, X, Y, v, w = len(data.target), data.tm, sol["Xt"], sol["Yt"], data.ty - sol["Yt"], sol["wt"]
    if n < 3:
        return None, []
    per = []
    for i in range(n):
        s = make()
        m = (idx != i) & (w > 0)
        try:
            s.fit(X[m], Y[m], v[m], w[m])
        except np.linalg.LinAlgError:
            return None, []
        sel = (idx == i) & (w > 0)
        per.append(_rms(s.raw(X[sel], Y[sel]) - v[sel]))
    return _rms(per), per


def fit_geometry(lamp: List[_LampLine], target: List[TargetLine], shape, y_ref: float, x_ref: Optional[float] = None,
                 dx_deg: Tuple[int, int] = (2, 1), dy_xdeg: int = 3, dy_ydeg="auto") -> Tuple[GeometryModel, dict]:
    """Joint smile + keystone model from lamp-line and target-bar traces of one optical state.

    Args:
        lamp: Lamp lines (:func:`lamp_points_from_analysis`), at least two.
        target: Target bars (:func:`trace_target_lines`), at least three.
        shape: ``(H, W)``.
        y_ref: Reference row (dx = 0).
        x_ref: Reference column (dy = 0); ``None``: the frame centre, moved inside the bars' span.
        dx_deg: ``(deg_y, deg_x)`` of the smile (deg_y >= 1).
        dy_xdeg: Degree of the keystone along the spectrum (>= 1).
        dy_ydeg: Degree along the slit, or ``"auto"`` (0 to ``MAX_DY_YDEG`` by leave-one-bar-out:
            the lowest degree within 10 % of the best).

    Returns:
        ``(model, qc)``; ``qc`` holds fit / leave-one-line-out residuals in px.
    """
    h, w = (int(v) for v in shape)
    if len(lamp) < 2:
        raise ValueError("At least two lamp lines covering the slit are needed for the joint geometry.")
    if len(target) < 3:
        raise ValueError("At least three target bars are needed for the keystone.")
    t_lo = float(np.median([t.x.min() for t in target])); t_hi = float(np.median([t.x.max() for t in target]))
    if x_ref is None:
        x_ref = float(np.clip((w - 1) / 2.0, t_lo, t_hi))
    x_ref = float(x_ref)
    dx_deg = (max(1, int(dx_deg[0])), max(0, int(dx_deg[1])))
    dx_deg = (dx_deg[0], min(dx_deg[1], len(lamp) - 1))
    dy_xdeg = max(1, int(dy_xdeg))
    data = _Data(lamp, target, (h, w), y_ref)
    Xk0 = [data.lamp_x_at(k, y_ref) for k in range(len(lamp))]
    Ym0 = [data.bar_y_at(m, x_ref) for m in range(len(target))]
    if np.any(np.diff(Xk0) < 1) or np.any(np.diff(sorted(Ym0)) < 1):
        raise ValueError("Lamp lines or target bars are duplicated / unresolved. Remove merged lines and rebuild.")
    if dy_ydeg == "auto":
        # odd powers along the slit are lens distortion of the slit image that changes with
        # wavelength (lateral colour); a real GigE spectrograph needed y**3..y**4
        cand = [d for d in range(0, MAX_DY_YDEG + 1) if d <= max(0, len(target) - 3)]
        d_y = min(1, cand[-1])                  # start of the leave-one-bar-out choice
    else:
        cand = [max(0, min(int(dy_ydeg), len(target) - 2))]
        d_y = cand[0]
    sdx, sdy = _surfaces((h, w), y_ref, x_ref, dx_deg, (d_y, dy_xdeg))
    sol = _solve(data, sdx, sdy, x_ref)
    choice = {}
    if len(cand) > 1:
        res = {d: _loo(data, sol, lambda d=d: _surfaces((h, w), y_ref, x_ref, dx_deg, (d, dy_xdeg))[1], "y")[0]
               for d in cand}
        valid = {d: v for d, v in res.items() if v is not None}
        if valid:
            best = min(valid, key=valid.get)
            for d in sorted(valid):
                if valid[d] <= 1.1 * valid[best]:
                    best = d
                    break
            d_y = best
        choice = {str(k): v for k, v in res.items()}
    sdx, sdy = _surfaces((h, w), y_ref, x_ref, dx_deg, (d_y, dy_xdeg))
    sol = _solve(data, sdx, sdy, x_ref)
    # measured boxes (destination coordinates)
    sdx.box = [float(sol["Xk"].min()), float(sol["Xk"].max()), float(sol["Yl"].min()), float(sol["Yl"].max())]
    sdy.box = [float(np.percentile(sol["Xt"], 0.5)), float(np.percentile(sol["Xt"], 99.5)),
               float(sol["Ym"].min()), float(sol["Ym"].max())]
    loo_x, per_x = _loo(data, sol, lambda: _surfaces((h, w), y_ref, x_ref, dx_deg, (d_y, dy_xdeg))[0], "x")
    loo_y, per_y = _loo(data, sol, lambda: _surfaces((h, w), y_ref, x_ref, dx_deg, (d_y, dy_xdeg))[1], "y")
    kl, kt = sol["wl"] > 0, sol["wt"] > 0
    qc = dict(
        lamp_lines=len(lamp), target_bars=len(target),
        before_lamp_rms_px=_rms(data.lx - sol["Xl"]), before_target_rms_px=_rms(data.ty - sol["Yt"]),
        fit_lamp_rms_px=_rms(sol["rl"][kl]), fit_target_rms_px=_rms(sol["rt"][kt]),
        fit_lamp_max_px=float(np.max(np.abs(sol["rl"][kl]))) if kl.any() else float("nan"),
        fit_target_max_px=float(np.max(np.abs(sol["rt"][kt]))) if kt.any() else float("nan"),
        loo_lamp_rms_px=loo_x, loo_target_rms_px=loo_y, loo_lamp_per_line_px=per_x, loo_target_per_bar_px=per_y,
        rejected_points=int((~kl).sum() + (~kt).sum()), degrees=dict(dx=list(dx_deg), dy=[d_y, dy_xdeg]),
        dy_ydeg_loo=choice,
        fit_lamp_per_line_px=[float(v) for v in sol["lamp_sigma"]],
        fit_target_per_bar_px=[float(v) for v in sol["bar_sigma"]],
        lamp_weight=[float(v) for v in sol["lamp_weight"]], bar_weight=[float(v) for v in sol["bar_weight"]])
    model = GeometryModel(shape=(h, w), y_ref=float(y_ref), x_ref=x_ref, dx=sdx, dy=sdy,
                          lamp_X=[float(v) for v in sol["Xk"]], target_Y=[float(v) for v in sol["Ym"]])
    return model, qc


# ----------------------------------------------------------------------------------------------- checks
def row_margin_from_map(map_y: np.ndarray, interp: str = "cubic", columns: Optional[Tuple[int, int]] = None
                        ) -> Tuple[int, int]:
    """Rows at the top / bottom whose source lies outside the frame in some valid column.

    Args:
        map_y: Row map ``(H, W)``.
        interp: Interpolation name (its kernel needs ``pad`` rows inside the frame).
        columns: ``(first, last)`` columns to consider (the valid columns); ``None`` = all.

    Returns:
        ``(top, bottom)`` row counts.
    """
    _, pad_l, pad_r = _INTERP[interp]
    h = map_y.shape[0]
    sel = map_y if columns is None else map_y[:, int(columns[0]):int(columns[1]) + 1]
    hi = h - 1.0 if interp == "linear" else h - pad_r - 1e-3
    ok = ((sel >= float(pad_l)) & (sel <= hi)).all(axis=1)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        raise ValueError("No valid rows after keystone correction: the shift is too large.")
    return int(idx[0]), int(h - 1 - idx[-1])


def _remap(img, mx, my, interp="cubic"):
    return cv2.remap(np.ascontiguousarray(img, np.float32), mx, my, interpolation=_INTERP[interp][0],
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _retrace(model: GeometryModel, lamp_corr, target_norm_corr, valid, lamp_halfwins, sat_level, noise, snr_min,
             polarity, band, bar_halfwins, step):
    """Re-measure lines on the REMAPPED frames: lamp lines must be vertical at X, bars horizontal at Y."""
    h, w = model.shape
    out = dict(lamp=[], target=[])
    per = dict(lamp=[], target=[])
    lo, hi = model.dx.box[2], model.dx.box[3]
    for X, hw in zip(model.lamp_X, lamp_halfwins):
        c = trace_line(lamp_corr, X, hw, sat_level, noise, snr_min)
        rows = np.flatnonzero(np.isfinite(c))
        rows = rows[(rows >= lo) & (rows <= hi)]
        cx = int(round(X))
        rows = rows[valid[rows, min(max(cx, 0), w - 1)]]
        v = c[rows] - X
        out["lamp"].extend(v.tolist())
        per["lamp"].append(dict(X=float(X), rms_px=_rms(v), max_px=float(np.max(np.abs(v))) if v.size else float("nan"),
                                n=int(v.size)))
    sign = -1.0 if polarity == "dark" else 1.0
    img = sign * _band_columns(target_norm_corr, band)
    noise_col = _column_noise(img)
    c0, c1 = model.dy.box[0], model.dy.box[1]
    cols = np.arange(int(math.ceil(c0)), int(math.floor(c1)) + 1, max(1, step))
    for Y, hw in zip(model.target_Y, bar_halfwins):
        vals = []
        r = int(round(Y))
        hw = int(hw)
        max_jump = max(1.0, 0.5 * hw)
        for c in cols:
            if r - hw < 0 or r + hw >= h or not valid[r, c]:
                continue
            yy, ok = _centroid_row(img[:, c], Y, hw, np.inf, snr_min * float(noise_col[c]))
            if ok and abs(yy - Y) <= max_jump:       # same acceptance as the tracer: never a neighbour
                vals.append(yy - Y)
        v = np.asarray(vals)
        out["target"].extend(vals)
        per["target"].append(dict(Y=float(Y), rms_px=_rms(v), max_px=float(np.max(np.abs(v))) if v.size else float("nan"),
                                  n=int(v.size)))
    lam, tar = np.asarray(out["lamp"]), np.asarray(out["target"])
    p95 = lambda v: float(np.percentile(np.abs(v), 95)) if v.size else float("nan")
    return dict(lamp_rms_px=_rms(lam), lamp_max_px=float(np.max(np.abs(lam))) if lam.size else float("nan"),
                target_rms_px=_rms(tar), target_max_px=float(np.max(np.abs(tar))) if tar.size else float("nan"),
                lamp_p95_px=p95(lam), target_p95_px=p95(tar),
                lamp_per_line=per["lamp"], target_per_bar=per["target"])


def _summary(model: GeometryModel, map_x, map_y) -> dict:
    h, w = model.shape
    xs = np.arange(w)[None, :]; ys = np.arange(h)[:, None]
    dx = map_x - xs; dy = map_y - ys
    bx, by = model.dx.box, model.dy.box
    x0, x1 = int(math.ceil(bx[0])), int(math.floor(bx[1]))
    c0, c1 = int(math.ceil(by[0])), int(math.floor(by[1]))
    r0, r1 = int(math.ceil(by[2])), int(math.floor(by[3]))
    smile = float(np.max(np.abs(dx[:, x0:x1 + 1]))) if x1 >= x0 else 0.0
    keystone = float(np.max(np.abs(dy[r0:r1 + 1, c0:c1 + 1]))) if (r1 >= r0 and c1 >= c0) else 0.0
    mid = int(round((r0 + r1) / 2))
    span = max(1, c1 - c0)
    tilt = float(dy[mid, c1] - dy[mid, c0])
    top = float(dy[r0, c1] - dy[r0, c0]); bottom = float(dy[r1, c1] - dy[r1, c0])
    out = dict(smile_max_px=smile, keystone_max_px=keystone, tilt_px=tilt,
               tilt_deg=float(math.degrees(math.atan2(tilt, span))), keystone_top_bottom_px=float(top - bottom),
               lamp_columns=[bx[0], bx[1]], target_columns=[by[0], by[1]], target_rows=[by[2], by[3]],
               extrapolated_columns=float((max(0.0, bx[0]) + max(0.0, (w - 1) - bx[1])) / max(1, w)))
    if abs(tilt) >= 2.0 and abs(out["keystone_top_bottom_px"]) < 0.35 * abs(tilt):
        out["verdict"] = ("Mostly tilt: the slit image is rotated %.2f deg on the detector; rotating the camera by "
                          "that angle would remove most of it." % out["tilt_deg"])
    elif abs(out["keystone_top_bottom_px"]) >= 1.0:
        out["verdict"] = "Keystone: the top and bottom of the slit move differently across the spectrum."
    elif keystone >= 1.0:
        out["verdict"] = "Mostly a uniform shift / tilt of the slit across the spectrum."
    else:
        out["verdict"] = "Small keystone: under 1 px in the measured area."
    return out


# ----------------------------------------------------------------------------------------------- analysis
@dataclass
class KeystoneAnalysis:
    """Result of :func:`analyze_keystone` (the joint smile + keystone geometry).

    Attributes:
        model: The fitted :class:`GeometryModel`.
        map_x, map_y: Destination -> source maps ``float32 (H, W)``.
        margin: ``(left, right)`` columns without a full source (as the smile-only model).
        row_margin: ``(top, bottom)`` rows without a full source in the valid columns.
        target_lines: Bars used.
        qc: Fit / leave-one-out / re-trace numbers (px); see :func:`keystone_report`.
        summary: Smile / keystone / tilt at a glance.
        warnings: Human-readable problems (empty when the result looks sound).
        target_corr, lamp_corr: Remapped frames (for display), float32.
        target_norm: Normalised raw target (for display), float32.
    """

    model: GeometryModel
    map_x: np.ndarray
    map_y: np.ndarray
    margin: Tuple[int, int]
    row_margin: Tuple[int, int]
    target_lines: List[TargetLine]
    qc: dict
    summary: dict
    warnings: List[str]
    target_corr: np.ndarray
    lamp_corr: np.ndarray
    target_norm: np.ndarray
    interp: str = "cubic"
    polarity: str = "dark"
    created: str = ""
    inputs: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        """JSON-able description stored in the model file (no images)."""
        qc = copy.deepcopy(self.qc)
        return dict(model=self.model.to_json(), margin=[int(v) for v in self.margin],
                    row_margin=[int(v) for v in self.row_margin], qc=_jsonable(qc), summary=_jsonable(self.summary),
                    warnings=list(self.warnings), interp=self.interp, polarity=self.polarity, created=self.created,
                    inputs=_jsonable(self.inputs),
                    target_bars=[dict(y_seed=t.y_seed, coverage=t.coverage, n=int(t.x.size)) for t in self.target_lines])


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, np.floating):
        v = float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def analyze_keystone(analysis, target, white=None, dark=None, polarity: str = "dark", x_ref: Optional[float] = None,
                     dy_xdeg: int = 3, dy_ydeg="auto", band: int = 5, step: Optional[int] = None,
                     snr_min: Optional[float] = None, interp: Optional[str] = None, inputs: Optional[dict] = None
                     ) -> KeystoneAnalysis:
    """Joint smile + keystone geometry from a smile analysis and a line-target frame.

    Args:
        analysis: :class:`~modules.smile.analysis.SmileAnalysis` of the lamp frame (same camera setting).
        target: Line-target frame ``(H, W)``; several frames should be averaged beforehand.
        white: Optional white frame (target removed) for normalisation; strongly recommended.
        dark: Optional dark frame subtracted from target and white (not from the lamp, which the
            smile analysis already used as given).
        polarity: ``"dark"`` or ``"bright"`` bars.
        x_ref: Reference column (dy = 0); ``None``: frame centre within the bars' span.
        dy_xdeg, dy_ydeg: Keystone degrees (see :func:`fit_geometry`).
        band, step: Column averaging / step of the bar tracing.
        snr_min: Bar / line detection threshold; ``None``: the smile analysis value (5).
        interp: Remap interpolation for the check images; ``None``: the smile analysis's.
        inputs: Free description of the input files (stored in the model).

    Returns:
        :class:`KeystoneAnalysis`.

    Raises:
        ValueError: Unusable frames (sizes differ, too few lines / bars, coverage too small).
    """
    lamp = analysis.frame_orig
    h, w = lamp.shape
    t = np.asarray(target, np.float32)
    if t.shape != (h, w):
        raise ValueError(f"The target frame {t.shape} differs from the lamp frame {(h, w)}: the same camera "
                         "setting (orientation, bin, ROI) is needed.")
    interp = interp or analysis.interp
    snr = 5.0 if snr_min is None else float(snr_min)
    notes: List[str] = []
    lamp_lines, lamp_rej = lamp_points_from_analysis(analysis)
    for col, why in lamp_rej:
        notes.append("lamp line at column %.0f not used for the joint fit: %s" % (col, why))
    norm = normalize_target(t, white, dark)
    bars, tinfo = trace_target_lines(t, polarity=polarity, band=band, step=step, snr_min=snr,
                                     normalized=norm, sat_level=None)
    for y, why in tinfo["rejected"]:
        notes.append("target bar near row %.0f not used: %s" % (y, why))
    sm = analysis.surface
    model, qc = fit_geometry(lamp_lines, bars, (h, w), analysis.y_ref, x_ref=x_ref,
                             dx_deg=(max(1, sm.deg_y), sm.deg_x), dy_xdeg=dy_xdeg, dy_ydeg=dy_ydeg)
    mx, my = model.maps()
    from .calibration import margin_from_map
    margin = margin_from_map(mx, interp)
    row_margin = row_margin_from_map(my, interp, (margin[0], w - 1 - margin[1]))
    lamp_corr = _remap(lamp, mx, my, interp)
    target_corr = _remap(norm, mx, my, interp)
    valid = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    by_col = {float(ln.column): ln for ln in analysis.used_lines}
    halfwins = []
    for L in lamp_lines:
        ln = by_col.get(L.column)
        halfwins.append(max(3, int(round(1.2 * ln.fwhm))) if ln is not None else 5)
    rt = _retrace(model, lamp_corr, target_corr, valid, halfwins, analysis.sat_level, analysis.noise, snr,
                  polarity, tinfo["band"], [t.halfwin or tinfo["halfwin"] for t in bars], tinfo["step"])
    qc.update(retrace_lamp_rms_px=rt["lamp_rms_px"], retrace_lamp_max_px=rt["lamp_max_px"],
              retrace_target_rms_px=rt["target_rms_px"], retrace_target_max_px=rt["target_max_px"],
              retrace_lamp_p95_px=rt["lamp_p95_px"], retrace_target_p95_px=rt["target_p95_px"],
              retrace_lamp_per_line=rt["lamp_per_line"], retrace_target_per_bar=rt["target_per_bar"],
              bar_period_px=tinfo["period"], bar_halfwin_px=tinfo["halfwin"], seed_column=tinfo["seed_column"],
              normalised_with_white=white is not None, notes=notes, merged_bar_pairs=tinfo.get("merged", []))
    # the median line / bar: robust to a few features whose own shape biases their centroid
    med = lambda per: (float(np.nanmedian([p["rms_px"] for p in per])) if per else float("nan"))
    qc["retrace_lamp_median_line_px"] = med(rt["lamp_per_line"])
    qc["retrace_target_median_bar_px"] = med(rt["target_per_bar"])
    for label, where_name, per in (("target bars", "rows", rt["target_per_bar"]),
                                   ("lamp lines", "columns", rt["lamp_per_line"])):
        weak = [p for p in per if np.isfinite(p["rms_px"]) and p["rms_px"] > 2 * RETRACE_WARN_PX]
        if weak:
            where = ", ".join("%.0f" % (p.get("Y", p.get("X"))) for p in weak[:12])
            notes.append("%d %s scatter over %.2f px RMS after correction (asymmetric profile or weak signal; "
                         "the fit weights them down), %s %s" % (len(weak), label, 2 * RETRACE_WARN_PX,
                                                               where_name, where))
    summ = _summary(model, mx, my)
    warn: List[str] = []
    for key, label in (("retrace_lamp_median_line_px", "lamp lines"), ("retrace_target_median_bar_px", "target bars")):
        v = qc.get(key)
        if v is None or not np.isfinite(v):
            warn.append("The %s could not be re-measured on the corrected frame." % label)
        elif v > RETRACE_WARN_PX:
            warn.append("The median of the %s is %.2f px RMS from straight after correction (limit %.2f px)."
                        % (label, v, RETRACE_WARN_PX))
    if white is None:
        warn.append("No white frame: the illumination was flattened with a running mean along the slit only.")
    if summ["extrapolated_columns"] > 0.25:
        warn.append("%.0f %% of the columns lie outside the lamp lines (the smile there is held at the edge "
                    "value)." % (100 * summ["extrapolated_columns"]))
    return KeystoneAnalysis(model=model, map_x=mx, map_y=my, margin=margin, row_margin=row_margin,
                            target_lines=bars, qc=qc, summary=summ, warnings=warn, target_corr=target_corr,
                            lamp_corr=lamp_corr, target_norm=norm, interp=interp, polarity=polarity,
                            created=time.strftime("%Y-%m-%dT%H:%M:%S"), inputs=dict(inputs or {}))


def keystone_report(k) -> str:
    """Human-readable report of a :class:`KeystoneAnalysis` (or its :meth:`~KeystoneAnalysis.to_json`)."""
    d = k.to_json() if hasattr(k, "to_json") else k      # duck-typed: also under ``python -m``
    q, s = d["qc"], d["summary"]
    f = lambda v: "-" if v is None else "%.3f px" % v
    lines = [
        "Smile + keystone geometry (%s)" % d.get("created", ""),
        "Lamp lines %d, target bars %d, degrees dx %s dy %s" % (q["lamp_lines"], q["target_bars"],
                                                               q["degrees"]["dx"], q["degrees"]["dy"]),
        "Before: lamp %s, target %s" % (f(q["before_lamp_rms_px"]), f(q["before_target_rms_px"])),
        "Fit RMS: lamp %s, target %s" % (f(q["fit_lamp_rms_px"]), f(q["fit_target_rms_px"])),
        "Leave-one-line-out: lamp %s, target %s" % (f(q["loo_lamp_rms_px"]), f(q["loo_target_rms_px"])),
        "Re-traced after remap (median line / bar): lamp %s, target %s" % (
            f(q.get("retrace_lamp_median_line_px")), f(q.get("retrace_target_median_bar_px"))),
        "Re-traced after remap (all points): lamp %s (p95 %s, max %s), target %s (p95 %s, max %s)" % (
            f(q["retrace_lamp_rms_px"]), f(q.get("retrace_lamp_p95_px")), f(q["retrace_lamp_max_px"]),
            f(q["retrace_target_rms_px"]), f(q.get("retrace_target_p95_px")), f(q["retrace_target_max_px"])),
        "Smile max %.2f px, keystone max %.2f px, tilt %.3f deg - %s" % (
            s["smile_max_px"], s["keystone_max_px"], s["tilt_deg"], s.get("verdict", "")),
        "Border: columns %s, rows %s" % (tuple(d["margin"]), tuple(d["row_margin"])),
    ]
    lines += ["Warning: " + w for w in d.get("warnings", [])]
    lines += q.get("notes", []) or []
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- CLI
def _read(paths: Sequence[str]) -> np.ndarray:
    from .core import load_frame
    frames = [load_frame(p) for p in paths]
    if any(f.shape != frames[0].shape for f in frames):
        raise SystemExit("All frames of one kind must have the same size.")
    return np.mean(np.stack(frames), axis=0).astype(np.float32) if len(frames) > 1 else frames[0]


def main(argv=None) -> int:
    """``python -m modules.smile.keystone --lamp L.tif --target T1.tif T2.tif [--white W.tif] [--out model.npz]``"""
    import argparse
    from .corrector import SmileCorrector
    ap = argparse.ArgumentParser(description="Smile + keystone model from a lamp frame and a line-target frame.")
    ap.add_argument("--lamp", nargs="+", required=True)
    ap.add_argument("--target", nargs="+", required=True)
    ap.add_argument("--white", nargs="+")
    ap.add_argument("--dark", nargs="+")
    ap.add_argument("--polarity", choices=("dark", "bright"), default="dark")
    ap.add_argument("--deg-y", type=int, default=2); ap.add_argument("--deg-x", type=int, default=1)
    ap.add_argument("--dy-xdeg", type=int, default=3)
    ap.add_argument("--interp", default="cubic", choices=sorted(_INTERP))
    ap.add_argument("--out", help="model .npz to write (no camera setting is stored)")
    a = ap.parse_args(argv)
    lamp = _read(a.lamp); target = _read(a.target)
    white = _read(a.white) if a.white else None
    dark = _read(a.dark) if a.dark else None
    if dark is not None:
        lamp = lamp - dark
    c = SmileCorrector(deg_y=a.deg_y, deg_x=a.deg_x, interp=a.interp).fit(lamp)
    k = c.fit_keystone(target, white=white, dark=dark, polarity=a.polarity, dy_xdeg=a.dy_xdeg,
                       inputs=dict(lamp=a.lamp, target=a.target, white=a.white, dark=a.dark))
    print(keystone_report(k))
    if a.out:
        c.save(a.out)
        print("Written %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
