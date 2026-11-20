"""Where along the slit and the spectrum a line-target frame is in focus (Qt-free).

A pushbroom spectrograph whose image surface is curved (field curvature) and whose focus
moves with wavelength (axial colour, or a detector tilted about the slit axis) is sharp on
a curved band of the ``(slit, wavelength)`` frame: on a frame of a bar target the bars are
crisp along that band and washed out elsewhere. :func:`focus_map` measures the local bar
contrast everywhere, finds the sharpest column of every slit row, fits a parabola through
them and condenses everything into numbers a person can focus by:

* ``worst`` - the bar contrast of the least sharp part of the lit frame (5th percentile).
  Turn the focus until this is highest: every part of the frame is then as sharp as the
  optics allow at once.
* ``centre_col`` / ``edge_col`` - the column of best focus at the slit centre and at the
  slit ends (with a wavelength calibration: the wavelengths).

Data conventions:
    ``image`` is a 2-D ``(slit rows, spectral columns)`` frame of a target whose bars run
    along the spectral axis (horizontal bars), any numeric dtype, dark-subtracted or not.
    Columns and rows are sensor pixels at the frame's own binning. Contrast is unit-less
    (bar modulation over the local level).

Threading:
    Pure functions; safe from any thread. One 1024 x 1280 frame takes about 50 ms.

Example:
    >>> import numpy as np
    >>> rows = np.arange(256)[:, None]
    >>> frame = 1000 + 400 * np.cos(2 * np.pi * rows / 12.0) * np.ones((1, 300))
    >>> fm = focus_map(frame)
    >>> round(fm.period, 1)
    12.0
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter

#: Columns averaged into one before filtering (the bars run along the columns, so this keeps them).
COLUMN_BLOCK = 4
#: Bar period limits (px) searched along the slit.
PERIOD_RANGE = (4.0, 120.0)
#: A pixel is lit when its smoothed level exceeds this share of the bright (99th percentile) level.
LIT_FRACTION = 0.15
#: The bar line in the slit-profile spectrum must exceed the median spectrum by this factor.
PEAK_RATIO = 6.0
#: Share of the slit rows that must show a clear best column before a best-focus curve is fitted.
MIN_ROW_SHARE = 0.3
#: Percentile of the lit contrast reported as ``worst``.
WORST_PERCENTILE = 5.0


@dataclass(frozen=True)
class FocusMap:
    """Result of :func:`focus_map` (immutable).

    Attributes:
        contrast: ``(rows, cols)`` float32 bar contrast on a reduced grid; NaN where unlit.
        rows: Sensor row of each ``contrast`` row (px).
        cols: Sensor column of each ``contrast`` column (px).
        period: Bar period along the slit (px).
        row_samples: Rows at which the best-focus column was searched (px).
        best_col: Column of best focus at each of ``row_samples`` (px; NaN: no lit pixels).
        best_contrast: Contrast at ``best_col``.
        fit: Parabola ``col = fit[0] + fit[1] * r + fit[2] * r**2`` with ``r = (row - centre_row) / half_height``,
            or ``None`` when too few rows had a clear best column.
        centre_row: Slit centre used by ``fit`` (middle of the lit rows, px).
        lit_rows: ``(first, last)`` lit row (px).
        lit_cols: ``(first, last)`` lit column (px).
        worst: 5th percentile of the lit contrast (the least sharp part of the frame).
        best: 95th percentile of the lit contrast.
        frame_shape: ``(height, width)`` of the input frame.
    """
    contrast: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    period: float
    row_samples: np.ndarray
    best_col: np.ndarray
    best_contrast: np.ndarray
    fit: Optional[np.ndarray]
    centre_row: float
    lit_rows: tuple
    lit_cols: tuple
    worst: float
    best: float
    frame_shape: tuple

    @property
    def balance(self) -> float:
        """``worst / best`` (0..1): 1 means the whole frame is equally sharp."""
        return float(self.worst / self.best) if self.best > 0 else 0.0

    def fitted_col(self, row) -> np.ndarray:
        """Best-focus column of the parabola at ``row`` (px; NaN without a fit)."""
        row = np.asarray(row, float)
        if self.fit is None:
            return np.full(row.shape, np.nan)
        half = max(1.0, 0.5 * (self.lit_rows[1] - self.lit_rows[0]))
        r = (row - self.centre_row) / half
        return self.fit[0] + self.fit[1] * r + self.fit[2] * r * r

    @property
    def centre_col(self) -> float:
        """Best-focus column at the slit centre (px; NaN without a fit)."""
        return float(self.fitted_col(self.centre_row))

    @property
    def edge_col(self) -> float:
        """Mean best-focus column of the two slit ends (lit rows; px; NaN without a fit)."""
        return float(np.mean(self.fitted_col(np.array(self.lit_rows, float))))

    @property
    def tilt_cols(self) -> float:
        """Best-focus column at the bottom slit end minus the top one (px): non-zero when the
        detector is tilted about the spectral axis (one slit end focuses at longer wavelengths)."""
        top, bottom = self.fitted_col(np.array(self.lit_rows, float))
        return float(bottom - top)


def bar_period(image: np.ndarray) -> float:
    """Bar period along the slit (px) from the spectrum of the central columns' slit profile.

    Raises:
        ValueError: No periodic pattern between :data:`PERIOD_RANGE` was found.
    """
    img = np.asarray(image, np.float32)
    h, w = img.shape
    prof = img[:, int(0.3 * w):max(int(0.3 * w) + 1, int(0.7 * w))].mean(axis=1)
    prof = prof - gaussian_filter1d(prof, 0.1 * h)
    prof = prof * np.hanning(h)
    n = 1 << int(np.ceil(np.log2(4 * h)))                 # zero-pad for a finer frequency grid
    spec = np.abs(np.fft.rfft(prof, n))
    freq = np.fft.rfftfreq(n)
    ok = (freq >= 1.0 / PERIOD_RANGE[1]) & (freq <= 1.0 / PERIOD_RANGE[0])
    if not ok.any() or spec[ok].max() <= 0:
        raise ValueError("No bar pattern found along the slit")
    idx = np.flatnonzero(ok)
    k = int(idx[np.argmax(spec[ok])])
    # a bar target gives one strong spectral line; noise or a scene gives a broad, low spectrum
    if spec[k] < PEAK_RATIO * float(np.median(spec[ok])) or k in (idx[0], idx[-1]):
        raise ValueError("No bar pattern found along the slit (is the bar target in view?)")
    if 0 < k < len(spec) - 1:                              # parabolic peak refinement
        y0, y1, y2 = np.log(spec[k - 1:k + 2] + 1e-12)
        k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    return float(1.0 / (k / n))


def focus_map(image: np.ndarray, period: Optional[float] = None) -> FocusMap:
    """Measure the bar contrast of a line-target frame and where each slit row is sharpest.

    Args:
        image: ``(slit, wavelength)`` frame of a target with horizontal bars.
        period: Bar period along the slit (px); ``None``: :func:`bar_period`.

    Returns:
        :class:`FocusMap`.

    Raises:
        ValueError: Not a 2-D frame of at least 32 x 32 pixels, no bar pattern, or nothing lit.
    """
    img = np.asarray(image, np.float32)
    if img.ndim != 2 or min(img.shape) < 32:
        raise ValueError("Use a 2-D frame of at least 32 x 32 pixels")
    h, w = img.shape
    img = np.nan_to_num(img)
    if period is None:
        period = bar_period(img)
    period = float(np.clip(period, *PERIOD_RANGE))

    # columns: block-average (the bars run along them); rows: keep full resolution
    nb = max(1, w // COLUMN_BLOCK)
    red = img[:, :nb * COLUMN_BLOCK].reshape(h, nb, COLUMN_BLOCK).mean(axis=2)
    cols = (np.arange(nb) + 0.5) * COLUMN_BLOCK - 0.5
    col_smooth = max(1.0, 0.004 * w / COLUMN_BLOCK)
    level = gaussian_filter(red, (period, col_smooth))
    fine = gaussian_filter(red, (max(0.6, period / 12.0), col_smooth))
    ripple = fine - level
    win_r = max(3, int(round(2 * period)))
    win_c = max(3, int(round(0.02 * w / COLUMN_BLOCK)))
    amp = np.sqrt(np.maximum(uniform_filter(ripple * ripple, (win_r, win_c)), 0.0)) * np.sqrt(2.0)
    bright = float(np.percentile(level, 99))
    if bright <= 0:
        raise ValueError("The frame is dark")
    lit = level > LIT_FRACTION * bright
    # keep away from the lit border, where the level gradient fakes contrast
    lit &= uniform_filter(lit.astype(np.float32), (win_r, win_c)) > 0.999
    if lit.sum() < 50:
        raise ValueError("Too few lit pixels")
    contrast = np.where(lit, amp / np.maximum(level, 1e-6), np.nan).astype(np.float32)

    lit_r = np.flatnonzero(lit.any(axis=1))
    lit_c = np.flatnonzero(lit.any(axis=0))
    lit_rows = (int(lit_r[0]), int(lit_r[-1]))
    lit_cols = (int(round(cols[lit_c[0]])), int(round(cols[lit_c[-1]])))

    # best column per row band
    band = max(4, int(round(period)))
    starts = np.arange(lit_rows[0], lit_rows[1] - band + 2, band)
    row_samples = starts + 0.5 * (band - 1)
    best_col = np.full(len(starts), np.nan)
    best_val = np.full(len(starts), np.nan)
    for i, s in enumerate(starts):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prof = np.nanmean(np.where(lit[s:s + band], contrast[s:s + band], np.nan), axis=0)
        good = np.isfinite(prof)
        if good.sum() < 5:
            continue
        sm = gaussian_filter1d(np.where(good, prof, 0.0), 3.0) / np.maximum(
            gaussian_filter1d(good.astype(float), 3.0), 1e-6)
        sm[~good] = -np.inf
        k = int(np.argmax(sm))
        # a clear peak: the best column must stand out from the row's median
        if sm[k] < 1.15 * np.median(sm[good]):
            continue
        if 0 < k < len(sm) - 1 and np.isfinite(sm[k - 1]) and np.isfinite(sm[k + 1]):
            y0, y1, y2 = sm[k - 1], sm[k], sm[k + 1]
            k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 - 1e-12)
        best_col[i] = float(np.interp(k, np.arange(nb), cols))
        best_val[i] = float(sm[int(round(k))])

    centre_row = 0.5 * (lit_rows[0] + lit_rows[1])
    half = max(1.0, 0.5 * (lit_rows[1] - lit_rows[0]))
    fit = None
    ok = np.isfinite(best_col)
    if ok.sum() >= max(5, MIN_ROW_SHARE * len(ok)):
        r = (row_samples[ok] - centre_row) / half
        x, wgt = best_col[ok], best_val[ok]
        keep = np.ones(len(r), bool)
        for _ in range(3):                                 # robust: drop rows far off the parabola
            coef = np.polynomial.polynomial.polyfit(r[keep], x[keep], 2, w=wgt[keep])
            res = x - np.polynomial.polynomial.polyval(r, coef)
            sig = 1.4826 * np.median(np.abs(res[keep])) + 0.5 * COLUMN_BLOCK
            new = np.abs(res) < 4 * sig
            if new.sum() < 5 or (new == keep).all():
                break
            keep = new
        fit = np.asarray(coef, float)
        # a curve whose centre lies far outside the lit spectrum is not a focus band but a bad fit
        span = max(1.0, lit_cols[1] - lit_cols[0])
        if not (lit_cols[0] - 0.25 * span <= fit[0] <= lit_cols[1] + 0.25 * span):
            fit = None

    vals = contrast[lit]
    return FocusMap(contrast=contrast, rows=np.arange(h, dtype=float), cols=cols, period=period,
                    row_samples=row_samples, best_col=best_col, best_contrast=best_val, fit=fit,
                    centre_row=float(centre_row), lit_rows=lit_rows, lit_cols=lit_cols,
                    worst=float(np.percentile(vals, WORST_PERCENTILE)),
                    best=float(np.percentile(vals, 100 - WORST_PERCENTILE)), frame_shape=(h, w))


def linear_wavelength(slope_nm_per_px: float, offset_nm: float) -> Callable[[np.ndarray], np.ndarray]:
    """``column -> nm`` for a linear calibration ``nm = slope * column + offset`` (columns at the frame's bin)."""
    def to_nm(col):
        return slope_nm_per_px * np.asarray(col, float) + offset_nm
    return to_nm


def focus_advice(fm: FocusMap, to_nm: Optional[Callable] = None) -> str:
    """One-line reading of where the focus band sits, for the person turning the focus.

    The best-focus band reaches from the slit centre (``centre_col``) to the slit ends
    (``edge_col``); the most even focus is when the middle of that band sits on the middle
    of the lit spectrum. Field curvature itself (the band's length) cannot be focused away.
    """
    if fm.fit is None:
        return "No clear best-focus band: the bars are equally blurred everywhere (far out of focus) or missing."
    mid_band = 0.5 * (fm.centre_col + fm.edge_col)
    mid_lit = 0.5 * (fm.lit_cols[0] + fm.lit_cols[1])
    off = mid_band - mid_lit
    span = abs(fm.edge_col - fm.centre_col)
    unit = (lambda c: "%.0f nm" % float(to_nm(c))) if to_nm else (lambda c: "col %.0f" % c)
    d_unit = (lambda d: "%+.0f nm" % (float(to_nm(mid_lit + d)) - float(to_nm(mid_lit)))) if to_nm \
        else (lambda d: "%+.0f px" % d)
    where = ("toward shorter wavelengths" if off > 0 else "toward longer wavelengths")
    text = ("Sharpest: slit centre at %s, slit ends at %s. Band middle is %s from the spectrum middle"
            % (unit(fm.centre_col), unit(fm.edge_col), d_unit(off)))
    if abs(off) > 0.05 * (fm.lit_cols[1] - fm.lit_cols[0]):
        text += " - move the focus %s." % where
    else:
        text += " - focus is balanced."
    if span > 0.5 * (fm.lit_cols[1] - fm.lit_cols[0]):
        text += " Field curvature is large (band covers %.0f %% of the spectrum): stop down to widen it." % (
            100 * span / max(1, fm.lit_cols[1] - fm.lit_cols[0]))
    return text
