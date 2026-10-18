"""FWHM / spectral resolution of calibration-lamp lines on a smile-corrected spectral image.

Method (per spectrum = one block of slit rows averaged *after* smile correction):

0. Optionally a smooth **baseline** is estimated on the usable (non smile-border) columns
   with one of :data:`~modules.spectral_resolution.baseline.BASELINE_METHODS` (ModPoly,
   IModPoly, ZhangFit/airPLS, AsLS, arPLS; ``baseline="none"`` by default) and subtracted.
   The fit below keeps its own linear background on top of it; saturation is always judged
   on the raw counts.
1. Every line starts at its calibrated pixel; the centre is refined to the highest *local
   maximum* within ±2 px (never past the midpoint to another listed line, so a weak line
   next to a strong one does not start on the strong line's flank; a guess up to 4 px off
   an isolated line still climbs to its peak) and a first width is
   taken from the half-maximum crossings (around this line's own peak when the strongest
   sample of the ±12 px segment is a neighbour's flank).
2. Each line gets a fit window of ``max(4, window_fwhm * FWHM)`` pixels per side (default
   3 × FWHM for a Gaussian, 5 × for a Lorentzian, whose wings still hold 10 % of the area
   beyond ±3 FWHM). Lines whose windows overlap - and any other peak found inside the
   window - are fitted **together**, so a neighbouring line can no longer narrow or widen
   the measured one. Lorentzian wings reach far beyond a window (they fall off only as
   1/d²): when the lines fitted in the *other* windows bend a window's background by more
   than a tenth of the noise, that window is fitted again with them subtracted.
3. The model is a sum of **pixel-integrated** line profiles of one shape (``profile``, see
   :data:`PROFILES`) on a linear background::

       counts(x) = b0 + b1·(x - x0) + Σ_k P(x; A_k, μ_k, w_k)
       gaussian:    P = A·[Φ((x+½-μ)/σ) - Φ((x-½-μ)/σ)]               FWHM = 2.3548·σ
       lorentzian:  P = (A/π)·[atan((x+½-μ)/γ) - atan((x-½-μ)/γ)]      FWHM = 2·γ

   A detector pixel integrates the light over its width; fitting point samples instead
   overestimates narrow lines (≈ +10 % at FWHM 1.5 px). The FWHM is the width of the line
   profile itself, ``A`` its area and the reported amplitude its peak height
   (``A/(σ√(2π))`` or ``A/(πγ)``). Pick the shape of the instrument's line profile: the
   wrong one biases the width (for 3-8 px lines a Lorentzian fitted to a Gaussian line
   comes out ≈ 15-20 % narrower, a Gaussian fitted to a Lorentzian line ≈ 9-14 % wider).
4. Lines with clipped (saturated) samples, low signal-to-noise or an unconverged fit (also:
   width or centre ended on a bound, the centre may move ±2 px) are flagged and left out of
   the statistics instead of reporting a wrong width; so is a line
   whose window does not reach the point where its profile has fallen to ≈ 4 % of the peak
   (``outside``: 2.5 σ = 1.06 FWHM for a Gaussian, 4.66 γ = 2.33 FWHM for a Lorentzian).
5. FWHM in nm = FWHM in px × dispersion ``m`` at the image's hardware bin; resolving power
   ``R = λ / Δλ``. Each line is summarised over all row blocks (median and 16-84 % spread).

Data conventions:
    ``frame``: 2-D ``(slit rows, spectral columns)``, oriented, dark-corrected and smile
    corrected (zero-filled invalid border columns are ignored; the border's width differs
    from row to row, so each block uses only the columns holding data in all of its rows,
    see :func:`usable_columns`). Pixels are spectral-column
    indices at the frame's hardware bin; wavelengths in nm; intensities in counts.

Threading:
    Pure functions; call from any thread. ``progress`` callbacks run in the calling thread.

Example:
    >>> import numpy as np
    >>> from modules.spectral_resolution import measure_spectrum
    >>> x = np.arange(300.0)
    >>> s = 100 + 2000 * np.exp(-0.5 * ((x - 120.3) / 1.2) ** 2)
    >>> w = measure_spectrum(s, [120], m=0.5, wavelengths=[460.0])[0]
    >>> w.flag, round(w.fwhm_px, 1), round(w.fwhm_nm, 2)   # point samples: narrower than 2.83
    ('ok', 2.7, 1.37)
    >>> g = 2.0                                      # Lorentzian half width (px)
    >>> s = 100 + 2000 * g * (np.arctan((x + 0.5 - 120.3) / g) - np.arctan((x - 0.5 - 120.3) / g))
    >>> w = measure_spectrum(s, [120], profile="lorentzian")[0]
    >>> w.flag, round(w.fwhm_px, 3), round(w.amplitude)
    ('ok', 4.0, 2000)
    >>> curved = s + 400 + 3e-2 * (x - 150) ** 2            # strong curved background
    >>> w = measure_spectrum(curved, [120], profile="lorentzian", baseline="arpls")[0]
    >>> w.flag, round(w.fwhm_px, 1)
    ('ok', 4.0)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .baseline import _prepare as _baseline_options
from .baseline import estimate_baseline

logger = logging.getLogger(__name__)

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))   # 2.3548...

#: Line profiles :func:`measure_spectrum` / :func:`measure_frame` can fit (``profile=``).
PROFILES = ("gaussian", "lorentzian")

#: Flags a measurement can carry; only ``"ok"`` and ``"undersampled"`` enter the statistics.
FLAGS = ("ok", "undersampled", "saturated", "low_snr", "fit_failed", "outside", "border")
VALID_FLAGS = ("ok", "undersampled")

# Lorentzian: a window is fitted again without the lines of the other windows when their wings
# bend its background by more than this many noise sigmas (beyond the linear background).
_WING_REFIT_NOISE = 0.1


class MeasurementCancelled(Exception):
    """Raised by :func:`fit_spectrum` / :func:`measure_spectrum` / :func:`measure_frame` when
    their ``cancel()`` callable returned True (checked between blocks, between fits and on
    every model evaluation, so even a long joint fit stops promptly)."""


def _check_cancel(cancel):
    """Raise :class:`MeasurementCancelled` when ``cancel`` is given and returns True."""
    if cancel is not None and cancel():
        raise MeasurementCancelled("FWHM measurement cancelled")


@dataclass(frozen=True)
class LineMeasurement:
    """Width of one line in one spectrum.

    Attributes:
        pixel: Starting position (px, as given).
        center_px: Fitted centre (px, sub-pixel), ``nan`` when not measured.
        fwhm_px: FWHM of the fitted pixel-integrated profile (px), ``nan`` unless valid.
        fwhm_px_halfmax: Model-free half-maximum width of the background- and
            neighbour-subtracted data (px), a cross-check; ``nan`` if not measurable.
        amplitude: Peak height of the fitted continuous profile above the background
            (counts; ``A/(σ√(2π))`` or ``A/(πγ)``).
        snr: ``amplitude / noise``.
        n_components: Lines/peaks fitted together with this one (1 = isolated).
        flag: One of :data:`FLAGS`.
        wavelength_nm: Line wavelength (given, else ``m·centre + c``), or ``nan``.
        fwhm_nm: ``fwhm_px·|m|`` (nm), ``nan`` without ``m`` or when not valid.
        resolving_power: ``λ / Δλ``, or ``nan``.
        profile: Line model of the fit, one of :data:`PROFILES`.
    """

    pixel: float
    center_px: float
    fwhm_px: float
    fwhm_px_halfmax: float
    amplitude: float
    snr: float
    n_components: int
    flag: str
    wavelength_nm: float = float("nan")
    fwhm_nm: float = float("nan")
    resolving_power: float = float("nan")
    profile: str = "gaussian"

    @property
    def valid(self) -> bool:
        """True when the width enters the statistics (flag ``ok`` or ``undersampled``)."""
        return self.flag in VALID_FLAGS


def _noise(values) -> float:
    """Robust per-sample noise from first differences (MAD / 0.6745 / √2)."""
    d = np.diff(np.asarray(values, dtype=float))
    if d.size < 3:
        return 0.0
    return float(np.median(np.abs(d - np.median(d))) / 0.67449 / np.sqrt(2.0))


def _halfmax_width(x, y):
    """Half-maximum width (px) of ``y`` around its maximum by linear interpolation, or ``nan``."""
    if len(y) < 3:
        return float("nan")
    peak = int(np.argmax(y))
    half = y[peak] / 2.0
    if half <= 0:
        return float("nan")
    left = peak
    while left > 0 and y[left] > half:
        left -= 1
    right = peak
    while right < len(y) - 1 and y[right] > half:
        right += 1
    if y[left] > half or y[right] > half:
        return float("nan")
    xl = x[left] + (half - y[left]) / (y[left + 1] - y[left]) * (x[left + 1] - x[left])
    xr = x[right - 1] + (half - y[right - 1]) / (y[right] - y[right - 1]) * (x[right] - x[right - 1])
    return float(xr - xl)


class _Gaussian:
    """Pixel-integrated Gaussian ``A·[Φ((x+½-μ)/σ) - Φ((x-½-μ)/σ)]``; width parameter σ (px).

    ``A`` is the line area (counts·px). ``core`` and ``extent`` are distances in units of the
    width parameter: within ``core`` the profile is above ≈49 % of its peak (where clipped
    samples spoil the fit); at ``extent`` it has fallen to e^-3.125 ≈ 4.4 % of its peak (the
    window must reach that far on both sides, else the line is flagged ``outside``).
    """

    name = "gaussian"
    label = "Gaussian"
    fwhm_per_width = FWHM_PER_SIGMA
    window_fwhm = 3.0           # ±3 FWHM = ±7 σ: the whole line
    heavy_wings = False         # nothing of a line reaches beyond its window
    core = 1.2
    extent = 2.5                # 1.06 FWHM

    @staticmethod
    def counts(x, area, mu, width):
        """Counts in the pixels centred at ``x`` of a line of ``area`` centred at ``mu``."""
        from scipy.special import ndtr
        return area * (ndtr((x + 0.5 - mu) / width) - ndtr((x - 0.5 - mu) / width))

    @staticmethod
    def area(height, width):
        """Area of a line whose continuous profile peaks at ``height``."""
        return height * width * np.sqrt(2 * np.pi)

    @staticmethod
    def height(area, width):
        """Continuous peak height of a line of ``area``."""
        return area / (width * np.sqrt(2 * np.pi))


class _Lorentzian:
    """Pixel-integrated Lorentzian ``(A/π)·[atan((x+½-μ)/γ) - atan((x-½-μ)/γ)]``; width γ = HWHM (px).

    ``core`` and ``extent`` mark the same fractions of the peak as for the Gaussian
    (1/(1+(d/γ)²) = e^(-c²/2)): 1.03 γ and 4.66 γ = 2.33 FWHM. The wings are heavy (10 % of
    the area lies beyond ±3 FWHM), so the default window is wider than the Gaussian's: at
    ±5 FWHM the wing is down to 1 % of the peak. On synthetic lamp spectra this gave a
    smaller width scatter than ±3 FWHM and kept wide lines (whose first width estimate is
    low) from being flagged ``outside``; joint fits grow, so dense spectra take longer.
    """

    name = "lorentzian"
    label = "Lorentzian"
    fwhm_per_width = 2.0
    window_fwhm = 5.0
    heavy_wings = True          # lines in other windows still reach into a window: refit without them
    core = float(np.sqrt(np.expm1(0.5 * _Gaussian.core ** 2)))
    extent = float(np.sqrt(np.expm1(0.5 * _Gaussian.extent ** 2)))

    @staticmethod
    def counts(x, area, mu, width):
        """Counts in the pixels centred at ``x`` of a line of ``area`` centred at ``mu``.

        Uses ``atan(a) - atan(b) = atan2(a - b, 1 + a·b)`` (exact for ``a > b``), scaled by γ²,
        which keeps full precision in the far wings where both arctangents approach ±π/2.
        """
        return area / np.pi * np.arctan2(width, (x - mu) ** 2 + width ** 2 - 0.25)

    @staticmethod
    def area(height, width):
        """Area of a line whose continuous profile peaks at ``height``."""
        return height * width * np.pi

    @staticmethod
    def height(area, width):
        """Continuous peak height of a line of ``area``."""
        return area / (width * np.pi)


_PROFILES = {"gaussian": _Gaussian, "lorentzian": _Lorentzian}

#: Default ``window_fwhm`` of each profile (used when ``window_fwhm=None``).
DEFAULT_WINDOW_FWHM = {name: model.window_fwhm for name, model in _PROFILES.items()}


def _line_profile(profile):
    """The line model for a :data:`PROFILES` name (case-insensitive); ``ValueError`` otherwise."""
    key = profile.lower() if isinstance(profile, str) else None
    if key not in _PROFILES:
        raise ValueError(f"profile must be one of {', '.join(PROFILES)}, not {profile!r}")
    return _PROFILES[key]


def _components(x, params, n, model=_Gaussian):
    """Per-component pixel-integrated line counts of ``model``, shape ``(n, len(x))``.

    ``params`` = ``[b0, b1, A_0, μ_0, w_0, A_1, ...]`` (``w`` = σ or γ).
    """
    out = np.empty((n, x.size))
    for k in range(n):
        area, mu, width = params[2 + 3 * k: 5 + 3 * k]
        out[k] = model.counts(x, area, mu, width)
    return out


def _is_peak(y, j) -> bool:
    """True when sample ``j`` is not below its neighbours (a missing neighbour does not count)."""
    return (j == 0 or y[j] >= y[j - 1]) and (j == len(y) - 1 or y[j] >= y[j + 1])


def _halfmax_width_at(x, y, i):
    """Half-maximum width (px) of the line whose peak is sample ``i`` of ``y`` (``y`` above its floor).

    Unlike :func:`_halfmax_width` it walks from ``i`` (not from the global maximum) and stops
    on a side where the data rise again by more than 10 % of the peak before falling to half
    (a valley above half: the flank of a neighbouring line). With one side found the width is
    twice that side's distance from ``x[i]``. ``nan`` when neither side is found.
    """
    half = y[i] / 2.0
    if not half > 0:
        return float("nan")
    rise = 0.1 * y[i]
    sides = []
    for step in (-1, 1):
        j, low = i, y[i]
        while 0 <= j + step < len(y) and y[j] > half:
            if y[j + step] > low + rise:                 # valley above half: a neighbour's flank
                break
            j += step
            low = min(low, y[j])
        if y[j] <= half:                                 # crossing between j - step and j
            k = j - step
            sides.append(abs(x[j] + (half - y[j]) / (y[k] - y[j]) * (x[k] - x[j]) - x[i]))
        else:
            sides.append(None)
    if sides[0] is not None and sides[1] is not None:
        return float(sides[0] + sides[1])
    one = sides[0] if sides[0] is not None else sides[1]
    return float(2.0 * one) if one is not None else float("nan")


def _initial(y, x, guess, lower=-np.inf, upper=np.inf):
    """Refine a start position to the local maximum (±2 px) and estimate its height and width.

    The start moves to the highest sample within ±2 px only when that sample is a local
    maximum (else to the highest local maximum in that range, else see :func:`_climb`): on
    the flank of a stronger neighbour the highest sample is at the edge of the range and
    belongs to the neighbour. Only samples strictly between ``lower`` and ``upper`` (px; the
    midpoints to the neighbouring listed lines) are candidates, so a weak listed line next
    to a strong one does not start on the strong line's peak. The width is the half-maximum
    width around the segment's maximum (a nearby lamp line has about the same width), or,
    when that cannot be measured (the maximum is a neighbour's flank at the segment edge),
    around this line's own peak.
    """
    i0 = int(np.clip(round(guess - x[0]), 0, x.size - 1))
    lo, hi = max(0, i0 - 2), min(x.size, i0 + 3)
    while lo < hi and not x[lo] > lower:
        lo += 1
    while hi > lo and not x[hi - 1] < upper:
        hi -= 1
    base = float(np.percentile(y, 10))
    above = np.clip(y - base, 0, None)
    if lo == hi:
        i = i0
    else:
        i = lo + int(np.argmax(y[lo:hi]))
        if not _is_peak(y, i):
            peaks = [j for j in range(lo, hi) if _is_peak(y, j)]
            i = max(peaks, key=lambda j: y[j]) if peaks else _climb(y, x, i0, i, lower, upper, above)
    width = _halfmax_width(x, above)
    if not np.isfinite(width) or width <= 0:
        width = _halfmax_width_at(x, above, i)
    if not np.isfinite(width) or width <= 0:
        width = 2.0
    return i, float(y[i] - base), float(width)


def _climb(y, x, i0, edge, lower, upper, above):
    """Start index when no local maximum lies within ±2 px of the guess ``i0`` (``edge``: the
    highest sample there, at the end of the range on a rising flank).

    Either the guess is a little off a line (its peak is a few px further uphill) or a weak
    line sits as a shoulder on the flank of a stronger neighbour. Uphill from ``edge`` to the
    next local maximum ``j``: the start moves there when ``j`` is at most 4 px from the guess
    (the reach of a start at the range edge plus its ±2 px centre bound) and the guess lies
    in that line's core (within half its half-maximum width) or on its convex flank. A
    clearly concave guess (second difference below -3 × its noise) is a line of its own: the
    start stays at ``i0``.
    """
    step = 1 if edge > i0 else -1
    j = edge
    while 0 <= j + step < y.size and lower < x[j + step] < upper and y[j + step] > y[j]:
        j += step
    if not _is_peak(y, j) or abs(x[j] - x[i0]) > 4.0:
        return i0
    half = _halfmax_width_at(x, above, j)
    if np.isfinite(half) and abs(x[j] - x[i0]) <= 0.5 * half:
        return j
    if 0 < i0 < y.size - 1:
        curvature = y[i0 - 1] - 2.0 * y[i0] + y[i0 + 1]
        if curvature < -3.0 * np.sqrt(6.0) * _noise(y):
            return i0
    return j


def baseline_settings(baseline: str = "none", baseline_params: Optional[Mapping] = None) -> Tuple[str, dict]:
    """Canonical baseline method name and its full effective parameters (defaults filled in).

    Args:
        baseline: One of :data:`~modules.spectral_resolution.baseline.BASELINE_METHODS`
            (case-insensitive).
        baseline_params: Parameters of that method (``None`` = all defaults).

    Returns:
        ``(method, params)``: lower-case name and a new dict of every parameter it uses
        (``{}`` for ``"none"``), validated and coerced as
        :func:`~modules.spectral_resolution.baseline.estimate_baseline` would.

    Raises:
        ValueError: unknown method or parameter, a value out of range, or
            ``baseline_params`` is not a mapping.
    """
    if baseline_params is None:
        baseline_params = {}
    if not isinstance(baseline_params, Mapping):
        raise ValueError(f"baseline_params must be a mapping of parameter names, not {baseline_params!r}")
    method, options = _baseline_options(baseline, dict(baseline_params))
    return method, dict(options)


def remove_spectrum_baseline(spectrum, baseline: str = "none", baseline_params: Optional[Mapping] = None,
                             valid_columns: Optional[tuple] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract a baseline estimated only on the usable columns of a spectrum.

    Args:
        spectrum: 1-D counts.
        baseline, baseline_params: See :func:`baseline_settings`.
        valid_columns: ``(first, last)`` usable columns, inclusive (``None`` = all). Columns
            outside (zero-filled smile border) keep their values and get a zero baseline, so
            the border never bends the baseline.

    Returns:
        ``(corrected, baseline)``: new ``float64`` arrays of the spectrum's length. With
        ``"none"`` the baseline is all zeros and ``corrected`` a copy of the input.

    Raises:
        ValueError: as :func:`baseline_settings`, ``spectrum`` not 1-D, or the usable range
            is too short / not finite for the method.
    """
    method, options = baseline_settings(baseline, baseline_params)
    y = np.array(spectrum, dtype=float)
    if y.ndim != 1:
        raise ValueError("spectrum must be 1-D")
    first, last = (0, y.size - 1) if valid_columns is None else (int(valid_columns[0]), int(valid_columns[1]))
    base = np.zeros(y.size)
    if method != "none" and last >= first:
        base[first:last + 1] = estimate_baseline(y[first:last + 1], method, **options)
    return y - base, base


@dataclass(frozen=True, eq=False)          # holds arrays: compared/hashed by identity
class FitWindow:
    """One joint least-squares fit of :func:`fit_spectrum` (lines and neighbours fitted together).

    The model in this window is ``b0 + b1·(x - x0) + Σ_k P(x; area_k, center_k, width_k)`` on
    the baseline-corrected spectrum, ``P`` the pixel-integrated profile ``profile``.

    Attributes:
        first, last: Column range of the window (inclusive, px).
        x0: Reference column of the linear background (px).
        background: ``(b0, b1)`` of the linear background (counts, counts/px).
        areas, centers, widths: Per component: area (counts·px), centre (px) and width
            parameter (σ or γ, px); read-only arrays.
        lines: Per component, the index into ``pixels`` of a listed line, ``-1`` for an
            unlisted neighbouring peak.
        profile: Line model, one of :data:`PROFILES`.
        success: ``False`` when the optimiser did not converge: the values are then its last
            iterate (or the start values if it raised).
        wings: ``None``, or (Lorentzian only) the counts per column ``first..last`` of the lines
            fitted in *other* windows, subtracted from the data before this window was fitted
            again (their heavy wings curve the background more than a straight line can follow).
            Read-only.
    """

    first: int
    last: int
    x0: float
    background: tuple
    areas: np.ndarray
    centers: np.ndarray
    widths: np.ndarray
    lines: tuple
    profile: str
    success: bool
    wings: Optional[np.ndarray] = None

    def background_at(self, x) -> np.ndarray:
        """Linear background at pixel positions ``x`` (counts)."""
        x = np.asarray(x, dtype=float)
        return self.background[0] + self.background[1] * (x - self.x0)

    def wings_at(self, x) -> np.ndarray:
        """:attr:`wings` interpolated at pixel positions ``x`` (zeros when ``None``)."""
        x = np.asarray(x, dtype=float)
        if self.wings is None:
            return np.zeros(x.shape)
        return np.interp(x, np.arange(self.first, self.last + 1, dtype=float), self.wings)

    def component(self, k: int, x) -> np.ndarray:
        """Counts of component ``k`` (without background) in pixels centred at ``x`` (fractional allowed)."""
        model = _line_profile(self.profile)
        return model.counts(np.asarray(x, dtype=float), self.areas[k], self.centers[k], self.widths[k])

    def total(self, x) -> np.ndarray:
        """Whole model of the corrected spectrum at ``x``: background + :attr:`wings` + every component."""
        x = np.asarray(x, dtype=float)
        return self.background_at(x) + self.wings_at(x) + sum(
            (self.component(k, x) for k in range(len(self.lines))), np.zeros(x.shape))

    def fwhm_px(self, k: int) -> float:
        """FWHM (px) of component ``k`` as fitted (also for rejected lines)."""
        return float(_line_profile(self.profile).fwhm_per_width * self.widths[k])


@dataclass(frozen=True, eq=False)
class SpectrumFit:
    """Everything :func:`fit_spectrum` computed for one spectrum.

    Attributes:
        measurements: One :class:`LineMeasurement` per input pixel (as :func:`measure_spectrum`).
        baseline: Subtracted baseline (``float64``, zeros outside ``valid_columns`` and for ``"none"``).
        corrected: ``spectrum - baseline`` (``float64``): the data the windows were fitted to.
        windows: The joint fits, in ascending column order.
        valid_columns: ``(first, last)`` usable columns used.
    """

    measurements: list
    baseline: np.ndarray
    corrected: np.ndarray
    windows: list
    valid_columns: tuple


def measure_spectrum(spectrum, pixels: Sequence[float], *, m: Optional[float] = None,
                     c: Optional[float] = None, wavelengths: Optional[Sequence[float]] = None,
                     saturation: Optional[float] = None, snr_min: float = 10.0,
                     window_fwhm: Optional[float] = None, valid_columns: Optional[tuple] = None,
                     find_neighbours: bool = True, clipped=None,
                     profile: str = "gaussian", baseline: str = "none",
                     baseline_params: Optional[Mapping] = None,
                     cancel: Optional[Callable[[], bool]] = None) -> List[LineMeasurement]:
    """Measure the FWHM of the lines at ``pixels`` in one spectrum (see the module docstring).

    Args:
        spectrum: 1-D counts (an unsmoothed spectrum).
        pixels: Approximate line positions (px).
        m: Dispersion (nm/px at the spectrum's bin) for widths in nm.
        c: Intercept (nm) used with ``m`` when a wavelength is not given.
        wavelengths: Known wavelengths matching ``pixels`` (``None``/``nan`` = unknown).
        saturation: Counts at which the sensor clips; lines touching it are flagged
            ``saturated`` (judged on the raw counts, before any baseline removal).
        snr_min: Minimum ``amplitude / noise`` of an accepted line.
        window_fwhm: Fit half window in units of the line's first FWHM estimate (min 4 px);
            ``None`` = the profile's default (3 for ``gaussian``, 5 for ``lorentzian``).
        valid_columns: ``(first, last)`` usable columns (smile border excluded), inclusive.
        find_neighbours: Also fit unlisted peaks found inside a window.
        clipped: Optional boolean array (one per column) of clipped samples; overrides
            ``saturation`` (used for averaged spectra, where clipping happens per row).
        profile: Line model, one of :data:`PROFILES` (case-insensitive). Every line and
            neighbour of a joint fit uses it.
        baseline: Baseline removed before fitting, one of
            :data:`~modules.spectral_resolution.baseline.BASELINE_METHODS` (default
            ``"none"``: results exactly as without the option). Estimated on
            ``valid_columns`` only.
        baseline_params: Parameters of ``baseline`` (``None`` = its defaults).
        cancel: Optional ``cancel() -> bool``; when it returns True the measurement stops with
            :class:`MeasurementCancelled`.

    Returns:
        One :class:`LineMeasurement` per pixel, in input order.

    Raises:
        ValueError: ``spectrum`` is not 1-D, ``profile`` is not in :data:`PROFILES`, or the
            baseline method/parameters are invalid or cannot be applied (see
            :func:`remove_spectrum_baseline`).
        MeasurementCancelled: ``cancel()`` returned True.
    """
    return fit_spectrum(spectrum, pixels, m=m, c=c, wavelengths=wavelengths, saturation=saturation,
                        snr_min=snr_min, window_fwhm=window_fwhm, valid_columns=valid_columns,
                        find_neighbours=find_neighbours, clipped=clipped, profile=profile,
                        baseline=baseline, baseline_params=baseline_params, cancel=cancel).measurements


def _fit_window(least_squares, x, y, x0, n, p0, lower, upper, model, cancel):
    """One bounded least-squares fit of ``n`` components + linear background: ``(params, success)``.

    On an optimiser exception the start values are returned with ``success=False``;
    :class:`MeasurementCancelled` is passed on.
    """
    def residual(p):
        _check_cancel(cancel)
        return p[0] + p[1] * (x - x0) + _components(x, p, n, model).sum(axis=0) - y
    try:
        fit = least_squares(residual, p0, bounds=(lower, upper), x_scale="jac", max_nfev=400)
        return fit.x, bool(fit.success)
    except MeasurementCancelled:
        raise
    except Exception:  # noqa: BLE001
        return p0, False


def _wings(windows, index, x, model):
    """Counts at ``x`` of every line component fitted in the windows other than ``index``."""
    out = np.zeros(x.size)
    for j, w in enumerate(windows):
        if j != index and w["success"]:
            params = w["params"]
            out += _components(x, params, w["n"], model).sum(axis=0)
    return out


def fit_spectrum(spectrum, pixels: Sequence[float], *, m: Optional[float] = None,
                 c: Optional[float] = None, wavelengths: Optional[Sequence[float]] = None,
                 saturation: Optional[float] = None, snr_min: float = 10.0,
                 window_fwhm: Optional[float] = None, valid_columns: Optional[tuple] = None,
                 find_neighbours: bool = True, clipped=None, profile: str = "gaussian",
                 baseline: str = "none", baseline_params: Optional[Mapping] = None,
                 cancel: Optional[Callable[[], bool]] = None) -> SpectrumFit:
    """:func:`measure_spectrum` that also returns the baseline and every fitted window.

    Same arguments and measurements as :func:`measure_spectrum`; use it to draw the fitted
    components (:meth:`FitWindow.component`) over the corrected spectrum.

    Returns:
        :class:`SpectrumFit`.

    Raises:
        ValueError: as :func:`measure_spectrum`.
        MeasurementCancelled: ``cancel()`` returned True.
    """
    from scipy.optimize import least_squares
    from scipy.signal import find_peaks

    model = _line_profile(profile)
    method, options = baseline_settings(baseline, baseline_params)
    if window_fwhm is None:
        window_fwhm = model.window_fwhm
    y_raw = np.asarray(spectrum, dtype=float)
    if y_raw.ndim != 1:
        raise ValueError("spectrum must be 1-D")
    _check_cancel(cancel)
    n_col = y_raw.size
    first, last = (0, n_col - 1) if valid_columns is None else (int(valid_columns[0]), int(valid_columns[1]))
    if clipped is None:                                          # saturation: always on the raw counts
        clipped = (y_raw >= 0.98 * float(saturation)) if saturation is not None else np.zeros(n_col, bool)
    clipped = np.asarray(clipped, dtype=bool)
    if method == "none":
        y_all, removed = y_raw, np.zeros(n_col)
    else:
        y_all, removed = remove_spectrum_baseline(y_raw, method, options, (first, last))
    given = list(wavelengths) if wavelengths is not None else [None] * len(pixels)
    noise_all = max(_noise(y_all[first:last + 1]), 1e-9)
    x_all = np.arange(n_col, dtype=float)

    # first estimates (a start never moves past the midpoint to a neighbouring listed line)
    starts = []
    listed = [float(q) for q in pixels]
    for p in pixels:
        if not first + 2 <= p <= last - 2:
            starts.append(None)
            continue
        below = [(q + p) / 2.0 for q in listed if q < p]
        above = [(q + p) / 2.0 for q in listed if q > p]
        lo, hi = max(first, int(round(p)) - 12), min(last + 1, int(round(p)) + 13)
        i, height, width = _initial(y_all[lo:hi], x_all[lo:hi], p, max(below, default=-np.inf),
                                    min(above, default=np.inf))
        starts.append((x_all[lo + i], height, width))

    # windows and clusters of overlapping windows
    windows = []
    for s in starts:
        if s is None:
            windows.append(None)
            continue
        half = max(4.0, window_fwhm * s[2])
        windows.append((max(first, int(np.floor(s[0] - half))), min(last, int(np.ceil(s[0] + half)))))
    order = [i for i in np.argsort([p for p in pixels]) if windows[i] is not None]
    clusters, current = [], []
    for i in order:
        if current and windows[i][0] <= max(windows[j][1] for j in current):
            current.append(i)
        else:
            if current:
                clusters.append(current)
            current = [i]
    if current:
        clusters.append(current)

    # one joint fit per cluster
    fits = []
    for cluster in clusters:
        _check_cancel(cancel)
        lo = min(windows[i][0] for i in cluster)
        hi = max(windows[i][1] for i in cluster)
        x = x_all[lo:hi + 1]
        y = y_all[lo:hi + 1]
        centers = [starts[i][0] for i in cluster]
        widths = [starts[i][2] for i in cluster]
        heights = [starts[i][1] for i in cluster]
        if find_neighbours:                                     # unlisted peaks inside the window
            peaks, props = find_peaks(y - np.percentile(y, 10), prominence=max(5 * noise_all, 1e-9))
            for pk in peaks:
                if all(abs(x[pk] - cc) > max(1.5, 0.75 * ww) for cc, ww in zip(centers, widths)):
                    centers.append(float(x[pk]))
                    widths.append(float(np.median(widths)))
                    heights.append(float(y[pk] - np.percentile(y, 10)))
        n = len(centers)
        x0 = float(x.mean())
        edge = np.r_[y[:2], y[-2:]]
        p0 = [float(np.min(edge)), 0.0]
        lower, upper = [-np.inf, -np.inf], [np.inf, np.inf]
        for cc, ww, hh in zip(centers, widths, heights):        # width parameter σ or γ >= 0.2 px
            width = max(ww / model.fwhm_per_width, 0.3)
            p0 += [model.area(max(hh, 1.0), width), cc, width]
            lower += [0.0, cc - 2.0, 0.2]
            upper += [np.inf, cc + 2.0, max(1.0, (hi - lo) / 2.0)]
        p0 = np.clip(p0, np.array(lower) + 1e-9, np.array(upper) - 1e-9)
        params, ok_fit = _fit_window(least_squares, x, y, x0, n, p0, lower, upper, model, cancel)
        fits.append({"cluster": cluster, "lo": lo, "hi": hi, "x": x, "y": y, "x0": x0, "n": n,
                     "lower": lower, "upper": upper, "params": params, "success": ok_fit, "wings": None})

    # Lorentzian: the wings of strong lines in other windows curve this window's background
    # (they fall off only as 1/d²); fit again with them subtracted (first-pass parameters).
    if model.heavy_wings and len(fits) > 1:
        refits = []
        for index, f in enumerate(fits):
            x = f["x"]
            wings = _wings(fits, index, x, model)
            if not wings.any():
                refits.append(None)
                continue
            design = np.vstack([np.ones(x.size), x - f["x0"]]).T
            bent = wings - design @ np.linalg.lstsq(design, wings, rcond=None)[0]
            refits.append(wings if np.max(np.abs(bent)) > _WING_REFIT_NOISE * noise_all else None)
        for f, wings in zip(fits, refits):
            if wings is None:
                continue
            _check_cancel(cancel)
            y = f["y"] - wings
            start = np.clip(np.asarray(f["params"], dtype=float),     # from the first-pass result
                            np.array(f["lower"]) + 1e-9, np.array(f["upper"]) - 1e-9)
            params, ok_fit = _fit_window(least_squares, f["x"], y, f["x0"], f["n"], start, f["lower"],
                                         f["upper"], model, cancel)
            frozen_wings = np.array(wings, dtype=float)
            frozen_wings.setflags(write=False)
            f.update(y=y, params=params, success=ok_fit, wings=frozen_wings)

    results = {}
    fitted = []
    for f in fits:
        cluster, lo, hi, x, y, x0, n = f["cluster"], f["lo"], f["hi"], f["x"], f["y"], f["x0"], f["n"]
        params, ok_fit, lower, upper = f["params"], f["success"], f["lower"], f["upper"]
        comps = _components(x, params, n, model)
        background = params[0] + params[1] * (x - x0)
        rms = float(np.sqrt(np.mean((background + comps.sum(axis=0) - y) ** 2)))
        noise = max(noise_all, 1e-9)
        frozen = np.array(params, dtype=float)
        frozen.setflags(write=False)
        fitted.append(FitWindow(int(lo), int(hi), x0, (float(params[0]), float(params[1])), frozen[2::3],
                                frozen[3::3], frozen[4::3], tuple(int(i) for i in cluster) + (-1,) * (n - len(cluster)),
                                model.name, ok_fit, f["wings"]))

        for k, index in enumerate(cluster):
            area, mu, width = params[2 + 3 * k: 5 + 3 * k]
            line_only = y - background - (comps.sum(axis=0) - comps[k])
            amp = float(model.height(area, width))
            near = np.abs(x - mu) <= max(1.0, model.core * width)
            reach = model.extent * width                        # the line core must fit in the window
            pinned = mu <= lower[3 + 3 * k] + 1e-6 or mu >= upper[3 + 3 * k] - 1e-6   # centre at its ±2 px bound
            flag = "ok"
            if np.any(clipped[lo:hi + 1][near]):
                flag = "saturated"
            elif not ok_fit or width <= 0.2 + 1e-6 or width >= upper[4 + 3 * k] - 1e-6 or pinned:
                flag = "fit_failed"
            elif not (x[0] + 1 <= mu <= x[-1] - 1) or mu - reach < lo - 0.5 or mu + reach > hi + 0.5:
                flag = "outside"
            elif amp / max(noise, rms) < snr_min:
                flag = "low_snr"
            fwhm = float(model.fwhm_per_width * width)
            if flag == "ok" and fwhm < 2.0:
                flag = "undersampled"
            valid = flag in VALID_FLAGS
            known = given[index]
            wl = float(known) if known is not None and np.isfinite(known) else (
                float(m * mu + c) if m is not None and c is not None else float("nan"))
            fwhm_px = fwhm if valid else float("nan")
            fwhm_nm = float(fwhm_px * abs(m)) if m is not None and valid else float("nan")
            power = float(wl / fwhm_nm) if np.isfinite(wl) and np.isfinite(fwhm_nm) and fwhm_nm > 0 else float("nan")
            results[index] = LineMeasurement(float(pixels[index]), float(mu), fwhm_px,
                                             _halfmax_width(x, line_only), amp, amp / noise, n, flag,
                                             wl, fwhm_nm, power, model.name)

    out = []
    for i, p in enumerate(pixels):
        if i in results:
            out.append(results[i])
        else:
            known = given[i]
            wl = float(known) if known is not None and np.isfinite(known) else float("nan")
            out.append(LineMeasurement(float(p), float("nan"), float("nan"), float("nan"),
                                       float("nan"), float("nan"), 0, "border", wl, profile=model.name))
    removed.setflags(write=False)
    corrected = np.array(y_all, dtype=float)
    corrected.setflags(write=False)
    return SpectrumFit(out, removed, corrected, sorted(fitted, key=lambda f: f.first), (first, last))


@dataclass
class LineSummary:
    """One line summarised over all row blocks.

    Attributes:
        wavelength_nm: Line wavelength (nm) or ``nan``.
        pixel: Calibrated position (px).
        n_valid: Blocks with a valid width.
        n_blocks: Blocks measured.
        fwhm_px_median, fwhm_px_p16, fwhm_px_p84: FWHM statistics (px).
        fwhm_nm_median, fwhm_nm_p16, fwhm_nm_p84: FWHM statistics (nm).
        resolving_power_median: Median ``λ/Δλ``.
        main_flag: Most frequent flag (explains why a line has no valid width).
    """

    wavelength_nm: float
    pixel: float
    n_valid: int
    n_blocks: int
    fwhm_px_median: float
    fwhm_px_p16: float
    fwhm_px_p84: float
    fwhm_nm_median: float
    fwhm_nm_p16: float
    fwhm_nm_p84: float
    resolving_power_median: float
    main_flag: str


@dataclass
class ResolutionResult:
    """Every measurement of :func:`measure_frame`.

    Attributes:
        block_rows: ``(n_blocks, 2)`` first/last slit row (inclusive) of each block.
        measurements: ``measurements[b][j]`` = :class:`LineMeasurement` of line ``j`` in block ``b``.
        lines: Per-line :class:`LineSummary` (same order as the input pixels).
        settings: Parameters used (rows per block, saturation, m, smile corrected, ...).
        block_columns: ``(n_blocks, 2)`` first/last usable column (inclusive) of each block
            (the block's rows can have a wider zero-filled smile border than other rows), or
            ``None``.
    """

    block_rows: np.ndarray
    measurements: list
    lines: List[LineSummary]
    settings: dict = field(default_factory=dict)
    block_columns: Optional[np.ndarray] = None

    def array(self, name: str) -> np.ndarray:
        """``(n_blocks, n_lines)`` float array of a :class:`LineMeasurement` attribute."""
        return np.array([[getattr(w, name) for w in row] for row in self.measurements], dtype=float)

    def flags(self) -> np.ndarray:
        """``(n_blocks, n_lines)`` array of flag strings."""
        return np.array([[w.flag for w in row] for row in self.measurements], dtype=object)

    def summary(self) -> dict:
        """Median/min/max over the lines' median FWHM (px and nm) and the median resolving power."""
        px = np.array([s.fwhm_px_median for s in self.lines], dtype=float)
        nm = np.array([s.fwhm_nm_median for s in self.lines], dtype=float)
        rp = np.array([s.resolving_power_median for s in self.lines], dtype=float)

        def stat(v, fn):
            v = v[np.isfinite(v)]
            return float(fn(v)) if v.size else float("nan")
        return {"n_lines_valid": int(np.isfinite(px).sum()), "n_lines": len(self.lines),
                "fwhm_px_median": stat(px, np.median), "fwhm_px_min": stat(px, np.min),
                "fwhm_px_max": stat(px, np.max), "fwhm_nm_median": stat(nm, np.median),
                "fwhm_nm_min": stat(nm, np.min), "fwhm_nm_max": stat(nm, np.max),
                "resolving_power_median": stat(rp, np.median)}


def _intersect(a, b) -> tuple:
    """Intersection of two inclusive column ranges (``first > last`` when they do not overlap)."""
    return (max(int(a[0]), int(b[0])), min(int(a[1]), int(b[1])))


def usable_columns(frame) -> tuple:
    """``(first, last)`` columns (inclusive) that hold data in *every* row of ``frame``.

    Smile correction zero-fills the columns it cannot map, and the width of that border
    differs from row to row; a column zero-filled in only some rows still lowers (and bends)
    any average of the rows. Each non-empty row contributes its first..last non-zero column;
    the result is the intersection. Zeros *inside* a row (dark-subtracted data may hold genuine
    zeros) do not count, and all-zero rows are ignored.

    Args:
        frame: 2-D ``(slit rows, spectral columns)`` image (a 1-D spectrum counts as one row).

    Returns:
        ``(first, last)``; the whole width when every pixel is zero; the union of the rows'
        ranges in the degenerate case that no column holds data in every row.
    """
    image = np.asarray(frame)
    if image.ndim == 1:
        image = image[None, :]
    nonzero = image != 0
    nonzero = nonzero[nonzero.any(axis=1)]
    width = image.shape[1]
    if not nonzero.size:
        return (0, width - 1)
    starts = np.argmax(nonzero, axis=1)                         # first non-zero column per row
    ends = width - 1 - np.argmax(nonzero[:, ::-1], axis=1)      # last non-zero column per row
    first, last = int(starts.max()), int(ends.min())
    if first > last:
        return (int(starts.min()), int(ends.max()))
    return (first, last)


_valid_columns = usable_columns         # former private name


def row_blocks(n_rows: int, rows_per_block: int) -> List[Tuple[int, int]]:
    """Row blocks :func:`measure_frame` averages: ``[(first, last), ...]`` (inclusive).

    Consecutive blocks of ``rows_per_block`` rows; a last block shorter than half a block is
    merged into the previous one.

    Raises:
        ValueError: ``rows_per_block`` < 1.
    """
    step = int(rows_per_block)
    if step < 1:
        raise ValueError("rows_per_block must be >= 1")
    rows = int(n_rows)
    bounds = [(r, min(rows, r + step) - 1) for r in range(0, rows, step)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] + 1 < step / 2:      # merge a short tail
        bounds[-2] = (bounds[-2][0], bounds[-1][1])
        bounds.pop()
    return bounds


def measure_frame(frame, pixels: Sequence[float], *, rows_per_block: int = 16, m: Optional[float] = None,
                  c: Optional[float] = None, wavelengths: Optional[Sequence[float]] = None,
                  saturation: Optional[float] = None, snr_min: float = 10.0,
                  window_fwhm: Optional[float] = None,
                  progress: Optional[Callable[[float, str], None]] = None,
                  settings: Optional[dict] = None, profile: str = "gaussian",
                  baseline: str = "none", baseline_params: Optional[Mapping] = None,
                  valid_columns: Optional[tuple] = None,
                  cancel: Optional[Callable[[], bool]] = None) -> ResolutionResult:
    """Measure every line in blocks of ``rows_per_block`` slit rows averaged together.

    Averaging rows is only valid on a smile-corrected frame (straight lines); on an
    uncorrected frame use ``rows_per_block=1``.

    Args:
        frame: 2-D ``(slit rows, spectral columns)`` image.
        pixels, m, c, wavelengths, saturation, snr_min, window_fwhm, profile: See
            :func:`measure_spectrum`. ``saturation`` is compared with the *unaveraged* rows
            of a block.
        rows_per_block: Slit rows averaged into one spectrum (>= 1).
        progress: ``progress(fraction, message)`` after each block.
        settings: Extra entries stored in ``ResolutionResult.settings``.
        baseline, baseline_params: Baseline removed from every block's averaged spectrum
            (estimated on that block's usable columns only) before fitting; see
            :func:`measure_spectrum`. Default ``"none"``.
        valid_columns: ``(first, last)`` columns known to be usable in every row (e.g. the
            smile model's valid range, which also excludes the interpolation margin);
            ``None`` = the whole width. Each block is further limited to the
            :func:`usable_columns` of its own rows, so a zero-filled smile border never enters
            a block's average, baseline or fit.
        cancel: Optional ``cancel() -> bool``, polled between blocks and during the fits; True
            stops the measurement with :class:`MeasurementCancelled`.

    Returns:
        :class:`ResolutionResult`; its ``settings`` hold the canonical ``profile`` name, the
        ``model`` description, the ``window_fwhm`` actually used, the ``baseline`` method and
        its full effective ``baseline_params`` (defaults included), and ``valid_columns`` (the
        columns usable in every block); ``block_columns`` holds each block's own range.

    Raises:
        ValueError: ``frame`` is not 2-D, ``rows_per_block`` < 1, ``pixels`` is empty,
            ``profile`` is not in :data:`PROFILES` or the baseline method/parameters are
            invalid (all checked before any block is measured).
        MeasurementCancelled: ``cancel()`` returned True.
    """
    image = np.asarray(frame)
    if image.ndim != 2:
        raise ValueError("frame must be 2-D (slit rows, spectral columns)")
    if int(rows_per_block) < 1:
        raise ValueError("rows_per_block must be >= 1")
    if not len(pixels):
        raise ValueError("no lines to measure")
    model = _line_profile(profile)
    method, options = baseline_settings(baseline, baseline_params)
    if window_fwhm is None:
        window_fwhm = model.window_fwhm
    started = time.perf_counter()
    rows = image.shape[0]
    step = int(rows_per_block)
    bounds = row_blocks(rows, step)
    width = image.shape[1]
    outer = (0, width - 1) if valid_columns is None else (max(0, int(valid_columns[0])),
                                                           min(width - 1, int(valid_columns[1])))
    columns = _intersect(outer, usable_columns(image))         # usable in every block
    logger.info("FWHM (%s, baseline %s): %d lines x %d blocks of %d rows, frame %dx%d, columns %d-%d",
                model.name, method, len(pixels), len(bounds), step, width, rows, columns[0], columns[1])
    measurements, block_columns = [], []
    for b, (r0, r1) in enumerate(bounds):
        _check_cancel(cancel)
        block = image[r0:r1 + 1].astype(float)
        own = _intersect(outer, usable_columns(block))
        clipped = (block.max(axis=0) >= 0.98 * float(saturation)) if saturation is not None else None
        widths = measure_spectrum(block.mean(axis=0), pixels, m=m, c=c, wavelengths=wavelengths,
                                  snr_min=snr_min, window_fwhm=window_fwhm, valid_columns=own,
                                  clipped=clipped, profile=model.name, baseline=method,
                                  baseline_params=options, cancel=cancel)
        measurements.append(widths)
        block_columns.append(own)
        if progress is not None:
            try:
                progress((b + 1) / len(bounds), f"FWHM: rows {r0}-{r1} ({b + 1}/{len(bounds)})")
            except MeasurementCancelled:
                raise
            except Exception:  # noqa: BLE001
                logger.debug("progress callback failed", exc_info=True)

    lines = []
    for j, p in enumerate(pixels):
        col = [row[j] for row in measurements]
        px = np.array([w.fwhm_px for w in col], dtype=float)
        nm = np.array([w.fwhm_nm for w in col], dtype=float)
        rp = np.array([w.resolving_power for w in col], dtype=float)

        def q(v, pct):
            v = v[np.isfinite(v)]
            return float(np.percentile(v, pct)) if v.size else float("nan")
        flags = [w.flag for w in col]
        main = max(set(flags), key=flags.count)
        wl = next((w.wavelength_nm for w in col if np.isfinite(w.wavelength_nm)), float("nan"))
        lines.append(LineSummary(wl, float(p), int(np.isfinite(px).sum()), len(col), q(px, 50), q(px, 16), q(px, 84),
                                 q(nm, 50), q(nm, 16), q(nm, 84), q(rp, 50), main))
    result = ResolutionResult(np.array(bounds, dtype=int), measurements, lines,
                              dict(settings or {}, rows_per_block=step, saturation=saturation, m_nm_per_px=m,
                                   c_nm=c, snr_min=snr_min, window_fwhm=window_fwhm,
                                   valid_columns=list(columns), profile=model.name,
                                   model=f"pixel-integrated {model.label} + linear background",
                                   baseline=method, baseline_params=dict(options)),
                              np.array(block_columns, dtype=int).reshape(-1, 2))
    s = result.summary()
    logger.info("FWHM done in %.1fs: %d/%d lines valid, median %.3f px / %.4f nm",
                time.perf_counter() - started, s["n_lines_valid"], s["n_lines"], s["fwhm_px_median"],
                s["fwhm_nm_median"])
    return result
