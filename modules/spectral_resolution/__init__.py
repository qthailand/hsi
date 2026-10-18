"""FWHM / spectral resolution of calibration-lamp lines on a smile-corrected frame.

Qt-free API (:mod:`modules.spectral_resolution.core`): :func:`measure_spectrum` (one
spectrum), :func:`fit_spectrum` (the same plus the baseline and every fitted window, for
plotting), :func:`measure_frame` (every block of slit rows), :class:`LineMeasurement`,
:class:`LineSummary`, :class:`ResolutionResult` (``cancel=`` stops a long measurement with
:class:`MeasurementCancelled`). Line shapes: :data:`PROFILES`
(pixel-integrated Gaussian or Lorentzian). Baseline removers
(:mod:`modules.spectral_resolution.baseline`): :func:`estimate_baseline`,
:func:`remove_baseline` with :data:`BASELINE_METHODS` (ModPoly, IModPoly, ZhangFit/airPLS,
AsLS, arPLS), their :data:`BASELINE_LABELS`, :data:`BASELINE_DEFAULTS` and
:data:`BASELINE_PARAM_HELP`. Excel export in :mod:`modules.spectral_resolution.excel_report`.

The panel :class:`SpectralResolutionWidget` grabs a frame, loads the wavelength profile and
smile model stored for the same camera setting (:class:`modules.contracts.CalibrationStore`),
pre-selects the peaks to measure (the calibration lines, else detected peaks), lets the user
add/move/remove peaks on the spectrum, choose the profile and the baseline, and only then
calculates. It is imported lazily.

Data conventions:
    Frames are ``(slit rows, spectral columns)``; pixels at the frame's hardware bin;
    wavelengths and widths in nm; intensities in counts. Widths come from pixel-integrated
    Gaussians or Lorentzians fitted jointly with neighbouring lines on a linear background,
    optionally after removing a smooth baseline estimated on the usable columns.

Threading:
    The Qt-free API is pure; the widget runs measurements in its own ``QThread``.

Example:
    >>> import numpy as np
    >>> from modules.spectral_resolution import measure_spectrum
    >>> x = np.arange(200.0)
    >>> s = 50 + 1000 * np.exp(-0.5 * ((x - 80.3) / 2.0) ** 2)
    >>> round(measure_spectrum(s, [80])[0].fwhm_px, 2)      # point samples: narrower than 4.71
    4.66
    >>> w = measure_spectrum(s + 0.02 * (x - 100) ** 2, [80], profile="gaussian", baseline="imodpoly")[0]
    >>> w.flag, round(w.fwhm_px, 1)
    ('ok', 4.7)
"""
from .baseline import BASELINE_LABELS, BASELINE_METHODS
from .baseline import DEFAULT_PARAMS as BASELINE_DEFAULTS
from .baseline import PARAM_HELP as BASELINE_PARAM_HELP
from .baseline import estimate_baseline, remove_baseline
from .core import (DEFAULT_WINDOW_FWHM, FLAGS, PROFILES, VALID_FLAGS, FitWindow, LineMeasurement, LineSummary,
                   MeasurementCancelled, ResolutionResult, SpectrumFit, baseline_settings, fit_spectrum,
                   measure_frame, measure_spectrum, remove_spectrum_baseline, row_blocks, usable_columns)

__all__ = ["measure_spectrum", "fit_spectrum", "measure_frame", "LineMeasurement", "LineSummary",
           "ResolutionResult", "SpectrumFit", "FitWindow", "FLAGS", "VALID_FLAGS", "PROFILES",
           "DEFAULT_WINDOW_FWHM", "BASELINE_METHODS", "BASELINE_LABELS", "BASELINE_DEFAULTS",
           "BASELINE_PARAM_HELP", "estimate_baseline", "remove_baseline", "baseline_settings",
           "remove_spectrum_baseline", "row_blocks", "usable_columns", "MeasurementCancelled",
           "SpectralResolutionWidget"]
__requires__ = ("contracts", "image", "smile")
__widgets__ = ("SpectralResolutionWidget",)


def __getattr__(name):
    """Import the Qt widget only on request, so ``import modules.spectral_resolution`` stays Qt-free."""
    if name == "SpectralResolutionWidget":
        from .widget import SpectralResolutionWidget
        return SpectralResolutionWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
