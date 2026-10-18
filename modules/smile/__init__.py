# smile/__init__.py
"""Smile Correction Module for Imaging Spectrometers.

Public API:
    Core:        load_frame, save_frame, analyze_smile
    Sklearn:     SmileCorrector
    Calibration: SmileCalibration, save_calibration, load_calibration
    Widget:      SmileCorrectionWidget (requires PyQt5)

Usage::

    from modules.smile import SmileCorrector

    corrector = SmileCorrector(deg_y=2, deg_x=1)
    corrector.fit(hgar_frame)
    corrector.save("model.npz")
    corrected = corrector.transform(raw_frame, crop=True)

Data conventions:
    A spectral image is a 2-D array ``(slit rows, spectral columns)``; rows and columns
    are zero-based pixels. ``smile`` is a curved-line distortion: a spectral line's
    column shifts with the slit row. Corrections remap columns only (rows are unchanged).
    Analysis works in ``float32``; the runtime path (:class:`modules.smile.runtime.SmileRuntime`)
    keeps uint8/uint16/float32 input dtype. Invalid border columns are zero-filled or cropped
    (``margin = (left, right)`` px). Calibrations are stored as ``.npz`` (see the README).
    The reference row defaults to :func:`modules.contracts.default_reference_row`.

Threading:
    Analysis and remap functions are pure and may run in any thread (OpenCV releases the
    GIL). :class:`SmileCalibration` caches a remap engine and is not safe to mutate from
    several threads. :class:`SmileCorrectionWidget` must be used from the GUI thread; it
    runs analysis in its own ``QThread`` and waits for it in ``closeEvent``.

Qt (``SmileCorrectionWidget`` etc.) is imported lazily, so ``import modules.smile`` does
not load PyQt5 or matplotlib's GUI backend.

Example:
    >>> import numpy as np
    >>> from modules.smile import SmileCorrector
    >>> h, w = 200, 300
    >>> y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    >>> frame = np.random.default_rng(0).normal(100, 2, (h, w)).astype(np.float32)
    >>> for column in (50, 100, 150, 200, 250):            # curved synthetic lamp lines
    ...     frame += 3000 * np.exp(-((x - column - 5e-5 * (y - h / 2) ** 2) ** 2) / 4.5)
    >>> corrector = SmileCorrector(deg_y=2, deg_x=1).fit(frame)
    >>> corrector.is_fitted, corrector.transform(frame, crop=True).ndim
    (True, 2)
"""

from .core import (
    Line,
    auto_sat_level,
    detect_lines,
    estimate_noise,
    load_frame,
    robust_sigma,
    save_frame,
    straightness_rms,
    trace_line,
)
from .surface import (
    Surface2D,
    cross_validate_lines,
    fit_surface_2d,
)
from .calibration import (
    SmileCalibration,
    build_maps,
    calibration_from_analysis,
    correct_smile,
    crop_margin,
    load_calibration,
    margin_from_map,
    save_calibration,
)
from .analysis import (
    SmileAnalysis,
    analyze_smile,
    verify_on_frame,
)
from .corrector import SmileCorrector

_GUI_EXPORTS = {
    "HAVE_QT", "PlotCanvas", "SmileCorrectionWidget",
    "SmileMainWindow", "launch_gui",
}


def __getattr__(name: str):
    """Load Qt-facing symbols only when a caller explicitly asks for them.

    Importing :mod:`smile` for image processing therefore does not import
    PyQt5, QtAgg, or Matplotlib's GUI backend.
    """
    if name in _GUI_EXPORTS:
        from . import widget

        value = getattr(widget, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Names for ``dir()``: the loaded globals plus the lazily loaded Qt exports."""
    return sorted(set(globals()) | _GUI_EXPORTS)

__version__ = "2.5.0"

__all__ = [
    # Core
    "load_frame", "save_frame", "robust_sigma", "auto_sat_level",
    "estimate_noise", "Line", "detect_lines", "trace_line",
    "straightness_rms",
    # Surface
    "Surface2D", "fit_surface_2d", "cross_validate_lines",
    # Remap
    "build_maps", "margin_from_map", "correct_smile", "crop_margin",
    # Analysis
    "SmileAnalysis", "analyze_smile", "verify_on_frame",
    # Calibration
    "SmileCalibration", "calibration_from_analysis",
    "save_calibration", "load_calibration",
    # Sklearn API
    "SmileCorrector",
    # Qt (optional)
    "SmileCorrectionWidget", "PlotCanvas", "HAVE_QT",
    "SmileMainWindow", "launch_gui",
]

from .quality import (SpectralReference, QualityReport, analyze_quality,
                      estimate_shift_uncertainty, correlation_shift)

from .config import SmileConfig

__all__ += ["SmileConfig", "SpectralReference", "QualityReport", "analyze_quality",
            "estimate_shift_uncertainty", "correlation_shift"]

__requires__ = ("contracts", "image")
__widgets__ = ("SmileCorrectionWidget",)
