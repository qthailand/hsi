"""Live focus map of a line-target frame: where the spectrograph is sharp, and a score to focus by.

* :func:`focus_map` - Qt-free bar-contrast map, best-focus column per slit row and the
  ``worst`` (least sharp part) score (``core.py``);
* :func:`focus_advice` - one-line reading of where the focus band sits;
* :class:`FocusMapWidget` - live window (``widget.py``, loaded lazily).

Data conventions:
    Frames are 2-D ``(slit, wavelength)`` of a target whose bars run along the spectral axis;
    any numeric dtype. Columns/rows are sensor pixels at the frame's binning; wavelengths in nm.

Threading:
    The core functions are pure (any thread); the widget is GUI-thread only.

Example:
    >>> import numpy as np
    >>> from modules.focus_map import focus_map
    >>> rows = np.arange(256)[:, None]
    >>> fm = focus_map(1000 + 400 * np.cos(2 * np.pi * rows / 12.0) * np.ones((1, 300)))
    >>> round(fm.period, 1)
    12.0
"""
from .core import FocusMap, bar_period, focus_advice, focus_map, linear_wavelength

__all__ = ["FocusMap", "bar_period", "focus_advice", "focus_map", "linear_wavelength", "FocusMapWidget"]
__requires__ = ()
__widgets__ = ("FocusMapWidget",)


def __getattr__(name):
    """Import the Qt widget only when it is requested, so ``import modules.focus_map`` stays Qt-free."""
    if name == "FocusMapWidget":
        from .widget import FocusMapWidget
        return FocusMapWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
