"""Profile peek: a small popup with a calibration's key curve and facts, before its full viewer.

A status card (Dark, Smile, Flat-field, Radiometric, Wavelength) opens a peek next to itself;
*Open ▸* in the peek opens the full viewer.

* :class:`PeekData`, :class:`PeekCurve` and the builders :func:`dark_peek`, :func:`smile_peek`,
  :func:`flatfield_peek`, :func:`reference_peek`, :func:`text_peek` - Qt-free (``core.py``);
* :class:`ProfilePeek` - the popup widget (``widget.py``, loaded lazily).

Data conventions:
    See :mod:`modules.profile_peek.core`: darks ``(slit, wavelength)`` counts, flat-field and
    reference tables ``(slit row, band)``, wavelengths in nm.

Threading:
    The builders are pure functions (any thread); ``ProfilePeek`` lives on the GUI thread.

Example:
    >>> import numpy as np
    >>> from modules.profile_peek import dark_peek
    >>> dark_peek({"mean": np.zeros((3, 5))}).color
    'green'
"""
from .core import (COLORS, PeekCurve, PeekData, dark_peek, flatfield_peek, reference_peek, smile_peek,
                   text_peek)

__all__ = ["COLORS", "PeekCurve", "PeekData", "dark_peek", "smile_peek", "flatfield_peek", "reference_peek",
           "text_peek", "ProfilePeek"]
__requires__ = ()                   # duck-typed: takes the objects of other modules, imports none
__widgets__ = ("ProfilePeek",)


def __getattr__(name):
    """Import the Qt widget only when it is requested, so ``import modules.profile_peek`` stays Qt-free."""
    if name == "ProfilePeek":
        from .widget import ProfilePeek
        return ProfilePeek
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
