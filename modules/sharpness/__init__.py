"""Live focus aid: a sharpness score and a sidebar widget that shows it.

* :func:`sharpness_score` / :class:`ImageSharpness` - Qt-free score of an image.
* :class:`SharpnessPanel` - narrow vertical widget with a large number and a bar
  (loaded lazily; needs PyQt5).

Data conventions:
    Images are ``(slit, wavelength)``; use ``axis=0`` to measure focus along the slit.

Threading:
    The score is computed in the acquisition worker; ``SharpnessPanel`` must be fed
    from the GUI thread (``set_value``).

Example:
    >>> import numpy as np
    >>> from modules.sharpness import sharpness_score
    >>> round(sharpness_score(np.zeros((8, 8)), axis=0), 3)
    0.0
"""
from .core import ImageSharpness, sharpness_score

__all__ = ["ImageSharpness", "sharpness_score", "SharpnessPanel"]
__requires__ = ()
# SharpnessPanel is constructible with no arguments.
__widgets__ = ("SharpnessPanel",)


def __getattr__(name):
    """Import the Qt widget only when it is requested."""
    if name == "SharpnessPanel":
        from .widget import SharpnessPanel
        return SharpnessPanel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
