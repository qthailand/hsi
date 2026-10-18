"""One-line summary of what this module does (shown in docs/MODULES.md).

Longer description: the problem it solves, and the public pieces:

* :func:`scale` - Qt-free logic (``core.py``);
* :class:`TemplateWidget` - the embeddable widget (``widget.py``, loaded lazily).

Data conventions:
    State the array axes, dtypes and units here, e.g. images are ``(slit, wavelength)``
    ``uint16``; wavelengths in nm; exposure in ms.

Threading:
    Say which calls are thread-safe and which thread owns what.

Example:
    >>> from modules._template import scale
    >>> scale([1, 2, 3], 2.0)
    [2.0, 4.0, 6.0]
"""
from .core import scale

__all__ = ["scale", "TemplateWidget"]
__requires__ = ()                 # other modules.* packages imported anywhere in this package
__widgets__ = ("TemplateWidget",)  # widgets constructible with NO arguments


def __getattr__(name):
    """Import the Qt widget only when it is requested, so ``import modules.<name>`` stays Qt-free."""
    if name == "TemplateWidget":
        from .widget import TemplateWidget
        return TemplateWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
