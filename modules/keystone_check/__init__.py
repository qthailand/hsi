"""Residual keystone of a corrected cube, measured on the horizontal edges of the scene.

* :func:`check_keystone` / :class:`KeystoneCheck` - Qt-free measurement (``core.py``): every
  horizontal edge of a reference band followed through the bands; shift per edge and band,
  wavelength-zone statistics and a PASS / CHECK verdict (95 % of shifts under ``PASS_PX``);
* :class:`KeystoneCheckWidget` - window to open a cube (or take an open one) and check it
  (``widget.py``, loaded lazily).

Data conventions:
    Cubes are ``(slit, scan, wavelength)`` (ENVI BIP order), wavelengths in nm, shifts in pixels
    along the slit (positive = lower in the frame than in the reference band).

Threading:
    The core is pure (any thread); the widget is GUI-thread only and measures on a ``QThread``.

Example:
    >>> import numpy as np
    >>> from modules.keystone_check import check_keystone
    >>> cube = np.ones((200, 40, 30), np.float32); cube[100:] = 3.0
    >>> check_keystone(cube, np.linspace(900, 1700, 30), band_step=1).passed
    True
"""
from .core import PASS_PX, KeystoneCheck, check_keystone

__all__ = ["PASS_PX", "KeystoneCheck", "check_keystone", "KeystoneCheckWidget"]
__requires__ = ("cube_io",)
__widgets__ = ("KeystoneCheckWidget",)


def __getattr__(name):
    """Import the Qt widget only when it is requested, so ``import modules.keystone_check`` stays Qt-free."""
    if name == "KeystoneCheckWidget":
        from .widget import KeystoneCheckWidget
        return KeystoneCheckWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
