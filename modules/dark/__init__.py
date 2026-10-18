"""Qt-independent dark-reference capture and cached correction API.

``DarkProcessor`` applies a dark mean and hot-pixel repair to 2-D
``(slit, wavelength)`` frames, held as an immutable ``DarkReference``; ``DarkFrameLibrary`` captures, stores and loads
the ``.npz`` dark references per (bin, exposure, gain, sensor temperature); ``select_dark`` picks
the one nearest the current temperature and checks it fits the camera frame; ``DarkFramePanel`` (lazy, needs PyQt5 and qtawesome) is the status/control widget.

Example:
    >>> import numpy as np
    >>> from modules.dark import DarkProcessor
    >>> proc = DarkProcessor()
    >>> proc.set_dark_reference(np.zeros((2, 2), np.uint16), None)
"""
from .config import DarkConfig
from .core import DarkProcessor, DarkReference
from .library import DarkFrameLibrary, UNKNOWN_TEMPERATURE_C, valid_temperature
from .selection import DarkSelection, select_dark

__all__ = ['DarkConfig', 'DarkProcessor', 'DarkReference', 'DarkFrameLibrary', 'DarkSelection', 'select_dark',
           'UNKNOWN_TEMPERATURE_C', 'valid_temperature', 'DarkFramePanel']
__requires__ = ()
# DarkFramePanel is a QGroupBox (PyQt5 + qtawesome, loaded lazily); constructible with no arguments.
__widgets__ = ('DarkFramePanel',)


def __getattr__(name):
    """Import the Qt panel only when it is requested."""
    if name == 'DarkFramePanel':
        from .widget import DarkFramePanel
        return DarkFramePanel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
