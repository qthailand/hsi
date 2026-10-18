"""ui_kit: small reusable PyQt5 widgets and mixins (range slider, zoom/pan, MDI resize, title bar).

Purpose:
    Application-independent UI building blocks used by the live view, cube
    preview and cube viewer modules; each can also be dropped into any other
    PyQt5 application.

    * ``RangeSlider`` - dual-handle slider, values are fractions ``0..1``
    * ``ZoomPanMixin`` - wheel zoom + middle-button pan for any ``QWidget``
    * ``ResizableMdiSubWindow`` - ``QMdiSubWindow`` resizable from all edges
    * ``IOSTitleBar`` - macOS-style custom title bar (needs ``qtawesome``)
    * ``TrafficLightButton`` - the round close/minimise/maximise button
    * ``dashboard_stylesheet`` - recolours the qdarktheme QSS to a green accent

Data conventions:
    * ``RangeSlider`` positions are floats in ``0.0 .. 1.0``.
    * Zoom/pan works in widget pixels; image coordinates are ``(x, y) =
      (column, row)``; zoom 1.0 means "fit to window".

Threading:
    Everything here is GUI-thread only.  Importing ``modules.ui_kit`` does not
    import PyQt5; the widget classes are loaded lazily on first attribute access.

Embedding:
    ``RangeSlider`` and ``ResizableMdiSubWindow`` can be built with no
    arguments (``__widgets__``).  ``IOSTitleBar(window, title)`` needs the
    window it controls and ``TrafficLightButton(color, hover_color)`` needs
    colours, so they are not in ``__widgets__``.

Example:
    >>> from PyQt5.QtWidgets import QApplication              # doctest: +SKIP
    >>> from modules.ui_kit import RangeSlider                # doctest: +SKIP
    >>> app = QApplication([])                                # doctest: +SKIP
    >>> slider = RangeSlider(low=0.1, high=0.9)               # doctest: +SKIP
    >>> slider.range_changed.connect(lambda lo, hi: print(lo, hi))  # doctest: +SKIP
    >>> slider.show()                                         # doctest: +SKIP

Run the demo with ``python -m modules.ui_kit``.
"""
from __future__ import annotations

import importlib

__all__ = (
    "RangeSlider",
    "ZoomPanMixin",
    "ResizableMdiSubWindow",
    "IOSTitleBar",
    "TrafficLightButton",
    "dashboard_stylesheet",
)
__requires__ = ()
__widgets__ = ("RangeSlider", "ResizableMdiSubWindow")

_LAZY = {
    "RangeSlider": "range_slider",
    "ZoomPanMixin": "zoom_pan",
    "ResizableMdiSubWindow": "resizable_subwindow",
    "IOSTitleBar": "title_bar",
    "TrafficLightButton": "title_bar",
    "dashboard_stylesheet": "theme",
}


def __getattr__(name):
    """Import the submodule that defines ``name`` on first access (keeps PyQt5 out of ``import``)."""
    try:
        submodule = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals()) + list(__all__))
