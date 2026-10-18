"""live_view: live camera-frame display (image canvas, display mapping, histogram, spectrum).

Purpose:
    Everything needed to show a live pushbroom/area-sensor stream: convert raw
    frames to 8 bit (``DisplayMapper``), show them with a selectable row and a
    focus ROI (``LiveImageCanvas``), toggle RAW/Enhance (``DisplayControls``),
    and plot the intensity histogram (``HistogramPlot``), the live spectrum of
    the selected row (``LiveLineSpectrumWidget``) or a static ROI/cube spectrum
    (``SpectrumPlot``).

Data conventions:
    * A frame is a 2-D array ``(rows, columns)`` = ``(slit, wavelength)``,
      normally uint16; a 3-D ``(rows, columns, C)`` array uses channel 0.
    * A spectrum is a 1-D array over bands; ``wavelengths`` are in nm.
    * ``selected_y`` / ROI values are 0-based image row indices.
    * ``bit_depth`` is the sensor full scale in bits (full scale ``2**bit_depth-1``).
    * A saturation ``ceiling`` is a frame (or spectrum) of the value a clipped
      sensor pixel shows at each position after dark subtraction, Smile and
      binning; ``saturated_mask`` flags values at >= 99 % of it.

Threading:
    The widgets are GUI-thread only: feed them from a camera thread through a
    queued signal.  ``DisplayMapper``/``DisplayMode`` are Qt-free numpy code and
    can be used from any thread.  ``import modules.live_view`` does not import
    PyQt5; widget classes load lazily on first access (``DisplayMapper`` and
    ``DisplayMode`` are imported eagerly because they are Qt-free).

Embedding:
    All widgets in ``__widgets__`` can be constructed with no arguments.

Example:
    >>> import numpy as np
    >>> from modules.live_view import DisplayMapper, DisplayMode
    >>> DisplayMapper(DisplayMode.RAW, 16).to_uint8(np.array([[4096]], np.uint16)).tolist()
    [[16]]
    >>> from PyQt5.QtWidgets import QApplication                    # doctest: +SKIP
    >>> from modules.live_view import LiveImageCanvas, DisplayControls  # doctest: +SKIP
    >>> app = QApplication([])                                      # doctest: +SKIP
    >>> canvas, controls = LiveImageCanvas(), DisplayControls()     # doctest: +SKIP
    >>> controls.mapper_changed.connect(canvas.set_display_mapper)  # doctest: +SKIP
    >>> canvas.set_array(np.zeros((512, 640), np.uint16))           # doctest: +SKIP

Run the demo with ``python -m modules.live_view``.
"""
from __future__ import annotations

import importlib

from .mapping import DisplayMapper, DisplayMode
from .saturation import saturated_mask

__all__ = (
    "DisplayMapper",
    "DisplayMode",
    "saturated_mask",
    "DisplayControls",
    "LiveImageCanvas",
    "LiveLineSpectrumWidget",
    "SpectrumPlot",
    "HistogramPlot",
)
__requires__ = ("image",)
__widgets__ = (
    "DisplayControls",
    "LiveImageCanvas",
    "LiveLineSpectrumWidget",
    "SpectrumPlot",
    "HistogramPlot",
)

_LAZY = {
    "DisplayControls": "display_controls",
    "LiveImageCanvas": "image_canvas",
    "LiveLineSpectrumWidget": "spectrum",
    "SpectrumPlot": "spectrum",
    "HistogramPlot": "histogram",
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
