"""cube_preview: realtime RGB preview of a pushbroom scan with click-to-plot spectra.

Purpose:
    While a hyperspectral cube is being scanned, show a growing false-colour
    image (one column per frame), let the user click pixels to plot their
    spectra and drag the R/G/B band lines to change the composite.  The heavy
    lifting (RGB composition, spectrum extraction) runs on a worker thread.

Data conventions:
    * Cube = ``(slit, scan, wavelength)``.  The preview is fed frame by frame:
      one frame is a 2-D ``(slit, wavelength)`` array, normally uint16.
    * The displayed image is ``(slit rows, scan columns)``; click coordinates
      are ``(x, y)`` = ``(scan column, slit row)``; ``cube[y, x, :] == frames[x][y, :]``.
    * Wavelengths are 1-D arrays in nm (one per band) or ``None``.
    * Saturation masks are boolean ``(n_frames, slit_rows)``.

Threading:
    ``RealtimeCubePreviewWidget`` is used from the GUI thread and owns a
    ``CubeRenderWorker`` on its own ``QThread``.  Frames are stored by
    reference (no copy).  The widget does not stop its thread by itself: call
    ``cleanup()`` before the widget is destroyed (e.g. in ``closeEvent`` or on
    ``QApplication.aboutToQuit``).  ``import modules.cube_preview`` does not
    import PyQt5; classes load lazily on first access (``find_closest_band``
    and the colour table need Qt/matplotlib too, so they are lazy as well).

Embedding:
    ``RealtimeCubePreviewWidget`` and ``CenteredPixmapLabel`` can be built with no
    arguments (``__widgets__``).  ``CubeRenderWorker`` is a ``QObject`` (not a
    widget): create it, connect ``rgb_ready`` / ``frames_count_ready`` /
    ``spectrum_ready`` and call ``start()``.

Example:
    >>> import numpy as np
    >>> from PyQt5.QtWidgets import QApplication                   # doctest: +SKIP
    >>> from modules.cube_preview import RealtimeCubePreviewWidget  # doctest: +SKIP
    >>> app = QApplication([])                                     # doctest: +SKIP
    >>> w = RealtimeCubePreviewWidget()                            # doctest: +SKIP
    >>> wl = np.linspace(400, 1000, 200)                           # doctest: +SKIP
    >>> w.push_frame(np.zeros((128, 200), np.uint16), wl, 200)     # doctest: +SKIP
    >>> w.show(); app.aboutToQuit.connect(w.cleanup)               # doctest: +SKIP

Run the demo with ``python -m modules.cube_preview``.
"""
from __future__ import annotations

import importlib

__all__ = (
    "RealtimeCubePreviewWidget",
    "CenteredPixmapLabel",
    "CubeRenderWorker",
    "find_closest_band",
    "RGB_TARGET_WAVELENGTHS_VIS",
    "RGB_TARGET_WAVELENGTHS_SWIR",
)
__requires__ = ("ui_kit",)
__widgets__ = ("RealtimeCubePreviewWidget", "CenteredPixmapLabel")

_LAZY = {
    "RealtimeCubePreviewWidget": "widget",
    "CenteredPixmapLabel": "widget",
    "find_closest_band": "widget",
    "RGB_TARGET_WAVELENGTHS_VIS": "widget",
    "RGB_TARGET_WAVELENGTHS_SWIR": "widget",
    "CubeRenderWorker": "render",
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
