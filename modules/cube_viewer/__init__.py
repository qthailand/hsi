"""Interactive hyperspectral cube viewer: band browser, RGB composite, point / rectangle / lasso spectra with CSV / XLSX export, whole-cube segmentation (SAM ... SVM) with separability and method comparison.

Public API:

* ``CubeViewerWindow`` - the full viewer (``QWidget``); needs a cube, so it is NOT
  constructible without arguments and is not listed in ``__widgets__``.
* ``CubeImageCanvas`` - single-band image canvas with zoom/pan and ``pixel_clicked``.
* ``RGBCompositeControls`` - single-band / RGB mode toggle, band spin boxes, stretch.

Spectra (submodules, imported directly): ``roi_stats`` (exact ROI statistics, numpy
only), ``spectra`` (``SpectrumSet`` / ``SpectrumRecord``), ``spectra_export`` (plain CSV,
XLSX, clipboard TSV), ``roi_draw`` (rectangle / lasso tools and overlays), ``jobs``
(background ``JobRunner``) and ``spectra_panel`` (the sidebar SPECTRA panel).
Segmentation (Phase 2): ``segment`` (numpy classifiers, thresholds, separability),
``ml`` (Random Forest / SVM, optional scikit-learn), ``compare`` (method comparison),
``segment_session`` (state, overlay, worker) and ``segment_dialog`` (the Segment dialog).

Data conventions:
    * Cube: 3-D array ``(slit, scan, wavelength)`` = ``(H, W, bands)``, any numeric
      dtype (usually uint16).  A ``numpy.memmap`` from
      ``modules.cube_io.load_envi_cube_for_viewer`` is fine.  Held by reference.
    * Wavelengths: 1-D array in nm, one value per band, or ``None``.
    * Image pixel ``(ix, iy)`` = ``(column, row)`` = ``(scan, slit)``; ``cube[iy, ix, :]``
      is the spectrum of that pixel.
    * ``config`` dict (``m`` nm/pixel at bin 1, ``c`` nm, optional ``pixel_size``) and
      ``camera_bin`` are forwarded to the export dialog; a numeric ``full_scale`` is also
      the saturation level counted in the spectra statistics, and the camera scalars go
      into the XLSX spectra metadata.

Threading:
    GUI thread only.  Large ROI statistics run on the viewer's ``JobRunner`` worker
    thread; exports run in a ``QThread`` owned by the export dialog, which uses the
    adapters in ``modules.cube_io.export_workers``.

Example:
    >>> import numpy as np
    >>> from modules.cube_viewer import CubeViewerWindow            # doctest: +SKIP
    >>> cube = np.random.randint(0, 4096, (32, 48, 40), dtype=np.uint16)
    >>> win = CubeViewerWindow(cube, wavelengths=np.linspace(400, 1000, 40))  # doctest: +SKIP
    >>> win.show()                                                   # doctest: +SKIP

Run ``python -m modules.cube_viewer [file.hdr]`` for a stand-alone window.  Importing
this package does not import PyQt5; the classes are loaded lazily on first access.
"""

__all__ = ("CubeViewerWindow", "CubeImageCanvas", "RGBCompositeControls")
__requires__ = ("cube_io", "cube_preview", "live_view", "ui_kit")
__widgets__ = ("CubeImageCanvas", "RGBCompositeControls")

_LAZY = {
    "CubeViewerWindow": "window",
    "CubeImageCanvas": "canvas",
    "RGBCompositeControls": "controls",
}


def __getattr__(name):
    """Lazily import the Qt classes so that importing the package never loads Qt.

    Raises:
        AttributeError: For any other unknown name.
    """
    if name in _LAZY:
        import importlib
        module = importlib.import_module("." + _LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(name)
