"""Cube assembly, ENVI loading and cube/spectrogram export. Existing axis ordering is preserved.

Public API:

* ``create_pushbroom_cube_from_corrected`` - stack corrected frames into a cube.
* ``load_envi_cube_for_viewer`` - open an ENVI cube lazily (memory map) plus wavelengths.
* ``as_analysis_cube`` - the cube as C-contiguous BIP in memory (BSQ/BIL transposed once).
* ``envi_value_fields`` - an ENVI header's 'data ignore value' / 'data full scale'.
* ``parse_envi_header`` / ``read_envi_band`` - minimal ENVI header and single-band reader.
* ``HypercubeWriter`` (ENVI BIP), ``GeoTIFFWriter``, ``SpectrogramWriter``
  (raw binary), ``save_hdf5`` - exporters.
* Qt adapters (loaded lazily, need PyQt5): ``HypercubeWorker``, ``GeoTIFFWorker``,
  ``SpectrogramWorker`` (``export_workers``), ``HDF5SaveWorker`` (``hdf5_worker``) and
  ``CubeWorker`` (``cube_worker``).

Data conventions:
    * Corrected frame: ``(slit, wavelength)``.
    * Cube in memory / viewer: ``(slit, scan, wavelength)`` - on disk this is an
      ENVI BIP file with ``lines = slit``, ``samples = scan``, ``bands = wavelength``.
    * Writers (``HypercubeWriter``, ``GeoTIFFWriter``) take a *band-first*
      cube ``(wavelength, slit, scan)`` plus a 1-D ``linear_spectrum`` of band
      wavelengths in nm; they keep the cube's dtype (ENVI ``data type`` follows it).
    * ``SpectrogramWriter`` / ``save_hdf5`` take a raw spectrogram
      ``(frames, spatial, spectral)``.
    * Wavelengths are nm.  ``config`` dicts carry the calibration ``m``
      (nm/pixel at hardware bin 1) and ``c`` (nm), needed by ``SpectrogramWriter``
      only; ``camera_bin`` is the camera's hardware binning factor.

Threading:
    The writers, loaders and parsers are Qt-free; importing this package never imports
    PyQt5 (the adapters below are lazy attributes).  Writers expose a blocking ``run()`` meant to be executed in a worker
    thread and report progress through an optional ``progress(int percent)``
    callback invoked on that thread.  ``SpectrogramWriter.cancel()`` may be
    called from another thread.  Loaders and parsers are plain functions.  The Qt
    adapters wrap the writers in ``QObject`` workers (move them to a ``QThread`` and
    connect ``started`` to ``run``) or, for ``HDF5SaveWorker``, a ``QThread`` subclass;
    results arrive as signals.

Example:
    >>> import numpy as np
    >>> from modules.cube_io import create_pushbroom_cube_from_corrected
    >>> frames = [np.zeros((4, 10), np.uint16) for _ in range(3)]   # (slit, wavelength)
    >>> create_pushbroom_cube_from_corrected(frames).shape           # (slit, scan, wavelength)
    (4, 3, 10)
"""
from .core import (as_analysis_cube, create_pushbroom_cube_from_corrected, envi_value_fields,
                   load_envi_cube_for_viewer)

from .envi import parse_envi_header, _read_envi_band_float32 as read_envi_band
from .writers import HypercubeWriter, GeoTIFFWriter, SpectrogramWriter
from .hdf5 import save_hdf5

__all__ = (
    "create_pushbroom_cube_from_corrected",
    "load_envi_cube_for_viewer",
    "as_analysis_cube",
    "envi_value_fields",
    "parse_envi_header",
    "read_envi_band",
    "HypercubeWriter",
    "GeoTIFFWriter",
    "SpectrogramWriter",
    "save_hdf5",
    "HypercubeWorker",
    "GeoTIFFWorker",
    "SpectrogramWorker",
    "HDF5SaveWorker",
    "CubeWorker",
)
__requires__ = ("spectral_processing",)

_LAZY_QT = {
    "HypercubeWorker": "export_workers",
    "GeoTIFFWorker": "export_workers",
    "SpectrogramWorker": "export_workers",
    "HDF5SaveWorker": "hdf5_worker",
    "CubeWorker": "cube_worker",
}


def __getattr__(name):
    """Lazily import the Qt adapters so that importing the package never loads Qt.

    Raises:
        AttributeError: For any other unknown name.
    """
    if name in _LAZY_QT:
        import importlib
        module = importlib.import_module("." + _LAZY_QT[name], __name__)
        return getattr(module, name)
    raise AttributeError(name)
