"""Frame-by-frame spectrogram viewer with HDF5 export.

Public API:

* ``SpectrogramViewer`` - ``QWidget`` (no-argument constructor) that browses a
  ``(n, h, v)`` spectrogram with a slider / keyboard and saves it as ``.h5``.

Data conventions:
    * Spectrogram: 3-D array ``(n, h, v)`` = ``(frames, spatial, spectral)``, any numeric
      dtype (typically uint16 raw frames); held by reference, not copied.
    * Each frame is min-max normalised to 8 bit for display only; data are never modified.
    * Saved file: HDF5 dataset ``"spectrogram"`` (see ``modules.cube_io.save_hdf5``).

Threading:
    GUI thread only.  Saving uses a ``QThread`` (``modules.cube_io.hdf5_worker.HDF5SaveWorker``)
    owned by the viewer and a progress dialog.

Example:
    >>> import numpy as np
    >>> from modules.spectrogram_viewer import SpectrogramViewer     # doctest: +SKIP
    >>> viewer = SpectrogramViewer()                                  # doctest: +SKIP
    >>> viewer.load(np.random.randint(0, 4096, (20, 64, 96), dtype=np.uint16))  # doctest: +SKIP
    >>> viewer.show()                                                 # doctest: +SKIP

Run ``python -m modules.spectrogram_viewer`` for a demo.  Importing this package does not
import PyQt5; ``SpectrogramViewer`` is loaded lazily on first access.
"""

__all__ = ("SpectrogramViewer",)
__requires__ = ("cube_io",)
__widgets__ = ("SpectrogramViewer",)


def __getattr__(name):
    """Lazily import ``SpectrogramViewer`` so that importing the package never loads Qt.

    Raises:
        AttributeError: For any other unknown name.
    """
    if name == "SpectrogramViewer":
        from .window import SpectrogramViewer
        return SpectrogramViewer
    raise AttributeError(name)
