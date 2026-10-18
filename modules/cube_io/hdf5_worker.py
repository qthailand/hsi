"""Background HDF5 save — keeps GUI responsive.

Data conventions:
    ``data`` is ``(frames, spatial, spectral)`` of any numeric dtype; the file layout is
    that of ``modules.cube_io.hdf5.save_hdf5`` (dataset ``"spectrogram"``).

Threading:
    ``HDF5SaveWorker`` is a ``QThread``: ``start()`` it from the GUI thread and connect
    its signals; the array is read from the worker thread, so do not modify it until
    ``finished`` / ``failed`` fires.  Requires ``h5py``.

Example:
    >>> import numpy as np
    >>> from modules.cube_io.hdf5_worker import HDF5SaveWorker       # doctest: +SKIP
    >>> w = HDF5SaveWorker("out.h5", np.zeros((10, 8, 8), np.uint16))  # doctest: +SKIP
    >>> w.finished.connect(print); w.start()                          # doctest: +SKIP
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class HDF5SaveWorker(QThread):
    """Save spectrogram to HDF5 in background thread.

    Args:
        path: Output ``.h5`` path (overwritten).
        data: ``(frames, spatial, spectral)`` array; kept by reference, not copied.
        parent: Optional QObject parent.
        metadata: Optional flat dict written as HDF5 root attributes (see ``save_hdf5``).

    Signals:
        progress(int):  percent 0-100
        finished(str):  saved file path
        failed(str):    error message

    ``progress`` fires about every 5 % of the frames and finally with 100;
    ``finished`` is emitted (with the path) after success, ``failed`` instead if any
    exception occurs (e.g. ``h5py`` missing).  Note that the interruption request made
    by a host is not checked while writing.
    """

    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, path: str, data: np.ndarray, parent=None, metadata=None):
        super().__init__(parent)
        self._path = path
        self._metadata = metadata
        self._data = data            # reference, not copy

    def run(self):
        """Thread body: write the file, then emit ``finished`` or ``failed`` (do not call directly; use ``start``)."""
        try:
            from modules.cube_io.hdf5 import save_hdf5
            save_hdf5(self._path, self._data, self.progress.emit, self._metadata)
            self.finished.emit(self._path)

        except Exception as e:
            logger.exception("HDF5 save failed: %s", e)
            self.failed.emit(str(e))