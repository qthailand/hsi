"""Qt worker that makes a C-contiguous copy of an array off the GUI thread.

Data conventions:
    Any ``numpy.ndarray``; ``np.ascontiguousarray`` returns the same array if it is
    already C-contiguous, otherwise a copy.

Threading:
    Create in the GUI thread, ``moveToThread`` a ``QThread`` and connect
    ``thread.started`` to ``run``.  Signals are delivered to the receiver's thread.

Example:
    >>> import numpy as np
    >>> from PyQt5.QtCore import QThread                            # doctest: +SKIP
    >>> from modules.cube_io.cube_worker import CubeWorker          # doctest: +SKIP
    >>> worker, thread = CubeWorker(np.zeros((4, 5, 6))[:, :, ::2]), QThread()  # doctest: +SKIP
    >>> worker.moveToThread(thread); thread.started.connect(worker.run)         # doctest: +SKIP
"""
import time

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal as Signal


class CubeWorker(QObject):
    """Make an array C-contiguous in a worker thread.

    Args:
        array: Array to convert (e.g. a strided/transposed cube view), or ``None``.
        parent: Optional QObject parent.

    Signals:
        ascontiguos(object): the C-contiguous ``numpy.ndarray`` (same object when
            the input already was contiguous); emitted from ``run`` before
            ``finished``.  (The name keeps its historical spelling.)
        finished(): always emitted when ``run`` is done (also for ``None`` input).
    """

    ascontiguos = Signal(object)
    finished = Signal()
    _d: np.ndarray = None

    def __init__(self, array: np.ndarray, parent: QObject = None):
        super().__init__(parent)
        self._d = array

    def run(self):
        """Convert the array and emit ``ascontiguos`` then ``finished``.

        Intended to run in the worker thread (connect to ``QThread.started``).
        Emits only ``finished`` when the array is ``None``.
        """
        if self._d is None:
            self.finished.emit()
            return

        t = time.time()
        a_c = np.ascontiguousarray(self._d)
        self.ascontiguos.emit(a_c)
        self.finished.emit()
