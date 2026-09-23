import time

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal as Signal


class CubeWorker(QObject):
    ascontiguos = Signal(object)
    finished = Signal()
    _d: np.ndarray = None

    def __init__(self, array: np.ndarray, parent: QObject = None):
        super().__init__(parent)
        self._d = array

    def run(self):
        if self._d is None:
            self.finished.emit()
            return

        t = time.time()
        a_c = np.ascontiguousarray(self._d)
        self.ascontiguos.emit(a_c)
        self.finished.emit()
