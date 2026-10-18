"""Qt wrapper: :class:`StageThread` runs a :class:`~modules.stage.sequence.StageSequence`.

Threading:
    ``start()`` runs the sequence on its own ``QThread``; the signals below are
    emitted from that thread and delivered queued to GUI receivers.
    ``request_stop()`` may be called from the GUI thread.
"""
from PyQt5.QtCore import QObject, QThread, pyqtSignal as Signal

from .sequence import StageSequence


class StageThread(StageSequence, QThread):
    """Thread that moves the stage: home -> scan -> return home.

    Signals:
        stagebefore: The scan move is about to start (start recording).
        stageafter: The scan move ended, at the endpoint or on a manual stop
            (build the cube).
        returned_home: Return-home finished successfully.
        error(str): Any hardware/Python failure; see ``failed_phase``.

    The motion parameters (``degree``, ``degree_per_second``, ``distance_mm``,
    ``speed_um_s``) and result flags (``failed``, ``completed``, ``cancelled``, ...)
    are those of :class:`StageSequence`.

    Args:
        parent: Optional Qt parent.
        stage_type: ``"rotation"`` or ``"linear"``.
        stage_factory: Optional zero-argument factory of an unopened stage driver
            (used by tests and simulators).
    """

    stagebefore = Signal()
    stageafter = Signal()
    returned_home = Signal()
    error = Signal(str)

    def __init__(self, parent: QObject = None, stage_type='rotation', stage_factory=None):
        QThread.__init__(self, parent)
        StageSequence.__init__(self, stage_type=stage_type, stage_factory=stage_factory)

    def _notify_before(self):
        self.stagebefore.emit()

    def _notify_after(self):
        self.stageafter.emit()

    def _notify_home(self):
        self.returned_home.emit()

    def _notify_error(self, message: str):
        self.error.emit(message)
