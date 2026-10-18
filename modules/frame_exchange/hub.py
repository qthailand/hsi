"""``FrameExchange``: Qt hub that routes one-shot snapshot requests to a single producer.

Protocol:
    1. A consumer calls ``request()`` and gets an integer id.
    2. ``capture_requested(id)`` is emitted (queued, after ``request`` returns) to the
       producer, which captures one frame.
    3. The producer calls ``publish(FrameSnapshot)`` or ``reject(id, message)``.
    4. Consumers receive ``frame_ready(snapshot)`` or ``failed(id, message)``; match
       the id with the one returned by ``request``.
    Only one request may be pending; it times out after 10 s with ``failed``.

Data conventions:
    See ``modules.frame_exchange.snapshot.FrameSnapshot`` (``image`` is
    ``(slit, wavelength)``, shared, read-only).

Threading:
    A ``QObject``: create and call it from the thread it lives in (normally the GUI
    thread). Producers in other threads should reach ``publish``/``reject`` through
    queued signal connections, as the HSI main window does.

Example:
    >>> from PyQt5.QtWidgets import QApplication
    >>> app = QApplication.instance() or QApplication([])
    >>> import numpy as np
    >>> from modules.frame_exchange import FrameExchange, FrameSnapshot
    >>> hub = FrameExchange()
    >>> got = []
    >>> _ = hub.frame_ready.connect(got.append)
    >>> token = hub.request()
    >>> hub.publish(FrameSnapshot(token, np.zeros((2, 2)), 0.0, 1, False, False, False))
    >>> got[0].request_id == token
    True
"""
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from .snapshot import FrameSnapshot

class FrameExchange(QObject):
    """Route one-shot requests and retain only the most recent owned snapshot.

    No live frame copying or broadcasting occurs without a request. Consumers
    treat image as read-only and copy only if their own algorithm modifies it.

    Signals:
        capture_requested(int): request id; connect to the producer.
        frame_ready(object): the published ``FrameSnapshot``.
        failed(int, str): request id and a human-readable reason.

    Attributes:
        latest (FrameSnapshot | None): most recently published snapshot.
    """
    capture_requested = pyqtSignal(int)
    frame_ready = pyqtSignal(object)
    failed = pyqtSignal(int, str)

    def __init__(self, parent=None):
        """Create an idle exchange.

        Args:
            parent: Optional parent ``QObject``.
        """
        super().__init__(parent)
        self.latest = None
        self._serial = 0
        self._pending = None
        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.timeout.connect(self._on_timeout)

    def request(self):
        """Start a snapshot request and arm the 10 s timeout.

        Returns:
            int: the request id; ``capture_requested`` is emitted with it from the
            event loop after this call returns, so the caller can register the id first.

        Raises:
            RuntimeError: a request is already pending.
        """
        if self._pending is not None:
            raise RuntimeError('A snapshot request is already pending')
        self._serial += 1
        self._pending = self._serial
        self._timeout.start(10000)
        # Delivery after returning lets the consumer register its request ID.
        QTimer.singleShot(0, self._dispatch)
        return self._serial

    def _dispatch(self):
        if self._pending is not None:
            self.capture_requested.emit(self._pending)

    def publish(self, packet):
        """Deliver the producer's frame for the pending request.

        Args:
            packet: ``FrameSnapshot`` whose ``request_id`` equals the pending id.
                Packets with any other id (stale or unsolicited) are ignored. The
                image is not copied; ``latest`` and all consumers share it.
        """
        if packet.request_id != self._pending:
            return
        self._timeout.stop()
        self._pending = None
        self.latest = packet
        self.frame_ready.emit(packet)

    def reject(self, request_id, message):
        """Fail the pending request.

        Args:
            request_id: Id of the request being rejected; ignored unless it is the
                pending one.
            message: Reason, forwarded through ``failed``.
        """
        if request_id != self._pending:
            return
        self._timeout.stop()
        self._pending = None
        self.failed.emit(request_id, message)

    def _on_timeout(self):
        if self._pending is not None:
            self.reject(self._pending, 'Snapshot timed out: no camera frame available')
