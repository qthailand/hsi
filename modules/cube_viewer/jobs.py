"""Background jobs of the cube viewer: one persistent worker thread, FIFO, cancellable per record.

``JobRunner`` (GUI thread) owns one ``QThread`` with one ``_JobWorker`` moved to it - the
ExportDialog pattern of ``window.py`` - started on the first ``submit``.  Each job is a
callable ``fn(cancel) -> result`` keyed by a record id (``rid``); ``cancel`` is a
``threading.Event.is_set`` bound method that long jobs poll (``roi_stats(...,
cancel=cancel)`` checks it once per ~1 MB chunk and raises ``RoiCancelled``).

Rules:
    * FIFO: jobs run one at a time in submission order (queued Qt calls into the
      worker's event loop).
    * At most one live job per ``rid``: submitting again cancels the earlier job.
    * ``cancel(rid)`` sets that job's event; a queued job is then skipped, a running one
      stops at its next poll.  After ``cancel`` returns, ``done`` / ``failed`` are never
      emitted for that job, even when its result is already on its way (results are
      matched to the live job on the GUI side), so a late result of a deleted record
      cannot reach the window.
    * ``failed(rid, message)`` carries ``'<ExceptionType>: <text>'``; a job that raises
      ``RoiCancelled`` without being cancelled reports ``failed(rid, 'cancelled')``.
    * ``shutdown()`` cancels everything, quits the thread and waits for it.  A running
      ``roi_stats`` job stops within one chunk and a worker-side ``polygon_mask`` within
      one batch of about 10 ms, so this takes milliseconds (see README.md); a job that
      ignores ``cancel`` delays it.  When the wait times out
      (default 5 s) the thread is parked - kept referenced until it has finished -
      instead of being destroyed while running; the job's closure keeps the data it
      reads alive meanwhile.

Data conventions:
    Jobs are keyed by the int record id (``rid``) of a
    :class:`~modules.cube_viewer.spectra.SpectrumRecord`.  The runner never touches a cube:
    a job closure holds its own references to the arrays it reads (normally the viewer's
    ``(slit, scan, wavelength)`` cube) and returns its result - a ``RoiStats`` for the
    viewer - by reference, without copying.

Threading:
    Create, submit, cancel and shut down from the GUI thread; ``done`` / ``failed`` are
    emitted in the GUI thread.  Job callables run in the worker thread and must not
    touch widgets; numpy releases the GIL in its heavy loops, so the GUI stays
    responsive.

Example:
    >>> from modules.cube_viewer.jobs import JobRunner                  # doctest: +SKIP
    >>> from modules.cube_viewer.roi_stats import roi_stats             # doctest: +SKIP
    >>> runner = JobRunner()                                             # doctest: +SKIP
    >>> runner.done.connect(lambda rid, stats: print(rid, stats.n_px))   # doctest: +SKIP
    >>> runner.submit(7, lambda cancel: roi_stats(cube, (0, 0, 400, 400), cancel=cancel))  # doctest: +SKIP
    >>> runner.shutdown()                                                # doctest: +SKIP
"""

from __future__ import annotations

import itertools
import logging
import threading

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from modules.cube_viewer.roi_stats import RoiCancelled

__all__ = ("JobRunner",)

logger = logging.getLogger(__name__)

# (thread, worker) pairs whose shutdown() wait timed out: kept referenced until the thread
# has finished, so that a running QThread is never destroyed (that aborts the process).
_PARKED: list = []


def _prune_parked() -> None:
    """Forget parked threads that have finished."""
    _PARKED[:] = [entry for entry in _PARKED if not entry[0].isFinished()]


class _Job:
    """One submitted job: its token, record id, callable and cancel event."""

    __slots__ = ("token", "rid", "fn", "event")

    def __init__(self, token: int, rid: int, fn):
        self.token = token
        self.rid = rid
        self.fn = fn
        self.event = threading.Event()


class _JobWorker(QObject):
    """Runs the jobs in the worker thread, one queued ``run`` call at a time (FIFO).

    Signals:
        finished(object, str, object): ``(job, outcome, payload)`` with outcome
            ``'done'`` (payload = result), ``'failed'`` (payload = message) or
            ``'cancelled'`` (payload None).
    """

    finished = pyqtSignal(object, str, object)

    @pyqtSlot(object)
    def run(self, job: _Job):
        """Run one job unless it was cancelled while queued, and report the outcome."""
        if job.event.is_set():
            self.finished.emit(job, "cancelled", None)
            return
        try:
            result = job.fn(job.event.is_set)
        except RoiCancelled:
            outcome, payload = "cancelled", None
        except Exception as exc:  # noqa: BLE001 - every job error is reported, never raised in the thread
            logger.debug("cube viewer job %s failed", job.rid, exc_info=True)
            outcome, payload = "failed", f"{type(exc).__name__}: {exc}"
        else:
            outcome, payload = ("cancelled", None) if job.event.is_set() else ("done", result)
        self.finished.emit(job, outcome, payload)


class JobRunner(QObject):
    """FIFO background jobs keyed by record id, on one persistent worker thread.

    Args:
        parent: Optional ``QObject`` parent (the viewer).  The owner must still call
            :meth:`shutdown` (e.g. in ``closeEvent``) before dropping the data the jobs
            read.
        name: Object name of the worker thread (shows in debuggers).

    Signals:
        done(int, object): ``(rid, result)`` of a job that finished and was not cancelled.
        failed(int, str): ``(rid, message)`` of a job that raised.
    """

    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    _submit = pyqtSignal(object)

    def __init__(self, parent=None, name: str = "cube-viewer-jobs"):
        super().__init__(parent)
        self._name = name
        self._thread: QThread | None = None
        self._worker: _JobWorker | None = None
        self._live: dict = {}                 # rid -> _Job (submitted, not yet reported)
        self._tokens = itertools.count(1)
        self._closed = False

    # ── public API ────────────────────────────────────────

    def submit(self, rid: int, fn) -> None:
        """Queue ``fn(cancel)`` for record ``rid``; an earlier job of ``rid`` is cancelled.

        Args:
            rid: Record id the result belongs to.
            fn: Callable run in the worker thread with one argument, ``cancel`` (a
                no-argument callable returning True once the job is cancelled).  Its
                return value is emitted with ``done``.

        Raises:
            RuntimeError: The runner was shut down.
        """
        if self._closed:
            raise RuntimeError("JobRunner.submit after shutdown()")
        self.cancel(rid)
        job = _Job(next(self._tokens), int(rid), fn)
        self._live[job.rid] = job
        self._ensure_thread()
        self._submit.emit(job)

    def cancel(self, rid: int) -> bool:
        """Cancel the job of ``rid``; nothing will be emitted for it afterwards.

        Returns:
            bool: True if ``rid`` had a queued or running job.
        """
        job = self._live.pop(rid, None)
        if job is None:
            return False
        job.event.set()
        return True

    def cancel_all(self) -> int:
        """Cancel every queued and running job; returns how many there were."""
        jobs = list(self._live.values())
        self._live.clear()
        for job in jobs:
            job.event.set()
        return len(jobs)

    def is_pending(self, rid: int) -> bool:
        """True while ``rid`` has a job that is queued or running."""
        return rid in self._live

    def pending(self) -> list:
        """Record ids with a queued or running job, in submission order."""
        return list(self._live)

    @property
    def is_running(self) -> bool:
        """bool: True while the worker thread is running."""
        return self._thread is not None and self._thread.isRunning()

    def shutdown(self, timeout_ms: int | None = 5000) -> bool:
        """Cancel all jobs, stop the worker thread and wait for it.  Idempotent.

        Args:
            timeout_ms: Longest wait in ms (default 5 s); None waits until the running
                job has stopped (it polls ``cancel`` between chunks).

        Returns:
            bool: True when the thread has finished (or never started); False when the
            timeout expired - the thread is then parked until it finishes.
        """
        self._closed = True
        self.cancel_all()
        _prune_parked()
        thread, worker = self._thread, self._worker
        self._thread = self._worker = None
        if thread is None:
            return True
        thread.quit()
        finished = thread.wait() if timeout_ms is None else thread.wait(int(timeout_ms))
        if not finished:
            logger.warning("cube viewer job thread still running after %s ms; parked", timeout_ms)
            _PARKED.append((thread, worker))
        return bool(finished)

    # ── internal ──────────────────────────────────────────

    def _ensure_thread(self) -> None:
        """Start the worker thread on first use."""
        if self._thread is not None:
            return
        thread = QThread()
        thread.setObjectName(self._name)
        worker = _JobWorker()
        worker.moveToThread(thread)
        self._submit.connect(worker.run)          # queued: the worker lives in the thread
        worker.finished.connect(self._on_finished)  # queued back to the GUI thread
        self._thread, self._worker = thread, worker
        thread.start()

    @pyqtSlot(object, str, object)
    def _on_finished(self, job: _Job, outcome: str, payload) -> None:
        """GUI thread: report a job's outcome unless it was cancelled or superseded."""
        if self._live.get(job.rid) is not job:
            return
        del self._live[job.rid]
        if outcome == "done":
            self.done.emit(job.rid, payload)
        elif outcome == "failed":
            self.failed.emit(job.rid, payload)
        else:                                  # raised RoiCancelled without being cancelled
            self.failed.emit(job.rid, "cancelled")
