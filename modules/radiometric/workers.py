"""Qt adapter: processing sees only a callable progress callback.

Data conventions:
    ``Worker`` carries no array data of its own; the result of the wrapped function
    is forwarded unchanged in ``finished(object)``.

Threading:
    Create the worker on the GUI thread, ``moveToThread`` a ``QThread`` and connect
    ``thread.started`` to ``Worker.run``; ``run`` then executes on that thread and
    the signals are delivered queued to the GUI thread.

Example:
    >>> thread = QThread(); worker = Worker(fn, arg)            # doctest: +SKIP
    >>> worker.moveToThread(thread); thread.started.connect(worker.run)  # doctest: +SKIP
    >>> worker.progress.connect(bar.setValue); thread.start()   # doctest: +SKIP
"""
import traceback
from PyQt5.QtCore import QObject, pyqtSignal

class Worker(QObject):
    """Run a callable on a thread and report progress, result and errors as signals.

    The wrapped function must accept a ``progress_callback`` keyword (all functions in
    ``modules.radiometric.core`` do); it receives ``Worker.progress.emit``.

    Signals:
        finished(object): return value of the function.
        error(str): formatted traceback if the function raised.
        progress(int): forwarded ``progress_callback`` values (band counts).
    """
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, fn, *args, **kwargs):
        """Store the callable and its arguments; nothing runs until :meth:`run`.

        Args:
            fn: Function to call.
            *args: Positional arguments for ``fn``.
            **kwargs: Keyword arguments for ``fn``; ``progress_callback`` is added by ``run``.
        """
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """Call ``fn`` (slot; execute on the worker thread) and emit ``finished`` or ``error``.
        """
        try:
            self.kwargs['progress_callback'] = self.progress.emit
            result = self.fn(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())

