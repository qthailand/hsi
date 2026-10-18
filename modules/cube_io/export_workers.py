"""Qt signal adapters for standalone cube export services.

Each worker wraps one Qt-free writer from ``modules.cube_io.writers`` so that it can
be run on a ``QThread`` and report through signals.

Data conventions:
    ``HypercubeWorker`` / ``GeoTIFFWorker`` take a band-first cube
    ``(wavelength, slit, scan)`` plus ``linear_spectrum`` (nm); ``SpectrogramWorker``
    takes ``(frames, spatial, spectral)``.  Keyword arguments are passed through to
    the writer unchanged (``save_path``, ``config``, ``camera_bin``, ...).

Threading:
    Build the worker in the GUI thread, ``moveToThread`` a ``QThread`` and connect
    ``thread.started`` to ``run``.  ``run`` blocks that thread; signals are queued to
    the receivers' threads.  ``SpectrogramWorker.cancel`` may be called from any thread.

Example:
    >>> from PyQt5.QtCore import QThread                              # doctest: +SKIP
    >>> from modules.cube_io.export_workers import HypercubeWorker    # doctest: +SKIP
    >>> w = HypercubeWorker(save_path="a.img", config={"m": .29, "c": 400.},
    ...                     hypercube=cube_band_first, linear_spectrum=wl_nm)   # doctest: +SKIP
    >>> t = QThread(); w.moveToThread(t); t.started.connect(w.run)    # doctest: +SKIP
    >>> w.finished.connect(lambda seconds: t.quit()); t.start()       # doctest: +SKIP
"""
import traceback
from PyQt5.QtCore import QObject, pyqtSignal as Signal
from modules.cube_io.writers import GeoTIFFWriter, HypercubeWriter, SpectrogramWriter

class _ExportWorker(QObject):
    """Base class: run ``writer_type(...).run()`` and report through signals.

    Subclasses set ``writer_type``.  Constructor keyword arguments are forwarded to the
    writer, with ``progress`` bound to the ``progressed`` signal.

    Signals:
        finished(float): seconds taken (the writer's ``run`` return value), emitted on success.
        progressed(int): percent 0-100, emitted from the worker thread during the write.
        error(str): formatted traceback if the writer raised; ``finished`` is then not emitted.
    """

    finished=Signal(float)
    progressed=Signal(int)
    error=Signal(str)
    writer_type=None

    def __init__(self, *, parent=None, **kwargs):
        super().__init__(parent)
        self.writer=self.writer_type(progress=self.progressed.emit, **kwargs)

    def run(self):
        """Run the writer (blocking) and emit ``finished`` or ``error``; call from the worker thread."""
        try:self.finished.emit(self.writer.run())
        except Exception:self.error.emit(traceback.format_exc())

class GeoTIFFWorker(_ExportWorker):
    """Export a band-first ``(wavelength, slit, scan)`` cube as GeoTIFF (see ``GeoTIFFWriter``)."""

    writer_type=GeoTIFFWriter

class HypercubeWorker(_ExportWorker):
    """Export a band-first ``(wavelength, slit, scan)`` cube as ENVI BIP (see ``HypercubeWriter``)."""

    writer_type=HypercubeWriter
    chunk_size=100*1024*1024

class SpectrogramWorker(_ExportWorker):
    """Export a ``(frames, spatial, spectral)`` spectrogram as raw binary (see ``SpectrogramWriter``)."""

    writer_type=SpectrogramWriter
    def cancel(self):
        """Request cancellation of the running export; safe to call from any thread."""
        self.writer.cancel()
