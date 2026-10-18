"""Frame acquisition and scan bookkeeping, usable with or without Qt.

* :class:`AcquisitionEngine` - runs one camera through *fetch -> dark/hot-pixel ->
  orientation -> Smile -> spectral binning -> record* and reports to an
  :class:`AcquisitionSink`. No Qt; run it on your own thread (``run_until_stopped``).
* :class:`CameraPreviewThread` - the same engine on a ``QThread`` with Qt signals
  (loaded lazily; needs PyQt5).
* :class:`CorrectionStatusDisplay` - Dark and Smile status cards fed from the engine's
  ``correction_status`` dict (lazy; needs PyQt5).
* :class:`ScanRecorder` - thread-safe "record every frame" buffers.
* :class:`ScanSession` - GUI-independent scan state machine (what to do when a scan
  starts, ends or fails, including the visual-selection preview scan).

Data conventions:
    Frames are ``(slit, spectral)`` ``uint16``; binned frames ``(slit, wavelength_bin)``;
    a scan's cube is ``(slit, scan, wavelength)``. Every processed frame is recorded -
    the 30 fps display throttle only affects what the GUI shows.

Threading:
    One worker thread owns the camera; all other methods are thread-safe requests.

Example:
    >>> from modules.acquisition import ScanRecorder
    >>> import numpy as np
    >>> r = ScanRecorder(); r.set_recording(True)
    >>> r.offer(np.zeros((2, 3), np.uint16), np.zeros((2, 6), np.uint16), keep_raw=True)
    True
    >>> len(r.take_frames()), len(r.take_raw())
    (1, 1)
"""
from .config import ProcessingConfig
from .engine import AcquisitionEngine, AcquisitionSink
from .metadata import build_raw_metadata, camera_bit_info
from .profiler import StageProfiler
from .recorder import ScanRecorder
from .session import ScanOutcome, ScanSession

__all__ = [
    "ProcessingConfig",
    "build_raw_metadata", "camera_bit_info",
    "AcquisitionEngine", "AcquisitionSink", "CameraPreviewThread", "ScanRecorder",
    "ScanSession", "ScanOutcome", "StageProfiler", "CorrectionStatusDisplay",
]
__requires__ = ("dark", "frame_exchange", "preprocessing", "sharpness", "smile", "spectral_processing", "status_card")
# CorrectionStatusDisplay is a QObject (two StatusCards), not a QWidget, so it is not listed.
__widgets__ = ()


def __getattr__(name):
    """Import the Qt classes (thread wrapper, status display) only when requested."""
    if name == "CameraPreviewThread":
        from .thread import CameraPreviewThread
        return CameraPreviewThread
    if name == "CorrectionStatusDisplay":
        from .widget import CorrectionStatusDisplay
        return CorrectionStatusDisplay
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
