"""Camera acquisition backends behind one contract (POA, GenICam, XIMEA).

Every backend implements :class:`AbstractCameraBackend`; applications select one by
name with :func:`create_backend` and never import vendor SDKs directly. SDKs are
imported only when their backend is requested, so a missing SDK for an unused camera
does not matter.

Data conventions:
    A frame is a 2-D ``uint16`` array ``(rows, columns) = (slit position, spectral
    pixel)``. ``read_frame_raw`` returns the untouched sensor frame; ``read_frame``
    additionally applies dark subtraction and hot-pixel repair when a dark reference
    is set (see :mod:`modules.dark`). Exposure is given in milliseconds by callers;
    ranges are reported in microseconds (``exposure_range_us``).

Threading:
    One thread (the acquisition worker) owns the streaming calls
    (``start_stream``/``stop_stream``/``read_frame*``/``set_params``/``set_bin``).
    ``info``, ``current_bin`` and the range properties are plain reads and may be
    used from the GUI thread.

Example:
    >>> from types import SimpleNamespace
    >>> cfg = SimpleNamespace(camera_backend='poa', default_gain=100, default_exposure_ms=10)
    >>> # camera = create_backend(cfg); camera.open(); camera.start_stream()  # doctest: +SKIP
    >>> # frame = camera.read_frame_raw(timeout_ms=1000)                       # doctest: +SKIP
"""
import importlib

from .base import AbstractCameraBackend, CameraBackendInfo
from .config import CameraConfig
from .factory import BACKENDS, create_backend

__all__ = [
    "CameraConfig",
    "AbstractCameraBackend",
    "CameraBackendInfo",
    "create_backend",
    "BACKENDS",
    "POACameraBackend",
    "GenICamBackend",
    "XIMEACameraBackend",
    "CameraControls",
]
__requires__ = ("dark",)
# CameraControls is a QGroupBox (PyQt5, loaded lazily); constructible with no arguments.
__widgets__ = ("CameraControls",)

_LAZY = {
    "POACameraBackend": ".poa_camera",
    "GenICamBackend": ".genicam",
    "XIMEACameraBackend": ".ximea_camera",
    "CameraControls": ".widget",
}


def __getattr__(name):
    """Import a backend class (and its SDK) or the Qt widget on first access."""
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
