from .base import AbstractCameraBackend, CameraBackendInfo

# Each backend SDK is loaded lazily so a missing/incompatible SDK
# for an unused backend does not prevent startup.
try:
    from .genicam import GenICamBackend
except Exception:
    GenICamBackend = None

try:
    from .ximea_camera import XIMEACameraBackend
except Exception:
    XIMEACameraBackend = None

try:
    from .poa_camera import POACameraBackend
except Exception:
    POACameraBackend = None

__all__ = [
    "AbstractCameraBackend",
    "CameraBackendInfo",
    "GenICamBackend",
    "POACameraBackend",
    "XIMEACameraBackend",
]
