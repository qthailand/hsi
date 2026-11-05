"""Settings the scan stage needs, owned by the stage module."""
from __future__ import annotations

from dataclasses import dataclass

#: Scan types of ``stage_type``: two motorised stages, and ``"free"`` - no stage, the scan
#: records for a set time (``free_duration_s``) while the scene moves past the slit by
#: itself (conveyor, vehicle, hand-held sweep...).
STAGE_TYPES = ("rotation", "linear", "free")
#: Range of the free scan time the scan window takes (seconds; up to 24 h).
FREE_SCAN_MIN_S = 0.1
FREE_SCAN_MAX_S = 86400.0


@dataclass
class StageConfig:
    """Stage selection and the scan move the window starts with.

    Attributes:
        stage_type: ``"rotation"`` (degrees), ``"linear"`` (millimetres) or ``"free"``
            (no stage: a timed scan of ``free_duration_s`` seconds); see :data:`STAGE_TYPES`.
        rotation_degree: Rotation scan angle (degrees).
        rotation_speed_dps: Rotation speed (degrees per second).
        linear_distance_mm: Linear scan distance (mm).
        linear_speed_um_s: Linear speed (micrometres per second).
        free_duration_s: Recording time of a free scan (seconds; no stage is opened).
        scan_trim_extra_ms: Extra time (ms) cut off both ends of the recorded scan, on
            top of the stage's acceleration and deceleration times (optional, default 0;
            only with a camera that timestamps its frames, GenICam; not applied to a free
            scan, which has no stage command to wait for).
        stage_device: Stage controller to open (optional): a device path such as
            ``/dev/serial/by-id/usb-...-if00`` (Linux) or a libximc URI. Empty = the device
            libximc enumerates, or on Linux a ``/dev/serial/by-id`` link naming XIMC.

    The five scan values are optional in ``scan.json``; the application writes them back
    when the user changes them, so the next start uses them.
    """

    stage_type: str = "rotation"
    rotation_degree: float = 10.0
    rotation_speed_dps: float = 1.0
    linear_distance_mm: float = 10.0
    linear_speed_um_s: float = 1000.0
    free_duration_s: float = 10.0
    scan_trim_extra_ms: float = 0.0
    stage_device: str = ""
