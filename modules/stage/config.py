"""Settings the scan stage needs, owned by the stage module."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StageConfig:
    """Stage selection and the scan move the window starts with.

    Attributes:
        stage_type: ``"rotation"`` (degrees) or ``"linear"`` (millimetres).
        rotation_degree: Rotation scan angle (degrees).
        rotation_speed_dps: Rotation speed (degrees per second).
        linear_distance_mm: Linear scan distance (mm).
        linear_speed_um_s: Linear speed (micrometres per second).
        scan_trim_extra_ms: Extra time (ms) cut off both ends of the recorded scan, on
            top of the stage's acceleration and deceleration times (optional, default 0;
            only with a camera that timestamps its frames, GenICam).
        stage_device: Stage controller to open (optional): a device path such as
            ``/dev/serial/by-id/usb-...-if00`` (Linux) or a libximc URI. Empty = the device
            libximc enumerates, or on Linux a ``/dev/serial/by-id`` link naming XIMC.

    The four scan values are optional in ``scan.json``; the application writes them back
    when the user changes them, so the next start uses them.
    """

    stage_type: str = "rotation"
    rotation_degree: float = 10.0
    rotation_speed_dps: float = 1.0
    linear_distance_mm: float = 10.0
    linear_speed_um_s: float = 1000.0
    scan_trim_extra_ms: float = 0.0
    stage_device: str = ""
