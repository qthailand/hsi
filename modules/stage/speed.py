"""Speed resolution of the XIMC controller, without the vendor SDK.

The controller takes a speed as whole full steps per second (``Speed``) plus a fraction
in microsteps per second (``uSpeed``) of the engine's microstep mode. :func:`encode_speed`
picks the nearest pair, so a stage runs within half a microstep per second of the request
instead of losing the fraction (150 um/s on the 50 steps/mm linear stage is 7.5 steps/s:
``Speed`` 7, ``uSpeed`` 128 at 1/256 microsteps; before, it ran at 7 steps/s = 140 um/s).

Data conventions:
    Speeds are full steps per second. Microsteps per full step are
    ``2 ** (MicrostepMode - 1)`` (mode 1 = full step, 2 = 1/2 ... 9 = 1/256).

Example:
    >>> encode_speed(7.5, 256), encode_speed(1.5, 2), decode_speed(7, 128, 256)
    ((7, 128), (1, 1), 7.5)
"""
from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

#: Fastest speed the drivers command, in full steps per second (linear 20 mm/s, rotation
#: 20 deg/s); a faster request is clamped with a warning.
MAX_SPEED_STEPS_S = 1000
#: Degrees per full step of the rotation stage.
ROTATION_DEG_PER_STEP = 0.02
#: Microsteps per full step on the rotation stage: its driver sets ``MICROSTEP_MODE_FRAC_2``.
ROTATION_MICROSTEPS = 2
#: Smallest rotation speed change the stage can run (half a step per second), deg/s.
ROTATION_SPEED_RESOLUTION_DPS = ROTATION_DEG_PER_STEP / ROTATION_MICROSTEPS
#: Fastest rotation speed, deg/s.
ROTATION_MAX_DPS = MAX_SPEED_STEPS_S * ROTATION_DEG_PER_STEP
#: Full steps per millimetre of the linear stage (1 step = 20 um).
LINEAR_STEPS_PER_MM = 50.0
#: Fastest linear speed, um/s.
LINEAR_MAX_UM_S = MAX_SPEED_STEPS_S / LINEAR_STEPS_PER_MM * 1000.0


def microsteps_per_step(microstep_mode) -> int:
    """Microsteps per full step of an engine ``MicrostepMode`` (1 = full step ... 9 = 1/256)."""
    return 2 ** max(int(microstep_mode) - 1, 0)


def encode_speed(steps_s: float, microsteps: int,
                 max_steps_s: float = MAX_SPEED_STEPS_S) -> tuple[int, int]:
    """``(Speed, uSpeed)`` closest to ``steps_s``.

    The speed is rounded to the nearest microstep per second (halves up), so a fraction
    that rounds up to a whole step carries into ``Speed`` (7.999 steps/s at 1/2 microsteps
    is ``(8, 0)``). A speed above ``max_steps_s`` is clamped to it, and a positive speed
    below one microstep per second is raised to one microstep (a zero speed never ends a
    move); both log a warning.

    Args:
        steps_s: Requested full steps per second (not negative).
        microsteps: Microsteps per full step of the engine (:func:`microsteps_per_step`).
        max_steps_s: Fastest speed to command, full steps per second.

    Returns:
        tuple: ``(Speed, uSpeed)`` ints, ``0 <= uSpeed < microsteps``.

    Raises:
        ValueError: ``steps_s`` is negative.
    """
    if steps_s < 0:
        raise ValueError("speed must be non-negative")
    microsteps = max(int(microsteps), 1)
    total = math.floor(steps_s * microsteps + 0.5)
    limit = math.floor(max_steps_s * microsteps)
    if total > limit:
        logger.warning("Stage speed %.6g steps/s is above the limit %.6g steps/s; clamped",
                       steps_s, max_steps_s)
        total = limit
    elif total == 0 and steps_s > 0:
        logger.warning("Stage speed %.6g steps/s is below one microstep (1/%d step) per second; "
                       "raised to it", steps_s, microsteps)
        total = 1
    return divmod(total, microsteps)


def decode_speed(speed: int, u_speed: int, microsteps: int) -> float:
    """Full steps per second of a ``(Speed, uSpeed)`` pair."""
    return speed + u_speed / max(int(microsteps), 1)
