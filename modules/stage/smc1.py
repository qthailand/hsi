"""XIMC (Standa 8SMC1) stage drivers: rotation (`Stage8SMC1`) and linear (`Stage8SMC1Linear`).

Imports the vendor library ``libximc``; import this module only on a machine that has it.
Full step 0.02 deg, minimum 1/8 step 0.0025 deg.

Data conventions:
    Rotation stage: positions in steps (``Position``) plus microsteps (``uPosition``),
    angles in degrees, speeds in degrees per second (``set_speed_dps``) or raw
    controller steps per second (``set_speed``). Linear stage: positions in steps
    (50 steps per mm), distances in mm, speed in mm/s. After every move ``Position`` and
    ``uPosition`` hold the controller's reported position.

Threading:
    A driver is not thread-safe. Motion calls block until the stage stops; to interrupt
    from another thread, call ``_axis.command_stop()`` (as ``StageSequence.request_stop``
    does).

Exceptions:
    Every method except ``open``/``close`` raises ``RuntimeError`` if the stage is not
    open; libximc errors propagate unchanged.

Example:
    >>> # with Stage8SMC1() as stage:        # doctest: +SKIP
    >>> #     stage.set_speed_dps(5)          # doctest: +SKIP
    >>> #     stage.set_degree(10)            # doctest: +SKIP
"""
import libximc.highlevel as ximc
from ctypes import *
import time
import math

# FULL STEP 0.02°
# MIN 1/8 STEP 0.0025°


class Stage8SMC1:
    """Rotation stage on a Standa 8SMC1 controller (first device found).

    Class attributes:
        STEPPERSTAGE: Full steps per revolution of the stage (18000).
        DEGREEPERSTEP: Degrees per full step (0.02).
    """

    STEPPERSTAGE = 18000
    DEGREEPERSTEP = 0.02

    def __init__(self, open_on_init=False):
        """Create the driver (does not touch the hardware unless asked).

        Args:
            open_on_init: Call ``open()`` immediately.

        Raises:
            RuntimeError: ``open_on_init`` is true and no stage is found.
        """
        self._axis = None
        self.device_id = None
        self._opened = False
        self._enum = None

        self.degree_per_step = 0.02

        if open_on_init:
            self.open()

    def __enter__(self):
        """Open the stage and return ``self``."""
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Close the stage; exceptions from the ``with`` body are not suppressed."""
        self.close()
        return False

    def open(self):
        """Connect to the first XIMC device found and read its settings (no-op if open).

        On the rotation stage this also sets the engine to half-microstep mode
        (``MICROSTEP_MODE_FRAC_2``) and reads move settings and position. Prints progress.
        If any step fails the device is closed again before the error propagates.

        Raises:
            RuntimeError: no XIMC stage detected, or a settings/position read failed.
        """
        if self._opened:
            return

        print("Library loaded successfully")
        devices = ximc.enumerate_devices(ximc.EnumerateFlags.ENUMERATE_PROBE)
        if len(devices) == 0:
            raise RuntimeError("No XIMC stage detected; check stage power and USB connection")
        else:
            device_uri = devices[0]['uri']
            self._axis = ximc.Axis(device_uri)
            self._axis.open_device()
            self._opened = True
            try:
                version = ximc.ximc_version()
                print("Opened:", version)
            except Exception:
                self.close()
                raise
            try:
                self.sync_engine()
                self.set_engine()
                self.sync_speed()
                self.sync_position()
            except Exception:
                self.close()
                raise

    def close(self):
        """Close the device (no-op if not open). The driver can be re-opened afterwards."""
        if not self._opened:
            return
        try:
            self._axis.close_device()
        finally:
            self._axis = None
            self._opened = False
            self._enum = None
            print("closed")

    def __del__(self):
        """Best-effort close; never raises."""
        try:
            self.close()
        except Exception:
            pass

    # ---------- helpers ----------
    def _ensure_open(self):
        """Raise ``RuntimeError`` if the device is not open."""
        if self._axis is None:
            raise RuntimeError(
                "Device not opened. Use with LinearStage8SMC1() as st: or call open().")

    # ---------- core ----------
    def zero(self):
        """Declare the current position to be zero (``command_zero``).

        Raises:
            RuntimeError: not open.
        """
        self._ensure_open()
        self._axis.command_zero()

    def sync_engine(self):
        """Read the engine settings into the driver (and print them).

        Raises:
            RuntimeError: not open.
        """
        self._ensure_open()
        self._engine = self._axis.get_engine_settings()
        print(self._engine)

    def set_engine(self):
        """Switch the engine to half-microstep mode and read the settings back.

        Raises:
            RuntimeError: not open or the engine settings were never read.
        """
        self._ensure_open()
        if self._engine is None:
            raise RuntimeError(f"get_engine_settings failed")
        self._engine.MicrostepMode = ximc.MicrostepMode.MICROSTEP_MODE_FRAC_2
        self._axis.set_engine_settings(self._engine)
        self._engine = self._axis.get_engine_settings()
        print(self._engine)

    def sync_position(self):
        """Read the controller position into ``Position`` (steps) and ``uPosition`` (microsteps).

        Raises:
            RuntimeError: not open, or the controller returned no position.
        """
        self._ensure_open()
        self._positions = self._axis.get_position()
        if self._positions is None:
            raise RuntimeError(f"get_position failed")

        self.Position = self._positions.Position
        self.uPosition = self._positions.uPosition

    def sync_speed(self):
        """Read the controller's move settings (speed, acceleration) into the driver.

        Raises:
            RuntimeError: not open, or the controller returned no move settings.
        """
        self._ensure_open()
        self._mvst = self._axis.get_move_settings()
        if self._mvst is None:
            raise RuntimeError(f"get_move_settings failed")

    def set_speed(self, speed):
        """Set the move speed in raw controller units (steps per second).

        Args:
            speed: Steps per second; truncated to int and capped at 1000.

        Raises:
            RuntimeError: not open (or move settings were never read).
        """
        self._ensure_open()
        speed = min(int(speed), 1000)
        self._mvst.Speed = speed
        self._axis.set_move_settings(self._mvst)

    def set_position(self, position, uPosition=0, wait_timeout_ms=10):
        """Move to an absolute position and block until the stage stops.

        Args:
            position: Target in full steps (truncated to int). Rotation: 0.02 deg per
                step; linear: 0.02 mm per step.
            uPosition: Target microsteps added to ``position``.
            wait_timeout_ms: Argument passed to ``command_wait_for_stop`` (milliseconds).

        Raises:
            RuntimeError: not open, or the position read-back failed.
        """
        self._ensure_open()
        self._axis.command_move(int(position), int(uPosition))
        self._axis.command_wait_for_stop(wait_timeout_ms)
        self.sync_position()

    def set_position_r(self, position, uPosition=0, wait_timeout_ms=10):
        """Move by a relative distance and block until the stage stops.

        Args:
            position: Offset in full steps (signed, truncated to int).
            uPosition: Offset microsteps.
            wait_timeout_ms: Argument passed to ``command_wait_for_stop`` (milliseconds).

        Raises:
            RuntimeError: not open, or the position read-back failed.
        """
        self._ensure_open()
        result = self._axis.command_movr(int(position), int(uPosition))
        self._axis.command_wait_for_stop(wait_timeout_ms)
        self.sync_position()

    def set_degree(self, degree):
        """Rotate to an absolute angle and block until the stage stops.

        Args:
            degree: Target angle in degrees from the zero position (converted with
                ``DEGREEPERSTEP``; the step count is truncated to an integer).

        Raises:
            RuntimeError: not open.
        """
        steps = degree/self.DEGREEPERSTEP
        self.set_position(steps)

    def set_speed_dps(self, degree_per_second: float,
                      accel_dps2: float | None = None,
                      decel_dps2: float | None = None) -> None:

        """Set the rotation speed and optionally the acceleration/deceleration.

        Args:
            degree_per_second: Speed in degrees per second, 0 to 20 inclusive. The
                whole-step part is written to ``Speed``; the microstep fraction is not
                applied (``uSpeed`` is set to 0).
            accel_dps2: Acceleration in degrees per second squared, or ``None`` to keep it.
            decel_dps2: Deceleration in degrees per second squared, or ``None`` to keep it.

        Raises:
            RuntimeError: not open, or engine/move settings were never read.
            ValueError: a value is negative, or ``degree_per_second`` exceeds 20 (the
                message wording says "higher than 20" but the check rejects values above 20).
        """
        self._ensure_open()
        if self._engine is None:
            raise RuntimeError(f"get_engine_settings failed")
        if self._mvst is None:
            raise RuntimeError(f"get_move_settings failed")
        if degree_per_second < 0:
            raise ValueError("degree_per_second must be non-negative")
        if degree_per_second > 20:
            raise ValueError("degree_per_second must be higher than 20")

        steps_per_rev = self.STEPPERSTAGE
        if steps_per_rev <= 0:
            raise RuntimeError("Invalid StepsPerRev from controller")

        # 2) แปลง deg/s -> steps/s
        speed_steps_per_s = degree_per_second / 360.0 * steps_per_rev

        fullSpeed = math.floor(speed_steps_per_s)
        micrSpeed = round((speed_steps_per_s - fullSpeed) * 256)

        print("Fraction Speed", speed_steps_per_s)
        print("Speed", fullSpeed)
        print("Micro Speed", micrSpeed)
        # Speed ใช้หน่วย full-steps/s
        self._mvst.Speed = fullSpeed
        self._mvst.uSpeed = 0

        if accel_dps2 is not None:
            if accel_dps2 < 0:
                raise ValueError("accel_dps2 must be non-negative")
            # Accel ใน libximc ใช้หน่วย steps/s^2
            self._mvst.Accel = int(round(accel_dps2 / 360.0 * steps_per_rev))

        if decel_dps2 is not None:
            if decel_dps2 < 0:
                raise ValueError("decel_dps2 must be non-negative")
            # Decel ใน libximc ใช้หน่วย steps/s^2
            self._mvst.Decel = int(round(decel_dps2 / 360.0 * steps_per_rev))

        self._axis.set_move_settings(self._mvst)
        print("moving Speed")
        print(self._mvst)

    def ramp_times_s(self):
        """Seconds to reach the set speed and to stop from it; see :func:`_ramp_times_s`."""
        self._ensure_open()
        return _ramp_times_s(self._axis, self._mvst)





class Stage8SMC1Linear:
    # length = 310mm
    # 1000 step = 2 cm = 20 mm
    """Linear stage on a Standa 8SMC1 controller (first device found).

    Class attributes:
        STEPS_PER_MM: Steps per millimetre (50, i.e. 0.02 mm per step); travel is 310 mm.
        MM_PER_STEP: Millimetres per step.
    """

    STEPS_PER_MM = 50.0
    MM_PER_STEP = 1.0 / STEPS_PER_MM  # 0.02 mm/step
    def __init__(self, open_on_init=False):
        """Create the driver (does not touch the hardware unless asked).

        Args:
            open_on_init: Call ``open()`` immediately.

        Raises:
            RuntimeError: ``open_on_init`` is true and no stage is found.
        """
        self._axis = None
        self.device_id = None
        self._opened = False
        self._enum = None

        # conversion
        self.steps_per_mm = self.STEPS_PER_MM
        self.mm_per_step = self.MM_PER_STEP

        if open_on_init:
            self.open()

    def __enter__(self):
        """Open the stage and return ``self``."""
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Close the stage; exceptions from the ``with`` body are not suppressed."""
        self.close()
        return False

    def open(self):
        """Connect to the first XIMC device found and read its settings (no-op if open).

        Reads move settings and position; the engine settings are left as configured on
        the controller. Prints progress. If any step fails the device is closed again
        before the error propagates.

        Raises:
            RuntimeError: no XIMC stage detected, or a settings/position read failed.
        """
        if self._opened:
            return

        print("Library loaded successfully")
        devices = ximc.enumerate_devices(ximc.EnumerateFlags.ENUMERATE_PROBE)
        if len(devices) == 0:
            raise RuntimeError("No XIMC stage detected; check stage power and USB connection")
        else:
            device_uri = devices[0]['uri']
            self._axis = ximc.Axis(device_uri)
            self._axis.open_device()
            self._opened = True
            try:
                version = ximc.ximc_version()
                print("Opened:", version)
            except Exception:
                self.close()
                raise
            try:
                self.sync_speed()
                self.sync_position()
            except Exception:
                self.close()
                raise

    def close(self):
        """Close the device (no-op if not open). The driver can be re-opened afterwards."""
        if not self._opened:
            return
        try:
            self._axis.close_device()
        finally:
            self._axis = None
            self._opened = False
            self._enum = None
            print("closed")

    def __del__(self):
        """Best-effort close; never raises."""
        try:
            self.close()
        except Exception:
            pass

    # ---------- helpers ----------
    def _ensure_open(self):
        """Raise ``RuntimeError`` if the device is not open."""
        if self._axis is None:
            raise RuntimeError(
                "Device not opened. Use with LinearStage8SMC1() as st: or call open().")

    def _mm_to_steps(self, mm: float) -> int:
        # round to nearest integer step
        """Convert millimetres to the nearest whole number of steps."""
        return int(round(mm * self.steps_per_mm))

    def _steps_to_mm(self, steps: int) -> float:
        """Convert steps to millimetres."""
        return steps * self.mm_per_step

    # ---------- core ----------
    def zero(self):
        """Declare the current position to be zero (``command_zero``).

        Raises:
            RuntimeError: not open.
        """
        self._ensure_open()
        self._axis.command_zero()

    def sync_position(self):
        """Read the controller position into ``Position`` (steps) and ``uPosition`` (microsteps).

        Prints the position.

        Raises:
            RuntimeError: not open, or the controller returned no position.
        """
        self._ensure_open()
        self._positions = self._axis.get_position()
        if self._positions is None:
            raise RuntimeError(f"get_position failed")

        self.Position = self._positions.Position
        self.uPosition = self._positions.uPosition

        print(self._positions)

    def sync_speed(self):
        """Read the controller's move settings (speed, acceleration) into the driver.

        Prints the settings.

        Raises:
            RuntimeError: not open, or the controller returned no move settings.
        """
        self._ensure_open()
        self._mvst = self._axis.get_move_settings()
        if self._mvst is None:
            raise RuntimeError(f"get_move_settings failed")
        print("Origial Move Settings")
        print(self._mvst)

    def set_speed(self, speed):
        """Set the move speed in raw controller units (steps per second).

        Args:
            speed: Steps per second; truncated to int and capped at 1000.

        Raises:
            RuntimeError: not open (or move settings were never read).
        """
        self._ensure_open()
        speed = min(int(speed), 1000)
        self._mvst.Speed = speed
        self._axis.set_move_settings(self._mvst)
        print("Current Move Settings")
        print(self._mvst)

    def get_position_mm(self) -> float:
        """Return current position in mm (based on Position only)."""
        self.sync_position()
        return self._steps_to_mm(self.Position)

    def set_position(self, position, uPosition=0, wait_timeout_ms=10):
        """Move to an absolute position and block until the stage stops.

        Args:
            position: Target in full steps (truncated to int). Rotation: 0.02 deg per
                step; linear: 0.02 mm per step.
            uPosition: Target microsteps added to ``position``.
            wait_timeout_ms: Argument passed to ``command_wait_for_stop`` (milliseconds).

        Raises:
            RuntimeError: not open, or the position read-back failed.
        """
        self._ensure_open()
        self._axis.command_move(int(position), int(uPosition))
        self._axis.command_wait_for_stop(wait_timeout_ms)
        self.sync_position()

    def set_position_r(self, position, uPosition=0, wait_timeout_ms=10):
        """Move by a relative distance and block until the stage stops.

        Args:
            position: Offset in full steps (signed, truncated to int).
            uPosition: Offset microsteps.
            wait_timeout_ms: Argument passed to ``command_wait_for_stop`` (milliseconds).

        Raises:
            RuntimeError: not open, or the position read-back failed.
        """
        self._ensure_open()
        result = self._axis.command_movr(int(position), int(uPosition))
        self._axis.command_wait_for_stop(wait_timeout_ms)
        self.sync_position()

    # ---------- mm API ----------
    def move_to_mm(self, mm: float, wait_timeout_ms=10) -> bool:
        """Absolute move to position in mm."""
        steps = self._mm_to_steps(mm)
        return self.set_position(steps, 0, wait_timeout_ms=wait_timeout_ms)

    def move_by_mm(self, mm: float, wait_timeout_ms=10) -> bool:
        """Relative move by mm (+ forward, - backward)."""
        steps = self._mm_to_steps(mm)
        return self.set_position_r(steps, 0, wait_timeout_ms=wait_timeout_ms)

    def move_to_motor_mm(self, mm: float, wait_timeout_ms=10) -> bool:
        """Your forward direction wrapper."""
        return self.move_by_mm(+mm, wait_timeout_ms=wait_timeout_ms)

    def move_le_motor_mm(self, mm: float, wait_timeout_ms=10) -> bool:
        """Your reverse direction wrapper."""
        return self.move_by_mm(-mm, wait_timeout_ms=wait_timeout_ms)

    def set_speed_mm_s(self, mm_s: float):
        """
        Convenience: set speed in mm/s assuming Speed unit is steps/s.
        With your calibration: 50 steps/mm => steps/s = mm/s * 50.
        """
        steps_s = mm_s * self.steps_per_mm
        self.set_speed(steps_s)

    def ramp_times_s(self):
        """Seconds to reach the set speed and to stop from it; see :func:`_ramp_times_s`."""
        self._ensure_open()
        return _ramp_times_s(self._axis, self._mvst)


def _ramp_times_s(axis, move_settings):
    """``(accel_s, decel_s)`` of a move at the current move settings.

    ``Speed`` (plus ``uSpeed`` microsteps of the engine's microstep mode), ``Accel`` and
    ``Decel`` are all in steps, so the time to reach the speed is ``Speed / Accel`` and to
    stop ``Speed / Decel``.  Both are 0 when the controller has acceleration off
    (``ENGINE_ACCEL_ON`` clear) or a value is 0.
    """
    engine = axis.get_engine_settings()
    if not int(engine.EngineFlags) & int(ximc.EngineFlags.ENGINE_ACCEL_ON):
        return 0.0, 0.0
    # MicrostepMode 1 = full step, 2 = 1/2, 3 = 1/4 ... 9 = 1/256
    microsteps = 2 ** max(int(engine.MicrostepMode) - 1, 0)
    speed = move_settings.Speed + move_settings.uSpeed / microsteps
    accel, decel = move_settings.Accel, move_settings.Decel
    return (speed / accel if accel > 0 else 0.0), (speed / decel if decel > 0 else 0.0)

