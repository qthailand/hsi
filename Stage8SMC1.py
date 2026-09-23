#Stage8SMC1.py
import libximc.highlevel as ximc
from ctypes import *
import time
import math

# FULL STEP 0.02°
# MIN 1/8 STEP 0.0025°


class Stage8SMC1:

    STEPPERSTAGE = 18000
    DEGREEPERSTEP = 0.02

    def __init__(self, open_on_init=False):
        self.device_id = None
        self._opened = False
        self._enum = None

        self.degree_per_step = 0.02

        if open_on_init:
            self.open()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def open(self):
        if self._opened:
            return

        print("Library loaded successfully")
        devices = ximc.enumerate_devices(ximc.EnumerateFlags.ENUMERATE_PROBE)
        if len(devices) == 0:
            raise RuntimeError("open_device failed")
        else:
            device_uri = devices[0]['uri']
            self._axis = ximc.Axis(device_uri)
            self._axis.open_device()
            version = ximc.ximc_version()

            self._opened = True
            print("Opened:", version)
            self.sync_engine()
            self.set_engine()
            self.sync_speed()
            self.sync_position()

    def close(self):
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
        try:
            self.close()
        except Exception:
            pass

    # ---------- helpers ----------
    def _ensure_open(self):
        if self._axis is None:
            raise RuntimeError(
                "Device not opened. Use with LinearStage8SMC1() as st: or call open().")

    # ---------- core ----------
    def zero(self):
        self._ensure_open()
        self._axis.command_zero()

    def sync_engine(self):
        self._ensure_open()
        self._engine = self._axis.get_engine_settings()
        print(self._engine)

    def set_engine(self):
        self._ensure_open()
        if self._engine is None:
            raise RuntimeError(f"get_engine_settings failed")
        self._engine.MicrostepMode = ximc.MicrostepMode.MICROSTEP_MODE_FRAC_2
        self._axis.set_engine_settings(self._engine)
        self._engine = self._axis.get_engine_settings()
        print(self._engine)

    def sync_position(self):
        self._ensure_open()
        self._positions = self._axis.get_position()
        if self._positions is None:
            raise RuntimeError(f"get_position failed")

        self.Position = self._positions.Position
        self.uPosition = self._positions.uPosition

    def sync_speed(self):
        self._ensure_open()
        self._mvst = self._axis.get_move_settings()
        if self._mvst is None:
            raise RuntimeError(f"get_move_settings failed")

    def set_speed(self, speed):
        self._ensure_open()
        speed = min(int(speed), 1000)
        self._mvst.Speed = speed
        self._axis.set_move_settings(self._mvst)

    def set_position(self, position, uPosition=0, wait_timeout_ms=100):
        self._ensure_open()
        self._axis.command_move(int(position), int(uPosition))
        self._axis.command_wait_for_stop(wait_timeout_ms)
        self.sync_position()

    def set_position_r(self, position, uPosition=0, wait_timeout_ms=100):
        self._ensure_open()
        result = self._axis.command_movr(int(position), int(uPosition))
        self._axis.command_wait_for_stop(wait_timeout_ms)
        self.sync_position()

    def set_degree(self, degree):
        steps = degree/self.DEGREEPERSTEP
        self.set_position(steps)

    def set_speed_dps(self, degree_per_second: float,
                      accel_dps2: float | None = None,
                      decel_dps2: float | None = None) -> None:

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





class Stage8SMC1Linear:
    # length = 310mm
    # 1000 step = 2 cm = 20 mm
    STEPS_PER_MM = 50.0
    MM_PER_STEP = 1.0 / STEPS_PER_MM  # 0.02 mm/step
    def __init__(self, open_on_init=False):
        self.device_id = None
        self._opened = False
        self._enum = None

        # conversion
        self.steps_per_mm = self.STEPS_PER_MM
        self.mm_per_step = self.MM_PER_STEP

        if open_on_init:
            self.open()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def open(self):
        if self._opened:
            return

        print("Library loaded successfully")
        devices = ximc.enumerate_devices(ximc.EnumerateFlags.ENUMERATE_PROBE)
        if len(devices) == 0:
            raise RuntimeError("open_device failed")
        else:
            device_uri = devices[0]['uri']
            self._axis = ximc.Axis(device_uri)
            self._axis.open_device()
            version = ximc.ximc_version()

            self._opened = True
            print("Opened:", version)
            self.sync_speed()
            self.sync_position()

    def close(self):
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
        try:
            self.close()
        except Exception:
            pass

    # ---------- helpers ----------
    def _ensure_open(self):
        if self._axis is None:
            raise RuntimeError(
                "Device not opened. Use with LinearStage8SMC1() as st: or call open().")

    def _mm_to_steps(self, mm: float) -> int:
        # round to nearest integer step
        return int(round(mm * self.steps_per_mm))

    def _steps_to_mm(self, steps: int) -> float:
        return steps * self.mm_per_step

    # ---------- core ----------
    def zero(self):
        self._ensure_open()
        self._axis.command_zero()

    def sync_position(self):
        self._ensure_open()
        self._positions = self._axis.get_position()
        if self._positions is None:
            raise RuntimeError(f"get_position failed")

        self.Position = self._positions.Position
        self.uPosition = self._positions.uPosition

        print(self._positions)

    def sync_speed(self):
        self._ensure_open()
        self._mvst = self._axis.get_move_settings()
        if self._mvst is None:
            raise RuntimeError(f"get_move_settings failed")
        print("Origial Move Settings")
        print(self._mvst)

    def set_speed(self, speed):
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

    def set_position(self, position, uPosition=0, wait_timeout_ms=100):
        self._ensure_open()
        self._axis.command_move(int(position), int(uPosition))
        self._axis.command_wait_for_stop(wait_timeout_ms)
        self.sync_position()

    def set_position_r(self, position, uPosition=0, wait_timeout_ms=100):
        self._ensure_open()
        result = self._axis.command_movr(int(position), int(uPosition))
        self._axis.command_wait_for_stop(wait_timeout_ms)
        self.sync_position()

    # ---------- mm API ----------
    def move_to_mm(self, mm: float, wait_timeout_ms=100) -> bool:
        """Absolute move to position in mm."""
        steps = self._mm_to_steps(mm)
        return self.set_position(steps, 0, wait_timeout_ms=wait_timeout_ms)

    def move_by_mm(self, mm: float, wait_timeout_ms=100) -> bool:
        """Relative move by mm (+ forward, - backward)."""
        steps = self._mm_to_steps(mm)
        return self.set_position_r(steps, 0, wait_timeout_ms=wait_timeout_ms)

    def move_to_motor_mm(self, mm: float, wait_timeout_ms=100) -> bool:
        """Your forward direction wrapper."""
        return self.move_by_mm(+mm, wait_timeout_ms=wait_timeout_ms)

    def move_le_motor_mm(self, mm: float, wait_timeout_ms=100) -> bool:
        """Your reverse direction wrapper."""
        return self.move_by_mm(-mm, wait_timeout_ms=wait_timeout_ms)

    def set_speed_mm_s(self, mm_s: float):
        """
        Convenience: set speed in mm/s assuming Speed unit is steps/s.
        With your calibration: 50 steps/mm => steps/s = mm/s * 50.
        """
        steps_s = mm_s * self.steps_per_mm
        self.set_speed(steps_s)

