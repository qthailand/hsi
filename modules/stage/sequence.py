"""Qt-free stage motion sequence: open -> home -> scan -> return home -> close.

Cube timing follows v1: ``on_after`` is called as soon as the scan move ends (at the
endpoint, or where a manual stop interrupted it), so the caller can build the cube
immediately. Return-home runs afterwards and is reported separately; a return-home
failure never invalidates scan data already published.

A manual stop (:meth:`StageSequence.request_stop`) interrupts the scan move but still
returns the stage home, as in v1. ``request_stop(return_home=False)`` aborts without
homing and is meant for application shutdown only.

Threading:
    :meth:`StageSequence.run` blocks and must run on a worker thread (the Qt wrapper
    :class:`modules.stage.StageThread` does this). :meth:`StageSequence.request_stop`
    may be called from any thread.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

#: Speed of the homing move before a scan, in controller full steps per second, for both
#: stages. The stage may start far away (e.g. the app was closed before it returned home).
#: Linear: 500 steps/s = 10 mm/s (0.02 mm per step; the driver caps raw speed at 1000).
#: Rotation: 500 steps/s = 10 deg/s (0.02 deg per step; the driver allows up to 20 deg/s).
HOMING_SPEED_STEPS_S = 500
#: Degrees per full step of the rotation stage (``Stage8SMC1.DEGREEPERSTEP``; smc1 needs the SDK).
_ROTATION_DEG_PER_STEP = 0.02


class StageSequence:
    """One scan: open the stage, home it, move through the scan, return home, close.

    Attributes:
        degree_per_second: Rotation speed during the scan (rotation stage).
        degree: Rotation angle of the scan move (rotation stage).
        distance_mm: Travel of the scan move (linear stage).
        speed_um_s: Speed during the scan (linear stage, micrometres per second).
        phase: One of ``IDLE, OPENING, HOMING, SCANNING, RETURNING, CLOSING``.
        failed: A hardware or Python error occurred (see ``failed_phase``).
        failed_phase: Phase in which the first failure happened, else ``None``.
        cancelled: A stop was requested.
        scan_finished: ``on_after`` was called (the scan move ended).
        motion_start_ns: ``time.perf_counter_ns()`` just before ``on_before`` and the scan
            move command, ``None`` until then or if a stop cancelled the move.
        motion_end_ns: ``time.perf_counter_ns()`` when the scan move ended (the driver
            polls the stop status every 10 ms), ``None`` until then or if it failed.
        ramp_s: ``(accel_s, decel_s)`` the controller takes to reach the scan speed and
            to stop from it, read when the scan speed is set (``(0, 0)`` if the driver
            cannot tell).
        trim_extra_ms: Extra time cut off both ends of the constant-speed window
            (:meth:`steady_window`), for command latency.
        device: Stage device path or libximc URI for the default drivers; empty = find it
            (``stage_device`` in ``config/scan.json``).
        completed: The whole sequence, including return-home, succeeded.

    Args:
        stage_type: ``"rotation"`` or ``"linear"``.
        stage_factory: Zero-argument callable returning an unopened stage driver
            (see :mod:`modules.stage.smc1`); defaults to the XIMC SMC1 drivers.
        on_before: Called just before the scan move (start recording).
        on_after: Called when the scan move ended (build the cube).
        on_home: Called after a successful return-home.
        on_error: Called with the error message of any failure.
    """

    IDLE, OPENING, HOMING, SCANNING, RETURNING, CLOSING = (
        'idle', 'opening', 'homing', 'scanning', 'returning', 'closing')

    def __init__(self, stage_type: str = 'rotation', stage_factory: Optional[Callable] = None,
                 on_before: Optional[Callable[[], None]] = None,
                 on_after: Optional[Callable[[], None]] = None,
                 on_home: Optional[Callable[[], None]] = None,
                 on_error: Optional[Callable[[str], None]] = None):
        self.stage_type = str(stage_type).lower()
        self.degree_per_second = 1
        self.degree = 10
        self.distance_mm = 10.0
        self.speed_um_s = 1000.0
        self._factory = stage_factory
        self._callbacks = (on_before, on_after, on_home, on_error)
        self._stage_ref = None
        self._lifecycle_lock = threading.Lock()
        self._stop = threading.Event()      # cancel the remaining scan
        self._abort = threading.Event()     # also skip return-home (shutdown)
        self.phase = self.IDLE
        self.failed = False
        self.failed_phase = None
        self.cancelled = False
        self.scan_finished = False          # on_after was called
        self.motion_start_ns = None
        self.motion_end_ns = None
        self.ramp_s = (0.0, 0.0)
        self.trim_extra_ms = 0.0
        self.device = ""
        self.completed = False              # whole sequence incl. return-home succeeded

    # ── notifications (overridden by the Qt wrapper to emit signals) ──────

    def _notify_before(self):
        """Report that the scan move is about to start."""
        if self._callbacks[0]:
            self._callbacks[0]()

    def _notify_after(self):
        """Report that the scan move ended."""
        if self._callbacks[1]:
            self._callbacks[1]()

    def _notify_home(self):
        """Report a successful return-home."""
        if self._callbacks[2]:
            self._callbacks[2]()

    def _notify_error(self, message: str):
        """Report a failure message."""
        if self._callbacks[3]:
            self._callbacks[3](message)

    # ── control (any thread) ─────────────────────────────

    def request_stop(self, return_home: bool = True):
        """Interrupt the scan move.

        The stage still returns home unless ``return_home`` is False. A stop during
        return-home is ignored unless aborting, so pressing Stop twice cannot strand
        the stage.

        Args:
            return_home: ``False`` aborts without homing (application shutdown only).
        """
        self._stop.set()
        if not return_home:
            self._abort.set()
        if self.phase == self.RETURNING and return_home:
            return
        if not self._lifecycle_lock.acquire(blocking=False):
            return  # opening/closing; the run loop checks the flags afterwards
        try:
            stage = self._stage_ref
            if stage is not None and getattr(stage, '_opened', False):
                try:
                    stage._axis.command_stop()
                except Exception:
                    logger.exception('Stage stop command failed')
        finally:
            self._lifecycle_lock.release()

    def steady_window(self):
        """Host-time bounds of the constant-speed part of the scan move.

        ``motion_start_ns`` plus the acceleration time and ``trim_extra_ms``, and
        ``motion_end_ns`` minus the deceleration time and ``trim_extra_ms``; a bound not
        known yet stays ``None``.  A move too short to reach constant speed keeps the
        whole motion window.

        Returns:
            tuple: ``(start_ns, end_ns)``, either ``None``.
        """
        start, end = self.motion_start_ns, self.motion_end_ns
        accel_s, decel_s = self.ramp_s
        extra_ns = int(self.trim_extra_ms * 1e6)
        if start is not None:
            start += int(accel_s * 1e9) + extra_ns
        if end is not None:
            end -= int(decel_s * 1e9) + extra_ns
        if start is not None and end is not None and start >= end:
            logger.warning('Scan move too short to reach constant speed; ramps not trimmed')
            return self.motion_start_ns, self.motion_end_ns
        return start, end

    # ── motion helpers (worker thread) ───────────────────

    def _set_scan_speed(self, stage):
        if self.stage_type == 'linear':
            stage.set_speed_mm_s(self.speed_um_s / 1000.0)
        else:
            stage.set_speed_dps(self.degree_per_second)

    def _read_ramp_times(self, stage):
        ramp_times = getattr(stage, 'ramp_times_s', None)
        try:
            self.ramp_s = tuple(float(t) for t in ramp_times()) if ramp_times else (0.0, 0.0)
        except Exception:
            logger.exception('Could not read the stage acceleration; ramps not trimmed')
            self.ramp_s = (0.0, 0.0)
        logger.info('Stage ramps: accel %.3f s, decel %.3f s, extra %.1f ms',
                    self.ramp_s[0], self.ramp_s[1], self.trim_extra_ms)

    def _scan_move(self, stage):
        if self.stage_type == 'linear':
            stage.move_le_motor_mm(self.distance_mm)
        else:
            stage.set_degree(self.degree)

    def _fail(self, exc):
        self.failed = True
        self.failed_phase = self.phase
        logger.exception('Stage %s failed', self.phase)
        self._notify_error(str(exc))

    # ── sequence ─────────────────────────────────────────

    def run(self):
        """Execute the sequence (blocking; call from a worker thread).

        Never raises: failures set ``failed``/``failed_phase`` and are reported through
        ``on_error``. The stage is always closed. Homing before the scan, the scan
        itself and the return-home are skipped according to stop/abort requests and
        to the phase in which a failure occurred.
        """
        stage = None
        moved = False
        try:
            if self.stage_type not in ('linear', 'rotation'):
                raise ValueError(f'Unsupported stage type: {self.stage_type}')
            factory = self._factory
            if factory is None:
                from .smc1 import Stage8SMC1, Stage8SMC1Linear
                driver = Stage8SMC1Linear if self.stage_type == 'linear' else Stage8SMC1
                # No device configured: the driver finds it, created exactly as before.
                factory = (lambda: driver(device=self.device)) if self.device else driver
            self.phase = self.OPENING
            stage = factory()
            with self._lifecycle_lock:
                self._stage_ref = stage
                stage.open()

            # 1) home before scanning
            if not self._stop.is_set():
                self.phase = self.HOMING
                if self.stage_type == 'linear':
                    stage.set_speed(HOMING_SPEED_STEPS_S)
                else:
                    stage.set_speed_dps(HOMING_SPEED_STEPS_S * _ROTATION_DEG_PER_STEP)
                moved = True
                stage.set_position(0, 0)

            # 2) scan; on_after is always called once the move ends so the
            #    caller builds the cube from what was recorded (v1 behavior)
            if not self._stop.is_set():
                self.phase = self.SCANNING
                self._set_scan_speed(stage)
                self._read_ramp_times(stage)
                # Noted before on_before, so its handler can already use steady_window().
                self.motion_start_ns = time.perf_counter_ns()
                self._notify_before()
                if not self._stop.is_set():   # stop may arrive between checks
                    self._scan_move(stage)
                    self.motion_end_ns = time.perf_counter_ns()
                else:
                    self.motion_start_ns = None   # the move never started
                self.scan_finished = True
                self._notify_after()
        except Exception as exc:
            self._fail(exc)

        # 3) return home after a normal end, a manual stop, or a failed scan
        #    move. Skipped on shutdown abort, if nothing moved, or if the
        #    failure happened while opening/homing (homing itself is suspect).
        try:
            scan_reached = self.scan_finished or self.failed_phase == self.SCANNING
            if (stage is not None and moved and not self._abort.is_set()
                    and (not self.failed or scan_reached)):
                self.phase = self.RETURNING
                if scan_reached:
                    self._set_scan_speed(stage)   # v1: return at scan speed
                stage.set_position(0, 0)
                if not self.failed:
                    self._notify_home()
        except Exception as exc:
            self._fail(exc)
        finally:
            self.cancelled = self._stop.is_set()
            self.phase = self.CLOSING
            with self._lifecycle_lock:
                try:
                    if stage is not None:
                        stage.close()
                except Exception as exc:
                    self._fail(exc)
                finally:
                    self._stage_ref = None
            self.completed = self.scan_finished and not self.failed
            self.phase = self.IDLE
