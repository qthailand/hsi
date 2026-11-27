"""Saturation-level measurement through the running acquisition engine (the Scan window's camera).

While the Scan window is open the acquisition worker owns the camera, so the saturation sweep
(:func:`modules.saturation_profile.measure`) cannot drive the backend itself. It goes through the
engine instead:

* :class:`RawTap` - the worker hands copies of the next raw frames to a tap, but only frames read
  once the requested setting (bin, gain, exposure) is applied and nothing is pending, after
  ``skip`` settling frames; the intensity pipeline is not changed (the tap copies, it never
  holds the frame);
* :class:`EngineFrameSource` - the :class:`modules.saturation_profile.FrameSource` on an engine:
  ``apply`` queues the bin / parameters with the engine's own requests and waits until they are
  applied, ``grab`` waits on a tap; :meth:`EngineFrameSource.snapshot` and
  :meth:`EngineFrameSource.restore` put the camera back (bin, gain, exposure) and say whether that
  worked.

The host (the Scan controller) also snapshots and checks the calibration state that follows the
bin (dark reference, Smile / keystone model, wavelength profile) and keeps scans locked until the
restore is confirmed.

Threading:
    :class:`EngineFrameSource` methods block; call them from one measurement thread. The tap is
    filled by the acquisition worker and read through its own lock / event.

Example:
    >>> source = EngineFrameSource(engine, "poa")                     # doctest: +SKIP
    >>> before = source.snapshot()                                     # doctest: +SKIP
    >>> try:                                                           # doctest: +SKIP
    ...     records = measure(source, plan)
    ... finally:
    ...     problems = source.restore(before)
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from modules.saturation_profile.sources import identity_of, nominal_full_scale_of


def _close(a, b) -> bool:
    return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))


class RawTap:
    """Copies of the next raw frames read with one setting (filled by the acquisition worker).

    Args:
        count: Frames wanted.
        target: ``(bin, gain, exposure_ms)`` the frames must be read with (``None``: any).
        skip: Matching frames dropped first (settling).
    """

    def __init__(self, count: int, target: Optional[Tuple[int, float, float]] = None, skip: int = 0):
        self.count = int(count)
        self.target = target
        self.skip = int(skip)
        self.frames: List[np.ndarray] = []
        self.done = threading.Event()
        self._lock = threading.Lock()

    def matches(self, backend) -> bool:
        """Whether ``backend`` is at the tap's setting now."""
        if self.target is None:
            return True
        camera_bin, gain, exposure = self.target
        try:
            return (int(backend.current_bin) == int(camera_bin) and _close(backend.gain, gain)
                    and _close(backend.exposure_ms, exposure))
        except Exception:                                      # noqa: BLE001 - unreadable: not yet
            return False

    def offer(self, frame, backend) -> bool:
        """Worker: take a copy of ``frame`` when it fits; returns True once the tap is full."""
        if self.done.is_set() or not self.matches(backend):
            return self.done.is_set()
        with self._lock:
            if self.skip > 0:
                self.skip -= 1
                return False
            self.frames.append(np.array(frame, copy=True))
            if len(self.frames) >= self.count:
                self.done.set()
        return self.done.is_set()

    def wait(self, timeout_s: float) -> List[np.ndarray]:
        """The frames collected within ``timeout_s`` (possibly fewer than ``count``)."""
        self.done.wait(max(0.0, float(timeout_s)))
        with self._lock:
            return list(self.frames)


class EngineFrameSource:
    """:class:`modules.saturation_profile.FrameSource` through an acquisition engine.

    Args:
        engine: The running :class:`modules.acquisition.AcquisitionEngine` (or its Qt thread).
        backend_name: The camera backend's name (``"poa"``, ...).
        settle_frames: Frames dropped after every change.
        apply_timeout_s: Longest wait for a bin / parameter change to be applied.
    """

    def __init__(self, engine, backend_name: str, settle_frames: int = 2, apply_timeout_s: float = 20.0):
        self.engine = engine
        self.backend = engine.backend
        self.backend_name = str(backend_name).lower()
        self.settle_frames = int(settle_frames)
        self.apply_timeout_s = float(apply_timeout_s)
        self._target = None

    # ── the camera ──────────────────────────────────────────

    def bins(self) -> Tuple[int, ...]:
        """Bins the camera supports."""
        return tuple(int(b) for b in self.backend.get_bins()) or (int(self.backend.current_bin),)

    def exposure_limits_ms(self) -> Tuple[float, float]:
        """``(min, max)`` exposure in ms."""
        low, high = self.backend.exposure_range_us
        return float(low) / 1000.0, float(high) / 1000.0

    def gain_limits(self) -> Tuple[float, float]:
        """``(min, max)`` gain."""
        low, high = self.backend.gain_range
        return float(low), float(high)

    def identity(self):
        """The configuration now applied."""
        return identity_of(self.backend, self.backend_name)

    def nominal_full_scale(self) -> int:
        """The frames' nominal full scale."""
        return nominal_full_scale_of(self.backend)

    def firmware(self) -> Optional[str]:
        """Firmware version when the backend reports one."""
        reader = getattr(self.backend, "firmware_version", None)
        try:
            return None if reader is None else (str(reader()) or None)
        except Exception:                                      # noqa: BLE001 - optional information
            return None

    # ── settings ────────────────────────────────────────────

    def snapshot(self) -> dict:
        """The camera setting to put back: ``{'bin', 'gain', 'exposure_ms'}``."""
        return {"bin": int(self.backend.current_bin), "gain": float(self.backend.gain),
                "exposure_ms": float(self.backend.exposure_ms)}

    def _settled(self, camera_bin, gain, exposure_ms) -> bool:
        engine = self.engine
        if engine.bin_pending or engine.params_pending:
            return False
        return RawTap(1, (camera_bin, gain, exposure_ms)).matches(self.backend)

    def apply(self, camera_bin: int, gain: float, exposure_ms: float) -> None:
        """Queue the setting with the engine and wait until it is applied.

        Raises:
            TimeoutError: not applied within ``apply_timeout_s`` (the engine refused it or the
                camera did not answer).
        """
        engine = self.engine
        if int(self.backend.current_bin) != int(camera_bin):
            engine.request_bin(int(camera_bin))
        engine.request_params(gain, float(exposure_ms))
        # the engine applies an int gain (request_params): compare with what it will hold
        target = (int(camera_bin), float(int(gain)), float(exposure_ms))
        deadline = time.monotonic() + self.apply_timeout_s
        while not self._settled(*target):
            if time.monotonic() > deadline:
                raise TimeoutError(f"bin {camera_bin}, gain {gain:g}, {exposure_ms:g} ms not applied within "
                                   f"{self.apply_timeout_s:g} s")
            time.sleep(0.01)
        self._target = target

    def grab(self, count: int, timeout_s: float) -> List[np.ndarray]:
        """``count`` raw frames read entirely with the applied setting (fewer on timeout)."""
        tap = self.engine.start_raw_tap(count, target=self._target, skip=self.settle_frames)
        try:
            exposure_s = (self._target[2] if self._target else 1000.0) / 1000.0
            return tap.wait(float(timeout_s) + self.settle_frames * (3.0 * exposure_s + 1.0))
        finally:
            self.engine.stop_raw_tap(tap)

    def restore(self, snapshot: dict, timeout_s: Optional[float] = None) -> List[str]:
        """Put the camera back to ``snapshot`` and check it: the problems found (``[]``: restored)."""
        try:
            saved, self.apply_timeout_s = self.apply_timeout_s, float(timeout_s or self.apply_timeout_s)
            try:
                self.apply(snapshot["bin"], snapshot["gain"], snapshot["exposure_ms"])
            finally:
                self.apply_timeout_s = saved
        except Exception as exc:                               # noqa: BLE001 - reported to the user
            return [f"camera setting not restored: {exc}"]
        now = self.snapshot()
        problems = [f"{key} {now[key]:g} != {snapshot[key]:g}" for key in ("bin", "gain", "exposure_ms")
                    if not _close(now[key], snapshot[key])]
        return problems


__all__ = ["RawTap", "EngineFrameSource"]
