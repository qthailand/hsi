"""Frame sources for :func:`modules.saturation_profile.measure`.

:class:`BackendFrameSource` drives an open camera backend directly (duck-typed: the
``modules.camera`` backends). Use it only when nothing else is acquiring from the camera (a
command-line measurement, hardware validation); the HSI application measures through its
acquisition engine instead, which owns the camera while the Scan window is open.

Settling: :meth:`BackendFrameSource.apply` stops the stream, applies bin, gain and exposure,
restarts the stream and discards ``settle_frames`` frames, so every frame :meth:`grab` returns
was exposed entirely with the new setting.

Data conventions:
    Frames are what ``backend.read_frame_raw()`` returns: the delivered raw frames, no dark
    subtraction or other correction. Exposure in ms, gain in backend units.

Threading:
    Call from one thread; the backend must not be used by another thread meanwhile.

Example:
    >>> source = BackendFrameSource(backend, "poa")          # doctest: +SKIP
    >>> records = measure(source, SweepPlan((1, 2), (1, 200), 250.0))   # doctest: +SKIP
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from modules.contracts.saturation_profile import SaturationIdentity


def identity_of(backend, backend_name: str) -> SaturationIdentity:
    """The :class:`SaturationIdentity` of ``backend``'s current configuration."""
    info = backend.info
    sensor = getattr(backend, "sensor_bit_depth", None)
    model = backend.model_name() if hasattr(backend, "model_name") else None
    serial = backend.serial_number() if hasattr(backend, "serial_number") else None
    return SaturationIdentity(backend_name, model, serial, getattr(info, "image_format", None),
                              int(info.bit_depth), int(sensor() if sensor is not None else info.bit_depth),
                              int(getattr(backend, "current_bin", 1)))


def nominal_full_scale_of(backend) -> int:
    """``(2**sensor_bits - 1) << (bit_depth - sensor_bits)``: the frames' nominal full scale
    (as ``modules.acquisition.camera_bit_info`` computes it)."""
    bits = int(backend.info.bit_depth)
    sensor = getattr(backend, "sensor_bit_depth", None)
    sensor_bits = int(sensor()) if sensor is not None else bits
    sensor_bits = min(max(sensor_bits, 1), bits)
    return ((1 << sensor_bits) - 1) << (bits - sensor_bits)


class BackendFrameSource:
    """:class:`modules.saturation_profile.FrameSource` on an open camera backend.

    Args:
        backend: An open ``modules.camera`` backend.
        backend_name: Its name (``"poa"``, ``"genicam"``, ``"ximea"``).
        settle_frames: Frames discarded after every change.
    """

    def __init__(self, backend, backend_name: str, settle_frames: int = 1):
        self.backend = backend
        self.backend_name = str(backend_name).lower()
        self.settle_frames = int(settle_frames)
        self._exposure_ms = None

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

    def _read(self, timeout_s: float):
        frame = self.backend.read_frame_raw(timeout_ms=int(max(1.0, timeout_s) * 1000))
        return None if frame is None else np.array(frame, copy=True)

    def apply(self, camera_bin: int, gain: float, exposure_ms: float) -> None:
        """Restart the stream with this setting and drop the settling frames."""
        backend = self.backend
        backend.stop_stream()
        if int(backend.current_bin) != int(camera_bin):
            backend.set_bin(int(camera_bin))
        backend.set_params(gain, exposure_ms)
        self._exposure_ms = float(exposure_ms)
        backend.start_stream()
        deadline = time.monotonic() + 5.0 + 3.0 * (self.settle_frames + 1) * exposure_ms / 1000.0
        dropped = 0
        while dropped < self.settle_frames:
            if time.monotonic() > deadline:
                raise TimeoutError(f"no frame within the settling time at {exposure_ms:g} ms")
            if self._read(exposure_ms / 1000.0 * 3 + 2) is not None:
                dropped += 1

    def grab(self, count: int, timeout_s: float) -> List[np.ndarray]:
        """``count`` frames, or fewer when ``timeout_s`` runs out."""
        frames, deadline = [], time.monotonic() + float(timeout_s)
        per_frame = (self._exposure_ms or 1000.0) / 1000.0 * 3 + 2
        while len(frames) < count and time.monotonic() < deadline:
            frame = self._read(min(per_frame, max(0.1, deadline - time.monotonic())))
            if frame is not None:
                frames.append(frame)
        return frames

    def identity(self) -> SaturationIdentity:
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


__all__ = ["BackendFrameSource", "identity_of", "nominal_full_scale_of"]
