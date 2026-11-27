"""A simulated camera for stand-alone use and demos of the saturation measurement (Qt-free).

:class:`SimulatedSource` is a :class:`~modules.saturation_profile.FrameSource`: a bright gradient
scene whose 12-bit values are shifted << 4 and clip at ``clip`` (the Apollo-M MAX reads 65504),
with optional hot pixels. :class:`SimulatedSession` wraps it the way the calibration window
expects (``source`` + ``finish()``).

Example:
    >>> from modules.saturation_profile import SweepPlan, measure
    >>> measure(SimulatedSource(), SweepPlan((1,), (1.0, 4.0), 10.0))[0].level
    65504
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from modules.contracts.saturation_profile import SaturationIdentity


class SimulatedSource:
    """Simulated camera (see the module docstring).

    Args:
        shape: Frame shape at bin 1 (binned frames are smaller).
        clip: Value a clipped pixel reads.
        hot: Number of stuck pixels (they read ``clip``).
        delay_s: Pause per frame (to watch the window work).
    """

    def __init__(self, shape=(240, 320), clip: int = 65504, hot: int = 0, delay_s: float = 0.0):
        rng = np.random.default_rng(3)
        self.shape, self.clip, self.delay_s = shape, int(clip), float(delay_s)
        self.camera_bin, self.gain, self.exposure_ms = 1, 1.0, 10.0
        self._noise = rng.normal(0, 0.003, shape)
        self._hot = (rng.integers(0, shape[0], hot), rng.integers(0, shape[1], hot)) if hot else None

    def bins(self) -> Tuple[int, ...]:
        """Bins offered."""
        return (1, 2)

    def exposure_limits_ms(self) -> Tuple[float, float]:
        """Exposure range (ms)."""
        return 0.1, 10000.0

    def gain_limits(self) -> Tuple[float, float]:
        """Gain range."""
        return 0.0, 500.0

    def apply(self, camera_bin: int, gain: float, exposure_ms: float) -> None:
        """Take the setting at once."""
        self.camera_bin, self.gain, self.exposure_ms = int(camera_bin), float(gain), float(exposure_ms)

    def _frame(self):
        h, w = self.shape
        radiance = np.outer(np.linspace(0.05, 1.0, h), np.linspace(0.2, 1.0, w))
        signal = radiance * self.exposure_ms * (1 + 0.02 * (self.gain - 1)) * 40.0 + self._noise * 4095
        counts = np.clip(np.round(signal), 0, 4094).astype(np.int64)
        frame = (counts << 4).astype(np.uint16)
        frame[counts >= 4094] = self.clip
        if self._hot is not None:
            frame[self._hot] = self.clip
        b = self.camera_bin
        return frame[: h - h % b: b, : w - w % b: b] if b > 1 else frame

    def grab(self, count: int, timeout_s: float) -> List[np.ndarray]:
        """``count`` frames."""
        frames = []
        for _ in range(count):
            if self.delay_s:
                time.sleep(self.delay_s)
            frames.append(self._frame())
        return frames

    def identity(self) -> SaturationIdentity:
        """The simulated configuration."""
        return SaturationIdentity("sim", "Simulated", "0000", "RAW16", 16, 12, self.camera_bin)

    def nominal_full_scale(self) -> int:
        """4095 << 4."""
        return 65520

    def firmware(self) -> Optional[str]:
        """No firmware."""
        return None


class SimulatedSession:
    """A session for the calibration window: the source, and nothing to restore."""

    def __init__(self, source: Optional[SimulatedSource] = None):
        self.source = source or SimulatedSource(delay_s=0.02)
        self.software = "demo"

    def finish(self) -> list:
        """Nothing to restore."""
        return []


__all__ = ["SimulatedSource", "SimulatedSession"]
