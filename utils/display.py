"""Display mapping: raw sensor → uint8 for GUI."""

from __future__ import annotations

from enum import Enum

import numpy as np


class DisplayMode(Enum):
    RAW = "raw"
    ENHANCE = "enhance"


class DisplayMapper:
    """Stateless helper — no per-frame allocation beyond the output array.

    RAW     : linear map  [0 .. 2^bit_depth-1] → [0 .. 255]
              16-bit value 4096  → uint8 value 16   (preserves true brightness)
              16-bit value 256   → uint8 value 1    (dark is dark)

    ENHANCE : min-max stretch (original behaviour)
              any frame → full 0-255 range (contrast-maximised)
    """

    __slots__ = ("mode", "bit_depth", "_scale", "_inv_range")

    def __init__(
        self,
        mode: DisplayMode = DisplayMode.RAW,
        bit_depth: int = 16,
    ):
        self.mode = mode
        self.bit_depth = bit_depth
        self._scale = 255.0 / float((1 << bit_depth) - 1)
        self._inv_range: float = 0.0  # unused until ENHANCE

    def set_bit_depth(self, bit_depth: int):
        self.bit_depth = int(bit_depth)
        self._scale = 255.0 / float((1 << bit_depth) - 1)

    def to_uint8(self, frame: np.ndarray) -> np.ndarray:
        """Convert *frame* (any integer dtype) → contiguous uint8."""
        if self.mode is DisplayMode.RAW:
            return self._raw(frame)
        return self._enhance(frame)

    # ── internals ────────────────────────────────────────────────

    @staticmethod
    def _enhance(frame: np.ndarray) -> np.ndarray:
        arr = frame.astype(np.float32)
        lo = arr.min()
        hi = arr.max()
        rng = hi - lo
        if rng > 0:
            arr = (arr - lo) * (255.0 / rng)
        else:
            arr[:] = 128.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _raw(self, frame: np.ndarray) -> np.ndarray:
        """Bit-shift or scale to 8-bit.  Pure numpy, no allocation beyond output."""
        # Fast path: already uint8
        if frame.dtype == np.uint8:
            return np.ascontiguousarray(frame)

        # Fast path: common bit depths — use right-shift (no float conversion)
        if frame.dtype == np.uint16:
            return self._uint16_to_uint8(frame)

        # Generic: float scale
        arr = frame.astype(np.float32) * self._scale
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _uint16_to_uint8(self, frame: np.ndarray) -> np.ndarray:
        """Vectorised uint16 → uint8 via integer arithmetic (zero float alloc)."""
        bd = self.bit_depth
        if bd == 16:
            return np.ascontiguousarray(
                np.right_shift(frame, 8).astype(np.uint8)
            )
        if bd == 12:
            return np.ascontiguousarray(
                np.right_shift(frame, 4).astype(np.uint8)
            )
        if bd == 10:
            # 10→8 : shift 2, but values only reach 1023→3 (dim).
            # Better: scale 0-1023 → 0-255  via  (x * 255 + 511) // 1023
            return np.ascontiguousarray(
                ((frame.astype(np.uint32) * 255 + 511) // 1023).astype(np.uint8)
            )
        if bd == 8:
            return np.ascontiguousarray(frame.astype(np.uint8))

        # Fallback
        arr = frame.astype(np.float32) * self._scale
        return np.clip(arr, 0, 255).astype(np.uint8)