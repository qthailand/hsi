"""Display mapping: raw sensor frame -> uint8 for the GUI (Qt-free).

Data conventions:
    Input frames are 2-D ``(rows, columns)`` = ``(slit, wavelength)`` integer
    (or float) arrays, normally uint16.  Output is a C-contiguous uint8 array
    of the same shape.  ``bit_depth`` is the sensor's full-scale in bits
    (``2**bit_depth - 1`` is white).

Threading:
    Pure numpy, no Qt.  ``DisplayMapper.to_uint8`` keeps no per-frame state and
    may be called from any thread; do not call ``set_bit_depth`` concurrently
    with it.

Example:
    >>> import numpy as np
    >>> from modules.live_view.mapping import DisplayMapper, DisplayMode
    >>> frame = np.array([[0, 4096, 65535]], dtype=np.uint16)
    >>> DisplayMapper(DisplayMode.RAW, 16).to_uint8(frame).tolist()
    [[0, 16, 255]]
    >>> DisplayMapper(DisplayMode.ENHANCE).to_uint8(frame).tolist()
    [[0, 15, 255]]
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class DisplayMode(Enum):
    """How a raw frame is mapped to 8 bit.

    Members:
        RAW: Linear map of the full sensor range ``0 .. 2**bit_depth-1`` to 0..255.
        ENHANCE: Per-frame min-max stretch to the full 0..255 range.
    """

    RAW = "raw"
    ENHANCE = "enhance"


class DisplayMapper:
    """Converts a raw frame to a displayable uint8 image.

    RAW     : linear map  [0 .. 2^bit_depth-1] -> [0 .. 255]
              16-bit value 4096  -> uint8 value 16   (preserves true brightness)
              16-bit value 256   -> uint8 value 1    (dark is dark)

    ENHANCE : min-max stretch of the frame (any frame -> full 0..255 range,
              a constant frame becomes mid-grey 128)

    Args:
        mode: A :class:`DisplayMode`; default ``RAW``.
        bit_depth: Sensor bit depth (e.g. 8, 10, 12, 16); only used by ``RAW``.

    Attributes:
        mode: Current :class:`DisplayMode` (may be reassigned).
        bit_depth: Current bit depth; change with :meth:`set_bit_depth`.

    Note:
        Stateless apart from the two settings above; instances are cheap and are
        created anew by ``DisplayControls`` on every change.
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
        """Change the sensor bit depth used by ``RAW`` mode.

        Args:
            bit_depth: Full-scale in bits, e.g. 12 for 0..4095.
        """
        self.bit_depth = int(bit_depth)
        self._scale = 255.0 / float((1 << bit_depth) - 1)

    def to_uint8(self, frame: np.ndarray) -> np.ndarray:
        """Convert ``frame`` to a contiguous uint8 image.

        Args:
            frame: 2-D ``(rows, columns)`` array of any integer (or float) dtype.
                Not modified.

        Returns:
            numpy.ndarray: uint8 array, same shape.  A new array (uint8 input in RAW
            mode is returned as-is when already contiguous).
        """
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