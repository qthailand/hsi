"""
workers/backends/base.py

Base class for all camera backends.

Responsibility split:
    Subclass implements   : _fetch_raw_frame() — raw frame from hardware, no processing.
    Base class implements : the shared correction pipeline (dark subtract,
                             hot-pixel repair, saturation check) that every
                             backend gets for free.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraBackendInfo:
    camera_id: int
    bit_depth: int
    pixel_size: float
    width: int
    height: int
    image_format: object

class AbstractCameraBackend(ABC):

    def __init__(self):
        self._dark_mean: np.ndarray | None = None
        self._hot_pixel_map: np.ndarray | None = None
        self._dark_enabled: bool = True
        self._hot_coords: np.ndarray | None = None
        self._hot_ys: np.ndarray | None = None
        self._hot_xs: np.ndarray | None = None
        self._hot_valid: np.ndarray | None = None
        self._neighbor_count: np.ndarray | None = None
        self._is_open: bool = False
        self._is_streaming: bool = False

    # ══════════════════════════════════════════════════════
    #  Context manager — ใช้ with ได้
    # ══════════════════════════════════════════════════════

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ══════════════════════════════════════════════════════
    #  Public lifecycle — base class จัดการ state
    # ══════════════════════════════════════════════════════

    def open(self) -> None:
        if self._is_open:
            return
        self._open_device()
        self._is_open = True

    def close(self) -> None:
        if not self._is_open:
            return
        if self._is_streaming:
            self.stop_stream()
        self._close_device()
        self._is_open = False

    def start_stream(self) -> None:
        if not self._is_open:
            raise RuntimeError("Camera not open — call open() first")
        if self._is_streaming:
            return
        self._start_device()
        self._is_streaming = True

    def stop_stream(self) -> None:
        if not self._is_streaming:
            return
        self._stop_device()
        self._is_streaming = False

    # ══════════════════════════════════════════════════════
    #  Subclass implements — hardware-specific
    # ══════════════════════════════════════════════════════

    @abstractmethod
    def _open_device(self) -> None:
        """Hardware init — connect, configure, tune."""
        pass

    @abstractmethod
    def _close_device(self) -> None:
        """Release hardware resources."""
        pass

    @abstractmethod
    def _start_device(self) -> None:
        """Start hardware acquisition."""
        pass

    @abstractmethod
    def _stop_device(self) -> None:
        """Stop hardware acquisition."""
        pass

    @abstractmethod
    def _fetch_raw_frame(self, timeout_ms: int = 1000) -> np.ndarray | None:
        """Fetch one raw frame from hardware — no processing."""
        pass

    # ══════════════════════════════════════════════════════
    #  Subclass implements — params
    # ══════════════════════════════════════════════════════

    @abstractmethod
    def set_params(self, gain: int, exposure_ms: int) -> None:
        pass

    @abstractmethod
    def set_bin(self, bin_value: int) -> None:
        pass

    @property
    @abstractmethod
    def info(self) -> CameraBackendInfo:
        pass

    @property
    @abstractmethod
    def exposure_range_us(self) -> tuple:
        pass

    @property
    @abstractmethod
    def gain_range(self) -> tuple:
        pass

    @abstractmethod
    def get_bins(self) -> tuple:
        pass

    @property
    @abstractmethod
    def current_bin(self) -> int:
        pass

    @abstractmethod
    def dropped_images_count(self) -> int:
        pass

    @abstractmethod
    def get_temperature(self) -> float:
        pass

    def set_stop_event(self, event) -> None:
        pass

    # ══════════════════════════════════════════════════════
    #  Frame acquisition — shared pipeline
    # ══════════════════════════════════════════════════════

    def read_frame(self, timeout_ms: int = 1000) -> np.ndarray | None:
        """Fetch a frame and apply dark subtract + hot-pixel correction."""
        frame = self._fetch_raw_frame(timeout_ms)
        if frame is None:
            return None
        if self._dark_enabled:
            frame = self._apply_dark_subtract(frame)
            frame = self._apply_hot_pixel(frame)
        return frame

    def read_frame_raw(self, timeout_ms: int = 1000) -> np.ndarray | None:
        """Fetch a frame with no correction applied."""
        return self._fetch_raw_frame(timeout_ms)

    # ══════════════════════════════════════════════════════
    #  Dark / hot-pixel — shared
    # ══════════════════════════════════════════════════════

    def set_dark_enabled(self, enabled: bool) -> None:
        self._dark_enabled = bool(enabled)

    def set_dark_reference(self, dark_mean, hot_pixel_map):
        self._dark_mean = dark_mean
        self._hot_pixel_map = hot_pixel_map

        if hot_pixel_map is None:
            self._hot_coords = None
            self._neighbor_count = None
            return

        coords = np.column_stack(np.nonzero(hot_pixel_map))
        if coords.size == 0:
            self._hot_coords = None
            self._neighbor_count = None
            return

        self._hot_coords = coords
        self._precompute_hot_pixel_neighbors(hot_pixel_map)

    def _precompute_hot_pixel_neighbors(self, hot_pixel_map):
        h, w = hot_pixel_map.shape
        offsets = np.array([
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ])

        ys = self._hot_coords[:, 0:1] + offsets[:, 0]
        xs = self._hot_coords[:, 1:2] + offsets[:, 1]

        in_bounds = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        ys_clipped = np.clip(ys, 0, h - 1)
        xs_clipped = np.clip(xs, 0, w - 1)

        neighbor_is_hot = hot_pixel_map[ys_clipped, xs_clipped].astype(bool)
        valid = in_bounds & ~neighbor_is_hot

        self._hot_ys = ys_clipped
        self._hot_xs = xs_clipped
        self._hot_valid = valid
        self._neighbor_count = np.maximum(valid.sum(axis=1, keepdims=True), 1)

    def _apply_dark_subtract(self, frame):
        if self._dark_mean is None:
            return frame
        if self._dark_mean.shape != frame.shape:
            return frame.copy()
        return cv2.subtract(frame, self._dark_mean)

    def _apply_hot_pixel(self, frame):
        if self._hot_coords is None:
            return frame

        neighbor_vals = frame[self._hot_ys, self._hot_xs].astype(np.float32)
        neighbor_vals = np.where(self._hot_valid, neighbor_vals, 0.0)
        neighbor_avg = neighbor_vals.sum(axis=1) / self._neighbor_count.ravel()

        frame[self._hot_coords[:, 0], self._hot_coords[:, 1]] = np.round(
            neighbor_avg
        ).astype(frame.dtype)
        return frame

    # ══════════════════════════════════════════════════════
    #  Diagnostics — shared
    # ══════════════════════════════════════════════════════

    def detect_saturation(self, raw_frame):
        bit_depth = self.info.bit_depth
        threshold = (1 << bit_depth) - 2
        return raw_frame >= threshold