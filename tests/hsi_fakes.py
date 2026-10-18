"""Fake camera shared by the acquisition and application smoke tests."""
import time
from types import SimpleNamespace

import numpy as np

from modules.camera import AbstractCameraBackend

W, H = 120, 40


class FakeCamera(AbstractCameraBackend):
    """Minimal backend: frame i is filled with value i, so tests can check ordering."""

    def __init__(self):
        super().__init__()
        self.index = 0
        self.bin = 1
        self.refuse = set()
        self.no_stream_bins = set()      # bins the camera accepts but cannot start streaming at
        self.closed = False
        self.exposure_ms = 1.0

    def _open_device(self): pass
    def _close_device(self): self.closed = True
    def _start_device(self):
        if self.bin in self.no_stream_bins:
            raise RuntimeError('GC_ERR_IO at AcquisitionStart')
    def _stop_device(self): pass

    def _fetch_raw_frame(self, timeout_ms=1000):
        time.sleep(0.001)
        self.index += 1
        return np.full((H, W // self.bin), self.index % 60000, np.uint16)

    def set_params(self, gain, exposure_ms): self.exposure_ms = float(exposure_ms)
    def frame_rate(self): return min(1000.0 / self.bin, 1000.0 / self.exposure_ms)

    def set_bin(self, bin_value):
        if bin_value in self.refuse:
            raise RuntimeError('camera refused bin %d' % bin_value)
        self.bin = bin_value

    info = property(lambda s: SimpleNamespace(height=H, width=W // s.bin, bit_depth=12))
    exposure_range_us = property(lambda s: (1, 100))
    gain_range = property(lambda s: (0, 10))
    current_bin = property(lambda s: s.bin)
    def get_bins(self): return (1, 2, 3)
    def dropped_images_count(self): return 0
    def get_temperature(self): return 20.0
