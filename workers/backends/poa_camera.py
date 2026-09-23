# workers/backends/poa_camera.py
"""POA camera backend."""

from __future__ import annotations
import logging
import time
import threading
import numpy as np
import workers.backends.poa.pyPOACamera as pyPOACamera
from .base import AbstractCameraBackend, CameraBackendInfo

logger = logging.getLogger(__name__)


class POACameraBackend(AbstractCameraBackend):

    def __init__(self, gain: int = 0, exposure_ms: int = 100):
        super().__init__()
        self.gain = gain
        self.exposure_ms = exposure_ms
        self.camera_id = -1
        self.bit_depth = 8
        self.pixel_size = 1.0
        self.img_height = 0
        self.img_width = 0
        self.img_format = None
        self.buf_array = None
        self._exposure_range_us: tuple = (0, 10_000_000)
        self._gain_range: tuple = (0, 255)
        self._bins: tuple = ()
        self._current_bin: int = 1
        self._stop_event: threading.Event | None = None

    # ══════════════════════════════════════════════════════
    #  Properties
    # ══════════════════════════════════════════════════════

    @property
    def current_bin(self) -> int:
        return self._current_bin

    @property
    def exposure_range_us(self) -> tuple:
        return self._exposure_range_us

    @property
    def gain_range(self) -> tuple:
        return self._gain_range

    def get_bins(self) -> tuple:
        return self._bins

    @property
    def info(self) -> CameraBackendInfo:
        return CameraBackendInfo(
            camera_id=self.camera_id,
            bit_depth=self.bit_depth,
            pixel_size=self.pixel_size,
            width=self.img_width,
            height=self.img_height,
            image_format=self.img_format,
        )

    # ══════════════════════════════════════════════════════
    #  Internal helpers
    # ══════════════════════════════════════════════════════

    def _allocate_image_buffer(self) -> None:
        img_size = pyPOACamera.ImageCalcSize(
            self.img_height, self.img_width, self.img_format)
        self.buf_array = np.zeros(img_size, dtype=np.uint8)

    def _refresh_image_state(self) -> None:
        _, self.img_width, self.img_height = pyPOACamera.GetImageSize(
            self.camera_id)
        _, self.img_format = pyPOACamera.GetImageFormat(self.camera_id)
        self._allocate_image_buffer()

    # ══════════════════════════════════════════════════════
    #  Params
    # ══════════════════════════════════════════════════════

    def set_params(self, gain: int, exposure_ms: int) -> None:
        self.gain = gain
        self.exposure_ms = exposure_ms
        if self.camera_id > -1:
            pyPOACamera.SetGain(self.camera_id, gain, False)
            pyPOACamera.SetExp(
                self.camera_id, int(round(exposure_ms * 1000)), False)

    def set_bin(self, bin_value: int) -> None:
        if self.camera_id < 0:
            return
        desired_bin = int(bin_value)
        if self._bins and desired_bin not in self._bins:
            raise ValueError(f"Unsupported bin value: {desired_bin}")

        pyPOACamera.SetImageBin(self.camera_id, desired_bin)
        self._current_bin = desired_bin
        self._refresh_image_state()

    # ══════════════════════════════════════════════════════
    #  Hardware lifecycle — base class calls these
    # ══════════════════════════════════════════════════════

    def _open_device(self) -> None:
        camera_count = pyPOACamera.GetCameraCount()
        if camera_count <= 0:
            raise RuntimeError("No POA camera found")

        _, camera_properties = pyPOACamera.GetCameraProperties(0)

        self.camera_id = camera_properties.cameraID
        self.bit_depth = camera_properties.bitDepth
        if self.bit_depth in (10, 12, 16):
            self.bit_depth = 16
        self.pixel_size = camera_properties.pixelSize
        self._bins = tuple(int(v) for v in camera_properties.bins)

        camera_error = pyPOACamera.OpenCamera(self.camera_id)
        if camera_error != pyPOACamera.POAErrors.POA_OK:
            raise RuntimeError(pyPOACamera.GetErrorString(camera_error))

        camera_error = pyPOACamera.InitCamera(self.camera_id)
        if camera_error != pyPOACamera.POAErrors.POA_OK:
            raise RuntimeError(pyPOACamera.GetErrorString(camera_error))

        # ranges
        _, exposure_attr = pyPOACamera.GetConfigAttributesByConfigID(
            self.camera_id, pyPOACamera.POAConfig.POA_EXPOSURE)
        self._exposure_range_us = (
            int(exposure_attr.minValue), int(exposure_attr.maxValue))

        _, gain_attr = pyPOACamera.GetConfigAttributesByConfigID(
            self.camera_id, pyPOACamera.POAConfig.POA_GAIN)
        self._gain_range = (int(gain_attr.minValue), int(gain_attr.maxValue))

        # defaults
        self.set_params(self.gain, self.exposure_ms)

        pyPOACamera.SetConfig(
            self.camera_id, pyPOACamera.POAConfig.POA_OFFSET, 20, False)
        pyPOACamera.SetConfig(
            self.camera_id, pyPOACamera.POAConfig.POA_USB_BANDWIDTH_LIMIT, 90, False)
        pyPOACamera.SetConfig(
            self.camera_id, pyPOACamera.POAConfig.POA_PIXEL_BIN_SUM, 0, False)

        pyPOACamera.SetImageFormat(
            self.camera_id, pyPOACamera.POAImgFormat.POA_RAW16)

        # default bin
        if 2 in self._bins:
            default_bin = 2
        elif 1 in self._bins:
            default_bin = 1
        elif self._bins:
            default_bin = self._bins[0]
        else:
            default_bin = 1
        self.set_bin(default_bin)

    def _close_device(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

        if self.camera_id > -1:
            try:
                pyPOACamera.CloseCamera(self.camera_id)
            except Exception:
                pass

        self.camera_id = -1

    def _start_device(self) -> None:
        camera_error = pyPOACamera.StartExposure(self.camera_id, False)
        if camera_error != pyPOACamera.POAErrors.POA_OK:
            raise RuntimeError(f"StartExposure failed: {camera_error}")

    def _stop_device(self) -> None:
        if self.camera_id > -1:
            try:
                pyPOACamera.StopExposure(self.camera_id)
            except Exception:
                pass

    # ══════════════════════════════════════════════════════
    #  Fetch
    # ══════════════════════════════════════════════════════

    def _fetch_raw_frame(self, timeout_ms: int = 1000) -> np.ndarray | None:
        if self.camera_id < 0:
            return None

        deadline = time.monotonic() + timeout_ms / 1000.0

        while time.monotonic() < deadline:
            if self._stop_event is not None and self._stop_event.is_set():
                return None

            try:
                _, p_is_ready = pyPOACamera.ImageReady(self.camera_id)
            except Exception:
                return None

            if p_is_ready:
                try:
                    _, live_w, live_h = pyPOACamera.GetImageSize(
                        self.camera_id)
                    if (live_w, live_h) != (self.img_width, self.img_height):
                        self.img_width, self.img_height = live_w, live_h
                        self._allocate_image_buffer()

                    pyPOACamera.GetImageData(
                        self.camera_id, self.buf_array, timeout_ms)

                    frame_nd = pyPOACamera.ImageDataConvert(
                        self.buf_array, self.img_height,
                        self.img_width, self.img_format,
                    )

                    if frame_nd is None:
                        return None

                    if frame_nd.ndim == 3:
                        frame = frame_nd[:, :, 0]
                    elif frame_nd.ndim == 2:
                        frame = frame_nd
                    else:
                        return None

                    self.img_height, self.img_width = frame.shape
                    return np.ascontiguousarray(frame)

                except Exception:
                    return None

            time.sleep(0.002)

        return None

    # ══════════════════════════════════════════════════════
    #  Misc
    # ══════════════════════════════════════════════════════

    def set_stop_event(self, event: threading.Event):
        self._stop_event = event

    def dropped_images_count(self) -> int:
        if self.camera_id < 0:
            return 0
        _, drop_count = pyPOACamera.GetDroppedImagesCount(self.camera_id)
        return drop_count

    def get_temperature(self) -> float:
        if self.camera_id < 0:
            return -999.0
        try:
            camera_error, temp = pyPOACamera.GetCameraTEMP(self.camera_id)
            if camera_error == pyPOACamera.POAErrors.POA_OK:
                return float(temp)
        except Exception:
            pass
        return -999.0