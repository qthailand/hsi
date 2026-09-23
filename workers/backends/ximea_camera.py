import logging
import threading
import numpy as np
from ctypes import c_ulong
from .ximea import xiapi
from .ximea import xidefs
from .base import AbstractCameraBackend, CameraBackendInfo

logger = logging.getLogger(__name__)


class XIMEACameraBackend(AbstractCameraBackend):
    def __init__(self, camera_id: int = 0, gain: int = 0, exposure_ms: int = 100):
        self._camera_id = camera_id
        self._camera = None
        self._info = None
        self._gain = gain
        self._exposure_ms = exposure_ms
        self._exposure_range_us: tuple = (0, 0)
        self._gain_range: tuple = (0, 0)
        self._bins: tuple = ()
        self._lock = threading.Lock()

    def open(self) -> None:
        with self._lock:
            self._camera = xiapi.Camera(self._camera_id)
            self._camera.open_device()
            self._camera.set_imgdataformat('XI_MONO16')
            try:
                sensor_bit_depth = int(self._camera.get_sensor_bit_depth())
            except Exception:
                sensor_bit_depth = 10
            self._info = CameraBackendInfo(
                camera_id=self._camera_id,
                bit_depth=sensor_bit_depth,
                pixel_size=self._get_pixel_size_unsafe(),
                width=self._camera.get_aeag_roi_width(),
                height=self._camera.get_aeag_roi_height(),
                image_format='XI_MONO16'
            )
            self._exposure_range_us = (
                self._camera.get_exposure_minimum(),
                self._camera.get_exposure_maximum(),
            )
            self._gain_range = (
                self._camera.get_gain_minimum(),
                self._camera.get_gain_maximum(),
            )
            self._set_params_unsafe(self._gain, self._exposure_ms)
            self._bins = self._collect_equal_bin_values_unsafe()

            logger.info("cameraID: %s", self._camera_id)
            logger.info("exposure time range(us): %s ~ %s",
                        self._exposure_range_us[0], self._exposure_range_us[1])
            logger.info("gain range: %s ~ %s",
                        self._gain_range[0], self._gain_range[1])
            logger.info("maxWidth: %s", self._info.width)
            logger.info("maxHeight: %s", self._info.height)
            logger.info("bitDepth: %s", self._info.bit_depth)
            try:
                logger.info("downsampling_type: %s",
                            self._camera.get_downsampling_type())
                logger.info("downsampling: %s",
                            self._camera.get_downsampling())
            except Exception as e:
                logger.debug("Unable to log current downsampling state: %s", e)

    def close(self) -> None:
        with self._lock:
            if self._camera is not None:
                self._camera.close_device()
                self._camera = None

    def start_stream(self) -> None:
        with self._lock:
            if self._camera is not None:
                self._camera.start_acquisition()

    def stop_stream(self) -> None:
        with self._lock:
            if self._camera is not None:
                self._camera.stop_acquisition()

    def set_params(self, gain: float, exposure_ms: float) -> None:
        with self._lock:
            self._set_params_unsafe(gain, exposure_ms)

    def _set_params_unsafe(self, gain: float, exposure_ms: float) -> None:
        """Internal set_params — must only be called while self._lock is held."""
        if self._camera is None:
            logger.warning("set_params called but camera is not initialized")
            return

        gain = float(max(self._gain_range[0], min(gain, self._gain_range[1])))
        exposure_us = int(max(
            self._exposure_range_us[0],
            min(exposure_ms * 1000, self._exposure_range_us[1])
        ))
        self._camera.set_gain(gain)
        self._camera.set_exposure(exposure_us)
        logger.info("Set camera params: gain=%s, exposure_ms=%s",
                    gain, exposure_ms)
        logger.info(
            "Current camera params: gain=%s, exposure_ms=%s",
            self._camera.get_gain(), self._camera.get_exposure() / 1000
        )

    def _get_pixel_size_unsafe(self) -> float:
        """Internal pixel size query — must only be called while self._lock is held."""
        if self._camera is None:
            return 1.0
        try:
            return float(self._camera.get_pixel_size())
        except AttributeError:
            return 1.0
        except Exception:
            return 1.0

    def _get_pixel_size(self) -> float:
        with self._lock:
            return self._get_pixel_size_unsafe()

    def _collect_equal_bin_values_unsafe(self) -> tuple:
        """Internal bin query — must only be called while self._lock is held."""
        if self._camera is None:
            return ()

        try:
            if self._camera.get_downsampling_type() != 'XI_BINNING':
                self._camera.set_downsampling_type('XI_BINNING')

            dmin = self._camera.get_downsampling_minimum()
            dmax = self._camera.get_downsampling_maximum()
        except Exception as e:
            logger.debug("Unable to query XIMEA binning support: %s", e)
            return ()

        def _parse_dwn(dwn_str: str) -> int | None:
            try:
                factor = dwn_str.split('_')[-1]
                return int(factor.split('x')[0])
            except (IndexError, ValueError):
                return None

        lo = _parse_dwn(dmin)
        hi = _parse_dwn(dmax)

        if lo is None or hi is None or lo > hi:
            return ()

        return tuple(range(lo, hi + 1))

    def _collect_equal_bin_values(self) -> tuple:
        with self._lock:
            return self._collect_equal_bin_values_unsafe()

    def _refresh_info_unsafe(self) -> None:
        """Internal info refresh — must only be called while self._lock is held."""
        if self._camera is None:
            return

        try:
            sensor_bit_depth = int(self._camera.get_sensor_bit_depth())
        except Exception:
            sensor_bit_depth = 12
        self._info = CameraBackendInfo(
            camera_id=self._camera_id,
            bit_depth=sensor_bit_depth,
            pixel_size=self._get_pixel_size_unsafe(),
            width=self._camera.get_aeag_roi_width(),
            height=self._camera.get_aeag_roi_height(),
            image_format='XI_MONO16'
        )

    def _refresh_info(self) -> None:
        with self._lock:
            self._refresh_info_unsafe()

    def set_bin(self, bin_value: int) -> None:
        with self._lock:
            if self._camera is None:
                return

            if not self._bins:
                self._bins = self._collect_equal_bin_values_unsafe()

            logger.info("bin_value %s bin %s", bin_value, self._bins)
            if bin_value not in self._bins:
                raise ValueError(
                    f"Unsupported equal bin value: {
                        bin_value}. Supported: {self._bins}"
                )

            true_bin_param = f'XI_DWN_{bin_value}x{bin_value}'
            self._camera.set_downsampling(true_bin_param)
            try:
                logger.info("current downsampling after set_bin: %s",
                            self._camera.get_downsampling())
            except Exception as e:
                logger.debug(
                    "Unable to read downsampling after set_bin: %s", e)

            self._refresh_info_unsafe()
            self._bins = self._collect_equal_bin_values_unsafe()

    def read_frame(self, timeout_ms: int = 1000) -> np.ndarray:
        with self._lock:
            if self._camera is not None:
                buf_array = xiapi.Image()
                try:
                    self._camera.get_image(buf_array, timeout_ms)
                    frame = buf_array.get_image_data_numpy()
                    h, w = buf_array.height, buf_array.width
                    return np.ascontiguousarray(frame, dtype=np.uint16).reshape(h, w)
                except Exception as e:
                    logger.error(f"Failed to read frame: {e}")
                    return None
        return None

    @property
    def info(self) -> CameraBackendInfo:
        return self._info

    @property
    def exposure_range_us(self) -> tuple:
        return self._exposure_range_us

    @property
    def gain_range(self) -> tuple:
        return self._gain_range

    def get_bins(self) -> tuple:
        return self._bins

    def dropped_images_count(self) -> int:
        with self._lock:
            if self._camera is None:
                return 0
            if hasattr(self._camera, 'get_dropped_images_count'):
                try:
                    return int(self._camera.get_dropped_images_count())
                except Exception:
                    return 0
        return 0
