# workers/backends/genicam.py
import os
import threading
import numpy as np
import faulthandler
import time
from harvesters.core import Harvester
from .base import AbstractCameraBackend, CameraBackendInfo

faulthandler.enable()


class GenICamBackend(AbstractCameraBackend):

    def __init__(
        self,
        camera_index: int = 0,
        gain: int = 0,
        exposure_ms: int = 100,
        cti_path: str = None,
        default_bin: int = None,
    ):
        super().__init__()
        self.camera_index = camera_index
        self.gain = gain
        self.exposure_ms = exposure_ms
        self.default_bin = default_bin
        self.CTI_PATH = cti_path or os.path.join(
            os.getenv("GENICAM_GENTL64_PATH", ""),
            "libsv_gev_tl_x64.cti",
        )

        # hardware handles
        self._h = None
        self._ia = None
        self._nm = None

        # state
        self._current_bin: int = 1
        self._stop_event: threading.Event | None = None
        self._info = CameraBackendInfo(
            camera_id=self.camera_index,
            bit_depth=12,
            pixel_size=1.0,
            width=0,
            height=0,
            image_format=None,
        )
        self._exposure_range_us = (1, 10_000_000)
        self._gain_range = (0, 255)
        self._supported_bins = ()

        # GigE config
        self.BUFFER_COUNT = 1
        self.ASSUME_JUMBO = True
        self.PACKET_SAFE = 9000
        self.SCPD_SAFE = 0
        self.SCPD_FAST = 0

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
        return self._supported_bins

    @property
    def info(self) -> CameraBackendInfo:
        return self._info

    # ══════════════════════════════════════════════════════
    #  Node helpers
    # ══════════════════════════════════════════════════════

    def _has_node(self, name: str) -> bool:
        return self._nm is not None and hasattr(self._nm, name)

    def _node(self, name: str):
        return getattr(self._nm, name) if self._has_node(name) else None

    def _read_node(self, name: str):
        node = self._node(name)
        if node is None:
            return None
        try:
            return node.value
        except Exception:
            return None

    def _set_node_value(self, name: str, value) -> bool:
        node = self._node(name)
        if node is None:
            return False
        try:
            node.value = value
            _ = node.value
            return True
        except Exception:
            return False

    def _set_node_max(self, name: str) -> bool:
        node = self._node(name)
        if node is None:
            return False
        try:
            node.value = node.max
            _ = node.value
            return True
        except Exception:
            return False

    def _exec_node(self, name: str) -> bool:
        node = self._node(name)
        if node is None:
            return False
        try:
            node.execute()
            return True
        except Exception:
            return False

    # ══════════════════════════════════════════════════════
    #  Refresh
    # ══════════════════════════════════════════════════════

    def _refresh_info(self) -> None:
        width = int(self._read_node("Width") or 0)
        height = int(self._read_node("Height") or 0)
        pixel_format = self._read_node("PixelFormat")
        self._info = CameraBackendInfo(
            camera_id=self.camera_index,
            bit_depth=12,
            pixel_size=1.0,
            width=width,
            height=height,
            image_format=pixel_format,
        )

        self._supported_bins = ()
        horiz = self._node("BinningHorizontal")
        vert = self._node("BinningVertical")
        if horiz is not None or vert is not None:
            try:
                source = horiz if horiz is not None else vert
                min_bin = int(source.min)
                max_bin = int(source.max)
                if max_bin >= min_bin:
                    self._supported_bins = tuple(range(min_bin, max_bin + 1))
            except Exception:
                self._supported_bins = ()

    def _refresh_ranges(self) -> None:
        if self._has_node("ExposureTime"):
            try:
                node = self._node("ExposureTime")
                self._exposure_range_us = (float(node.min), float(node.max))
            except Exception:
                pass

        if self._has_node("Gain"):
            try:
                node = self._node("Gain")
                self._gain_range = (float(node.min), float(node.max))
            except Exception:
                pass

    # ══════════════════════════════════════════════════════
    #  Tuning
    # ══════════════════════════════════════════════════════

    def _tune_gige(self):
        self._exec_node("GevSCPSFireTest")

        if self._has_node("GevSCPSPacketSize"):
            if self.ASSUME_JUMBO:
                self._set_node_max("GevSCPSPacketSize")
            else:
                self._set_node_value("GevSCPSPacketSize", int(self.PACKET_SAFE))

        if self._has_node("DeviceLinkThroughputLimit"):
            self._set_node_max("DeviceLinkThroughputLimit")

        delay = int(self.SCPD_FAST if self.ASSUME_JUMBO else self.SCPD_SAFE)
        self._set_node_value("GevSCPD", delay)

    def _tune_framerate(self):
        if self._has_node("AcquisitionFrameRate"):
            self._set_node_max("AcquisitionFrameRate")
        if self._has_node("AcquisitionFrameRateEnable"):
            if not self._set_node_value("AcquisitionFrameRateEnable", False):
                self._set_node_value("AcquisitionFrameRateEnable", "Off")

    def _tune_image(self):
        self._set_node_value("ExposureAuto", "Off")
        self._set_node_value("AcquisitionMode", "Continuous")
        self._set_node_value("TriggerMode", "Off")

    # ══════════════════════════════════════════════════════
    #  Params
    # ══════════════════════════════════════════════════════

    def _set_gain(self, gain):
        if self._nm is None:
            return
        node = self._node("Gain")
        if node is None:
            return
        try:
            node.value = float(gain)
        except Exception:
            try:
                g = float(gain)
                if hasattr(node, "min"):
                    g = max(g, float(node.min))
                if hasattr(node, "max"):
                    g = min(g, float(node.max))
                node.value = g
            except Exception:
                pass

    def _set_exposure(self, expo_us):
        if self._nm is None:
            return
        self._set_node_value("ExposureAuto", "Off")
        node = self._node("ExposureTime")
        if node is None:
            return
        try:
            node.value = float(expo_us)
        except Exception:
            try:
                e = float(expo_us)
                if hasattr(node, "min"):
                    e = max(e, float(node.min))
                if hasattr(node, "max"):
                    e = min(e, float(node.max))
                node.value = e
            except Exception:
                pass

    def set_params(self, gain: int, exposure_ms: int) -> None:
        self.gain = max(self.gain_range[0], min(float(gain), self.gain_range[1]))
        self.exposure_ms = max(
            self.exposure_range_us[0] / 1000.0,
            min(float(exposure_ms), self.exposure_range_us[1] / 1000.0),
        )

        if self._nm is not None:
            self._set_gain(self.gain)
            self._set_exposure(self.exposure_ms * 1000.0)
            self._refresh_info()

    def set_bin(self, bin_value: int) -> None:
        if self._nm is None:
            return

        desired_bin = int(bin_value)
        if self._supported_bins and desired_bin not in self._supported_bins:
            raise ValueError(
                f"Unsupported bin value: {desired_bin}. "
                f"Supported: {self._supported_bins}"
            )

        if not self._set_node_value("BinningHorizontal", desired_bin):
            raise ValueError("GenICam backend does not support BinningHorizontal")

        if self._has_node("BinningVertical"):
            self._set_node_value("BinningVertical", desired_bin)

        self._current_bin = desired_bin
        self._refresh_info()

    # ══════════════════════════════════════════════════════
    #  Hardware lifecycle — base class calls these
    # ══════════════════════════════════════════════════════

    def _open_device(self) -> None:
        self._h = Harvester()
        self._h.add_file(self.CTI_PATH)
        self._h.update()

        if len(self._h.device_info_list) == 0:
            raise RuntimeError(
                "No device found. ตรวจสอบ IP/subnet/firewall"
            )

        self._ia = self._h.create(self.camera_index)
        self._ia.buffer_handling_mode = "NewestOnly"
        self._ia.num_buffers = self.BUFFER_COUNT
        self._nm = self._ia.remote_device.node_map

        self._set_node_value("DefectPixelCorrectionEnable_Control", "On")
        self._set_node_value("DefectPixelCorrectionMark_Control", "Off")

        self._refresh_ranges()
        self._refresh_info()
        self._tune_gige()
        self._tune_framerate()
        self._tune_image()
        self.set_params(self.gain, self.exposure_ms)

        if self.default_bin is not None:
            try:
                self.set_bin(int(self.default_bin))
            except Exception as e:
                print(f"[GenICam] Warning: could not apply default_bin={self.default_bin}: {e}")

    def _close_device(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

        if self._ia is not None:
            try:
                self._ia.destroy()
            except Exception:
                pass
        self._ia = None
        self._nm = None

        if self._h is not None:
            try:
                self._h.reset()
            except Exception:
                pass
        self._h = None

    def _start_device(self) -> None:
        if self._ia is None:
            raise RuntimeError("GenICam backend is not opened")
        self._ia.start()

    def _stop_device(self) -> None:
        if self._ia is not None:
            try:
                self._ia.stop()
            except Exception:
                pass

    # ══════════════════════════════════════════════════════
    #  Fetch — แค่ fetch + scale 12→16 bit
    # ══════════════════════════════════════════════════════

    def _fetch_raw_frame(self, timeout_ms: int = 1000) -> np.ndarray | None:
        if self._ia is None:
            return None

        if self._stop_event is not None and self._stop_event.is_set():
            return None

        try:
            with self._ia.fetch(timeout=timeout_ms) as buffer:
                if getattr(buffer, 'is_incomplete', False):
                    return None

                comp = buffer.payload.components[0]
                h, w = comp.height, comp.width

                if self._info.height > 0 and self._info.width > 0:
                    if h != self._info.height or w != self._info.width:
                        return None

                return (comp.data.reshape(h, w) << 4).copy()

        except Exception:
            return None

    # ══════════════════════════════════════════════════════
    #  Misc
    # ══════════════════════════════════════════════════════

    def set_stop_event(self, event: threading.Event):
        self._stop_event = event

    def dropped_images_count(self) -> int:
        return 0

    def get_temperature(self) -> float:
        if self._nm is None:
            return -999.0
        for node_name in ("DeviceTemperature", "Temperature", "TemperatureAbs"):
            val = self._read_node(node_name)
            if val is not None:
                return float(val)
        return -999.0