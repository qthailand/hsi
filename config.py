from __future__ import annotations
import os
import json
import logging
import math
from dataclasses import dataclass, fields
from pathlib import Path

from modules.acquisition.config import ProcessingConfig
from modules.camera.config import CameraConfig
from modules.dark.config import DarkConfig
from modules.smile.config import SmileConfig
from modules.stage.config import StageConfig

logger = logging.getLogger("HSI")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class AppConfig(CameraConfig, StageConfig, ProcessingConfig, DarkConfig, SmileConfig):
    """All application settings as one flat object.

    Each group of settings is a dataclass owned by the module that uses it
    (:class:`CameraConfig`, :class:`StageConfig`, :class:`ProcessingConfig`,
    :class:`DarkConfig`, :class:`SmileConfig`); this class only composes them and fixes
    the default folders relative to the project.
    """

    dark_frame_folder: str = os.path.join(PROJECT_ROOT, "dark_frames")
    smile_map_folder: str = os.path.join(PROJECT_ROOT, "smile_maps")


#: Settings-file sections -> the module-owned dataclass whose fields they store.
#: ``m``/``c`` live in the wavelength base file, not in a section.
SECTION_CLASSES = {
    "camera": CameraConfig,
    "scan": StageConfig,
    "processing": ProcessingConfig,
    "dark": DarkConfig,
    "smile": SmileConfig,
}


def config_sections() -> dict:
    """Section name -> tuple of setting names, derived from the module-owned dataclasses."""
    return {
        section: tuple(f.name for f in fields(cls) if f.name not in ("m", "c"))
        for section, cls in SECTION_CLASSES.items()
    }


def load_app_config(path: str | None = None) -> AppConfig:
    """Read the split config directory; missing/invalid files fail explicitly."""
    from modules.configuration import ConfigStore
    store = ConfigStore(path or str(Path(PROJECT_ROOT) / 'config'), sections=config_sections())
    data = store.read()
    missing = set(AppConfig.__dataclass_fields__) - set(data) - {"preprocessing_backend", "sensor_pixel_size", "profile_timing",
                                                                 "resolution_source", "fwhm_step_factor",
                                                                 "target_temperature_c", "rotation_degree",
                                                                 "rotation_speed_dps", "linear_distance_mm",
                                                                 "linear_speed_um_s", "scan_trim_extra_ms"}
    target_temperature = data.get("target_temperature_c")
    stage_defaults = StageConfig()
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")

    config = AppConfig(
        camera_backend=str(data.get("camera_backend", "poa")),
        camera_id=int(data.get("camera_id", 0)),
        camera_index=int(data.get("camera_index", data.get("camera_id", 0))),
        cti_path=data.get("cti_path"),
        default_gain=int(data.get("default_gain", 100)),
        default_exposure_ms=float(data.get("default_exposure_ms", 50.0)),
        default_bin=int(data.get("default_bin", 1)),
        sensor_pixel_size=int(data.get("sensor_pixel_size", 10)),
        target_temperature_c=None if target_temperature is None else float(target_temperature),
        stage_type=str(data.get("stage_type", "rotation")).lower(),
        rotation_degree=float(data.get("rotation_degree", stage_defaults.rotation_degree)),
        rotation_speed_dps=float(data.get("rotation_speed_dps", stage_defaults.rotation_speed_dps)),
        linear_distance_mm=float(data.get("linear_distance_mm", stage_defaults.linear_distance_mm)),
        linear_speed_um_s=float(data.get("linear_speed_um_s", stage_defaults.linear_speed_um_s)),
        scan_trim_extra_ms=float(data.get("scan_trim_extra_ms", stage_defaults.scan_trim_extra_ms)),
        m=float(data.get("m", 0.2903)),
        c=float(data.get("c", 399.7910)),
        resolution=float(data.get("resolution", 2)),
        resolution_source=str(data.get("resolution_source", "config")).lower(),
        fwhm_step_factor=float(data.get("fwhm_step_factor", 1.0)),
        raw_inverted=bool(data.get("raw_inverted", True)),
        preprocessing_backend=str(data.get("preprocessing_backend", "auto")).lower(),
        dark_frame_folder=str(data.get("dark_frame_folder", "dark_frames")),
        dark_frame_n_frames=int(data.get("dark_frame_n_frames", 50)),
        smile_map_folder=str(data.get("smile_map_folder", os.path.join(PROJECT_ROOT, "smile_maps"))),
        rgb_target_wavelengths=tuple(float(v) for v in data.get("rgb_target_wavelengths", [645.0, 555.0, 465.0])),
        profile_timing=bool(data.get("profile_timing", True)),
    )

    for name in ('dark_frame_folder', 'smile_map_folder'):
        value = Path(getattr(config, name))
        if not value.is_absolute():
            value = store.root / value
        # Dark/hot-pixel references and smile models are specific to a camera
        # (sensor size, bin, response): keep one folder per backend.
        setattr(config, name, str(value / config.camera_backend))
    if config.resolution_source not in ("config", "fwhm"):
        raise ValueError(f"resolution_source must be 'config' or 'fwhm', not {config.resolution_source!r}")
    for name in ("resolution", "fwhm_step_factor"):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive number, not {value}")
    if not math.isfinite(config.scan_trim_extra_ms) or config.scan_trim_extra_ms < 0:
        raise ValueError(f"scan_trim_extra_ms must be zero or more, not {config.scan_trim_extra_ms}")
    config._config_store = store
    config._wavelength_profile = data.get('_wavelength_profile')
    logger.info("Loaded config: %s", config)
    return config