from __future__ import annotations
import os
import json
import logging
import math
from dataclasses import dataclass, fields
from pathlib import Path

from modules.acquisition.config import ProcessingConfig, parse_wavelength_ranges
from modules.camera.config import CameraConfig
from modules.dark.config import DarkConfig
from modules.smile.config import SmileConfig
from modules.stage.config import FREE_SCAN_MAX_S, FREE_SCAN_MIN_S, STAGE_TYPES, StageConfig

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
    camera_profile_folder: str = os.path.join(PROJECT_ROOT, "camera_profiles")


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
                                                                 "linear_speed_um_s", "free_duration_s", "scan_trim_extra_ms", "stage_device",
                                                                 "wavelength_range_nm", "camera_profile_folder"}
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
        free_duration_s=float(data.get("free_duration_s", stage_defaults.free_duration_s)),
        scan_trim_extra_ms=float(data.get("scan_trim_extra_ms", stage_defaults.scan_trim_extra_ms)),
        stage_device=str(data.get("stage_device") or ""),
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
        camera_profile_folder=str(data.get("camera_profile_folder", "camera_profiles")),
        rgb_target_wavelengths=tuple(float(v) for v in data.get("rgb_target_wavelengths", [645.0, 555.0, 465.0])),
        profile_timing=bool(data.get("profile_timing", True)),
        wavelength_range_nm=parse_wavelength_ranges(data.get("wavelength_range_nm")),
    )

    for name in ('dark_frame_folder', 'smile_map_folder'):
        value = Path(getattr(config, name))
        if not value.is_absolute():
            value = store.root / value
        # Older layout, one folder per backend: still read as a fallback. New darks and smile
        # models go to the open camera's profile (camera_profile_folder, see CameraProfile).
        setattr(config, name, str(value / config.camera_backend))
    profiles = Path(config.camera_profile_folder)
    config.camera_profile_folder = str(profiles if profiles.is_absolute() else store.root / profiles)
    if config.resolution_source not in ("config", "fwhm"):
        raise ValueError(f"resolution_source must be 'config' or 'fwhm', not {config.resolution_source!r}")
    for name in ("resolution", "fwhm_step_factor"):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive number, not {value}")
    if not math.isfinite(config.scan_trim_extra_ms) or config.scan_trim_extra_ms < 0:
        raise ValueError(f"scan_trim_extra_ms must be zero or more, not {config.scan_trim_extra_ms}")
    if config.stage_type not in STAGE_TYPES:
        raise ValueError(f"stage_type must be one of {', '.join(STAGE_TYPES)}, not {config.stage_type!r}")
    if not (math.isfinite(config.free_duration_s)
            and FREE_SCAN_MIN_S <= config.free_duration_s <= FREE_SCAN_MAX_S):
        raise ValueError(f"free_duration_s must be {FREE_SCAN_MIN_S:g} to {FREE_SCAN_MAX_S:g} seconds, "
                         f"not {config.free_duration_s}")
    config._config_store = store
    config._wavelength_profile = data.get('_wavelength_profile')
    config.camera_profile = None      # set by the scan controller once the camera is open
    logger.info("Loaded config: %s", config)
    return config


def legacy_wavelength_folder(config) -> Path:
    """Per-backend wavelength/FWHM folder of the older layout (read only as a fallback)."""
    store = getattr(config, "_config_store", None)
    if store is not None:
        return Path(store.wavelength_folder(config.camera_backend))
    return Path(PROJECT_ROOT) / "calibration" / "wavelength" / str(config.camera_backend).lower()


def calibration_store(config):
    """The ``CalibrationStore`` of the open camera's profile, falling back to the older folders.

    Smile, Wavelength and FWHM save into ``config.camera_profile`` (set by the scan controller
    once the camera is open) and also read the per-backend folders of the older layout when the
    profile has no file for a bin. Without a profile (no camera opened yet) it is the older
    layout only.
    """
    from modules.contracts import CalibrationStore
    legacy_wavelength = legacy_wavelength_folder(config)
    profile = getattr(config, "camera_profile", None)
    if profile is None:
        return CalibrationStore(legacy_wavelength, config.smile_map_folder)
    return CalibrationStore(profile.wavelength_folder, profile.smile_folder,
                            fallback_wavelength_folder=legacy_wavelength,
                            fallback_smile_folder=config.smile_map_folder)