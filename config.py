from __future__ import annotations
import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("HSI")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

@dataclass
class AppConfig:
    camera_backend: str = "poa"
    camera_id: int = 0
    default_gain: int = 100
    default_exposure_ms: float = 50.0
    default_bin: int = 1
    stage_type: str = "rotation"

    m: float = 0.2903
    c: float = 399.7910
    resolution: int = 2
    raw_inverted: bool = True

    dark_frame_folder: str = os.path.join(PROJECT_ROOT, "dark_frames")
    dark_frame_n_frames: int = 50

    rgb_target_wavelengths: tuple[float, float, float] = (645.0, 555.0, 465.0)
    smile_map_folder: str = os.path.join(PROJECT_ROOT, "smile_maps")


def load_app_config(path: str = "backend.json") -> AppConfig:
    config_path = Path(path)

    if not config_path.exists():
        logger.warning("Config file not found: %s. Using defaults.", path)
        return AppConfig()

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    config = AppConfig(
        camera_backend=str(data.get("camera_backend", "poa")),
        camera_id=int(data.get("camera_id", 0)),
        default_gain=int(data.get("default_gain", 100)),
        default_exposure_ms=float(data.get("default_exposure_ms", 50.0)),
        default_bin=int(data.get("default_bin", 1)),
        stage_type=str(data.get("stage_type", "rotation")).lower(),
        m=float(data.get("m", 0.2903)),
        c=float(data.get("c", 399.7910)),
        resolution=int(data.get("resolution", 2)),
        raw_inverted=bool(data.get("raw_inverted", True)),
        dark_frame_folder=str(data.get("dark_frame_folder", "dark_frames")),
        dark_frame_n_frames=int(data.get("dark_frame_n_frames", 50)),
        rgb_target_wavelengths=tuple(float(v) for v in data.get("rgb_target_wavelengths", [645.0, 555.0, 465.0])),
    )

    logger.info("Loaded config: %s", config)
    return config