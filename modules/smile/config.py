"""Settings for stored Smile models, owned by the smile module."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SmileConfig:
    """Where Smile models (``smile_model_bin<N>.npz``) are stored.

    Attributes:
        smile_map_folder: Folder of Smile model files (the application appends the
            camera backend name, one folder per camera).
    """

    smile_map_folder: str = "smile_maps"
