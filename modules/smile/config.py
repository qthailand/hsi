"""Settings for stored Smile models, owned by the smile module."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SmileConfig:
    """Where Smile models (``smile_model_bin<N>.npz``) are stored.

    Attributes:
        smile_map_folder: Older folder of Smile model files (the application appends the
            camera backend name). New models go to the open camera's profile
            (``camera_profiles/<backend>/<model>_<serial>/smile``, see
            :class:`modules.contracts.CameraProfile`); this folder is only read as a fallback.
    """

    smile_map_folder: str = "smile_maps"
