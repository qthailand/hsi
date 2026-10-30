"""Settings for dark-reference storage, owned by the dark module."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DarkConfig:
    """Where dark references live and how many frames a capture averages.

    Attributes:
        dark_frame_folder: Older folder of ``dark_b<bin>_e<exposure>_g<gain>_t<temperature>.npz``
            files (the application appends the camera backend name). New darks go to the open
            camera's profile (``camera_profiles/<backend>/<model>_<serial>/dark``, see
            :class:`modules.contracts.CameraProfile`); this folder is only read as a fallback.
        dark_frame_n_frames: Frames averaged per dark capture.
    """

    dark_frame_folder: str = "dark_frames"
    dark_frame_n_frames: int = 50
