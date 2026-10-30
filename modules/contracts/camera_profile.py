"""Where one physical camera's calibration data lives: its *camera profile* folder.

Dark current, hot pixels, the Smile model and the wavelength/FWHM calibration all belong to
one sensor (and, for Smile and wavelength, to the spectrograph it is mounted on), so they are
kept together, one folder per camera::

    camera_profiles/<backend>/<model>_<serial>/
        profile.json        backend, model, serial, created / last opened
        dark/               dark_b<bin>_e<exposure>_g<gain>_t<temperature>.npz  (modules.dark)
        smile/              smile_model_bin<N>.npz, history/                    (modules.smile)
        wavelength/         wavelength_bin<N>.json, resolution_bin<N>.json, history/

``<model>`` and ``<serial>`` are what the camera SDK reports (characters outside
``A-Z a-z 0-9 . _ -`` become ``_``); a camera that reports neither gets ``unknown``. Files
stored before camera profiles existed (``dark_frames/<backend>``, ``smile_maps/<backend>``,
``calibration/wavelength/<backend>``) are not moved: hosts read them only as a fallback
(see :class:`~modules.contracts.CalibrationStore`), since nothing records which camera made them.

Threading:
    Pure path logic plus one small JSON write (:meth:`CameraProfile.write_info`); call from one
    thread.

Example:
    >>> profile = CameraProfile.for_camera("camera_profiles", "POA", "Apollo-M MAX", "CAMP7161")
    >>> profile.name, profile.dark_folder.as_posix()
    ('Apollo-M_MAX_CAMP7161', 'camera_profiles/poa/Apollo-M_MAX_CAMP7161/dark')
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = ["CameraProfile", "profile_name"]

#: Name of the profile folder of a camera that reports neither model nor serial.
UNKNOWN_CAMERA = "unknown"
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _name_part(text: Optional[str]) -> str:
    """``text`` made safe for one folder name (``""`` when empty)."""
    return _UNSAFE_NAME_CHARS.sub("_", str(text or "").strip()).strip("._")


def profile_name(model: Optional[str], serial: Optional[str]) -> str:
    """Folder name of a camera's profile.

    Returns:
        ``<model>_<serial>``; ``<model>`` or ``SN<serial>`` when only one is known;
        ``"unknown"`` when neither is.

    >>> profile_name(" MQ022 HG/IM ", "12:34"), profile_name(None, "77"), profile_name("..", "")
    ('MQ022_HG_IM_12_34', 'SN77', 'unknown')
    """
    model, serial = _name_part(model), _name_part(serial)
    if model and serial:
        return f"{model}_{serial}"
    if model:
        return model
    if serial:
        return f"SN{serial}"
    return UNKNOWN_CAMERA


@dataclass(frozen=True)
class CameraProfile:
    """The profile folder of one camera and the sub-folders each calibration uses.

    Attributes:
        root: ``<profiles folder>/<backend>/<name>``.
        backend: Camera backend (lower case), e.g. ``"poa"``.
        model: Camera model from the SDK, or ``None``.
        serial: Camera serial number from the SDK, or ``None``.
    """

    root: Path
    backend: str
    model: Optional[str] = None
    serial: Optional[str] = None

    @classmethod
    def for_camera(cls, profiles_folder, backend: str, model: Optional[str],
                   serial: Optional[str]) -> "CameraProfile":
        """Profile of the camera ``(backend, model, serial)`` under ``profiles_folder``."""
        backend = str(backend).lower()
        root = Path(profiles_folder) / backend / profile_name(model, serial)
        return cls(root, backend, model or None, serial or None)

    @property
    def name(self) -> str:
        """Folder name of this camera (``<model>_<serial>``)."""
        return self.root.name

    @property
    def known(self) -> bool:
        """True when the SDK identified the camera (model or serial)."""
        return self.model is not None or self.serial is not None

    @property
    def dark_folder(self) -> Path:
        """Dark / hot-pixel references (``modules.dark.DarkFrameLibrary``)."""
        return self.root / "dark"

    @property
    def smile_folder(self) -> Path:
        """Smile models ``smile_model_bin<N>.npz``."""
        return self.root / "smile"

    @property
    def wavelength_folder(self) -> Path:
        """Wavelength profiles and FWHM measurements per bin."""
        return self.root / "wavelength"

    @property
    def info_path(self) -> Path:
        """``profile.json``: which camera this folder belongs to."""
        return self.root / "profile.json"

    def write_info(self, **extra) -> Path:
        """Create the folder and write ``profile.json`` (keeps the first ``created_at``).

        Args:
            **extra: JSON-serialisable keys stored too (e.g. ``frame_width``).

        Returns:
            The path of ``profile.json``.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        created = now
        try:
            created = json.loads(self.info_path.read_text(encoding="utf-8")).get("created_at", now)
        except (OSError, ValueError):
            pass
        data = {"format_version": 1, "backend": self.backend, "model": self.model,
                "serial": self.serial, "created_at": created, "last_opened_at": now, **extra}
        tmp = self.info_path.with_name(self.info_path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.info_path)
        return self.info_path
