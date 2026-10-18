"""Camera settings and the per-setting calibration store shared by Smile, Wavelength and FWHM.

A calibration is only valid for the camera configuration it was measured with. This file
defines that configuration (:class:`CameraSetting`) and *where* the calibration for a given
setting lives (:class:`CalibrationStore`), so that every module finds the same files without
asking the user:

* wavelength profile - ``<wavelength_folder>/wavelength_bin<N>.json``
  (a :class:`~modules.contracts.wavelength.WavelengthCalibration` profile plus a
  ``camera_setting`` block; loadable by ``WavelengthCalibration.load`` like any profile);
* smile model - ``<smile_folder>/smile_model_bin<N>.npz`` (the file the HSI runtime already
  loads per bin; read/written by ``modules.smile``);
* spectral resolution - ``<wavelength_folder>/resolution_bin<N>.json`` (the median FWHM in nm
  over the measured lamp lines, written by ``modules.spectral_resolution``; the acquisition
  engine can use it as the wavelength bin width).

The host chooses one folder pair per camera backend (``.../<backend>/``); files inside are
keyed by hardware bin. Before a file is replaced, the old one is copied to
``<folder>/history/<name>.<timestamp><ext>``.

Data conventions:
    ``frame_shape`` is ``(slit rows, spectral columns)`` of the oriented sensor frame at the
    hardware bin; ``()`` means "unknown" and is not compared.

Threading:
    Pure path logic and small file writes; call from one thread at a time per folder.

Example:
    >>> from modules.contracts.calibration_set import CameraSetting
    >>> s = CameraSetting("genicam", 0, 2, True, (552, 804))
    >>> s.label()
    'genicam #0 · bin 2 · 804×552 px · inverted'
    >>> s.mismatch(CameraSetting("genicam", 0, 1, True, (1104, 1608)))
    ['bin 1 ≠ 2', 'frame 1608×1104 ≠ 804×552']
"""
from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import List, Optional

__all__ = ["CameraSetting", "CalibrationStore", "StoredWavelength", "StoredResolution"]


@dataclass(frozen=True)
class CameraSetting:
    """The camera configuration a calibration belongs to.

    Attributes:
        camera_backend: ``"poa"``, ``"genicam"`` or ``"ximea"`` (lower case).
        camera_index: Device index of that backend.
        camera_bin: Hardware bin (>= 1).
        raw_inverted: Whether raw frames are flipped before processing.
        frame_shape: ``(rows, columns)`` of the oriented frame, or ``()`` if unknown.
    """

    camera_backend: str
    camera_index: int
    camera_bin: int
    raw_inverted: bool
    frame_shape: tuple = ()

    def label(self) -> str:
        """Short human-readable description, e.g. ``genicam #0 · bin 2 · 804×552 px``."""
        text = f"{self.camera_backend} #{self.camera_index} · bin {self.camera_bin}"
        if self.frame_shape:
            text += f" · {self.frame_shape[1]}×{self.frame_shape[0]} px"
        return text + (" · inverted" if self.raw_inverted else "")

    def with_frame(self, frame_shape, camera_bin: Optional[int] = None) -> "CameraSetting":
        """Copy with another frame shape (and optionally bin), e.g. for an image loaded from disk."""
        return replace(self, frame_shape=tuple(int(v) for v in frame_shape[:2]),
                       camera_bin=int(self.camera_bin if camera_bin is None else camera_bin))

    def mismatch(self, other: "CameraSetting") -> List[str]:
        """Differences that make a calibration of ``other`` invalid for ``self`` (empty = compatible).

        ``frame_shape`` is compared only when both are known.
        """
        problems = []
        if other.camera_backend.lower() != self.camera_backend.lower():
            problems.append(f"camera {other.camera_backend} ≠ {self.camera_backend}")
        if int(other.camera_index) != int(self.camera_index):
            problems.append(f"camera index {other.camera_index} ≠ {self.camera_index}")
        if int(other.camera_bin) != int(self.camera_bin):
            problems.append(f"bin {other.camera_bin} ≠ {self.camera_bin}")
        if bool(other.raw_inverted) != bool(self.raw_inverted):
            problems.append("orientation (raw_inverted) differs")
        if self.frame_shape and other.frame_shape and tuple(other.frame_shape) != tuple(self.frame_shape):
            problems.append(f"frame {other.frame_shape[1]}×{other.frame_shape[0]} ≠ "
                            f"{self.frame_shape[1]}×{self.frame_shape[0]}")
        return problems

    def to_dict(self) -> dict:
        """JSON-serialisable dict (``frame_shape`` as a list)."""
        data = asdict(self)
        data["frame_shape"] = list(self.frame_shape)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CameraSetting":
        """Inverse of :meth:`to_dict`; missing ``frame_shape`` means unknown."""
        return cls(str(data["camera_backend"]).lower(), int(data.get("camera_index", 0)),
                   int(data["camera_bin"]), bool(data.get("raw_inverted", False)),
                   tuple(int(v) for v in data.get("frame_shape") or ()))


@dataclass(frozen=True)
class StoredWavelength:
    """A wavelength profile found in the store.

    Attributes:
        profile: The :class:`~modules.contracts.wavelength.WavelengthCalibration`.
        path: File it was read from.
        setting: Camera setting recorded in the file (``None`` for a plain profile).
        problems: :meth:`CameraSetting.mismatch` against the requested setting (empty = valid).
        saved_at: Modification time of the file (seconds since the epoch).
    """

    profile: object
    path: Path
    setting: Optional[CameraSetting]
    problems: tuple
    saved_at: float

    @property
    def ok(self) -> bool:
        """True when the profile is valid for the requested setting."""
        return not self.problems


@dataclass(frozen=True)
class StoredResolution:
    """A spectral-resolution (FWHM) measurement found in the store.

    Attributes:
        fwhm_nm: Median FWHM over the measured lines (nm), finite and > 0.
        path: File it was read from.
        setting: Camera setting recorded in the file.
        problems: :meth:`CameraSetting.mismatch` against the requested setting (empty = valid).
        saved_at: Modification time of the file (seconds since the epoch).
        details: The other keys of the file (per-line values, fit settings, ...).
    """

    fwhm_nm: float
    path: Path
    setting: CameraSetting
    problems: tuple
    saved_at: float
    details: dict

    @property
    def ok(self) -> bool:
        """True when the measurement is valid for the requested setting."""
        return not self.problems


class CalibrationStore:
    """Locates, saves and loads calibrations by :class:`CameraSetting`.

    Args:
        wavelength_folder: Folder for ``wavelength_bin<N>.json`` and ``resolution_bin<N>.json``
            (one camera backend).
        smile_folder: Folder for ``smile_model_bin<N>.npz`` (one camera backend).
    """

    def __init__(self, wavelength_folder, smile_folder):
        self.wavelength_folder = Path(wavelength_folder)
        self.smile_folder = Path(smile_folder)

    # ── paths ──────────────────────────────────────────────
    def wavelength_path(self, setting: CameraSetting) -> Path:
        """Path of the wavelength profile for ``setting``'s bin."""
        return self.wavelength_folder / f"wavelength_bin{int(setting.camera_bin)}.json"

    def smile_path(self, setting: CameraSetting) -> Path:
        """Path of the smile model for ``setting``'s bin (the HSI runtime's file name)."""
        return self.smile_folder / f"smile_model_bin{int(setting.camera_bin)}.npz"

    def resolution_path(self, setting: CameraSetting) -> Path:
        """Path of the spectral-resolution (FWHM) measurement for ``setting``'s bin."""
        return self.wavelength_folder / f"resolution_bin{int(setting.camera_bin)}.json"

    @staticmethod
    def backup(path) -> Optional[Path]:
        """Copy an existing ``path`` to ``history/<stem>.<timestamp><suffix>``; returns the copy or None."""
        path = Path(path)
        if not path.exists():
            return None
        history = path.parent / "history"
        history.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(path.stat().st_mtime))
        target = history / f"{path.stem}.{stamp}{path.suffix}"
        n = 1
        while target.exists():
            target = history / f"{path.stem}.{stamp}-{n}{path.suffix}"
            n += 1
        shutil.copy2(path, target)
        return target

    # ── wavelength ─────────────────────────────────────────
    def save_wavelength(self, profile, setting: CameraSetting) -> Path:
        """Write ``profile`` as the wavelength calibration of ``setting`` (old file kept in history).

        Raises:
            ValueError: the profile is invalid, or its bin/frame shape disagrees with ``setting``.
        """
        profile.validate()
        if int(profile.camera_bin) != int(setting.camera_bin):
            raise ValueError(f"Profile bin {profile.camera_bin} ≠ camera bin {setting.camera_bin}")
        if profile.frame_shape and setting.frame_shape and tuple(profile.frame_shape) != tuple(setting.frame_shape):
            raise ValueError("Profile image size differs from the camera frame size")
        if not setting.frame_shape and profile.frame_shape:
            setting = setting.with_frame(profile.frame_shape)
        path = self.wavelength_path(setting)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.backup(path)
        data = asdict(profile)
        data.update(format_version=1, units="nm", stage="post_dark_hot_pixel_orientation",
                    camera_setting=setting.to_dict(), saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
        tmp.replace(path)
        return path

    def load_wavelength(self, setting: CameraSetting) -> Optional[StoredWavelength]:
        """Read the wavelength profile stored for ``setting``'s bin, or ``None`` if there is none.

        The result's ``problems`` lists every mismatch (camera, index, orientation, frame size);
        callers must not use a profile whose ``ok`` is False.

        Raises:
            ValueError, KeyError, OSError: the file exists but is not a valid profile.
        """
        from .wavelength import WavelengthCalibration
        path = self.wavelength_path(setting)
        if not path.exists():
            return None
        profile = WavelengthCalibration.load(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        saved = CameraSetting.from_dict(data["camera_setting"]) if "camera_setting" in data else None
        if saved is None:
            saved = replace(setting, camera_bin=int(profile.camera_bin),
                            frame_shape=tuple(profile.frame_shape))
        problems = tuple(setting.mismatch(saved))
        return StoredWavelength(profile, path, saved, problems, path.stat().st_mtime)

    # ── spectral resolution ────────────────────────────────
    def save_resolution(self, fwhm_nm: float, setting: CameraSetting, details: Optional[dict] = None) -> Path:
        """Write the median FWHM (nm) measured at ``setting`` (old file kept in history).

        Args:
            fwhm_nm: Median FWHM over the measured lines, nm.
            setting: Camera setting of the measured frame.
            details: Extra JSON-serialisable keys stored alongside (per-line values, fit settings).

        Raises:
            ValueError: ``fwhm_nm`` is not finite and > 0, or ``details`` holds NaN/inf.
        """
        fwhm_nm = float(fwhm_nm)
        if not math.isfinite(fwhm_nm) or fwhm_nm <= 0:
            raise ValueError(f"FWHM must be a positive number of nm, not {fwhm_nm}")
        path = self.resolution_path(setting)
        data = dict(details or {})
        data.update(format_version=1, units="nm", fwhm_nm=fwhm_nm,
                    camera_setting=setting.to_dict(), saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        text = json.dumps(data, indent=2, allow_nan=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.backup(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        return path

    def load_resolution(self, setting: CameraSetting) -> Optional[StoredResolution]:
        """Read the FWHM stored for ``setting``'s bin, or ``None`` if there is none.

        Callers must not use a result whose ``ok`` is False.

        Raises:
            ValueError, KeyError, OSError: the file exists but is not a valid measurement.
        """
        path = self.resolution_path(setting)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("units") != "nm":
            raise ValueError(f"{path}: units must be nm")
        fwhm_nm = float(data.pop("fwhm_nm"))
        if not math.isfinite(fwhm_nm) or fwhm_nm <= 0:
            raise ValueError(f"{path}: FWHM {fwhm_nm} is not a positive number of nm")
        saved = CameraSetting.from_dict(data.pop("camera_setting"))
        problems = tuple(setting.mismatch(saved))
        return StoredResolution(fwhm_nm, path, saved, problems, path.stat().st_mtime, data)
