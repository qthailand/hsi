"""Where one physical camera's calibration data lives: its *camera profile* folder.

Dark current, hot pixels, the Smile model and the wavelength/FWHM calibration all belong to
one sensor (and, for Smile and wavelength, to the spectrograph it is mounted on), so they are
kept together, one folder per camera::

    camera_profiles/<backend>/<model>_<serial>/
        profile.json        backend, model, serial, created / last opened, sensor pixel pitch
        dark/              dark_b<bin>_e<exposure>_g<gain>_t<temperature>.npz  (modules.dark)
        smile/              smile_model_bin<N>.npz, history/                    (modules.smile)
        wavelength/         wavelength_bin<N>.json, resolution_bin<N>.json, history/

``<model>`` and ``<serial>`` are what the camera SDK reports (characters outside
``A-Z a-z 0-9 . _ -`` become ``_``); a camera that reports neither gets ``unknown``. Files
stored before camera profiles existed (``dark_frames/<backend>``, ``smile_maps/<backend>``,
``calibration/wavelength/<backend>``) are not moved: hosts read them only as a fallback
(see :class:`~modules.contracts.CalibrationStore`), since nothing records which camera made them.

The sensor's pixel pitch is entered once by the user (most SDKs do not report it) and kept in
``profile.json`` under ``"sensor"``; earlier entries move to ``"sensor_history"``::

    "sensor": {"pixel_pitch_um": 9.0, "note": "datasheet", "entered_at": "2026-11-03T10:15:00",
               "model": "Apollo-M MAX", "serial": "CAMP7161"}

Data conventions:
    The pixel pitch is in micrometres **at bin 1** (the sensor's own pixel); at hardware bin N a
    frame pixel spans ``N * pixel_pitch_um``.

Threading:
    Pure path logic plus small JSON writes (:meth:`CameraProfile.write_info`,
    :meth:`CameraProfile.set_pixel_pitch`); call from one thread.

Example:
    >>> profile = CameraProfile.for_camera("camera_profiles", "POA", "Apollo-M MAX", "CAMP7161")
    >>> profile.name, profile.dark_folder.as_posix()
    ('Apollo-M_MAX_CAMP7161', 'camera_profiles/poa/Apollo-M_MAX_CAMP7161/dark')
    >>> sdk_pixel_pitch_um(9.0), sdk_pixel_pitch_um(1.0)   # 1.0 = "the SDK does not know"
    (9.0, None)
"""
from __future__ import annotations

import json
import logging
import math
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

__all__ = ["CameraProfile", "SensorInfo", "profile_name", "sdk_pixel_pitch_um", "PIXEL_PITCH_RANGE_UM"]

#: Name of the profile folder of a camera that reports neither model nor serial.
UNKNOWN_CAMERA = "unknown"
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
#: Pixel pitches accepted from the user, micrometres (anything else is a typing error).
PIXEL_PITCH_RANGE_UM = (0.5, 50.0)
#: ``CameraBackendInfo.pixel_size`` of a backend whose SDK does not report the pitch.
SDK_PLACEHOLDER_PITCH_UM = 1.0
#: Relative difference between the entered and the SDK pitch that is reported.
PITCH_MISMATCH_TOLERANCE = 0.01


def sdk_pixel_pitch_um(value) -> Optional[float]:
    """The pitch a camera SDK reported, or ``None`` when it is not a real value.

    Backends report ``1.0`` when the SDK has no pitch (GenICam, XIMEA); that, ``None``,
    non-numbers and values outside :data:`PIXEL_PITCH_RANGE_UM` mean "unknown".
    """
    try:
        pitch = float(value)
    except (TypeError, ValueError):
        return None
    low, high = PIXEL_PITCH_RANGE_UM
    if not math.isfinite(pitch) or pitch == SDK_PLACEHOLDER_PITCH_UM or not low <= pitch <= high:
        return None
    return pitch


@dataclass(frozen=True)
class SensorInfo:
    """The ``"sensor"`` block of ``profile.json``: the pixel pitch the user entered.

    Attributes:
        pixel_pitch_um: Pixel pitch at bin 1, micrometres.
        note: Where the value comes from (datasheet, sensor type), may be ``""``.
        entered_at: Local time it was saved (``YYYY-MM-DDTHH:MM:SS``).
        model: Camera model the value was entered for (``None`` if the SDK reported none).
        serial: Camera serial the value was entered for (``None`` if the SDK reported none).
    """

    pixel_pitch_um: float
    note: str = ""
    entered_at: str = ""
    model: Optional[str] = None
    serial: Optional[str] = None

    def to_dict(self) -> dict:
        """JSON-serialisable dict (the ``"sensor"`` block)."""
        return {"pixel_pitch_um": self.pixel_pitch_um, "note": self.note, "entered_at": self.entered_at,
                "model": self.model, "serial": self.serial}

    @classmethod
    def from_dict(cls, data) -> Optional["SensorInfo"]:
        """Inverse of :meth:`to_dict`; ``None`` when the block is missing or its pitch is not valid."""
        if not isinstance(data, dict):
            return None
        try:
            pitch = float(data.get("pixel_pitch_um"))
        except (TypeError, ValueError):
            return None
        low, high = PIXEL_PITCH_RANGE_UM
        if not (math.isfinite(pitch) and low <= pitch <= high):
            return None
        return cls(pitch, str(data.get("note") or ""), str(data.get("entered_at") or ""),
                   data.get("model") or None, data.get("serial") or None)


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

    def read_info(self) -> dict:
        """Contents of ``profile.json``; ``{}`` when it is missing, unreadable or not a JSON object.

        A UTF-8 byte-order mark (Notepad, PowerShell ``Set-Content -Encoding utf8``) is accepted.
        """
        try:
            data = json.loads(self.info_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _read_for_update(self) -> dict:
        """``profile.json`` as a dict to update and write back.

        A file that cannot be parsed (a broken hand edit) is copied to
        ``profile.json.unreadable-<time>`` first, so writing a fresh file never silently loses
        what the user entered.

        Raises:
            OSError: the file exists but cannot be read (e.g. locked by another program);
                nothing is written then.
        """
        try:
            text = self.info_path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            return {}
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, dict):
            return data
        backup = self.info_path.with_name(f"{self.info_path.name}.unreadable-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(self.info_path, backup)
        logger.warning("%s is not a JSON object; kept as %s and written anew", self.info_path, backup.name)
        return {}

    def _write(self, data: dict) -> Path:
        """Write ``profile.json`` atomically (temporary file, then rename)."""
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.info_path.with_name(self.info_path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.info_path)
        return self.info_path

    def write_info(self, **extra) -> Path:
        """Create the folder and write ``profile.json``.

        Keys already in the file that are not given here (the first ``created_at``, the
        ``sensor`` block the user entered, ...) are kept; an unparsable file is backed up first.

        Args:
            **extra: JSON-serialisable keys stored too (e.g. ``frame_width``).

        Returns:
            The path of ``profile.json``.

        Raises:
            OSError: ``profile.json`` exists but cannot be read, or cannot be written.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        data = self._read_for_update()
        data.update({"format_version": 1, "backend": self.backend, "model": self.model,
                     "serial": self.serial, "created_at": data.get("created_at", now),
                     "last_opened_at": now, **extra})
        return self._write(data)

    # ── sensor pixel pitch ─────────────────────────────────

    @property
    def sensor(self) -> Optional[SensorInfo]:
        """The pixel pitch the user entered for this camera, or ``None`` if there is none (re-read
        from ``profile.json`` on every access)."""
        return SensorInfo.from_dict(self.read_info().get("sensor"))

    @property
    def pixel_pitch_um(self) -> Optional[float]:
        """Pixel pitch at bin 1 in micrometres, or ``None`` until the user has entered it."""
        sensor = self.sensor
        return None if sensor is None else sensor.pixel_pitch_um

    def set_pixel_pitch(self, pitch_um: float, note: str = "") -> SensorInfo:
        """Save the pixel pitch (bin 1, micrometres) the user entered; the previous entry, if any,
        is appended to ``"sensor_history"``.

        Raises:
            ValueError: ``pitch_um`` is not a number within :data:`PIXEL_PITCH_RANGE_UM`.
            OSError: ``profile.json`` cannot be read or written.
        """
        try:
            pitch = float(pitch_um)
        except (TypeError, ValueError):
            raise ValueError(f"pixel pitch must be a number, got {pitch_um!r}") from None
        low, high = PIXEL_PITCH_RANGE_UM
        if not (math.isfinite(pitch) and low <= pitch <= high):
            raise ValueError(f"pixel pitch must be {low:g}-{high:g} µm, got {pitch_um!r}")
        sensor = SensorInfo(pitch, str(note or "").strip(), time.strftime("%Y-%m-%dT%H:%M:%S"),
                            self.model, self.serial)
        data = self._read_for_update()
        if isinstance(data.get("sensor"), dict):
            data["sensor_history"] = list(data.get("sensor_history") or []) + [data["sensor"]]
        data.setdefault("format_version", 1)
        data.update(backend=self.backend, model=self.model, serial=self.serial)
        data["sensor"] = sensor.to_dict()
        self._write(data)
        return sensor

    def pitch_warnings(self, sdk_pitch_um: Optional[float] = None, sharing: bool = True) -> List[str]:
        """Reasons to double-check this profile's pixel pitch (empty when there are none).

        * the profile may be shared: the SDK reported no serial (or nothing at all), so
          other cameras of the same model (or any camera) use this folder too;
        * the pitch was entered for another model/serial (``profile.json`` copied by hand);
        * the SDK reports a pitch that differs by more than 1 % from the entered one.

        Args:
            sdk_pitch_um: :func:`sdk_pixel_pitch_um` of the open camera, if known.
            sharing: Include the "profile may be shared" reasons (the first point).
        """
        warnings = []
        if sharing and not self.known:
            warnings.append("the camera reports neither model nor serial: every such camera shares this profile")
        elif sharing and self.serial is None:
            warnings.append(f"the camera reports no serial: every {self.model} shares this profile")
        sensor = self.sensor
        if sensor is None:
            return warnings
        if (sensor.model, sensor.serial) != (self.model, self.serial):
            warnings.append(f"the pixel pitch was entered for {sensor.model or '?'} #{sensor.serial or '?'}, "
                            f"not for this camera")
        if sdk_pitch_um and abs(sensor.pixel_pitch_um - sdk_pitch_um) > PITCH_MISMATCH_TOLERANCE * sdk_pitch_um:
            warnings.append(f"the camera reports {sdk_pitch_um:g} µm, the profile has {sensor.pixel_pitch_um:g} µm")
        return warnings
