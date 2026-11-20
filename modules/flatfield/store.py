"""Flat-field profiles kept per camera, and the rule that decides when one may be used.

Layout of a store folder (in the application: the camera profile's ``flatfield/``)::

    <folder>/
        flat_<YYYYMMDD>_<HHMMSS>[_<n>]/   one profile, never modified after it is written
            flatfield.json             format_version, id, created_at, acquired_at, key, ROI,
                                       sensor_temperature_c, info
            profile.npy                (lines, bands) float32   (save_factor_profile)
            profile.wavelength.npy     (bands,) float64 band centres, nm
            roi_mask.npy               bool (lines, samples) ROI of the flat-field scan

A profile belongs to a camera setting, a :class:`~modules.radiometric.reference.ReferenceKey`
(camera, bin, orientation, slit rows, bands, gain, corrections). Exposure, stage type and the
data full scale are not compared: on a linear sensor they scale every row alike, and the profile
is relative. The profile also belongs to the lamp and target geometry, which no file records:
after moving a lamp or changing the working distance a new profile must be made.

:meth:`FlatFieldStore.find` returns the newest profile whose setting fits a scan.

Threading:
    Plain file I/O on the calling thread; use one store from one thread at a time. A profile
    folder appears only once complete.

Example:
    >>> store = FlatFieldStore('camera_profiles/poa/Apollo-M_MAX_CAMP7161/flatfield')   # doctest: +SKIP
    >>> saved = store.save(result['profile'], key, mask=mask)                           # doctest: +SKIP
    >>> store.find(scan_key).profile.id == saved.id                                     # doctest: +SKIP
    True
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from modules.radiometric.core import load_factor_profile, save_factor_profile
from modules.radiometric.reference import ReferenceKey, _aware, _sync_files, _temperature

from .core import check_profile

logger = logging.getLogger(__name__)

__all__ = ["FlatFieldProfile", "FlatFieldStore", "FlatFieldMatch", "flatfield_mismatch",
           "FLATFIELD_FORMAT_VERSION"]

#: ``format_version`` of ``flatfield.json``.
FLATFIELD_FORMAT_VERSION = 1
#: Key values that scale every slit row alike and are therefore not compared.
IGNORED_KEY_FIELDS = ("exposure_ms", "stage_type", "full_scale")

_PROFILE_ID = re.compile(r"^flat_\d{8}_\d{6}(?:_\d+)?$")
_INFO_FILE = "flatfield.json"
_PROFILE_FILE = "profile.npy"
_MASK_FILE = "roi_mask.npy"


def _comparable(key: ReferenceKey) -> ReferenceKey:
    return dataclasses.replace(key, **{name: None for name in IGNORED_KEY_FIELDS})


def flatfield_mismatch(profile_key: ReferenceKey, scan_key: ReferenceKey) -> List[str]:
    """Differences that forbid using a flat-field profile of ``profile_key`` on a scan (empty = fits).

    As :meth:`ReferenceKey.mismatch`, without exposure, stage type and data full scale.
    """
    return _comparable(profile_key).mismatch(_comparable(scan_key))


def _key_problems(key: ReferenceKey) -> List[str]:
    problems = [f"{label} unknown" for label, value in (("camera bin", key.camera_bin), ("slit rows", key.lines))
                if value is None]
    if not key.wavelengths:
        problems.append("band wavelengths unknown")
    if key.dark_corrected is not True:
        problems.append("dark subtraction was " + ("off" if key.dark_corrected is False else "unknown")
                        + ": a flat-field profile needs dark-subtracted counts")
    return problems


@dataclass(frozen=True)
class FlatFieldProfile:
    """One stored flat-field profile (read from its ``flatfield.json``).

    Attributes:
        id: Folder name, ``flat_<YYYYMMDD>_<HHMMSS>[_<n>]`` (of the save time).
        folder: The profile's folder.
        created_at: When it was saved, local ISO 8601 time with UTC offset.
        acquired_at: When the flat-field target was scanned.
        key: Camera setting of the flat-field scan.
        span: First and last slit row measured.
        coverage: Share of the slit rows measured (the rest is interpolated or held).
        sensor_temperature_c: Mean sensor temperature of the flat-field scan (degrees C), or
            ``None`` when the camera gave none (or the profile predates the field).
        info: Further JSON data given when it was saved (target, seam rows, source, ...).
    """

    id: str
    folder: Path
    created_at: str
    acquired_at: str
    key: ReferenceKey
    span: Tuple[int, int]
    coverage: float
    sensor_temperature_c: Optional[float] = None
    info: dict = field(default_factory=dict)

    @property
    def created(self) -> datetime.datetime:
        """``created_at`` as an aware datetime."""
        return _aware(self.created_at)

    def profile(self) -> np.ndarray:
        """The profile ``(lines, bands)`` float32.

        Raises:
            ValueError: The stored file does not fit the key any more, or is not a valid profile.
        """
        values, wavelengths = load_factor_profile(self.folder / _PROFILE_FILE)
        values = check_profile(values, self.key.lines, len(self.key.wavelengths))
        if wavelengths is not None and not np.allclose(wavelengths, self.key.wavelengths, rtol=0, atol=0.01):
            raise ValueError(f"{self.id}: stored wavelengths do not match its key")
        return values

    def roi_mask(self) -> np.ndarray:
        """Boolean ROI ``(lines, samples)`` of the flat-field scan."""
        return np.load(self.folder / _MASK_FILE).astype(bool)

    @classmethod
    def _read(cls, folder: Path) -> "FlatFieldProfile":
        data = json.loads((folder / _INFO_FILE).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError(f"{_INFO_FILE} is not a JSON object")
        if data.get("format_version") != FLATFIELD_FORMAT_VERSION:
            raise ValueError(f"format_version {data.get('format_version')!r} is not {FLATFIELD_FORMAT_VERSION}")
        if data.get("id") not in (None, folder.name):
            raise ValueError(f"{_INFO_FILE} names {data.get('id')!r}, not its folder {folder.name!r}")
        created_at = str(data["created_at"])
        acquired_at = str(data.get("acquired_at") or created_at)
        _aware(created_at), _aware(acquired_at)
        roi, key, info = data.get("roi"), data.get("key"), data.get("info") or {}
        if not isinstance(roi, dict) or not isinstance(key, dict) or not isinstance(info, dict):
            raise ValueError("'roi', 'key' and 'info' must be objects")
        span = roi.get("span")
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError("'roi.span' is not [first, last]")
        return cls(id=folder.name, folder=folder, created_at=created_at, acquired_at=acquired_at,
                   key=ReferenceKey.from_dict(key), span=(int(span[0]), int(span[1])),
                   coverage=float(roi.get("coverage") or 0.0),
                   sensor_temperature_c=_temperature(data.get("sensor_temperature_c")), info=dict(info))


@dataclass(frozen=True)
class FlatFieldMatch:
    """Result of :meth:`FlatFieldStore.find`.

    Attributes:
        profile: The newest profile that fits the scan, or ``None``.
        rejected: ``{profile id: reasons}`` for the profiles that do not.
    """

    profile: Optional[FlatFieldProfile]
    rejected: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when a profile was found."""
        return self.profile is not None

    def reason(self) -> str:
        """One line for the user: the profile used, why none fits (the closest one), or that none exists."""
        if self.profile is not None:
            return f"flat-field {self.profile.id}"
        if not self.rejected:
            return "no flat-field profile"
        ref_id, reasons = min(self.rejected.items(), key=lambda item: len(item[1]))
        return f"{ref_id} not used: " + "; ".join(reasons)


class FlatFieldStore:
    """Flat-field profiles of one camera in one folder (see the module docstring for the layout).

    Args:
        folder: Store folder; created on the first save. The application passes the camera
            profile's ``flatfield_folder``.
    """

    def __init__(self, folder):
        self.folder = Path(folder)

    def save(self, profile, key: ReferenceKey, *, mask, coverage: Optional[float] = None,
             info: Optional[dict] = None, acquired_at=None,
             sensor_temperature_c: Optional[float] = None,
             created_at: Optional[datetime.datetime] = None) -> FlatFieldProfile:
        """Store a new profile.

        Args:
            profile: ``(lines, bands)`` profile (:func:`modules.flatfield.compute_flatfield`).
            key: Camera setting of the flat-field scan (bin, slit rows and bands known, dark
                subtraction on).
            mask: Boolean ROI ``(lines, samples)`` the profile was fitted on.
            coverage: Share of the slit rows measured; default: the rows of ``mask``.
            info: Further JSON-serialisable data (target, seam rows, source cube, ...).
            acquired_at: When the target was scanned (ISO text or datetime). Default: ``created_at``.
            sensor_temperature_c: Mean sensor temperature of the flat-field scan (degrees C,
                :func:`~modules.radiometric.reference.scan_temperature_c`); ``None`` or a camera
                "no reading" value (-999) is stored as ``null``.
            created_at: Save time (default: now, local time); names the folder.

        Raises:
            ValueError: The key is incomplete, or profile / mask do not fit it.
            OSError: The folder cannot be written.
        """
        problems = _key_problems(key)
        if problems:
            raise ValueError("Cannot store this flat-field profile: " + "; ".join(problems))
        profile = check_profile(profile, key.lines, len(key.wavelengths))
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2 or mask.shape[0] != key.lines or not mask.any():
            raise ValueError(f"ROI mask {mask.shape} must be (lines={key.lines}, samples) and select pixels")
        created = created_at or datetime.datetime.now().astimezone()
        if created.tzinfo is None:
            created = created.astimezone()
        if acquired_at is None:
            acquired = created
        elif isinstance(acquired_at, datetime.datetime):
            acquired = acquired_at if acquired_at.tzinfo is not None else acquired_at.astimezone()
        else:
            acquired = _aware(str(acquired_at).replace("Z", "+00:00"))
        info = dict(info or {})
        json.dumps(info)                                   # fail before writing anything
        rows = np.where(mask.any(axis=1))[0]
        record = {
            "format_version": FLATFIELD_FORMAT_VERSION,
            "created_at": created.isoformat(timespec="seconds"),
            "acquired_at": acquired.isoformat(timespec="seconds"),
            "key": key.to_dict(),
            "roi": {"span": [int(rows.min()), int(rows.max())], "pixels": int(mask.sum()),
                    "coverage": float(rows.size / mask.shape[0] if coverage is None else coverage),
                    "samples": int(mask.shape[1])},
            "sensor_temperature_c": _temperature(sensor_temperature_c),
            "info": info,
        }
        self.folder.mkdir(parents=True, exist_ok=True)
        base = "flat_" + created.strftime("%Y%m%d_%H%M%S")
        staging = Path(tempfile.mkdtemp(prefix=f".{base}.", dir=self.folder))
        try:
            save_factor_profile(staging / _PROFILE_FILE, profile, np.asarray(key.wavelengths))
            np.save(staging / _MASK_FILE, mask)
            for n in range(1, 1000):
                profile_id = base if n == 1 else f"{base}_{n}"
                target = self.folder / profile_id
                if target.exists():
                    continue
                record["id"] = profile_id
                (staging / _INFO_FILE).write_text(json.dumps(record, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
                _sync_files(staging)
                try:
                    staging.rename(target)
                except FileExistsError:
                    continue
                break
            else:
                raise OSError(f"no free profile name for {base} in {self.folder}")
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        logger.info("Flat-field profile %s stored for %s", profile_id, key.label())
        return self.load(profile_id)

    def load(self, profile_id: str) -> FlatFieldProfile:
        """Read one profile.

        Raises:
            ValueError: ``profile_id`` is malformed, or its ``flatfield.json`` is not a profile.
            OSError: It cannot be read (e.g. does not exist).
        """
        if not isinstance(profile_id, str) or not _PROFILE_ID.match(profile_id):
            raise ValueError(f"{profile_id!r} is not a flat-field profile id")
        try:
            return FlatFieldProfile._read(self.folder / profile_id)
        except OSError:
            raise
        except KeyError as exc:
            raise ValueError(f"{profile_id}: {_INFO_FILE} lacks {exc}") from None
        except Exception as exc:
            raise ValueError(f"{profile_id}: {exc}") from None

    def profiles(self) -> List[FlatFieldProfile]:
        """Every readable profile, newest first (unreadable folders are logged and skipped)."""
        found = []
        if not self.folder.is_dir():
            return found
        for path in self.folder.iterdir():
            if not path.is_dir() or not _PROFILE_ID.match(path.name):
                continue
            try:
                found.append(self.load(path.name))
            except (OSError, ValueError) as exc:
                logger.warning("Skipping flat-field profile %s: %s", path.name, exc)
        return sorted(found, key=lambda p: (p.created, p.id), reverse=True)

    def status(self, camera_bin: Optional[int] = None,
               now: Optional[datetime.datetime] = None) -> Tuple[str, str, Optional[FlatFieldProfile]]:
        """State of the flat-field correction for a status badge.

        Args:
            camera_bin: The bin selected now; the newest profile of that bin is reported.
                ``None``: the newest profile of any bin.
            now: Time the age is measured to (default: the current time).

        Returns:
            ``(state, text, profile)``: ``state`` ``"on"`` (a profile to report) or ``"off"``;
            ``text`` e.g. ``Flat-field: bin 4 - 503/512 rows - response 0.46-1.18 - 7 joint rows -
            2.0 h old`` (short: it shares a row of the scan window); ``profile`` the one reported,
            or ``None``.
            Whether it fits a scan is decided per scan (:meth:`find`: gain, bands, corrections too).
        """
        profiles = self.profiles()
        if not profiles:
            return "off", "Flat-field: no profile", None
        fitting = [p for p in profiles if camera_bin is None or p.key.camera_bin == int(camera_bin)]
        if not fitting:
            bins = sorted({p.key.camera_bin for p in profiles if p.key.camera_bin is not None})
            return "off", (f"Flat-field: no profile for bin {int(camera_bin)} "
                           f"(have bin {', '.join(str(b) for b in bins)})"), None
        profile = fitting[0]
        key = profile.key
        parts = [f"bin {key.camera_bin}", f"{int(round(profile.coverage * (key.lines or 0)))}/{key.lines} rows"]
        low, high = profile.info.get("response_min"), profile.info.get("response_max")
        if low is not None and high is not None:
            parts.append(f"response {float(low):.2f}-{float(high):.2f}")
        joints = profile.info.get("seam_rows") or []
        if joints:
            parts.append(f"{len(joints)} joint rows")
        steps = profile.info.get("steps") or []
        if steps:
            parts.append(f"{len(steps)} brightness step{'s' if len(steps) != 1 else ''} (!)")
        now = now or datetime.datetime.now().astimezone()
        hours = (now - _aware(profile.acquired_at)).total_seconds() / 3600.0
        parts.append(f"{hours:.1f} h old" if hours < 48 else f"{hours / 24:.0f} days old")
        return "on", "Flat-field: " + " - ".join(parts), profile

    def find(self, scan_key: ReferenceKey) -> FlatFieldMatch:
        """The newest profile whose camera setting fits the scan of ``scan_key``, with its array readable."""
        rejected: Dict[str, Tuple[str, ...]] = {}
        for candidate in self.profiles():
            problems = flatfield_mismatch(candidate.key, scan_key)
            if not problems:
                try:
                    candidate.profile()
                except Exception as exc:
                    rejected[candidate.id] = (f"cannot be read: {exc or type(exc).__name__}",)
                    continue
                return FlatFieldMatch(candidate, rejected)
            rejected[candidate.id] = tuple(problems)
        return FlatFieldMatch(None, rejected)
