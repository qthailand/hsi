"""Saturation level profile: the measured value a clipped raw pixel reads, per camera setting.

A *saturation profile* records what a saturated pixel of one camera actually reads, measured on
real frames, for one configuration: camera (backend, model, serial), pixel format, delivered and
sensor bit depth, and hardware bin. It is quality data only - clipping detection,
near-saturation and quality tracking. It never takes part in intensity correction, and it is
not the *nominal full scale* (``camera_bit_info().full_scale``), which stays the metadata /
radiometric-reference value.

| Parameter | Purpose |
|---|---|
| nominal full scale | metadata, intensity representation, radiometric reference keys |
| measured saturation level | clipping detection, near-saturation, quality tracking |

Files (``modules.saturation_profile.SaturationProfileStore`` writes them)::

    camera_profiles/<backend>/<model>_<serial>/saturation/
        saturation_<format>_d<delivered>s<sensor>_bin<N>.json     the active profile
        history/<same stem>.<YYYYmmdd-HHMMSS>[_n].json             replaced profiles, attempts

The delivered / sensor bits are in the name, so one pixel format read at two bit depths never
shares a file.

Statuses: ``verified`` (enough evidence: only these give a level to check frames against),
``inconclusive`` (too little or conflicting evidence) and ``failed`` (the measurement could
not run). A verified profile is valid only inside the gain range it was measured over.

Limits (also stored in every record): the level is *digital* clipping (the largest value the
frames reach), not sensor non-linearity or full-well saturation; with hardware binning a binned
pixel reads the level only when every sensor pixel of its block clipped, so a partly clipped
block reads less (the near-saturation count catches it).

Data conventions:
    Levels and plateau values in raw counts of the delivered frames; exposure in ms; gain in the
    backend's own units; times ISO 8601 local time.

Threading:
    Pure data (frozen dataclasses) and JSON conversion; no I/O.

Example:
    >>> identity = SaturationIdentity("poa", "Apollo-M MAX", "CAMP7161", "RAW16", 16, 12, 2)
    >>> identity.file_stem()
    'saturation_RAW16_d16s12_bin2'
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

#: Version of the profile files this code reads and writes.
SATURATION_PROFILE_FORMAT_VERSION = 1
STATUS_VERIFIED = "verified"
STATUS_INCONCLUSIVE = "inconclusive"
STATUS_FAILED = "failed"
STATUSES = (STATUS_VERIFIED, STATUS_INCONCLUSIVE, STATUS_FAILED)
#: The limits every record states (see the module docstring).
LIMITATIONS = (
    "digital clipping only: the largest value the frames reach, not sensor non-linearity or full-well saturation",
    "hardware binning: a binned pixel reads the level only when every sensor pixel of its block clipped; "
    "a partly clipped block reads less and is counted as near saturation",
)


def format_token(image_format) -> str:
    """Pixel format as a short file-name token: ``POAImgFormat.POA_RAW16`` -> ``RAW16``,
    ``Mono12Packed`` -> ``Mono12Packed``; ``unknown`` when empty."""
    text = str(image_format if image_format is not None else "").strip()
    text = text.rsplit(".", 1)[-1]
    text = re.sub(r"^POA_", "", text)
    text = re.sub(r"[^A-Za-z0-9]+", "", text)
    return text or "unknown"


@dataclass(frozen=True)
class SaturationIdentity:
    """The camera configuration a saturation level belongs to.

    Attributes:
        backend: Camera backend (lower case), e.g. ``"poa"``.
        model, serial: What the camera SDK reports (``None`` when unknown).
        pixel_format: :func:`format_token` of the delivered pixel format.
        delivered_bits: Bit depth of the delivered frames (``info.bit_depth``).
        sensor_bits: Real data bits before any scaling (``sensor_bit_depth()``).
        bin: Hardware bin.
    """

    backend: str
    model: Optional[str]
    serial: Optional[str]
    pixel_format: str
    delivered_bits: int
    sensor_bits: int
    bin: int

    def __post_init__(self):
        object.__setattr__(self, "backend", str(self.backend).lower())
        object.__setattr__(self, "pixel_format", format_token(self.pixel_format))
        for name in ("delivered_bits", "sensor_bits", "bin"):
            object.__setattr__(self, name, int(getattr(self, name)))

    def file_stem(self) -> str:
        """Profile file name without ``.json`` (camera folder holds backend / model / serial)."""
        return f"saturation_{self.pixel_format}_d{self.delivered_bits}s{self.sensor_bits}_bin{self.bin}"

    def label(self) -> str:
        """Short text for logs and tooltips."""
        return (f"{self.model or '?'} {self.serial or '?'} {self.pixel_format} "
                f"{self.sensor_bits}->{self.delivered_bits} bit, bin {self.bin}")

    def mismatch(self, other: "SaturationIdentity") -> List[str]:
        """What differs from ``other`` (empty: the same configuration)."""
        names = (("backend", "backend"), ("model", "model"), ("serial", "serial number"),
                 ("pixel_format", "pixel format"), ("delivered_bits", "delivered bit depth"),
                 ("sensor_bits", "sensor bit depth"), ("bin", "hardware bin"))
        return [f"{label} {getattr(other, name)!r} != {getattr(self, name)!r}"
                for name, label in names if getattr(self, name) != getattr(other, name)]

    def to_dict(self) -> dict:
        """JSON form."""
        return {"backend": self.backend, "model": self.model, "serial": self.serial,
                "pixel_format": self.pixel_format, "delivered_bits": self.delivered_bits,
                "sensor_bits": self.sensor_bits, "bin": self.bin}

    @classmethod
    def from_dict(cls, data: dict) -> "SaturationIdentity":
        """Inverse of :meth:`to_dict` (``KeyError`` / ``ValueError`` on a bad record)."""
        return cls(data["backend"], data.get("model"), data.get("serial"), data["pixel_format"],
                   int(data["delivered_bits"]), int(data["sensor_bits"]), int(data["bin"]))


@dataclass(frozen=True)
class PlateauValue:
    """One exact raw value many pixels sat on.

    Attributes:
        value: Raw counts.
        role: ``"clipping"`` (evidence that it is a clipping plateau) or ``"unconfirmed"``.
        reason: Why (one line).
        evidence: Per gain: ``{"gain", "first_exposure_ms", "counts": [[exposure_ms, mean pixels], ...]}``.
    """

    value: int
    role: str
    reason: str
    evidence: Tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        """JSON form."""
        return {"value": int(self.value), "role": self.role, "reason": self.reason,
                "evidence": [dict(e) for e in self.evidence]}

    @classmethod
    def from_dict(cls, data: dict) -> "PlateauValue":
        """Inverse of :meth:`to_dict`."""
        return cls(int(data["value"]), str(data["role"]), str(data.get("reason", "")),
                   tuple(dict(e) for e in data.get("evidence", ())))


@dataclass(frozen=True)
class SaturationProfileRecord:
    """One measurement of one configuration (an active profile or a history entry).

    Attributes:
        identity: The configuration measured.
        status: ``verified`` / ``inconclusive`` / ``failed``.
        level: Measured saturation level (``None`` unless verified).
        nominal_full_scale: The frames' nominal full scale, for reference only.
        plateaus: Every plateau value found, with its role and evidence.
        gain_range: ``(low, high)`` gains the level was verified over (``None`` unless verified).
        gains_tested: Gains measured.
        exposures_ms: Exposures measured (all gains).
        frames: Frames analysed.
        steps: Evidence per exposure step (gain, exposure, pixels, top values, plateau counts).
        reasons: Why the status is what it is (failure / inconclusive reasons, notes).
        measured_at: ISO time of the measurement.
        software: Git revision (or ``"unknown"``).
        firmware: Camera firmware when the SDK reports it.
        limitations: :data:`LIMITATIONS`.
        format_version: :data:`SATURATION_PROFILE_FORMAT_VERSION`.
    """

    identity: SaturationIdentity
    status: str
    level: Optional[int]
    nominal_full_scale: Optional[int]
    plateaus: Tuple[PlateauValue, ...] = ()
    gain_range: Optional[Tuple[float, float]] = None
    gains_tested: Tuple[float, ...] = ()
    exposures_ms: Tuple[float, ...] = ()
    frames: int = 0
    steps: Tuple[dict, ...] = ()
    reasons: Tuple[str, ...] = ()
    measured_at: str = ""
    software: str = "unknown"
    firmware: Optional[str] = None
    limitations: Tuple[str, ...] = LIMITATIONS
    format_version: int = SATURATION_PROFILE_FORMAT_VERSION

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"unknown saturation profile status {self.status!r}")
        if self.status == STATUS_VERIFIED:
            if self.level is None or int(self.level) <= 0:
                raise ValueError("a verified saturation profile needs a positive level")
            if self.gain_range is None:
                raise ValueError("a verified saturation profile needs the gain range it was verified over")
        elif self.level is not None:
            raise ValueError(f"a {self.status} saturation profile has no level")

    @property
    def verified(self) -> bool:
        """Whether the level may be used for confirmed clipping."""
        return self.status == STATUS_VERIFIED

    def covers_gain(self, gain) -> bool:
        """Whether ``gain`` lies in the verified gain range (``None`` gain: unknown, not covered)."""
        if not self.verified or gain is None:
            return False
        low, high = self.gain_range
        return float(low) - 1e-9 <= float(gain) <= float(high) + 1e-9

    def profile_id(self) -> str:
        """``<file stem>@<measured_at>``: names this measurement (ENVI header, provenance)."""
        return f"{self.identity.file_stem()}@{self.measured_at}"

    def provenance(self) -> str:
        """One line: what the level is and where it comes from."""
        if not self.verified:
            return f"saturation profile {self.profile_id()}: {self.status}"
        low, high = self.gain_range
        return (f"measured saturation profile {self.profile_id()}: saturated pixels read {self.level} "
                f"(gain {low:g}-{high:g})")

    def to_dict(self) -> dict:
        """JSON form (the file content)."""
        return {"format_version": self.format_version, "identity": self.identity.to_dict(),
                "status": self.status, "level": None if self.level is None else int(self.level),
                "nominal_full_scale": None if self.nominal_full_scale is None else int(self.nominal_full_scale),
                "plateaus": [p.to_dict() for p in self.plateaus],
                "gain_range": None if self.gain_range is None else [float(v) for v in self.gain_range],
                "gains_tested": [float(g) for g in self.gains_tested],
                "exposures_ms": [float(e) for e in self.exposures_ms], "frames": int(self.frames),
                "steps": [dict(s) for s in self.steps], "reasons": list(self.reasons),
                "measured_at": self.measured_at, "software": self.software, "firmware": self.firmware,
                "limitations": list(self.limitations)}

    @classmethod
    def from_dict(cls, data: dict) -> "SaturationProfileRecord":
        """Inverse of :meth:`to_dict`.

        Raises:
            ValueError: unsupported ``format_version`` or an inconsistent record.
            KeyError: a required field is missing.
        """
        if not isinstance(data, dict):
            raise ValueError("a saturation profile must be a JSON object")
        version = data.get("format_version")
        if version != SATURATION_PROFILE_FORMAT_VERSION:
            raise ValueError(f"unsupported saturation profile format_version {version!r} "
                             f"(this software reads {SATURATION_PROFILE_FORMAT_VERSION})")
        gain_range = data.get("gain_range")
        return cls(identity=SaturationIdentity.from_dict(data["identity"]), status=str(data["status"]),
                   level=None if data.get("level") is None else int(data["level"]),
                   nominal_full_scale=None if data.get("nominal_full_scale") is None
                   else int(data["nominal_full_scale"]),
                   plateaus=tuple(PlateauValue.from_dict(p) for p in data.get("plateaus", ())),
                   gain_range=None if gain_range is None else (float(gain_range[0]), float(gain_range[1])),
                   gains_tested=tuple(float(g) for g in data.get("gains_tested", ())),
                   exposures_ms=tuple(float(e) for e in data.get("exposures_ms", ())),
                   frames=int(data.get("frames", 0)), steps=tuple(dict(s) for s in data.get("steps", ())),
                   reasons=tuple(str(r) for r in data.get("reasons", ())),
                   measured_at=str(data.get("measured_at", "")), software=str(data.get("software", "unknown")),
                   firmware=data.get("firmware"), limitations=tuple(data.get("limitations", LIMITATIONS)),
                   format_version=int(version))

    def with_reasons(self, *reasons: str) -> "SaturationProfileRecord":
        """A copy with ``reasons`` appended."""
        return replace(self, reasons=tuple(self.reasons) + tuple(reasons))


__all__ = ["SATURATION_PROFILE_FORMAT_VERSION", "STATUS_VERIFIED", "STATUS_INCONCLUSIVE", "STATUS_FAILED",
           "STATUSES", "LIMITATIONS", "format_token", "SaturationIdentity", "PlateauValue",
           "SaturationProfileRecord"]
