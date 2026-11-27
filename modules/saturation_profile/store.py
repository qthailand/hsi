"""Saturation profiles of one camera on disk: save (atomic, with history), load, check, look up.

Layout and format: :mod:`modules.contracts.saturation_profile`. One active file per
configuration (``<stem>.json``); every file it replaces, and every measurement that did not
replace it, goes to ``history/<stem>.<YYYYmmdd-HHMMSS>[_n].json``.

Saving rules (:meth:`SaturationProfileStore.save`):

* a **verified** record becomes the active profile (the previous one moves to history);
* any other record (inconclusive, failed) becomes active only when the active one is not
  verified - a failed measurement never replaces a verified profile; it goes to history.

Writes are atomic: a unique temporary file in the same folder, then ``os.replace``; the old
active file is copied to history before it is replaced. No other calibration file is touched.

Lookups (:meth:`SaturationProfileStore.lookup`) return a :class:`ProfileLookup` whose
``state`` says what the host may do: only ``verified`` gives a level for confirmed clipping;
``missing``, ``inconclusive``, ``failed``, ``incompatible`` (the file is for another camera or
configuration), ``invalid`` (unreadable, unsupported version) and ``gain_outside`` (verified for
another gain range) all mean *not checked*.

Threading:
    Plain file I/O; one writer at a time per folder (the calibration window). Lookups only read.

Example:
    >>> store = SaturationProfileStore(profile.saturation_folder)          # doctest: +SKIP
    >>> store.lookup(identity, gain=1).state                               # doctest: +SKIP
    'verified'
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from modules.contracts.saturation_profile import SaturationIdentity, SaturationProfileRecord

logger = logging.getLogger(__name__)

STATE_VERIFIED = "verified"
STATE_MISSING = "missing"
STATE_INCONCLUSIVE = "inconclusive"
STATE_FAILED = "failed"
STATE_INCOMPATIBLE = "incompatible"
STATE_INVALID = "invalid"
STATE_GAIN_OUTSIDE = "gain_outside"


@dataclass(frozen=True)
class ProfileLookup:
    """What the store holds for one configuration.

    Attributes:
        state: See the module docstring; only ``verified`` is usable.
        record: The record read (``None`` when missing or unreadable).
        reason: One line for logs and tooltips.
        path: The active file looked at.
    """

    state: str
    record: Optional[SaturationProfileRecord]
    reason: str
    path: Path

    @property
    def usable(self) -> bool:
        """Whether the level may be used for confirmed clipping."""
        return self.state == STATE_VERIFIED


@dataclass(frozen=True)
class SaveResult:
    """Outcome of :meth:`SaturationProfileStore.save`.

    Attributes:
        path: Where the record was written (the active file or a history file).
        active: Whether it became the active profile.
        replaced: History copy of the profile it replaced, or ``None``.
    """

    path: Path
    active: bool
    replaced: Optional[Path]


def software_revision(root=None) -> str:
    """Git revision of the software (``.git/HEAD`` read directly), or ``"unknown"``."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    git = root / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            path = git / ref
            if path.exists():
                return path.read_text(encoding="utf-8").strip()[:12]
            for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0][:12]
            return "unknown"
        return head[:12]
    except OSError:
        return "unknown"


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write ``payload`` to a unique temporary file next to ``path``, then move it over ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _json_bytes(data: dict) -> bytes:
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")


class SaturationProfileStore:
    """The ``saturation/`` folder of one camera profile.

    Args:
        folder: ``CameraProfile.saturation_folder``; created on the first save.
    """

    def __init__(self, folder):
        self.folder = Path(folder)

    @property
    def history_folder(self) -> Path:
        """Replaced profiles and measurements that did not become active."""
        return self.folder / "history"

    def path_for(self, identity: SaturationIdentity) -> Path:
        """The active file of ``identity``."""
        return self.folder / f"{identity.file_stem()}.json"

    # ── reading ─────────────────────────────────────────────

    def read(self, path) -> SaturationProfileRecord:
        """One profile file.

        Raises:
            OSError: it cannot be read; ValueError / KeyError: it is not a supported profile.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return SaturationProfileRecord.from_dict(data)

    def load(self, identity: SaturationIdentity) -> ProfileLookup:
        """The active profile of ``identity``, checked against it (no gain check)."""
        path = self.path_for(identity)
        if not path.exists():
            return ProfileLookup(STATE_MISSING, None, f"no saturation profile for {identity.label()}", path)
        try:
            record = self.read(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return ProfileLookup(STATE_INVALID, None, f"{path.name} unreadable: {exc}", path)
        problems = record.identity.mismatch(identity)
        if problems:
            return ProfileLookup(STATE_INCOMPATIBLE, record,
                                 f"{path.name} is for another configuration: " + "; ".join(problems), path)
        if record.status != STATE_VERIFIED:
            why = "; ".join(record.reasons[:2]) or record.status
            return ProfileLookup(record.status, record, f"saturation profile {record.status}: {why}", path)
        return ProfileLookup(STATE_VERIFIED, record, record.provenance(), path)

    def lookup(self, identity: SaturationIdentity, gain) -> ProfileLookup:
        """:meth:`load`, and a verified profile is usable only for a gain in its verified range."""
        found = self.load(identity)
        if found.usable and not found.record.covers_gain(gain):
            low, high = found.record.gain_range
            return ProfileLookup(STATE_GAIN_OUTSIDE, found.record,
                                 f"gain {gain} outside the verified range {low:g}-{high:g} of {found.path.name}",
                                 found.path)
        return found

    def history(self, identity: Optional[SaturationIdentity] = None) -> List[Path]:
        """History files (of ``identity`` only, when given), oldest first."""
        if not self.history_folder.exists():
            return []
        prefix = f"{identity.file_stem()}." if identity is not None else ""
        return sorted(p for p in self.history_folder.glob(f"{prefix}*.json") if p.is_file())

    def active(self) -> List[Tuple[Path, Optional[SaturationProfileRecord], str]]:
        """Every active file: ``(path, record or None, error text)``."""
        out = []
        for path in sorted(self.folder.glob("saturation_*.json")) if self.folder.exists() else []:
            try:
                out.append((path, self.read(path), ""))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                out.append((path, None, str(exc)))
        return out

    # ── writing ─────────────────────────────────────────────

    def _history_path(self, stem: str, when: Optional[str] = None) -> Path:
        try:
            moment = datetime.datetime.fromisoformat(when) if when else datetime.datetime.now()
        except ValueError:
            moment = datetime.datetime.now()
        stamp = moment.strftime("%Y%m%d-%H%M%S")
        for n in range(1, 1000):
            path = self.history_folder / (f"{stem}.{stamp}.json" if n == 1 else f"{stem}.{stamp}_{n}.json")
            if not path.exists():
                return path
        raise OSError(f"no free history name for {stem} in {self.history_folder}")

    def save(self, record: SaturationProfileRecord) -> SaveResult:
        """Store ``record`` by the rules of the module docstring.

        Raises:
            OSError: the folder cannot be written (nothing is half written).
        """
        payload = _json_bytes(record.to_dict())                # fail before touching any file
        path = self.path_for(record.identity)
        current = self.load(record.identity)
        if not record.verified and current.state == STATE_VERIFIED:
            target = self._history_path(record.identity.file_stem(), record.measured_at)
            _atomic_write(target, payload)
            logger.info("Saturation measurement %s (%s) kept in history; the verified profile stays active",
                        record.profile_id(), record.status)
            return SaveResult(target, False, None)
        replaced = None
        if path.exists():                                      # keep it, byte for byte, before replacing it
            old = current.record.measured_at if current.record is not None else None
            replaced = self._history_path(record.identity.file_stem(), old)
            _atomic_write(replaced, path.read_bytes())
        _atomic_write(path, payload)
        logger.info("Saturation profile %s saved (%s, level %s)", path.name, record.status, record.level)
        return SaveResult(path, True, replaced)


__all__ = ["SaturationProfileStore", "ProfileLookup", "SaveResult", "software_revision", "STATE_VERIFIED",
           "STATE_MISSING", "STATE_INCONCLUSIVE", "STATE_FAILED", "STATE_INCOMPATIBLE", "STATE_INVALID",
           "STATE_GAIN_OUTSIDE"]
