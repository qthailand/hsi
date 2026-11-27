"""The saturation level of the camera's active configuration, from its verified profile.

:class:`ProfileLevelSource` is what the acquisition engine's saturation tracker asks for the
level frames are checked against: it builds the active :class:`SaturationIdentity` from the
backend (camera, pixel format, bits, bin), looks it up in the camera profile's ``saturation/``
folder with the current gain, and answers a :class:`modules.saturation_mask.LevelEvidence`:

* a **verified** profile for this exact configuration, with the gain inside its verified range:
  ``level`` set, ``source='measured'``, the provenance and the profile id;
* anything else: ``level=None`` (*not checked*) and why - ``missing``, ``inconclusive``,
  ``failed``, ``incompatible``, ``invalid``, ``gain_outside``; ``error`` when the camera or the
  file could not be read (the caller asks again a little later).

It reads one small JSON file per call: the caller caches the answer per configuration and asks
again only when that changes (bin, format, bits, gain, a saved profile). It never computes a
level from the bit depth.

Threading:
    No state of its own besides the store path; call from the acquisition worker.

Example:
    >>> source = ProfileLevelSource(profile.saturation_folder, "poa")     # doctest: +SKIP
    >>> source(backend, np.uint16).level                                  # doctest: +SKIP
    65504
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from modules.saturation_mask import LevelEvidence

from .sources import identity_of
from .store import SaturationProfileStore


class ProfileLevelSource:
    """Level evidence from verified saturation profiles (see the module docstring).

    Args:
        folder: The camera profile's ``saturation_folder``, or ``None`` (no camera profile:
            every frame is not checked).
        backend_name: The backend's name as the profile folders use it (``"poa"``, ...).
    """

    def __init__(self, folder, backend_name: str):
        self.store: Optional[SaturationProfileStore] = SaturationProfileStore(folder) if folder is not None else None
        self.backend_name = str(backend_name or "").lower()

    def identity(self, backend):
        """The active configuration's identity (raises when the backend cannot say)."""
        return identity_of(backend, self.backend_name)

    def __call__(self, backend, dtype) -> LevelEvidence:
        """The level for ``backend``'s frames now (``dtype``: the raw frames' dtype)."""
        if self.store is None:
            return LevelEvidence(None, "missing", "no camera profile: the saturation level is not measured")
        try:
            identity = self.identity(backend)
            gain = getattr(backend, "gain", None)
        except Exception as exc:                               # noqa: BLE001 - an SDK read failed: later
            return LevelEvidence(None, "error", f"camera configuration unreadable: {exc}")
        try:
            found = self.store.lookup(identity, gain)
        except OSError as exc:
            return LevelEvidence(None, "error", f"saturation profile unreadable: {exc}")
        record = found.record
        profile_id = record.profile_id() if record is not None else None
        status = record.status if record is not None else None
        if not found.usable:
            return LevelEvidence(None, found.state, found.reason, None, profile_id, status)
        dtype = np.dtype(dtype)
        if dtype.kind in "ui" and record.level > np.iinfo(dtype).max:
            return LevelEvidence(None, "incompatible", f"measured level {record.level} exceeds the {dtype} frames",
                                 None, profile_id, status)
        return LevelEvidence(int(record.level), "measured", found.reason, found.reason, profile_id, status)


__all__ = ["ProfileLevelSource"]
