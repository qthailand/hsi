"""Readiness of the production calibration profiles for the camera setting under test (read only).

The profiles are found and read with the application's own mechanisms - ``config.load_app_config``,
``CameraProfile.for_camera``, ``config.calibration_store`` (Smile, wavelength, FWHM with the older
folders as fallback), ``modules.dark.select_dark`` on the dark library, and
``SaturationProfileStore.lookup`` - but nothing is written: no ``profile.json`` refresh, no folder
creation, no profile generated or migrated. A missing or mismatching profile is reported with its
reason and blocks only the tests that need it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

READY, MISSING, MISMATCH, OPTIONAL_MISSING = "READY", "MISSING", "MISMATCH", "NOT USED"


@dataclass
class Readiness:
    """One profile item for the setting under test."""

    item: str
    state: str
    reason: str
    files: List[str] = field(default_factory=list)
    revision: Dict = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        """Whether tests needing it may run."""
        return self.state == READY

    def to_dict(self) -> dict:
        """JSON form."""
        return asdict(self)


def camera_profile_for(config, identity: dict):
    """The application's ``CameraProfile`` of ``identity`` (no ``profile.json`` written)."""
    from modules.contracts import CameraProfile
    return CameraProfile.for_camera(config.camera_profile_folder, config.camera_backend,
                                    identity.get("model"), identity.get("serial"))


def check_profiles(config, identity: dict) -> Dict[str, Readiness]:
    """Every profile item of ``identity``'s setting (bin, gain, exposure, format, frame shape)."""
    out: Dict[str, Readiness] = {}
    profile = camera_profile_for(config, identity)
    root = Path(profile.root)
    info = profile.read_info()
    if not root.exists():
        out["camera_profile"] = Readiness("camera_profile", MISSING,
                                          f"no camera profile folder {root} (calibrate in the application first)")
        for item in ("dark", "smile", "wavelength", "fwhm", "saturation"):
            out[item] = Readiness(item, MISSING, "no camera profile")
        return out
    wrong = [f"{k} {info.get(k)!r} != {identity.get(k)!r}" for k in ("model", "serial")
             if info.get(k) is not None and identity.get(k) is not None and info.get(k) != identity.get(k)]
    out["camera_profile"] = Readiness("camera_profile", MISMATCH if wrong else READY,
                                      "; ".join(wrong) or f"{root.name} (profile.json: {info.get('model')} "
                                                           f"{info.get('serial')})", [str(root)])
    camera_bin = int(identity["bin"])
    shape = tuple(identity.get("frame_shape") or ())

    # dark reference: the scan controller's lookup, without creating the folder
    if not Path(profile.dark_folder).exists() and not Path(config.dark_frame_folder).exists():
        out["dark"] = Readiness("dark", MISSING, "no dark folder")
    else:
        try:
            from modules.dark import DarkFrameLibrary, select_dark
            # the library creates a missing folder: only open folders that exist (read only)
            own, legacy = Path(profile.dark_folder), Path(config.dark_frame_folder)
            library = DarkFrameLibrary(folder=str(own if own.exists() else legacy), n_frames=config.dark_frame_n_frames,
                                       fallback_folder=str(legacy) if own.exists() and legacy.exists() else None,
                                       camera_model=profile.model, camera_serial=profile.serial)
            selection = select_dark(library, camera_bin, float(identity["exposure_ms"]), float(identity["gain"]),
                                    shape or None, temperature=identity.get("sensor_temperature_c"))
            path = library.find(camera_bin, float(identity["exposure_ms"]), float(identity["gain"]),
                                temperature=identity.get("sensor_temperature_c"))
            state = {"loaded": READY, "missing": MISSING}.get(selection.kind, MISMATCH)
            reason = {"loaded": f"dark for bin {camera_bin}, {identity['exposure_ms']:g} ms, gain {identity['gain']:g}",
                      "missing": f"no dark for bin {camera_bin}, {identity['exposure_ms']:g} ms, gain "
                                 f"{identity['gain']:g} (capture it in the application)"}.get(
                selection.kind, f"dark {selection.dark_shape} does not fit frames {shape}")
            revision = {}
            if selection.result is not None:
                revision = {"captured_at": selection.result.get("captured_at"),
                            "temperature": selection.result.get("temperature")}
            out["dark"] = Readiness("dark", state, reason, [str(path)] if path else [], revision)
        except Exception as exc:                               # noqa: BLE001 - reported
            out["dark"] = Readiness("dark", MISMATCH, f"dark lookup failed: {exc}")

    # Smile / keystone, wavelength, FWHM: the application's calibration store
    try:
        import config as app_config
        from modules.contracts import CameraSetting
        config.camera_profile = profile                        # in memory only (as the controller does)
        store = app_config.calibration_store(config)
        setting = CameraSetting(config.camera_backend, int(getattr(config, "camera_index", 0)), camera_bin,
                                bool(getattr(config, "raw_inverted", True)), shape)
        smile = store.find_smile(setting)
        out["smile"] = (Readiness("smile", READY, f"{Path(smile).name}", [str(smile)])
                        if smile else Readiness("smile", MISSING, f"no Smile model for bin {camera_bin}"))
        if smile:
            try:
                from modules.smile.calibration import load_calibration
                model = load_calibration(str(smile))
                out["smile"].revision = {"created": str(getattr(model, "created", None)),
                                         "image_shape": list(getattr(model, "image_shape", ()) or ())}
                if shape and tuple(getattr(model, "image_shape", shape)) != shape:
                    out["smile"].state, out["smile"].reason = MISMATCH, (
                        f"{Path(smile).name} is for {tuple(model.image_shape)}, frames are {shape}")
            except Exception as exc:                           # noqa: BLE001
                out["smile"] = Readiness("smile", MISMATCH, f"{Path(smile).name} unreadable: {exc}", [str(smile)])
        wavelength = store.load_wavelength(setting)
        out["wavelength"] = (Readiness("wavelength", READY, f"wavelength profile for bin {camera_bin}")
                             if wavelength is not None else
                             Readiness("wavelength", MISSING, f"no wavelength profile for bin {camera_bin}"))
        resolution = store.load_resolution(setting)
        needed = getattr(config, "resolution_source", "config") == "fwhm"
        out["fwhm"] = (Readiness("fwhm", READY, f"FWHM for bin {camera_bin}") if resolution is not None else
                       Readiness("fwhm", MISSING if needed else OPTIONAL_MISSING,
                                 f"no FWHM for bin {camera_bin}" + ("" if needed else " (resolution_source=config)")))
    except Exception as exc:                                   # noqa: BLE001 - reported
        for item in ("smile", "wavelength", "fwhm"):
            out.setdefault(item, Readiness(item, MISMATCH, f"calibration store failed: {exc}"))

    # saturation profile: what the acquisition engine will look up
    try:
        from modules.contracts import SaturationIdentity
        from modules.saturation_profile import SaturationProfileStore
        ident = SaturationIdentity(config.camera_backend, identity.get("model"), identity.get("serial"),
                                   identity.get("pixel_format"), int(identity["delivered_bits"]),
                                   int(identity["sensor_bits"]), camera_bin)
        found = SaturationProfileStore(profile.saturation_folder).lookup(ident, identity.get("gain"))
        record = found.record
        revision = ({"profile_id": record.profile_id(), "status": record.status, "level": record.level,
                     "gain_range": record.gain_range, "software": record.software} if record is not None else {})
        state = READY if found.usable else (MISSING if found.state == "missing" else MISMATCH)
        out["saturation"] = Readiness("saturation", state, found.reason,
                                      [str(found.path)] if Path(found.path).exists() else [], revision)
    except Exception as exc:                                   # noqa: BLE001 - reported (e.g. pre-2E code)
        out["saturation"] = Readiness("saturation", MISSING, f"saturation profile lookup failed: {exc}")
    return out


__all__ = ["Readiness", "check_profiles", "camera_profile_for", "READY", "MISSING", "MISMATCH", "OPTIONAL_MISSING"]
