"""Automatic radiometric calibration of a scan with the camera's active reference.

The scan's :class:`~modules.radiometric.reference.ReferenceKey` is built from its metadata and the
camera's active references are searched (:meth:`RadiometricReferenceStore.find`: same key,
reference at most ``max_age_h`` old, sensor temperature within the tolerance). The cube is then
multiplied by the reference's row-wise factors with the arithmetic of :func:`apply_rowwise_factors`
(``float32(counts) * factor``, rounded once), the values the Calibration window writes, so an
automatic and a manual calibration with the same factors are equal.

Two ways, both Qt-free:

* Frame by frame (the default of the Scan window): :func:`plan_frame_calibration` chooses the
  reference when the scan starts; the acquisition recorder multiplies every recorded, lambda-binned
  frame by its factors (``ScanRecorder.set_recording(factors=...)``; the displayed frames stay in
  counts); :func:`finish_frame_calibration` checks again with the final metadata (mean sensor
  temperature). The counts are never lost: :func:`counts_from_reflectance` rebuilds them exactly
  (the factors are finite and positive), for Save as Radiometric Reference, the Calibration window,
  or when the reference may not stay applied.
* After the scan: :func:`plan_auto_calibration` + :func:`apply_reference` on a cube recorded in
  counts (auto calibration ticked after the scan, or a reference whose factors have NaN bands).

When nothing fits, the reason is reported and the scan stays in counts.

Example:
    >>> plan, frame = plan_frame_calibration(store, start_metadata, wavelengths, lines)  # doctest: +SKIP
    >>> engine.set_recording(True, factors=frame.factors)                                # doctest: +SKIP
    >>> final, problems = finish_frame_calibration(frame, end_metadata)                  # doctest: +SKIP
    >>> header = reflectance_export_metadata(metadata, final.header_fields())            # doctest: +SKIP
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from modules.cube_io.writers import COUNT_HEADER_FIELDS

from .core import apply_rowwise_factors
from .reference import (DEFAULT_MAX_AGE_H, DEFAULT_TEMPERATURE_TOLERANCE_C, RadiometricReference,
                        RadiometricReferenceStore, ReferenceKey, ReferenceMatch, _aware, _temperature,
                        reference_problems, scan_temperature_c)

logger = logging.getLogger(__name__)

#: ``reflectance reference mode`` of a cube calibrated after the scan without the user.
AUTO_MODE = "automatic"
#: ... and of a cube whose frames were calibrated while recorded.
AUTO_FRAME_MODE = "automatic, frame by frame"
#: Largest share of the free memory the reflectance cube (float32) may take.
MEMORY_SHARE = 0.8


def scan_start_time(metadata: dict) -> Optional[datetime.datetime]:
    """The scan's ``started_at`` as an aware datetime, or ``None`` when absent or unreadable."""
    started = (metadata or {}).get("started_at")
    if not started:
        return None
    try:
        text = str(started).strip()
        if text.endswith("Z"):                      # Python 3.10 fromisoformat has no 'Z'
            text = text[:-1] + "+00:00"
        return _aware(text)
    except ValueError:
        return None


@dataclass(frozen=True)
class AutoCalibrationPlan:
    """What to do with a finished scan (:func:`plan_auto_calibration`).

    Attributes:
        key: The scan's reference key.
        match: Result of searching the active references.
        scan_started: Scan start (the reference age is measured to it), or ``None`` (now).
        scan_temperature_c: Mean sensor temperature of the scan, or ``None``.
    """

    key: ReferenceKey
    match: ReferenceMatch
    scan_started: Optional[datetime.datetime] = None
    scan_temperature_c: Optional[float] = None

    @property
    def ok(self) -> bool:
        """True when a reference may calibrate the scan."""
        return self.match.ok

    @property
    def reference(self) -> Optional[RadiometricReference]:
        """The reference to apply, or ``None``."""
        return self.match.reference

    @property
    def wanted(self) -> bool:
        """True when the camera has an active reference at all (applied or rejected): only then is
        a scan left in counts worth a message."""
        return self.match.ok or bool(self.match.rejected)

    def age_hours(self) -> Optional[float]:
        """Hours from the reference scan to this scan, or ``None`` without a reference."""
        ref = self.reference
        return None if ref is None else ref.age_hours(self.scan_started)

    def delta_t_c(self) -> Optional[float]:
        """Scan minus reference sensor temperature (degrees C), or ``None`` when either is unknown."""
        ref = self.reference
        scan_t = _temperature(self.scan_temperature_c)
        if ref is None or ref.sensor_temperature_c is None or scan_t is None:
            return None
        return float(scan_t - ref.sensor_temperature_c)

    def summary(self) -> str:
        """One line for the user: the reference applied with its age and temperature difference,
        or why none applies (:meth:`ReferenceMatch.reason`)."""
        ref = self.reference
        if ref is None:
            return self.match.reason()
        parts = [f"reference scanned {self.age_hours():.1f} h before"]
        dt = self.delta_t_c()
        if dt is not None:
            parts.append(f"dT {dt:+.1f} C")
        return f"calibrated with {ref.id} ({', '.join(parts)})"

    def header_fields(self) -> dict:
        """Export-header metadata of the calibrated cube (``SCAN_HEADER_FIELDS`` keys): the
        reference's table, id and rows, plus the mode, the reference scan time, its age, the
        temperature difference, the ROI tool and a warning for bands outside the table."""
        ref = self.reference
        if ref is None:
            return {}
        fields = dict(ref.header_fields())
        fields["reflectance_reference_mode"] = AUTO_MODE
        fields["reflectance_reference_scanned"] = ref.acquired_at
        fields["reflectance_reference_age_h"] = round(self.age_hours(), 3)
        dt = self.delta_t_c()
        if dt is not None:
            fields["reflectance_reference_delta_t_c"] = round(dt, 2)
        roi = ref.info.get("roi")
        if isinstance(roi, dict) and roi.get("tool"):
            fields["reflectance_reference_roi"] = str(roi["tool"])
        n_out = ref.info.get("out_of_range_bands")
        table = ref.info.get("table_range_nm")
        if n_out and isinstance(table, (list, tuple)) and len(table) == 2:
            fields["calibration_warnings"] = (f"{int(n_out)} bands outside the reference table "
                                              f"({float(table[0]):g}-{float(table[1]):g} nm) "
                                              "have no reflectance (NaN)")
        return fields


def plan_auto_calibration(store: RadiometricReferenceStore, metadata: dict,
                          wavelengths: Optional[Sequence[float]], lines: Optional[int], *,
                          max_age_h: Optional[float] = DEFAULT_MAX_AGE_H,
                          temperature_tolerance_c: Optional[float] = DEFAULT_TEMPERATURE_TOLERANCE_C
                          ) -> AutoCalibrationPlan:
    """Find the active reference that may calibrate a finished scan.

    Args:
        store: The camera's reference store.
        metadata: The scan's metadata, merged as for *Save as Radiometric Reference* (raw metadata,
            export-header metadata, the bit depth recorded at the scan start): the reference key
            is built the same way, so the same camera setting gives the same key.
        wavelengths: Band centres of the cube (nm).
        lines: Slit rows of the cube.
        max_age_h, temperature_tolerance_c: As in :func:`reference_problems`.
    """
    metadata = metadata or {}
    key = ReferenceKey.from_scan(metadata, wavelengths, lines)
    started = scan_start_time(metadata)
    scan_t = scan_temperature_c(metadata)
    match = store.find(key, scan_temperature_c=scan_t, now=started, max_age_h=max_age_h,
                       temperature_tolerance_c=temperature_tolerance_c)
    return AutoCalibrationPlan(key=key, match=match, scan_started=started, scan_temperature_c=scan_t)


@dataclass(frozen=True)
class FrameCalibration:
    """A reference chosen when a scan starts, multiplied into every recorded frame.

    Attributes:
        plan: The match made at the start (key and start temperature).
        factors: Read-only float32 ``(lines, bands)`` factors, all finite and positive (so the
            counts can always be rebuilt: :func:`counts_from_reflectance`).
    """

    plan: AutoCalibrationPlan
    factors: np.ndarray

    @property
    def reference(self) -> RadiometricReference:
        """The reference applied."""
        return self.plan.reference


def plan_frame_calibration(store: RadiometricReferenceStore, metadata: dict,
                           wavelengths: Optional[Sequence[float]], lines: Optional[int], *,
                           max_age_h: Optional[float] = DEFAULT_MAX_AGE_H,
                           temperature_tolerance_c: Optional[float] = DEFAULT_TEMPERATURE_TOLERANCE_C):
    """Choose the reference for a scan about to start (frame-by-frame calibration).

    ``metadata`` is the scan's metadata at its start (settings, corrections, start time and the
    start sensor temperature); the rules are those of :func:`plan_auto_calibration`.

    Returns:
        ``(plan, frame)``: the :class:`AutoCalibrationPlan`, and a :class:`FrameCalibration` when
        its reference may be multiplied into the frames, else ``None`` (no reference fits, or its
        factors have NaN / non-positive values, e.g. bands outside the reflectance table: such a
        scan is recorded in counts and can be calibrated after it).
    """
    plan = plan_auto_calibration(store, metadata, wavelengths, lines, max_age_h=max_age_h,
                                 temperature_tolerance_c=temperature_tolerance_c)
    if not plan.ok:
        return plan, None
    factors = np.array(plan.reference.factors(), dtype=np.float32, order="C")
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0):
        return plan, None
    factors.setflags(write=False)
    return plan, FrameCalibration(plan, factors)


def finish_frame_calibration(frame: FrameCalibration, metadata: dict, *,
                             max_age_h: Optional[float] = DEFAULT_MAX_AGE_H,
                             temperature_tolerance_c: Optional[float] = DEFAULT_TEMPERATURE_TOLERANCE_C):
    """Check a frame-by-frame calibration again when its scan has ended.

    The key is rebuilt from the final metadata and the rules applied with the scan's mean sensor
    temperature (start and end), exactly as after a scan calibrated afterwards.

    Returns:
        ``(plan, problems)``: the final :class:`AutoCalibrationPlan` (its header records the real
        temperature difference) and the reasons the reference may not stay applied (empty = it may).
    """
    reference = frame.reference
    key = ReferenceKey.from_scan(metadata or {}, frame.plan.key.wavelengths, frame.plan.key.lines)
    scan_t = scan_temperature_c(metadata or {})
    started = scan_start_time(metadata or {}) or frame.plan.scan_started
    problems = reference_problems(reference, key, scan_temperature_c=scan_t, now=started,
                                  max_age_h=max_age_h, temperature_tolerance_c=temperature_tolerance_c)
    plan = AutoCalibrationPlan(key=key, match=ReferenceMatch(None if problems else reference,
                                                             {} if not problems else {reference.id: tuple(problems)}),
                               scan_started=started, scan_temperature_c=scan_t)
    return plan, problems


def counts_from_reflectance(cube, factors, rows_per_chunk: int = 64) -> np.ndarray:
    """The counts a reflectance cube was made of: ``rint(cube / factors)`` as uint16.

    Exact for a cube made by :func:`apply_rowwise_factors` (or frame by frame) from integer counts
    with finite positive float32 factors: ``float32(c) * f`` rounded once differs from ``c * f`` by
    at most 2**-24 relative, so dividing again in float64 gives ``c`` within 0.004 counts.

    Args:
        cube: ``(lines, samples, bands)`` reflectance (float32), or one ``(lines, bands)`` frame.
        factors: ``(lines, bands)`` factors it was multiplied by.
        rows_per_chunk: Slit rows converted per step (no float64 copy of the whole cube).
    """
    factors = np.asarray(factors, dtype=np.float64)
    cube = np.asarray(cube)
    if cube.ndim == 2:
        cube3, f3 = cube[:, None, :], factors[:, None, :]
    else:
        cube3, f3 = cube, factors[:, None, :]
    if cube3.shape[0] != factors.shape[0] or cube3.shape[2] != factors.shape[1]:
        raise ValueError(f"factors {factors.shape} do not fit the cube {cube.shape}")
    out = np.empty(cube3.shape, dtype=np.uint16)
    step = max(1, int(rows_per_chunk))
    for y in range(0, cube3.shape[0], step):
        counts = np.rint(cube3[y:y + step].astype(np.float64) / f3[y:y + step])
        np.clip(np.nan_to_num(counts, nan=0.0), 0, 65535, out=counts)
        out[y:y + step] = counts
    return out[:, 0, :] if cube.ndim == 2 else out


def memory_problem(shape: Sequence[int], available: Optional[int] = None) -> Optional[str]:
    """Why a float32 reflectance cube of ``shape`` should not be made now (``None`` = it may).

    ``available`` is the free physical memory in bytes (default: asked from ``psutil``; without
    ``psutil`` nothing is refused here).
    """
    need = int(np.prod([int(n) for n in shape], dtype=np.int64)) * 4
    if available is None:
        try:
            import psutil
            available = int(psutil.virtual_memory().available)
        except Exception:
            return None
    if need > MEMORY_SHARE * available:
        return (f"not enough free memory for the reflectance cube ({need / 2 ** 30:.1f} GB needed, "
                f"{available / 2 ** 30:.1f} GB free)")
    return None


def apply_reference(cube, reference: RadiometricReference, wavelengths=None, progress_callback=None) -> np.ndarray:
    """The float32 reflectance cube of ``cube`` calibrated with ``reference`` (a new array; ``cube``
    is not changed). Heavy: run it off the GUI thread.

    Raises:
        ValueError: The reference's factors cannot be read or do not fit the cube.
    """
    return apply_rowwise_factors(cube, reference.factors(), reference.key.wavelengths or None, wavelengths,
                                 progress_callback=progress_callback)


def reflectance_export_metadata(metadata: Optional[dict], fields: dict) -> dict:
    """Export-header metadata of a calibrated cube: the scan's ``metadata`` without the raw-count
    fields (bit depths, full scale: a reflectance cube has none), plus ``fields``
    (:meth:`AutoCalibrationPlan.header_fields`). ``calibration_warnings`` of both are kept, each once.
    """
    counts = {key for key, _name, _kind in COUNT_HEADER_FIELDS}
    out = {k: v for k, v in (metadata or {}).items() if k not in counts}
    out = {k: v for k, v in out.items() if not k.startswith("reflectance_reference")}
    for key, value in fields.items():
        if value is None:
            continue
        if key == "calibration_warnings" and out.get(key):
            carried = str(out[key]).split("; ")
            value = "; ".join(carried + [w for w in str(value).split("; ") if w not in carried])
        out[key] = value
    return out
