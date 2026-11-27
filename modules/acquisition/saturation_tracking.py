"""Saturation in the acquisition loop: basic detection, tracking on/off, level evidence, mailboxes, cost.

Basic detection (always on): the worker calls :meth:`SaturationTracker.count` with every acquired
raw frame, before any correction. It counts clipped (``raw >= level``) and near-saturated
(``raw >= NEAR_RATIO * level``) pixels in one pass (:class:`modules.saturation_mask.RawCounter`)
and keeps, since the last :meth:`SaturationTracker.reset_peak` (every Start scan, and the Clipped
tile's reset button), the peak clipped / near share and the frames that clipped. The GUI reads
the newest :class:`RawSaturationStats` from a mailbox (:meth:`SaturationTracker.latest_raw`) at
its own refresh rate; nothing is queued per frame and no mask is built. A frame without a
stated level is *not checked* (never "0 % clipped"); a window with some frames not checked is
*partial*.

Full tracking (optional, off by default): the worker calls :meth:`SaturationTracker.track` once per corrected frame with the frame's own
correction snapshot (``PreprocessingRuntime.last_prepared``) and binner; the result goes to the
recorder with the frame and the newest one into a mailbox the GUI reads when it wants
(:meth:`SaturationTracker.latest`) - nothing is queued per frame.

A frame is *checked* only when tracking is on, the backend's frames go through the shared
preprocessing (a backend that corrects its own frames hides the raw values) and the camera's
active configuration has a **verified saturation profile** (the ``level_source``, in the
application ``modules.saturation_profile.ProfileLevelSource``: the value a clipped pixel was
measured to read, for this camera, pixel format, bits, bin and gain range); otherwise its
record is ``None`` (*not checked*), never an empty mask. Basic detection and full tracking use
that one level. The nominal full scale of the bit depth is never used for it.

Level provenance: the level is asked again whenever the camera configuration it depends on
changes (frame dtype, bin, gain, ``info.bit_depth`` / ``info.image_format``, or
:meth:`reset_level` after a bin / parameter change, a restart or a saved profile), and after
:data:`LEVEL_RETRY_S` when a lookup failed (a transient failure does not leave the rest of the
session unchecked); never per frame. Each scan keeps the levels it was checked against with
their provenance and profile id (:meth:`SaturationTracker.scan_info`).

Mailbox: :meth:`SaturationTracker.latest` is the newest *tracked* frame, not necessarily the
frame on screen. A consumer that shows a mask over a displayed frame (Phase 4) must pair them
by the same frame identity, never by reading the display mailbox and this one separately.

Cost: :meth:`SaturationTracker.note_frame` is told how long the worker spent on each frame
(read to recorder), so :meth:`live_info` and :meth:`scan_info` can set the mask's time against
the whole frame's and the camera's frame period, with tracking on or off.

Threading:
    ``track`` / ``note_frame`` / ``reset_level`` on the worker; ``set_enabled``, ``latest``,
    ``live_info``, ``start_scan`` and ``scan_info`` from any thread (one internal lock).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import numpy as np

from modules.saturation_mask import (STATUS_CHECKED, STATUS_NOT_CHECKED, STATUS_PARTIAL, FrameSaturation,
                                     LevelEvidence, RawCounter, SaturationPropagator, SaturationRecord,
                                     near_threshold)

logger = logging.getLogger(__name__)

#: Seconds of frames :meth:`SaturationTracker.live_info` averages over.
LIVE_WINDOW_S = 2.0
#: Seconds before a failed level query (SDK error) is asked again.
LEVEL_RETRY_S = 1.0


class _NoProfiler:
    @staticmethod
    def time(_name):
        return nullcontext()


_NO_PROFILER = _NoProfiler()


@dataclass(frozen=True)
class SaturationLatest:
    """The newest frame's saturation (the mailbox).

    Attributes:
        sequence: Increases by one per frame the tracker saw (checked or not).
        status: ``'checked'``, ``'not_checked'`` or ``'off'`` (tracking switched off).
        reason: Why it is not checked, or the level's provenance when it is.
        saturation: :class:`~modules.saturation_mask.FrameSaturation` (``None`` unless checked).
        level: Full scale used (``None`` unless checked).
        provenance: Where that level comes from (the backend's statement).
    """

    sequence: int
    status: str
    reason: str
    saturation: Optional[FrameSaturation] = None
    level: Optional[int] = None
    provenance: Optional[str] = None


@dataclass(frozen=True)
class RawSaturationStats:
    """Basic detection mailbox: the newest acquired frame and the totals since the last reset.

    Attributes:
        sequence: Increases by one per acquired frame (checked or not) and per reset.
        status: ``'checked'`` or ``'not_checked'`` (this frame).
        reason: Why it is not checked, or the level's provenance when it is.
        pixels: Raw pixels of the frame.
        clipped, near: Its pixels ``>= level`` and ``>= near_level`` (``near`` includes clipped).
        level, near_level, provenance: Thresholds used and where the level comes from
            (``None`` unless checked).
        count_ms: Wall-clock time of the count (``None`` unless checked).
        frames, checked_frames: Frames since the reset, and those checked.
        saturated_frames: Checked frames with at least one clipped pixel.
        near_frames: Checked frames with at least one near-saturated pixel.
        peak_clipped_percent, peak_near_percent: Highest shares (%) over the checked frames
            since the reset (``None`` before one).
        profile_id: The saturation profile the level comes from (``None`` unless checked).
    """

    sequence: int
    status: str
    reason: str
    pixels: int = 0
    clipped: int = 0
    near: int = 0
    level: Optional[int] = None
    near_level: Optional[int] = None
    provenance: Optional[str] = None
    count_ms: Optional[float] = None
    frames: int = 0
    checked_frames: int = 0
    saturated_frames: int = 0
    near_frames: int = 0
    peak_clipped_percent: Optional[float] = None
    peak_near_percent: Optional[float] = None
    profile_id: Optional[str] = None

    @property
    def checked(self) -> bool:
        """Whether this frame was checked."""
        return self.status == STATUS_CHECKED

    @property
    def clipped_percent(self) -> Optional[float]:
        """Clipped share of this frame (%), ``None`` unless checked."""
        return 100.0 * self.clipped / self.pixels if self.checked and self.pixels else None

    @property
    def near_percent(self) -> Optional[float]:
        """Near-saturated share of this frame (%, clipped included), ``None`` unless checked."""
        return 100.0 * self.near / self.pixels if self.checked and self.pixels else None

    @property
    def window_status(self) -> str:
        """Since the reset: ``'checked'`` (every frame), ``'partial'`` or ``'not_checked'`` (none)."""
        if self.checked_frames == 0:
            return STATUS_NOT_CHECKED
        return STATUS_CHECKED if self.checked_frames == self.frames else STATUS_PARTIAL


class _Since:
    """Basic-detection totals since a reset."""

    __slots__ = ("frames", "checked", "saturated", "near", "peak_clipped", "peak_near")

    def __init__(self):
        self.frames = self.checked = self.saturated = self.near = 0
        self.peak_clipped = self.peak_near = None

    def add(self, clipped_percent, near_percent):
        """Count one frame; ``None`` percentages: not checked."""
        self.frames += 1
        if clipped_percent is None:
            return
        self.checked += 1
        self.saturated += clipped_percent > 0
        self.near += near_percent > 0
        self.peak_clipped = clipped_percent if self.peak_clipped is None else max(self.peak_clipped, clipped_percent)
        self.peak_near = near_percent if self.peak_near is None else max(self.peak_near, near_percent)

    def status(self) -> str:
        """``'checked'`` (every frame), ``'partial'`` or ``'not_checked'`` (none)."""
        if self.checked == 0:
            return STATUS_NOT_CHECKED
        return STATUS_CHECKED if self.checked == self.frames else STATUS_PARTIAL


class RawCount(NamedTuple):
    """Basic detection of one frame: ``clipped`` / ``near`` pixels of ``pixels`` against ``level``
    (from the saturation profile ``profile_id``). Indexes as ``(clipped, near, level)`` first."""

    clipped: int
    near: int
    level: int
    pixels: int
    profile_id: Optional[str] = None


def _no_level(backend, dtype) -> LevelEvidence:
    """Default level source: none (frames are not checked)."""
    return LevelEvidence(None, "missing", "no saturation profile source: the saturation level is not measured")


class _Totals:
    """Sum / max / count of one quantity."""

    __slots__ = ("n", "total", "peak")

    def __init__(self):
        self.n, self.total, self.peak = 0, 0.0, 0.0

    def add(self, value: float):
        """Count one value."""
        self.n += 1
        self.total += value
        self.peak = max(self.peak, value)

    def mean(self):
        """Mean of the values, or ``None`` without any."""
        return self.total / self.n if self.n else None


class SaturationTracker:
    """Per-frame saturation of the acquisition loop (see the module docstring).

    Args:
        backend: The camera backend (its configuration: bin, gain, bit depths, pixel format).
        shared_preprocessing: Whether its frames go raw through ``modules.preprocessing``.
        enabled: Track from the start.
        level_source: ``(backend, dtype) -> LevelEvidence``: the level for the active
            configuration (``modules.saturation_profile.ProfileLevelSource``); ``None``: none,
            every frame is not checked.
    """

    def __init__(self, backend, shared_preprocessing: bool, enabled: bool = False,
                 level_source: Optional[Callable] = None):
        self.backend = backend
        self.level_source = level_source if level_source is not None else _no_level
        self.shared_preprocessing = bool(shared_preprocessing)
        self._enabled = bool(enabled)
        self._propagator = SaturationPropagator()
        self._evidence_key = None
        self._evidence: Optional[LevelEvidence] = None
        self._evidence_time = 0.0                    # monotonic time of the last query
        self._evidence_generation = 0                # bumped by reset_level
        self._logged_evidence = None                 # the last evidence logged (log on change only)
        self._frame_outcome = None                   # (level, provenance) or the not-checked reason
        self._counter = RawCounter(self._propagator.near_ratio)
        self._frame_count: Optional[RawCount] = None   # of the frame counted last
        self._frame_pixels = 0
        self._count_ms: Optional[float] = None
        self._count_error_logged = False
        self._since = _Since()
        self._raw_sequence = 0
        self._raw_latest: Optional[RawSaturationStats] = None
        self._lock = threading.Lock()
        self._sequence = 0
        self._latest: Optional[SaturationLatest] = None
        self._frame_ms: Optional[float] = None       # the last track() of this frame, for note_frame
        self._frame_record = None
        self._live: deque = deque()                  # (t, sat_ms or None, busy_ms, near, clipped)
        self._scan = None                            # per-recording totals (start_scan)
        self._error_logged = False

    # ── switches (any thread) ───────────────────────────────

    @property
    def enabled(self) -> bool:
        """Whether frames are being checked."""
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Track (True) or skip (False) from the next frame on."""
        enabled = bool(enabled)
        if enabled != self._enabled:
            logger.info("Saturation tracking %s", "on" if enabled else "off")
        self._enabled = enabled
        with self._lock:
            self._live.clear()                       # the meter shows one setting at a time

    def reset_level(self) -> None:
        """Ask the backend for its level again on the next frame (bin / parameters changed,
        acquisition restarted)."""
        self._evidence_generation += 1

    def reset_peak(self) -> None:
        """Start the basic-detection totals again (peak, saturated frames) from the next frame:
        every Start scan, and the Clipped tile's reset button. The mailbox shows it at once."""
        with self._lock:
            self._since = _Since()
            latest = self._raw_latest
            if latest is not None:
                self._raw_sequence += 1
                self._raw_latest = RawSaturationStats(
                    self._raw_sequence, latest.status, latest.reason, latest.pixels, latest.clipped, latest.near,
                    latest.level, latest.near_level, latest.provenance, latest.count_ms,
                    profile_id=latest.profile_id)

    # ── worker ──────────────────────────────────────────────

    def warm_up(self) -> None:
        """Compile / load the count kernels and start their threads before the first frame
        (on the worker, before acquisition starts). Never raises."""
        t0 = time.perf_counter()
        try:
            self._counter.warm_up()
        except Exception:                            # noqa: BLE001 - the first frame then compiles
            logger.exception("Saturation count warm-up failed")
            return
        logger.info("Saturation count ready in %.0f ms", (time.perf_counter() - t0) * 1000.0)

    def count(self, frame, profiler=None):
        """Basic detection of one acquired raw frame: ``(clipped, near, level)``, or ``None`` when
        it is not checked (no raw values, no stated level, the count failed). Updates the
        totals and the mailbox. Never raises; the frame is not modified."""
        self._frame_count, self._count_ms = None, None
        pixels = self._frame_pixels = int(np.size(frame))
        if not self.shared_preprocessing:
            self._post_raw(STATUS_NOT_CHECKED, "the camera backend corrects its frames itself (no raw values)", pixels)
            return None
        evidence = self.evidence(frame)
        if not evidence.trusted:
            self._post_raw(STATUS_NOT_CHECKED, evidence.reason, pixels)
            return None
        profiler = profiler if profiler is not None else _NO_PROFILER
        t0 = time.perf_counter()
        try:
            with profiler.time("sat_count"):
                clipped, near = self._counter.count(frame, evidence.level)
        except Exception as exc:                     # noqa: BLE001 - detection must never stop acquisition
            if not self._count_error_logged:
                logger.exception("Saturation count failed; frames are not checked until it works")
                self._count_error_logged = True
            self._post_raw(STATUS_NOT_CHECKED, f"saturation count failed: {exc}", pixels)
            return None
        self._count_error_logged = False
        self._count_ms = (time.perf_counter() - t0) * 1000.0
        self._frame_count = RawCount(int(clipped), int(near), int(evidence.level), pixels, evidence.profile_id)
        self._post_raw(STATUS_CHECKED, evidence.provenance or evidence.reason, pixels, clipped, near, evidence)
        return self._frame_count

    def _post_raw(self, status, reason, pixels, clipped=0, near=0, evidence=None):
        checked = status == STATUS_CHECKED and pixels > 0
        clipped_percent = 100.0 * clipped / pixels if checked else None
        near_percent = 100.0 * near / pixels if checked else None
        level = evidence.level if checked else None
        with self._lock:
            since = self._since
            since.add(clipped_percent, near_percent)
            self._raw_sequence += 1
            self._raw_latest = RawSaturationStats(
                self._raw_sequence, STATUS_CHECKED if checked else STATUS_NOT_CHECKED, reason, pixels, clipped, near,
                level, near_threshold(level, self._counter.near_ratio) if checked else None,
                evidence.provenance if checked else None, self._count_ms if checked else None,
                since.frames, since.checked, since.saturated, since.near, since.peak_clipped, since.peak_near,
                profile_id=evidence.profile_id if checked else None)

    def _configuration_key(self, frame):
        """What the level depends on: frame dtype, bin, delivered bits / pixel format, resets."""
        backend = self.backend
        try:
            info = backend.info
            bits, image_format = getattr(info, "bit_depth", None), getattr(info, "image_format", None)
        except Exception:                            # noqa: BLE001 - info unreadable: ask every time
            bits = image_format = object()
        try:
            gain = getattr(backend, "gain", None)
        except Exception:                            # noqa: BLE001 - unreadable: ask every time
            gain = object()
        return (np.asarray(frame).dtype.str, getattr(backend, "current_bin", None), bits, image_format, gain,
                self._evidence_generation)

    def evidence(self, frame) -> LevelEvidence:
        """The level evidence for frames like ``frame``: asked again when the camera configuration
        changes, and :data:`LEVEL_RETRY_S` after a failed query; logged when it changes."""
        key = self._configuration_key(frame)
        now = time.monotonic()
        stale = (key != self._evidence_key or self._evidence is None
                 or (self._evidence.transient and now - self._evidence_time >= LEVEL_RETRY_S))
        if stale:
            try:
                self._evidence = self.level_source(self.backend, np.asarray(frame).dtype)
            except Exception as exc:                 # noqa: BLE001 - a broken source: not checked, retry
                self._evidence = LevelEvidence(None, "error", f"saturation level lookup failed: {exc}")
            self._evidence_key, self._evidence_time = key, now
            shown = (self._evidence.level, self._evidence.source, self._evidence.reason)
            if shown != self._logged_evidence:
                self._logged_evidence = shown
                if self._evidence.trusted:
                    logger.info("Saturation level: %s (%s)", self._evidence.level, self._evidence.reason)
                else:
                    logger.warning("Saturation not checked: %s", self._evidence.reason)
        return self._evidence

    def track(self, frame, prepared, binner, row_crop=(0, 0), profiler=None,
              counts=None) -> Optional[SaturationRecord]:
        """Saturation of one corrected frame, or ``None`` when it is not checked.

        Args:
            frame: The raw frame the intensities were computed from (not modified).
            prepared: ``PreprocessingRuntime.last_prepared`` of that frame (its dark-reference
                snapshot, hot-pixel switch, orientation and model).
            binner: The binner the frame was binned with.
            row_crop: Keystone rows its binned line loses (``keystone_row_margin``).
            profiler: The loop's profiler (spans ``sat_total``, ``sat_pack`` and those of
                :meth:`SaturationPropagator.frame`).
            counts: This frame's :meth:`count` result: used as the mask's fast path (a frame
                with nothing near saturation is not read again) when it was counted against
                the same level.

        Never raises: a failure is logged once and the frame is not checked.
        """
        self._frame_ms, self._frame_record, self._frame_outcome = None, None, None
        if not self._enabled:
            self._post("off", "saturation tracking is off")
            return None
        if not self.shared_preprocessing:
            self._post(STATUS_NOT_CHECKED, "the camera backend corrects its frames itself (no raw values)")
            return None
        evidence = self.evidence(frame)
        if not evidence.trusted:
            self._post(STATUS_NOT_CHECKED, evidence.reason)
            return None
        if prepared is None or prepared.frame is not frame:
            self._post(STATUS_NOT_CHECKED, "no correction snapshot for this frame")
            return None
        profiler = profiler if profiler is not None else _NO_PROFILER
        known = counts[:2] if counts is not None and counts[2] == evidence.level else None
        t0 = time.perf_counter()
        try:
            with profiler.time("sat_total"):
                result = self._propagator.frame(frame, evidence.level, prepared.hot, prepared.enabled,
                                                prepared.calibration, prepared.inverted, binner,
                                                row_crop=row_crop, profiler=profiler, counts=known)
                with profiler.time("sat_pack"):
                    record = SaturationRecord.from_frame(result, binner.w)
        except Exception as exc:                     # noqa: BLE001 - the mask must never stop acquisition
            if not self._error_logged:
                logger.exception("Saturation mask failed; frames are not checked until it works")
                self._error_logged = True
            self._post(STATUS_NOT_CHECKED, f"saturation mask failed: {exc}")
            return None
        self._error_logged = False
        self._frame_ms = (time.perf_counter() - t0) * 1000.0
        self._frame_record = record
        self._frame_outcome = (evidence.level, evidence.provenance, evidence.profile_id)
        self._post(STATUS_CHECKED, f"level {evidence.level}: {evidence.provenance}", result, evidence.level,
                   evidence.provenance)
        return record

    def note_frame(self, busy_ms: float, recorded: bool) -> None:
        """The worker spent ``busy_ms`` on the frame just tracked (read to recorder);
        ``recorded``: the recorder kept it (counted in :meth:`scan_info`)."""
        sat_ms, record = self._frame_ms, self._frame_record
        near = bool(record is not None and record.any_near())
        clipped = bool(record is not None and record.any_clipped())
        count, count_ms, pixels = self._frame_count, self._count_ms, self._frame_pixels
        now = time.monotonic()
        with self._lock:
            self._live.append((now, sat_ms, float(busy_ms), near, clipped, count_ms))
            while self._live and now - self._live[0][0] > LIVE_WINDOW_S:
                self._live.popleft()
            scan = self._scan
            if scan is not None and recorded:
                scan["frames"] += 1
                scan["busy"].add(float(busy_ms))
                if count is not None and pixels:
                    scan["raw"].add(100.0 * count[0] / pixels, 100.0 * count[1] / pixels)
                    scan["count"].add(count_ms)
                else:
                    scan["raw"].add(None, None)
                if sat_ms is not None:
                    scan["checked"] += 1
                    scan["sat"].add(sat_ms)
                    scan["near_frames"] += near
                    scan["clipped_frames"] += clipped
                    levels = scan["levels"]
                    levels[self._frame_outcome] = levels.get(self._frame_outcome, 0) + 1
                else:
                    reason = self._latest.reason if self._latest is not None else "unknown"
                    scan["not_checked"][reason] = scan["not_checked"].get(reason, 0) + 1

    def _post(self, status, reason, saturation=None, level=None, provenance=None):
        with self._lock:
            self._sequence += 1
            self._latest = SaturationLatest(self._sequence, status, reason, saturation, level, provenance)

    # ── readers (any thread) ────────────────────────────────

    def latest_raw(self) -> Optional[RawSaturationStats]:
        """The basic-detection mailbox: newest frame and totals since the reset (``None`` before
        the first frame)."""
        with self._lock:
            return self._raw_latest

    def latest(self) -> Optional[SaturationLatest]:
        """The newest frame's saturation (``None`` before the first frame)."""
        with self._lock:
            return self._latest

    def live_info(self, period_ms: Optional[float] = None, period_source: Optional[str] = None) -> dict:
        """The last :data:`LIVE_WINDOW_S` seconds: ``frames``, ``tracking``, ``status`` /
        ``reason`` of the newest frame, ``sat_ms`` (mean mask time of the checked frames: the
        saturation *overhead*), ``busy_ms`` (mean worker time per frame, mask included),
        ``near_frames``, ``clipped_frames`` and, with ``period_ms`` (the frame period;
        ``period_source`` ``'reported'`` by the camera or ``'measured'`` between processed
        frames, an estimate that cannot show frames lost before the worker), ``load`` =
        ``busy_ms / period_ms`` and ``overhead`` = ``sat_ms / period_ms``."""
        with self._lock:
            live = list(self._live)
            latest = self._latest
        sat = [e[1] for e in live if e[1] is not None]
        busy = [e[2] for e in live]
        counted = [e[5] for e in live if e[5] is not None]
        info = {"tracking": self._enabled, "frames": len(live),
                "status": latest.status if latest else None, "reason": latest.reason if latest else "",
                "sat_ms": float(np.mean(sat)) if sat else None,
                "sat_max_ms": float(np.max(sat)) if sat else None,
                "count_ms": float(np.mean(counted)) if counted else None,
                "count_max_ms": float(np.max(counted)) if counted else None,
                "busy_ms": float(np.mean(busy)) if busy else None,
                "near_frames": sum(1 for e in live if e[3]),
                "clipped_frames": sum(1 for e in live if e[4])}
        if period_ms and info["busy_ms"] is not None:
            info["period_ms"] = float(period_ms)
            info["period_source"] = period_source or "measured"
            info["load"] = info["busy_ms"] / float(period_ms)
            info["overhead"] = None if info["sat_ms"] is None else info["sat_ms"] / float(period_ms)
        return info

    def start_scan(self) -> None:
        """Start the per-recording totals (recording armed) and :meth:`reset_peak`."""
        with self._lock:
            self._scan = {"tracking": self._enabled, "frames": 0, "checked": 0, "near_frames": 0,
                          "clipped_frames": 0, "sat": _Totals(), "busy": _Totals(), "levels": {},
                          "not_checked": {}, "raw": _Since(), "count": _Totals()}
        self.reset_peak()

    def scan_info(self, period_ms: Optional[float] = None, period_source: Optional[str] = None) -> dict:
        """Totals since :meth:`start_scan` over the recorded frames (before any trim):
        ``tracking`` (at the start), ``frames``, ``checked``, ``near_frames``, ``clipped_frames``,
        ``sat_ms`` / ``sat_max_ms`` and ``busy_ms`` / ``busy_max_ms`` (mean / max per frame),
        ``levels`` (``[{'level', 'provenance', 'frames'}]``: every level frames were checked
        against), ``not_checked`` (``{reason: frames}``), ``period_ms`` / ``period_source`` (as
        in :meth:`live_info`; ``busy_max_ms > period_ms`` means some frame took longer than the
        camera's period). Basic detection of the same frames: ``raw_status``, ``raw_checked``,
        ``raw_saturated_frames``, ``raw_near_frames``, ``raw_peak_clipped_percent``,
        ``raw_peak_near_percent`` and ``count_ms`` / ``count_max_ms``. ``{}`` before."""
        with self._lock:
            scan = self._scan
            if scan is None:
                return {}
            info = {k: scan[k] for k in ("tracking", "frames", "checked", "near_frames", "clipped_frames")}
            info.update(sat_ms=scan["sat"].mean(), sat_max_ms=scan["sat"].peak if scan["sat"].n else None,
                        busy_ms=scan["busy"].mean(), busy_max_ms=scan["busy"].peak if scan["busy"].n else None,
                        levels=[{"level": level, "provenance": provenance, "profile_id": profile_id, "frames": n}
                                for (level, provenance, profile_id), n in scan["levels"].items()],
                        not_checked=dict(scan["not_checked"]))
            raw, counted = scan["raw"], scan["count"]
            info.update(raw_status=raw.status(), raw_checked=raw.checked, raw_saturated_frames=raw.saturated,
                        raw_near_frames=raw.near, raw_peak_clipped_percent=raw.peak_clipped,
                        raw_peak_near_percent=raw.peak_near,
                        count_ms=counted.mean(), count_max_ms=counted.peak if counted.n else None)
        if period_ms:
            info["period_ms"] = float(period_ms)
            info["period_source"] = period_source or "measured"
        return info


def scan_saturation_metadata(raw_counts, frames: Optional[int] = None) -> dict:
    """Cube-header keys of a scan's basic detection, from the kept frames' :class:`RawCount`
    (``None`` = not checked), index for index with the cube's lines (after the recorder's trim).

    The level, its profile id and status are the ones the frames were *counted with* (kept with
    each frame), not the profile active at export. A scan counted against more than one level
    reports no single ``saturation_level``: its source lists each level and profile with their
    frame counts. With ``frames`` given and a different number of entries, nothing is paired:
    every line counts as not checked.

    Returns:
        dict: ``saturation_level``, ``saturation_level_source``, ``saturation_profile_status``,
        ``saturation_profile_id``, ``saturation_checked_frames``, ``saturation_unchecked_frames``,
        ``saturation_clipped_frames``, ``saturation_peak_clipped_percent`` (only those that apply).
    """
    counts = list(raw_counts or [])
    if frames is not None and len(counts) != int(frames):
        logger.error("Saturation counts (%d) do not match the frames (%d); scan not checked", len(counts), frames)
        counts = [None] * int(frames)
    checked = [c for c in counts if c is not None]
    meta = {"saturation_checked_frames": len(checked), "saturation_unchecked_frames": len(counts) - len(checked)}
    if not checked:
        meta.update(saturation_level_source="not checked (no verified saturation profile for the camera setting)",
                    saturation_profile_status="none")
        return meta
    meta["saturation_clipped_frames"] = sum(1 for c in checked if c.clipped > 0)
    meta["saturation_peak_clipped_percent"] = max(100.0 * c.clipped / c.pixels if c.pixels else 0.0 for c in checked)
    groups: dict = {}
    for c in checked:
        groups[(c.level, c.profile_id)] = groups.get((c.level, c.profile_id), 0) + 1
    if len(groups) == 1:
        (level, profile_id), = groups
        meta.update(saturation_level=int(level), saturation_profile_id=profile_id or "unknown",
                    saturation_profile_status="verified",
                    saturation_level_source=f"measured saturation profile {profile_id or 'unknown'}")
    else:
        meta.update(saturation_profile_status="verified", saturation_profile_id="mixed",
                    saturation_level_source="mixed: " + "; ".join(
                        f"{level} ({profile_id}) x{n}" for (level, profile_id), n in groups.items()))
    return meta


def format_scan_info(info: dict, summary: dict) -> str:
    """One log line for a scan: the record summary (``scan_summary``) and the cost totals."""
    def ms(value):
        return "?" if value is None else f"{value:.2f}"
    period = info.get("period_ms")
    cost = (f"mask {ms(info.get('sat_ms'))} ms mean / {ms(info.get('sat_max_ms'))} ms max; "
            f"frame {ms(info.get('busy_ms'))} ms mean / {ms(info.get('busy_max_ms'))} ms max")
    if period:
        cost += f" of a {period:.2f} ms frame period ({info.get('period_source', 'measured')})"
    peak = info.get("raw_peak_clipped_percent")
    raw = (f"raw {info.get('raw_status', '?')}: {info.get('raw_checked', 0)}/{info.get('frames', 0)} frames counted, "
           f"{info.get('raw_saturated_frames', 0)} clipped (peak {'?' if peak is None else f'{peak:.3f}'} %), "
           f"{info.get('raw_near_frames', 0)} near, count {ms(info.get('count_ms'))} ms mean / "
           f"{ms(info.get('count_max_ms'))} ms max; ")
    levels = "; ".join(f"level {e['level']} ({e['provenance']}) x{e['frames']}" for e in info.get("levels") or [])
    skipped = "; ".join(f"{n} not checked: {reason}" for reason, n in (info.get("not_checked") or {}).items())
    return (raw + f"mask {summary.get('status')}: {summary.get('checked', 0)}/{summary.get('frames', 0)} "
            f"frames checked, "
            f"{summary.get('near_frames', 0)} near, {summary.get('clipped_frames', 0)} clipped, "
            f"{summary.get('bytes', 0) / 1024:.0f} KiB; {cost}"
            + (f"; {levels}" if levels else "") + (f"; {skipped}" if skipped else ""))
