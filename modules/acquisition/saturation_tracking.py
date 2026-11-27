"""Saturation tracking in the acquisition loop: on/off, level evidence, mailbox and cost meter.

The worker calls :meth:`SaturationTracker.track` once per corrected frame with the frame's own
correction snapshot (``PreprocessingRuntime.last_prepared``) and binner; the result goes to the
recorder with the frame and the newest one into a mailbox the GUI reads when it wants
(:meth:`SaturationTracker.latest`) - nothing is queued per frame.

A frame is *checked* only when tracking is on, the backend's frames go through the shared
preprocessing (a backend that corrects its own frames hides the raw values) and the camera
states its full scale (:func:`modules.saturation_mask.level_evidence`); otherwise its record is
``None`` (*not checked*), never an empty mask.

Level provenance: the evidence is asked again whenever the camera configuration it depends on
changes (frame dtype, bin, ``info.bit_depth`` / ``info.image_format``, or :meth:`reset_level`
after a bin / parameter change or a restart), and after :data:`LEVEL_RETRY_S` when an SDK
query failed (a transient failure does not leave the rest of the session unchecked). Each scan
keeps the levels it was checked against with their provenance (:meth:`SaturationTracker.scan_info`).

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
from typing import Optional

import numpy as np

from modules.saturation_mask import (STATUS_CHECKED, STATUS_NOT_CHECKED, FrameSaturation, LevelEvidence,
                                     SaturationPropagator, SaturationRecord, level_evidence)

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
        backend: The camera backend (its ``saturation_level`` and bit depths).
        shared_preprocessing: Whether its frames go raw through ``modules.preprocessing``.
        enabled: Track from the start.
    """

    def __init__(self, backend, shared_preprocessing: bool, enabled: bool = False):
        self.backend = backend
        self.shared_preprocessing = bool(shared_preprocessing)
        self._enabled = bool(enabled)
        self._propagator = SaturationPropagator()
        self._evidence_key = None
        self._evidence: Optional[LevelEvidence] = None
        self._evidence_time = 0.0                    # monotonic time of the last query
        self._evidence_generation = 0                # bumped by reset_level
        self._logged_evidence = None                 # the last evidence logged (log on change only)
        self._frame_outcome = None                   # (level, provenance) or the not-checked reason
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

    # ── worker ──────────────────────────────────────────────

    def _configuration_key(self, frame):
        """What the level depends on: frame dtype, bin, delivered bits / pixel format, resets."""
        backend = self.backend
        try:
            info = backend.info
            bits, image_format = getattr(info, "bit_depth", None), getattr(info, "image_format", None)
        except Exception:                            # noqa: BLE001 - info unreadable: ask every time
            bits = image_format = object()
        return (np.asarray(frame).dtype.str, getattr(backend, "current_bin", None), bits, image_format,
                self._evidence_generation)

    def evidence(self, frame) -> LevelEvidence:
        """The level evidence for frames like ``frame``: asked again when the camera configuration
        changes, and :data:`LEVEL_RETRY_S` after a failed query; logged when it changes."""
        key = self._configuration_key(frame)
        now = time.monotonic()
        stale = (key != self._evidence_key or self._evidence is None
                 or (self._evidence.transient and now - self._evidence_time >= LEVEL_RETRY_S))
        if stale:
            self._evidence = level_evidence(self.backend, np.asarray(frame).dtype)
            self._evidence_key, self._evidence_time = key, now
            shown = (self._evidence.level, self._evidence.source, self._evidence.reason)
            if shown != self._logged_evidence:
                self._logged_evidence = shown
                if self._evidence.trusted:
                    logger.info("Saturation level: %s (%s)", self._evidence.level, self._evidence.reason)
                else:
                    logger.warning("Saturation not checked: %s", self._evidence.reason)
        return self._evidence

    def track(self, frame, prepared, binner, row_crop=(0, 0), profiler=None) -> Optional[SaturationRecord]:
        """Saturation of one corrected frame, or ``None`` when it is not checked.

        Args:
            frame: The raw frame the intensities were computed from (not modified).
            prepared: ``PreprocessingRuntime.last_prepared`` of that frame (its dark-reference
                snapshot, hot-pixel switch, orientation and model).
            binner: The binner the frame was binned with.
            row_crop: Keystone rows its binned line loses (``keystone_row_margin``).
            profiler: The loop's profiler (spans ``sat_total``, ``sat_pack`` and those of
                :meth:`SaturationPropagator.frame`).

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
        t0 = time.perf_counter()
        try:
            with profiler.time("sat_total"):
                result = self._propagator.frame(frame, evidence.level, prepared.hot, prepared.enabled,
                                                prepared.calibration, prepared.inverted, binner,
                                                row_crop=row_crop, profiler=profiler)
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
        self._frame_outcome = (evidence.level, evidence.provenance)
        self._post(STATUS_CHECKED, f"level {evidence.level}: {evidence.provenance}", result, evidence.level,
                   evidence.provenance)
        return record

    def note_frame(self, busy_ms: float, recorded: bool) -> None:
        """The worker spent ``busy_ms`` on the frame just tracked (read to recorder);
        ``recorded``: the recorder kept it (counted in :meth:`scan_info`)."""
        sat_ms, record = self._frame_ms, self._frame_record
        near = bool(record is not None and record.any_near())
        clipped = bool(record is not None and record.any_clipped())
        now = time.monotonic()
        with self._lock:
            self._live.append((now, sat_ms, float(busy_ms), near, clipped))
            while self._live and now - self._live[0][0] > LIVE_WINDOW_S:
                self._live.popleft()
            scan = self._scan
            if scan is not None and recorded:
                scan["frames"] += 1
                scan["busy"].add(float(busy_ms))
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
        sat = [s for _, s, _, _, _ in live if s is not None]
        busy = [b for _, _, b, _, _ in live]
        info = {"tracking": self._enabled, "frames": len(live),
                "status": latest.status if latest else None, "reason": latest.reason if latest else "",
                "sat_ms": float(np.mean(sat)) if sat else None,
                "sat_max_ms": float(np.max(sat)) if sat else None,
                "busy_ms": float(np.mean(busy)) if busy else None,
                "near_frames": sum(1 for *_, n, _ in live if n),
                "clipped_frames": sum(1 for *_, c in live if c)}
        if period_ms and info["busy_ms"] is not None:
            info["period_ms"] = float(period_ms)
            info["period_source"] = period_source or "measured"
            info["load"] = info["busy_ms"] / float(period_ms)
            info["overhead"] = None if info["sat_ms"] is None else info["sat_ms"] / float(period_ms)
        return info

    def start_scan(self) -> None:
        """Start the per-recording totals (recording armed)."""
        with self._lock:
            self._scan = {"tracking": self._enabled, "frames": 0, "checked": 0, "near_frames": 0,
                          "clipped_frames": 0, "sat": _Totals(), "busy": _Totals(), "levels": {},
                          "not_checked": {}}

    def scan_info(self, period_ms: Optional[float] = None, period_source: Optional[str] = None) -> dict:
        """Totals since :meth:`start_scan` over the recorded frames (before any trim):
        ``tracking`` (at the start), ``frames``, ``checked``, ``near_frames``, ``clipped_frames``,
        ``sat_ms`` / ``sat_max_ms`` and ``busy_ms`` / ``busy_max_ms`` (mean / max per frame),
        ``levels`` (``[{'level', 'provenance', 'frames'}]``: every level frames were checked
        against), ``not_checked`` (``{reason: frames}``), ``period_ms`` / ``period_source`` (as
        in :meth:`live_info`; ``busy_max_ms > period_ms`` means some frame took longer than the
        camera's period). ``{}`` before."""
        with self._lock:
            scan = self._scan
            if scan is None:
                return {}
            info = {k: scan[k] for k in ("tracking", "frames", "checked", "near_frames", "clipped_frames")}
            info.update(sat_ms=scan["sat"].mean(), sat_max_ms=scan["sat"].peak if scan["sat"].n else None,
                        busy_ms=scan["busy"].mean(), busy_max_ms=scan["busy"].peak if scan["busy"].n else None,
                        levels=[{"level": level, "provenance": provenance, "frames": n}
                                for (level, provenance), n in scan["levels"].items()],
                        not_checked=dict(scan["not_checked"]))
        if period_ms:
            info["period_ms"] = float(period_ms)
            info["period_source"] = period_source or "measured"
        return info


def format_scan_info(info: dict, summary: dict) -> str:
    """One log line for a scan: the record summary (``scan_summary``) and the cost totals."""
    def ms(value):
        return "?" if value is None else f"{value:.2f}"
    period = info.get("period_ms")
    cost = (f"mask {ms(info.get('sat_ms'))} ms mean / {ms(info.get('sat_max_ms'))} ms max; "
            f"frame {ms(info.get('busy_ms'))} ms mean / {ms(info.get('busy_max_ms'))} ms max")
    if period:
        cost += f" of a {period:.2f} ms frame period ({info.get('period_source', 'measured')})"
    levels = "; ".join(f"level {e['level']} ({e['provenance']}) x{e['frames']}" for e in info.get("levels") or [])
    skipped = "; ".join(f"{n} not checked: {reason}" for reason, n in (info.get("not_checked") or {}).items())
    return (f"{summary.get('status')}: {summary.get('checked', 0)}/{summary.get('frames', 0)} frames checked, "
            f"{summary.get('near_frames', 0)} near, {summary.get('clipped_frames', 0)} clipped, "
            f"{summary.get('bytes', 0) / 1024:.0f} KiB; {cost}"
            + (f"; {levels}" if levels else "") + (f"; {skipped}" if skipped else ""))
