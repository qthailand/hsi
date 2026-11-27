"""Plateau analysis and the exposure sweep that measures a camera's saturation level (Qt-free).

A clipped pixel reads one exact value (the ADC's largest output, scaled and maybe averaged by the
camera). Raise the exposure on a bright scene and more and more pixels pile up on that value:
a *plateau*. This module finds plateaus in the top of the raw histogram and decides from their
behaviour over exposure and gain whether they are clipping:

1. **Candidates per step** (:func:`step_candidates`): an exact value in the top band
   (``>= (1 - band) * nominal full scale``) holding at least ``min_pixels`` pixels and
   ``spike_ratio`` times more than the median of its neighbouring values, in *every* frame of
   the step (stable between frames). A raw maximum alone never decides anything.
2. **Clipping evidence per gain** (:func:`analyse`): over the exposure sweep a candidate is a
   clipping plateau when it stays once it appeared (persistent), its pixel count does not fall
   as the exposure rises, and either a shorter exposure without it was measured (the plateau's
   *entry*) or its count grew. A value that holds a constant count from the shortest exposure
   (hot pixels, fixed pattern) or a frame saturated everywhere at every exposure gives no such
   evidence: *inconclusive*, never a guess. Secondary peaks are judged the same way, one by one.
3. **Level**: the lowest clipping plateau of each gain, accepted as one scalar threshold only
   when the pixels between it and the highest clipping plateau are (almost) all on clipping
   plateaus. Every gain measured must give the same level (at least ``min_gains`` gains):
   then *verified* over that gain range; levels that differ between gains are recorded and the
   result is *inconclusive* (no gain-dependent profile is made).

:func:`measure` drives a :class:`FrameSource` through the sweep: per bin and gain it starts at
the given exposure, steps down (``/ factor``) while the plateau is already there so its entry
is seen, then up (``x factor``) until the plateau has been followed ``steps_after_plateau``
more steps or an exposure limit is reached. Cancel with a ``threading.Event``.

Data conventions:
    Frames are 2-D integer raw frames (any orientation) as the camera delivers them; levels in
    raw counts; exposure in ms; gain in backend units.

Threading:
    :func:`analyse` and friends are pure. :func:`measure` blocks on the source; run it on a
    worker thread (``modules.saturation_profile.widget`` does).

Example:
    >>> import numpy as np
    >>> frame = np.full((40, 40), 1000, np.uint16); frame[:10] = 65504
    >>> step_candidates(step_from_frames([frame, frame], gain=1, exposure_ms=10, nominal_full_scale=65520),
    ...                 65520)
    {65504: 400.0}
"""
from __future__ import annotations

import datetime
import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from modules.contracts.saturation_profile import (STATUS_FAILED, STATUS_INCONCLUSIVE, STATUS_VERIFIED,
                                                  PlateauValue, SaturationIdentity, SaturationProfileRecord)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlateauParams:
    """Thresholds of the plateau analysis (defaults suit 10-16 bit cameras).

    Attributes:
        band: Top band analysed, as a share of the nominal full scale (``0.10``: the top 10 %).
        min_pixels: Fewest pixels a plateau holds (also ``min_fraction`` of the frame, the larger).
        min_fraction: Fewest pixels a plateau holds as a share of the frame.
        spike_ratio: A plateau holds this many times the median count of its neighbouring values.
        spike_window: Neighbourhood of a value, as a share of the nominal full scale.
        full_fraction: Share of the frame above which a plateau means "fully saturated".
        growth: A count has grown when its last step holds this many times its first.
        shrink_tolerance: A count may dip to this share of the previous step (noise) and still
            count as not falling.
        between_tolerance: Pixels between the lowest and highest clipping plateau that are on
            no clipping plateau, as a share of the plateau pixels, above which one scalar
            threshold is not accepted.
        min_level_ratio: A level below this share of the nominal full scale is inconclusive
            (check the pixel format).
        min_gains: Gains that must agree on the level.
    """

    band: float = 0.10
    min_pixels: int = 50
    min_fraction: float = 2e-5
    spike_ratio: float = 10.0
    spike_window: float = 0.01
    full_fraction: float = 0.98
    growth: float = 1.2
    shrink_tolerance: float = 0.9
    between_tolerance: float = 0.05
    min_level_ratio: float = 0.9
    min_gains: int = 2

    def floor(self, nominal_full_scale: int) -> int:
        """Lowest raw value analysed."""
        return int(math.floor((1.0 - self.band) * float(nominal_full_scale)))

    def plateau_pixels(self, pixels: int) -> int:
        """Fewest pixels a plateau holds in a frame of ``pixels`` pixels."""
        return max(int(self.min_pixels), int(math.ceil(self.min_fraction * pixels)))


DEFAULT_PARAMS = PlateauParams()


# ── per frame / per step ──────────────────────────────────────────────────

def top_histogram(frame, floor: int) -> Dict[int, int]:
    """``{value: pixels}`` of the integer frame's values ``>= floor`` (exact values)."""
    frame = np.asarray(frame)
    if frame.dtype.kind not in "ui":
        raise ValueError(f"raw frames must be integers, got {frame.dtype}")
    top = frame[frame >= floor]
    if top.size == 0:
        return {}
    offset = int(floor)
    counts = np.bincount((top.astype(np.int64) - offset).ravel())
    nz = np.flatnonzero(counts)
    return {int(v + offset): int(counts[v]) for v in nz}


@dataclass(frozen=True)
class StepFrames:
    """The frames of one exposure step, reduced to their top histograms.

    Attributes:
        gain, exposure_ms: The setting the frames were taken with.
        pixels: Pixels per frame.
        histograms: :func:`top_histogram` of each frame.
        maxima: Largest value of each frame.
    """

    gain: float
    exposure_ms: float
    pixels: int
    histograms: Tuple[Dict[int, int], ...]
    maxima: Tuple[int, ...]

    def to_dict(self, top: int = 8, candidates: Optional[Dict[int, float]] = None) -> dict:
        """Evidence for a profile: the setting, the top ``top`` values by mean count, the plateaus."""
        total: Dict[int, int] = {}
        for hist in self.histograms:
            for value, count in hist.items():
                total[value] = total.get(value, 0) + count
        n = max(1, len(self.histograms))
        best = sorted(total.items(), key=lambda item: (-item[1], -item[0]))[:top]
        out = {"gain": float(self.gain), "exposure_ms": float(self.exposure_ms), "frames": len(self.histograms),
               "pixels": int(self.pixels), "max": int(max(self.maxima)) if self.maxima else None,
               "top_values": [[int(v), round(c / n, 1)] for v, c in best]}
        if candidates is not None:
            out["plateaus"] = [[int(v), round(float(c), 1)] for v, c in sorted(candidates.items())]
        return out


def step_from_frames(frames: Sequence, gain, exposure_ms, nominal_full_scale: int,
                     params: PlateauParams = DEFAULT_PARAMS) -> StepFrames:
    """Reduce the raw ``frames`` of one step to a :class:`StepFrames` (frames are not kept)."""
    frames = [np.asarray(f) for f in frames]
    if not frames:
        raise ValueError("a step needs at least one frame")
    floor = params.floor(nominal_full_scale)
    return StepFrames(float(gain), float(exposure_ms), int(frames[0].size),
                      tuple(top_histogram(f, floor) for f in frames), tuple(int(f.max()) for f in frames))


def _frame_candidates(hist: Dict[int, int], minimum: int, ratio: float, window: int) -> Dict[int, int]:
    """Exact values of one histogram that hold a plateau (see the module docstring)."""
    if not hist:
        return {}
    values = np.fromiter(hist.keys(), dtype=np.int64)
    counts = np.fromiter(hist.values(), dtype=np.int64)
    order = np.argsort(values)
    values, counts = values[order], counts[order]
    out = {}
    for i in np.flatnonzero(counts >= minimum):
        v = values[i]
        lo, hi = np.searchsorted(values, v - window), np.searchsorted(values, v + window, side="right")
        near = np.concatenate([counts[lo:i], counts[i + 1:hi]])
        background = float(np.median(near)) if near.size else 0.0
        if counts[i] >= ratio * max(1.0, background):
            out[int(v)] = int(counts[i])
    return out


def step_candidates(step: StepFrames, nominal_full_scale: int,
                    params: PlateauParams = DEFAULT_PARAMS) -> Dict[int, float]:
    """``{value: mean pixels}`` of the plateaus present in *every* frame of ``step``."""
    minimum = params.plateau_pixels(step.pixels)
    window = max(1, int(round(params.spike_window * nominal_full_scale)))
    per_frame = [_frame_candidates(h, minimum, params.spike_ratio, window) for h in step.histograms]
    if not per_frame:
        return {}
    common = set(per_frame[0]).intersection(*per_frame[1:])
    return {v: float(np.mean([c[v] for c in per_frame])) for v in sorted(common)}


def fully_saturated(candidates: Dict[int, float], pixels: int, params: PlateauParams = DEFAULT_PARAMS) -> bool:
    """Whether the plateaus of a step cover (almost) the whole frame."""
    return bool(candidates) and sum(candidates.values()) >= params.full_fraction * pixels


# ── analysis ──────────────────────────────────────────────────────────────

@dataclass
class _Judgement:
    value: int
    confirmed: bool
    reason: str
    series: List[Tuple[float, float]] = field(default_factory=list)


def _judge(value: int, steps: List[StepFrames], cands: List[Dict[int, float]], params: PlateauParams) -> _Judgement:
    series = [(s.exposure_ms, float(c.get(value, 0.0))) for s, c in zip(steps, cands)]
    present = [value in c for c in cands]
    first = present.index(True)
    counts = [c[value] for c in cands[first:] if value in c]
    pixels = steps[first].pixels
    if not all(present[first:]):
        return _Judgement(value, False, "disappears at a longer exposure (not a clipping plateau)", series)
    if any(b < params.shrink_tolerance * a for a, b in zip(counts, counts[1:])):
        return _Judgement(value, False, "pixel count falls as the exposure rises", series)
    entry = first > 0
    grew = counts[-1] >= params.growth * counts[0]
    if entry or grew:
        how = (f"absent at {steps[first - 1].exposure_ms:g} ms, appears at {steps[first].exposure_ms:g} ms"
               if entry else f"pixels grow x{counts[-1] / counts[0]:.1f} with exposure")
        return _Judgement(value, True, f"clipping plateau: {how}", series)
    if all(c >= params.full_fraction * pixels for c in counts):
        return _Judgement(value, False, "frame fully saturated at every exposure measured: the plateau's "
                                        "entry was not seen (measure a shorter exposure)", series)
    if len(counts) == 1:
        return _Judgement(value, False, "seen at one exposure only: no entry or growth to confirm it", series)
    return _Judgement(value, False, "constant pixel count from the shortest exposure (hot pixels or fixed "
                                    "pattern?): no entry or growth seen", series)


def _gain_level(steps: List[StepFrames], cands: List[Dict[int, float]], nominal: int,
                params: PlateauParams):
    """``(level or None, judgements, reasons)`` of one gain's sweep (steps sorted by exposure)."""
    values = sorted(set().union(*cands)) if cands else []
    judgements = [_judge(v, steps, cands, params) for v in values]
    gain = steps[0].gain if steps else None
    if not values:
        return None, judgements, [f"gain {gain:g}: no plateau in the top {params.band:.0%} up to "
                                  f"{max(s.exposure_ms for s in steps):g} ms (scene too dark or exposure limit)"]
    clipping = [j for j in judgements if j.confirmed]
    if not clipping:
        return None, judgements, [f"gain {gain:g}: {j.value}: {j.reason}" for j in judgements]
    level, top = min(j.value for j in clipping), max(j.value for j in clipping)
    last_hist = _mean_hist(steps[-1])
    clip_values = {j.value for j in clipping}
    on_plateaus = sum(last_hist.get(v, 0.0) for v in clip_values)
    between = sum(c for v, c in last_hist.items() if level < v < top and v not in clip_values)
    reasons = []
    if on_plateaus and between > params.between_tolerance * on_plateaus:
        reasons.append(f"gain {gain:g}: {between:.0f} pixels between the clipping plateaus {level} and {top} "
                       f"are on none of them: one threshold does not describe this clipping")
    if level < params.min_level_ratio * nominal:
        reasons.append(f"gain {gain:g}: level {level} is below {params.min_level_ratio:.0%} of the nominal full "
                       f"scale {nominal} (check the pixel format)")
    return (None if reasons else level), judgements, reasons


def _mean_hist(step: StepFrames) -> Dict[int, float]:
    total: Dict[int, float] = {}
    for hist in step.histograms:
        for v, c in hist.items():
            total[v] = total.get(v, 0.0) + c
    n = max(1, len(step.histograms))
    return {v: c / n for v, c in total.items()}


def analyse(steps: Iterable[StepFrames], identity: SaturationIdentity, nominal_full_scale: int,
            params: PlateauParams = DEFAULT_PARAMS, measured_at: Optional[str] = None,
            software: str = "unknown", firmware: Optional[str] = None,
            notes: Sequence[str] = ()) -> SaturationProfileRecord:
    """Decide the saturation level of one configuration from its sweep (see the module docstring).

    Args:
        steps: Every exposure step of every gain (any order).
        identity: The configuration measured.
        nominal_full_scale: The frames' nominal full scale (reference, and the top band).
        params: Thresholds.
        measured_at, software, firmware: Stored in the record (``measured_at`` default: now).
        notes: Extra lines appended to the record's reasons.

    Returns:
        SaturationProfileRecord: ``verified`` with a level and gain range, or ``inconclusive``
        with the reasons; ``failed`` when there are no steps.
    """
    steps = sorted(steps, key=lambda s: (s.gain, s.exposure_ms))
    when = measured_at or datetime.datetime.now().isoformat(timespec="seconds")
    common = dict(identity=identity, nominal_full_scale=int(nominal_full_scale), measured_at=when,
                  software=software, firmware=firmware)
    if not steps:
        return SaturationProfileRecord(status=STATUS_FAILED, level=None,
                                       reasons=("no frames were measured",) + tuple(notes), **common)
    gains = sorted({s.gain for s in steps})
    per_gain, reasons, evidence = {}, [], {}
    step_dicts = []
    for gain in gains:
        g_steps = [s for s in steps if s.gain == gain]
        cands = [step_candidates(s, nominal_full_scale, params) for s in g_steps]
        step_dicts += [s.to_dict(candidates=c) for s, c in zip(g_steps, cands)]
        level, judgements, why = _gain_level(g_steps, cands, nominal_full_scale, params)
        per_gain[gain] = level
        reasons += why
        for j in judgements:
            evidence.setdefault(j.value, []).append(j)
            item = {"gain": float(gain), "confirmed": j.confirmed, "reason": j.reason,
                    "counts": [[float(e), round(c, 1)] for e, c in j.series]}
            j.series = item                                    # reuse: the JSON evidence of this gain
    plateaus = []
    for value in sorted(evidence):
        js = evidence[value]
        confirmed = any(j.confirmed for j in js)
        reason = next((j.reason for j in js if j.confirmed), js[0].reason)
        plateaus.append(PlateauValue(value, "clipping" if confirmed else "unconfirmed", reason,
                                     tuple(j.series for j in js)))
    levels = {g: lv for g, lv in per_gain.items() if lv is not None}
    distinct = sorted(set(levels.values()))
    if len(distinct) > 1:
        reasons.append("levels differ between gains: " + ", ".join(f"gain {g:g} -> {lv}" for g, lv in levels.items())
                       + " (no gain-independent level; a gain-dependent profile is not made)")
    elif len(levels) < len(gains):
        pass                                                   # the failing gain's reason is already there
    elif len(gains) < params.min_gains:
        reasons.append(f"measured at {len(gains)} gain(s); at least {params.min_gains} must agree")
    verified = len(distinct) == 1 and len(levels) == len(gains) >= params.min_gains and not any(
        r for r in reasons)
    exposures = tuple(sorted({s.exposure_ms for s in steps}))
    record = SaturationProfileRecord(
        status=STATUS_VERIFIED if verified else STATUS_INCONCLUSIVE, level=distinct[0] if verified else None,
        plateaus=tuple(plateaus), gain_range=(gains[0], gains[-1]) if verified else None,
        gains_tested=tuple(gains), exposures_ms=exposures, frames=sum(len(s.histograms) for s in steps),
        steps=tuple(step_dicts), reasons=tuple(reasons) + tuple(notes), **common)
    return record


def failed_record(identity: SaturationIdentity, nominal_full_scale, reason: str, steps: Sequence[StepFrames] = (),
                  software: str = "unknown", firmware: Optional[str] = None) -> SaturationProfileRecord:
    """A ``failed`` record (the measurement could not run): the reason and whatever was measured."""
    return SaturationProfileRecord(
        identity=identity, status=STATUS_FAILED, level=None,
        nominal_full_scale=None if nominal_full_scale is None else int(nominal_full_scale),
        gains_tested=tuple(sorted({s.gain for s in steps})), exposures_ms=tuple(sorted({s.exposure_ms for s in steps})),
        frames=sum(len(s.histograms) for s in steps), steps=tuple(s.to_dict() for s in steps), reasons=(reason,),
        measured_at=datetime.datetime.now().isoformat(timespec="seconds"), software=software, firmware=firmware)


# ── the sweep ─────────────────────────────────────────────────────────────

class MeasurementCancelled(Exception):
    """The user cancelled the measurement (nothing is saved)."""


class FrameSource(Protocol):
    """What :func:`measure` needs from the camera (the host adapts its acquisition to it).

    ``apply`` returns once the setting is active and ``grab`` returns only frames exposed with
    it (the source discards frames in flight); both raise ``TimeoutError`` when the camera does
    not deliver in time and any other exception on a camera error.
    """

    def bins(self) -> Tuple[int, ...]: ...
    def exposure_limits_ms(self) -> Tuple[float, float]: ...
    def gain_limits(self) -> Tuple[float, float]: ...
    def apply(self, camera_bin: int, gain: float, exposure_ms: float) -> None: ...
    def grab(self, count: int, timeout_s: float) -> List[np.ndarray]: ...
    def identity(self) -> SaturationIdentity: ...
    def nominal_full_scale(self) -> int: ...
    def firmware(self) -> Optional[str]: ...


@dataclass(frozen=True)
class SweepPlan:
    """What :func:`measure` does.

    Attributes:
        bins: Hardware bins to measure, in order.
        gains: Gains to measure (at least two for a verified level).
        start_exposure_ms: First exposure of every sweep.
        factor: Exposure ratio between steps.
        frames_per_step: Frames analysed per step (after the source's settling).
        max_steps: Most steps per gain.
        steps_after_plateau: Steps followed after the first plateau appeared.
        down_steps: Most steps below the start (to see the plateau's entry).
        max_exposure_ms: Longest exposure used (also the camera's limit).
    """

    bins: Tuple[int, ...]
    gains: Tuple[float, ...]
    start_exposure_ms: float
    factor: float = 2.0
    frames_per_step: int = 4
    max_steps: int = 10
    steps_after_plateau: int = 2
    down_steps: int = 4
    max_exposure_ms: float = 10000.0


def frame_timeout_s(exposure_ms: float, frames: int) -> float:
    """Time allowed for ``frames`` frames at ``exposure_ms`` (3 frame periods each, plus 5 s)."""
    return 5.0 + 3.0 * frames * float(exposure_ms) / 1000.0


def _check(cancel):
    if cancel is not None and cancel.is_set():
        raise MeasurementCancelled("cancelled")


def _sweep_gain(source: FrameSource, camera_bin: int, gain: float, plan: SweepPlan, params: PlateauParams,
                nominal: int, cancel, progress) -> List[StepFrames]:
    low, high = source.exposure_limits_ms()
    high = min(float(high), float(plan.max_exposure_ms))
    steps: Dict[float, StepFrames] = {}

    def take(exposure):
        exposure = float(min(max(exposure, low), high))
        if exposure in steps:
            return steps[exposure], step_candidates(steps[exposure], nominal, params)
        _check(cancel)
        source.apply(camera_bin, gain, exposure)
        _check(cancel)
        frames = source.grab(plan.frames_per_step, frame_timeout_s(exposure, plan.frames_per_step))
        if len(frames) < plan.frames_per_step:
            raise TimeoutError(f"{len(frames)} of {plan.frames_per_step} frames at {exposure:g} ms")
        step = step_from_frames(frames, gain, exposure, nominal, params)
        steps[exposure] = step
        cands = step_candidates(step, nominal, params)
        if progress is not None:
            progress({"bin": camera_bin, "gain": gain, "exposure_ms": exposure, "step": step,
                      "plateaus": cands, "frame": frames[-1]})
        return step, cands

    start = float(min(max(plan.start_exposure_ms, low), high))
    _, cands = take(start)
    exposure = start
    down = 0
    while cands and down < plan.down_steps and exposure / plan.factor >= low and len(steps) < plan.max_steps:
        exposure /= plan.factor                                # plateau already there: find its entry
        previous = cands
        _, cands = take(exposure)
        down += 1
        if cands and all(v in cands and cands[v] >= params.shrink_tolerance * c for v, c in previous.items()):
            break                                              # not exposure-driven (hot pixels): stop
    baseline = step_candidates(steps[min(steps)], nominal, params)

    def evidence(found):
        """A plateau the shortest exposure did not have, or one that has grown since."""
        return any(v not in baseline or c >= params.growth * baseline[v] for v, c in found.items())

    exposure, after = start, 0
    _, cands = take(exposure)
    while len(steps) < plan.max_steps:
        if evidence(cands):
            after += 1
            if after > plan.steps_after_plateau:
                break
        nxt = exposure * plan.factor
        if nxt > high + 1e-9:
            break                                              # exposure limit
        exposure = nxt
        _, cands = take(exposure)
    return list(steps.values())


def measure(source: FrameSource, plan: SweepPlan, params: PlateauParams = DEFAULT_PARAMS,
            cancel: Optional[threading.Event] = None, progress: Optional[Callable[[dict], None]] = None,
            software: str = "unknown") -> List[SaturationProfileRecord]:
    """Measure every bin of ``plan`` (see the module docstring); one record per bin.

    A camera timeout or error during one bin gives that bin a ``failed`` record and goes on with
    the next. The caller restores the camera afterwards (this function changes bin, gain and
    exposure and does not put them back).

    Raises:
        MeasurementCancelled: ``cancel`` was set (no records are returned).
        ValueError: the plan asks for a bin or gain the camera does not support.
    """
    bins = tuple(int(b) for b in plan.bins)
    supported = tuple(int(b) for b in source.bins())
    if not bins or any(b not in supported for b in bins):
        raise ValueError(f"bins {bins} not all supported by the camera ({supported})")
    g_low, g_high = source.gain_limits()
    gains = tuple(float(g) for g in plan.gains)
    if not gains or any(g < g_low or g > g_high for g in gains):
        raise ValueError(f"gains {gains} outside the camera's range {g_low:g}-{g_high:g}")
    records = []
    for camera_bin in bins:
        steps: List[StepFrames] = []
        identity = None
        try:
            for gain in gains:
                steps += _sweep_gain(source, camera_bin, gain, plan, params, source.nominal_full_scale(),
                                     cancel, progress)
                if identity is None:
                    identity = source.identity()
            records.append(analyse(steps, identity, source.nominal_full_scale(), params, software=software,
                                   firmware=source.firmware()))
        except MeasurementCancelled:
            raise
        except Exception as exc:                               # noqa: BLE001 - one bin failed: record it
            logger.warning("Saturation measurement of bin %d failed: %s", camera_bin, exc)
            try:
                identity = identity or source.identity()
                nominal = source.nominal_full_scale()
            except Exception:                                  # noqa: BLE001 - the camera is gone
                identity, nominal = None, None
            if identity is None:
                raise
            records.append(failed_record(identity, nominal, f"{type(exc).__name__}: {exc}", steps,
                                         software=software))
    return records


__all__ = ["PlateauParams", "DEFAULT_PARAMS", "top_histogram", "StepFrames", "step_from_frames",
           "step_candidates", "fully_saturated", "analyse", "failed_record", "MeasurementCancelled",
           "FrameSource", "SweepPlan", "frame_timeout_s", "measure"]
