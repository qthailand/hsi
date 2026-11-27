"""Phase 2E.1: saturation level profile - plateau analysis, the exposure sweep and the profile store.

Measurement (10.1): single / multiple plateaus, fully saturated frames, hot pixels only, too few
plateau pixels, an unstable plateau, a secondary peak that is not clipping, a gain-dependent
level, the exposure limit, a camera timeout and cancel - on a simulated camera - and the plateaus
of real Apollo-M MAX frames (tests/fixtures/saturation, 2026-11-27).
Store (10.2): round trip, invalid JSON, unsupported version, serial / format / bits / bin
mismatch, no collision between bit depths, atomic save, history, a failed measurement never
replacing a verified profile, the verified gain range.
"""
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.contracts import CameraProfile, SaturationIdentity, SaturationProfileRecord  # noqa: E402
from modules.contracts.saturation_profile import format_token  # noqa: E402
from modules.saturation_profile import (STATE_FAILED, STATE_GAIN_OUTSIDE, STATE_INCOMPATIBLE,  # noqa: E402
                                        STATE_INCONCLUSIVE, STATE_INVALID, STATE_MISSING, STATE_VERIFIED,
                                        MeasurementCancelled, SaturationProfileStore, StepFrames, SweepPlan,
                                        analyse, measure, step_candidates, step_from_frames, top_histogram)
from modules.saturation_profile import store as store_module  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "saturation"
NOMINAL = 65520
IDENT = SaturationIdentity("poa", "Apollo-M MAX", "CAMP7161", "RAW16", 16, 12, 1)


# ---------------------------------------------------------------- simulated camera
class FakeCamera:
    """A bright gradient scene: value = min(radiance * exposure * gain, ADC max), 12 bits << 4.

    ``clip(gain)`` is the value clipped pixels read; ``secondary`` = (value, share): that share of
    the clipped pixels reads another value (bin 3 of the Apollo-M MAX); ``hot`` pixels always read
    ``hot_value``; ``jitter`` alternates the clip value between frames (an unstable plateau);
    ``timeout_at`` exposures deliver no frames.
    """

    def __init__(self, shape=(120, 160), bright=1.0, clip=lambda gain: 65504, secondary=None, hot=0,
                 hot_value=65504, jitter=False, timeout_at=(), max_exposure_ms=10000.0, bins=(1, 2, 3, 4)):
        rng = np.random.default_rng(7)
        h, w = shape
        self.radiance = np.outer(np.linspace(0.05, 1.0, h), np.linspace(0.2, 1.0, w)) * bright
        self.noise = rng.normal(0, 0.003, shape)
        self.clip, self.secondary, self.jitter = clip, secondary, jitter
        self.hot = (rng.integers(0, h, hot), rng.integers(0, w, hot)) if hot else None
        self.hot_value = hot_value
        self.timeout_at = set(timeout_at)
        self.max_exposure_ms = max_exposure_ms
        self._bins = bins
        self.bin, self.gain, self.exposure = 1, 1.0, 10.0
        self.applied, self.frame_no = [], 0
        self.share_mask = rng.random(shape)

    def bins(self):
        return self._bins

    def exposure_limits_ms(self):
        return 0.1, self.max_exposure_ms

    def gain_limits(self):
        return 0.0, 500.0

    def apply(self, camera_bin, gain, exposure_ms):
        self.bin, self.gain, self.exposure = camera_bin, gain, exposure_ms
        self.applied.append((camera_bin, gain, exposure_ms))

    def frame(self):
        self.frame_no += 1
        signal = self.radiance * self.exposure * (1 + 0.02 * (self.gain - 1)) * 40.0 + self.noise * 4095
        counts = np.clip(np.round(signal), 0, 4094).astype(np.int64)
        out = (counts << 4).astype(np.uint16)
        clipped = counts >= 4094
        level = self.clip(self.gain)
        if self.jitter and self.frame_no % 2:
            level -= 16
        out[clipped] = level
        if self.secondary is not None:
            value, share = self.secondary
            out[clipped & (self.share_mask < share)] = value
        if self.hot is not None:
            out[self.hot] = self.hot_value
        return out

    def grab(self, count, timeout_s):
        if self.exposure in self.timeout_at:
            return []
        return [self.frame() for _ in range(count)]

    def identity(self):
        return SaturationIdentity("fake", "Sim", "S1", "RAW16", 16, 12, self.bin)

    def nominal_full_scale(self):
        return NOMINAL

    def firmware(self):
        return "1.0"


def plan(bins=(1,), gains=(1.0, 4.0), start=10.0, **kw):
    return SweepPlan(bins=bins, gains=gains, start_exposure_ms=start, frames_per_step=3, **kw)


# ---------------------------------------------------------------- per step
def test_top_histogram_and_candidates_need_an_exact_value_in_every_frame():
    frame = np.full((50, 50), 30000, np.uint16)
    frame[:5] = 65504                                                  # 250 pixels on one value
    frame[10, :30] = np.arange(60000, 60030)                           # a ramp: no plateau
    assert top_histogram(frame, 58968)[65504] == 250
    step = step_from_frames([frame, frame], 1, 5.0, NOMINAL)
    assert step_candidates(step, NOMINAL) == {65504: 250.0}
    other = frame.copy()
    other[:5] = 65488                                                  # the next frame clips elsewhere
    assert step_candidates(step_from_frames([frame, other], 1, 5.0, NOMINAL), NOMINAL) == {}
    few = np.full((50, 50), 30000, np.uint16)
    few[0, :20] = 65504                                                # 20 pixels < 50
    assert step_candidates(step_from_frames([few], 1, 5.0, NOMINAL), NOMINAL) == {}
    with pytest.raises(ValueError):
        top_histogram(frame.astype(np.float32), 0)


# ---------------------------------------------------------------- the sweep (10.1)
def test_single_plateau_is_verified_over_the_gains_measured():
    camera = FakeCamera()
    (record,) = measure(camera, plan())
    assert record.status == STATE_VERIFIED and record.level == 65504
    assert record.gain_range == (1.0, 4.0) and record.gains_tested == (1.0, 4.0)
    assert [p.value for p in record.plateaus if p.role == "clipping"] == [65504]
    assert record.nominal_full_scale == NOMINAL and record.firmware == "1.0"
    assert record.identity.serial == "S1" and record.frames == 3 * len(record.steps)
    assert all("exposure_ms" in s and "top_values" in s for s in record.steps)
    assert record.limitations and "digital clipping" in record.limitations[0]


def test_two_clipping_plateaus_give_the_lower_as_threshold():
    camera = FakeCamera(secondary=(65535, 0.98))                       # bin 3: 98 % read 65535, 2 % 65504
    (record,) = measure(camera, plan())
    assert record.status == STATE_VERIFIED and record.level == 65504
    roles = {p.value: p.role for p in record.plateaus}
    assert roles == {65504: "clipping", 65535: "clipping"}


def test_fully_saturated_start_steps_down_to_see_the_entry():
    camera = FakeCamera(bright=40.0)                                   # saturated everywhere at 10 ms
    (record,) = measure(camera, plan(start=10.0))
    assert record.status == STATE_VERIFIED and min(record.exposures_ms) < 10.0
    full = np.full((60, 80), 65504, np.uint16)                         # every pixel clipped
    steps = [step_from_frames([full] * 3, gain, e, NOMINAL) for gain in (1, 4) for e in (10.0, 20.0)]
    for s in steps:
        assert s.histograms[0] == {65504: s.pixels}
    result = analyse(steps, IDENT, NOMINAL)                            # never saw it form
    assert result.status == STATE_INCONCLUSIVE and result.level is None
    assert any("fully saturated" in r for r in result.reasons)


def test_hot_pixels_alone_are_not_a_clipping_plateau():
    camera = FakeCamera(bright=0.0005, hot=200)                        # dark scene, 200 stuck pixels
    (record,) = measure(camera, plan(max_exposure_ms=80.0))
    assert record.status == STATE_INCONCLUSIVE and record.level is None
    (plateau,) = record.plateaus
    assert plateau.value == 65504 and plateau.role == "unconfirmed" and "constant" in plateau.reason


def test_hot_pixels_on_the_clip_value_do_not_stop_the_sweep_early():
    """Stuck pixels already on the clip value at the shortest exposure must not count as the
    plateau: the sweep goes on until the real clipping grows it (regression)."""
    camera = FakeCamera(hot=200, hot_value=65504)
    (record,) = measure(camera, plan(start=2.0))
    assert record.status == STATE_VERIFIED and record.level == 65504
    assert max(record.exposures_ms) >= 20.0                            # followed past the hot pixels


def test_too_few_plateau_pixels_or_exposure_limit_is_inconclusive():
    camera = FakeCamera(bright=0.0005)                                 # never saturates below 80 ms
    (record,) = measure(camera, plan(max_exposure_ms=80.0))
    assert record.status == STATE_INCONCLUSIVE and not record.plateaus
    assert any("no plateau" in r and "exposure limit" in r for r in record.reasons)
    assert max(record.exposures_ms) == 80.0


def test_an_unstable_plateau_is_not_accepted():
    camera = FakeCamera(jitter=True)                                   # clip value changes frame to frame
    (record,) = measure(camera, plan())
    assert record.status == STATE_INCONCLUSIVE and record.level is None


def test_a_secondary_peak_that_is_not_clipping_is_recorded_but_not_used():
    camera = FakeCamera(hot=300, hot_value=62000)                      # stuck pixels below the clip
    (record,) = measure(camera, plan())
    assert record.status == STATE_VERIFIED and record.level == 65504
    roles = {p.value: p.role for p in record.plateaus}
    assert roles[62000] == "unconfirmed" and roles[65504] == "clipping"


def test_a_gain_dependent_level_is_inconclusive_and_keeps_every_result():
    camera = FakeCamera(clip=lambda gain: 65504 if gain < 2 else 65472)
    (record,) = measure(camera, plan())
    assert record.status == STATE_INCONCLUSIVE and record.level is None and record.gain_range is None
    assert any("levels differ between gains" in r and "65504" in r and "65472" in r for r in record.reasons)
    assert {p.value for p in record.plateaus} == {65472, 65504}


def test_one_gain_is_not_enough():
    (record,) = measure(FakeCamera(), plan(gains=(1.0,)))
    assert record.status == STATE_INCONCLUSIVE and any("at least 2" in r for r in record.reasons)


def test_a_camera_timeout_fails_that_bin_and_the_next_bin_goes_on():
    camera = FakeCamera(timeout_at={20.0})
    first, second = measure(camera, plan(bins=(1, 2), start=10.0))
    assert first.status == STATE_FAILED and "TimeoutError" in first.reasons[0]
    assert second.status == STATE_FAILED                               # the same exposure times out again
    healthy = FakeCamera()
    assert [r.status for r in measure(healthy, plan(bins=(1, 2)))] == [STATE_VERIFIED, STATE_VERIFIED]


def test_cancel_stops_the_sweep_and_returns_nothing():
    camera = FakeCamera()
    cancel = threading.Event()
    seen = []

    def progress(info):
        seen.append(info["exposure_ms"])
        if len(seen) == 2:
            cancel.set()

    with pytest.raises(MeasurementCancelled):
        measure(camera, plan(), cancel=cancel, progress=progress)
    assert len(seen) == 2 and len(camera.applied) == 2


def test_the_plan_is_checked_against_the_camera():
    with pytest.raises(ValueError):
        measure(FakeCamera(bins=(1, 2)), plan(bins=(3,)))
    with pytest.raises(ValueError):
        measure(FakeCamera(), plan(gains=(1.0, 900.0)))


# ---------------------------------------------------------------- real frames (10.5 reference)
def _real_steps():
    data = json.loads((FIXTURES / "poa_apollo_m_max_20261127_probe.json").read_text())
    out = {}
    for s in data["steps"]:
        hist = tuple({int(k): v for k, v in h.items()} for h in s["histograms"])
        out.setdefault(s["bin"], []).append(StepFrames(s["gain"], s["exposure_ms"], s["pixels"], hist,
                                                       tuple(s["maxima"])))
    return data, out


def test_real_apollo_frames_show_the_plateaus_the_probe_found():
    data, steps = _real_steps()
    assert data["serial"] == "CAMP7161A470A1209000"
    found = {(b, s.gain, s.exposure_ms): step_candidates(s, NOMINAL) for b, ss in steps.items() for s in ss}
    assert found[(1, 1, 500)] == {} and set(found[(1, 1, 2000)]) == {65504}
    assert set(found[(1, 200, 1000)]) == {65504} and set(found[(2, 1, 2000)]) == {65504}
    assert set(found[(3, 1, 2000)]) == {65535} and set(found[(3, 200, 1000)]) == {65520, 65535}
    assert found[(4, 1, 2000)] == {}                                   # 20 pixels: too few at bin 4
    # the probe was not a full sweep: nothing may be verified from it
    for b, ss in steps.items():
        ident = SaturationIdentity("poa", "Apollo-M MAX", data["serial"], "RAW16", 16, 12, b)
        record = analyse(ss, ident, NOMINAL)
        assert record.status == STATE_INCONCLUSIVE, b
    bin1 = analyse(steps[1], IDENT, NOMINAL)
    clipping = {p.value: p for p in bin1.plateaus}
    assert clipping[65504].role == "clipping" and "absent at 500 ms" in clipping[65504].reason
    crops = np.load(FIXTURES / "poa_apollo_m_max_20261127_crops.npz")
    assert crops["bin1_gain1_2000ms"].max() == 65504 and np.all(crops["bin1_gain1_2000ms"] % 16 == 0)
    assert 65535 in crops["bin3_gain200_1000ms"] and 65520 in crops["bin3_gain200_1000ms"]
    assert crops["bin2_gain1_2000ms"].max() == 65504


# ---------------------------------------------------------------- the store (10.2)
def _verified(identity=IDENT, level=65504, when="2026-11-27T10:00:00", gains=(1.0, 200.0)):
    return SaturationProfileRecord(identity, "verified", level, NOMINAL, gain_range=gains, gains_tested=gains,
                                   measured_at=when)


def _inconclusive(identity=IDENT, when="2026-11-28T10:00:00"):
    return SaturationProfileRecord(identity, "inconclusive", None, NOMINAL, reasons=("too few pixels",),
                                   measured_at=when)


def test_store_round_trip_and_states(tmp_path):
    store = SaturationProfileStore(CameraProfile.for_camera(tmp_path, "poa", "Apollo-M MAX", "CAMP7161")
                                   .saturation_folder)
    assert store.folder.name == "saturation"
    assert store.lookup(IDENT, 1).state == STATE_MISSING
    result = store.save(_verified())
    assert result.active and result.replaced is None and result.path.name == "saturation_RAW16_d16s12_bin1.json"
    found = store.lookup(IDENT, 1)
    assert found.usable and found.record == _verified() and "65504" in found.reason
    assert store.lookup(IDENT, 250).state == STATE_GAIN_OUTSIDE
    assert store.lookup(IDENT, None).state == STATE_GAIN_OUTSIDE        # unknown gain: not covered
    assert [p.name for p, r, e in store.active()] == ["saturation_RAW16_d16s12_bin1.json"]


def test_store_rejects_bad_files_and_other_configurations(tmp_path):
    store = SaturationProfileStore(tmp_path)
    path = store.path_for(IDENT)
    path.write_text("{not json")
    assert store.lookup(IDENT, 1).state == STATE_INVALID
    data = _verified().to_dict()
    data["format_version"] = 99
    path.write_text(json.dumps(data))
    found = store.lookup(IDENT, 1)
    assert found.state == STATE_INVALID and "format_version" in found.reason
    for change in ({"serial": "OTHER"}, {"pixel_format": "Mono12"}, {"delivered_bits": 8}, {"bin": 2}):
        data = _verified().to_dict()
        data["identity"].update(change)
        path.write_text(json.dumps(data))
        found = store.lookup(IDENT, 1)
        assert found.state == STATE_INCOMPATIBLE, change
    other_bits = SaturationIdentity("poa", "Apollo-M MAX", "CAMP7161", "RAW16", 16, 10, 1)
    assert store.path_for(other_bits) != store.path_for(IDENT)        # no collision between bit depths
    assert format_token("POAImgFormat.POA_RAW16") == "RAW16" and format_token(None) == "unknown"


def test_store_keeps_history_and_never_replaces_verified_with_a_failed_measurement(tmp_path):
    store = SaturationProfileStore(tmp_path)
    store.save(_verified(level=65504, when="2026-11-27T10:00:00"))
    second = store.save(_verified(level=65504, when="2026-11-27T11:00:00"))
    assert second.active and second.replaced is not None and second.replaced.parent == store.history_folder
    assert SaturationProfileRecord.from_dict(json.loads(second.replaced.read_text())).measured_at.endswith("10:00:00")
    attempt = store.save(_inconclusive())
    assert not attempt.active and attempt.path.parent == store.history_folder
    assert store.lookup(IDENT, 1).record.measured_at.endswith("11:00:00")    # the verified one stays
    failed = SaturationProfileRecord(IDENT, "failed", None, NOMINAL, reasons=("TimeoutError",),
                                     measured_at="2026-11-28T12:00:00")
    assert not store.save(failed).active
    assert len(store.history(IDENT)) == 3
    fresh = SaturationProfileStore(tmp_path / "other")                 # nothing verified yet: it becomes active
    assert fresh.save(_inconclusive()).active and fresh.lookup(IDENT, 1).state == STATE_INCONCLUSIVE


def test_store_save_is_atomic(tmp_path, monkeypatch):
    store = SaturationProfileStore(tmp_path)
    store.save(_verified(when="2026-11-27T10:00:00"))
    before = store.path_for(IDENT).read_bytes()

    def broken(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(store_module.os, "replace", broken)
    with pytest.raises(OSError):
        store.save(_verified(when="2026-11-27T12:00:00"))
    assert store.path_for(IDENT).read_bytes() == before                # old content intact
    assert not list(tmp_path.rglob("*.tmp"))                           # no temporary file left


def test_record_rules():
    with pytest.raises(ValueError):
        SaturationProfileRecord(IDENT, "verified", None, NOMINAL, gain_range=(1, 2))
    with pytest.raises(ValueError):
        SaturationProfileRecord(IDENT, "verified", 65504, NOMINAL)       # no gain range
    with pytest.raises(ValueError):
        SaturationProfileRecord(IDENT, "inconclusive", 65504, NOMINAL)
    with pytest.raises(ValueError):
        SaturationProfileRecord(IDENT, "maybe", None, NOMINAL)
    record = _verified()
    assert record.covers_gain(1) and record.covers_gain(200) and not record.covers_gain(201)
    assert record.profile_id() == "saturation_RAW16_d16s12_bin1@2026-11-27T10:00:00"
    assert SaturationProfileRecord.from_dict(record.to_dict()) == record
