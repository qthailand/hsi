"""Phase 2: saturation tracking in acquisition - level evidence, records, recorder alignment,
the tracker (on/off, not-checked reasons, mailbox, cost) and the engine loop end to end.

The engine test runs the real loop on a fake camera with a cubic smile + keystone model, hot
pixels, an inverted slit and a smile-trimmed (subset) binner, records with Keep Spectrogram on,
and checks every recorded frame twice against an independent recomputation from its raw frame:
the intensities equal the untracked pipeline's (tracking changes no pixel), and the saturation
record equals ``SaturationPropagator.frame`` with the same snapshot (frame identity).
"""
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules.acquisition import AcquisitionEngine, AcquisitionSink, ScanRecorder  # noqa: E402
from modules.acquisition.engine import crop_keystone_rows, keystone_row_margin  # noqa: E402
from modules.acquisition.saturation_tracking import SaturationTracker, format_scan_info  # noqa: E402
from modules.camera import AbstractCameraBackend  # noqa: E402
from modules.dark import DarkProcessor  # noqa: E402
from modules.preprocessing import PreparedFrame, PreprocessingRuntime  # noqa: E402
from modules.contracts import CameraProfile, SaturationProfileRecord  # noqa: E402
from modules.saturation_mask import (STATUS_CHECKED, STATUS_NOT_CHECKED, STATUS_PARTIAL,  # noqa: E402
                                     LevelEvidence, SaturationPropagator, SaturationRecord, bin_masks,
                                     scan_status, scan_summary)
from modules.saturation_profile import SaturationProfileStore  # noqa: E402
from modules.saturation_profile.sources import identity_of  # noqa: E402
from modules.spectral_processing import LambdaBinner  # noqa: E402

from test_saturation_mask import smile_model  # noqa: E402

LEVEL = 65520


# ---------------------------------------------------------------- the level
class _Info:
    def __init__(self, bits, image_format="Mono16"):
        self.bit_depth, self.image_format = bits, image_format


class StatedLevel:                       # a camera whose level the test level source knows
    def __init__(self, provenance="test profile: saturated pixels read 65520"):
        self.provenance, self.calls, self.fail = provenance, 0, 0
        self.current_bin, self.info, self.gain = 1, _Info(16), 1.0

    def saturation_level(self):
        self.calls += 1
        if self.fail:
            self.fail -= 1
            raise RuntimeError("GenTL timeout")
        return LEVEL


def stated(backend, dtype):
    """Test level source: the backend's level as if a verified profile measured it (no profile
    when the backend has no provenance)."""
    level = backend.saturation_level()
    provenance = getattr(backend, "provenance", None)
    if not provenance:
        return LevelEvidence(None, "missing", "no verified saturation profile (test)")
    return LevelEvidence(level, "measured", provenance, provenance, "test-profile@2026-11-27", "verified")


class BaseLevel(AbstractCameraBackend):
    """The base class: a nominal level (``2**bit_depth - 1``), never measured."""

    def __init__(self, delivered):
        super().__init__()
        self._delivered = delivered

    info = property(lambda s: _Info(s._delivered))

    def _open_device(self): pass
    def _close_device(self): pass
    def _start_device(self): pass
    def _stop_device(self): pass
    def _fetch_raw_frame(self, timeout_ms=1000): return None
    def set_params(self, gain, exposure_ms): pass
    def set_bin(self, bin_value): pass
    exposure_range_us = property(lambda s: (1, 100))
    gain_range = property(lambda s: (0, 10))
    current_bin = property(lambda s: 1)
    def get_bins(self): return (1,)
    def dropped_images_count(self): return 0
    def get_temperature(self): return 20.0


def verified_profile(folder, backend_name, camera, level=LEVEL, gains=(0.0, 100.0)):
    """Save a verified saturation profile for ``camera``'s configuration (tests of the engine)."""
    record = SaturationProfileRecord(identity_of(camera, backend_name), "verified", level, 65520, gain_range=gains,
                                     gains_tested=gains, measured_at="2026-11-27T10:00:00")
    SaturationProfileStore(folder).save(record)
    return record


def _poa(sensor_bits):
    from modules.camera.poa_camera import POACameraBackend
    camera = POACameraBackend.__new__(POACameraBackend)
    camera.bit_depth, camera._sensor_bits = 16, sensor_bits
    camera._model_name, camera._current_bin = "Apollo-M MAX", 1
    camera.camera_id, camera.pixel_size, camera.img_width, camera.img_height, camera.img_format = 0, 9.0, 8, 8, 1
    return camera


def test_backends_give_only_the_nominal_full_scale():
    """The measured level lives in the saturation profile; a backend's level is nominal (display
    ceilings) and no backend states one any more."""
    assert _poa(12).saturation_level() == 4095 << 4 == 65520
    assert _poa(10).saturation_level() == 1023 << 6 and _poa(16).saturation_level() == 65535
    from modules.camera.genicam import GenICamBackend
    from modules.camera.ximea_camera import XIMEACameraBackend
    for backend in (AbstractCameraBackend, GenICamBackend, XIMEACameraBackend, type(_poa(12))):
        assert not hasattr(backend, "saturation_level_source"), backend
    tracker = SaturationTracker(_poa(12), shared_preprocessing=True, enabled=True)    # no level source
    assert tracker.evidence(np.zeros((2, 2), np.uint16)).level is None


def test_tracker_asks_again_when_the_configuration_changes_or_a_query_failed(monkeypatch, caplog):
    import modules.acquisition.saturation_tracking as tracking
    backend = StatedLevel()
    tracker = SaturationTracker(backend, shared_preprocessing=True, enabled=True, level_source=stated)
    frame = np.zeros(SHAPE, np.uint16)
    caplog.set_level("INFO", logger=tracking.__name__)
    for _ in range(3):
        assert tracker.evidence(frame).trusted
    assert backend.calls == 1                                        # cached while nothing changes
    backend.current_bin = 2
    tracker.evidence(frame)
    backend.info = _Info(16, "Mono12Packed")
    tracker.evidence(frame)
    tracker.reset_level()                                            # bin / parameters applied, restart
    tracker.evidence(frame)
    tracker.evidence(frame.astype(np.uint8))                         # another dtype
    backend.gain = 50.0                                              # another gain (verified ranges)
    tracker.evidence(frame)
    assert backend.calls == 6
    assert sum("Saturation level" in r.getMessage() for r in caplog.records) == 1   # logged on change only
    backend.fail = 1                                                 # a transient SDK failure
    tracker.reset_level()
    assert tracker.evidence(frame).transient
    assert tracker.evidence(frame).transient and backend.calls == 7   # not hammered every frame
    monkeypatch.setattr(tracking, "LEVEL_RETRY_S", 0.0)
    assert tracker.evidence(frame).trusted and backend.calls == 8     # ... but asked again: recovered


# ---------------------------------------------------------------- records
def _frame_saturation(seed=0, width=40, rows=12, wide=False, density=0.1):
    rng = np.random.default_rng(seed)
    binner = LambdaBinner(m=0.5, c=399.791, step=40.0 if wide else 1.5).fit(width)
    near = rng.random((rows, width)) < density
    clipped = near & (rng.random((rows, width)) < 0.5)
    near_any, near_fraction = bin_masks(near, binner)
    clip_any, clip_fraction = bin_masks(clipped, binner)
    from modules.saturation_mask import FrameSaturation
    return FrameSaturation(clip_any, near_any, clip_fraction, near_fraction, 0.01, 0.02, LEVEL, 0.98), binner


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("density, sparse", [(0.004, True), (0.1, False), (1.0, False)])
def test_record_round_trip_is_exact(wide, density, sparse):
    frame, binner = _frame_saturation(wide=wide, rows=60, width=400, density=density)
    record = SaturationRecord.from_frame(frame, binner.w)
    sparse = sparse and not wide                                      # 80-pixel bins: most bands hit
    assert (record.near_index is not None) == sparse and (record.near_packed is None) == sparse
    assert record.near_count.dtype == (np.uint8 if binner.w.max() <= 255 else np.uint16)
    back = record.to_frame(binner.w)
    for name in ("clipped_mask", "near_saturation_mask", "affected_fraction", "near_affected_fraction"):
        a, b = getattr(frame, name), getattr(back, name)
        assert a.dtype == b.dtype and np.array_equal(a, b), name
    assert back.raw_clipped_fraction == frame.raw_clipped_fraction and back.level == LEVEL
    dense = sum(getattr(frame, n).nbytes for n in ("clipped_mask", "near_saturation_mask",
                                                    "affected_fraction", "near_affected_fraction"))
    assert record.nbytes < dense / 4                                  # what a recording keeps per frame
    packed = 2 * ((frame.near_saturation_mask.size + 7) // 8)        # two bit-packed masks
    if sparse:
        assert record.nbytes < packed / 2


def test_clean_record_and_scan_status():
    frame, binner = _frame_saturation()
    empty = np.zeros_like(frame.near_saturation_mask)
    from modules.saturation_mask import FrameSaturation
    clean = SaturationRecord.from_frame(
        FrameSaturation(empty, empty, np.zeros(empty.shape, np.float32), np.zeros(empty.shape, np.float32),
                        0.0, 0.0, LEVEL, 0.98), binner.w)
    assert not clean.any_near() and not clean.any_clipped() and clean.nbytes == 0
    assert not clean.to_frame(binner.w).near_saturation_mask.any()
    hit = SaturationRecord.from_frame(frame, binner.w)
    assert scan_status([clean, hit]) == STATUS_CHECKED
    assert scan_status([clean, None]) == STATUS_PARTIAL
    assert scan_status([None, None]) == scan_status([]) == STATUS_NOT_CHECKED
    summary = scan_summary([clean, hit, None])
    assert summary == {"status": STATUS_PARTIAL, "frames": 3, "checked": 2, "near_frames": 1,
                       "clipped_frames": 1, "bytes": hit.nbytes}


# ---------------------------------------------------------------- recorder
def test_recorder_keeps_saturation_with_its_frame_through_the_trim():
    recorder = ScanRecorder()
    recorder.set_recording(True)
    marks = [object() for _ in range(6)]
    for i, mark in enumerate(marks):
        recorder.offer(np.full((1, 1), i, np.uint16), np.zeros((1, 2), np.uint16), False,
                       stamp=100 * i, saturation=None if i == 3 else mark)
    frames = recorder.take_frames(window=(100, 400))                 # keeps frames 1..4
    kept = recorder.take_saturation()
    assert [int(f[0, 0]) for f in frames] == [1, 2, 3, 4]
    assert kept == [marks[1], marks[2], None, marks[4]]               # index for index, None = not checked
    assert recorder.take_saturation() == []                           # handed over once
    recorder.set_recording(True)
    recorder.offer(np.zeros((1, 1), np.uint16), np.zeros((1, 2), np.uint16), False)
    assert recorder.take_frames() and recorder.take_saturation() == [None]   # offer without: not checked


# ---------------------------------------------------------------- tracker
SHAPE = (18, 40)


def _snapshot(frame, calibration=None, inverted=False):
    proc = DarkProcessor()
    rng = np.random.default_rng(3)
    hot = np.zeros(SHAPE, np.uint8)
    hot[5, 7] = hot[10, 30] = 1
    proc.set_dark_reference(rng.integers(0, 50, SHAPE, dtype=np.uint16), hot)
    return PreparedFrame(frame, proc.dark_reference, True, inverted, calibration)


def _raw():
    raw = np.full(SHAPE, 1000, np.uint16)
    raw[4, 6:9] = LEVEL                                              # around a hot pixel
    raw[12, 20] = int(0.99 * LEVEL)                                  # near only
    return raw


def test_tracker_checks_only_with_evidence_and_the_frames_own_snapshot():
    binner = LambdaBinner(m=0.5, c=399.791, step=1.5).fit(SHAPE[1])
    raw = _raw()
    prepared = _snapshot(raw, smile_model("cubic", keystone=True), inverted=True)
    tracker = SaturationTracker(StatedLevel(), shared_preprocessing=True, level_source=stated)
    assert tracker.track(raw, prepared, binner) is None and tracker.latest().status == "off"
    tracker.set_enabled(True)
    record = tracker.track(raw, prepared, binner, row_crop=(2, 1))
    expected = SaturationPropagator().frame(raw, LEVEL, prepared.hot, True, prepared.calibration, True, binner,
                                            row_crop=(2, 1))
    got = record.to_frame(binner.w)
    assert np.array_equal(got.clipped_mask, expected.clipped_mask) and got.clipped_mask.any()
    assert np.array_equal(got.near_affected_fraction, expected.near_affected_fraction)
    latest = tracker.latest()
    assert latest.status == STATUS_CHECKED and latest.level == LEVEL and latest.saturation is not None
    assert latest.provenance == "test profile: saturated pixels read 65520"
    # not checked: another frame's snapshot, a backend correcting its own frames, a guessed level
    assert tracker.track(raw.copy(), prepared, binner) is None
    assert "snapshot" in tracker.latest().reason
    legacy = SaturationTracker(StatedLevel(), shared_preprocessing=False, enabled=True, level_source=stated)
    assert legacy.track(raw, prepared, binner) is None and legacy.latest().status == STATUS_NOT_CHECKED
    guessed = SaturationTracker(BaseLevel(16), shared_preprocessing=True, enabled=True)
    assert guessed.track(raw, prepared, binner) is None and "not measured" in guessed.latest().reason


def test_tracker_failure_is_not_checked_and_never_raises(caplog):
    raw = _raw()
    tracker = SaturationTracker(StatedLevel(), shared_preprocessing=True, enabled=True, level_source=stated)
    wrong = LambdaBinner(m=0.5, c=399.791, step=1.5).fit(SHAPE[1] + 4)   # fitted for another width
    for _ in range(3):
        assert tracker.track(raw, _snapshot(raw), wrong) is None
    assert tracker.latest().status == STATUS_NOT_CHECKED and "failed" in tracker.latest().reason
    assert sum("Saturation mask failed" in r.getMessage() for r in caplog.records) == 1   # logged once


def test_tracker_meters_live_and_per_scan_cost():
    binner = LambdaBinner(m=0.5, c=399.791, step=1.5).fit(SHAPE[1])
    tracker = SaturationTracker(StatedLevel(), shared_preprocessing=True, enabled=True, level_source=stated)
    tracker.start_scan()
    clean = np.full(SHAPE, 1000, np.uint16)
    for i in range(6):
        raw = _raw() if i % 2 else clean
        tracker.track(raw, _snapshot(raw), binner)
        tracker.note_frame(4.0, recorded=i < 5)
    live = tracker.live_info(period_ms=10.0, period_source="reported")
    assert live["frames"] == 6 and live["near_frames"] == 3 and live["clipped_frames"] == 3
    assert live["busy_ms"] == pytest.approx(4.0) and live["load"] == pytest.approx(0.4)
    assert live["period_source"] == "reported" and live["overhead"] == pytest.approx(live["sat_ms"] / 10.0)
    assert tracker.live_info(period_ms=10.0)["period_source"] == "measured"
    assert live["sat_ms"] is not None and live["sat_max_ms"] >= live["sat_ms"]
    scan = tracker.scan_info(period_ms=10.0, period_source="reported")
    assert scan["frames"] == scan["checked"] == 5 and scan["near_frames"] == scan["clipped_frames"] == 2
    assert scan["levels"] == [{"level": LEVEL, "provenance": "test profile: saturated pixels read 65520",
                               "profile_id": "test-profile@2026-11-27", "frames": 5}]
    assert scan["not_checked"] == {}
    assert scan["busy_max_ms"] == pytest.approx(4.0) and scan["period_ms"] == 10.0
    line = format_scan_info(scan, {"status": "checked", "checked": 5, "frames": 5, "near_frames": 2,
                                   "clipped_frames": 2, "bytes": 2048})
    assert "mask checked: 5/5 frames checked, 2 near, 2 clipped, 2 KiB; mask " in line
    assert "of a 10.00 ms frame period (reported)" in line
    tracker.set_enabled(False)
    assert tracker.live_info()["frames"] == 0 and not tracker.live_info()["tracking"]


# ---------------------------------------------------------------- engine loop
H, W = 18, 40


class ClipCamera(AbstractCameraBackend):
    """Frames with a ramp, a few clipped pixels (moving) and a near-saturated one on odd frames."""

    def __init__(self):
        super().__init__()
        self.index = 0

    def _open_device(self): pass
    def _close_device(self): pass
    def _start_device(self): pass
    def _stop_device(self): pass

    def _fetch_raw_frame(self, timeout_ms=1000):
        time.sleep(0.002)
        self.index += 1
        i = self.index
        frame = (np.add.outer(np.arange(H) * 37, np.arange(W) * 11) % 3000 + 500).astype(np.uint16)
        if i % 2:
            frame[(i // 2) % H, (3 * i) % W] = LEVEL
            frame[(i + 5) % H, (7 * i) % W] = LEVEL
            frame[(i + 9) % H, (5 * i) % W] = int(0.985 * LEVEL)
        frame[0, 0] = i % 60000 // 1                                  # frame number, for alignment
        return frame

    gain = 1.0

    def set_params(self, gain, exposure_ms): pass
    def frame_rate(self): return 400.0
    def set_bin(self, bin_value): pass
    info = property(lambda s: SimpleNamespace(height=H, width=W, bit_depth=16))
    exposure_range_us = property(lambda s: (1, 100))
    gain_range = property(lambda s: (0, 10))
    current_bin = property(lambda s: 1)
    def get_bins(self): return (1,)
    def dropped_images_count(self): return 0
    def get_temperature(self): return 20.0


def _wait(condition, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def clip_engine(tmp_path):
    camera = ClipCamera()
    hot = np.zeros((H, W), np.uint8)
    hot[3, 9] = hot[11, 25] = hot[(4, 4, 5), (14, 15, 14)] = 1
    camera.set_dark_reference(np.random.default_rng(5).integers(0, 60, (H, W), dtype=np.uint16), hot)
    camera.set_dark_enabled(True)
    profile = CameraProfile.for_camera(tmp_path / "profiles", "test", None, None)
    verified_profile(profile.saturation_folder, "test", camera)       # the level: a measured profile
    config = SimpleNamespace(smile_map_folder=str(tmp_path), preprocessing_backend="cpu", m=0.5, c=400.0,
                             resolution=1.5, saturation_tracking=True, camera_backend="test",
                             camera_profile=profile)
    sink = AcquisitionSink()
    frames_seen = []
    sink.frame = lambda frame, fps, timestamp: frames_seen.append(frame)
    engine = AcquisitionEngine(camera, config, sink=sink, profile=False)
    calibration = smile_model("cubic", keystone=True, shape=(H, W))
    engine._active_calibration = lambda: calibration
    engine.set_raw_inverted(True)
    thread = threading.Thread(target=engine.run_until_stopped, daemon=True)
    thread.start()
    assert _wait(lambda: frames_seen)
    yield engine, camera, calibration
    engine.request_stop()
    thread.join(5)


def _record(engine, frames=30):
    engine.set_keep_spectrogram(True)
    engine.set_recording(True)
    assert _wait(lambda: engine.recorded_count() >= frames)
    images = engine.take_recorded_frames()
    return images, engine.take_recorded_raw(), engine.take_recorded_saturation()


def test_engine_records_each_frames_own_saturation_and_unchanged_intensities(clip_engine):
    engine, camera, calibration = clip_engine
    images, raw, saturation = _record(engine)
    assert len(images) == len(raw) == len(saturation) >= 30
    binner = engine._binner
    assert binner.n_bins < LambdaBinner(m=0.5, c=400.0, step=1.5).fit(W).n_bins     # smile-trimmed subset
    crop = keystone_row_margin(calibration, H)
    assert crop == (2, 1)
    reference = PreprocessingRuntime("cpu")
    propagator = SaturationPropagator()
    clipped_frames = 0
    for image, frame, record in zip(images, raw, saturation):
        expected, _, _ = reference.process_binned(frame, camera, binner, calibration=calibration, inverted=True)
        assert np.array_equal(image, crop_keystone_rows(expected, calibration))   # not one pixel changed
        truth = propagator.frame(frame, LEVEL, camera.dark_reference, True, calibration, True, binner,
                                 row_crop=crop)
        got = record.to_frame(binner.w)
        assert got.clipped_mask.shape == image.shape
        for name in ("clipped_mask", "near_saturation_mask", "affected_fraction", "near_affected_fraction"):
            assert np.array_equal(getattr(got, name), getattr(truth, name)), name
        clipped_frames += got.any_clipped()
    assert clipped_frames >= len(images) // 3                         # the odd frames clip
    assert scan_status(saturation) == STATUS_CHECKED
    latest = engine.latest_saturation()
    assert latest.status == STATUS_CHECKED and latest.level == LEVEL
    info = engine.saturation_scan_info()
    assert info["checked"] >= len(images) and info["sat_ms"] > 0 and info["busy_ms"] >= info["sat_ms"]
    live = engine.saturation_live_info()
    assert live["period_ms"] == pytest.approx(2.5) and live["period_source"] == "reported"   # 400 Hz camera
    assert "measured saturation profile" in info["levels"][0]["provenance"]
    assert info["levels"][0]["profile_id"] == "saturation_unknown_d16s16_bin1@2026-11-27T10:00:00"


def test_a_ceiling_recomputed_on_the_frame_does_not_take_its_snapshot(clip_engine):
    """The saturation ceiling runs process_binned on a synthetic frame after the real one; the
    frame must still be checked with its own snapshot (regression: last_prepared was read late)."""
    engine, camera, calibration = clip_engine
    original = engine._update_saturation_ceiling

    def every_frame(*args, **kwargs):
        engine._ceiling_key = ()                                    # a setting changed: recompute now
        return original(*args, **kwargs)

    engine._update_saturation_ceiling = every_frame
    images, raw, saturation = _record(engine, frames=12)
    assert scan_status(saturation) == STATUS_CHECKED
    binner, crop = engine._binner, keystone_row_margin(calibration, H)
    propagator = SaturationPropagator()
    for frame, record in zip(raw, saturation):
        truth = propagator.frame(frame, LEVEL, camera.dark_reference, True, calibration, True, binner,
                                 row_crop=crop)
        assert np.array_equal(record.to_frame(binner.w).clipped_mask, truth.clipped_mask)


def test_engine_toggle_gives_not_checked_and_partial(clip_engine):
    engine, _, _ = clip_engine
    engine.set_saturation_tracking(False)
    assert not engine.saturation_tracking
    _, _, saturation = _record(engine, frames=10)
    assert saturation and scan_status(saturation) == STATUS_NOT_CHECKED
    assert engine.latest_saturation().status == "off"
    engine.set_keep_spectrogram(False)
    engine.set_recording(True)
    assert _wait(lambda: engine.recorded_count() >= 5)
    engine.set_saturation_tracking(True)                              # switched during a recording
    start = engine.recorded_count()
    assert _wait(lambda: engine.recorded_count() >= start + 5)
    engine.take_recorded_frames()
    assert scan_status(engine.take_recorded_saturation()) == STATUS_PARTIAL


def test_preprocessing_records_the_snapshot_it_used():
    runtime = PreprocessingRuntime("cpu")
    frame = np.full(SHAPE, 100, np.uint16)
    proc = DarkProcessor()
    proc.set_dark_reference(np.zeros(SHAPE, np.uint16), np.zeros(SHAPE, np.uint8))
    runtime.process(frame, proc)
    prepared = runtime.last_prepared
    assert prepared.frame is frame and prepared.hot is proc.dark_reference and prepared.enabled
    other = np.full((4, 4), 1, np.uint16)                             # a reference of another shape: none used
    runtime.process(other, proc)
    assert runtime.last_prepared.hot is not proc.dark_reference


def test_cost_label_shows_overhead_and_load_apart_and_marks_an_estimated_period():
    from PyQt5.QtWidgets import QApplication
    global _APP                                                       # kept alive for the widget
    _APP = QApplication.instance() or QApplication([])
    from modules.camera.widget import CameraControls
    controls = CameraControls()
    controls.set_saturation_tracking(True)
    info = {"tracking": True, "status": "checked", "sat_ms": 0.5, "sat_max_ms": 2.0, "busy_ms": 6.0,
            "period_ms": 100.0, "period_source": "reported", "load": 0.06, "overhead": 0.005,
            "frames": 20, "near_frames": 0, "clipped_frames": 0, "reason": "level 65520: test"}
    controls.show_saturation_cost(info)
    label = controls.saturation_cost_label
    assert label.text() == "mask +0.5 ms · load 6 %"
    assert "reported by the camera" in label.toolTip() and "0 % of the frame period" in label.toolTip()
    controls.show_saturation_cost(dict(info, period_source="measured"))
    assert label.text() == "mask +0.5 ms · load ~6 %" and "estimate" in label.toolTip()
    controls.show_saturation_cost({"tracking": True, "status": "not_checked", "reason": "assumed level"})
    assert label.text() == "not checked" and label.toolTip() == "assumed level"
    controls.set_saturation_tracking(False)
    assert label.isHidden()
