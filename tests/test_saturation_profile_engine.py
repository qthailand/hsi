"""Phase 2E.2: the measured saturation level in acquisition, metadata and the GUI.

* ProfileLevelSource answers a level only for a verified profile of the exact configuration and
  gain range; everything else is not checked, with the reason;
* the tracker reads the profile once per configuration (not per frame), again after a bin / gain
  change or a saved profile (reload);
* the engine counts against the profile's level, keeps each frame's level and profile id with it
  through the recorder's trim, and a scan's ENVI header gets the saturation keys while
  ``data full scale`` (nominal) is unchanged; a level change during a scan is reported as mixed;
* the Clipped card never rounds a few clipped pixels away; the status bar shows the final scan
  statistics apart from the live card; the spectrum calls its ceiling bands near, not clipped.
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
from modules.acquisition import AcquisitionEngine, AcquisitionSink  # noqa: E402
from modules.acquisition.saturation_tracking import (RawCount, SaturationTracker,  # noqa: E402
                                                     scan_saturation_metadata)
from modules.contracts import CameraProfile, SaturationIdentity, SaturationProfileRecord  # noqa: E402
from modules.cube_io.writers import metadata_header_lines  # noqa: E402
from modules.saturation_profile import ProfileLevelSource, SaturationProfileStore  # noqa: E402
from modules.saturation_profile.sources import identity_of, nominal_full_scale_of  # noqa: E402

from test_saturation_tracking import H, W, ClipCamera, _wait  # noqa: E402

LEVEL = 65504


class Backend:
    """Just what the level source reads."""

    def __init__(self, bin_=1, gain=1.0, image_format="POAImgFormat.POA_RAW16"):
        self.current_bin, self.gain = bin_, gain
        self.info = SimpleNamespace(bit_depth=16, image_format=image_format)

    def sensor_bit_depth(self):
        return 12

    def model_name(self):
        return "Apollo-M MAX"

    def serial_number(self):
        return "CAMP7161"


def _save(folder, backend, level=LEVEL, gains=(1.0, 200.0), status="verified", when="2026-11-27T10:00:00"):
    ident = identity_of(backend, "poa")
    record = SaturationProfileRecord(ident, status, level if status == "verified" else None, 65520,
                                     gain_range=gains if status == "verified" else None, gains_tested=gains,
                                     measured_at=when)
    SaturationProfileStore(folder).save(record)
    return record


# ---------------------------------------------------------------- the level source
def test_level_source_answers_only_for_a_verified_profile_of_this_configuration(tmp_path):
    backend = Backend()
    source = ProfileLevelSource(tmp_path, "poa")
    missing = source(backend, np.uint16)
    assert missing.level is None and missing.source == "missing" and not missing.transient
    record = _save(tmp_path, backend)
    found = source(backend, np.uint16)
    assert found.level == LEVEL and found.source == "measured" and found.trusted
    assert found.profile_id == record.profile_id() and found.profile_status == "verified"
    assert "saturated pixels read 65504" in found.provenance and "gain 1-200" in found.provenance
    backend.gain = 300.0
    outside = source(backend, np.uint16)
    assert outside.level is None and outside.source == "gain_outside" and outside.profile_id == record.profile_id()
    backend.gain, backend.current_bin = 1.0, 2
    assert source(backend, np.uint16).source == "missing"                # bin 2 not measured
    backend.current_bin = 1
    assert source(backend, np.uint8).source == "incompatible"            # 65504 cannot be an 8-bit value
    other = Backend()
    other.serial_number = lambda: "ANOTHER"                              # a profile folder copied from another camera
    wrong = source(other, np.uint16)
    assert wrong.source == "incompatible" and wrong.level is None and "serial number" in wrong.reason
    _save(tmp_path, backend, status="inconclusive", when="2026-11-28T10:00:00")
    assert source(backend, np.uint16).level == LEVEL                     # ...an inconclusive one never replaces it
    assert ProfileLevelSource(None, "poa")(backend, np.uint16).source == "missing"
    broken = Backend()
    broken.info = property(lambda s: 1 / 0)
    broken.sensor_bit_depth = lambda: 1 / 0
    assert source(broken, np.uint16).source == "error"                   # transient: asked again later


def test_the_nominal_full_scale_is_never_the_level(tmp_path):
    backend = Backend()
    assert nominal_full_scale_of(backend) == 65520                      # camera_bit_info's value
    assert ProfileLevelSource(tmp_path, "poa")(backend, np.uint16).level is None


def test_tracker_reads_the_profile_once_per_configuration(tmp_path, monkeypatch):
    backend = Backend()
    _save(tmp_path, backend)
    source = ProfileLevelSource(tmp_path, "poa")
    reads = []
    original = source.store.lookup
    monkeypatch.setattr(source.store, "lookup", lambda *a: reads.append(a) or original(*a))
    tracker = SaturationTracker(backend, shared_preprocessing=True, level_source=source)
    frame = np.full((20, 30), 1000, np.uint16)
    frame[0, :6] = LEVEL
    for _ in range(50):
        assert tracker.count(frame)[:3] == (6, 6, LEVEL)
    assert len(reads) == 1                                               # no per-frame file I/O
    backend.gain = 50.0
    tracker.count(frame)
    assert len(reads) == 2                                               # another gain: asked again
    _save(tmp_path, backend, level=65520, when="2026-11-27T12:00:00")
    tracker.count(frame)
    assert len(reads) == 2 and tracker.count(frame).level == LEVEL       # not until reloaded
    tracker.reset_level()                                                # engine.reload_saturation_profile()
    assert tracker.count(frame).level == 65520 and len(reads) == 3
    stats = tracker.latest_raw()
    assert stats.profile_id.endswith("@2026-11-27T12:00:00") and stats.level == 65520


# ---------------------------------------------------------------- scan metadata
def _rc(clipped, level=LEVEL, pid="p1", pixels=1000):
    return RawCount(clipped, clipped, level, pixels, pid)


def test_scan_metadata_reports_the_kept_frames_and_their_level():
    meta = scan_saturation_metadata([_rc(0), _rc(5), None, _rc(30)], frames=4)
    assert meta == {"saturation_checked_frames": 3, "saturation_unchecked_frames": 1,
                    "saturation_clipped_frames": 2, "saturation_peak_clipped_percent": 3.0,
                    "saturation_level": LEVEL, "saturation_profile_id": "p1",
                    "saturation_profile_status": "verified",
                    "saturation_level_source": "measured saturation profile p1"}
    mixed = scan_saturation_metadata([_rc(1), _rc(1, level=65520, pid="p2")])
    assert "saturation_level" not in mixed and mixed["saturation_profile_id"] == "mixed"
    assert "65504 (p1) x1" in mixed["saturation_level_source"] and "65520 (p2) x1" in mixed["saturation_level_source"]
    none = scan_saturation_metadata([None, None])
    assert none["saturation_checked_frames"] == 0 and none["saturation_profile_status"] == "none"
    assert "saturation_level" not in none and "not checked" in none["saturation_level_source"]
    misaligned = scan_saturation_metadata([_rc(1)], frames=3)            # never pair with other frames
    assert misaligned["saturation_checked_frames"] == 0 and misaligned["saturation_unchecked_frames"] == 3


def test_header_gets_the_saturation_keys_and_keeps_the_nominal_full_scale():
    meta = scan_saturation_metadata([_rc(0), _rc(5)])
    lines = metadata_header_lines({"full_scale": 65520, **meta})
    assert "data full scale = 65520" in lines                           # nominal, unchanged
    assert "saturation level = 65504" in lines and "saturation profile id = p1" in lines
    assert "saturation clipped frames = 1" in lines and "saturation checked frames = 2" in lines
    assert "saturation peak clipped percent = 0.5" in lines
    assert not [l for l in metadata_header_lines({"full_scale": 65520}) if l.startswith("saturation")]


# ---------------------------------------------------------------- the engine
@pytest.fixture
def profiled_engine(tmp_path):
    camera = ClipCamera()
    profile = CameraProfile.for_camera(tmp_path / "profiles", "test", None, None)
    config = SimpleNamespace(smile_map_folder=str(tmp_path), preprocessing_backend="cpu", m=0.5, c=400.0,
                             resolution=1.5, saturation_tracking=False, camera_backend="test",
                             camera_profile=profile)
    seen = []
    sink = AcquisitionSink()
    sink.frame = lambda frame, fps, timestamp: seen.append(frame)
    engine = AcquisitionEngine(camera, config, sink=sink, profile=False)
    thread = threading.Thread(target=engine.run_until_stopped, daemon=True)
    thread.start()
    assert _wait(lambda: seen)
    yield engine, camera, profile
    engine.request_stop()
    thread.join(5)


def test_engine_without_a_profile_is_not_checked_then_reloads_a_saved_one(profiled_engine):
    from test_saturation_tracking import LEVEL as CLIP
    engine, camera, profile = profiled_engine
    assert _wait(lambda: engine.latest_raw_saturation() is not None)
    stats = engine.latest_raw_saturation()
    assert stats.status == "not_checked" and "no saturation profile" in stats.reason
    ident = identity_of(camera, "test")
    SaturationProfileStore(profile.saturation_folder).save(SaturationProfileRecord(
        ident, "verified", CLIP, 65520, gain_range=(0.0, 10.0), gains_tested=(0.0, 10.0),
        measured_at="2026-11-27T10:00:00"))
    time.sleep(0.1)
    assert engine.latest_raw_saturation().status == "not_checked"        # not re-read per frame
    engine.reload_saturation_profile()
    assert _wait(lambda: engine.latest_raw_saturation().status == "checked")
    engine.set_recording(True)
    assert _wait(lambda: engine.recorded_count() >= 20)
    images = engine.take_recorded_frames()
    counts = engine.take_recorded_raw_counts()
    assert len(counts) == len(images) and all(c is not None for c in counts)
    assert {c.profile_id for c in counts} == {f"{ident.file_stem()}@2026-11-27T10:00:00"}
    meta = scan_saturation_metadata(counts, len(images))
    assert meta["saturation_level"] == CLIP and meta["saturation_clipped_frames"] > 0


def test_recorder_trims_the_counts_with_their_frames():
    from modules.acquisition import ScanRecorder
    recorder = ScanRecorder()
    recorder.set_recording(True)
    for i in range(6):
        recorder.offer(np.zeros((2, 2), np.uint16), np.zeros((2, 2), np.uint16), False, stamp=i * 10,
                       raw_count=_rc(i))
    kept = recorder.take_frames(window=(15, 45))
    assert len(kept) == 3 and [c.clipped for c in recorder.take_raw_counts()] == [2, 3, 4]
    assert recorder.take_raw_counts() == []                              # handed over once


# ---------------------------------------------------------------- the GUI
@pytest.fixture
def qt_app():
    from PyQt5.QtWidgets import QApplication
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def test_a_few_clipped_pixels_never_round_away(qt_app):
    from test_basic_saturation_detection import _stats
    from windows.scan_inspector import _RED_BAR, ClippedTile, format_clipped_percent
    assert format_clipped_percent(0.0) == "0.00 %" and format_clipped_percent(0.004) == "<0.01 %"
    assert format_clipped_percent(None) == "—" and format_clipped_percent(1.234) == "1.23 %"
    tile = ClippedTile(0.1)
    tile.set_raw_saturation(_stats(pixels=1_775_232, clipped=3, near=3, saturated_frames=1,
                                   peak_clipped_percent=100 * 3 / 1_775_232))
    assert tile.value_label.text() == "<0.01 %" and _RED_BAR in tile.value_label.styleSheet()
    assert tile.max_label.text() == "max <0.01 %"


def test_status_bar_shows_the_last_scan_apart_from_the_live_card(qt_app):
    import config as app_config
    from windows.scan_window import ScanWindow, scan_clipping_text
    meta = scan_saturation_metadata([_rc(0), _rc(5), None])
    assert scan_clipping_text(meta) == "Scan clipped 1/2 lines (peak 0.50 %), 1 not checked"
    assert scan_clipping_text(scan_saturation_metadata([None])) == "Scan clipping not checked"
    assert scan_clipping_text(None) == ""
    win = ScanWindow(app_config.AppConfig())
    win.set_scan_clipping(meta)
    assert "Scan clipped 1/2 lines" in win.status_info()
    win.set_scan_clipping(None)
    assert "Scan clipped" not in win.status_info()
    win.close()


def test_spectrum_calls_its_ceiling_bands_near_not_clipped(qt_app):
    import inspect
    from modules.live_view import spectrum
    source = inspect.getsource(spectrum.LiveLineSpectrumWidget)
    assert "NEAR CEILING" in source and "SATURATED {" not in source
