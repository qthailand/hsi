"""Phase 2B: basic saturation detection - clipped / near counted on every acquired raw frame.

* the one-pass count equals NumPy for every dtype, thread count, layout and threshold edge, does
  not modify the frame and leaves the caller's Numba thread count (the binner's) as it was;
* the full mask reuses the counts (``detect_raw(counts=...)`` equals the uncounted result, and a
  frame counted clean is not read again);
* the tracker keeps the peak / saturated frames since the reset, says *not checked* / *partial*
  without a stated level, and its mailbox shows a reset at once;
* the engine counts every acquired frame (each frame's count checked against its own truth),
  hands the counts to full tracking and changes no intensity;
* the Clipped tile shows the engine's numbers (confirmed clipping), near full scale as the
  exposure warning, *not checked* / *partial*, and its reset reaches the engine.
"""
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numba
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules.acquisition.engine import crop_keystone_rows  # noqa: E402
from modules.acquisition.saturation_tracking import (RawSaturationStats, SaturationTracker,  # noqa: E402
                                                     format_scan_info)
from modules.preprocessing import PreprocessingRuntime  # noqa: E402
from modules.saturation_mask import (STATUS_CHECKED, STATUS_NOT_CHECKED, STATUS_PARTIAL,  # noqa: E402
                                     RawCounter, SaturationPropagator, count_saturated, detect_raw,
                                     near_threshold)
from modules.saturation_mask import count as count_module  # noqa: E402
from modules.spectral_processing import LambdaBinner  # noqa: E402

from test_saturation_tracking import (H, LEVEL, W, BaseLevel, StatedLevel, _record, _wait,  # noqa: E402,F401
                                      clip_engine, stated)

NEAR = near_threshold(LEVEL)


def _truth(frame, level=LEVEL, ratio=0.98):
    return int(np.count_nonzero(frame >= level)), int(np.count_nonzero(frame >= ratio * level))


# ---------------------------------------------------------------- the count
@pytest.mark.parametrize("threads", [1, 2, 4, 8, 16, 24, None])
@pytest.mark.parametrize("dtype,level", [(np.uint16, 65520), (np.uint16, 4095), (np.uint8, 255),
                                         (np.int32, 1000), (np.float32, 1000.0)])
def test_count_equals_numpy(threads, dtype, level):
    rng = np.random.default_rng(int(level) + (threads or 0))
    top = 256 if dtype == np.uint8 else int(level) + 30
    frame = rng.integers(0, top, (37, 53)).astype(dtype)
    before = frame.copy()
    assert count_saturated(frame, level, threads=threads) == _truth(frame, level)
    assert np.array_equal(frame, before)                                 # never modified


def test_count_thresholds_layouts_and_errors():
    frame = np.array([[NEAR - 1, NEAR, LEVEL - 1, LEVEL, 65535]], np.uint16)
    assert NEAR == 64210 and count_saturated(frame, LEVEL) == (2, 4)    # ceil(0.98 * 65520) is near
    assert detect_raw(frame, LEVEL).raw_near_fraction == pytest.approx(4 / 5)   # the same pixels
    big = np.random.default_rng(1).integers(60000, 65536, (300, 400)).astype(np.uint16)
    view = big[::3, 7:]                                                  # strided, not contiguous
    for threads in (1, 4, None):
        assert count_saturated(view, LEVEL, threads=threads) == _truth(view)
        assert count_saturated(big.T, LEVEL, threads=threads) == _truth(big.T)
    assert count_saturated(np.zeros((0, 5), np.uint16), LEVEL) == (0, 0)
    assert count_saturated(np.full((2, 2), 250, np.uint8), 300) == (0, 0)        # level above uint8
    assert count_saturated(frame, LEVEL, near_ratio=1.0) == (2, 2)
    for bad in (dict(raw=np.zeros(5, np.uint16), level=10), dict(raw=frame, level=0),
                dict(raw=frame, level=10, near_ratio=0.0)):
        with pytest.raises(ValueError):
            count_saturated(**bad)


def test_count_leaves_the_callers_thread_count_alone():
    """``set_num_threads`` is thread-local: the count restores it, and another thread (the GUI,
    or a binner with its own setting) never sees the change."""
    frame = np.random.default_rng(2).integers(0, 65536, (600, 700)).astype(np.uint16)
    seen = {}

    def worker():
        numba.set_num_threads(3)                                         # e.g. a binner's own setting
        count_saturated(frame, LEVEL, threads=8)
        seen["after"] = numba.get_num_threads()

    main = numba.get_num_threads()
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert seen["after"] == 3 and numba.get_num_threads() == main
    count_saturated(frame, LEVEL, threads=2)
    assert numba.get_num_threads() == main


def test_thread_choice_by_frame_size_and_warm_up(monkeypatch):
    assert count_module.threads_for(10) == count_module.threads_for(200_000)     # small: one table entry
    assert count_module.threads_for(2_000_000) >= 1
    assert count_module.threads_for(10, table=((100, 1), (np.inf, 4))) == 1
    assert count_module.threads_for(1000, table=((100, 1), (np.inf, 4))) == min(4, numba.config.NUMBA_NUM_THREADS)
    counter = RawCounter()
    counter.warm_up()
    assert counter.warm and counter.count(np.full((3, 3), LEVEL, np.uint16), LEVEL) == (9, 9)


# ---------------------------------------------------------------- full tracking reuses the count
def test_detect_raw_with_counts_equals_without():
    rng = np.random.default_rng(4)
    for case in range(6):
        raw = rng.integers(50000, 65536, (20, 30)).astype(np.uint16)
        if case == 0:
            raw[:] = 1000                                                # nothing near
        if case == 1:
            raw[raw >= NEAR] = LEVEL                                     # every near pixel clipped
        counts = count_saturated(raw, LEVEL)
        a, b = detect_raw(raw, LEVEL), detect_raw(raw, LEVEL, counts=counts)
        assert np.array_equal(a.clipped, b.clipped) and np.array_equal(a.near, b.near)
        assert (a.raw_clipped_fraction, a.raw_near_fraction) == (b.raw_clipped_fraction, b.raw_near_fraction)
        assert (b.clipped is b.near) == (counts[0] == counts[1])
    # a frame counted clean is not read again (the counts are the fast path)
    hot = np.full((4, 4), LEVEL, np.uint16)
    assert not detect_raw(hot, LEVEL, counts=(0, 0)).any()
    with pytest.raises(ValueError):
        detect_raw(hot, LEVEL, counts=(5, 2))                            # clipped > near


def test_propagator_with_counts_equals_without():
    from test_saturation_mask import smile_model
    rng = np.random.default_rng(6)
    binner = LambdaBinner(m=0.5, c=399.791, step=1.5).fit(W)
    calibration = smile_model("cubic", keystone=True, shape=(H, W))
    for _ in range(4):
        raw = rng.integers(1000, 60000, (H, W)).astype(np.uint16)
        raw[rng.integers(0, H, 5), rng.integers(0, W, 5)] = LEVEL
        raw[rng.integers(0, H, 3), rng.integers(0, W, 3)] = NEAR
        a = SaturationPropagator().frame(raw, LEVEL, None, False, calibration, True, binner, row_crop=(2, 1))
        b = SaturationPropagator().frame(raw, LEVEL, None, False, calibration, True, binner, row_crop=(2, 1),
                                         counts=count_saturated(raw, LEVEL))
        for name in ("clipped_mask", "near_saturation_mask", "affected_fraction", "near_affected_fraction"):
            assert np.array_equal(getattr(a, name), getattr(b, name)), name


# ---------------------------------------------------------------- the tracker
def _frame(clipped=0, near=0, shape=(10, 20)):
    frame = np.full(shape, 1000, np.uint16)
    flat = frame.reshape(-1)
    flat[:clipped] = LEVEL
    flat[clipped:clipped + near] = NEAR
    return frame


def test_tracker_keeps_the_peak_and_saturated_frames_since_reset():
    tracker = SaturationTracker(StatedLevel(), shared_preprocessing=True, level_source=stated)       # full tracking off
    assert tracker.latest_raw() is None
    for clipped, near in ((0, 0), (4, 2), (1, 0), (0, 3)):
        assert tracker.count(_frame(clipped, near))[:3] == (clipped, clipped + near, LEVEL)
    stats = tracker.latest_raw()
    assert isinstance(stats, RawSaturationStats) and stats.status == STATUS_CHECKED
    assert (stats.clipped, stats.near, stats.pixels) == (0, 3, 200)
    assert stats.clipped_percent == 0.0 and stats.near_percent == pytest.approx(1.5)
    assert stats.level == LEVEL and stats.near_level == NEAR and stats.count_ms >= 0
    assert stats.provenance == "test profile: saturated pixels read 65520"
    assert stats.profile_id == "test-profile@2026-11-27"
    assert (stats.frames, stats.checked_frames, stats.saturated_frames, stats.near_frames) == (4, 4, 2, 3)
    assert stats.peak_clipped_percent == pytest.approx(2.0) and stats.peak_near_percent == pytest.approx(3.0)
    assert stats.window_status == STATUS_CHECKED
    sequence = stats.sequence
    tracker.reset_peak()                                                 # the mailbox shows it at once
    reset = tracker.latest_raw()
    assert reset.sequence > sequence and reset.peak_clipped_percent is None and reset.saturated_frames == 0
    assert reset.clipped_percent == 0.0                                  # the newest frame is kept
    tracker.count(_frame(2))
    assert tracker.latest_raw().peak_clipped_percent == pytest.approx(1.0)
    tracker.start_scan()                                                 # Start scan resets too
    assert tracker.latest_raw().peak_clipped_percent is None


def test_tracker_without_evidence_is_not_checked_or_partial(caplog):
    backend = StatedLevel()
    tracker = SaturationTracker(backend, shared_preprocessing=True, level_source=stated)
    tracker.count(_frame(3))
    backend.provenance = None                                            # no verified profile any more
    tracker.reset_level()
    assert tracker.count(_frame(3)) is None
    stats = tracker.latest_raw()
    assert stats.status == STATUS_NOT_CHECKED and stats.clipped_percent is None and "no verified" in stats.reason
    assert stats.window_status == STATUS_PARTIAL and stats.peak_clipped_percent == pytest.approx(1.5)
    tracker.reset_peak()
    tracker.count(_frame(3))
    assert tracker.latest_raw().window_status == STATUS_NOT_CHECKED
    assert SaturationTracker(BaseLevel(16), shared_preprocessing=True).count(_frame(1)) is None
    legacy = SaturationTracker(StatedLevel(), shared_preprocessing=False, level_source=stated)
    assert legacy.count(_frame(1)) is None and "itself" in legacy.latest_raw().reason
    broken = SaturationTracker(StatedLevel(), shared_preprocessing=True, level_source=stated)
    for _ in range(3):
        assert broken.count(np.zeros(5, np.uint16)) is None             # 1-D: the count raises
    assert "count failed" in broken.latest_raw().reason
    assert sum("Saturation count failed" in r.getMessage() for r in caplog.records) == 1   # logged once


def test_scan_totals_count_only_the_recorded_frames():
    tracker = SaturationTracker(StatedLevel(), shared_preprocessing=True, level_source=stated)
    tracker.start_scan()
    for i, clipped in enumerate((0, 6, 2, 9)):
        tracker.count(_frame(clipped))
        tracker.note_frame(3.0, recorded=i < 3)                          # the last one not recorded
    info = tracker.scan_info()
    assert info["frames"] == info["raw_checked"] == 3 and info["raw_status"] == STATUS_CHECKED
    assert info["raw_saturated_frames"] == 2 and info["raw_peak_clipped_percent"] == pytest.approx(3.0)
    assert info["count_ms"] is not None and info["count_max_ms"] >= info["count_ms"]
    assert tracker.latest_raw().peak_clipped_percent == pytest.approx(4.5)   # the tile sees every frame
    line = format_scan_info(info, {"status": "not_checked", "checked": 0, "frames": 3})
    assert line.startswith("raw checked: 3/3 frames counted, 2 clipped (peak 3.000 %), 2 near, count ")
    live = tracker.live_info()
    assert live["count_ms"] is not None and live["frames"] == 4


# ---------------------------------------------------------------- the engine
def test_engine_counts_every_frame_and_hands_the_count_to_tracking(clip_engine):
    engine, camera, calibration = clip_engine
    tracker = engine._saturation_tracker
    assert tracker._counter.warm                                         # warmed up before the first frame
    seen, handed, lock = [], [], threading.Lock()
    count, frame_fn = tracker.count, tracker._propagator.frame

    def spy_count(frame, profiler=None):
        result = count(frame, profiler)
        with lock:
            seen.append((result, _truth(frame), frame))
        return result

    def spy_frame(*args, **kwargs):
        handed.append(kwargs.get("counts"))
        return frame_fn(*args, **kwargs)

    tracker.count, tracker._propagator.frame = spy_count, spy_frame
    assert _wait(lambda: len(seen) >= 3)                                 # no frame in flight from before
    images, raw, saturation = _record(engine, frames=40)
    with lock:
        checked = list(seen)
    assert len(checked) >= len(images)
    for result, truth, _ in checked:                                     # every frame, not the first
        assert result is not None and result[:2] == truth and result[2] == LEVEL
    assert any(t[0] for _, t, _ in checked) and any(not t[0] for _, t, _ in checked)
    counted = {f.tobytes(): r for r, _, f in checked}                   # the recorder keeps copies
    for frame in raw:                                                    # each recorded raw frame counted,
        assert counted[frame.tobytes()][:2] == _truth(frame)             # with that frame's own counts
    assert handed and all(h is not None for h in handed)                 # tracking reused the count
    reference = PreprocessingRuntime("cpu")
    for image, frame in zip(images, raw):                                # not one intensity changed
        expected, _, _ = reference.process_binned(frame, camera, engine._binner, calibration=calibration,
                                                  inverted=True)
        assert np.array_equal(image, crop_keystone_rows(expected, calibration))
    stats = engine.latest_raw_saturation()
    assert stats.status == STATUS_CHECKED and stats.saturated_frames > 0 and stats.peak_clipped_percent > 0
    info = engine.saturation_scan_info()
    assert info["raw_checked"] == info["frames"] >= len(images) and info["raw_saturated_frames"] > 0
    engine.reset_saturation_peak()
    assert engine.latest_raw_saturation().saturated_frames == 0


def test_engine_with_tracking_off_still_counts(clip_engine):
    engine, _, _ = clip_engine
    engine.set_saturation_tracking(False)
    # a frame in flight when the switch flipped is still tracked: record from the first frame after it
    assert _wait(lambda: getattr(engine.latest_saturation(), "status", None) == "off")
    _, _, saturation = _record(engine, frames=10)
    assert all(s is None for s in saturation)                            # no mask
    info = engine.saturation_scan_info()
    assert info["raw_status"] == STATUS_CHECKED and info["raw_checked"] == info["frames"] >= 10
    assert info["raw_saturated_frames"] > 0 and info["checked"] == 0


# ---------------------------------------------------------------- the Clipped tile
@pytest.fixture
def qt_app():
    from PyQt5.QtWidgets import QApplication
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _stats(**kw):
    base = dict(sequence=1, status=STATUS_CHECKED, reason="test", pixels=1000, clipped=0, near=0, level=LEVEL,
                near_level=NEAR, provenance="test camera", count_ms=0.1, frames=5, checked_frames=5,
                saturated_frames=0, near_frames=0, peak_clipped_percent=0.0, peak_near_percent=0.0)
    base.update(kw)
    return RawSaturationStats(**base)


def test_tile_shows_the_engines_numbers(qt_app):
    from windows.scan_inspector import _AMBER, _RED_BAR, ClippedTile
    tile = ClippedTile(0.1)
    assert tile.value_label.text() == "—" and tile.saturated_percent() is None
    tile.update_histogram(np.full((10, 100), 65535, np.uint16))          # bars only: no number from them
    assert tile.saturated_percent() is None and tile.bars.counts is not None
    tile.set_raw_saturation(_stats(clipped=5, near=7, saturated_frames=3, peak_clipped_percent=1.2))
    assert tile.value_label.text() == "0.50 %" and _RED_BAR in tile.value_label.styleSheet()   # confirmed
    assert tile.max_label.text() == "max 1.20 %" and _RED_BAR in tile.max_label.styleSheet()
    assert tile.frames_label.text() == "3 fr" and _RED_BAR in tile.frames_label.styleSheet()
    assert "5 of 1,000" in tile.value_label.toolTip() and "test camera" in tile.value_label.toolTip()
    tile.set_raw_saturation(_stats(near=4))                              # near only: exposure warning
    assert tile.value_label.text() == "0.00 %" and _AMBER in tile.value_label.styleSheet()
    assert tile.near_percent() == pytest.approx(0.4) and "lower the exposure" in tile.value_label.toolTip()
    tile.set_raw_saturation(_stats(frames=9, peak_clipped_percent=0.05))
    assert (tile.max_label.text(), tile.frames_label.text()) == ("max ≥0.05 %", "≥0 fr")   # partial
    assert "4 of 9 frames not checked" in tile.max_label.toolTip()
    tile.set_raw_saturation(_stats(status=STATUS_NOT_CHECKED, reason="level assumed", checked_frames=0,
                                   peak_clipped_percent=None))
    assert tile.value_label.text() == "not checked" and tile.saturated_percent() is None
    assert "level assumed" in tile.value_label.toolTip() and tile.max_label.text() == "—"
    assert tile.check_status() == STATUS_NOT_CHECKED
    resets = []
    tile.peak_reset.connect(lambda: resets.append(True))
    tile.set_raw_saturation(_stats(saturated_frames=2, peak_clipped_percent=0.3))
    tile.reset_btn.click()
    assert resets == [True] and tile.max_label.text() == "—" and tile.max_saturated_percent() is None


def test_controller_feeds_the_tile_from_the_mailbox_and_resets_the_engine(qt_app):
    from controllers.scan_controller import HISTOGRAM_UPDATE_INTERVAL_SEC, ScanController
    from windows.scan_inspector import ClippedTile
    tile = ClippedTile(0.1)
    stats = [_stats(clipped=20)]
    resets = []
    controller = ScanController.__new__(ScanController)
    controller.win = SimpleNamespace(histogram=tile)
    controller.preview_thread = SimpleNamespace(latest_raw_saturation=lambda: stats[0],
                                                reset_saturation_peak=lambda: resets.append(True))
    controller.last_histogram_update = 0.0
    controller.enable_histogram = False                                  # the numbers still update
    controller.saturation_ceiling = None
    controller._update_live_histogram(np.zeros((4, 4), np.uint16), now=100.0)
    assert tile.value_label.text() == "2.00 %"
    stats[0] = _stats(clipped=0)
    controller._update_live_histogram(None, now=100.0 + HISTOGRAM_UPDATE_INTERVAL_SEC / 2)   # throttled
    assert tile.value_label.text() == "2.00 %"
    controller._update_live_histogram(None, now=101.0)
    assert tile.value_label.text() == "0.00 %"
    tile.peak_reset.connect(controller._reset_saturation_peak)
    tile.reset_peak()
    assert resets == [True]


def test_ready_list_warns_clipped_near_and_not_checked(qt_app):
    import config as app_config
    from windows.scan_window import ScanWindow
    win = ScanWindow(app_config.AppConfig())
    win.histogram.set_raw_saturation(_stats(clipped=21))
    assert "2.10 % clipped" in [t for _s, t, _a, _c in win.ready_items()[0]]
    win.histogram.set_raw_saturation(_stats(near=5))
    assert "0.50 % near full scale" in [t for _s, t, _a, _c in win.ready_items()[0]]
    win.histogram.set_raw_saturation(_stats())
    assert "No clipped pixels" in win.ready_items()[1]
    win.histogram.set_raw_saturation(_stats(status=STATUS_NOT_CHECKED, checked_frames=0))
    assert "Clipping not checked" in [t for _s, t, _a, _c in win.ready_items()[0]]
    win.close()
