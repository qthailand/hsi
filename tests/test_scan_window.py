"""Scan trimmed to the stage motion by camera timestamps, and lost-frame counting."""
import inspect
import logging
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PyQt5.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules.acquisition import AcquisitionEngine
from modules.acquisition.profiler import StageProfiler
from modules.acquisition.recorder import ScanRecorder
from modules.cube_preview import RealtimeCubePreviewWidget
from modules.cube_preview.render import CubeRenderWorker
from modules.stage import smc1
from modules.stage.sequence import StageSequence
from test_camera_backends import _Node, genicam  # noqa: F401  (pytest fixture)
from test_reference_roi import controller


# ── recorder ────────────────────────────────────────────────────────────

def recorded(stamps, keep_raw=True):
    recorder = ScanRecorder()
    recorder.set_recording(True)
    for i, stamp in enumerate(stamps):
        recorder.offer(np.full((2, 3), i, np.uint16), np.full((2, 6), i, np.uint16), keep_raw, stamp)
    return recorder


def test_take_frames_drops_frames_outside_the_motion_window():
    recorder = recorded([100, 200, 300, 400, 500])
    frames = recorder.take_frames((200, 400))
    assert [int(f[0, 0]) for f in frames] == [1, 2, 3]
    assert [int(r[0, 0]) for r in recorder.take_raw()] == [1, 2, 3]       # raw stays aligned
    assert recorder.last_trim == (1, 1) and not recorder.enabled


def test_open_bounds_and_unknown_stamps_keep_frames():
    recorder = recorded([100, 200, 300])
    assert len(recorder.take_frames((150, None))) == 2                     # manual stop: no end time
    recorder = recorded([100, None, 300])                                  # a camera without timestamps
    assert len(recorder.take_frames((150, 250))) == 3 and recorder.last_trim == (0, 0)
    recorder = recorded([100, 200, 300])
    assert len(recorder.take_frames()) == 3                                # no window: as before


def test_without_raw_frames_only_binned_are_trimmed():
    recorder = recorded([100, 200, 300], keep_raw=False)
    assert len(recorder.take_frames((150, 250))) == 1 and recorder.take_raw() == []


# ── engine ──────────────────────────────────────────────────────────────

def bare_engine(offset_ns):
    engine = AcquisitionEngine.__new__(AcquisitionEngine)
    engine._lock = threading.Lock()
    engine.recorder = ScanRecorder()
    engine._clock_offset_ns = None
    engine._armed_ns = None
    engine._record_start_ns = None
    engine._clock_sync_pending = False
    engine._last_frame_id = None
    engine.backend = SimpleNamespace(camera_clock_offset_ns=lambda: offset_ns)
    return engine


def test_arming_latches_the_clock_and_take_converts_host_times(caplog):
    engine = bare_engine(offset_ns=1_000)                   # camera clock = host clock + 1000 ns
    engine.set_recording(True)
    engine._apply_pending_clock_sync()                      # next loop iteration (worker)
    assert engine._clock_offset_ns == 1_000
    for i, stamp in enumerate((1_100, 1_500, 2_500, 3_200)):
        engine.recorder.offer(np.full((1, 1), i, np.uint16), np.zeros((1, 1), np.uint16), False, stamp)
    engine._armed_ns = 100 - 3_000_000                      # armed 3 ms before the move
    with caplog.at_level(logging.INFO, logger='modules.acquisition.engine'):
        frames = engine.take_recorded_frames(motion_start_ns=200, motion_end_ns=2_000,   # host time
                                             reference_ns=100)
    assert [int(f[0, 0]) for f in frames] == [1, 2]         # camera window 1200..3000
    assert trim_line(caplog) == ('Scan trim: dropped 1 before (0 while recording), 1 after; 4 recorded, 2 kept; armed -3 ms, '
                                 'frames +0..+0 ms, keep +0..+0 ms (from stage move start)')


def trim_line(caplog):
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith('Scan trim:')]
    assert len(lines) == 1
    return lines[0]


def test_trim_log_gives_frame_times_from_the_move_start(caplog):
    engine = bare_engine(offset_ns=5_000_000_000)
    engine.set_recording(True)
    engine._apply_pending_clock_sync()
    engine._armed_ns = 1_250_000_000                        # GUI armed 250 ms after the move
    for stamp in (6_260_000_000, 6_400_000_000, 9_000_000_000):
        engine.recorder.offer(np.zeros((1, 1), np.uint16), np.zeros((1, 1), np.uint16), False, stamp)
    with caplog.at_level(logging.INFO, logger='modules.acquisition.engine'):
        engine.take_recorded_frames(1_200_000_000, 3_500_000_000, reference_ns=1_000_000_000)
    assert trim_line(caplog) == ('Scan trim: dropped 0 before (0 while recording), 1 after; 3 recorded, 2 kept; armed +250 ms, '
                                 'frames +260..+3000 ms, keep +200..+2500 ms (from stage move start)')


def test_trim_log_says_why_nothing_was_trimmed(caplog):
    engine = bare_engine(offset_ns=0)
    engine.set_recording(True)
    engine._apply_pending_clock_sync()
    for stamp in (5, None):
        engine.recorder.offer(np.zeros((1, 1), np.uint16), np.zeros((1, 1), np.uint16), False, stamp)
    with caplog.at_level(logging.INFO, logger='modules.acquisition.engine'):
        assert len(engine.take_recorded_frames(motion_start_ns=10, motion_end_ns=20)) == 2
    assert trim_line(caplog).startswith('Scan trim: not trimmed: 1 frames without camera time; 2 recorded, 2 kept')


def test_no_latch_means_no_trim(caplog):
    engine = bare_engine(offset_ns=None)
    engine.set_recording(True)
    engine._apply_pending_clock_sync()
    engine.recorder.offer(np.zeros((1, 1), np.uint16), np.zeros((1, 1), np.uint16), False, 5)
    with caplog.at_level(logging.INFO, logger='modules.acquisition.engine'):
        assert len(engine.take_recorded_frames(motion_start_ns=10, motion_end_ns=20)) == 1
    assert trim_line(caplog).startswith('Scan trim: not trimmed: camera clock not latched; 1 recorded')


def test_lost_frames_counted_from_frame_id_gaps():
    engine = bare_engine(offset_ns=0)
    profiler = SimpleNamespace(counts=[], add_count=lambda name, n: profiler.counts.append((name, n)))
    stamps = []
    for frame_id in (5, 6, 9, 10, 1, 2):                    # 2 lost, then IDs restart (new stream)
        engine.backend.last_frame_info = (frame_id * 100, frame_id)
        stamps.append(engine._note_frame_info(profiler))
    assert stamps == [500, 600, 900, 1000, 100, 200]
    assert [n for _, n in profiler.counts] == [0, 0, 2, 0, 0, 0]


def test_profiler_shows_counts_in_the_header(caplog):
    prof = StageProfiler(log_every=2)
    with caplog.at_level(logging.INFO, logger='modules.acquisition.profiler'):
        for lost in (1, 2):
            prof._record('fetch', 1.0)
            prof.add_count('lost', lost)
            prof.tick()
    assert ' lost=3]' in caplog.records[0].getMessage()


# ── camera ──────────────────────────────────────────────────────────────

class _StampedBuffer:
    def __init__(self, inner, timestamp, frame_id):
        self.inner, self.timestamp, self.frame_id = inner, timestamp, frame_id
        self.payload, self.is_incomplete = inner.payload, False

    def __enter__(self): return self
    def __exit__(self, *exc): return False


def test_genicam_reports_frame_time_id_and_clock_offset(genicam):
    camera = genicam()
    camera.open()
    camera._tick_hz = 1e9
    camera.start_stream()
    fetch = camera._ia.fetch
    camera._ia.fetch = lambda timeout=0: _StampedBuffer(fetch(timeout), 123_456_789, 42)
    assert camera.read_frame_raw(timeout_ms=100) is not None
    assert camera.last_frame_info == (123_456_789, 42)
    camera._tick_hz = 125e6                                    # 8 ns ticks
    camera.read_frame_raw(timeout_ms=100)
    assert camera.last_frame_info == (987_654_312, 42)
    assert camera.camera_clock_offset_ns() is None             # fake camera: no latch node
    camera._nm.n['GevTimestampControlLatch'] = _Node(camera._nm, 'GevTimestampControlLatch', None)
    camera._nm.n['GevTimestampValue'] = _Node(camera._nm, 'GevTimestampValue', 10_000_000_000)
    camera._tick_hz = 1e9
    offset = camera.camera_clock_offset_ns()
    assert abs((offset + time.perf_counter_ns()) - 10_000_000_000) < 50_000_000
    camera.close()


# ── stage ───────────────────────────────────────────────────────────────

class _Stage:
    _opened = False
    def __init__(self): self.calls = []
    def open(self): self._opened = True
    def close(self): self._opened = False
    def set_speed_dps(self, value): pass
    def set_position(self, *args): pass
    def set_degree(self, value): self.calls.append('move')


def test_sequence_records_the_motion_window():
    stage = _Stage()
    sequence = StageSequence('rotation', stage_factory=lambda: stage)
    seen = []
    sequence._notify_before = lambda: seen.append(('before', sequence.motion_start_ns))
    sequence._notify_after = lambda: seen.append(('after', sequence.motion_start_ns, sequence.motion_end_ns))
    sequence.run()
    _, noted = seen[0]                                      # already set when on_before runs
    _, start, end = seen[1]
    assert noted is not None and start == noted and end is not None and start <= end


def test_stop_before_the_move_leaves_no_motion_start():
    stage = _Stage()
    sequence = StageSequence('rotation', stage_factory=lambda: stage)
    sequence._notify_before = lambda: sequence._stop.set()
    sequence.run()
    assert sequence.motion_start_ns is None and 'move' not in stage.calls


def test_sequence_reads_the_ramps_after_setting_the_scan_speed():
    stage = _Stage()
    stage.ramp_times_s = lambda: (0.25, 0.5)
    sequence = StageSequence('rotation', stage_factory=lambda: stage)
    sequence.run()
    assert sequence.ramp_s == (0.25, 0.5)
    sequence = StageSequence('rotation', stage_factory=_Stage)        # driver without ramps
    sequence.run()
    assert sequence.ramp_s == (0.0, 0.0)


def test_steady_window_cuts_the_ramps_and_the_extra_time():
    sequence = StageSequence('rotation')
    assert sequence.steady_window() == (None, None)                    # not moved yet
    sequence.motion_start_ns, sequence.motion_end_ns = 1_000_000_000, 5_000_000_000
    sequence.ramp_s, sequence.trim_extra_ms = (0.5, 0.25), 10.0
    assert sequence.steady_window() == (1_510_000_000, 4_740_000_000)
    sequence.motion_end_ns = None                                      # manual stop
    assert sequence.steady_window() == (1_510_000_000, None)
    sequence.motion_end_ns = 1_600_000_000                             # never at constant speed
    assert sequence.steady_window() == (1_000_000_000, 1_600_000_000)


def test_ramp_times_from_the_controller_move_settings():
    engine = SimpleNamespace(EngineFlags=smc1.ximc.EngineFlags.ENGINE_ACCEL_ON,
                             MicrostepMode=smc1.ximc.MicrostepMode.MICROSTEP_MODE_FRAC_2)
    axis = SimpleNamespace(get_engine_settings=lambda: engine)
    move = SimpleNamespace(Speed=50, uSpeed=1, Accel=101, Decel=202)   # 50.5 steps/s
    assert smc1._ramp_times_s(axis, move) == (0.5, 0.25)
    move.Decel = 0
    assert smc1._ramp_times_s(axis, move) == (0.5, 0.0)
    engine.EngineFlags = 0                                             # acceleration off
    assert smc1._ramp_times_s(axis, move) == (0.0, 0.0)


def test_stage_polls_its_stop_status_every_10_ms():
    for cls in (smc1.Stage8SMC1, smc1.Stage8SMC1Linear):
        for name in ('set_position', 'set_position_r'):
            assert inspect.signature(getattr(cls, name)).parameters['wait_timeout_ms'].default == 10
    for name in ('move_to_mm', 'move_by_mm', 'move_to_motor_mm', 'move_le_motor_mm'):
        assert inspect.signature(getattr(smc1.Stage8SMC1Linear, name)).parameters['wait_timeout_ms'].default == 10


def test_finish_scan_passes_the_constant_speed_window():
    c = controller()
    taken = []
    c.preview_thread.take_recorded_frames = lambda *window, reference_ns: taken.append(window + (reference_ns,)) or []
    c.state = c.SCANNING
    c.stage = StageSequence('rotation')
    c.stage.motion_start_ns, c.stage.motion_end_ns = 1_000, 9_000
    c.stage.ramp_s, c.stage.trim_extra_ms = (1e-6, 2e-6), 0.001        # 1000 / 2000 / 1000 ns
    c.finish_scan()
    assert taken == [(3_000, 6_000, 1_000)]


def test_stage_start_arms_recording_from_the_constant_speed_start():
    c = controller()
    armed = []
    c.preview_thread.set_recording = lambda active, start_ns=None: armed.append((active, start_ns))
    c.state = c.SCANNING
    c.stage = StageSequence('rotation')
    c.stage.motion_start_ns, c.stage.ramp_s, c.stage.trim_extra_ms = 1_000, (2e-6, 0.0), 0.001
    c.on_stage_started()
    assert armed == [(True, 4_000)]                                    # 1000 + 2000 + 1000 ns


# ── recording-time drop and the shared preview list ─────────────────────

def test_recorder_refuses_frames_before_the_min_stamp():
    recorder = recorded([100, 200])
    recorder.set_min_stamp(250)
    for stamp in (240, None, 300):
        recorder.offer(np.zeros((2, 3), np.uint16), np.zeros((2, 6), np.uint16), False, stamp)
    assert recorder.count() == 4 and recorder.last_live_dropped == 0   # 100, 200 stay; 240 refused
    assert len(recorder.live_frames()) == 4                            # refused frames never reach the preview
    recorder.take_frames()
    assert recorder.last_live_dropped == 1
    recorder.set_recording(True)                                       # re-arming clears the bound
    recorder.offer(np.zeros((2, 3), np.uint16), np.zeros((2, 6), np.uint16), False, 1)
    assert recorder.count() == 1


def test_live_list_is_the_cube_list_unless_trimmed():
    recorder = recorded([100, 200, 300])
    live = recorder.live_frames()
    assert recorder.take_frames((None, None)) is live                  # nothing dropped: same list object
    recorder = recorded([100, 200, 300])
    live = recorder.live_frames()
    kept = recorder.take_frames((150, None))
    assert kept is not live and len(kept) == 2 and all(a is b for a, b in zip(kept, live[1:]))


def test_latch_sets_the_recording_bound_in_camera_time(caplog):
    engine = bare_engine(offset_ns=1_000)
    engine.set_recording(True, start_ns=500)
    engine._apply_pending_clock_sync()
    for i, stamp in enumerate((1_400, 1_600)):                         # camera bound 1500
        engine.recorder.offer(np.full((1, 1), i, np.uint16), np.zeros((1, 1), np.uint16), False, stamp)
    assert len(engine.live_recorded_frames()) == 1
    with caplog.at_level(logging.INFO, logger='modules.acquisition.engine'):
        frames = engine.take_recorded_frames(500, None)
    assert [int(f[0, 0]) for f in frames] == [1]
    assert trim_line(caplog).startswith('Scan trim: dropped 1 before (1 while recording), 0 after; 1 recorded, 1 kept')


def render_worker():
    worker = CubeRenderWorker()                                        # not started: _tick is driven here
    widths = []
    worker.rgb_ready.connect(lambda rgb, h, w: widths.append(w))
    return worker, widths


def frame(i):
    return np.full((4, 3), i, np.uint16)


def test_render_worker_reads_a_growing_shared_list():
    worker, widths = render_worker()
    shared = [frame(0), frame(1)]
    worker.set_frames(shared)
    worker._tick()
    shared.append(frame(2))                                            # the recorder appends
    worker._tick()
    assert widths == [2, 3] and worker._frames is shared


def test_render_worker_keeps_columns_when_the_list_is_cut_at_the_end():
    worker, widths = render_worker()
    live = [frame(i) for i in range(5)]
    worker.set_frames(live)
    worker._tick()
    cache = worker._raw
    worker.set_frames(live[:3])                                        # trimmed cube list, same frames
    worker._tick()
    assert widths == [5, 3] and worker._raw is cache and worker._filled == 3
    assert worker._vmax == 2.0                                         # normalised over what is left


def test_render_worker_redraws_when_the_start_was_cut():
    worker, widths = render_worker()
    live = [frame(i) for i in range(5)]
    worker.set_frames(live)
    worker._tick()
    worker.set_frames(live[2:])
    worker._tick()
    assert widths == [5, 3] and worker._vmin == 2.0                    # rendered again from the new first frame


def test_preview_widget_keeps_points_and_shifts_them_with_a_cut_start():
    app = QApplication.instance() or QApplication([])
    widget = RealtimeCubePreviewWidget()
    try:
        live = []
        widget.set_frame_source(live, np.array([500.0, 600.0, 700.0]))
        assert widget._bands == 0                                      # nothing recorded yet
        live.extend(frame(i) for i in range(6))
        widget._on_frames_count(len(live))                             # first tick with frames
        assert widget._bands == 3
        widget._rgb_band_indices = [2, 1, 0]                           # user moved the R/G/B lines
        for x in (1, 4):
            widget.image_label.add_click_point(x, 2)
            widget._spectrum_history.append((x, 2, np.zeros(3, np.float32)))
        cube = live[2:]
        widget.set_frame_source(cube, shift=2)                         # cube list, 2 cut off the start
        assert widget.image_label.click_points() == [(2, 2)]           # x=1 left the image
        assert [h[:2] for h in widget._spectrum_history] == [(2, 2)]
        assert widget._rgb_band_indices == [2, 1, 0] and widget._worker._frames is cube
    finally:
        widget.cleanup()
        app.processEvents()


def test_controller_points_the_preview_at_the_recording_then_the_cube():
    c = controller()
    c.enable_cube_preview, c.current_wavelengths = True, None
    shown = []
    c.win.cube_preview = SimpleNamespace(set_frame_source=lambda frames, wl, shift: shown.append((frames, shift)),
                                         set_saturation_mask=lambda mask: None)
    live = [frame(i) for i in range(6)]
    c.preview_thread.live_recorded_frames = lambda: live
    c.state = c.SCANNING
    c.on_stage_started()
    kept = live[2:5]
    c.preview_thread.recorder = SimpleNamespace(last_trim=(2, 1))
    c.preview_thread.take_recorded_frames = lambda *window, **kw: kept
    c.finish_scan()
    assert shown[0][0] is live and shown[0][1] == 0
    assert shown[1][0] is kept and shown[1][1] == 2


# ── settings pane during a scan ─────────────────────────────────────────

def test_a_scan_locks_the_settings_in_view_and_the_end_unlocks_them():
    import config as app_config
    from windows.scan_window import ScanWindow
    app = QApplication.instance() or QApplication([])
    win = ScanWindow(app_config.AppConfig())
    win.resize(1500, 800); win.show(); app.processEvents()
    controls = win.camera_controls
    controls.set_cooling_supported(True, (-20.0, 20.0))         # a TEC target is a setting
    win.set_scanning(True); win.set_scanning(True)              # twice: no harm
    assert win.settings_scroll.isVisible() and controls.isVisible()   # still readable in their column
    assert not any(w.isEnabled() for w in (controls.gain_spin, controls.exp_spin, controls.temp_spin,
                                           controls.keep_spectrogram_check, win.rotation_group, win.linear_group))
    win.set_scanning(False)
    assert controls.temp_spin.isEnabled() and win.rotation_group.isEnabled()
    win.set_scanning(False)                                     # no scan to end: unchanged
    assert controls.gain_spin.isEnabled()
    controls.set_cooling_supported(False)
    win.set_scanning(True)
    assert controls.temp_spin.isEnabled() and not controls.gain_spin.isEnabled()   # a sensor readout stays readable
    win.set_scanning(False)
    win.close()


def test_three_columns_live_view_cube_preview_settings():
    """Live view | cube preview (the one that grows) | settings with Start / Stop pinned below."""
    import config as app_config
    from windows.scan_window import INSPECTOR_MIN_WIDTH, ScanWindow
    app = QApplication.instance() or QApplication([])
    win = ScanWindow(app_config.AppConfig())
    win.resize(1500, 800); win.show(); app.processEvents()
    columns = win.columns_splitter
    assert [columns.widget(i) for i in range(3)][1:] == [win.cube_preview_group, win.inspector]
    left, centre, right = columns.sizes()
    assert centre > left and centre > right and right >= INSPECTOR_MIN_WIDTH
    win.resize(1900, 800); app.processEvents()
    assert columns.sizes()[1] > centre and columns.sizes()[2] == right           # only the preview grows
    start, stop = win.start_scan_btn, win.stop_scan_btn
    bottom = win.inspector.mapTo(win, win.inspector.rect().bottomLeft()).y()
    assert start.isVisible() and not stop.isVisible() and start.mapTo(win, start.rect().bottomLeft()).y() <= bottom
    assert win.viewer.start_btn is start and win.viewer.stop_btn is stop   # the image has no buttons of its own
    win.close()


def test_the_pinned_start_and_stop_drive_the_scan_and_follow_its_state():
    import config as app_config
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from windows.scan_window import ScanWindow
    app = QApplication.instance() or QApplication([])
    win = ScanWindow(app_config.AppConfig())
    win.show(); app.processEvents()
    asked = []
    win.viewer.start_scan_requested.connect(lambda: asked.append("start"))
    win.viewer.stop_scan_requested.connect(lambda: asked.append("stop"))
    win.start_scan_btn.click()
    win.viewer.set_scanning_ui(True)                           # the controller: the scan runs
    assert win.stop_scan_btn.isVisible() and win.stop_scan_btn.isEnabled() and not win.start_scan_btn.isVisible()
    win.stop_scan_btn.click()
    win.viewer.set_scanning_ui(False)
    assert win.start_scan_btn.isVisible() and win.start_scan_btn.isEnabled()
    win.activateWindow(); win.setFocus(); app.processEvents()
    QTest.keyClick(win, Qt.Key_F5)                              # F5 still starts it
    assert asked[:2] == ["start", "stop"] and asked[2:] in ([], ["start"])
    win.close()


def test_the_status_line_shows_the_scan_time_while_the_stage_moves(monkeypatch):
    import config as app_config
    import windows.scan_window as scan_window_module
    app = QApplication.instance() or QApplication([])
    win = scan_window_module.ScanWindow(app_config.AppConfig())
    now = [100.0]
    monkeypatch.setattr(scan_window_module, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(win, 'stage_type', lambda: 'rotation')
    win.degree_spin.setValue(10.0); win.degree_per_second_spin.setValue(2.0)
    assert '/ ~' not in win.status_info()
    win.set_scanning(True); win.start_scan_clock()              # the stage move starts
    now[0] = 101.5
    assert 'Scan 1.5 s / ~5.0 s' in win.status_info()
    win.set_scanning(False)                                     # the scan ended
    assert '/ ~' not in win.status_info()
    monkeypatch.setattr(win, 'stage_type', lambda: 'linear')
    win.distance_mm_spin.setValue(150.0); win.speed_um_s_spin.setValue(1000.0)
    assert win.expected_scan_seconds() == 150.0
    win.start_scan_clock(); now[0] += 61
    assert 'Scan 1:01 / ~2:30' in win.status_info()
    win.close()


def test_the_theme_makes_splitter_handles_stand_out_without_shrinking_them():
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QSplitter, QWidget
    from modules.ui_kit.theme import HANDLE, dashboard_stylesheet
    app = QApplication.instance() or QApplication([])
    base = ('QSplitter::handle { background-color: #3f4042; margin: 1px 3px; }'      # as qdarktheme dark
            'QSplitter::handle:horizontal { width: 5px; } QSplitter::handle:vertical { height: 5px; }')
    def handle(orientation, sheet):
        splitter = QSplitter(orientation)
        splitter.addWidget(QWidget()); splitter.addWidget(QWidget())
        splitter.setStyleSheet(sheet)
        splitter.resize(200, 200); splitter.show(); app.processEvents()
        h = splitter.handle(1)
        size = h.width() if orientation == Qt.Horizontal else h.height()
        centre = h.grab().toImage().pixelColor(h.width() // 2, h.height() // 2).name()
        splitter.close()
        return size, centre
    for orientation in (Qt.Horizontal, Qt.Vertical):
        base_size, base_centre = handle(orientation, base)
        size, centre = handle(orientation, dashboard_stylesheet(base))
        assert base_centre == '#3f4042' and centre == HANDLE.lower()   # the bar is lighter
        assert size == base_size                                       # same grab area


def test_a_scan_shows_its_settings_in_the_preview_title_and_the_viewer_buttons_share_a_row(monkeypatch):
    import config as app_config
    from windows.scan_window import CUBE_PREVIEW_TITLE, ScanWindow
    app = QApplication.instance() or QApplication([])
    win = ScanWindow(app_config.AppConfig())
    win.resize(1200, 800); win.show(); app.processEvents()
    spans = [(b.mapTo(win, b.rect().topLeft()).y(), b.mapTo(win, b.rect().bottomLeft()).y())
             for b in (win.open_cube_viewer_btn, win.open_calibration_btn, win.open_spectrogram_btn)]
    assert max(top for top, _ in spans) < min(bottom for _, bottom in spans)   # one row
    monkeypatch.setattr(win, 'stage_type', lambda: 'rotation')
    win.set_bin_options((1, 2), 2)
    win.camera_controls.gain_spin.setValue(3); win.camera_controls.exp_spin.setValue(1.5)
    win.degree_spin.setValue(10.0); win.degree_per_second_spin.setValue(3.5)
    win.set_wavelength_range_axis({'range_nm': (500.0, 600.0), 'bands': (5, 22), 'full_bands': 120,
                                   'centres_nm': (500.9, 599.8), 'calibrated_nm': (400.0, 700.0), 'step_nm': 6.0})
    assert win.cube_preview_group.title() == CUBE_PREVIEW_TITLE
    win.set_scanning(True)
    assert win.cube_preview_group.title() == (f'{CUBE_PREVIEW_TITLE}  —  Bin 2  ·  1.500 ms  ·  Gain 3  ·  '
                                              '17 bands 500.9–599.8 nm  ·  10.000° @ 3.500°/s')
    win.set_scanning(False)
    assert win.cube_preview_group.title() == CUBE_PREVIEW_TITLE
    monkeypatch.setattr(win, 'stage_type', lambda: 'linear')
    win.distance_mm_spin.setValue(25.0); win.speed_um_s_spin.setValue(1500.0)
    assert win.scan_settings_summary().endswith('17 bands 500.9–599.8 nm  ·  25.000 mm @ 1500 µm/s')
    win.close()


def test_the_status_bar_shows_the_temperature_with_a_thermometer_icon():
    from PyQt5.QtWidgets import QMainWindow, QMdiArea, QWidget
    import config as app_config
    import windows.main_window as main_window_module
    import windows.scan_window as scan_window_module
    app = QApplication.instance() or QApplication([])
    win = scan_window_module.ScanWindow(app_config.AppConfig())
    assert win.status_temperature() is None                     # -999: unknown
    win.set_temperature(39.14)
    assert win.status_temperature() == 39.14
    assert '\U0001f321' not in win.status_info() and '\u00b0C' not in win.status_info()

    class Shell(QMainWindow):                                   # the shell's status bar only
        _create_status_timer = main_window_module.MainWindow._create_status_timer
        _update_status_bar = main_window_module.MainWindow._update_status_bar

    host = Shell()
    host.mdi = QMdiArea()
    host.setCentralWidget(host.mdi)
    host._create_status_timer()
    host._status_timer.stop()
    assert not host.temp_status_icon.pixmap().isNull()
    sub = host.mdi.addSubWindow(win)
    host.mdi.setActiveSubWindow(sub)
    host._update_status_bar()
    assert not host.temp_status.isHidden() and host.temp_status_text.text() == '39.1\u00b0C'
    host.mdi.setActiveSubWindow(host.mdi.addSubWindow(QWidget()))  # no temperature: hidden
    host._update_status_bar()
    assert host.temp_status.isHidden()
    host.close()


# ── the settings column (inspector) ─────────────────────────────────────

def test_status_cards_are_the_correction_rows():
    import config as app_config
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest
    from windows.scan_window import ScanWindow
    app = QApplication.instance() or QApplication([])
    win = ScanWindow(app_config.AppConfig())
    win.resize(1500, 1100); win.show(); app.processEvents()
    opened, switched = [], []
    win.status_card_clicked.connect(opened.append)
    card = win.correction_status.dark
    assert win.correction_rows["dark"] is card and card.is_row() and card.isVisible()
    card.value_clicked.connect(lambda: switched.append(True))
    card.set_content("Ready", "dark_b1.npz", color="green")
    assert card.row_state.text() == "Ready" and "dark_b1.npz" in card.toolTip()
    assert card.row_chevron.isVisible()                         # no dark yet: its peek captures one
    QTest.mouseClick(card, Qt.LeftButton, pos=QPoint(20, card.height() // 2))
    assert opened == ["dark"]                                   # the row opens its view (peek)
    assert card.switch_btn.isVisible() and not card.value.isVisible()
    card.switch_btn.click()
    assert switched == [True]                                   # the owner switches, the card shows it
    win.close()


def test_ready_to_scan_lists_only_what_needs_attention():
    import config as app_config
    from windows.scan_window import ScanWindow
    app = QApplication.instance() or QApplication([])
    win = ScanWindow(app_config.AppConfig())
    captured = []
    win.dark_panel.capture_requested.connect(lambda: captured.append(True))
    win._refresh_inspector()
    items, passed = win.ready_items()
    assert [text for _s, text, _a, _c in items][:2] == ["No dark frame", "No smile model"]
    items[0][3]()                                               # its Capture fix
    assert captured == [True]
    import numpy as np
    clipped = np.zeros((10, 100), np.uint16); clipped[0, :21] = 65535      # 2.1 % at full scale
    win.histogram.update_histogram(clipped)
    win.set_ready_issue("pixel_pitch", "Pixel pitch not set", "Set", lambda: None)
    assert "2.10 % clipped" in win.ready_list.texts() and "Pixel pitch not set" in win.ready_list.texts()
    win.set_ready_issue("pixel_pitch", "")
    win.histogram.update_histogram(np.zeros((10, 100), np.uint16))
    win._refresh_inspector()
    assert "Pixel pitch not set" not in win.ready_list.texts() and "No clipped pixels" in win.ready_items()[1]
    win.close()


def test_bin_buttons_drive_the_bin_and_lock_with_it():
    import config as app_config
    from windows.scan_window import ScanWindow
    app = QApplication.instance() or QApplication([])
    win = ScanWindow(app_config.AppConfig())
    controls = win.camera_controls
    assert not hasattr(controls, "bin_combo")                 # the bar holds the bin
    win.set_bin_options((1, 2, 4), 2)
    picked = []
    controls.bin_changed.connect(lambda: picked.append(win.selected_bin()))
    buttons = controls._bin_group.buttons()
    assert [b.isChecked() for b in buttons] == [False, True, False]
    buttons[2].click()
    assert picked == [4] and win.selected_bin() == 4
    win.set_bin_enabled(False)
    assert not controls.bin_segments.isEnabled()
    win.close()


def test_stop_shows_the_scan_progress_and_time_left(monkeypatch):
    import config as app_config
    import windows.scan_window as scan_window_module
    app = QApplication.instance() or QApplication([])
    win = scan_window_module.ScanWindow(app_config.AppConfig())
    now = [100.0]
    monkeypatch.setattr(scan_window_module, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(win, 'stage_type', lambda: 'rotation')
    win.degree_spin.setValue(10.0); win.degree_per_second_spin.setValue(2.0)   # 5 s
    win.viewer.set_scanning_ui(True); win.set_scanning(True); win.start_scan_clock()
    now[0] = 102.0
    win._refresh_inspector()
    assert win.scan_progress.value() == 400 and win.stop_scan_btn.text().endswith("3.0 s left")
    win.set_scanning(False); win.viewer.set_scanning_ui(False)
    assert win.stop_scan_btn.text().strip() == "Stop scan" and win.scan_progress.isHidden()
    win.close()


def test_the_dark_lives_in_its_row_its_peek_and_ready_to_scan():
    """No Dark Frame box: the row switch subtracts, the peek captures / reloads / cancels."""
    import config as app_config
    from windows.scan_window import ScanWindow
    app = QApplication.instance() or QApplication([])
    win = ScanWindow(app_config.AppConfig())
    win.resize(1500, 1000); win.show(); app.processEvents()
    panel = win.dark_panel
    from modules.dark.state import DarkFrameState
    assert isinstance(panel, DarkFrameState) and win.correction_rows["dark"].isVisible()   # no box: a state
    asked = []
    for name in ("capture_requested", "reload_requested", "cancel_requested"):
        getattr(panel, name).connect(lambda name=name: asked.append(name))
    assert win.peek_actions("dark") == (("capture", "Capture"), ("reload", "Reload"))
    win.run_peek_action("dark", "capture"); win.run_peek_action("dark", "reload")
    panel.set_capturing(12, 50, 2, 6.0, 1)
    assert win.peek_actions("dark") == (("cancel", "Cancel capture"),)
    items, _ = win.ready_items()
    assert items[0][1] == "Capturing the dark frame…" and items[0][2] == "Cancel"
    win._refresh_inspector()
    assert win.correction_rows["dark"].row_state.text() == "12 / 50"      # the count: on the row only
    panel.update_progress(30, 50); win._refresh_inspector()
    assert win.correction_rows["dark"].row_state.text() == "30 / 50"
    items[0][3]()
    assert asked == ["capture_requested", "reload_requested", "cancel_requested"]
    panel.set_loaded(2, 6.0, 1.0, 50, 3)                         # the capture finished
    win._refresh_inspector()
    assert win.correction_rows["dark"].row_state.text() == win.correction_status.dark.value.text()
    assert win.peek_actions("dark") == (("capture", "Capture"), ("reload", "Reload"))
    win.set_scanning(True)
    assert win.peek_actions("dark") == ()                       # locked during a scan
    win.set_scanning(False)
    win.set_dark_viewable(True, "lines at 912 nm")
    texts = [text for _s, text, _a, _c in win.ready_items()[0]]
    assert "Light leak in the dark" in texts and "⚠ lines at 912 nm" in win.correction_status.dark.toolTip()
    assert win.dark_setting_text().endswith("bin 1")
    win.close()


def test_the_main_window_dark_peek_without_a_dark():
    from types import SimpleNamespace as NS
    from windows.main_window import MainWindow
    panel = NS(is_capturing=lambda: False)
    win = NS(peek_actions=lambda key: (("capture", "Capture"), ("reload", "Reload")), dark_panel=panel,
             dark_setting_text=lambda: "6 ms · gain 1 · bin 2")
    data = MainWindow._dark_peek_data(NS(win=win, dark_peek_data=lambda: None))
    assert data.title == "Dark" and data.open_label == "" and data.color == "red"
    assert data.lines[:2] == ("No dark frame", "for 6 ms · gain 1 · bin 2")
    assert data.actions == (("capture", "Capture"), ("reload", "Reload"))
    import numpy as np
    from modules.profile_peek import dark_peek
    loaded = MainWindow._dark_peek_data(NS(win=win, dark_peek_data=lambda: dark_peek({"mean": np.zeros((2, 3))})))
    assert loaded.open_label == "View ▸" and loaded.actions == data.actions


def test_live_column_cards_follow_sharpness_and_clipping():
    import config as app_config
    import numpy as np
    from windows.scan_window import ScanWindow
    app = QApplication.instance() or QApplication([])
    win = ScanWindow(app_config.AppConfig())
    win.resize(1500, 1000); win.show(); app.processEvents()
    assert not hasattr(win, "sharpness_panel")                 # the tile is the sharpness display
    assert win.sharpness_tile.isVisible() and win.clipped_tile.isVisible() and win.histogram is win.clipped_tile
    for score in (40.0, 72.4, 71.8):
        win.set_sharpness_score(score)
    assert win.sharpness_tile.value_label.text() == "71.8" and win.sharpness_tile.best() == 72.4
    win.sharpness_tile.reset_btn.click()                       # resets every sharpness display
    assert win.sharpness_tile.value_label.text() == "\u2014" and win.sharpness_tile.best() is None
    frame = np.full((40, 60), 100, np.uint16)
    frame[:, :3] = 65535
    win.histogram.update_histogram(frame)
    win._refresh_inspector()
    assert win.clipped_tile.value_label.text() == "5.00 %"
    assert win.clipped_tile.bars.saturated[-1] > 0                # the clipped bar is red
    win.set_fps(37.8)
    assert win.live_fps_label.text() == "37.8 fps"
    assert win.save_frame_btn.menu() is not None and win.save_frame_btn.text() == ""
    win.close()


def test_clipped_keeps_the_max_since_reset_and_a_scan_resets_it():
    """Now (this frame) and on the way (max since reset): Start scan resets the max."""
    import config as app_config
    import numpy as np
    from windows.scan_window import ScanWindow
    app = QApplication.instance() or QApplication([])
    win = ScanWindow(app_config.AppConfig())
    tile = win.clipped_tile
    assert tile.max_label.text() == "\u2014"
    frame = np.zeros((10, 100), np.uint16)
    frame[0, :30] = 65535
    tile.update_histogram(frame)                               # 3 % clipped
    tile.update_histogram(np.zeros((10, 100), np.uint16))      # then none
    assert tile.value_label.text() == "0.00 %" and tile.max_label.text() == "3.00 %"
    assert "#F85149" in tile.max_label.styleSheet()             # it clipped on the way: red
    assert tile.max_saturated_percent() == 3.0
    tile.reset_btn.click()
    assert tile.max_label.text() == "\u2014" and "#F85149" not in tile.max_label.styleSheet()   # grey
    tile.update_histogram(frame)
    win.set_scanning(True)                                     # Start scan: the max starts again
    assert tile.max_saturated_percent() is None
    tile.update_histogram(np.zeros((10, 100), np.uint16))
    win.set_scanning(True)                                     # already scanning: no reset
    assert tile.max_saturated_percent() == 0.0
    win.set_scanning(False)
    assert tile.max_saturated_percent() == 0.0                 # the scan's max stays readable after it
    win.close()


def test_choose_bin_acts_like_a_click_on_its_button():
    """Scripts (tools/hardware_smoke.py) switch the bin with choose_bin: it emits bin_changed."""
    import config as app_config
    from windows.scan_window import ScanWindow
    app = QApplication.instance() or QApplication([])
    win = ScanWindow(app_config.AppConfig())
    controls = win.camera_controls
    win.set_bin_options((1, 2, 4), 2)
    picked = []
    controls.bin_changed.connect(lambda: picked.append(win.selected_bin()))
    assert controls.choose_bin(4) and picked == [4] and controls._bin_group.checkedId() == 2
    assert controls.choose_bin(4) and picked == [4]            # already the bin: no change
    assert not controls.choose_bin(3) and win.selected_bin() == 4
    controls.set_selected_bin(1)                               # shows only: no bin_changed
    assert picked == [4] and win.selected_bin() == 1
    win.close()


def test_a_single_bin_button_is_rounded_on_both_sides():
    import config as app_config
    from windows.scan_window import ScanWindow
    app = QApplication.instance() or QApplication([])
    win = ScanWindow(app_config.AppConfig())
    controls = win.camera_controls
    win.set_bin_options((1,), 1)
    assert [b.property("segment") for b in controls._bin_group.buttons()] == ["only"]
    win.set_bin_options((1, 2, 4), 1)
    assert [b.property("segment") for b in controls._bin_group.buttons()] == ["first", "middle", "last"]
    assert '[segment="only"]' in win.inspector.styleSheet()
    win.close()


def test_the_wavelength_peek_has_the_cards_colour():
    """A missing or mismatched calibration (red / orange card) must not peek green."""
    from types import SimpleNamespace as NS
    import config as app_config
    from windows.main_window import MainWindow
    from windows.scan_window import ScanWindow
    app = QApplication.instance() or QApplication([])
    win = ScanWindow(app_config.AppConfig())
    card = win.status_card("wavelength")
    host = NS(scan_controller=NS(win=win), integrations=[])
    for value, color in (("Missing", "red"), ("Bin 2?", "orange"), ("Stored", "green")):
        card.set_content(value, "detail", color=color)
        data = MainWindow._profile_peek_data(host, "wavelength")
        assert data.color == color and data.lines[0] == value
    win.close()


def test_ready_list_locks_its_buttons_and_updates_text_in_place():
    from PyQt5.QtCore import QEvent
    from PyQt5.QtWidgets import QPushButton
    from windows.scan_inspector import ReadyList
    app = QApplication.instance() or QApplication([])
    ready = ReadyList()
    calls = []
    ready.set_items([("warning", "Dark subtraction is off", "Turn on", lambda: calls.append("on"))])
    buttons = ready.findChildren(QPushButton)
    assert [b.text() for b in buttons] == ["Turn on"]
    # the scan starts: same text, no callback -> the button goes (a kept one would act mid-cube)
    ready.set_items([("warning", "Dark subtraction is off", "Turn on", None)])
    app.sendPostedEvents(None, QEvent.DeferredDelete)
    assert [b.text() for b in ready.findChildren(QPushButton) if not b.isHidden()] == []
    # and comes back after the scan
    ready.set_items([("warning", "Dark subtraction is off", "Turn on", lambda: calls.append("on"))])
    app.sendPostedEvents(None, QEvent.DeferredDelete)
    # a new clipped % every tick: the rows (and the button under the cursor) stay, the text changes
    for percent in ("0.51", "0.52", "0.53"):
        ready.set_items([("warning", "Dark subtraction is off", "Turn on", lambda: calls.append("on")),
                         ("warning", f"{percent} % clipped", "", None)])
    rows = list(ready.rows)
    ready.set_items([("warning", "Dark subtraction is off", "Turn on", lambda: calls.append("on2")),
                     ("warning", "0.60 % clipped", "", None)])
    assert ready.rows == rows and ready.texts()[1] == "0.60 % clipped"
    ready.findChildren(QPushButton)[-1].click()
    assert calls == ["on2"]                                  # the latest callback runs
    ready.deleteLater()


def test_the_wavelength_peek_shows_its_value_once():
    from types import SimpleNamespace as NS
    import config as app_config
    from windows.main_window import MainWindow
    from windows.scan_window import ScanWindow
    app = QApplication.instance() or QApplication([])
    win = ScanWindow(app_config.AppConfig())
    win.status_card("wavelength").set_content("Applied", "bin 2, 224 bands", color="green")
    data = MainWindow._profile_peek_data(NS(scan_controller=NS(win=win), integrations=[]), "wavelength")
    assert data.lines == ("Applied", "bin 2, 224 bands")
    win.close()


def test_the_inspector_pauses_while_the_window_is_hidden():
    import config as app_config
    from windows.scan_window import ScanWindow
    app = QApplication.instance() or QApplication([])
    win = ScanWindow(app_config.AppConfig())
    win.show()
    app.processEvents()
    assert win._inspector_timer.isActive()
    win.hide()
    assert not win._inspector_timer.isActive()
    win.show()
    assert win._inspector_timer.isActive()
    win.close()
