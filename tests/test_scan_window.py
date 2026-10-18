"""Scan trimmed to the stage motion by camera timestamps, and lost-frame counting."""
import inspect
import logging
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules.acquisition import AcquisitionEngine
from modules.acquisition.profiler import StageProfiler
from modules.acquisition.recorder import ScanRecorder
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
    with caplog.at_level(logging.INFO, logger='modules.acquisition.engine'):
        frames = engine.take_recorded_frames(motion_start_ns=200, motion_end_ns=2_000)   # host time
    assert [int(f[0, 0]) for f in frames] == [1, 2]         # camera window 1200..3000
    assert any('1 frames before start and 1 after stop' in r.getMessage() for r in caplog.records)


def test_no_latch_means_no_trim():
    engine = bare_engine(offset_ns=None)
    engine.set_recording(True)
    engine._apply_pending_clock_sync()
    engine.recorder.offer(np.zeros((1, 1), np.uint16), np.zeros((1, 1), np.uint16), False, 5)
    assert len(engine.take_recorded_frames(motion_start_ns=10, motion_end_ns=20)) == 1


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
    assert seen[0] == ('before', None)                      # set just before the move command
    _, start, end = seen[1]
    assert start is not None and end is not None and start <= end


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
    c.preview_thread.take_recorded_frames = lambda *window: taken.append(window) or []
    c.state = c.SCANNING
    c.stage = StageSequence('rotation')
    c.stage.motion_start_ns, c.stage.motion_end_ns = 1_000, 9_000
    c.stage.ramp_s, c.stage.trim_extra_ms = (1e-6, 2e-6), 0.001        # 1000 / 2000 / 1000 ns
    c.finish_scan()
    assert taken == [(3_000, 6_000)]
