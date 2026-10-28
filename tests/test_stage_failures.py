"""Stage failures must not escape QThread.run or leave scan recording armed."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import importlib
import logging
import sys
import threading
import types
from PyQt5.QtWidgets import QApplication
import pytest

ROOT=Path(__file__).resolve().parents[1]

def stage_class():
    spec=importlib.util.spec_from_file_location('stage_fix_test',ROOT/'workers/stage_thread.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.StageThread

class FakeStage:
    def __init__(self,fail=None):
        self.fail=fail;self._opened=False;self.calls=[];self._axis=SimpleNamespace(command_stop=lambda:self.calls.append('stop'))
    def open(self):
        self.calls.append('open')
        if self.fail=='open':raise RuntimeError('No stage detected')
        self._opened=True
    def close(self):self.calls.append('close');self._opened=False
    def set_speed_dps(self,value):self.calls.append(('speed',value))
    def set_position(self,*args):self.calls.append(('position',args))
    def set_degree(self,value):
        self.calls.append(('degree',value))
        if self.fail=='move':raise RuntimeError('motion failed')

@pytest.mark.parametrize('fail',['open','move'])
def test_failure_signalled_and_closed(fail):
    app=QApplication.instance() or QApplication([]);stage=FakeStage(fail);worker=stage_class()(stage_factory=lambda:stage);errors=[];worker.error.connect(errors.append);worker.run()
    assert worker.failed and not worker.completed and len(errors)==1
    assert worker._stage_ref is None and stage.calls[-1]=='close'


def test_success_close_and_prestart_cancel():
    app=QApplication.instance() or QApplication([]);stage=FakeStage();worker=stage_class()(stage_factory=lambda:stage);worker.run();assert worker.completed and not worker.failed;assert stage.calls[-1]=='close'
    stage=FakeStage();worker=stage_class()(stage_factory=lambda:stage);worker.request_stop();worker.run();assert worker.cancelled;assert stage.calls==['open','close']

def test_stop_does_not_block_on_open_lock():
    app=QApplication.instance() or QApplication([]);worker=stage_class()(stage_factory=FakeStage);worker._lifecycle_lock.acquire()
    try:worker.request_stop();assert worker._stop.is_set()
    finally:worker._lifecycle_lock.release()

def controller_methods():
    """The real ScanController class without its camera/window constructor."""
    from controllers.scan_controller import ScanController
    from modules.acquisition import ScanSession
    class Controller(ScanController):
        def __init__(self):
            self.session=ScanSession();self.reference_preview_ready=None;self.reference_preview_failed=None
    return Controller

def _capture_dialogs(monkeypatch):
    import controllers.scan_controller as module
    dialogs=[]
    monkeypatch.setattr(module.QMessageBox,'critical',lambda *a:dialogs.append(a[1:]))
    return dialogs

def test_controller_recovers_preview_and_releases_stage(monkeypatch):
    dialogs=_capture_dialogs(monkeypatch)
    c=controller_methods()();c.PREVIEW='PREVIEW';c.SCANNING='SCANNING';c.state=c.SCANNING;c.recording=True;flags=[];statuses=[]
    c.preview_thread=SimpleNamespace(set_scanning=lambda v:flags.append(v),set_keep_spectrogram=lambda v:flags.append(v),set_recording=lambda *a, **kw:None,take_recorded_frames=lambda *window, **kw:[],take_recorded_raw=lambda:[],live_recorded_frames=lambda:[]);c.win=SimpleNamespace(set_status=statuses.append);c._set_state=lambda v:setattr(c,'state',v)
    deleted=[];c.stage=SimpleNamespace(completed=False,failed=True,cancelled=False,deleteLater=lambda:deleted.append(True))
    c.on_stage_error('open failed');assert c.state==c.PREVIEW and not c.recording and flags==[False,False]
    c.on_stage_thread_finished();assert c.stage is None and deleted
    assert dialogs==[('Stage Error','Scan discarded:\nopen failed')]    # one dialog, not repeated at thread end

def test_failure_in_real_qthread():
    app=QApplication.instance() or QApplication([])
    stage=FakeStage('open')
    worker=stage_class()(stage_factory=lambda:stage)
    errors=[]
    worker.error.connect(errors.append)
    worker.start()
    assert worker.wait(1000)
    app.processEvents()
    assert worker.failed and errors==['No stage detected']
    assert worker._stage_ref is None and stage.calls[-1]=='close'


# ── v1 cube timing: cube at scan end / manual stop; return-home separate ──

class RecordingStage(FakeStage):
    """FakeStage that also logs signal emissions into the same call list."""
    def __init__(self, fail=None, block_scan=False):
        super().__init__(fail)
        self.block_scan = block_scan
        self.release = threading.Event()
        self.moving = threading.Event()
        self._axis = SimpleNamespace(command_stop=self._command_stop)
    def _command_stop(self):
        self.calls.append('stop'); self.release.set()
    def set_degree(self, value):
        self.calls.append(('degree', value))
        if self.block_scan:
            self.moving.set(); assert self.release.wait(2)
        if self.fail == 'move': raise RuntimeError('motion failed')
    def set_position(self, *args):
        self.calls.append(('position', args))
        if self.fail == 'return' and any(c == 'stageafter' for c in self.calls):
            raise RuntimeError('return failed')

def make_worker(stage, **kw):
    w = stage_class()(stage_factory=lambda: stage, **kw)
    w.degree, w.degree_per_second = 30, 2
    for name in ('stagebefore', 'stageafter', 'returned_home'):
        getattr(w, name).connect(lambda n=name: stage.calls.append(n))
    errors = []; w.error.connect(errors.append)
    return w, errors

def test_cube_signal_before_return_home():
    app = QApplication.instance() or QApplication([])
    stage = RecordingStage(); w, errors = make_worker(stage); w.run()
    assert stage.calls == ['open', ('speed', 10), ('position', (0, 0)), ('speed', 2), 'stagebefore',
                           ('degree', 30), 'stageafter', ('speed', 2), ('position', (0, 0)),
                           'returned_home', 'close']
    assert w.completed and w.scan_finished and not w.failed and not errors

def test_homing_before_a_scan_runs_at_500_steps_per_second_on_both_stages():
    # The stage may start far from home (the app was closed before it returned home).
    from modules.stage.sequence import HOMING_SPEED_STEPS_S, StageSequence
    class LinearStage:
        def __init__(self): self.calls = []
        def open(self): pass
        def close(self): pass
        def set_speed(self, v): self.calls.append(('steps/s', v))
        def set_speed_mm_s(self, v): self.calls.append(('mm/s', v))
        def set_position(self, *args): self.calls.append(('position', args))
        def move_le_motor_mm(self, v): self.calls.append(('move', v))
    linear = LinearStage()
    seq = StageSequence('linear', stage_factory=lambda: linear)
    seq.distance_mm, seq.speed_um_s = 5.0, 2000.0
    seq.run()
    assert HOMING_SPEED_STEPS_S == 500 and seq.completed
    assert linear.calls[:3] == [('steps/s', 500), ('position', (0, 0)), ('mm/s', 2.0)]   # 10 mm/s (was 0.2)
    app = QApplication.instance() or QApplication([])
    rotation = RecordingStage(); w, errors = make_worker(rotation); w.run()
    assert rotation.calls[1] == ('speed', 10) and not errors     # 500 steps/s = 10 deg/s, as before


def test_return_home_failure_keeps_scan():
    app = QApplication.instance() or QApplication([])
    stage = RecordingStage('return'); w, errors = make_worker(stage); w.run()
    assert w.scan_finished and w.failed and w.failed_phase == 'returning'
    assert stage.calls.index('stageafter') < stage.calls.index(('position', (0, 0)), 3)
    assert 'returned_home' not in stage.calls and stage.calls[-1] == 'close' and len(errors) == 1

def test_scan_move_failure_still_returns_home():
    app = QApplication.instance() or QApplication([])
    stage = RecordingStage('move'); w, errors = make_worker(stage); w.run()
    assert w.failed and w.failed_phase == 'scanning' and not w.scan_finished
    assert 'stageafter' not in stage.calls and stage.calls[-2] == ('position', (0, 0))

def test_manual_stop_mid_scan_returns_home_in_real_qthread():
    app = QApplication.instance() or QApplication([])
    stage = RecordingStage(block_scan=True); w, errors = make_worker(stage)
    w.start(); assert stage.moving.wait(2)
    w.request_stop()                       # GUI presses Stop during the scan move
    assert w.wait(2000); app.processEvents()
    assert w.cancelled and w.scan_finished and not w.failed and not errors
    after = stage.calls[stage.calls.index('stop'):]
    assert 'stageafter' in after and ('position', (0, 0)) in after and 'returned_home' in after

def test_stop_during_return_home_is_ignored():
    app = QApplication.instance() or QApplication([])
    stage = RecordingStage(); w, _ = make_worker(stage)
    stage._opened = True; w._stage_ref = stage; w.phase = w.RETURNING
    w.request_stop(); assert 'stop' not in stage.calls

def test_shutdown_abort_skips_return_home():
    app = QApplication.instance() or QApplication([])
    stage = RecordingStage(block_scan=True); w, _ = make_worker(stage)
    w.start(); assert stage.moving.wait(2)
    w.request_stop(return_home=False); assert w.wait(2000); app.processEvents()
    after = stage.calls[stage.calls.index('stop'):]
    assert ('position', (0, 0)) not in after and 'close' in after and 'returned_home' not in after

def _controller(state='SCANNING', running=True):
    c = controller_methods()(); c.PREVIEW = 'PREVIEW'; c.SCANNING = 'SCANNING'; c.state = state
    c.recording = True; c.statuses = []; c.built = []; c.stops = []
    c.preview_thread = SimpleNamespace(set_scanning=lambda v: None, set_keep_spectrogram=lambda v: None,
                                  set_recording=lambda *a, **kw: None, take_recorded_frames=lambda *window, **kw: [], take_recorded_raw=lambda: [], live_recorded_frames=lambda: [])
    c.win = SimpleNamespace(set_status=c.statuses.append)
    c._set_state = lambda v: setattr(c, 'state', v)
    def finish():
        if c.state == c.SCANNING: c.built.append(True); c.state = c.PREVIEW; c.recording = False
    c.finish_scan = finish
    c.stage = SimpleNamespace(isRunning=lambda: running, request_stop=lambda **k: c.stops.append(k),
                              failed=False, completed=False, cancelled=False, deleteLater=lambda: None)
    return c

def test_controller_builds_cube_at_scan_end():
    c = _controller(); c.on_stage_finished()
    assert c.built == [True] and c.state == 'PREVIEW' and 'returning home' in c.statuses[-1]

def test_controller_stop_builds_partial_cube_immediately():
    c = _controller(); c.stop_scan_manually()
    assert c.stops == [{}] and c.built == [True] and c.state == 'PREVIEW'
    c.on_stage_finished(); assert c.built == [True]      # late stageafter is a no-op

def test_controller_return_home_error_keeps_cube(monkeypatch):
    dialogs = _capture_dialogs(monkeypatch)
    c = _controller(state='PREVIEW'); c.on_stage_error('return failed')
    assert c.state == 'PREVIEW' and 'kept' in c.statuses[-1]
    assert dialogs == [('Stage Error', 'Scan data kept:\nreturn failed')]


# ── speed resolution of the XIMC drivers (fake libximc, no hardware) ──

class FakeAxis:
    """libximc ``Axis`` double: engine/move settings in memory; logs the speeds written
    (``(Speed, uSpeed)``) and the moves."""
    engine = dict(MicrostepMode=9, EngineFlags=0, NomSpeed=0, uNomSpeed=0)   # 1/256 microsteps
    drop_u_speed = False          # a controller that ignores uSpeed
    created = []
    def __init__(self, uri):
        self.written, self.moves = [], []
        self._engine = SimpleNamespace(**type(self).engine)
        self._move = SimpleNamespace(Speed=100, uSpeed=0, Accel=500, Decel=500)
        FakeAxis.created.append(self)
    def open_device(self): pass
    def close_device(self): pass
    def get_engine_settings(self): return copy.copy(self._engine)
    def set_engine_settings(self, engine): self._engine = copy.copy(engine)
    def get_move_settings(self): return copy.copy(self._move)
    def set_move_settings(self, move):
        self.written.append((move.Speed, move.uSpeed))
        self._move = copy.copy(move)
        if self.drop_u_speed: self._move.uSpeed = 0
    def get_position(self): return SimpleNamespace(Position=0, uPosition=0)
    def command_move(self, position, u_position): self.moves.append((position, u_position))
    def command_movr(self, position, u_position): self.moves.append(('r', position, u_position))
    def command_wait_for_stop(self, ms): pass
    def command_zero(self): pass
    def command_stop(self): pass

@pytest.fixture
def smc1(monkeypatch):
    """``modules.stage.smc1`` imported over a fake ``libximc.highlevel`` (undone afterwards)."""
    import modules.stage as stage_package
    highlevel = types.ModuleType('libximc.highlevel')
    highlevel.Axis, highlevel.ximc_version = FakeAxis, lambda: 'fake'
    highlevel.MicrostepMode = SimpleNamespace(MICROSTEP_MODE_FULL=1, MICROSTEP_MODE_FRAC_2=2, MICROSTEP_MODE_FRAC_256=9)
    highlevel.EngineFlags = SimpleNamespace(ENGINE_ACCEL_ON=0x10, ENGINE_LIMIT_RPM=0x80)
    highlevel.EnumerateFlags = SimpleNamespace(ENUMERATE_PROBE=1)
    highlevel.enumerate_devices = lambda flags: []
    package = types.ModuleType('libximc'); package.highlevel = highlevel
    monkeypatch.setitem(sys.modules, 'libximc', package)
    monkeypatch.setitem(sys.modules, 'libximc.highlevel', highlevel)
    monkeypatch.delitem(sys.modules, 'modules.stage.smc1', raising=False)
    monkeypatch.delattr(stage_package, 'smc1', raising=False)
    monkeypatch.setattr(FakeAxis, 'created', [])
    module = importlib.import_module('modules.stage.smc1')
    yield module
    sys.modules.pop('modules.stage.smc1', None)
    if getattr(stage_package, 'smc1', None) is module: delattr(stage_package, 'smc1')

def _opened(driver):
    driver = driver(device='xi-emu:///fake'); driver.open(); return driver

def test_speed_encoding_rounds_to_the_nearest_microstep_and_carries():
    from modules.stage.speed import decode_speed, encode_speed, microsteps_per_step
    assert [microsteps_per_step(mode) for mode in (1, 2, 4, 9)] == [1, 2, 8, 256]
    assert encode_speed(7.5, 256) == (7, 128) and encode_speed(7.5, 1) == (8, 0)
    assert encode_speed(7.999, 256) == (8, 0) and encode_speed(7.8, 2) == (8, 0)      # carry into Speed
    assert encode_speed(0, 256) == (0, 0) and decode_speed(7, 128, 256) == 7.5
    with pytest.raises(ValueError): encode_speed(-1, 256)

@pytest.mark.parametrize('mode, microsteps', [(9, 256), (4, 8), (2, 2)])
def test_linear_speed_keeps_the_fraction_of_a_step(smc1, monkeypatch, mode, microsteps):
    monkeypatch.setattr(FakeAxis, 'engine', {**FakeAxis.engine, 'MicrostepMode': mode})
    stage = _opened(smc1.Stage8SMC1Linear)
    assert stage.set_speed_mm_s(0.150) == pytest.approx(0.150)       # 7.5 steps/s; ran at 140 um/s
    assert stage._axis.written[-1] == (7, microsteps // 2) and stage.speed_steps_s == 7.5
    assert stage.set_speed_mm_s(0.250) == pytest.approx(0.250)       # ran at 240 um/s
    assert stage._axis.written[-1] == (12, microsteps // 2)

def test_rotation_speed_keeps_its_half_step(smc1):
    from modules.stage.speed import ROTATION_MICROSTEPS, microsteps_per_step
    stage = _opened(smc1.Stage8SMC1)
    assert microsteps_per_step(stage._axis.get_engine_settings().MicrostepMode) == ROTATION_MICROSTEPS
    assert stage.set_speed_dps(0.03) == pytest.approx(0.03)          # 1.5 steps/s; ran at 0.02 deg/s
    assert stage._axis.written[-1] == (1, 1)
    assert stage.set_speed(7.8) == 8.0 and stage._axis.written[-1] == (8, 0)   # rounds up into Speed

def test_speed_limits_are_clamped_with_a_warning(smc1, monkeypatch, caplog):
    stage = _opened(smc1.Stage8SMC1Linear)
    with caplog.at_level(logging.WARNING):
        assert stage.set_speed_mm_s(30.0) == 20.0                    # 1000 steps/s
    assert stage._axis.written[-1] == (1000, 0) and 'clamped' in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        stage.set_speed_mm_s(1e-6)                                   # below one microstep per second
    assert stage._axis.written[-1] == (0, 1) and 'raised' in caplog.text
    rotation = _opened(smc1.Stage8SMC1)
    assert rotation.set_speed_dps(0.001) == pytest.approx(0.01) and rotation._axis.written[-1] == (0, 1)
    with pytest.raises(ValueError): rotation.set_speed_dps(20.5)
    monkeypatch.setattr(FakeAxis, 'engine', {**FakeAxis.engine, 'EngineFlags': 0x80, 'NomSpeed': 400})
    limited = _opened(smc1.Stage8SMC1Linear)                         # controller holds the motor below NomSpeed
    assert limited.set_speed(500) == 400 and limited._axis.written[-1] == (400, 0)

def test_the_speed_reported_is_the_controllers_read_back(smc1, monkeypatch, caplog):
    monkeypatch.setattr(FakeAxis, 'drop_u_speed', True)
    stage = _opened(smc1.Stage8SMC1Linear)
    with caplog.at_level(logging.WARNING):
        assert stage.set_speed_mm_s(0.150) == pytest.approx(0.140)
    assert stage._axis.written[-1] == (7, 128) and 'instead of' in caplog.text

def test_set_degree_rounds_to_the_nearest_step(smc1):
    stage = _opened(smc1.Stage8SMC1)
    stage.set_degree(2.3)                                            # 114.99999999999999 steps
    assert stage._axis.moves[-1] == (115, 0)

@pytest.mark.parametrize('stage_type', ['linear', 'rotation'])
def test_sequence_reports_the_speed_the_driver_runs(smc1, stage_type):
    from modules.stage.sequence import StageSequence
    driver = smc1.Stage8SMC1Linear if stage_type == 'linear' else smc1.Stage8SMC1
    seq = StageSequence(stage_type, stage_factory=lambda: driver(device='xi-emu:///fake'))
    seq.distance_mm, seq.speed_um_s, seq.degree, seq.degree_per_second = 5.0, 150.0, 30, 0.03
    seq.run()
    axis = FakeAxis.created[-1]
    assert seq.completed
    if stage_type == 'linear':
        assert seq.speed_um_s_achieved == pytest.approx(150.0) and seq.degree_per_second_achieved is None
        assert axis.written == [(500, 0), (7, 128), (7, 128)]        # homing, scan, return at scan speed
        assert axis.moves == [(0, 0), ('r', -250, 0), (0, 0)]
    else:
        assert seq.degree_per_second_achieved == pytest.approx(0.03) and seq.speed_um_s_achieved is None
        assert axis.written == [(500, 0), (1, 1), (1, 1)]
        assert axis.moves == [(0, 0), (1500, 0), (0, 0)]

def test_scan_metadata_and_clock_carry_the_speed_the_stage_runs():
    from modules.stage.sequence import StageSequence
    c = controller_methods()(); c.SCANNING = 'SCANNING'; c.state = 'SCANNING'; c.enable_cube_preview = False
    c.preview_thread = SimpleNamespace(set_recording=lambda *a, **kw: None, live_recorded_frames=lambda: [])
    clocks = []; c.win = SimpleNamespace(start_scan_clock=clocks.append)
    c._raw_metadata = {'stage_type': 'linear', 'speed_um_s': 150.0}
    c.stage = StageSequence('linear'); c.stage.speed_um_s_achieved = 140.0
    c.on_stage_started()
    assert c._raw_metadata == {'stage_type': 'linear', 'speed_um_s': 150.0, 'speed_um_s_achieved': 140.0}
    c._raw_metadata = {'stage_type': 'rotation', 'degree_per_second': 0.03}
    c.stage = StageSequence('rotation'); c.stage.degree_per_second_achieved = 0.03
    c.on_stage_started()
    assert c._raw_metadata['degree_per_second_achieved'] == 0.03
    c.stage = StageSequence('rotation'); c._raw_metadata = {}       # driver without a read-back
    c.on_stage_started()
    assert c._raw_metadata == {} and clocks == [140.0, 0.03, None]

def test_scan_window_runs_the_stage_speed_steps_and_times_the_achieved_speed(monkeypatch):
    import config as app_config
    from windows.scan_window import ScanWindow
    app = QApplication.instance() or QApplication([])
    win = ScanWindow(app_config.AppConfig())
    rotation = win.degree_per_second_spin
    assert (rotation.minimum(), rotation.maximum(), rotation.decimals()) == (0.01, 20.0, 2)
    rotation.setValue(0.033); assert rotation.value() == 0.03       # what the stage can run
    assert win.speed_um_s_spin.maximum() == 20000.0
    monkeypatch.setattr(win, 'stage_type', lambda: 'linear')
    win.distance_mm_spin.setValue(15.0); win.speed_um_s_spin.setValue(150.0)
    win.set_scanning(True); win.start_scan_clock(140.0)             # the stage reported 140 um/s
    assert win._scan_clock[1] == pytest.approx(15.0 / 0.140)
    assert win.cube_preview_group.title().endswith('15.000 mm @ 150 µm/s (stage runs 140.000 µm/s)')
    win.start_scan_clock(150.0)
    assert win.cube_preview_group.title().endswith('15.000 mm @ 150 µm/s')
    win.set_scanning(False); win.close()
