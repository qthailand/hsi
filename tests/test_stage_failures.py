"""Stage failures must not escape QThread.run or leave scan recording armed."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import ast
import threading
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
    c.preview_thread=SimpleNamespace(set_scanning=lambda v:flags.append(v),set_keep_spectrogram=lambda v:flags.append(v),set_recording=lambda v:None,take_recorded_frames=lambda *window:[],take_recorded_raw=lambda:[]);c.win=SimpleNamespace(set_status=statuses.append);c._set_state=lambda v:setattr(c,'state',v)
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
                                  set_recording=lambda v: None, take_recorded_frames=lambda *window: [], take_recorded_raw=lambda: [])
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
