"""Profile boundaries preserve correction order, raw capture, and overrides."""
import ast
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

def methods(path, class_name, names, namespace=None):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    node = ast.ClassDef(name=class_name, bases=[], keywords=[], body=body, decorator_list=[])
    ns = namespace or {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])), 'profile_test', 'exec'), ns)
    return ns[class_name]

class Profiler:
    def __init__(self): self.events = []
    @contextmanager
    def time(self, name):
        self.events.append(('start', name))
        yield
        self.events.append(('end', name))
    def _record(self, name, value): self.events.append((name, value))

def backend():
    cls = methods('modules/camera/base.py', 'AbstractCameraBackend', {'read_frame', 'read_frame_profiled'})
    b = cls(); b._dark_enabled = True
    b._fetch_raw_frame = lambda timeout: 10
    b._apply_dark_subtract = lambda frame: frame - 2
    b._apply_hot_pixel = lambda frame: frame * 3
    return b

def test_profiled_matches_normal_and_kernel_order():
    b = backend(); p = Profiler()
    assert b.read_frame() == b.read_frame_profiled(p) == 24
    assert p.events == [('start','fetch'),('end','fetch'),('start','dark'),('end','dark'),('start','hot_pixel'),('end','hot_pixel')]

def test_disabled_and_empty_frames():
    b = backend(); b._dark_enabled = False; p = Profiler()
    assert b.read_frame_profiled(p) == 10
    assert p.events[-2:] == [('dark',0.0),('hot_pixel',0.0)]
    b._fetch_raw_frame = lambda timeout: None; p = Profiler()
    assert b.read_frame_profiled(p) is None
    assert p.events == [('start','fetch'),('end','fetch')]

def test_custom_override_preserved():
    base = type(backend())
    class Custom(base):
        def read_frame(self, timeout_ms=1000): return 99
    p = Profiler()
    assert Custom().read_frame_profiled(p) == 99
    assert p.events == [('start','fetch'),('end','fetch')]

def engine():
    from modules.acquisition import AcquisitionEngine
    return AcquisitionEngine.__new__(AcquisitionEngine)

def test_raw_dark_capture_bypasses_correction():
    worker = engine(); worker._shared_preprocessing = False; p = Profiler()
    worker.backend = SimpleNamespace(read_frame_raw=lambda timeout_ms: 123)
    assert worker._read_camera_frame(p, True) == 123
    assert p.events[-2:] == [('dark',0.0),('hot_pixel',0.0)]

def test_shared_pipeline_fetches_raw_without_double_correction():
    worker = engine(); p = Profiler(); calls = []
    worker._shared_preprocessing = True
    provider = lambda shape: None
    worker._preprocessor = SimpleNamespace(input_buffer=provider)
    worker.backend = SimpleNamespace(
        read_frame_raw_into=lambda provide, timeout_ms: calls.append(('raw', provide is provider)) or 321,
        read_frame=lambda timeout_ms: calls.append('corrected') or 654)
    assert worker._read_camera_frame(p, False) == 321
    assert calls == [('raw', True)] and p.events == [('start','fetch'),('end','fetch')]
