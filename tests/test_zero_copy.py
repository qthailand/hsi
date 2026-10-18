"""Zero-copy camera input: GenICam converts into the CuPy backend's mapped buffer, read in place."""
import contextlib
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import modules.compute_runtime as compute
from cuda_emulation import PinnedMemory, fake_cupy, fake_cv2
from hsi_fakes import FakeCamera
from modules.acquisition import AcquisitionEngine, AcquisitionSink
from modules.compute_runtime import BackendCapability, ComputeCapabilities, set_processing_backend
from modules.preprocessing import PreprocessingRuntime, cupy_cuda, opencv_cuda
from modules.preprocessing.cpu import CPUBackend
from test_camera_backends import _Node, genicam, poa  # noqa: F401  (pytest fixtures)
from test_preprocessing import Calibration, fixture
from test_preprocessing_binned import binner_for, _Silent

GPU_CAPS = ComputeCapabilities(BackendCapability(True, 1, 0, 'emulated', 'test'),
                               BackendCapability(True, 1, 0, 'emulated', 'test'))


@pytest.fixture
def emulated(monkeypatch):
    monkeypatch.setattr(cupy_cuda, 'cp', fake_cupy())
    monkeypatch.setattr(opencv_cuda, 'cv2', fake_cv2())


# ── camera ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize('fmt,sensor,value', [
    ('Mono8', 'SensorBpp8', 200), ('Mono10', 'SensorBpp10', 1000), ('Mono12', 'SensorBpp10', 4000),
    ('Mono12p', 'SensorBpp12', 4095), ('Mono16', 'SensorBpp10', 1000), ('Mono16', None, 60000),
])
def test_genicam_read_into_buffer_matches_read_frame_raw(genicam, fmt, sensor, value):
    camera = genicam()
    camera.open()
    camera._nm.n['PixelFormat']._v = fmt
    if sensor is not None:
        camera._nm.n['SensorPixelSize'] = _Node(camera._nm, 'SensorPixelSize', sensor)
    camera._refresh_info()
    camera._ia.fixed_value = value
    camera.start_stream()
    expected = camera.read_frame_raw(timeout_ms=100)
    shapes, buffer = [], np.full(expected.shape, 7, np.uint16)

    def provide(shape):
        shapes.append(shape)
        return buffer

    frame = camera.read_frame_raw_into(provide, timeout_ms=100)
    assert frame is buffer and shapes == [expected.shape]
    np.testing.assert_array_equal(frame, expected)
    fresh = camera.read_frame_raw_into(lambda shape: None, timeout_ms=100)     # no buffer: a new array
    assert fresh is not buffer and fresh.dtype == np.uint16
    np.testing.assert_array_equal(fresh, expected)
    camera.close()


def test_default_read_into_ignores_the_buffer(poa):
    camera = poa()
    camera.open()
    camera.start_stream()
    frame = camera.read_frame_raw_into(lambda shape: pytest.fail('POA has no zero-copy path'), timeout_ms=100)
    assert frame is not None and frame.dtype == np.uint16
    camera.close()


# ── CuPy backend ────────────────────────────────────────────────────────

@pytest.mark.parametrize('dark_enabled', [True, False])
@pytest.mark.parametrize('interp', [None, 'cubic'])
def test_zero_copy_input_matches_cpu_and_is_not_modified(emulated, dark_enabled, interp):
    frame, processor = fixture()
    model = Calibration(frame.shape, interp) if interp else None
    dark = processor._dark_mean if dark_enabled else None           # without dark: hot pixels from the input
    gpu = cupy_cuda.CuPyCUDABackend(0)
    before = PinnedMemory.allocations
    buffer = gpu.input_buffer(frame.shape)
    assert gpu.input_buffer(frame.shape) is buffer and PinnedMemory.allocations == before + 1   # reused
    buffer[...] = frame
    binner = binner_for(frame.shape[1])
    binned, output, grabbed = gpu.process_binned(buffer, dark, processor, True, True, model, True,
                                                 binner, True, _Silent())
    np.testing.assert_array_equal(buffer, frame)                    # the caller's frame is untouched
    reference, reference_grab = CPUBackend().process(frame, dark, processor, True, True, model, True, _Silent())
    np.testing.assert_array_equal(output, reference)
    np.testing.assert_array_equal(binned, binner.transform(reference[None])[0])
    np.testing.assert_array_equal(grabbed, reference_grab)


def test_zero_copy_skips_the_upload(emulated, monkeypatch):
    frame, processor = fixture()
    gpu = cupy_cuda.CuPyCUDABackend(0)
    binner = binner_for(frame.shape[1])
    rest = (processor._dark_mean, processor, True, False, None, False)
    gpu.process_binned(frame, *rest, binner, False, _Silent())       # allocates the device buffers
    uploads = []
    monkeypatch.setattr(type(gpu._input), 'set', lambda self, array: uploads.append(array.shape))
    buffer = gpu.input_buffer(frame.shape)
    buffer[...] = frame
    gpu.process_binned(buffer, *rest, binner, False, _Silent())
    assert uploads == []                                             # read in place
    gpu.process_binned(frame, *rest, binner, False, _Silent())
    assert uploads == [frame.shape]                                  # an ordinary array is uploaded


def test_new_shape_gets_a_new_buffer(emulated):
    gpu = cupy_cuda.CuPyCUDABackend(0)
    first = gpu.input_buffer((4, 6))
    second = gpu.input_buffer((2, 3))
    assert second.shape == (2, 3) and gpu.input_buffer((2, 3)) is second
    assert first.shape == (4, 6)                                     # the old view stays valid (ref-counted)


# ── runtime ─────────────────────────────────────────────────────────────

def test_runtime_offers_zero_copy_only_on_cupy(emulated):
    frame, processor = fixture()
    for requested, offered in (('cupy_cuda', True), ('opencv_cuda', False), ('cpu', False)):
        runtime = PreprocessingRuntime(requested, GPU_CAPS)
        assert runtime.input_buffer(frame.shape) is None             # nothing before selection
        runtime.process_binned(frame, processor, binner_for(frame.shape[1]))
        assert (runtime.input_buffer(frame.shape) is not None) == offered, requested


def test_broken_zero_copy_is_disabled_but_cupy_is_kept(emulated, monkeypatch):
    real = cupy_cuda.CuPyCUDABackend.input_buffer

    def not_aliased(self, shape):                                     # the GPU would read other memory
        host = real(self, shape)
        if host is not None:
            self._mapped_device = self._mapped_device.copy()
        return host

    monkeypatch.setattr(cupy_cuda.CuPyCUDABackend, 'input_buffer', not_aliased)
    frame, processor = fixture()
    runtime = PreprocessingRuntime('cupy_cuda', GPU_CAPS)
    runtime.process_binned(frame, processor, binner_for(frame.shape[1]))
    assert runtime.backend_name == 'cupy_cuda'
    assert runtime.input_buffer(frame.shape) is None                  # zero-copy switched off


# ── engine ──────────────────────────────────────────────────────────────

class _ZeroCopyCamera(FakeCamera):
    """Fake camera that converts into the provided buffer, as GenICam does."""

    def __init__(self):
        super().__init__()
        self.provided = 0

    def read_frame_raw_into(self, provide, timeout_ms=1000):
        frame = self.read_frame_raw(timeout_ms)
        if frame is None:
            return None
        buffer = provide(frame.shape)
        if buffer is None:
            return frame
        self.provided += 1
        buffer[...] = frame
        return buffer


class _Sink(AcquisitionSink):
    def __init__(self):
        self.frames, self.errors = [], []

    def frame(self, frame, fps, timestamp):
        self.frames.append(frame)

    def error(self, message):
        self.errors.append(message)


def test_engine_runs_zero_copy_end_to_end(emulated, monkeypatch, tmp_path):
    camera = _ZeroCopyCamera()
    hot = np.zeros((camera.info.height, camera.info.width), np.uint8)
    hot[2, 3] = 255
    camera.set_dark_reference(np.full(hot.shape, 3, np.uint16), hot)
    monkeypatch.setattr(compute, '_capabilities', GPU_CAPS)
    config = SimpleNamespace(smile_map_folder=str(tmp_path), preprocessing_backend='cupy_cuda',
                             m=.5, c=400., resolution=2)
    sink = _Sink()
    engine = AcquisitionEngine(camera, config, sink=sink, profile=False)
    thread = threading.Thread(target=engine.run_until_stopped, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while camera.provided < 20 and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        engine.request_stop()
        thread.join(5)
        set_processing_backend('cpu')
    assert not sink.errors and camera.provided >= 20 and sink.frames
    assert engine._preprocessor.backend_name == 'cupy_cuda', engine._preprocessor.fallback_reason


# ── fetch split: cam_wait / convert ─────────────────────────────────────

def test_genicam_records_wait_and_convert_times(genicam):
    camera = genicam()
    camera.open()
    camera.start_stream()
    assert camera.read_frame_raw_into(lambda shape: None, timeout_ms=100) is not None
    wait_ms, convert_ms = camera.last_fetch_timing
    assert wait_ms >= 0 and convert_ms >= 0
    camera._ia.fetch = lambda timeout=0: (_ for _ in ()).throw(TimeoutError())   # no frame
    assert camera.read_frame_raw_into(lambda shape: None, timeout_ms=100) is None
    assert camera.last_fetch_timing is None
    camera.close()


def test_engine_records_cam_wait_and_convert():
    class Profiler:
        def __init__(self): self.records = []
        def time(self, name): return contextlib.nullcontext()
        def _record(self, name, ms): self.records.append((name, ms))
    worker = AcquisitionEngine.__new__(AcquisitionEngine)
    worker._shared_preprocessing = True
    worker._preprocessor = SimpleNamespace(input_buffer=lambda shape: None)
    worker.backend = SimpleNamespace(read_frame_raw_into=lambda provide, timeout_ms: 'frame',
                                     last_fetch_timing=(8.5, 0.4))
    profiler = Profiler()
    assert worker._read_camera_frame(profiler, False) == 'frame'
    assert profiler.records == [('cam_wait', 8.5), ('convert', 0.4)]
    worker.backend = SimpleNamespace(read_frame_raw_into=lambda provide, timeout_ms: 'frame')   # POA/XIMEA
    profiler = Profiler()
    worker._read_camera_frame(profiler, False)
    assert profiler.records == []
