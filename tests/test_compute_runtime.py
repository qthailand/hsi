"""Detection is cached, immutable, thread-safe, and tolerates GPU failures."""
import sys
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import compute_runtime as runtime

class ComputeRuntimeTests(unittest.TestCase):
    def setUp(self):
        active_patch = patch.object(runtime, '_processing_backend', 'cpu')
        active_patch.start()
        self.addCleanup(active_patch.stop)
        self.cache_patch = patch.object(runtime, '_capabilities', None)
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)

    def test_cached_global_and_concurrent_initialization(self):
        calls = []
        def probe():
            calls.append(True)
            return runtime.BackendCapability(True, 1, 0, 'GPU', 'test')
        with patch.object(runtime, '_probe_opencv', probe), patch.object(runtime, '_probe_cupy', probe):
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _: runtime.initialize_compute(), range(16)))
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(result is results[0] for result in results))
        self.assertIs(runtime.get_compute_capabilities(), results[0])
        self.assertTrue(results[0].cuda_available)
        self.assertEqual(results[0].processing_backend, 'cpu')
        with self.assertRaises(FrozenInstanceError):
            results[0].opencv_cuda.available = False

    def test_missing_optional_dependencies(self):
        with patch.dict(sys.modules, {'cv2': None, 'cupy': None}):
            result = runtime.initialize_compute()
        self.assertFalse(result.cuda_available)
        self.assertTrue(result.opencv_cuda.reason and result.cupy_cuda.reason)

    def test_opencv_device_or_driver_unavailable(self):
        for count in (0, -1):
            cv = SimpleNamespace(cuda=SimpleNamespace(getCudaEnabledDeviceCount=lambda: count))
            with patch.dict(sys.modules, {'cv2': cv, 'numpy': SimpleNamespace()}):
                result = runtime._probe_opencv()
            self.assertFalse(result.available)
            self.assertTrue(result.reason)

    def test_device_present_but_remap_missing(self):
        cv = SimpleNamespace(cuda=SimpleNamespace(getCudaEnabledDeviceCount=lambda: 1))
        with patch.dict(sys.modules, {'cv2': cv, 'numpy': SimpleNamespace()}):
            result = runtime._probe_opencv()
        self.assertEqual(result.reason, 'OpenCV CUDA remap is unavailable')

    def test_opencv_without_cublas_is_available(self):
        # GPU binning no longer uses cv2.cuda.gemm, so an OpenCV build without cuBLAS qualifies.
        class Mat:
            def upload(self, array): self.data = array
            def download(self): return self.data
        def remap(src, x, y, **kw):
            mat = Mat(); mat.upload(src.data); return mat
        cuda = SimpleNamespace(getCudaEnabledDeviceCount=lambda: 1, getDevice=lambda: 0, setDevice=lambda d: None,
                               remap=remap, DeviceInfo=lambda d: SimpleNamespace(name=lambda: 'GPU'))
        cv = SimpleNamespace(cuda=cuda, cuda_GpuMat=Mat, INTER_LINEAR=1, BORDER_CONSTANT=0)
        with patch.dict(sys.modules, {'cv2': cv}):
            result = runtime._probe_opencv()
        self.assertTrue(result.available, result.reason)

    def test_cupy_kernel_failure_is_not_available(self):
        class Device:
            def __init__(self, index): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
        def fail(*args, **kwargs): raise RuntimeError('kernel launch failed')
        cp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 1), Device=Device),
                             uint16='uint16', arange=fail)
        with patch.dict(sys.modules, {'cupy': cp}):
            result = runtime._probe_cupy()
        self.assertFalse(result.available)
        self.assertEqual(result.device_count, 1)
        self.assertIn('kernel launch failed', result.reason)

if __name__ == '__main__':
    unittest.main()
