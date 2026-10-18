"""The sharpness scale restarts when gain, exposure or bin change (scores are not comparable)."""
import threading
import time
from types import SimpleNamespace

import pytest

from controllers.scan_controller import ScanController
from hsi_fakes import FakeCamera
from modules.acquisition import AcquisitionEngine, AcquisitionSink


class Sink(AcquisitionSink):
    def __init__(self):
        self.events = []

    def frame(self, frame, fps, timestamp): self.events.append('frame')
    def params_applied(self): self.events.append('applied')
    def params_failed(self, message): self.events.append('failed')


def wait_for(condition, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def running(tmp_path):
    camera = FakeCamera()
    config = SimpleNamespace(smile_map_folder=str(tmp_path), preprocessing_backend='cpu', m=.5, c=400., resolution=2)
    sink = Sink()
    engine = AcquisitionEngine(camera, config, sink=sink, profile=False)
    worker = threading.Thread(target=engine.run_until_stopped, daemon=True)
    worker.start()
    yield engine, camera, sink
    engine.request_stop()
    worker.join(5)


def test_engine_reports_applied_params_but_not_failed_ones(running):
    engine, camera, sink = running
    assert wait_for(lambda: 'frame' in sink.events)
    engine.request_params(2, 5.0)
    assert wait_for(lambda: 'applied' in sink.events) and camera.exposure_ms == 5.0

    def refuse(gain, exposure_ms):
        raise RuntimeError('refused')

    camera.set_params = refuse
    engine.request_params(3, 7.0)
    assert wait_for(lambda: 'failed' in sink.events)
    assert sink.events.count('applied') == 1


def test_controller_resets_the_scale_on_applied_params_and_bin():
    resets = []
    controller = ScanController.__new__(ScanController)            # no camera window: only the slots
    controller.win = SimpleNamespace(reset_sharpness_scale=lambda: resets.append(1),
                                     camera_controls=SimpleNamespace(set_selected_bin=lambda _: None))
    controller._refresh_dark_reference = lambda _: None
    controller._on_params_applied()
    controller._on_bin_applied(2)
    assert len(resets) == 2
