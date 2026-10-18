"""Wavelength calibrations stored per camera setting are used at start and after every bin change."""
import logging
import threading
import time
from types import SimpleNamespace

import pytest

from controllers.scan_controller import ScanController
from hsi_fakes import FakeCamera, H, W
from modules.acquisition import AcquisitionEngine, AcquisitionSink
from modules.configuration import ConfigStore
from modules.contracts import CalibrationStore, CameraSetting
from modules.wavelength_calibration import fit_wavelength
from windows.integrations import calibration_store


class Sink(AcquisitionSink):
    def __init__(self):
        self.frames, self.bins = [], []

    def frame(self, frame, fps, timestamp): self.frames.append(frame)
    def bin_applied(self, bin_value): self.bins.append(bin_value)


def wait_for(condition, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_engine_uses_each_bins_calibration_and_keeps_m_c_without_one(tmp_path, caplog):
    camera = FakeCamera()
    profile = fit_wavelength([0, 50], [900.0, 960.0], 2)          # bin 2: 1.2 nm/px, base 0.6

    def lookup():
        if camera.current_bin == 2:
            return profile
        raise LookupError('no file')

    config = SimpleNamespace(smile_map_folder=str(tmp_path), preprocessing_backend='cpu', m=.5, c=400., resolution=2)
    sink = Sink()
    engine = AcquisitionEngine(camera, config, sink=sink, profile=False)
    engine.set_wavelength_lookup(lookup)
    caplog.set_level(logging.WARNING, logger='modules.acquisition.engine')
    worker = threading.Thread(target=engine.run_until_stopped, daemon=True)
    worker.start()
    try:
        assert wait_for(lambda: sink.frames)
        assert (config.m, config.c) == (.5, 400.)                 # bin 1 has no file: config kept
        assert 'Bin 1: no usable wavelength calibration (no file)' in caplog.text
        engine.request_bin(2)
        assert wait_for(lambda: sink.bins == [2])
        assert config.m == pytest.approx(profile.base_m) and config.c == pytest.approx(profile.c)
        seen = len(sink.frames)
        assert wait_for(lambda: len(sink.frames) > seen + 1)
        assert engine.binner_lambda_out[0] == pytest.approx(profile.c + 1.0, abs=0.05)   # refitted, 2 nm bins
        engine.request_bin(1)
        assert wait_for(lambda: sink.bins == [2, 1])
        assert config.m == pytest.approx(profile.base_m)          # still none for bin 1: last one kept
    finally:
        engine.request_stop()
        worker.join(5)


def test_controller_lookup_reads_the_file_of_the_current_setting(tmp_path):
    configuration = ConfigStore(tmp_path / 'config')
    folder = configuration.wavelength_folder('GenICam')
    assert folder == tmp_path.resolve() / 'calibration' / 'wavelength' / 'genicam'
    host = SimpleNamespace(config=SimpleNamespace(_config_store=configuration, camera_backend='GenICam',
                                                  smile_map_folder=str(tmp_path / 'smile')))
    assert calibration_store(host).wavelength_folder == folder    # the modules save where the scan reads
    store = CalibrationStore(folder, tmp_path / 'smile')
    store.save_wavelength(fit_wavelength([0, 50], [900.0, 960.0], 2, (H, W // 2), H // 2),
                          CameraSetting('genicam', 0, 2, True, (H, W // 2)))

    controller = ScanController.__new__(ScanController)            # no camera window: only the lookup
    controller.config = SimpleNamespace(camera_backend='genicam', camera_index=0, raw_inverted=True)
    controller.backend = FakeCamera()
    controller._wavelength_store = store
    controller.backend.set_bin(2)
    assert controller._stored_wavelength_profile().c == pytest.approx(900.0)
    controller.backend.set_bin(1)
    with pytest.raises(LookupError, match='wavelength_bin1.json'):
        controller._stored_wavelength_profile()
    controller.backend.set_bin(2)
    controller.config.raw_inverted = False
    with pytest.raises(LookupError, match='orientation'):
        controller._stored_wavelength_profile()
