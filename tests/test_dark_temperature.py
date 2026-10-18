"""Dark references keyed by sensor temperature, cooler target UI, saved settings and header temperatures."""
import json
import os
import threading

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from modules.dark import DarkFrameLibrary, UNKNOWN_TEMPERATURE_C, select_dark


def capture(lib, value, temperature, bin=1, exposure=10.0, gain=1.0):
    lib.start_capture(bin=bin, exposure=exposure, gain=gain)
    for _ in range(lib.n_frames):
        done = lib.add_frame(np.full((4, 6), value, np.uint16), temperature=temperature)
    assert done


def test_capture_names_file_by_mean_temperature_and_records_range(tmp_path):
    lib = DarkFrameLibrary(str(tmp_path), n_frames=3)
    lib.start_capture(bin=2, exposure=10.0, gain=1.0)
    for t in (24.0, -999.0, 25.0):                     # -999 = camera could not read it
        lib.add_frame(np.full((4, 6), 5, np.uint16), temperature=t)
    assert os.listdir(tmp_path) == ['dark_b2_e10.0_g1.0_t+24.npz']   # mean 24.5 rounds to 24
    result = lib.load(2, 10.0, 1.0, temperature=24.0)
    assert (result['temperature'], result['temperature_min'], result['temperature_max']) == (24.5, 24.0, 25.0)
    assert lib.list_keys()[0]['temperature'] == 24.5


def test_nearest_temperature_wins_and_unknown_assumes_60(tmp_path):
    lib = DarkFrameLibrary(str(tmp_path), n_frames=2)
    capture(lib, 10, 20.0)
    capture(lib, 30, 40.0)
    capture(lib, 50, -5.0)
    assert lib.load(1, 10.0, 1.0, temperature=23.0)['mean'][0, 0] == 10
    assert lib.load(1, 10.0, 1.0, temperature=33.0)['mean'][0, 0] == 30
    assert lib.load(1, 10.0, 1.0, temperature=-3.0)['filename'] == 'dark_b1_e10.0_g1.0_t-5.npz'
    assert UNKNOWN_TEMPERATURE_C == 60.0
    assert lib.load(1, 10.0, 1.0, temperature=-999.0)['mean'][0, 0] == 30     # nearest 60 C
    assert lib.load(1, 10.0, 1.0)['mean'][0, 0] == 30
    assert lib.load(1, 10.0, 2.0, temperature=20.0) is None                    # other gain


def test_recapture_at_same_degree_replaces_and_legacy_is_only_a_fallback(tmp_path):
    lib = DarkFrameLibrary(str(tmp_path), n_frames=2)
    capture(lib, 7, None)                                   # no temperature: legacy name
    assert lib.load(1, 10.0, 1.0, temperature=25.0)['temperature'] is None
    capture(lib, 8, 25.2)
    capture(lib, 9, 24.8)                                   # same whole degree: replaces
    assert sorted(os.listdir(tmp_path)) == ['dark_b1_e10.0_g1.0.npz', 'dark_b1_e10.0_g1.0_t+25.npz']
    assert lib.load(1, 10.0, 1.0, temperature=90.0)['mean'][0, 0] == 9         # legacy never preferred
    lib.delete(1, 10.0, 1.0, temperature=25.0)
    assert lib.load(1, 10.0, 1.0, temperature=25.0)['mean'][0, 0] == 7
    lib.delete(1, 10.0, 1.0)
    assert not lib.exists(1, 10.0, 1.0)


def test_select_dark_reports_temperatures(tmp_path):
    lib = DarkFrameLibrary(str(tmp_path), n_frames=2)
    capture(lib, 10, 21.0)
    selection = select_dark(lib, 1, 10.0, 1.0, (4, 6), temperature=23.5)
    assert selection.kind == 'loaded' and selection.dark_temperature == 21.0 and selection.temperature == 23.5
    assert select_dark(lib, 1, 10.0, 1.0, (5, 6), temperature=-999).kind == 'mismatch'
    assert select_dark(lib, 1, 10.0, 1.0, (5, 6), temperature=-999).temperature is None


def test_update_section_keeps_other_keys_and_load_reads_new_fields(tmp_path):
    from dataclasses import asdict
    from config import AppConfig, load_app_config
    from modules.configuration import ConfigStore
    data = asdict(AppConfig())
    data.update(m=.29, c=400)
    for optional in ('target_temperature_c', 'rotation_degree', 'rotation_speed_dps',
                     'linear_distance_mm', 'linear_speed_um_s'):
        data.pop(optional)                                   # older settings files lack them
    directory = tmp_path / 'config'
    ConfigStore(directory).initialize(data)
    config = load_app_config(str(directory))
    assert config.target_temperature_c is None and config.rotation_degree == 10.0
    store = config._config_store
    assert store.update_section('camera', {'default_gain': 42, 'target_temperature_c': -10.0})
    assert store.update_section('scan', {'rotation_degree': 33.5, 'linear_speed_um_s': 250.0})
    assert not store.update_section('scan', {'rotation_degree': 33.5})        # unchanged: not rewritten
    camera = json.loads((directory / 'camera.json').read_text())
    assert camera['camera_backend'] == 'poa' and camera['default_gain'] == 42
    loaded = load_app_config(str(directory))
    assert (loaded.default_gain, loaded.target_temperature_c) == (42, -10.0)
    assert (loaded.rotation_degree, loaded.linear_speed_um_s) == (33.5, 250.0)


def test_camera_controls_temperature_row():
    from modules.camera.widget import CameraControls
    app = QApplication.instance() or QApplication([])
    controls = CameraControls()
    emitted = []
    controls.target_temperature_changed.connect(emitted.append)
    assert controls.temp_spin.isReadOnly() and controls.temp_label.text() == 'Sensor Temp'
    assert controls.target_temperature() is None
    controls.show_sensor_temperature(31.24)
    assert controls.temp_spin.value() == pytest.approx(31.2)
    controls.show_sensor_temperature(-999.0)
    assert controls.temp_spin.text() == '—'                # unknown
    controls.set_cooling_supported(True, (-20.0, 30.0))
    assert not controls.temp_spin.isReadOnly() and controls.temp_label.text() == 'Target Temp'
    controls.set_target_temperature(-5.0)
    controls.show_sensor_temperature(28.0)                  # ignored while it is the target
    assert controls.target_temperature() == -5.0 and emitted == []
    controls.temp_spin.setValue(-12.0)
    controls.temp_spin.editingFinished.emit()
    controls.temp_spin.editingFinished.emit()               # same value: once only
    assert emitted == [-12.0]
    controls.set_target_temperature(99.0)                   # clamped to the camera's range
    assert controls.target_temperature() == 30.0


def test_envi_header_has_scan_temperatures(tmp_path):
    from modules.cube_io.writers import HypercubeWriter
    path = str(tmp_path / 'cube.img')
    HypercubeWriter(save_path=path, config={'m': .29, 'c': 400.0, 'sensor_temperature_c_start': 24.5,
                                            'sensor_temperature_c_end': 25.0, 'dark_temperature_c': 24.0},
                    hypercube=np.zeros((3, 2, 2), np.uint16), linear_spectrum=np.arange(3.0)).run()
    header = open(path + '.hdr').read()
    assert 'sensor temperature start = 24.50' in header and 'dark temperature = 24.00' in header
    assert 'target temperature' not in header


def test_engine_applies_target_temperature_between_frames():
    from types import SimpleNamespace
    from modules.acquisition.engine import AcquisitionEngine, AcquisitionSink

    class Sink(AcquisitionSink):
        def __init__(self): self.events = []
        def target_temperature_applied(self, t): self.events.append(('ok', t))
        def target_temperature_failed(self, m): self.events.append(('fail', m))

    class Backend:
        def __init__(self): self.targets = []
        def set_target_temperature(self, t):
            if t > 50:
                raise RuntimeError('out of range')
            self.targets.append(t)

    engine = AcquisitionEngine.__new__(AcquisitionEngine)
    engine.backend, engine.sink = Backend(), Sink()
    engine._lock = threading.Lock()
    engine._pending_target_temperature = None
    engine._apply_pending_target_temperature()               # nothing queued
    engine.request_target_temperature(-10)
    engine._apply_pending_target_temperature()
    engine.request_target_temperature(80)
    engine._apply_pending_target_temperature()
    assert engine.backend.targets == [-10.0]
    assert engine.sink.events == [('ok', -10.0), ('fail', 'out of range')]
