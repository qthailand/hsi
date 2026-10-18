"""The FWHM stored per camera setting can set the wavelength bin width (config resolution as fallback)."""
import json
import logging
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from controllers.scan_controller import ScanController
from hsi_fakes import FakeCamera, H, W
from modules.acquisition import AcquisitionEngine, AcquisitionSink, build_raw_metadata
from modules.contracts import CalibrationStore, CameraSetting


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


def config(**updates):
    data = dict(smile_map_folder='.', preprocessing_backend='cpu', m=.5, c=400., resolution=2.0)
    data.update(updates)
    return SimpleNamespace(**data)


def run_engine(camera, cfg, lookup, body):
    """Stream ``camera`` through an engine with ``lookup`` and call ``body(engine, sink)``."""
    sink = Sink()
    engine = AcquisitionEngine(camera, cfg, sink=sink, profile=False)
    engine.set_resolution_lookup(lookup)
    worker = threading.Thread(target=engine.run_until_stopped, daemon=True)
    worker.start()
    try:
        assert wait_for(lambda: sink.frames and engine.binner_lambda_out is not None)
        body(engine, sink)
    finally:
        engine.request_stop()
        worker.join(5)


def spacing(engine):
    return float(np.median(np.diff(engine.binner_lambda_out)))


# ───────────────────────────── store ─────────────────────────────

def test_store_round_trip_mismatch_and_history(tmp_path):
    store = CalibrationStore(tmp_path / 'wl', tmp_path / 'smile')
    setting = CameraSetting('poa', 0, 2, True, (552, 804))
    assert store.load_resolution(setting) is None
    path = store.save_resolution(1.5, setting, {'lines': [{'fwhm_nm_median': 1.4}], 'profile': 'gaussian'})
    assert path == tmp_path / 'wl' / 'resolution_bin2.json'
    stored = store.load_resolution(setting)
    assert stored.ok and stored.fwhm_nm == 1.5 and stored.details['profile'] == 'gaussian'
    assert store.load_resolution(CameraSetting('poa', 0, 2, False, (552, 804))).problems   # other orientation
    assert store.load_resolution(CameraSetting('poa', 0, 1, True, ())) is None             # other bin: other file
    store.save_resolution(1.6, setting)
    assert store.load_resolution(setting).fwhm_nm == 1.6
    assert len(list((tmp_path / 'wl' / 'history').glob('resolution_bin2.*.json'))) == 1
    for bad in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            store.save_resolution(bad, setting)
    assert store.load_resolution(setting).fwhm_nm == 1.6                                   # untouched
    data = json.loads(path.read_text(encoding='utf-8'))
    data['fwhm_nm'] = -2
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError):
        store.load_resolution(setting)


# ───────────────────────────── engine ─────────────────────────────

def test_config_source_never_reads_the_fwhm():
    calls = []

    def lookup():
        calls.append(1)
        return 1.5

    def body(engine, sink):
        assert spacing(engine) == pytest.approx(2.0, abs=0.05)
        assert engine.binning_info() == {'resolution_nm': 2.0, 'resolution_source': 'config'}

    run_engine(FakeCamera(), config(), lookup, body)                      # resolution_source missing = config
    run_engine(FakeCamera(), config(resolution_source='config'), lookup, body)
    assert calls == []


def test_fwhm_source_uses_each_bins_fwhm_and_falls_back_to_config(caplog):
    camera = FakeCamera()

    def lookup():
        if camera.current_bin == 1:
            return 1.5
        raise LookupError('no resolution_bin2.json')

    caplog.set_level(logging.WARNING, logger='modules.acquisition.engine')

    def body(engine, sink):
        assert spacing(engine) == pytest.approx(1.5, abs=0.05)
        assert engine.binning_info() == {'resolution_nm': 1.5, 'resolution_source': 'fwhm',
                                         'fwhm_nm': 1.5, 'fwhm_step_factor': 1.0}
        engine.request_bin(2)
        assert wait_for(lambda: sink.bins == [2] and engine.binner_lambda_out is not None)
        assert spacing(engine) == pytest.approx(2.0, abs=0.05)                     # no FWHM: config resolution
        assert engine.binning_info()['resolution_source'] == 'config'
        assert 'Bin 2: no usable FWHM (no resolution_bin2.json)' in caplog.text

    run_engine(camera, config(resolution_source='fwhm'), lookup, body)


def test_fwhm_step_factor_and_two_pixel_minimum():
    def body_half(engine, sink):
        assert spacing(engine) == pytest.approx(1.5, abs=0.05)                     # 0.5 × 3 nm

    run_engine(FakeCamera(), config(resolution_source='fwhm', fwhm_step_factor=0.5), lambda: 3.0, body_half)

    def body_min(engine, sink):
        info = engine.binning_info()
        assert info['resolution_nm'] == pytest.approx(1.0) and info['fwhm_nm'] == 0.3   # 2 px × 0.5 nm/px
        assert spacing(engine) == pytest.approx(1.0, abs=0.05)

    run_engine(FakeCamera(), config(resolution_source='fwhm'), lambda: 0.3, body_min)


def test_bad_fwhm_falls_back_and_reload_picks_up_a_new_one():
    value = {'fwhm': float('nan')}

    def body(engine, sink):
        assert engine.binning_info()['resolution_source'] == 'config'               # NaN refused
        value['fwhm'] = 1.25
        engine.set_scanning(True)
        engine.request_resolution_reload()
        time.sleep(0.2)
        assert engine.binning_info()['resolution_source'] == 'config'               # a scan keeps its bins
        engine.set_scanning(False)
        assert wait_for(lambda: engine.binning_info()['resolution_source'] == 'fwhm')
        assert wait_for(lambda: engine.binner_lambda_out is not None
                        and spacing(engine) == pytest.approx(1.25, abs=0.05))

    run_engine(FakeCamera(), config(resolution_source='fwhm'), lambda: value['fwhm'], body)


def test_metadata_records_the_bin_width_actually_used():
    camera = SimpleNamespace(info=SimpleNamespace(camera_id=0, bit_depth=16, pixel_size=3.75, width=804, height=552),
                             current_bin=2)
    cfg = SimpleNamespace(camera_backend='poa', camera_index=0, raw_inverted=True, m=.2903, c=399.791, resolution=2)
    assert build_raw_metadata(backend=camera, config=cfg)['resolution_nm'] == 2.0
    meta = build_raw_metadata(backend=camera, config=cfg, binning={
        'resolution_nm': 1.4, 'resolution_source': 'fwhm', 'fwhm_nm': 1.4, 'fwhm_step_factor': 1.0})
    assert (meta['resolution_nm'], meta['resolution_source'], meta['fwhm_nm']) == (1.4, 'fwhm', 1.4)


# ───────────────────────────── controller ─────────────────────────────

def test_controller_lookup_reads_the_fwhm_of_the_current_setting(tmp_path):
    store = CalibrationStore(tmp_path / 'wl', tmp_path / 'smile')
    store.save_resolution(1.7, CameraSetting('genicam', 0, 2, True, (H, W // 2)))
    controller = ScanController.__new__(ScanController)            # no camera window: only the lookup
    controller.config = SimpleNamespace(camera_backend='genicam', camera_index=0, raw_inverted=True)
    controller.backend = FakeCamera()
    controller._wavelength_store = store
    controller.backend.set_bin(2)
    assert controller._stored_fwhm_nm() == 1.7
    controller.backend.set_bin(1)
    with pytest.raises(LookupError, match='resolution_bin1.json'):
        controller._stored_fwhm_nm()
    controller.backend.set_bin(2)
    controller.config.raw_inverted = False
    with pytest.raises(LookupError, match='orientation'):
        controller._stored_fwhm_nm()


# ───────────────────────────── config ─────────────────────────────

def test_config_reads_source_factor_and_float_resolution(tmp_path):
    from config import load_app_config
    from test_configuration import settings
    p = settings(tmp_path)
    cfg = load_app_config(str(p))
    assert (cfg.resolution_source, cfg.fwhm_step_factor, cfg.resolution) == ('config', 1.0, 2.0)
    proc = p / 'processing.json'
    data = json.loads(proc.read_text())
    data.update(resolution=1.5, resolution_source='FWHM', fwhm_step_factor=0.5)
    proc.write_text(json.dumps(data))
    cfg = load_app_config(str(p))
    assert (cfg.resolution_source, cfg.fwhm_step_factor, cfg.resolution) == ('fwhm', 0.5, 1.5)
    for key, value in (('resolution_source', 'median'), ('fwhm_step_factor', 0), ('resolution', -1)):
        proc.write_text(json.dumps({**data, key: value}))
        with pytest.raises(ValueError):
            load_app_config(str(p))
    del data['resolution_source'], data['fwhm_step_factor']           # older processing.json
    proc.write_text(json.dumps(data))
    assert load_app_config(str(p)).resolution_source == 'config'
