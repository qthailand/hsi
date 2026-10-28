"""Scan metadata in ENVI headers: written by the exporter, read back, kept by radiometric outputs."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from modules.cube_io import envi_scan_metadata
from modules.cube_io.envi import parse_envi_header, release_band_maps
from modules.cube_io.writers import (HypercubeWriter, carried_header_lines, metadata_from_header,
                                     metadata_header_lines)

L, S, B = 4, 6, 3
WL = np.array([400.0, 410.0, 420.0])

SCAN = {
    'started_at': '2026-10-27T10:15:30+07:00',
    'bit_depth': 16, 'sensor_bit_depth': 12, 'full_scale': 65520,
    'sensor_temperature_c_start': -9.5, 'sensor_temperature_c_end': -9.25,
    'sensor_temperature_c_min': -9.75, 'sensor_temperature_c_max': -9.0,
    'dark_temperature_c': -10.0, 'target_temperature_c': -10.0,
    'exposure_ms': 12.5, 'gain': 100, 'frame_rate_hz': 79.87, 'camera_bin': 2,
    'dark_corrected': True, 'hot_pixel_corrected': True, 'hot_pixel_count': 37,
    'smile_corrected': False,
    'stage_type': 'linear', 'speed_um_s': 150.0, 'speed_um_s_achieved': 149.921875,
}


@pytest.fixture(autouse=True)
def _release_maps():
    yield
    release_band_maps()


def export(path, config):
    cube = np.arange(B * L * S, dtype=np.uint16).reshape(B, L, S)
    HypercubeWriter(save_path=str(path), config=config, hypercube=cube, linear_spectrum=WL).run()
    return parse_envi_header(str(path))


def test_scan_fields_are_written(tmp_path):
    export(tmp_path / 'cube.img', SCAN)
    text = (tmp_path / 'cube.img.hdr').read_text()
    for line in ('acquisition time = 2026-10-27T03:15:30.000000Z', 'exposure time (ms) = 12.5',
                 'sensor gain = 100', 'frame rate (Hz) = 79.87', 'camera bin = 2',
                 'dark corrected = 1', 'hot pixel corrected = 1', 'hot pixel count = 37',
                 'smile corrected = 0', 'stage type = linear', 'linear speed (um/s) = 150',
                 'linear speed achieved (um/s) = 149.921875', 'sensor temperature start = -9.50',
                 'dark temperature = -10.00', 'data full scale = 65520'):
        assert line + '\n' in text, line
    assert 'smile model' not in text and 'rotation speed' not in text     # unknown: left out


def test_header_reads_back_into_the_same_metadata(tmp_path):
    meta = export(tmp_path / 'cube.img', SCAN)
    back = metadata_from_header(meta)
    assert back.pop('started_at') == '2026-10-27T03:15:30.000000Z'
    expected = {k: v for k, v in SCAN.items() if k != 'started_at'}
    assert back == expected


def test_reexport_of_an_opened_cube_keeps_its_metadata(tmp_path):
    """Open ENVI -> Cube Viewer -> Export: acquisition time and scan fields survive."""
    export(tmp_path / 'scan.img', SCAN)
    config = envi_scan_metadata(tmp_path / 'scan.img.hdr')
    export(tmp_path / 'again.img', config)
    first = (tmp_path / 'scan.img.hdr').read_text().splitlines()
    again = (tmp_path / 'again.img.hdr').read_text().splitlines()
    assert first == again


def test_a_header_without_metadata_gives_nothing(tmp_path):
    (tmp_path / 'x.img.hdr').write_text('ENVI\nlines = 1\nsamples = 1\nbands = 1\ndata type = 12\n')
    assert envi_scan_metadata(tmp_path / 'x.img.hdr') == {}
    assert envi_scan_metadata(tmp_path / 'missing.hdr') == {}       # unreadable: empty, no error
    assert metadata_header_lines({}) == []


def test_calibrated_cubes_drop_the_raw_count_fields():
    meta = {'acquisition time': '2026-10-27T03:15:30.000000Z', 'data full scale': '65520',
            'data bit depth': '16', 'exposure time (ms)': '12.5', 'dark corrected': '1'}
    calibrated = carried_header_lines(meta, calibrated=True)
    assert calibrated == ['acquisition time = 2026-10-27T03:15:30.000000Z',
                          'exposure time (ms) = 12.5', 'dark corrected = 1']
    assert 'data full scale = 65520' in carried_header_lines(meta, calibrated=False)


def _source(tmp_path):
    export(tmp_path / 'src.img', SCAN)
    return str(tmp_path / 'src.img'), parse_envi_header(str(tmp_path / 'src.img'))


@pytest.mark.parametrize('resample', [False, True])
@pytest.mark.parametrize('calibrated', [False, True])
def test_radiometric_outputs_keep_the_scan_metadata(tmp_path, calibrated, resample):
    from modules.radiometric import core
    src, meta = _source(tmp_path)
    out = str(tmp_path / 'out.bip')
    params = dict(raw_filepath=src, raw_meta=meta, save_path=out, crop_coords=(0, S, 0, L),
                  is_wavelength=True, band_count=B, factors=None,
                  rowwise_factors=np.full((L, B), 0.5, np.float32) if calibrated else None,
                  binning_factor=1, lam_target=WL.astype(np.float32))
    (core.create_datacube_resampled_logic if resample else core.create_datacube_logic)(params)
    back = metadata_from_header(parse_envi_header(out))
    assert back['started_at'] == '2026-10-27T03:15:30.000000Z'
    for key in ('exposure_ms', 'gain', 'dark_corrected', 'hot_pixel_count', 'smile_corrected',
                'sensor_temperature_c_start', 'dark_temperature_c', 'speed_um_s_achieved'):
        assert back[key] == pytest.approx(SCAN[key]), key
    assert ('full_scale' in back) is (not calibrated)


def test_calibration_window_keeps_the_scan_metadata_of_an_in_memory_cube():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from modules.radiometric import CalibrationWindow
    window = CalibrationWindow()
    try:
        window.load_from_memory(cube_array=np.ones((L, S, B), np.float32), wavelengths=WL,
                                name='ScanMeta', metadata=SCAN)
        back = metadata_from_header(window.raw_meta)
        assert back['started_at'] == '2026-10-27T03:15:30.000000Z'
        assert back['exposure_ms'] == 12.5 and back['dark_corrected'] is True
        assert back['sensor_temperature_c_max'] == -9.0
    finally:
        window.close()
        release_band_maps()
        app.processEvents()


def _engine(status, hot=None, hz=80.0, smile_path=None):
    from modules.acquisition.engine import AcquisitionEngine
    engine = object.__new__(AcquisitionEngine)
    engine._camera_hz = hz
    engine._last_correction_status = status
    engine.backend = NS(_hot_coords=hot)
    return engine


def test_correction_info_reports_what_the_last_frame_went_through():
    from modules.acquisition.engine import AcquisitionEngine   # noqa: F401
    model = object()
    hot = np.zeros((5, 2), int)
    applied = (False, True, 1, True, id(model), 'C:/maps/smile_model_bin2.npz', 2, False)
    assert _engine(applied, hot).correction_info() == {
        'frame_rate_hz': 80.0, 'dark_corrected': True, 'hot_pixel_corrected': True,
        'hot_pixel_count': 5, 'smile_corrected': True, 'smile_model': 'smile_model_bin2.npz'}
    # dark off, no smile model
    off = (False, False, 1, True, id(None), None, 2, False)
    assert _engine(off, hot).correction_info() == {
        'frame_rate_hz': 80.0, 'dark_corrected': False, 'hot_pixel_corrected': False,
        'smile_corrected': False}
    # dark enabled but the reference does not match; smile model failed on the frame
    mismatch = (False, True, 1, False, id(model), 'x.npz', 2, True)
    info = _engine(mismatch, None).correction_info()
    assert info['dark_corrected'] is False and info['hot_pixel_corrected'] is False
    assert info['smile_corrected'] is False and 'smile_model' not in info


@pytest.mark.parametrize('status', [None, (True, True, 1, True, 1, 'x', 2, False)])
def test_correction_state_is_unknown_without_a_processed_scan_frame(status):
    assert _engine(status, hz=None).correction_info() == {}


def test_scan_controller_collects_the_header_metadata():
    from controllers.scan_controller import ScanController
    controller = object.__new__(ScanController)
    controller._raw_metadata = {'started_at': SCAN['started_at'], 'exposure_ms': 12.5, 'gain': 100,
                                'camera_bin': 2, 'stage_type': 'linear', 'speed_um_s': 150.0,
                                'speed_um_s_achieved': 149.921875, 'camera_backend': 'poa',
                                'frame_width': 804}
    controller.last_scan_temperatures = {'sensor_temperature_c_start': -9.5}
    controller.preview_thread = NS(correction_info=lambda: {'dark_corrected': True})
    meta = controller._scan_header_metadata()
    assert meta == {'started_at': SCAN['started_at'], 'exposure_ms': 12.5, 'gain': 100,
                    'camera_bin': 2, 'stage_type': 'linear', 'speed_um_s': 150.0,
                    'speed_um_s_achieved': 149.921875, 'sensor_temperature_c_start': -9.5,
                    'dark_corrected': True}
