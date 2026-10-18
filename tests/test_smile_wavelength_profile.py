"""Smile Analysis must load the profile saved by the Wavelength Calibration panel."""
import json
import numpy as np
import pytest
from modules.wavelength_calibration import fit_wavelength
from modules.smile.quality import normalize_wavelength_config

def saved_profile(tmp_path, row=60, bin_=2):
    x = np.array([40., 120., 300., 650.]); y = .58 * x + 400
    path = tmp_path / 'profile.json'
    fit_wavelength(x, y, camera_bin=bin_, frame_shape=(120, 800), selected_row=row).save(path)
    return path, x, y

def test_profile_is_converted_to_smile_layout(tmp_path):
    path, x, y = saved_profile(tmp_path)
    config = normalize_wavelength_config(json.loads(path.read_text()))
    assert config['m_nm_per_px'] == pytest.approx(.58) and config['c_nm'] == pytest.approx(400)
    assert [s['x_px'] for s in config['standards']] == list(x)
    assert [s['wavelength_nm'] for s in config['standards']] == pytest.approx(list(y))
    assert config['reference_row_y'] == 60

def test_smile_layout_passes_through_and_garbage_is_rejected():
    smile = dict(m_nm_per_px=.5, c_nm=500, standards=[dict(wavelength_nm=540, x_px=80)])
    assert normalize_wavelength_config(smile) is smile
    for bad in ({}, {'standards': [1]}, {'m': 1, 'c': 2}, [], {'pixels': [], 'wavelengths': [], 'm': 1, 'c': 2}):
        with pytest.raises(ValueError):
            normalize_wavelength_config(bad)

def test_widget_loads_profile_and_checks_reference_row(tmp_path):
    from types import SimpleNamespace
    from PyQt5.QtWidgets import QApplication
    from modules.smile.widget import SmileCorrectionWidget
    app = QApplication.instance() or QApplication([])
    path, _, _ = saved_profile(tmp_path, row=60)
    widget = SmileCorrectionWidget()
    widget.load_wavelength_calibration(str(path))          # used to fail with the format error
    assert widget._wavelength_config['standards'][0]['x_px'] == 40.
    assert widget.lbl_quality.text() == 'profile.json'
    widget._result = SimpleNamespace(y_ref=90)             # other reference row -> explained, not a format error
    widget.run_quality_analysis()
    text = widget.lbl_quality.text()
    assert 'row 60' in text and '90' in text and 'Reference row mismatch' in text
    widget.close()

@pytest.mark.parametrize('height', [2, 7, 8, 9, 552, 553])
def test_calibration_row_defaults_to_smile_reference_row(height):
    from PyQt5.QtWidgets import QApplication
    from modules.smile.analysis import _default_reference_row
    from modules.wavelength_calibration import WavelengthCalibrationWidget
    app = QApplication.instance() or QApplication([])
    widget = WavelengthCalibrationWidget(); widget.set_frame(np.zeros((height, 50)), 1)
    assert widget.row.value() == _default_reference_row(height)
    widget.row.setValue(0); widget.set_frame(np.zeros((height, 50)), 1)      # a new image resets to the rule
    assert widget.row.value() == _default_reference_row(height)
    widget.close()
