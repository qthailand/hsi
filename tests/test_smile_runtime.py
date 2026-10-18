"""Sensor-coordinate remap, dtype, model switching and explicit mismatch checks."""
import numpy as np
import pytest
from modules.smile.calibration import SmileCalibration, save_calibration
from modules.smile.runtime import SmileRuntime

def make_model(path,shape=(8,16),shift=0):
    h,w=shape
    cal=SmileCalibration(map_x=np.tile(np.arange(w,dtype=np.float32)+shift,(h,1)),margin=(0,0),image_shape=shape,deg_y=0,deg_x=0,surface_coef=np.array([0.]),y_rng=(0,h-1),x_rng=(0,w-1),interp='linear',fit_rms=0.,cv_rms=0.,rms_before=0.,rms_after=0.,n_lines_used=1,n_lines_total=1,created='test',y_ref=3.)
    save_calibration(str(path),cal)

def test_identity_dtype_and_no_crop(tmp_path):
    path=tmp_path/'smile_model_bin2.npz';make_model(path)
    rt=SmileRuntime(tmp_path);assert rt.set_bin(2)
    a=np.arange(128,dtype=np.uint16).reshape(8,16)
    out=rt.apply(a)
    assert out.dtype==a.dtype and out.shape==a.shape
    np.testing.assert_array_equal(out,a)
    engine=rt.calibration._engine
    rt.apply(a);assert rt.calibration._engine is engine
    assert rt.calibration.y_ref==3.

def test_shift_and_zero_border(tmp_path):
    path=tmp_path/'model.npz';make_model(path,shift=1)
    rt=SmileRuntime(tmp_path);rt.load(path,1)
    a=np.arange(128,dtype=np.uint16).reshape(8,16)
    out=rt.apply(a)
    np.testing.assert_array_equal(out[:,:-1],a[:,1:]);assert not out[:,-1].any()

def test_bin_change_disables_previous_model(tmp_path):
    make_model(tmp_path/'smile_model_bin2.npz')
    rt=SmileRuntime(tmp_path);rt.set_bin(2);assert not rt.set_bin(4)
    a=np.zeros((8,16),np.uint16);assert rt.apply(a) is a

def test_mismatch_rejected(tmp_path):
    make_model(tmp_path/'model.npz')
    rt=SmileRuntime(tmp_path);rt.load(tmp_path/'model.npz',1)
    with pytest.raises(ValueError,match='does not match'):rt.apply(np.zeros((4,8),np.uint16))
    assert rt.calibration is None

def test_integration_source_order_and_syntax():
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    for rel in ['controllers/scan_controller.py','workers/camera_preview.py','windows/main_window.py','config.py']:
        ast.parse((root/rel).read_text())
    src=(root/'workers/camera_preview.py').read_text()
    assert 'workers.smile_corrector' not in (root/'controllers/scan_controller.py').read_text()
