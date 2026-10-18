"""Regression coverage for package boundaries and preserved array semantics."""
import subprocess
import sys
import numpy as np
import pytest
from modules.dark import DarkProcessor, DarkFrameLibrary
from modules.spectral_processing import LambdaBinner
from modules.image import apply_raw_orientation
from modules.cube_io import create_pushbroom_cube_from_corrected, load_envi_cube_for_viewer
from modules.contracts import CubeData

def test_headless_imports():
    subprocess.run([sys.executable, '-c', 'import sys; import modules.dark, modules.spectral_processing, modules.image, modules.cube_io; assert not any(n.startswith(("PyQt5", "matplotlib", "rasterio")) for n in sys.modules)'], check=True)

def test_binning_and_cube_axes():
    raw = np.arange(2*5*64, dtype=np.uint16).reshape(2,5,64)
    b = LambdaBinner(.58,399.791,2,threads=2).fit(64)
    out=b.transform(raw)
    expected=np.empty_like(out)
    for k,(l,r,w) in enumerate(zip(b.iL,b.iR,b.w)):
        expected[:,:,k]=(raw[:,:,l:r].sum(axis=2,dtype=np.uint64)+w//2)//w
    np.testing.assert_array_equal(out,expected)
    cube=create_pushbroom_cube_from_corrected(list(out))
    np.testing.assert_array_equal(cube,out.transpose(1,0,2))
    np.testing.assert_array_equal(b.transform_to_cube(raw)[0],out.transpose(2,1,0))
    CubeData(cube,wavelengths=b.lambda_out).validate()

def test_dark_edges_and_inplace_behavior():
    p=DarkProcessor();f=np.arange(20,dtype=np.uint16).reshape(4,5)
    assert p._apply_dark_subtract(f) is f
    hot=np.zeros_like(f,np.uint8);hot[0,0]=255;hot[0,1]=255
    p.set_dark_reference(np.ones_like(f)*2,hot)
    corrected=p._apply_dark_subtract(f)
    np.testing.assert_array_equal(corrected,np.maximum(f.astype(int)-2,0))
    before=corrected.copy()
    assert p._apply_hot_pixel(corrected) is corrected
    assert corrected[0,0]==round(float(before[1,0:2].mean()))
    p.set_dark_reference(None,None)
    assert p._apply_hot_pixel(f) is f

def test_dark_persistence(tmp_path):
    lib=DarkFrameLibrary(str(tmp_path),n_frames=3)
    lib.start_capture(bin=2,exposure=10,gain=180)
    for v in [10,12,14]:lib.add_frame(np.full((4,6),v,np.uint16))
    result=lib.load(bin=2,exposure=10,gain=180)
    np.testing.assert_array_equal(result['mean'],np.full((4,6),12,np.uint16))
    assert result['n_frames']==3 and not lib.is_collecting

def test_envi_loader(tmp_path):
    import spectral
    f=tmp_path/'test.hdr';a=np.arange(3*4*5,dtype=np.uint16).reshape(3,4,5)
    spectral.envi.save_image(str(f),a,interleave='bip',metadata={'wavelength':[400,402,404,406,408],'wavelength units':'Nanometers'})
    cube,wl=load_envi_cube_for_viewer(f)
    assert isinstance(cube,np.memmap)
    np.testing.assert_array_equal(cube,a)
    np.testing.assert_array_equal(wl,[400,402,404,406,408])

def test_orientation_and_contract():
    a=np.arange(20,dtype=np.uint16).reshape(4,5)
    assert apply_raw_orientation(a,False) is a
    np.testing.assert_array_equal(apply_raw_orientation(a,True),a[::-1])
    with pytest.raises(ValueError):CubeData(a).validate()
    with pytest.raises(ValueError):CubeData(a[None],wavelengths=np.arange(2)).validate()

