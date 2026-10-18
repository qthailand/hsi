"""Compare extracted processing and file bytes with the pre-refactor source."""
import ast
import os
from pathlib import Path
import struct
import sys
import textwrap
import subprocess
import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from modules.radiometric import core
from modules.cube_io import envi
from modules.cube_io.writers import HypercubeWriter,SpectrogramWriter

# Captured processing-only baseline: no old GUI or SDK imports.
BASE=Path(__file__).parent/'fixtures/stage2_baseline.py'

def baseline():
    import importlib.util
    spec=importlib.util.spec_from_file_location('stage2_baseline',BASE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class Progress:
    def __init__(self):self.values=[]
    def emit(self,v):self.values.append(v)

def cube_file(tmp_path,interleave='bip'):
    a=(np.arange(4*6*12,dtype=np.uint16).reshape(4,6,12)+20)
    f=tmp_path/(interleave+'.bin')
    disk={'bip':a,'bsq':a.transpose(2,0,1),'bil':a.transpose(0,2,1)}[interleave]
    disk.copy().tofile(f)
    Path(str(f)+'.hdr').write_text('ENVI\nlines = 4\nsamples = 6\nbands = 12\ndata type = 12\ninterleave = '+interleave+'\nbyte order = 0\nwavelength = {'+','.join(str(x) for x in np.arange(400,424,2))+'}\n')
    return f,a,envi.parse_envi_header(str(f))

@pytest.mark.parametrize('interleave',['bip','bsq','bil'])
def test_envi_bands(tmp_path,interleave):
    f,a,meta=cube_file(tmp_path,interleave);old=baseline()
    for b in range(12):
        np.testing.assert_array_equal(envi._read_envi_band_float32(str(f),meta,b),old._read_envi_band_float32(str(f),meta,b))
        np.testing.assert_array_equal(envi._read_envi_band_float32(str(f),meta,b),a[:,:,b])

def test_factors_match_original(tmp_path):
    f,a,meta=cube_file(tmp_path);old=baseline();mask=np.ones((4,6),bool);mask[0]=False
    sheet=tmp_path/'ref.xlsx';pd.DataFrame([[390,.3],[400,.4],[430,.7],[450,.9]]).to_excel(sheet,index=False,header=False)
    p=Progress();values=[]
    x=old.rowwise_calculation_heavy_logic(str(f),meta,str(sheet),2,mask,p)
    y=core.rowwise_calculation_heavy_logic(str(f),meta,str(sheet),2,mask,values.append)
    np.testing.assert_array_equal(x['factors'],y['factors']);assert x['span']==y['span'] and values==p.values
    p=Progress();x=old.fluo_calculation_heavy_logic(str(f),meta,mask,p);y=core.fluo_calculation_heavy_logic(str(f),meta,mask)
    np.testing.assert_array_equal(x['factors'],y['factors'])

@pytest.mark.parametrize('resample',[False,True])
@pytest.mark.parametrize('calibrated',[False,True])
def test_bsq_processing_bytes(tmp_path,resample,calibrated):
    f,a,meta=cube_file(tmp_path);old=baseline();params=dict(raw_filepath=str(f),raw_meta=meta,save_path=str(tmp_path/'old.bin'),crop_coords=(0,6,0,4),is_wavelength=True,band_count=12,factors=None,rowwise_factors=np.ones((4,12),np.float32)*.7 if calibrated else None,binning_factor=2,lam_target=np.array([401,405,411],np.float32))
    name='create_datacube_resampled_logic' if resample else 'create_datacube_logic'
    getattr(old,name)(params,Progress());newparams=dict(params,save_path=str(tmp_path/'new.bin'));getattr(core,name)(newparams)
    assert Path(params['save_path']).read_bytes()==Path(newparams['save_path']).read_bytes()
    assert Path(params['save_path']+'.hdr').read_bytes()==Path(newparams['save_path']+'.hdr').read_bytes()

@pytest.mark.parametrize('calibrated',[False,True])
def test_temporary_cube_bytes(tmp_path,calibrated):
    f,a,meta=cube_file(tmp_path);old=baseline();params=dict(raw_filepath=str(f),folder=str(f),raw_meta=meta,save_path=str(tmp_path/'old.tmp'),crop=(1,5,0,4),factors=np.arange(12,dtype=np.float32)+1)
    name='build_cal_cube_tmp' if calibrated else 'build_raw_cube_tmp'
    x=getattr(old,name)(params,Progress());q=dict(params,save_path=str(tmp_path/'new.tmp'));y=getattr(core,name)(q)
    assert Path(x['path']).read_bytes()==Path(y['path']).read_bytes()
    assert {k:v for k,v in x.items() if k!='path'}=={k:v for k,v in y.items() if k!='path'}

def test_spectrogram_format_and_cancel(tmp_path):
    a=np.arange(3*4*5,dtype=np.uint16).reshape(3,4,5);f=tmp_path/'scan.raw';values=[]
    writer=SpectrogramWriter(save_path=str(f),config={'m':.29,'c':400},spectrogram=a,pixel_size=.1,camera_bin=2,progress=values.append,fsync=False)
    writer.run();data=f.read_bytes();assert data[:16]==struct.pack('<fffHH',.58,400,.1,4,5);assert data[16:]==a.tobytes();assert values[-1]==100
    writer=SpectrogramWriter(save_path=str(f),config={'m':.29,'c':400},spectrogram=a,pixel_size=.1,fsync=False);before=f.read_bytes();writer.cancel();writer.run();assert f.read_bytes()==before;assert not Path(str(f)+'.tmp').exists()

def test_export_roundtrip(tmp_path):
    from modules.cube_io import load_envi_cube_for_viewer
    a=np.arange(3*4*5,dtype=np.uint16).reshape(3,4,5);wl=np.array([400,402,404],np.float32);f=tmp_path/'out.bin'
    HypercubeWriter(save_path=str(f),config={'m':.29,'c':400},hypercube=a,linear_spectrum=wl).run()
    loaded,wavelengths=load_envi_cube_for_viewer(str(f)+'.hdr');np.testing.assert_array_equal(loaded,a.transpose(1,2,0));np.testing.assert_array_equal(wavelengths,wl)

def test_hdf5_data(tmp_path):
    import h5py
    from modules.cube_io.hdf5 import save_hdf5
    a=np.arange(3*4*5,dtype=np.uint16).reshape(3,4,5);f=tmp_path/'out.h5';progress=[];save_hdf5(str(f),a,progress.append)
    with h5py.File(f) as h:np.testing.assert_array_equal(h['spectrogram'][:],a);assert h['spectrogram'].compression=='gzip';assert h['dtype']==str(a.dtype) if 'dtype' in h else h.attrs['dtype']==str(a.dtype)
    assert progress[-1]==100

def test_headless_packages():
    subprocess.run([sys.executable,'-c','import sys; import modules.radiometric, modules.cube_io.writers; assert not any(n.startswith(("PyQt5", "matplotlib.backends.backend_qt")) for n in sys.modules)'],check=True)

def export_baseline():
    import importlib.util
    spec=importlib.util.spec_from_file_location('export_baseline',BASE.with_name('stage2_export_baseline.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_envi_writer_matches_original_bytes(tmp_path,monkeypatch):
    pytest.importorskip('rasterio')   # the frozen baseline module imports it
    import datetime
    import modules.cube_io.writers as writers
    old=export_baseline()
    class FixedDate:
        @staticmethod
        def now():return datetime.datetime(2020,1,2,3,4,5)
    monkeypatch.setattr(old,'datetime',FixedDate);monkeypatch.setattr(writers,'datetime',FixedDate)
    a=np.arange(3*4*5,dtype=np.uint16).reshape(3,4,5);wl=np.array([400,402,404],np.float32)
    kwargs=dict(config={'m':.29,'c':400},hypercube=a,linear_spectrum=wl)
    old.HypercubeWorker(save_path=str(tmp_path/'old.bin'),**kwargs).run()
    writers.HypercubeWriter(save_path=str(tmp_path/'new.bin'),**kwargs).run()
    assert (tmp_path/'old.bin').read_bytes()==(tmp_path/'new.bin').read_bytes()
    assert (tmp_path/'old.bin.hdr').read_bytes()==(tmp_path/'new.bin.hdr').read_bytes()

def test_geotiff_matches_original_data_and_metadata(tmp_path):
    rasterio=pytest.importorskip('rasterio')
    from modules.cube_io.writers import GeoTIFFWriter
    old=export_baseline();a=np.arange(3*4*5,dtype=np.uint16).reshape(3,4,5);wl=np.array([400,402,404],np.float32)
    kwargs=dict(config={'m':.29,'c':400},hypercube=a,linear_spectrum=wl)
    errors=[];worker=old.GeoTIFFWorker(save_path=str(tmp_path/'old.tif'),**kwargs);worker.error.connect(errors.append);worker.run();assert not errors
    GeoTIFFWriter(save_path=str(tmp_path/'new.tif'),**kwargs).run()
    with rasterio.open(tmp_path/'old.tif') as x,rasterio.open(tmp_path/'new.tif') as y:
        np.testing.assert_array_equal(x.read(),y.read());assert x.profile==y.profile;assert x.descriptions==y.descriptions
        for b in range(1,4):assert x.tags(b)==y.tags(b)

def test_qt_adapters_deliver_progress_and_finished(tmp_path):
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'workers/export_workers.py';spec=importlib.util.spec_from_file_location('export_adapter_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    a=np.arange(3*4*5,dtype=np.uint16).reshape(3,4,5)
    worker=m.HypercubeWorker(save_path=str(tmp_path/'test.bin'),config={'m':.29,'c':400},hypercube=a,linear_spectrum=np.array([400,402,404]))
    done=[];progress=[];errors=[];worker.finished.connect(done.append);worker.progressed.connect(progress.append);worker.error.connect(errors.append);worker.run()
    assert len(done)==1 and progress[-1]==100 and not errors
    from modules.radiometric.workers import Worker
    f,_,meta=cube_file(tmp_path);mask=np.ones((4,6),bool);w=Worker(core.fluo_calculation_heavy_logic,str(f),meta,mask);done=[];errors=[];w.finished.connect(done.append);w.error.connect(errors.append);w.run();assert len(done)==1 and not errors
