"""Standalone BSQ GUI or offline ENVI crop/calibration/resampling CLI."""
# Allow running this file directly (IDE 'run file', runfile): make the project importable.
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "modules.radiometric"
import argparse
import numpy as np
from .core import create_datacube_logic, create_datacube_resampled_logic, rowwise_calculation_heavy_logic, fluo_calculation_heavy_logic, save_factor_profile, load_factor_profile
from modules.cube_io.envi import parse_envi_header

def main():
    """Run the stand-alone window, or the offline CLI when ``--input`` is given.

    Without ``--input`` the Qt calibration window is shown. With ``--input`` one of three
    jobs runs: ``--compute rowwise|fluorescence`` (needs ``--roi-mask`` NPY, rowwise also
    ``--reference-spectrum`` XLSX; saves factors to ``--output`` as ``.npy`` plus
    ``<name>.wavelength.npy``), or a cube
    export (optionally ``--crop``, ``--factors``/``--rowwise`` NPY, ``--spatial-bin``,
    ``--target-grid START STOP STEP`` for resampling) written as ENVI BIP to ``--output``.
    Argument errors exit via ``argparse``.
    """
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',help='ENVI binary cube path');p.add_argument('--output')
    p.add_argument('--crop',type=int,nargs=4,metavar=('X0','X1','Y0','Y1'))
    p.add_argument('--factors',help='Per-band NPY factors');p.add_argument('--rowwise',help='Row-by-band NPY factors')
    p.add_argument('--spatial-bin',type=int,default=1)
    p.add_argument('--target-grid',type=float,nargs=3,metavar=('START','STOP','STEP'),help='Resample wavelengths; STOP excluded')
    p.add_argument('--compute',choices=['rowwise','fluorescence'],help='Compute factors instead of exporting a cube')
    p.add_argument('--roi-mask',help='Boolean/0-1 NPY mask matching reference spatial axes')
    p.add_argument('--reference-spectrum',help='Two numeric columns in XLSX without header')
    p.add_argument('--avg-window',type=int,default=20)
    a=p.parse_args()
    if not a.input:
        from PyQt5.QtWidgets import QApplication,QMainWindow
        from .widget import CalibrationWindow
        app=QApplication.instance() or QApplication([]);window=QMainWindow();window.setWindowTitle('BSQ Generation / Reference Calibration');window.setCentralWidget(CalibrationWindow());window.resize(1400,900);window.show();app.exec_();return
    if not a.output:p.error('--output is required with --input')
    if a.spatial_bin<1:p.error('--spatial-bin must be positive')
    meta=parse_envi_header(a.input)
    if a.compute:
        if not a.roi_mask:p.error('--roi-mask required for factor computation')
        mask=np.load(a.roi_mask,allow_pickle=False).astype(bool)
        if mask.shape!=(meta['lines'],meta['samples']) or not mask.any():p.error('ROI mask must match reference spatial dimensions and be nonempty')
        if a.compute=='rowwise':
            if not a.reference_spectrum or a.avg_window<1 or meta['wavelength'] is None:p.error('Rowwise calibration requires reference spectrum, wavelengths and positive window')
            result=rowwise_calculation_heavy_logic(a.input,meta,a.reference_spectrum,a.avg_window,mask)
        else:result=fluo_calculation_heavy_logic(a.input,meta,mask)
        save_factor_profile(a.output,result['factors'],meta['wavelength']);print(a.output);return
    crop=a.crop or (0,meta['samples'],0,meta['lines'])
    x0,x1,y0,y1=crop
    if not (0<=x0<x1<=meta['samples'] and 0<=y0<y1<=meta['lines']):p.error('Crop outside input dimensions')
    if min(x1-x0,y1-y0)<a.spatial_bin:p.error('Crop smaller than spatial bin')
    if a.factors and a.rowwise:p.error('Choose --factors or --rowwise')
    factors,factor_wl=load_factor_profile(a.factors or a.rowwise) if (a.factors or a.rowwise) else (None,None)
    params=dict(raw_filepath=a.input,raw_meta=meta,save_path=a.output,crop_coords=crop,is_wavelength=meta['wavelength'] is not None,band_count=meta['bands'],factors=factors if a.factors else None,rowwise_factors=factors if a.rowwise else None,factor_wavelengths=factor_wl,binning_factor=a.spatial_bin)
    if a.target_grid:
        start,stop,step=a.target_grid
        if step<=0 or stop<=start or meta['wavelength'] is None:p.error('Positive target grid and input wavelength metadata required')
        target=np.arange(start,stop,step,dtype=np.float32);wl=meta['wavelength']
        if np.any(np.diff(wl)<=0) or len(target)==0 or target[0]<wl[0] or target[-1]>wl[-1]:p.error('Target wavelengths must be within increasing input wavelength range')
        params['lam_target']=target;result=create_datacube_resampled_logic(params)
    else:result=create_datacube_logic(params)
    print(result)
if __name__=='__main__':main()
