"""Standalone GUI, or headless fitting from two-column CSV."""
# Allow running this file directly (IDE 'run file', runfile): make the project importable.
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "modules.wavelength_calibration"
import argparse
from .core import fit_wavelength

def main():
    """Entry point: fit from a CSV (``--points``) or open the GUI.

    With ``--points`` the CSV (no header, columns ``pixel,wavelength_nm``) is fitted at
    ``--bin`` and the profile JSON is written to ``--output``; otherwise the
    calibration widget opens in its own window.
    """
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--points',help='CSV pixel,wavelength; no header');p.add_argument('--output');p.add_argument('--bin',type=int,default=1);a=p.parse_args()
    if a.points:
        import numpy as np
        if not a.output:p.error('--output required with --points')
        data=np.loadtxt(a.points,delimiter=',',ndmin=2)
        if data.shape[1]!=2:p.error('CSV must have two columns')
        profile=fit_wavelength(data[:,0],data[:,1],a.bin);profile.save(a.output);print(f'm={profile.m}, c={profile.c}, R2={profile.r2}, RMS={profile.rms_nm} nm');return
    from modules._support import launch
    from .widget import WavelengthCalibrationWidget
    launch(WavelengthCalibrationWidget,'Wavelength Calibration',(1200,900))
if __name__=='__main__':main()
