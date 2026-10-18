"""Inspect ENVI, or export an NPY cube without importing the HSI GUI."""
# Allow running this file directly (IDE 'run file', runfile): make the project importable.
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "modules.cube_io"
import argparse
import json
import numpy as np
from . import load_envi_cube_for_viewer

def main():
    """Command-line entry point.

    ``python -m modules.cube_io INPUT`` (default ``--format inspect``) prints the
    shape, dtype and wavelength count of an ENVI cube as JSON.  With
    ``--format envi|geotiff|spectrogram|hdf5 --output OUT`` a 3-D ``.npy`` file
    is exported; ``envi``/``geotiff`` additionally need ``--wavelengths W.npy``
    (1-D, one value per band-first axis 0) and accept ``--m``, ``--c``,
    ``--bin``; ``spectrogram`` accepts ``--m``, ``--c``, ``--bin``, ``--pixel-size``.
    """
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input')
    p.add_argument('--format',choices=['inspect','envi','geotiff','spectrogram','hdf5'],default='inspect')
    p.add_argument('--output');p.add_argument('--wavelengths',help='1-D NPY for band-first ENVI/GeoTIFF export')
    p.add_argument('--m',type=float,default=.2903);p.add_argument('--c',type=float,default=399.791)
    p.add_argument('--bin',type=int,default=1);p.add_argument('--pixel-size',type=float,default=.1)
    a=p.parse_args()
    if a.format=='inspect':
        cube,wl=load_envi_cube_for_viewer(a.input)
        print(json.dumps({'shape':list(cube.shape),'dtype':str(cube.dtype),'wavelength_count':None if wl is None else len(wl)},indent=2));return
    if not a.output:p.error('--output is required for export')
    data=np.load(a.input,allow_pickle=False,mmap_mode='r')
    if data.ndim!=3:p.error('Input must be a 3-D array')
    if a.format=='hdf5':
        from .hdf5 import save_hdf5
        save_hdf5(a.output,data)
    elif a.format=='spectrogram':
        from .writers import SpectrogramWriter
        SpectrogramWriter(save_path=a.output,config={'m':a.m,'c':a.c},spectrogram=data,pixel_size=a.pixel_size,camera_bin=a.bin).run()
    else:
        if not a.wavelengths:p.error('--wavelengths required for ENVI/GeoTIFF')
        wl=np.load(a.wavelengths,allow_pickle=False)
        if wl.ndim!=1 or len(wl)!=data.shape[0]:p.error('Wavelength count must match band-first cube axis 0')
        from .writers import HypercubeWriter,GeoTIFFWriter
        writer=HypercubeWriter if a.format=='envi' else GeoTIFFWriter
        writer(save_path=a.output,config={'m':a.m,'c':a.c},hypercube=data,linear_spectrum=wl,camera_bin=a.bin).run()
    print(a.output)
if __name__=='__main__':main()
