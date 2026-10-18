"""Bin a saved (frames, spatial, spectral) uint16 NPY without Qt."""
# Allow running this file directly (IDE 'run file', runfile): make the project importable.
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "modules.spectral_processing"
import argparse
import numpy as np
from . import LambdaBinner

def main():
    """Parse the command line, bin the NPY spectrogram and write an NPZ.

    Usage: ``python -m modules.spectral_processing INPUT.npy OUTPUT.npz --m M --c C
    [--step NM] [--threads N]``.  The output archive holds ``frames`` (uint16,
    ``(frames, spatial, n_bins)``) and ``wavelengths`` (float32 bin centres, nm).
    """
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input');p.add_argument('output')
    p.add_argument('--m',type=float,required=True);p.add_argument('--c',type=float,required=True)
    p.add_argument('--step',type=float,default=2);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();frame=np.load(a.input,allow_pickle=False)
    if frame.ndim!=3:p.error('input must have axes (frames, spatial, spectral)')
    b=LambdaBinner(a.m,a.c,a.step,a.threads).fit_from(frame)
    np.savez(a.output,frames=b.transform(frame),wavelengths=b.lambda_out)
if __name__=='__main__':main()
