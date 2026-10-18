"""Apply a saved dark reference to a raw 2-D NPY frame without Qt."""
# Allow running this file directly (IDE 'run file', runfile): make the project importable.
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "modules.dark"
import argparse
import numpy as np
from . import DarkProcessor

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input');p.add_argument('reference');p.add_argument('output')
    a=p.parse_args();f=np.load(a.input,allow_pickle=False)
    with np.load(a.reference,allow_pickle=False) as ref:
        mean=ref['mean'];hot=ref['hot_pixel_map']
    if f.ndim!=2 or f.shape!=mean.shape or f.shape!=hot.shape:p.error('frame and reference must have identical 2-D shapes')
    d=DarkProcessor();d.set_dark_reference(mean,hot)
    np.save(a.output,d._apply_hot_pixel(d._apply_dark_subtract(f)))
if __name__=='__main__':main()
