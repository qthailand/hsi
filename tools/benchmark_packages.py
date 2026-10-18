"""Compare original and packaged kernels on the same PC, outside GUI/acquisition.

Supply an extracted original hsi directory. Timings exclude fetch, GUI, disk,
and smile. This is a relative regression check, not an end-to-end FPS claim.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modules.dark import DarkProcessor
from modules.image import apply_raw_orientation
from modules.spectral_processing import LambdaBinner

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy',type=Path,required=True)
    p.add_argument('--height',type=int,default=1024);p.add_argument('--width',type=int,default=2048)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--repeat',type=int,default=300)
    p.add_argument('--m',type=float,default=.5806);p.add_argument('--c',type=float,default=399.791);p.add_argument('--step',type=float,default=2)
    args=p.parse_args()
    if min(args.height,args.width,args.threads,args.repeat)<=0:p.error('dimensions, threads and repeat must be positive')
    oldbin=load(args.legacy/'BiningTools.py','baseline_binning').LambdaBinner
    oldbase=load(args.legacy/'workers/backends/base.py','baseline_backend').AbstractCameraBackend
    oldimage=load(args.legacy/'utils/image.py','baseline_image')
    frame=np.random.default_rng(7).integers(0,60000,(args.height,args.width),dtype=np.uint16)
    dark=np.full_like(frame,50);hot=np.zeros_like(frame,np.uint8);hot[::71,::97]=255
    old=object.__new__(type('LegacyProcessor',(),{n:getattr(oldbase,n) for n in ['__init__','set_dark_reference','_precompute_hot_pixel_neighbors','_apply_dark_subtract','_apply_hot_pixel']}));old.__init__()
    new=DarkProcessor();pairs=[]
    for name,obj,cls,orient in [('legacy',old,oldbin,oldimage.apply_raw_orientation),('packaged',new,LambdaBinner,apply_raw_orientation)]:
        obj.set_dark_reference(dark,hot)
        b=cls(args.m,args.c,args.step,threads=args.threads).fit(args.width)
        def run(obj=obj,b=b,orient=orient):
            f=obj._apply_dark_subtract(frame);f=obj._apply_hot_pixel(f);f=orient(f,True)
            return b.transform(f.reshape(1,*f.shape))
        for _ in range(10):run()
        pairs.append((name,run))
    np.testing.assert_array_equal(pairs[0][1](),pairs[1][1]())
    timings={name:[] for name,_ in pairs}
    for i in range(args.repeat):
        # Alternate order to reduce systematic thermal/cache bias.
        for name,run in pairs[::1 if i%2==0 else -1]:
            t=time.perf_counter_ns();run();timings[name].append((time.perf_counter_ns()-t)/1e6)
    result={'shape':list(frame.shape),'threads':args.threads,'repeat':args.repeat,'exact_equal':True,'timing_scope':'dark + hot pixels + orientation + binning; excludes fetch, GUI, recording, smile and other diagnostics'}
    for name,v in timings.items():result[name+'_ms']={k:float(x) for k,x in zip(['median','p95','p99'],np.percentile(v,[50,95,99]))}
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
