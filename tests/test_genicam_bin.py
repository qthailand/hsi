"""GenICam bin switching: ROI-limited cameras, refused bins, and not leaving a half-configured camera."""
import importlib
import sys
import types
import pytest

class Node:
    def __init__(self,cam,name,value,mn=None,mx=None):self.cam,self.name,self._v,self.min,self._max=cam,name,value,mn,mx
    @property
    def max(self):return self._max() if callable(self._max) else self._max
    @property
    def value(self):return self._v
    @value.setter
    def value(self,v):self.cam.write(self.name,v);self._v=v

class Cam:
    def __init__(self,reject=None):
        self.bin=1;self.reject=reject;self.log=[];n={}
        n['BinningHorizontal']=Node(self,'BinningHorizontal',1,1,4);n['BinningVertical']=Node(self,'BinningVertical',1,1,4)
        n['Width']=Node(self,'Width',1600,8,lambda:1600//self.bin);n['Height']=Node(self,'Height',1200,8,lambda:1200//self.bin)
        n['OffsetX']=Node(self,'OffsetX',0,0,0);n['OffsetY']=Node(self,'OffsetY',0,0,0);self.n=n
    def write(self,name,v):
        if name.startswith('Binning'):
            if v==self.reject:raise RuntimeError('bin not supported')
            if v>2 and self.n['Width'].value>1600//v:raise RuntimeError('Width exceeds max')   # ROI-limited behaviour
            self.bin=v
            for k in ('Width','Height'):self.n[k]._v=min(self.n[k]._v,self.n[k].max)
        elif name in ('Width','Height') and v>self.n[name].max:raise RuntimeError('out of range')
        self.log.append((name,v))
    def __getattr__(self,k):
        if k in ('n','log','bin','reject'):raise AttributeError(k)
        try:return self.n[k]
        except KeyError:raise AttributeError(k)

@pytest.fixture
def backend_class(monkeypatch):
    harvesters=types.ModuleType('harvesters');core=types.ModuleType('harvesters.core');core.Harvester=object
    monkeypatch.setitem(sys.modules,'harvesters',harvesters);monkeypatch.setitem(sys.modules,'harvesters.core',core)
    monkeypatch.delitem(sys.modules,'modules.camera.genicam',raising=False)
    cls=importlib.import_module('modules.camera.genicam').GenICamBackend
    yield cls
    sys.modules.pop('modules.camera.genicam',None)

def make(cls,cam):
    b=cls();b._nm=cam;b._refresh_info();return b

def test_plain_bin_change_leaves_roi_alone(backend_class):
    cam=Cam();b=make(backend_class,cam);b.set_bin(2)
    assert (b.current_bin,b.info.width,b.info.height)==(2,800,600)
    assert not any(name in ('Width','Height') for name,_ in cam.log)

def test_roi_limited_camera_is_retried_with_reset_roi(backend_class):
    cam=Cam();b=make(backend_class,cam);b.set_bin(3)
    assert (b.current_bin,b.info.width,b.info.height)==(3,533,400)

def test_refused_bin_keeps_previous_bin_and_full_roi(backend_class):
    cam=Cam(reject=4);b=make(backend_class,cam);b.set_bin(2)
    with pytest.raises(RuntimeError,match='bin 4'):b.set_bin(4)
    assert (b.current_bin,cam.bin,b.info.width,b.info.height)==(2,2,800,600)
