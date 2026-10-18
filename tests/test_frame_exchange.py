"""One-shot snapshot ownership, delivery and post-orientation placement."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
from modules.frame_exchange import FrameExchange

@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])

def load_worker():
    path=Path(__file__).resolve().parents[1]/'workers/camera_preview.py'
    spec=importlib.util.spec_from_file_location('snapshot_worker_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.CameraPreviewThread

def test_one_copy_only_on_request(app,tmp_path):
    cls=load_worker()
    backend=SimpleNamespace(current_bin=2,_dark_mean=None,_dark_enabled=True)
    worker=cls(backend,SimpleNamespace(smile_map_folder=str(tmp_path)))
    received=[];worker.snapshot_ready.connect(received.append)
    f=np.arange(20,dtype=np.uint16).reshape(4,5)[::-1].copy()
    worker._publish_snapshot_if_requested(f);assert not received
    worker.request_snapshot(7);worker._publish_snapshot_if_requested(f)
    worker._publish_snapshot_if_requested(f);assert len(received)==1
    packet=received[0];assert packet.request_id==7
    assert packet.stage=='post_dark_hot_pixel_orientation'
    assert not packet.dark_applied and packet.dark_enabled
    before=f.copy();f[:]=0
    np.testing.assert_array_equal(packet.image,before)
    assert not packet.image.flags.writeable

def test_exchange_routes_and_timeout(app):
    hub=FrameExchange();requests=[];errors=[]
    hub.capture_requested.connect(requests.append);hub.failed.connect(lambda i,m:errors.append((i,m)))
    token=hub.request();app.processEvents();assert requests==[token]
    with pytest.raises(RuntimeError):hub.request()
    hub._on_timeout();assert errors[0][0]==token
    token2=hub.request();assert token2!=token
    from modules.frame_exchange import FrameSnapshot
    stale=FrameSnapshot(token,np.zeros((2,2)),0,1,False,False,False)
    hub.publish(stale);assert hub.latest is None
    fresh=FrameSnapshot(token2,np.zeros((2,2)),0,1,False,False,False)
    hub.publish(fresh);assert hub.latest is fresh

def test_widget_receives_orientation_once(app):
    from modules.smile import SmileCorrectionWidget
    from modules.frame_exchange import FrameSnapshot
    w=SmileCorrectionWidget();w.set_grab_available(True);w.begin_grab(1)
    f=np.arange(20,dtype=np.uint16).reshape(4,5)
    packet=FrameSnapshot(1,f,0,2,True,True,False)
    captured=[];w.set_frame=lambda image:captured.append(image)
    w.accept_snapshot(packet);assert captured[0] is f
    assert w.btn_grab.isEnabled();w.close()

def test_worker_snapshot_position():
    """The grabbed frame is oriented but not yet Smile-corrected (behaviour, not source order)."""
    from modules.preprocessing import PreprocessingRuntime
    from modules.dark import DarkProcessor
    class Calibration:
        interp='nearest';image_shape=(4,5)
        def apply_native(self,image):return image+1          # stands in for Smile
    frame=np.arange(20,dtype=np.uint16).reshape(4,5)
    output,grabbed=PreprocessingRuntime('cpu').process(frame,DarkProcessor(),calibration=Calibration(),inverted=True,snapshot=True)
    np.testing.assert_array_equal(grabbed,frame[::-1])
    np.testing.assert_array_equal(output,frame[::-1]+1)
