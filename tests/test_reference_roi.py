import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QPointF
from modules.reference_roi import ReferenceROIProfile, ReferenceROIStore, select_rgb, compose_preview

CONTEXT=dict(camera_backend='poa',camera_id=0,camera_index=0,camera_bin=1,raw_inverted=True,spatial_height=120)

def test_persistence_and_deactivation(tmp_path):
    store=ReferenceROIStore(tmp_path)
    saved=store.save_active(ReferenceROIProfile('White reference',20,40,CONTEXT))
    assert ReferenceROIStore(tmp_path).load_active()==saved
    store.clear_active();assert store.load_active() is None
    assert (store.folder/(saved.profile_id+'.json')).exists()

@pytest.mark.parametrize('key,value',[('camera_bin',2),('raw_inverted',False),('spatial_height',60),('camera_id',1)])
def test_compatibility(key,value):
    profile=ReferenceROIProfile('ref',20,40,CONTEXT)
    assert profile.matches(CONTEXT)
    assert not profile.matches(dict(CONTEXT,**{key:value}))

@pytest.mark.parametrize('start,end',[(-1,40),(40,40),(20,121)])
def test_invalid_rows(start,end):
    with pytest.raises(ValueError):ReferenceROIProfile('ref',start,end,CONTEXT).validate()

def test_rgb_keeps_spatial_rows_and_source():
    frame=np.arange(120*5,dtype=np.uint16).reshape(120,5);original=frame.copy()
    columns=[select_rgb(frame,np.array([400,465,555,645,800])) for _ in range(7)]
    assert np.array_equal(columns[0],frame[:,[3,2,1]])
    image=compose_preview(columns)
    assert image.shape==(120,7,3) and image.dtype==np.uint8
    assert np.array_equal(frame,original)
    with pytest.raises(ValueError):compose_preview([])

def test_widget_row_mapping_and_busy():
    from modules.reference_roi.widget import ReferenceROIWidget
    app=QApplication.instance() or QApplication([])
    widget=ReferenceROIWidget();widget.resize(900,650)
    image=np.zeros((120,400,3),dtype=np.uint8)
    image[:]=[30,80,30];image[20:40]=220
    widget.accept_preview(image,CONTEXT);widget.show();app.processEvents()
    rect=widget.canvas._rect
    assert widget.canvas._row(QPointF(rect.center().x(),rect.top()+30.5*rect.height()/120))==30
    widget.canvas._anchor=20
    widget.canvas._set_drag(QPointF(rect.center().x(),rect.top()+39.5*rect.height()/120))
    assert (widget.start.value(),widget.end.value())==(20,40)
    assert widget.save_button.isEnabled()
    widget.set_busy(True);assert not widget.canvas.isEnabled() and not widget.save_button.isEnabled()
    widget.set_busy(False);widget.grab().save('/tmp/reference_roi_widget.png');widget.close()

def controller():
    """The real ScanController (no __init__: no camera/window), driven with fakes."""
    from controllers.scan_controller import ScanController
    from modules.acquisition import ScanSession
    class Controller(ScanController):
        def __init__(self):
            self.session=ScanSession();self.reference_preview_ready=None;self.reference_preview_failed=None
    ns={'Controller':Controller}
    c=ns['Controller']();c.PREVIEW='PREVIEW';c.enable_cube_preview=False;c.SCANNING='SCANNING';c.state=c.PREVIEW;c.stage=None
    c.dark_lib=SimpleNamespace(is_collecting=False);
    c.preview_thread=SimpleNamespace(set_scanning=lambda _:None,set_keep_spectrogram=lambda _:None,set_recording=lambda *a, **kw:None,take_recorded_frames=lambda *window, **kw:[],take_recorded_raw=lambda:[],live_recorded_frames=lambda:[])
    c._set_state=lambda state:setattr(c,'state',state)
    c.config=SimpleNamespace(**{k:v for k,v in CONTEXT.items() if k not in ('spatial_height','camera_bin')})
    c.backend=SimpleNamespace(current_bin=1);c.win=SimpleNamespace(set_status=lambda _:None,start_scan_clock=lambda:None,set_recorded_count=lambda _:None)
    c.ready=[];c.failed=[];c.reference_preview_ready=lambda *args:c.ready.append(args);c.reference_preview_failed=lambda *args:c.failed.append(args)
    return c

def test_preview_busy_and_failure_recovery(monkeypatch):
    import controllers.scan_controller as module
    monkeypatch.setattr(module.QMessageBox,'critical',lambda *a:None)
    c=controller();c.stage=object()
    with pytest.raises(RuntimeError):c.start_reference_preview(1)
    assert c.session.reference_preview_id is None
    c.stage=None;c.start_scan=lambda:setattr(c,'state',c.SCANNING)
    c.start_reference_preview(2);c.on_stage_error('open failed')
    assert c.failed==[(2,'open failed')] and c.session.reference_preview_id is None and c.state==c.PREVIEW

def test_start_scan_waits_for_a_bin_change_still_being_applied():
    # Started now, the worker would apply the bin mid-scan: two frame widths in one cube.
    c=controller();statuses=[];c.win=SimpleNamespace(set_status=statuses.append)
    c.preview_thread.bin_pending=True
    c.start_scan()
    assert c.state==c.PREVIEW and 'Bin change' in statuses[-1] and 'press Start again' in statuses[-1]
    with pytest.raises(RuntimeError,match='could not be started'):c.start_reference_preview(5)   # the caller learns at once
    assert c.session.reference_preview_id is None

def test_preview_finalization_without_cube_and_empty_failure():
    c=controller();c.state=c.SCANNING;c.session.reference_preview_id=7
    c.session.reference_columns=[np.ones((120,3),np.float32)]*4
    c.finish_scan();assert c.ready[0][0]==7 and c.ready[0][1].shape==(120,4,3)
    assert c.session.reference_preview_id is None and c.session.reference_columns==[]
    c.state=c.SCANNING;c.session.reference_preview_id=8;c.finish_scan()
    assert c.failed[0][0]==8

def test_preview_has_one_column_per_recorded_frame():
    c=controller();c.state=c.SCANNING;c.session.reference_preview_id=9;c.current_wavelengths=None
    frames=[np.full((120,10),i,np.uint16) for i in range(6)]
    c.preview_thread.take_recorded_frames=lambda *window, **kw:frames
    c.finish_scan();assert c.ready[0][0]==9 and c.ready[0][1].shape==(120,6,3)

def test_overlay_compatibility_and_cached_update():
    c=controller();bands=[];c.viewer=SimpleNamespace(image=SimpleNamespace(set_reference_roi=bands.append))
    c.set_reference_roi_profile(ReferenceROIProfile('ref',20,40,CONTEXT))
    c._refresh_reference_overlay(120);c._refresh_reference_overlay(120)
    assert bands==[None,(20,40,120)]
    c.backend.current_bin=2;c._refresh_reference_overlay(120)
    assert bands[-1] is None

def test_live_overlay_does_not_modify_pixels():
    from widgets.live_image_canvas import LiveImageCanvas
    app=QApplication.instance() or QApplication([])
    canvas=LiveImageCanvas();canvas.resize(600,360)
    frame=np.full((120,100),4096,np.uint16);original=frame.copy()
    canvas.set_array(frame);canvas.set_reference_roi((20,40,120));canvas.show();app.processEvents()
    image=canvas.grab().toImage()
    # Away from focus/selected-row outlines, the reference area is yellow.
    band=image.pixelColor(300,90);outside=image.pixelColor(300,30)
    assert band.red()>outside.red() and band.green()>outside.green()
    assert np.array_equal(frame,original)
    canvas.set_reference_roi(None);app.processEvents();canvas.close()

def test_cancel_discards_pending_and_stops_existing_stage():
    from windows.integrations import ReferenceROIIntegration
    messages=[];stops=[];widget=SimpleNamespace(fail_preview=messages.append)
    integration=ReferenceROIIntegration.__new__(ReferenceROIIntegration)      # no store/config needed for this path
    integration.host=SimpleNamespace(scan_controller=SimpleNamespace(stop_scan_manually=lambda:stops.append(True)))
    integration._pending=(5,widget)
    integration._cancel_preview(widget)
    assert integration._pending is None and stops==[True] and 'cancelled' in messages[0]
