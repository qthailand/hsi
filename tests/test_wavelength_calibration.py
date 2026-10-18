"""Scientific fit, profile integrity, bin conversion and interactive regression."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from modules.wavelength_calibration import fit_wavelength,WavelengthCalibration,row_spectrum,detect_peaks

def test_fit_and_bin_roundtrip(tmp_path):
    x=np.array([10,30,70,150]);y=.58*x+400
    p=fit_wavelength(x,y,camera_bin=2,frame_shape=(20,200))
    assert p.m==pytest.approx(.58) and p.base_m==pytest.approx(.29)
    assert p.c==pytest.approx(400) and p.r2==pytest.approx(1) and p.rms_nm<1e-10
    f=tmp_path/'profile.json';p.save(f);q=WavelengthCalibration.load(f);assert q==p
    np.testing.assert_allclose(q.base_m*2*x+q.c,y)

@pytest.mark.parametrize('x,y',[([1],[400]),([1,1],[400,500]),([1,2],[400,400]),([1,2],[500,400]),([1,2],[400,float('nan')])])
def test_invalid_fit(x,y):
    with pytest.raises(ValueError):fit_wavelength(x,y)

def test_row_and_peak():
    a=np.zeros((4,100));a[3,20:23]=[10,100,10];a[3,60:63]=[10,80,10]
    s=row_spectrum(a,999);peaks=detect_peaks(s)
    assert 21 in peaks and 61 in peaks
    assert len(row_spectrum(np.zeros((2,2)),0))==2

def test_widget_repeated_detection_and_invalidation():
    from PyQt5.QtWidgets import QApplication,QTableWidgetItem
    from modules.wavelength_calibration import WavelengthCalibrationWidget
    app=QApplication.instance() or QApplication([])
    w=WavelengthCalibrationWidget();a=np.zeros((4,100));a[:,20:23]=[10,100,10];a[:,60:63]=[10,80,10];w.set_frame(a,2)
    w.find_peaks();n=w.table.rowCount();w.find_peaks();assert w.table.rowCount()==n
    w.table.setRowCount(2)
    for i,(x,y) in enumerate([(10,406),(50,430)]):
        w.table.setItem(i,0,QTableWidgetItem(str(x)));w.table.setItem(i,1,QTableWidgetItem(str(y)))
    w.calculate();assert w.profile is not None
    w.table.item(0,1).setText('407');assert w.profile is None
    w.close()

def test_worker_calibration_reset(tmp_path):
    path=Path(__file__).resolve().parents[1]/'workers/camera_preview.py';spec=importlib.util.spec_from_file_location('wave_worker_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    config=SimpleNamespace(smile_map_folder=str(tmp_path),m=.1,c=300)
    worker=m.CameraPreviewThread(SimpleNamespace(current_bin=2),config)
    worker._binner=object();worker._binner_fitted=True
    profile=fit_wavelength([0,100],[400,458],2)
    worker.request_wavelength(profile);assert config.m==.1
    worker._apply_pending_wavelength();assert config.m==pytest.approx(.29);assert config.c==pytest.approx(400)
    assert worker._binner is None and not worker._binner_fitted


def test_smoothing_is_optional_and_adjustable():
    rng=np.random.default_rng(0);a=rng.normal(100,20,(2,200));a[1,50]=900
    raw=row_spectrum(a,1,smooth=False);np.testing.assert_array_equal(raw,a[1])
    strong=row_spectrum(a,1,window=21,polyorder=2);weak=row_spectrum(a,1,window=5,polyorder=2)
    assert strong[50]<weak[50]<raw[50]
    assert len(row_spectrum(a,1,window=8,polyorder=30))==200      # even window / large order are clamped

def _lamp_spectrum(m=.58,c=395.,extra=()):
    from modules.wavelength_calibration import HGAR_LINES
    lines=[l for l in HGAR_LINES if 410<l<880];px=[(l-c)/m for l in lines];x=np.arange(1000.)
    s=np.full(1000,5.)
    for p in px+list(extra):s+=600*np.exp(-.5*((x-p)/1.6)**2)
    return s,lines,px

def test_auto_calibration_extends_three_peaks():
    from modules.wavelength_calibration import auto_calibrate
    s,lines,px=_lamp_spectrum(extra=[333.])
    idx=(1,6,12);r=auto_calibrate(s,[px[i] for i in idx],[lines[i] for i in idx])
    assert r.sources.count('user')==3 and r.sources.count('auto')>=8
    assert r.calibration.m==pytest.approx(.58,abs=2e-3) and r.calibration.c==pytest.approx(395,abs=.5)
    assert any(abs(u-333)<1 for u in r.unmatched)                      # non-lamp peak is not forced onto a line
    for i in idx:assert any(abs(x-px[i])<1 and y==lines[i] for x,y in zip(r.pixels,r.wavelengths))

@pytest.mark.parametrize('xs,ys',[([10,20],[400,410]),([10,10,30],[400,401,420])])
def test_auto_calibration_rejects_bad_anchors(xs,ys):
    from modules.wavelength_calibration import auto_calibrate
    with pytest.raises(ValueError):auto_calibrate(_lamp_spectrum()[0],xs,ys)

def test_auto_calibration_rejects_reversed_assignment():
    from modules.wavelength_calibration import auto_calibrate
    s,lines,px=_lamp_spectrum()
    with pytest.raises(ValueError):auto_calibrate(s,[px[1],px[6],px[12]],[lines[12],lines[6],lines[1]])

def test_widget_auto_with_three_anchor_wavelengths():
    from PyQt5.QtWidgets import QApplication,QTableWidgetItem
    from modules.wavelength_calibration import WavelengthCalibrationWidget
    app=QApplication.instance() or QApplication([])
    s,lines,px=_lamp_spectrum();w=WavelengthCalibrationWidget();w.set_frame(np.tile(s,(4,1)),1);w.find_peaks()
    assert w.btn_auto.isEnabled()                                     # anchored (3+ wavelengths) or full search
    rows={i:float(w.table.item(i,0).text()) for i in range(w.table.rowCount())}
    chosen=[(i,lines[j]) for j in (1,6,12) for i,p in rows.items() if abs(p-px[j])<2]
    for i,wl in chosen:w.table.setItem(i,1,QTableWidgetItem(str(wl)))
    w.auto_calibrate()
    assert w.profile is not None and w.profile.m==pytest.approx(.58,abs=2e-3)
    assert sum(1 for i in range(w.table.rowCount()) if w.table.item(i,1) and w.table.item(i,1).text())>3
    w.smooth.setChecked(False);assert w.profile is not None          # smoothing never discards the calibration
    w.close()


def test_bin_is_read_only_and_comes_from_the_camera(tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QApplication
    from modules.wavelength_calibration import WavelengthCalibrationWidget
    app=QApplication.instance() or QApplication([])
    w=WavelengthCalibrationWidget();assert w.bin.isReadOnly()
    w.set_frame(np.zeros((6,50)),2);assert w.bin.value()==2                   # grabbed frame carries the controller's bin
    path=tmp_path/'image.npy';np.save(path,np.zeros((6,50)))
    monkeypatch.setattr('PyQt5.QtWidgets.QFileDialog.getOpenFileName',lambda *a,**k:(str(path),''))
    w.set_bin_provider(lambda:4);w.load_image();assert w.bin.value()==4       # loaded image: camera's current bin
    w.set_bin_provider(lambda:None);w.set_bin(1);w.load_image();assert w.bin.value()==1
    w.close()


def test_fit_plots_always_match_the_current_profile():
    """Editing points or a failed Calculate must not leave the previous fit/residuals on screen."""
    from unittest import mock
    from PyQt5.QtWidgets import QApplication, QTableWidgetItem
    from modules.wavelength_calibration import WavelengthCalibrationWidget
    app=QApplication.instance() or QApplication([])
    w=WavelengthCalibrationWidget();w.set_frame(np.zeros((6,100)),1)
    def fill(rows):
        w.table.setRowCount(len(rows))
        for i,(x,y) in enumerate(rows):
            w.table.setItem(i,0,QTableWidgetItem(str(x)));w.table.setItem(i,1,QTableWidgetItem(str(y)))
    def residual():
        return [np.round(i.yData,3).tolist() for i in w.residual_plot.listDataItems()]
    fill([(10,406),(50,430),(80,449)]);w.calculate();first=residual();assert first
    fill([(10,406),(50,432),(80,447)])                       # edited, not recalculated
    assert w.profile is None and residual()==[] and not w.fit_plot.listDataItems()
    w.calculate();second=residual();assert second and second!=first       # second calculation replaces the first
    with mock.patch('PyQt5.QtWidgets.QMessageBox.warning'):
        fill([(10,406),(10,430)]);w.calculate()                         # invalid: duplicate pixel
    assert w.profile is None and residual()==[] and not w.apply.isEnabled()
    w.close()


def _lines_spectrum(n=804):
    x=np.arange(float(n));truth=[(120.4,1.2),(300.7,2.0),(512.2,3.1),(700.0,0.8)]
    s=20+sum(900*np.exp(-.5*((x-p)/sg)**2) for p,sg in truth)
    return s,truth

def test_report_is_tab_separated_and_complete(tmp_path, monkeypatch):
    """Export Report writes fit, points with residuals and spectrum sections (FWHM moved to spectral_resolution)."""
    from PyQt5.QtWidgets import QApplication,QTableWidgetItem
    from modules.wavelength_calibration import WavelengthCalibrationWidget
    app=QApplication.instance() or QApplication([])
    s,truth=_lines_spectrum();w=WavelengthCalibrationWidget();w.set_frame(np.tile(s,(5,1)),2);w.find_peaks()
    w._loading=True
    for i,(p,_) in enumerate(truth):w.table.setItem(i,1,QTableWidgetItem(str(400+.58*p)))
    w._loading=False;w.calculate()
    path=tmp_path/'report.txt'
    monkeypatch.setattr('PyQt5.QtWidgets.QFileDialog.getSaveFileName',lambda *a,**k:(str(path),''))
    w.export_report()
    text=path.read_text(encoding='utf-8-sig')
    sections={}
    for block in text.strip().split('\n\n'):
        lines=[l for l in block.splitlines() if l.strip()]
        name=next((l[3:-1] for l in lines if l.startswith('# [')),None)
        if name:sections[name]=[l.split('\t') for l in lines if not l.startswith('#')]
    assert set(sections)=={'fit','points','spectrum'}
    fit=dict(sections['fit'][1:]);assert float(fit['m_nm_per_px'])==pytest.approx(w.profile.m) and fit['hardware_bin']=='2'
    points=sections['points'];assert points[0]==['pixel','wavelength_nm','fit_nm','residual_nm','residual_px'] and len(points)==1+len(truth)
    assert float(points[1][3])==pytest.approx(float(points[1][1])-float(points[1][2]))
    assert len(sections['spectrum'])==1+len(s)
    spectrum=np.array(sections['spectrum'][1:],dtype=float)                # numeric, one row per pixel
    np.testing.assert_allclose(spectrum[:,2],s)
    w.close()



AR_FILE=__import__('pathlib').Path(r"C:/Users/TSC/Downloads/OceanOptics_AR-2_wavelengths_nm.txt")

def test_line_list_formats(tmp_path):
    from modules.wavelength_calibration import load_line_list
    f=tmp_path/'list.txt'
    f.write_text("\n".join(["# argon","wavelength_nm\tintensity","912.297\t100","866.794, Ar I","","; note","922.450","912.297",""]),encoding='utf-8')
    assert load_line_list(f)==(866.794,912.297,922.45)
    (tmp_path/'bad.txt').write_text("only 500\n",encoding='utf-8')
    with pytest.raises(ValueError):load_line_list(tmp_path/'bad.txt')
    if AR_FILE.exists():
        lines=load_line_list(AR_FILE);assert len(lines)==40 and lines[0]==866.794 and lines[-1]==1704.288

def _ar_spectrum(lines,m=1.3,c=850.,n=640,seed=4):
    rng=np.random.default_rng(seed);x=np.arange(float(n))
    keep=[l for l in lines if c<l<c+m*(n-1) and rng.random()>.25]
    s=20+sum(rng.uniform(200,900)*np.exp(-.5*((x-(l-c)/m)/1.6)**2) for l in keep)
    s+=400*np.exp(-.5*((x-n*.37)/1.6)**2)+rng.normal(0,3,n)            # one stray peak
    return s,keep

@pytest.mark.parametrize('m,c,n',[(1.3,850.,640),(2.6,840.,320),(.65,880.,1280)])
def test_full_search_finds_the_calibration_without_anchors(m,c,n):
    from modules.wavelength_calibration import search_calibration
    lines=(866.794,912.297,922.45,935.422,965.779,978.45,1047.005,1067.357,1107.887,1144.183,1148.811,1156.871,1171.949,
           1211.233,1213.974,1234.339,1243.932,1248.766,1270.228,1273.342,1280.274,1295.666,1300.826,1322.811,1327.264,
           1331.321,1336.711,1350.419,1362.266,1367.855,1371.858,1382.572,1390.748,1409.364,1504.65,1517.269,1532.934,
           1598.949,1694.058,1704.288)
    s,keep=_ar_spectrum(lines,m,c,n)
    r=search_calibration(s,lines)
    assert r.calibration.m==pytest.approx(m,rel=2e-3) and r.calibration.c==pytest.approx(c,abs=1.0)
    from modules.wavelength_calibration import detect_peaks
    peaks=len(detect_peaks(s))                     # blended lines (coarse dispersion) are one peak
    assert len(r.pixels)>=.6*peaks and r.calibration.rms_nm<.35*m

def test_widget_loads_list_and_searches_without_wavelengths(tmp_path):
    from PyQt5.QtWidgets import QApplication
    from modules.wavelength_calibration import WavelengthCalibrationWidget
    app=QApplication.instance() or QApplication([])
    lines=(866.794,912.297,922.45,965.779,1047.005,1107.887,1211.233,1295.666,1350.419,1409.364,1504.65,1598.949)
    f=tmp_path/'ar.txt';f.write_text("\n".join(map(str,lines)),encoding='utf-8')
    s,_=_ar_spectrum(lines,1.3,850.,640,seed=1)
    w=WavelengthCalibrationWidget();w.set_frame(np.tile(s,(5,1)),1);w.load_line_list(str(f))
    assert 'ar.txt' in w.catalog_label.text() and w.catalog==tuple(sorted(lines))
    w.auto_calibrate()
    assert w.profile is not None and w.profile.m==pytest.approx(1.3,rel=3e-3)
    assert 'full search' in w.status.text()
    w.close()
