"""PNG export of grabbed frames and of the processing stages of one scan frame: bit depth kept,
values unscaled, metadata in a tEXt chunk."""
import threading

import cv2
import numpy as np
import pytest
from types import SimpleNamespace

import json

from modules.image import png_name, png_text, save_png


def read_png(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)


@pytest.mark.parametrize("dtype, top", [(np.uint16, 65504), (np.uint16, 4095), (np.uint8, 255)])
def test_png_keeps_dtype_and_values(tmp_path, dtype, top):
    frame = (np.arange(6 * 7).reshape(6, 7) * (top // 41)).astype(dtype)
    path = tmp_path / "ภาพ_grab.png"                         # non-ASCII path (cv2.imwrite cannot)
    save_png(path, frame)
    back = read_png(path)
    assert back.dtype == dtype and back.shape == frame.shape
    np.testing.assert_array_equal(back, frame)


@pytest.mark.parametrize("frame", [np.zeros((3, 4), np.float32), np.zeros((3, 4, 3), np.uint8)])
def test_png_refuses_what_it_cannot_store_exactly(tmp_path, frame):
    with pytest.raises(ValueError):
        save_png(tmp_path / "x.png", frame)
    assert not (tmp_path / "x.png").exists()


def test_png_text_chunks_are_kept_and_the_image_is_unchanged(tmp_path):
    frame = (np.arange(5 * 9).reshape(5, 9) * 1000).astype(np.uint16)
    info = {"stage": "valid", "rows_left_out": [3, 4]}
    path = tmp_path / "stage.png"
    save_png(path, frame, text={"Software": "HSI System", "hsi": json.dumps(info)})
    np.testing.assert_array_equal(read_png(path), frame)       # decoders still read it
    text = png_text(path)
    assert text["Software"] == "HSI System" and json.loads(text["hsi"]) == info


def test_png_name():
    name = png_name("scan", 2, ("dark", "smile"))
    assert name.startswith("scan_bin2_") and name.endswith("_dark_smile.png")
    assert png_name("smile").startswith("smile_2") and "bin" not in png_name("smile", None)


# ───────────────────────────── scan: processing stages of one frame ─────────────────────────────

from hsi_fakes import FakeCamera, W, H  # noqa: E402
from modules.acquisition import AcquisitionEngine, AcquisitionSink  # noqa: E402
from modules.frame_exchange import FRAME_STAGES  # noqa: E402


class StageCollector(AcquisitionSink):
    def __init__(self):
        self.frames, self.processed = [], []

    def frame(self, frame, fps, timestamp): self.frames.append(frame)
    def frame_stages(self, packet): self.processed.append(packet)


def wait_for(condition, timeout=20.0):
    import time
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


def _engine(tmp_path):
    camera = FakeCamera()
    config = SimpleNamespace(smile_map_folder=str(tmp_path), preprocessing_backend='cpu', m=.5, c=400., resolution=2)
    sink = StageCollector()
    eng = AcquisitionEngine(camera, config, sink=sink, profile=False)
    thread = threading.Thread(target=eng.run_until_stopped, daemon=True)
    thread.start()
    return camera, sink, eng, thread


def test_every_stage_comes_from_one_frame(tmp_path):
    camera, sink, eng, thread = _engine(tmp_path)
    try:
        assert wait_for(lambda: sink.frames)
        assert not sink.processed                                # nothing copied unless requested
        dark = np.zeros((H, W), np.uint16)
        dark[:, W // 2:] = 5
        camera.set_dark_reference(dark, None)
        camera.set_dark_enabled(True)

        eng.request_frame_stages(FRAME_STAGES)
        assert wait_for(lambda: sink.processed)
        packet = sink.processed[0]
        assert tuple(packet.images) == FRAME_STAGES              # pipeline order
        raw, darked, binned = packet.images['raw'], packet.images['dark'], packet.images['binned']
        assert raw.shape == darked.shape == (H, W) and raw.dtype == np.uint16
        assert packet.dark_applied and not packet.smile_applied
        diff = raw.astype(int) - darked.astype(int)
        assert (diff[:, -1] == np.minimum(5, raw[:, -1])).all() and (diff[:, 0] == 0).all()
        # no smile model: the geometry stages are the dark stage, nothing is cropped
        np.testing.assert_array_equal(packet.images['geometry'], darked)
        np.testing.assert_array_equal(packet.images['valid'], darked)
        assert packet.valid_rows == (0, 0) and packet.valid_columns == (0, 0)
        assert binned.shape[0] == H and binned.shape[1] != W     # the recorded line is binned
        assert len(packet.wavelengths_nm) == binned.shape[1]
        assert all(not image.flags.writeable for image in packet.images.values())

        eng.request_frame_stages(['binned'])
        assert wait_for(lambda: len(sink.processed) == 2)
        assert tuple(sink.processed[1].images) == ('binned',)    # only what was asked
        assert len(sink.processed) == 2                          # one packet per request
    finally:
        eng.request_stop(); thread.join(5)


def test_unknown_stage_is_refused(tmp_path):
    eng = AcquisitionEngine(FakeCamera(), SimpleNamespace(smile_map_folder=str(tmp_path), m=.5, c=400.,
                                                          resolution=2), sink=StageCollector(), profile=False)
    with pytest.raises(ValueError):
        eng.request_frame_stages(['smile'])
    with pytest.raises(ValueError):
        eng.request_frame_stages([])


def test_valid_area_leaves_out_the_keystone_rows_and_smile_columns(tmp_path):
    sink = StageCollector()
    eng = AcquisitionEngine(FakeCamera(), SimpleNamespace(smile_map_folder=str(tmp_path), m=.5, c=400.,
                                                          resolution=2), sink=sink, profile=False)
    calibration = SimpleNamespace(has_keystone=True, row_margin=(3, 2), margin=(4, 6), image_shape=(20, 30))
    eng._active_calibration = lambda: calibration
    raw = np.arange(20 * 30, dtype=np.uint16).reshape(20, 30)
    corrected = raw + 1
    eng._publish_frame_stages(frozenset(FRAME_STAGES), raw, raw, corrected, corrected[3:18, :8],
                              calibration, SimpleNamespace(lambda_out=np.arange(8.)))
    packet = sink.processed[0]
    assert packet.smile_applied and packet.keystone_applied
    assert packet.valid_rows == (3, 2) and packet.valid_columns == (4, 6)
    np.testing.assert_array_equal(packet.images['valid'], corrected[3:18, 4:24])
    np.testing.assert_array_equal(packet.images['geometry'], corrected)


# ───────────────────────────── scan window: Save frame drop-down ─────────────────────────────

@pytest.fixture
def scan_window(app):
    import config as app_config
    from windows.scan_window import ScanWindow
    window = ScanWindow(app_config.AppConfig())
    yield window
    window.close()


def _packet(images, **kw):
    from modules.frame_exchange import FrameStages
    base = dict(timestamp=0.0, camera_bin=2, raw_inverted=False, dark_applied=True, smile_applied=True,
                keystone_applied=True, valid_rows=(3, 2), valid_columns=(4, 6), wavelengths_nm=None)
    base.update(kw)
    return FrameStages(images, **base)


def test_menu_offers_the_geometry_stages_only_with_a_model(scan_window):
    w = scan_window
    w._update_save_frame_menu()
    assert not w._save_frame_actions['geometry'].isEnabled()
    assert 'geometry' not in w._save_frame_stage_keys()
    w._geometry = dict(model=True, keystone_available=True, smile=True, keystone=True)
    w._update_save_frame_menu()
    assert w._save_frame_actions['valid'].isEnabled()
    assert w._save_frame_actions['valid'].text() == 'Smile + keystone corrected, valid area'
    assert w._save_frame_stage_keys() == FRAME_STAGES


def test_one_stage_is_saved_with_its_metadata(scan_window, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog
    target = tmp_path / "one.png"
    asked = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: (asked.append(a[2]) or str(target), "PNG (*.png)"))
    image = np.full((15, 20), 1234, np.uint16)
    scan_window.save_frame_stages(_packet({'valid': image}))
    assert asked[0].endswith('_smile-keystone_valid.png')
    np.testing.assert_array_equal(read_png(target), image)
    info = json.loads(png_text(target)['hsi'])
    assert info['stage'] == 'valid' and info['rows_left_out'] == [3, 2] and info['columns_left_out'] == [4, 6]
    assert info['dark_applied'] and info['keystone_applied'] and info['camera_bin'] == 2


def test_all_stages_share_one_name(scan_window, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "set.png"), ""))
    raw = np.full((4, 6), 9, np.uint16)
    binned = np.full((4, 3), 7, np.uint16)
    scan_window.save_frame_stages(_packet({'raw': raw, 'dark': raw, 'binned': binned}, smile_applied=False,
                                          keystone_applied=False, wavelengths_nm=np.array([450., 550., 650.])))
    assert sorted(p.name for p in tmp_path.iterdir()) == ['set_binned.png', 'set_dark.png', 'set_raw.png']
    raw_info = json.loads(png_text(tmp_path / 'set_raw.png')['hsi'])
    assert not raw_info['dark_applied']                          # raw is before the dark subtraction
    assert json.loads(png_text(tmp_path / 'set_binned.png')['hsi'])['wavelengths_nm'] == [450., 550., 650.]


# ───────────────────────────── module widgets: Export PNG ─────────────────────────────

@pytest.fixture
def app():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _smile():
    from modules.smile import SmileCorrectionWidget
    return SmileCorrectionWidget()


def _wavelength():
    from modules.wavelength_calibration import WavelengthCalibrationWidget
    return WavelengthCalibrationWidget()


def _fwhm():
    from modules.spectral_resolution import SpectralResolutionWidget
    return SpectralResolutionWidget()


@pytest.mark.parametrize("make, export", [(_smile, "_on_export_png"), (_wavelength, "export_png"),
                                          (_fwhm, "_on_export_png")])
def test_widget_exports_the_frame_as_received(app, tmp_path, monkeypatch, make, export):
    from PyQt5.QtWidgets import QFileDialog
    widget = make()
    assert not widget.btn_export_png.isEnabled()
    frame = (np.arange(40 * 60).reshape(40, 60) * 27).astype(np.uint16)
    widget.set_frame(frame)
    assert widget.btn_export_png.isEnabled()
    target = tmp_path / "grab.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "PNG (*.png)"))
    getattr(widget, export)()
    back = read_png(target)
    assert back.dtype == np.uint16
    np.testing.assert_array_equal(back, frame)
    widget.close()
