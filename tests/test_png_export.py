"""PNG export of grabbed and preprocessed frames: bit depth kept, values unscaled."""
import threading

import cv2
import numpy as np
import pytest
from types import SimpleNamespace

from modules.image import png_name, save_png


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


def test_png_name():
    name = png_name("scan", 2, ("dark", "smile"))
    assert name.startswith("scan_bin2_") and name.endswith("_dark_smile.png")
    assert png_name("smile").startswith("smile_2") and "bin" not in png_name("smile", None)


# ───────────────────────────── scan: preprocessed frame before binning ─────────────────────────────

from hsi_fakes import FakeCamera, W, H  # noqa: E402
from modules.acquisition import AcquisitionEngine, AcquisitionSink  # noqa: E402


class ProcessedCollector(AcquisitionSink):
    def __init__(self):
        self.frames, self.processed = [], []

    def frame(self, frame, fps, timestamp): self.frames.append(frame)
    def processed_frame(self, packet): self.processed.append(packet)


def wait_for(condition, timeout=20.0):
    import time
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_processed_frame_is_after_dark_and_before_binning(tmp_path):
    camera = FakeCamera()
    config = SimpleNamespace(smile_map_folder=str(tmp_path), preprocessing_backend='cpu', m=.5, c=400., resolution=2)
    sink = ProcessedCollector()
    eng = AcquisitionEngine(camera, config, sink=sink, profile=False)
    thread = threading.Thread(target=eng.run_until_stopped, daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: sink.frames)
        assert not sink.processed                                # nothing copied unless requested

        eng.request_processed_frame()
        assert wait_for(lambda: sink.processed)
        plain = sink.processed[0]
        assert plain.image.shape == (H, W) and plain.image.dtype == np.uint16   # full sensor width
        assert sink.frames[-1].shape[1] != W                     # the display frame is binned
        assert not plain.dark_applied and plain.stage == 'post_dark_hot_pixel_orientation'
        assert not plain.image.flags.writeable

        dark = np.zeros((H, W), np.uint16)
        dark[:, W // 2:] = 5
        camera.set_dark_reference(dark, None)
        camera.set_dark_enabled(True)
        eng.request_processed_frame()
        assert wait_for(lambda: len(sink.processed) == 2)
        darked = sink.processed[1]
        assert darked.dark_applied
        image = darked.image.astype(int)
        assert (image[:, 0] - image[:, -1] == 5).all()           # dark subtracted per pixel
        assert len(sink.processed) == 2                          # one frame per request
    finally:
        eng.request_stop(); thread.join(5)


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
