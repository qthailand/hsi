"""Live cube preview: every scan starts fitted to the canvas, on a white background."""
import numpy as np
from PyQt5.QtWidgets import QApplication

from modules.cube_preview import CenteredPixmapLabel, RealtimeCubePreviewWidget


def _rgb(h, w):
    return np.random.default_rng(0).random((h, w, 3)).astype(np.float32)


def test_first_image_after_reset_is_fitted():
    app = QApplication.instance() or QApplication([])
    preview = RealtimeCubePreviewWidget()
    try:
        label = preview.image_label
        label.resize(400, 300)
        preview._on_rgb_ready(_rgb(64, 10), 64, 10)
        label.zoom_in(); label.zoom_in()                  # zoomed during the previous scan
        label._zp_pan_x, label._zp_pan_y = 120.0, -40.0
        preview.reset()
        preview._on_rgb_ready(_rgb(64, 1), 64, 1)          # first frame of the next scan
        assert label.zoom_percent == 100.0 and (label._zp_pan_x, label._zp_pan_y) == (0.0, 0.0)
        label.zoom_in()
        preview._on_rgb_ready(_rgb(64, 2), 64, 2)          # later frames keep the user's zoom
        assert label.zoom_percent > 100.0
    finally:
        preview.cleanup()
        preview.close()
        app.processEvents()


def test_live_preview_canvas_is_white_and_viewer_default_stays_dark():
    app = QApplication.instance() or QApplication([])
    preview = RealtimeCubePreviewWidget()
    try:
        preview.image_label.resize(50, 40)
        assert preview.image_label.grab().toImage().pixelColor(5, 5).name() == "#ffffff"
        plain = CenteredPixmapLabel()
        plain.resize(50, 40)
        assert plain.grab().toImage().pixelColor(5, 5).name() == "#0d0d0d"
    finally:
        preview.cleanup()
        preview.close()
        app.processEvents()
