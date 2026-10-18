"""Check Circle / Check Square: target detection and the preview buttons."""
import cv2
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from modules.cube_preview import RealtimeCubePreviewWidget
from modules.cube_preview.shape_check import detect_shape

WHITE = (255, 255, 255)


def _canvas(h=240, w=320, value=0):
    return np.full((h, w, 3), value, np.uint8)


@pytest.mark.parametrize("radius", [20, 40, 80])
def test_circle_scores_one_and_no_stretch(radius):
    img = _canvas()
    cv2.circle(img, (160, 120), radius, WHITE, -1)
    r = detect_shape(img, "circle")
    assert r.score > 0.98
    assert abs(r.aspect - 1.0) < 0.01
    assert abs(r.center[0] - 160) < 0.5 and abs(r.center[1] - 120) < 0.5


def test_stretched_circle_reports_scan_over_slit_ratio():
    img = _canvas()
    cv2.ellipse(img, (160, 120), (55, 50), 0, 0, 360, WHITE, -1)   # 10 % wider along scan (x)
    assert detect_shape(img, "circle").aspect == pytest.approx(1.1, abs=0.01)
    img = _canvas()
    cv2.ellipse(img, (160, 120), (50, 55), 0, 0, 360, WHITE, -1)
    assert detect_shape(img, "circle").aspect == pytest.approx(1 / 1.1, abs=0.01)


def test_square_scores_one_rectangle_less_and_rotation_does_not_matter():
    img = _canvas()
    cv2.rectangle(img, (110, 70), (190, 150), WHITE, -1)
    square = detect_shape(img, "square")
    assert square.score > 0.98 and abs(square.aspect - 1.0) < 0.01
    assert detect_shape(img, "circle").score < 0.85                # a square is not round

    img = _canvas()
    cv2.rectangle(img, (110, 70), (198, 150), WHITE, -1)          # 89 x 81
    rect = detect_shape(img, "square")
    assert rect.score < 0.93 and rect.aspect == pytest.approx(89 / 81, abs=0.01)

    img = _canvas()
    cv2.fillPoly(img, [cv2.boxPoints(((160, 120), (80, 80), 30)).astype(np.int32)], WHITE)
    rotated = detect_shape(img, "square")
    assert rotated.score > 0.95 and abs(rotated.aspect - 1.0) < 0.02


def test_picks_the_requested_shape_when_both_are_in_view_with_noise():
    img = _canvas(240, 400, 20)
    cv2.circle(img, (100, 120), 50, (240,) * 3, -1)
    cv2.rectangle(img, (250, 70), (350, 170), (240,) * 3, -1)
    img = np.clip(img + np.random.default_rng(0).normal(0, 15, img.shape), 0, 255).astype(np.uint8)
    assert detect_shape(img, "circle").center[0] < 200
    assert detect_shape(img, "square").center[0] > 200


def test_dark_target_on_bright_background_and_nothing_to_find():
    img = _canvas(value=230)
    cv2.rectangle(img, (110, 70), (190, 150), (10,) * 3, -1)
    assert detect_shape(img, "square").score > 0.98
    assert detect_shape(_canvas(), "circle") is None
    cut = _canvas()
    cv2.circle(cut, (0, 120), 60, WHITE, -1)                       # touches the border: ignored
    assert detect_shape(cut, "circle") is None
    with pytest.raises(ValueError):
        detect_shape(img, "triangle")


def test_buttons_outline_the_target_and_clear_removes_it():
    app = QApplication.instance() or QApplication([])
    preview = RealtimeCubePreviewWidget()
    try:
        label = preview.image_label
        label.resize(400, 300)
        preview._check_shape("circle")
        assert "No image" in preview.shape_label.text()

        rgb = np.zeros((120, 160, 3), np.float32)
        cv2.circle(rgb, (80, 60), 30, (1.0, 1.0, 1.0), -1)
        preview._on_rgb_ready(rgb, 120, 160)
        preview._btn_check_circle.click()
        assert label._shape_outline is not None
        assert "Circularity" in preview.shape_label.text() and "W/H" in preview.shape_label.text()
        label.grab()                                               # paints the overlay

        preview._btn_clear_shape.click()
        assert label._shape_outline is None and preview.shape_label.text() == ""

        preview._btn_check_square.click()
        assert "Squareness" in preview.shape_label.text()
        preview.reset()
        assert label._shape_outline is None
    finally:
        preview.cleanup()
        preview.close()
        app.processEvents()
