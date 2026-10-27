"""Check Circle / Check Square: target detection and the preview buttons."""
import cv2
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from modules.cube_preview import RealtimeCubePreviewWidget
from modules.cube_preview.shape_check import detect_shape, detect_shapes, summarize_aspects

WHITE = (255, 255, 255)


def _canvas(h=240, w=320, value=0):
    return np.full((h, w, 3), value, np.uint8)


def _scene(h, w, shapes, background=0.1, blur=1.0, noise=0.02, seed=0):
    """Float RGB image like the preview worker's: shapes ``(kind, cx, cy, size, aspect, angle, value)``.

    A circle has radius ``size``, a square side ``size``; ``aspect`` stretches along x (the scan).
    Edges are anti-aliased (4 x 4 supersampling), then blurred and noised.
    """
    ss = 4
    yy, xx = np.mgrid[0:h * ss, 0:w * ss]
    xs, ys = (xx + 0.5) / ss - 0.5, (yy + 0.5) / ss - 0.5
    img = np.full((h * ss, w * ss), background, np.float64)
    for kind, cx, cy, size, aspect, angle, value in shapes:
        u, v = (xs - cx) / aspect, ys - cy
        if kind == "circle":
            inside = u * u + v * v <= size * size
        else:
            c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
            inside = (np.abs(u * c + v * s) <= size / 2) & (np.abs(-u * s + v * c) <= size / 2)
        img[inside] = value
    img = img.reshape(h, ss, w, ss).mean(axis=(1, 3))
    img = cv2.GaussianBlur(img, (0, 0), blur) + np.random.default_rng(seed).normal(0, noise, (h, w))
    return np.repeat(img[:, :, None], 3, axis=2).astype(np.float32)


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
    assert detect_shape(img, "circle") is None                     # a square is not a circle

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
    assert [r.center[0] < 200 for r in detect_shapes(img, "circle")] == [True]
    assert [r.center[0] > 200 for r in detect_shapes(img, "square")] == [True]


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


def test_finds_every_target_of_a_grid_in_reading_order():
    centres = [(70 + 120 * j * 1.08, 60 + 110 * i) for i in range(3) for j in range(4)]
    img = _scene(330, 520, [("circle", x, y, 30, 1.08, 0, 0.9) for x, y in centres])
    found = detect_shapes(img, "circle")
    assert len(found) == 12
    for r, (x, y) in zip(found, centres):                          # row by row, left to right
        assert abs(r.center[0] - x) < 0.2 and abs(r.center[1] - y) < 0.2
        assert r.aspect == pytest.approx(1.08, abs=0.004)
    summary = summarize_aspects(found)
    assert summary.count == 12 and summary.outliers == ()
    assert summary.aspect == pytest.approx(1.08, abs=0.002)


def test_small_dim_and_touching_targets_are_all_measured():
    shapes = [("circle", 80, 80, 50, 1.1, 0, 0.9),                   # large
              ("circle", 200, 60, 7, 1.1, 0, 0.9),                    # small
              ("circle", 320, 90, 30, 1.1, 0, 0.3),                   # dim (a quarter of the contrast)
              ("circle", 120, 240, 35, 1.1, 0, 0.9),                  # these two touch
              ("circle", 120 + 70 * 1.1, 240, 35, 1.1, 0, 0.9)]
    found = detect_shapes(_scene(320, 420, shapes), "circle")
    assert len(found) == 5
    for r in found:
        assert r.aspect == pytest.approx(1.1, rel=0.006)
    assert sorted(r.touching for r in found) == [False, False, False, True, True]


def test_stretch_does_not_confuse_circles_and_squares():
    shapes = [("circle", 90, 90, 35, 1.35, 0, 0.9), ("square", 260, 90, 70, 1.35, 25, 0.9),
              ("circle", 90, 230, 35, 0.75, 0, 0.9), ("square", 260, 230, 70, 0.75, 40, 0.9)]
    img = _scene(320, 380, shapes)
    circles, squares = detect_shapes(img, "circle"), detect_shapes(img, "square")
    assert [round(r.aspect, 2) for r in circles] == [1.35, 0.75]
    assert [round(r.aspect, 2) for r in squares] == [1.35, 0.75]


def test_dead_slit_row_and_dropped_frames_do_not_cut_targets():
    img = _scene(260, 400, [("circle", 100, 130, 45, 1.05, 0, 0.9), ("circle", 280, 130, 45, 1.05, 0, 0.9)])
    img[125] = 0.0                       # dead slit pixel: a constant row through both targets
    img[:, 275:277] = 0.0                # two dropped frames: dark columns through one of them
    img[:, 350] = 1.0                    # one saturated frame
    found = detect_shapes(img, "circle")
    assert len(found) == 2
    for r in found:
        assert r.aspect == pytest.approx(1.05, abs=0.005)


def test_ring_and_specular_spot_count_once():
    img = _scene(240, 360, [("circle", 100, 120, 50, 1.0, 0, 0.9), ("circle", 100, 120, 30, 1.0, 0, 0.1),   # ring
                            ("circle", 260, 120, 45, 1.0, 0, 0.1), ("circle", 248, 108, 6, 1.0, 0, 1.0)],   # spot
                 background=0.5)
    found = detect_shapes(img, "circle")
    assert [(round(r.center[0]), round(r.center[1])) for r in found] == [(100, 120), (260, 120)]


def test_other_inputs_and_degenerate_images():
    img = _canvas()
    cv2.circle(img, (160, 120), 40, WHITE, -1)
    assert len(detect_shapes(img[:, :, 0], "circle")) == 1                         # grey
    assert len(detect_shapes(img.astype(np.float64) / 255, "circle")) == 1         # float
    for blank in (np.zeros((2, 2, 3), np.uint8), np.full((50, 60, 3), 7, np.uint8),
                  np.zeros((50, 60, 4), np.uint8), np.zeros((50,), np.uint8)):
        assert detect_shapes(blank, "circle") == []
    noise = np.random.default_rng(1).integers(0, 256, (200, 300, 3)).astype(np.uint8)
    assert detect_shapes(noise, "circle") == []


def test_summary_leaves_out_a_target_far_from_the_rest():
    class R:
        def __init__(self, aspect):
            self.aspect = aspect
    summary = summarize_aspects([R(1.050), R(1.052), R(1.049), R(1.051), R(1.12)])
    assert summary.outliers == (4,) and summary.count == 4
    assert summary.aspect == pytest.approx(1.0505, abs=1e-4)
    assert summarize_aspects([R(1.0), R(1.2)]).outliers == ()         # too few to tell
    assert summarize_aspects([]) is None


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
        assert len(label.shape_overlays()) == 1 and not preview.shape_table.isVisibleTo(preview)
        assert "Circularity" in preview.shape_label.text() and "W/H" in preview.shape_label.text()
        label.grab()                                               # paints the overlay

        preview._btn_check_square.click()
        assert "No square found" in preview.shape_label.text() and not label.shape_overlays()

        rgb = np.zeros((120, 160, 3), np.float32)
        cv2.rectangle(rgb, (50, 30), (110, 90), (1.0, 1.0, 1.0), -1)
        preview._on_rgb_ready(rgb, 120, 160)
        preview._btn_check_square.click()
        assert "Squareness" in preview.shape_label.text()

        preview._btn_clear_shape.click()
        assert not label.shape_overlays() and preview.shape_label.text() == ""
        preview._btn_check_square.click()
        preview.reset()
        assert not label.shape_overlays()
    finally:
        preview.cleanup()
        preview.close()
        app.processEvents()


def test_several_targets_fill_the_table_and_a_row_highlights_one():
    app = QApplication.instance() or QApplication([])
    preview = RealtimeCubePreviewWidget()
    try:
        label = preview.image_label
        label.resize(500, 300)
        shapes = [("circle", 60 + 110 * j, 70 + 120 * i, 30, 1.03, 0, 0.9) for i in range(2) for j in range(3)]
        shapes[4] = ("circle", 170, 190, 30, 1.25, 0, 0.9)             # one target far off the others
        preview._on_rgb_ready(_scene(260, 360, shapes), 260, 360)
        preview._btn_check_circle.click()

        assert len(label.shape_overlays()) == 6
        assert [text for _, text in label.shape_overlays()] == ["1", "2", "3", "4", "5", "6"]
        assert preview.shape_table.isVisibleTo(preview) and preview.shape_table.rowCount() == 6
        assert "6 circles" in preview.shape_label.text() and "1 left out" in preview.shape_label.text()
        assert label._shape_flagged == frozenset({4})
        assert "1.030" in preview.shape_label.text()                   # mean W/H of the other five

        preview.shape_table.selectRow(2)
        assert label._shape_selected == 2 and "W/H" in label.shape_overlays()[2][1]
        label.grab()

        preview._btn_copy_shape.click()
        lines = QApplication.clipboard().text().splitlines()
        assert lines[0].startswith("#\tkind") and len(lines) == 8
        assert lines[5].endswith("\tno") and lines[1].endswith("\tyes")

        preview._btn_clear_shape.click()
        assert preview.shape_table.rowCount() == 0 and not preview._btn_copy_shape.isEnabled()
    finally:
        preview.cleanup()
        preview.close()
        app.processEvents()
