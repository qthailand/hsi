"""``modules.saturation_mask``: each stage against its own ground truth, independent of the mask code.

Stage            Ground truth
Raw detection    the threshold on the camera level
Hot pixel        the neighbours ``DarkReference._apply_hot_pixel`` really reads (impulse per raw pixel)
Geometry         OpenCV's float32 impulse response through the intensity remap (non-zero, any sign);
                 ``'nearest'``: the model's own ``apply_native``
Crop             ``keystone_row_margin`` of the acquisition engine, and the output shape
Binning          OR of the corrected pixels over ``iL:iR`` (explicit loops)
Subset           the subset's bands equal the full binner's bands k0:k1
"""
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.compute_runtime import BackendCapability, ComputeCapabilities  # noqa: E402
from modules.dark import DarkProcessor  # noqa: E402
from modules.preprocessing import PreprocessingRuntime  # noqa: E402
from modules.preprocessing.maps import FLAGS, prepare_maps  # noqa: E402
from modules.saturation_mask import (GeometryMask, SaturationPropagator, bin_masks, crop_rows,  # noqa: E402
                                     detect_raw, hot_pixel_mask, pack_mask, tap_support, tap_weights,
                                     unpack_mask)
from modules.spectral_processing import LambdaBinner  # noqa: E402

SHAPE = (18, 40)
INTERPS = ("nearest", "linear", "cubic", "lanczos")
CPU_CAPS = ComputeCapabilities(BackendCapability(), BackendCapability())


# ---------------------------------------------------------------- models
def _maps(shape, keystone):
    """Maps whose fractional parts take every 1/32 level (in x, and in y with keystone), with
    integer shifts that push taps past every border."""
    h, w = shape
    y, x = np.indices(shape, dtype=np.float32)
    shift_x = np.round(np.sin(y / 3.0) * 2.0) - (x < 3) * 2 + (x > w - 4) * 2
    map_x = x + ((x + 5 * y) % 32) / 32.0 + shift_x
    map_y = None
    if keystone:
        shift_y = np.round(np.cos(x / 4.0) * 1.0) - (y < 2) * 2 + (y > h - 3) * 2
        map_y = (y + ((3 * x + y) % 32) / 32.0 + shift_y).astype(np.float32)
    return map_x.astype(np.float32), map_y


def smile_model(interp, keystone=False, shape=SHAPE):
    """A real ``SmileCalibration`` (linear / cubic / lanczos)."""
    from modules.smile.calibration import SmileCalibration
    map_x, map_y = _maps(shape, keystone)
    return SmileCalibration(
        map_x=map_x, margin=(2, 2), image_shape=shape, deg_y=1, deg_x=1, surface_coef=np.zeros(4),
        y_rng=(0.0, 1.0), x_rng=(0.0, 1.0), interp=interp, fit_rms=0.0, cv_rms=0.0, rms_before=0.0,
        rms_after=0.0, n_lines_used=0, n_lines_total=0, created="", backend="cpu", map_y=map_y,
        keystone={"row_margin": [2, 1]} if keystone else None)


class NearestModel:
    """A ``'nearest'`` model: ``SmileCalibration`` has none; the pipeline calls ``apply_native``,
    which (like the app's fixed-point path) differs from ``prepare_maps``' rounded nearest maps."""

    def __init__(self, keystone=False, shape=SHAPE):
        self.image_shape, self.interp = shape, "nearest"
        self.map_x, map_y = _maps(shape, keystone)
        self.map_y = map_y
        self.has_keystone = keystone
        self.row_margin = (2, 1) if keystone else (0, 0)

    def _get_map_y(self):
        return self.map_y if self.map_y is not None else np.indices(self.image_shape, dtype=np.float32)[0]

    def apply_native(self, frame):
        xy, fraction = cv2.convertMaps(self.map_x, self._get_map_y(), cv2.CV_16SC2)
        return cv2.remap(frame, xy, fraction, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def model(interp, keystone):
    return NearestModel(keystone) if interp == "nearest" else smile_model(interp, keystone)


def processor(shape=SHAPE, hot=((0, 0), (0, 1), (3, 4), (3, 5), (SHAPE[0] - 1, SHAPE[1] - 1))):
    """A dark processor with hot pixels: two adjacent pairs, a corner, and a 3x3 hot block whose
    centre has no valid neighbour."""
    rng = np.random.default_rng(7)
    dark = rng.integers(0, 200, shape, dtype=np.uint16)
    hot_map = np.zeros(shape, dtype=np.uint8)
    for y, x in hot:
        hot_map[y, x] = 1
    hot_map[8:11, 20:23] = 1
    proc = DarkProcessor()
    proc.set_dark_reference(dark, hot_map)
    return proc


def reference_of(proc):
    """The snapshot the intensity pipeline corrects a frame with (``_prepare``)."""
    return getattr(proc, "dark_reference", proc)


def binner(width=SHAPE[1]):
    """Bins with overlapping edges (a pixel on an edge belongs to both bins, as the binner has
    them for this slope and step)."""
    return LambdaBinner(m=0.5, c=399.791, step=1.5).fit(width)


# ---------------------------------------------------------------- raw detection
@pytest.mark.parametrize("level", [4095, 65520])
def test_raw_thresholds_follow_the_camera_level(level):
    near_edge = int(np.ceil(0.98 * level))
    raw = np.array([[0, near_edge - 1, near_edge, level - 1, level, min(level + 1, 65535)]], dtype=np.uint16)
    sat = detect_raw(raw, level)
    assert sat.clipped.tolist() == [[False, False, False, False, True, True]]
    assert sat.near.tolist() == [[False, False, True, True, True, True]]
    assert sat.raw_clipped_fraction == pytest.approx(2 / 6) and sat.raw_near_fraction == pytest.approx(4 / 6)
    assert not sat.clipped.flags.writeable


def test_raw_detection_fast_path_and_bad_arguments():
    sat = detect_raw(np.full(SHAPE, 100, np.uint16), 4095)
    assert not sat.any() and sat.clipped is sat.near and not sat.near.any()
    with pytest.raises(ValueError):
        detect_raw(np.zeros(SHAPE), 0)
    with pytest.raises(ValueError):
        detect_raw(np.zeros(5), 4095)


# ---------------------------------------------------------------- hot pixels
def _hot_truth(ref, enabled):
    """For each raw pixel: the output pixels it reaches through the real ``_apply_hot_pixel``."""
    h, w = SHAPE
    truth = np.zeros((h, w, h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            frame = np.zeros(SHAPE, np.uint16)
            frame[y, x] = 8000                       # / at most 8 neighbours: >= 1000 counts
            out = ref._apply_hot_pixel(frame.copy()) if enabled else frame
            truth[y, x] = out != 0
    return truth


@pytest.mark.parametrize("enabled", [True, False])
def test_hot_pixel_mask_matches_the_neighbours_the_repair_reads(enabled):
    ref = reference_of(processor())
    truth = _hot_truth(ref, enabled)
    h, w = SHAPE
    for y in range(h):
        for x in range(w):
            mask = np.zeros(SHAPE, bool)
            mask[y, x] = True
            got = hot_pixel_mask(mask, ref, enabled)
            assert np.array_equal(got, truth[y, x]), (y, x)


def test_hot_pixel_with_no_valid_neighbour_and_saturated_neighbours():
    ref = reference_of(processor())
    mask = np.zeros(SHAPE, bool)
    mask[7:12, 19:24] = True                         # everything around the hot block clipped
    got = hot_pixel_mask(mask, ref, True)
    assert not got[9, 21]                            # block centre: no valid neighbour, reads none
    assert got[8, 20] and got[10, 22]                # block edge: valid clipped neighbours
    lone = np.zeros(SHAPE, bool)
    lone[3, 3] = True                                # a valid neighbour of hot (3, 4) only
    got = hot_pixel_mask(lone, ref, True)
    assert got[3, 3] and got[3, 4] and not got[3, 5]  # (3, 3) is not a neighbour of (3, 5)
    hot_only = np.zeros(SHAPE, bool)
    hot_only[3, 4] = True                            # the hot pixel itself clipped: repaired away
    assert not hot_pixel_mask(hot_only, ref, True)[3, 4]
    assert hot_pixel_mask(hot_only, ref, False)[3, 4]
    assert hot_pixel_mask(hot_only, None, True)[3, 4]


# ---------------------------------------------------------------- geometry
def _geometry_truth(cal, inverted):
    """``truth[y, x]``: the corrected pixels with a non-zero response to raw pixel (y, x)."""
    h, w = SHAPE
    truth = np.zeros((h, w, h, w), dtype=bool)
    if cal is not None and cal.interp != "nearest":
        _x, _y, fixed, fractions, flag = prepare_maps(cal, inverted)
    for y in range(h):
        for x in range(w):
            frame = np.zeros(SHAPE, np.float32)
            frame[y, x] = 1.0
            if cal is None:
                out = frame[::-1] if inverted else frame
            elif cal.interp == "nearest":
                out = cal.apply_native(np.ascontiguousarray(frame[::-1] if inverted else frame))
            else:
                out = cv2.remap(frame, fixed, fractions, flag, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            truth[y, x] = out != 0
    return truth


@pytest.mark.parametrize("loop", [False, True], ids=["dilation", "tap-loop"])
@pytest.mark.parametrize("keystone", [False, True], ids=["smile", "smile+keystone"])
@pytest.mark.parametrize("inverted", [False, True], ids=["normal", "inverted"])
@pytest.mark.parametrize("interp", INTERPS)
def test_geometry_mask_matches_the_opencv_impulse_response(interp, inverted, keystone, loop):
    cal = model(interp, keystone)
    if interp != "nearest":
        fractions = prepare_maps(cal, inverted)[3]
        assert len(np.unique(fractions & 31)) == 32                       # every 1/32 level in x
        if keystone:
            assert len(np.unique((fractions >> 5) & 31)) == 32           # and in y
    geometry = GeometryMask(cal, inverted, SHAPE)
    geometry.reference_loop = loop
    truth = _geometry_truth(cal, inverted)
    h, w = SHAPE
    for y in range(h):
        for x in range(w):
            mask = np.zeros(SHAPE, bool)
            mask[y, x] = True
            assert np.array_equal(geometry.apply(mask), truth[y, x]), (interp, y, x)
    rng = np.random.default_rng(3)                    # many pixels at once: the OR of their responses
    mask = rng.random(SHAPE) < 0.04
    assert np.array_equal(geometry.apply(mask), truth[mask].any(axis=0))
    assert truth.any(axis=(0, 1)).sum() < h * w       # the border really reads outside


@pytest.mark.parametrize("inverted", [False, True])
def test_without_a_model_the_mask_is_only_oriented(inverted):
    mask = np.zeros(SHAPE, bool)
    mask[2, 5] = True
    got = GeometryMask(None, inverted, SHAPE).apply(mask)
    assert np.array_equal(got, _geometry_truth(None, inverted)[2, 5])


@pytest.mark.parametrize("interp", ["linear", "cubic", "lanczos"])
def test_every_uint16_intensity_effect_is_inside_the_mask(interp):
    """Floating-point check apart: what the uint16 pipeline lets a clipped pixel change is never
    outside the mask (exact non-zero support may flag a band where the rounded effect is 0)."""
    cal = smile_model(interp, keystone=True)
    geometry = GeometryMask(cal, False, SHAPE)
    _x, _y, fixed, fractions, flag = prepare_maps(cal, False)
    h, w = SHAPE
    for y in range(0, h, 3):
        for x in range(0, w, 3):
            frame = np.zeros(SHAPE, np.uint16)
            frame[y, x] = 65535
            changed = cv2.remap(frame, fixed, fractions, flag, borderMode=cv2.BORDER_CONSTANT) != 0
            mask = np.zeros(SHAPE, bool)
            mask[y, x] = True
            assert not (changed & ~geometry.apply(mask)).any()


@pytest.mark.parametrize("interp,taps", [("linear", 2), ("cubic", 4), ("lanczos", 8)])
def test_tap_tables_have_one_tap_at_fraction_zero_and_all_taps_otherwise(interp, taps):
    support = tap_support(interp)
    assert support.shape == (32, taps)
    assert support[0].sum() == 1
    assert (support[1:].sum(axis=1) == taps).all()
    assert np.allclose(tap_weights(interp).sum(axis=1), 1.0, atol=1e-5)
    assert tap_support("nearest").shape == (32, 1)


# ---------------------------------------------------------------- binning, subset, crop
def test_bin_masks_are_the_or_over_each_bins_columns():
    b = binner()
    assert (b.iL[1:] < b.iR[:-1]).any()               # overlapping edges, as the binner has them
    rng = np.random.default_rng(5)
    mask = rng.random(SHAPE) < 0.1
    got, fraction = bin_masks(mask, b)
    for r in range(SHAPE[0]):
        for k in range(len(b.iL)):
            cols = mask[r, b.iL[k]:b.iR[k]]
            assert got[r, k] == cols.any()
            assert fraction[r, k] == pytest.approx(cols.sum() / b.w[k])
    assert ((fraction >= 0) & (fraction <= 1)).all()


def test_a_subset_binner_gives_the_full_masks_bands():
    full = binner()
    part = full.subset(3, 11)
    mask = np.random.default_rng(9).random(SHAPE) < 0.1
    assert np.array_equal(bin_masks(mask, part)[0], bin_masks(mask, full)[0][:, 3:11])
    assert np.array_equal(bin_masks(mask, part)[1], bin_masks(mask, full)[1][:, 3:11])


def test_row_crop_is_the_engines_keystone_margin():
    from modules.acquisition.engine import crop_keystone_rows, keystone_row_margin
    cal = smile_model("cubic", keystone=True)
    values = np.arange(SHAPE[0] * 5).reshape(SHAPE[0], 5)
    margin = keystone_row_margin(cal, SHAPE[0])
    assert margin == (2, 1)
    assert np.array_equal(crop_rows(values, margin), crop_keystone_rows(values, cal))
    assert crop_rows(values, (0, 0)) is values


# ---------------------------------------------------------------- the whole frame
@contextmanager
def _nothing():
    yield


class Spans:
    def __init__(self):
        self.names = []

    def time(self, name):
        self.names.append(name)
        return _nothing()


def _frame(rng, clipped=()):
    frame = rng.integers(500, 3000, SHAPE, dtype=np.uint16)
    for y, x in clipped:
        frame[y, x] = 4095
    return frame


@pytest.mark.parametrize("interp", (None,) + INTERPS)
@pytest.mark.parametrize("inverted", [False, True])
@pytest.mark.parametrize("keystone", [False, True])
def test_frame_matches_the_intensity_layout_and_leaves_everything_unchanged(interp, inverted, keystone):
    from modules.acquisition.engine import crop_keystone_rows, keystone_row_margin
    if interp is None and keystone:
        pytest.skip("no model: no keystone")
    cal = model(interp, keystone) if interp else None
    proc = processor()
    proc.set_dark_enabled(True)
    b = binner()
    rng = np.random.default_rng(11)
    raw = _frame(rng, clipped=[(5, 9), (12, 30), (3, 3)])
    runtime = PreprocessingRuntime("cpu", CPU_CAPS)
    before = runtime.process_binned(raw, proc, b, calibration=cal, inverted=inverted, full=True)
    raw_copy = raw.copy()
    result = SaturationPropagator().frame(raw, 4095, reference_of(proc), True, cal, inverted, b,
                                          row_crop=keystone_row_margin(cal, SHAPE[0]))
    after = runtime.process_binned(raw, proc, b, calibration=cal, inverted=inverted, full=True)
    assert np.array_equal(raw, raw_copy)                                   # input untouched
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])
    intensity = crop_keystone_rows(before[0], cal)
    assert result.clipped_mask.shape == intensity.shape == result.affected_fraction.shape
    assert result.clipped_mask.any()
    assert ((result.affected_fraction >= 0) & (result.affected_fraction <= 1)).all()
    assert (result.near_saturation_mask >= result.clipped_mask).all()    # near includes clipped


@pytest.mark.parametrize("interp", INTERPS)
def test_frame_is_the_stages_in_order(interp):
    """The end-to-end mask equals hot -> geometry -> bin -> crop composed from their ground truths."""
    from modules.acquisition.engine import keystone_row_margin
    cal = model(interp, keystone=True)
    ref = reference_of(processor())
    b = binner()
    rng = np.random.default_rng(2)
    raw = _frame(rng, clipped=[(4, 6), (9, 21), (13, 2), (17, 39), (0, 1)])
    clipped = raw >= 4095
    hot = _hot_truth(ref, True)
    geo = _geometry_truth(cal, True)
    after_hot = hot[clipped].any(axis=0)
    corrected = geo[after_hot].any(axis=0)
    expected = crop_rows(np.stack([corrected[:, b.iL[k]:b.iR[k]].any(axis=1) for k in range(len(b.iL))], 1),
                         keystone_row_margin(cal, SHAPE[0]))
    result = SaturationPropagator().frame(raw, 4095, ref, True, cal, True, b,
                                          row_crop=keystone_row_margin(cal, SHAPE[0]))
    assert np.array_equal(result.clipped_mask, expected)


def test_one_clipped_pixel_is_missed_by_the_ceiling_and_found_by_the_mask():
    """Why: averaged with its neighbours, a clipped pixel's band stays below the full-scale ceiling."""
    from modules.live_view import saturated_mask
    cal = smile_model("cubic")
    proc = processor(hot=())
    b = binner()
    raw = np.full(SHAPE, 3000, np.uint16)
    raw[9, 20] = 4095
    runtime = PreprocessingRuntime("cpu", CPU_CAPS)
    binned = runtime.process_binned(raw, proc, b, calibration=cal)[0]
    ceiling = runtime.process_binned(np.full(SHAPE, 4095, np.uint16), proc, b, calibration=cal)[0]
    assert not saturated_mask(binned, ceiling).any()
    assert SaturationPropagator().frame(raw, 4095, reference_of(proc), False, cal, False, b).clipped_mask.any()


def test_all_zero_single_and_all_clipped_frames():
    b = binner()
    prop = SaturationPropagator()
    spans = Spans()
    zero = prop.frame(np.zeros(SHAPE, np.uint16), 4095, None, False, None, False, b, profiler=spans)
    assert not zero.clipped_mask.any() and zero.raw_clipped_fraction == 0 and spans.names == ["sat_detect"]
    single = np.zeros(SHAPE, np.uint16)
    single[4, 10] = 4095
    one = prop.frame(single, 4095, None, False, None, False, b)
    assert one.clipped_mask.sum() == sum(int(b.iL[k] <= 10 < b.iR[k]) for k in range(len(b.iL)))
    assert one.raw_clipped_fraction == pytest.approx(1 / single.size)
    full = prop.frame(np.full(SHAPE, 4095, np.uint16), 4095, None, False, None, False, b)
    assert full.clipped_mask.all() and np.allclose(full.affected_fraction, 1.0)


def test_near_saturation_without_clipping():
    b = binner()
    raw = np.full(SHAPE, 100, np.uint16)
    raw[6, 12] = 4050                                 # >= 0.98 * 4095, < 4095
    spans = Spans()
    result = SaturationPropagator().frame(raw, 4095, None, False, None, False, b, profiler=spans)
    assert not result.clipped_mask.any() and result.near_saturation_mask.any()
    assert "sat_remap" in spans.names and "sat_bin" in spans.names


def test_pack_and_unpack_round_trip():
    mask = np.random.default_rng(1).random((7, 5, 3)) < 0.3
    packed, shape = pack_mask(mask)
    assert packed.dtype == np.uint8 and packed.size == (mask.size + 7) // 8
    assert np.array_equal(unpack_mask(packed, shape), mask)
