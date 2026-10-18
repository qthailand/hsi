"""Regression tests for review findings on the FWHM core (start positions, wings, smile border).

* A weak line next to a stronger one: the start used to move onto the neighbour's flank and
  the centre ended pinned on its ±2 px bound with a width 15-20 % too large, flagged ``ok``.
* The first width estimate was taken around a stronger neighbour's flank: well resolved
  lines were rejected as ``outside``.
* Lorentzian: a strong listed line just outside the window bent the weak line's background.
* A row-dependent smile border (zero in some rows only) entered the baseline and the fit.
* ``cancel=`` stops a measurement.
"""
import numpy as np
import pytest
from scipy.special import ndtr

from modules.spectral_resolution import (MeasurementCancelled, fit_spectrum, measure_frame, measure_spectrum,
                                         remove_spectrum_baseline, usable_columns)
from modules.spectral_resolution.core import FWHM_PER_SIGMA

N = 1608
X = np.arange(N, dtype=float)


def line(profile, height, center, fwhm, x=X):
    """Pixel-integrated line of continuous peak ``height``."""
    if profile == "gaussian":
        s = fwhm / FWHM_PER_SIGMA
        return height * s * np.sqrt(2 * np.pi) * (ndtr((x + .5 - center) / s) - ndtr((x - .5 - center) / s))
    g = fwhm / 2.0
    return height * g * (np.arctan((x + .5 - center) / g) - np.arctan((x - .5 - center) / g))


@pytest.mark.parametrize("profile", ["gaussian", "lorentzian"])
def test_weak_line_next_to_a_strong_one_is_not_pinned_on_its_flank(profile):
    fwhm, weak = 3.5, 458.6
    strong = weak + 1.5 * fwhm                                    # 1.5 FWHM apart
    errors, centres = [], []
    for seed in range(8):
        y = (300 + line(profile, 600, weak, fwhm) + line(profile, 4400, strong, fwhm)
             + np.random.default_rng(seed).normal(0, 5, N))
        w = measure_spectrum(y, [round(weak), round(strong)], profile=profile)[0]
        assert w.flag == "ok"
        errors.append(w.fwhm_px / fwhm - 1)
        centres.append(w.center_px - weak)
    assert abs(np.mean(errors)) < 0.01 and np.max(np.abs(errors)) < 0.04     # was +18 % / +20 %
    assert np.max(np.abs(centres)) < 0.1                           # was pinned at 459.0 (0.4 px off)


def test_centre_on_its_bound_is_fit_failed_not_ok():
    x = np.arange(400.0)
    y = 300 + line("gaussian", 3000, 104.8, 3.0, x) + np.random.default_rng(0).normal(0, 5, 400)
    w = fit_spectrum(y, [100], find_neighbours=False).measurements[0]
    assert w.center_px == pytest.approx(102.0) and w.flag == "fit_failed"   # 4.8 px off: out of reach


@pytest.mark.parametrize("offset", [1.5, 2.6, 3.5])
@pytest.mark.parametrize("fwhm", [2.5, 8.0])
def test_a_guess_a_few_px_off_an_isolated_line_still_finds_it(offset, fwhm):
    x = np.arange(400.0)
    y = 300 + line("gaussian", 3000, 100 + offset, fwhm, x) + np.random.default_rng(0).normal(0, 5, 400)
    w = measure_spectrum(y, [100])[0]
    assert w.flag == "ok" and w.center_px == pytest.approx(100 + offset, abs=0.02)
    assert w.fwhm_px == pytest.approx(fwhm, rel=0.01)


@pytest.mark.parametrize("profile, fwhm", [("gaussian", 8.0), ("lorentzian", 5.0), ("lorentzian", 8.0)])
def test_resolved_weak_line_beside_a_strong_one_is_not_rejected_as_outside(profile, fwhm):
    for seed in range(5):
        y = (300 + line(profile, 400, 800, fwhm) + line(profile, 8000, 820, fwhm)
             + np.random.default_rng(seed).normal(0, 5, N))
        weak, strong = measure_spectrum(y, [800, 820], profile=profile)
        assert weak.flag == "ok" and weak.fwhm_px == pytest.approx(fwhm, rel=0.04)   # was "outside"
        assert strong.flag == "ok" and strong.fwhm_px == pytest.approx(fwhm, rel=0.01)


@pytest.mark.parametrize("fwhm, apart", [(3.0, 12), (8.0, 10)])
def test_lorentzian_wing_of_a_strong_listed_line_outside_the_window_is_removed(fwhm, apart):
    with_neighbour, alone = [], []
    for seed in range(10):
        weak = 300 + line("lorentzian", 300, 800, fwhm) + np.random.default_rng(seed).normal(0, 5, N)
        y = weak + line("lorentzian", 8000, 800 + apart * fwhm, fwhm)
        fit = fit_spectrum(y, [800, 800 + apart * fwhm], profile="lorentzian")
        assert len(fit.windows) == 2 and fit.measurements[0].flag == "ok"
        refitted = next(win for win in fit.windows if win.lines == (0,))
        assert refitted.wings is not None and not refitted.wings.flags.writeable
        xs = np.arange(refitted.first, refitted.last + 1, dtype=float)
        assert np.abs(refitted.total(xs) - fit.corrected[refitted.first:refitted.last + 1]).max() < 30   # model incl. wings
        with_neighbour.append(fit.measurements[0].fwhm_px / fwhm - 1)
        alone.append(measure_spectrum(weak, [800], profile="lorentzian")[0].fwhm_px / fwhm - 1)
    assert abs(np.mean(with_neighbour) - np.mean(alone)) < 0.005           # was -4.8 % / -6.5 %


def test_gaussian_windows_are_not_refitted_for_wings():
    y = 300 + line("gaussian", 300, 800, 3.0) + line("gaussian", 8000, 836, 3.0)
    assert all(win.wings is None for win in fit_spectrum(y, [800, 836]).windows)


def smile_border_frame(rows=128, cols=600):
    """Flat 300 counts + three lines; a zero border whose width depends on the row (as cv2.remap)."""
    x = np.arange(cols, dtype=float)
    spectrum = 300 + sum(line("gaussian", 2000, c, 3.0, x) for c in (40.3, 300.3, 560.3))
    frame = spectrum[None, :] + np.random.default_rng(1).normal(0, 3, (rows, cols))
    for r in range(rows):
        t = ((r - rows / 2) / (rows / 2)) ** 2
        right, left = int(np.ceil(6 * t)), int(np.ceil(4 * t))
        if right:
            frame[r, cols - right:] = 0
        if left:
            frame[r, :left] = 0
    return frame


def test_columns_zero_filled_in_some_rows_are_not_usable():
    frame = smile_border_frame()
    assert usable_columns(frame) == (4, 593)                       # was (0, 599): any row non-zero
    assert usable_columns(np.zeros((3, 10))) == (0, 9)
    inner = np.ones((4, 10))
    inner[1, 5] = 0                                               # a genuine zero inside a row does not count
    assert usable_columns(inner) == (0, 9)
    preview = frame.mean(axis=0)                                  # what the panel previews
    for method in ("modpoly", "asls", "arpls"):
        _, base = remove_spectrum_baseline(preview, method, None, usable_columns(frame))
        assert np.abs(base[4:594] - 300).max() < 3                 # was -255 (ModPoly) / -159 (AsLS)


def test_measure_frame_limits_every_block_to_its_own_usable_columns():
    frame = smile_border_frame()
    result = measure_frame(frame, [40.3, 300.3, 560.3], rows_per_block=16, baseline="arpls")
    columns = result.block_columns
    assert columns.shape == (8, 2) and columns.tolist()[0] == [4, 593] and columns.tolist()[3] == [1, 598]
    assert result.settings["valid_columns"] == [4, 593]
    for summary in result.lines:
        assert summary.n_valid == 8 and summary.fwhm_px_median == pytest.approx(3.0, rel=0.005)
    limited = measure_frame(frame, [300.3], rows_per_block=16, valid_columns=(10, 580))
    assert limited.block_columns.tolist() == [[10, 580]] * 8 and limited.settings["valid_columns"] == [10, 580]


def test_cancel_stops_a_measurement():
    frame = smile_border_frame()
    blocks = []
    with pytest.raises(MeasurementCancelled):
        measure_frame(frame, [300.3], rows_per_block=16, progress=lambda f, m: blocks.append(f),
                      cancel=lambda: len(blocks) >= 2)
    assert len(blocks) == 2

    def stop(fraction, message):
        raise MeasurementCancelled("from the progress callback")
    with pytest.raises(MeasurementCancelled):                     # passes through (other errors are logged)
        measure_frame(frame, [300.3], rows_per_block=16, progress=stop)
    evaluations = []

    def during_the_fit():
        evaluations.append(1)
        return len(evaluations) > 3                              # polled on every model evaluation
    with pytest.raises(MeasurementCancelled):
        fit_spectrum(frame[64], [40.3, 300.3, 560.3], cancel=during_the_fit)
    assert len(evaluations) == 4
