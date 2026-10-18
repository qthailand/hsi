"""FWHM line profiles: pixel-integrated Lorentzian next to the default Gaussian.

The Gaussian results must not change (checked against values recorded before the profile
option existed); the Lorentzian must recover its own width, centre and peak height, fit
neighbours jointly, flag saturation and window problems, and work through ``measure_frame``.
Fitting the wrong profile biases the width with a known sign - the reason to offer both.
"""
import dataclasses

import numpy as np
import pytest
from scipy.special import erf

from modules.spectral_resolution import LineMeasurement, measure_frame, measure_spectrum
from modules.spectral_resolution.core import DEFAULT_WINDOW_FWHM, PROFILES

X = np.arange(400.0)


def lorentz_line(center, fwhm, height, x=X):
    """What a detector records from a Lorentzian of peak ``height``: its integral over each pixel.

    ``(A/π)·[atan((x+½-μ)/γ) - atan((x-½-μ)/γ)]`` with ``γ = FWHM/2`` and area ``A = π·γ·height``.
    """
    g = fwhm / 2.0
    return height * g * (np.arctan((x + .5 - center) / g) - np.arctan((x - .5 - center) / g))


def gauss_line(center, fwhm, height, x=X):
    """Pixel-integrated Gaussian of peak ``height`` (same helper as tests/test_spectral_resolution.py)."""
    s = fwhm / 2.3548
    return height * s * np.sqrt(np.pi / 2) * (erf((x + .5 - center) / (np.sqrt(2) * s))
                                             - erf((x - .5 - center) / (np.sqrt(2) * s)))


def identical(a, b):
    """Two lists of measurements are equal field by field, NaN equal to NaN."""
    def key(w):
        return tuple("nan" if isinstance(v, float) and np.isnan(v) else v for v in dataclasses.astuple(w))
    return [key(w) for w in a] == [key(w) for w in b]


# ───────────────────────────── Lorentzian accuracy ─────────────────────────────

@pytest.mark.parametrize("fwhm", [1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0])
def test_lorentzian_width_centre_and_height_are_recovered(fwhm):
    rng = np.random.default_rng(11)
    s = 100 + lorentz_line(200.3, fwhm, 3000) + rng.normal(0, 3, X.size)
    w = measure_spectrum(s, [200], m=0.5, profile="lorentzian")[0]
    assert w.valid and w.profile == "lorentzian" and w.n_components == 1
    assert w.fwhm_px == pytest.approx(fwhm, rel=0.01) and w.center_px == pytest.approx(200.3, abs=0.02)
    assert w.amplitude == pytest.approx(3000, rel=0.01)
    assert w.fwhm_nm == pytest.approx(w.fwhm_px * 0.5)
    if abs(fwhm - 2.0) > 0.1:                                     # 2 px is the threshold itself
        assert (w.flag == "undersampled") == (fwhm < 2.0)


def test_lorentzian_amplitude_is_the_peak_height_of_the_continuous_profile():
    for fwhm in (1.5, 4.0):
        s = 100 + lorentz_line(200.3, fwhm, 2500)                 # noise free: exact
        w = measure_spectrum(s, [200], profile="lorentzian")[0]
        assert w.amplitude == pytest.approx(2500, rel=1e-6) and w.fwhm_px == pytest.approx(fwhm, rel=1e-6)
        assert w.center_px == pytest.approx(200.3, abs=1e-6)
    assert lorentz_line(200.3, 1.5, 2500).max() < 0.85 * 2500    # pixels average a narrow peak down


def test_profile_name_is_case_insensitive_and_invalid_names_raise():
    s = 100 + lorentz_line(200.3, 3, 3000)
    assert identical(measure_spectrum(s, [200], profile="Lorentzian"),
                     measure_spectrum(s, [200], profile="lorentzian"))
    for bad in ("voigt", "", None, 1):
        with pytest.raises(ValueError):
            measure_spectrum(s, [200], profile=bad)
        with pytest.raises(ValueError):
            measure_frame(np.tile(s, (4, 1)), [200], profile=bad)
    assert PROFILES == ("gaussian", "lorentzian")
    assert DEFAULT_WINDOW_FWHM == {"gaussian": 3.0, "lorentzian": 5.0}


def test_neighbouring_lorentzians_are_fitted_jointly():
    rng = np.random.default_rng(12)
    s = 100 + lorentz_line(200.3, 3, 3000) + lorentz_line(206.5, 3, 2000) + rng.normal(0, 3, X.size)
    alone = measure_spectrum(s, [200], profile="lorentzian")[0]   # the neighbour is not listed
    both = measure_spectrum(s, [200, 206], profile="lorentzian")
    assert alone.n_components == 2 and alone.fwhm_px == pytest.approx(3.0, rel=0.02)
    assert [w.n_components for w in both] == [2, 2]
    assert [w.fwhm_px for w in both] == [pytest.approx(3.0, rel=0.02)] * 2
    assert [w.center_px for w in both] == [pytest.approx(200.3, abs=0.02), pytest.approx(206.5, abs=0.02)]
    single = measure_spectrum(s, [200], profile="lorentzian", find_neighbours=False)[0]
    assert not single.valid                       # one profile cannot fit the blend: rejected, not reported


def test_lorentzian_on_a_sloped_background():
    w = measure_spectrum(100 + 2 * (X - 200) + lorentz_line(200.3, 3, 3000), [200], profile="lorentzian")[0]
    assert w.flag == "ok" and w.fwhm_px == pytest.approx(3.0, rel=0.005)


def test_saturated_and_weak_lorentzians_are_flagged():
    clipped = np.minimum(100 + lorentz_line(200.3, 3, 8000), 4095.0)
    w = measure_spectrum(clipped, [200], saturation=4095, profile="lorentzian")[0]
    assert w.flag == "saturated" and np.isnan(w.fwhm_px) and not w.valid
    rng = np.random.default_rng(13)                               # SNR 5 < 10 (at SNR ~2, pure noise,
    weak = measure_spectrum(100 + lorentz_line(200.3, 3, 30) + rng.normal(0, 6, X.size), [200],
                            profile="lorentzian")[0]              # either profile may also hit a bound)
    assert weak.flag == "low_snr" and np.isnan(weak.fwhm_px)


def test_outside_means_the_lorentzian_core_does_not_fit_in_the_window():
    cut = dict(valid_columns=(195, X.size - 1))                   # usable data starts 5 px left of the line
    lorentz, gauss = 100 + lorentz_line(200.3, 3, 3000), 100 + gauss_line(200.3, 3, 3000)
    assert measure_spectrum(lorentz, [200], profile="lorentzian", **cut)[0].flag == "outside"   # needs ±7 px
    assert measure_spectrum(gauss, [200], profile="gaussian", **cut)[0].flag == "ok"            # needs ±3.2 px
    # ±3 FWHM is enough for a well estimated Lorentzian, but a wide line's first estimate is low
    # (wings inside the ±12 px start region): the wider Lorentzian default keeps it.
    assert measure_spectrum(100 + lorentz_line(200.3, 5, 3000), [200], profile="lorentzian",
                            window_fwhm=3)[0].flag == "ok"
    wide = 100 + lorentz_line(200.3, 16, 3000)
    assert measure_spectrum(wide, [200], profile="lorentzian", window_fwhm=3)[0].flag == "outside"
    w = measure_spectrum(wide, [200], profile="lorentzian")[0]
    assert w.flag == "ok" and w.fwhm_px == pytest.approx(16, rel=1e-4)


def test_frame_with_lorentzian_profile():
    rng = np.random.default_rng(14)
    frame = np.zeros((96, X.size))
    truth = [(60, 2.2), (150, 2.6), (240, 3.0), (330, 3.4)]
    for c, f in truth:
        frame += lorentz_line(c + 0.3, f, 2000)[None, :]
    frame = 100 + frame + rng.normal(0, 4, frame.shape)
    frame[5, 239:243] = 4095                                      # one clipped row inside block 0
    r = measure_frame(frame, [c for c, _ in truth], rows_per_block=16, m=0.5, c=400, saturation=4095,
                      wavelengths=[430.0, 475.0, 520.0, 565.0], profile="LORENTZIAN")
    assert r.settings["profile"] == "lorentzian" and r.settings["window_fwhm"] == 5.0
    assert r.settings["model"] == "pixel-integrated Lorentzian + linear background"
    assert {w.profile for row in r.measurements for w in row} == {"lorentzian"}
    assert r.block_rows.shape == (6, 2) and r.flags()[0, 2] == "saturated"
    for line, (_, f) in zip(r.lines, truth):
        assert line.fwhm_px_median == pytest.approx(f, rel=0.02)
        assert line.fwhm_nm_median == pytest.approx(f * 0.5, rel=0.02)
    assert r.lines[2].n_valid == 5 and r.summary()["n_lines_valid"] == 4
    assert np.allclose(r.array("amplitude")[1:, 0], 2000, rtol=0.03)


# ───────────────────────────── Gaussian default unchanged ─────────────────────────────

def test_default_profile_is_gaussian_and_unchanged():
    rng = np.random.default_rng(2)                                # tests/test_spectral_resolution.py case
    s = 100 + gauss_line(200.3, 3, 3000) + gauss_line(206.5, 3, 2000) + rng.normal(0, 3, X.size)
    default = measure_spectrum(s, [200, 206])
    assert identical(default, measure_spectrum(s, [200, 206], profile="gaussian"))
    assert identical(default, measure_spectrum(s, [200, 206], profile="GAUSSIAN", window_fwhm=3.0))
    # values recorded with the code before the profile option (identical there to the last bit;
    # the tolerance only allows for other platforms' floating-point rounding)
    recorded = [(200.30071468355504, 2.9985098220529562, 3.1743827829721454, 2998.8756704059124),
                (206.49937905053724, 3.0096004257050084, 3.3046952200102737, 1992.0674898707591)]
    for w, (c, f, f_half, amp) in zip(default, recorded):
        assert w.profile == "gaussian" and w.flag == "ok" and w.n_components == 2
        assert (w.center_px, w.fwhm_px, w.fwhm_px_halfmax, w.amplitude) == pytest.approx((c, f, f_half, amp),
                                                                                       rel=1e-6)
    frame = np.full((20, X.size), 100.0) + gauss_line(200.3, 3, 2000)[None, :]
    frame[:, :30] = 0
    r = measure_frame(frame, [10, 200], rows_per_block=20)
    assert r.settings["model"] == "pixel-integrated Gaussian + linear background"
    assert r.settings["profile"] == "gaussian" and r.settings["window_fwhm"] == 3.0
    assert [w.profile for w in r.measurements[0]] == ["gaussian", "gaussian"]   # a border entry too
    assert dataclasses.fields(LineMeasurement)[-1].name == "profile"
    assert LineMeasurement(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1, "ok").profile == "gaussian"


# ───────────────────────────── model mismatch ─────────────────────────────

@pytest.mark.parametrize("fwhm", [2.0, 3.0, 5.0, 8.0])
def test_fitting_the_wrong_profile_biases_the_width_with_a_known_sign(fwhm):
    """Why the profile is a choice: the width is only right with the line's own shape.

    A Lorentzian fitted to a Gaussian line must keep its heavy wings low where the data
    have none, so it narrows (≈ -15 to -28 % here); a Gaussian fitted to a Lorentzian line
    widens to reach the wings (≈ +9 to +19 %). The sizes depend on window and background
    freedom (not derived here), so only the sign and a clear margin are asserted.
    """
    gauss = 100 + gauss_line(200.3, fwhm, 3000)
    lorentz = 100 + lorentz_line(200.3, fwhm, 3000)
    right_g = measure_spectrum(gauss, [200], profile="gaussian")[0]
    right_l = measure_spectrum(lorentz, [200], profile="lorentzian")[0]
    wrong_l = measure_spectrum(gauss, [200], profile="lorentzian")[0]     # Gaussian line, Lorentzian model
    wrong_g = measure_spectrum(lorentz, [200], profile="gaussian")[0]     # Lorentzian line, Gaussian model
    assert right_g.fwhm_px == pytest.approx(fwhm, rel=1e-4) and right_l.fwhm_px == pytest.approx(fwhm, rel=1e-4)
    assert wrong_l.valid and wrong_g.valid                        # the flags do not reveal a wrong model
    assert wrong_l.fwhm_px < 0.9 * fwhm
    assert wrong_g.fwhm_px > 1.05 * fwhm
