"""Physical synthetic checks for quality diagnostics and failure handling."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from smile.quality import (SpectralReference, analyze_quality,
                           correlation_shift, estimate_shift_uncertainty)
from smile.surface import fit_surface_2d


def frames():
    x = np.arange(160)
    shifts = np.linspace(-1.5, 1.5, 21)
    before = np.array([20+0.02*x+1000*np.exp(-0.5*((x-80-s)/3)**2) for s in shifts])
    after = np.tile(20+0.02*x+1000*np.exp(-0.5*((x-80)/3)**2), (21, 1))
    return before, after


def test_absolute_error_width_and_area(tmp_path):
    before, after = frames()
    report = analyze_quality(before, after, [SpectralReference(540, 80)],
                             m_nm_per_px=.5, c_nm=500, resolution_nm=5)
    assert report.summary['wavelength_before_nm']['rmse'] > .4
    assert report.summary['wavelength_after_nm']['rmse'] < 1e-5
    np.testing.assert_allclose(report.after['fwhm_nm'], 3*2.354820045*.5, atol=1e-4)
    assert report.summary['area_counts_px_change_percent']['max_abs'] < .01
    assert report.summary['correlation_after_px']['max_abs'] < 1e-3
    path = tmp_path/'report.json'
    report.save(path)
    assert json.loads(path.read_text())['metadata']['independent_frame'] is False


def test_blends_and_saturation_are_missing(tmp_path):
    before, after = frames()
    report = analyze_quality(before, after,
                             [SpectralReference(540,80), SpectralReference(543,86)],
                             m_nm_per_px=.5, c_nm=500)
    assert report.metadata['excluded_blends'] == [True, True]
    assert report.summary['wavelength_after_nm']['count'] == 0
    path = tmp_path/'missing.json'
    report.save(path)
    assert json.loads(path.read_text())['after']['center_px'][0] == [None, None]
    report = analyze_quality(before, after, [SpectralReference(540,80)],
                             m_nm_per_px=.5, c_nm=500, saturation=500)
    assert report.summary['wavelength_after_nm']['count'] == 0


def test_cross_correlation_sign_and_flat_rejection():
    x = np.arange(101)
    ref = np.exp(-.5*((x-50)/4)**2)
    target = np.exp(-.5*((x-51.25)/4)**2)
    shift, score = correlation_shift(ref, target)
    assert shift == pytest.approx(1.25, abs=.04)
    assert score > .999
    assert np.isnan(correlation_shift(np.ones(101), np.ones(101))[0])


def test_bootstrap_support_and_reproducibility():
    rng = np.random.default_rng(5)
    y, x = np.meshgrid(np.arange(30), [30., 60., 90.], indexing='ij')
    y, x = y.ravel(), x.ravel()
    v = .01*(y-15)**2 + .1*rng.normal(size=y.size)
    surface, _ = fit_surface_2d(y,x,v,2,1)
    analysis = SimpleNamespace(meas_y=y, meas_x=x, meas_v=v, surface=surface, y_ref=15)
    kwargs = dict(rows=np.array([0.,15.,29.]), columns=np.array([10.,60.,100.]), repeats=20)
    a = estimate_shift_uncertainty(analysis, **kwargs)
    b = estimate_shift_uncertainty(analysis, **kwargs)
    np.testing.assert_equal(a['std_px'], b['std_px'])
    assert a['supported'].tolist() == [False, True, False]
    assert a['std_px'][1] == 0
    assert a['std_px'][0] > 0


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match='uncropped'):
        analyze_quality(np.ones((20,30)), np.ones((20,28)),
                        [SpectralReference(500,10)], m_nm_per_px=1,c_nm=490)


def test_detects_wavelength_bias_and_broadening():
    before, _ = frames()
    x = np.arange(160)
    after = np.tile(20+0.02*x+900*np.exp(-.5*((x-81)/3.6)**2), (21,1))
    report = analyze_quality(before, after, [SpectralReference(540,80)],
                             m_nm_per_px=.5,c_nm=500)
    assert report.summary['wavelength_after_nm']['bias'] == pytest.approx(.5,abs=.001)
    assert report.summary['fwhm_nm_change_percent']['bias'] == pytest.approx(20,abs=.01)
    assert report.summary['peak_height_change_percent']['bias'] == pytest.approx(-10,abs=.01)


def test_noise_and_window_edge_are_rejected():
    from smile.quality import _peak
    assert _peak(np.ones(100), 50,20,5,None) is None
    profile = np.zeros(100); profile[30] = 1000
    assert _peak(profile,50,20,5,None) is None
