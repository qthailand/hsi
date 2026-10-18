"""Spectral quality analysis reports progress, logs, and keeps its results (analytic Jacobian)."""
import logging
import time

import numpy as np
import pytest

from modules.smile import analyze_smile
from modules.smile import quality as q


@pytest.fixture(scope="module")
def setup():
    h, w = 160, 300
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    frame = np.random.default_rng(1).normal(100, 2, (h, w)).astype(np.float32)
    cols = (50, 100, 150, 200, 250)
    for c in cols:
        frame += 3000 * np.exp(-((x - c - 5e-5 * (y - h / 2) ** 2) ** 2) / 4.5)
    analysis = analyze_smile(frame)
    refs = [q.SpectralReference(400 + .5 * c, c, 12) for c in cols]
    return analysis, refs


def test_analytic_jacobian_matches_finite_differences():
    from scipy.optimize._numdiff import approx_derivative
    x = np.arange(20, 45, dtype=float)
    origin = x.mean()
    v = 800 * np.exp(-0.5 * ((x - 31.3) / 2.1) ** 2) + 50 + 0.4 * (x - origin)
    p = np.array([700., 31., 2.5, 40., 0.1])
    numeric = approx_derivative(lambda pp: q._gaussian_residual(pp, x, origin, v), p, method='3-point')
    np.testing.assert_allclose(q._gaussian_jacobian(p, x, origin, v), numeric, rtol=1e-5, atol=1e-6)


def test_progress_is_monotonic_reaches_one_and_logs(setup, caplog):
    analysis, refs = setup
    seen = []
    with caplog.at_level(logging.INFO, logger="modules.smile.quality"):
        report = q.analyze_quality(analysis.frame_orig, analysis.frame_corr, refs, m_nm_per_px=.5, c_nm=400,
                                   row_step=8, y_ref=int(analysis.y_ref), saturation=analysis.sat_level,
                                   progress=lambda f, m: seen.append((f, m)))
        unc_seen = []
        q.estimate_shift_uncertainty(analysis, rows=report.rows[:, None],
                                     columns=np.array([r.column_px for r in refs])[None, :], repeats=20,
                                     progress=lambda f, m: unc_seen.append(f))
    fractions = [f for f, _ in seen]
    assert len(seen) == 1 + 2 * len(refs)                       # start + every line of both frames
    assert fractions == sorted(fractions) and fractions[-1] == pytest.approx(1.0)
    assert any("after: line 5/5" in m for _, m in seen)
    assert unc_seen == sorted(unc_seen) and unc_seen[-1] == pytest.approx(1.0)
    text = caplog.text
    assert "Spectral quality: 5 lines" in text and "Spectral quality done" in text
    assert "Shift uncertainty done" in text


def test_a_failing_progress_callback_does_not_stop_the_analysis(setup):
    analysis, refs = setup

    def broken(fraction, message):
        raise RuntimeError("UI gone")
    report = q.analyze_quality(analysis.frame_orig, analysis.frame_corr, refs, m_nm_per_px=.5, c_nm=400,
                               row_step=16, y_ref=int(analysis.y_ref), progress=broken)
    assert report.summary["paired_wavelength_after_nm"]["count"] > 0


def test_widget_shows_progress_while_running(setup, tmp_path):
    import json
    from PyQt5.QtWidgets import QApplication
    from modules.smile.widget import SmileCorrectionWidget
    app = QApplication.instance() or QApplication([])
    analysis, refs = setup
    path = tmp_path / "wl.json"
    path.write_text(json.dumps({"m_nm_per_px": .5, "c_nm": 400, "reference_row_y": int(analysis.y_ref),
                                "standards": [{"wavelength_nm": r.wavelength_nm, "x_px": r.column_px,
                                               "half_window_px": 12} for r in refs]}))
    widget = SmileCorrectionWidget()
    widget._on_analysis_done(analysis)
    widget.load_wavelength_calibration(str(path))
    values = []
    widget.run_quality_analysis()
    assert widget.quality_progress.isVisible() or not widget.isVisible()
    widget._quality_worker.progress.connect(lambda p, m: values.append(p))
    end = time.time() + 60
    while widget._quality_worker is not None and time.time() < end:
        app.processEvents(); time.sleep(0.01)
    assert widget.get_quality_report() is not None
    assert values and values == sorted(values) and max(values) == 100
    assert not widget.quality_progress.isVisible()
    widget.close()
