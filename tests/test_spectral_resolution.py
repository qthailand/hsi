"""FWHM module, per-setting calibration store, Smile Apply and automatic loading by camera setting."""
import time

import numpy as np
import pytest
from scipy.special import erf

from modules.contracts import CalibrationStore, CameraSetting
from modules.spectral_resolution import measure_frame, measure_spectrum

X = np.arange(400.0)


def pixel_line(center, fwhm, amplitude, x=X):
    """What a detector records: a Gaussian integrated over each pixel."""
    s = fwhm / 2.3548
    return amplitude * s * np.sqrt(np.pi / 2) * (erf((x + .5 - center) / (np.sqrt(2) * s))
                                                - erf((x - .5 - center) / (np.sqrt(2) * s)))


# ───────────────────────────── measurement accuracy ─────────────────────────────

@pytest.mark.parametrize("fwhm", [1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 14.0])
def test_width_is_exact_from_undersampled_to_wide_lines(fwhm):
    rng = np.random.default_rng(1)
    w = measure_spectrum(100 + pixel_line(200.3, fwhm, 3000) + rng.normal(0, 3, X.size), [200], m=0.5)[0]
    assert w.valid and w.fwhm_px == pytest.approx(fwhm, rel=0.01) and w.center_px == pytest.approx(200.3, abs=0.02)
    assert w.fwhm_nm == pytest.approx(w.fwhm_px * 0.5)
    if abs(fwhm - 2.0) > 0.1:                                     # 2 px is the threshold itself
        assert (w.flag == "undersampled") == (fwhm < 2.0)


def test_neighbouring_lines_are_fitted_together():
    rng = np.random.default_rng(2)
    s = 100 + pixel_line(200.3, 3, 3000) + pixel_line(206.5, 3, 2000) + rng.normal(0, 3, X.size)
    alone = measure_spectrum(s, [200])[0]                       # the neighbour is not listed
    both = measure_spectrum(s, [200, 206])
    assert alone.n_components == 2 and alone.fwhm_px == pytest.approx(3.0, rel=0.02)
    assert [round(w.fwhm_px, 1) for w in both] == [3.0, 3.0]


def test_saturated_and_weak_lines_are_flagged_not_reported_wrong():
    clipped = np.minimum(100 + pixel_line(200.3, 3, 8000), 4095.0)
    w = measure_spectrum(clipped, [200], saturation=4095)[0]
    assert w.flag == "saturated" and np.isnan(w.fwhm_px) and not w.valid
    rng = np.random.default_rng(3)
    weak = measure_spectrum(100 + pixel_line(200.3, 3, 10) + rng.normal(0, 6, X.size), [200])[0]
    assert weak.flag == "low_snr" and np.isnan(weak.fwhm_px)


def test_sloped_background_does_not_bias_the_width():
    w = measure_spectrum(100 + 2 * (X - 200) + pixel_line(200.3, 3, 3000), [200])[0]
    assert w.fwhm_px == pytest.approx(3.0, rel=0.005)


def test_frame_blocks_summary_and_saturation_in_a_single_row():
    rng = np.random.default_rng(4)
    frame = np.zeros((96, X.size))
    truth = [(60, 2.2), (150, 2.6), (240, 3.0), (330, 3.4)]
    for c, f in truth:
        frame += pixel_line(c + 0.3, f, 2000)[None, :]
    frame = 100 + frame + rng.normal(0, 4, frame.shape)
    frame[5, 239:243] = 4095                                      # one clipped row inside block 0
    steps = []
    r = measure_frame(frame, [c for c, _ in truth], rows_per_block=16, m=0.5, c=400, saturation=4095,
                      wavelengths=[430.0, 475.0, 520.0, 565.0], progress=lambda f, m: steps.append(f))
    assert r.block_rows.shape == (6, 2) and steps[-1] == pytest.approx(1.0)
    assert r.flags()[0, 2] == "saturated"                         # clipping in any row of a block rejects it
    for line, (_, f) in zip(r.lines, truth):
        assert line.fwhm_px_median == pytest.approx(f, rel=0.02)
        assert line.fwhm_nm_median == pytest.approx(f * 0.5, rel=0.02)
        assert line.resolving_power_median == pytest.approx(line.wavelength_nm / (f * 0.5), rel=0.03)
    assert r.lines[2].n_valid == 5 and r.summary()["n_lines_valid"] == 4


def test_smile_border_columns_are_ignored():
    frame = np.full((20, X.size), 100.0) + pixel_line(200.3, 3, 2000)[None, :]
    frame[:, :30] = 0
    r = measure_frame(frame, [10, 200], rows_per_block=20)
    assert r.lines[0].main_flag == "border" and r.lines[1].n_valid == 1


# ───────────────────────────── calibration store ─────────────────────────────

SETTING = CameraSetting("genicam", 0, 2, True, (64, 400))


def test_store_saves_by_setting_keeps_history_and_reports_mismatches(tmp_path):
    from modules.wavelength_calibration import WavelengthCalibration, fit_wavelength
    store = CalibrationStore(tmp_path / "wl", tmp_path / "smile")
    assert store.load_wavelength(SETTING) is None
    p1 = fit_wavelength([10, 200, 390], [400.0, 495.0, 590.0], 2, (64, 400), 32)
    path = store.save_wavelength(p1, SETTING)
    assert path.name == "wavelength_bin2.json" and WavelengthCalibration.load(path).m == pytest.approx(p1.m)
    store.save_wavelength(fit_wavelength([10, 200, 390], [401.0, 496.0, 591.0], 2, (64, 400), 32), SETTING)
    assert len(list((tmp_path / "wl" / "history").iterdir())) == 1     # previous profile kept
    stored = store.load_wavelength(SETTING)
    assert stored.ok and stored.profile.c == pytest.approx(401.0 - stored.profile.m * 10, abs=1e-6)
    other = CameraSetting("genicam", 0, 2, False, (64, 400))           # flipped orientation
    assert not store.load_wavelength(other).ok and "orientation" in store.load_wavelength(other).problems[0]
    with pytest.raises(ValueError):
        store.save_wavelength(fit_wavelength([10, 200], [400.0, 495.0], 1), SETTING)   # bin 1 profile, bin 2 camera
    assert store.smile_path(SETTING).name == "smile_model_bin2.npz"


def test_setting_label_and_mismatch():
    assert SETTING.label() == "genicam #0 · bin 2 · 400×64 px · inverted"
    assert SETTING.mismatch(SETTING.with_frame((64, 400))) == []
    assert SETTING.mismatch(CameraSetting("poa", 0, 2, True, (64, 400))) == ["camera poa ≠ genicam"]


# ───────────────────────────── widgets ─────────────────────────────

def _lamp(h=64, w=400, curvature=4e-4):
    """Curved (smile) lamp frame and the line positions at the reference row."""
    from modules.contracts import default_reference_row
    y = np.arange(h)[:, None].astype(float)
    y_ref = default_reference_row(h)
    centers = [60.3, 150.3, 240.3, 330.3]
    frame = np.full((h, w), 100.0)
    for c in centers:
        shift = curvature * (y - y_ref) ** 2
        frame += np.vstack([pixel_line(c + s, 3.0, 3000, np.arange(w, dtype=float)) for s in shift[:, 0]])
    frame += np.random.default_rng(5).normal(0, 3, frame.shape)
    return frame.astype(np.float32), centers, y_ref


@pytest.fixture
def app():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _pump(app, condition, timeout=30):
    end = time.time() + timeout
    while time.time() < end and not condition():
        app.processEvents()
        time.sleep(0.01)
    return condition()


def test_smile_apply_stores_for_the_setting_and_fwhm_loads_everything_by_itself(app, tmp_path):
    from modules.frame_exchange import FrameSnapshot
    from modules.smile.widget import SmileCorrectionWidget
    from modules.spectral_resolution.widget import SpectralResolutionWidget
    from modules.wavelength_calibration import fit_wavelength
    frame, centers, y_ref = _lamp()
    setting = CameraSetting("genicam", 0, 2, True, frame.shape)
    store = CalibrationStore(tmp_path / "wl", tmp_path / "smile")
    profile = fit_wavelength(centers, [430.0, 475.0, 520.0, 565.0], 2, frame.shape, y_ref)
    store.save_wavelength(profile, setting)                       # what Wavelength *Apply* writes

    smile = SmileCorrectionWidget()
    smile.set_calibration_source(store, lambda: setting)
    applied = []
    smile.applyRequested.connect(applied.append)
    smile._grab_request_id = 7
    smile.accept_snapshot(FrameSnapshot(7, frame, 0.0, 2, True, True, True))
    assert "✓ wavelength_bin2.json" in smile.lbl_quality.text()   # quality analysis needs no file dialog
    assert not smile.btn_apply.isEnabled()
    from modules.smile import analyze_smile
    smile._on_analysis_done(analyze_smile(frame))
    assert smile.btn_apply.isEnabled() and "bin 2" in smile.btn_apply.text()
    path = smile.apply_to_camera()
    assert path == str(store.smile_path(setting)) and applied == [path]
    smile.close()

    w = SpectralResolutionWidget()
    w.set_calibration_source(store, lambda: setting)
    w.set_grab_available(True)
    w.begin_grab(3)
    w.accept_snapshot(FrameSnapshot(3, frame, 0.0, 2, True, True, True))
    assert w.profile is not None and w.smile is not None
    assert w.lbl_wavelength.text().startswith("✓") and w.lbl_smile.text().startswith("✓")
    w.analyze()
    assert _pump(app, lambda: w.result is not None)
    for line in w.result.lines:                                   # smile corrected, so blocks of 16 rows stay sharp
        assert line.n_valid == line.n_blocks and line.fwhm_px_median == pytest.approx(3.0, rel=0.03)
    assert w.table.rowCount() == 4 and "Δλ" in w.summary.text()
    xlsx = w.export_excel(tmp_path / "fwhm")
    import pandas as pd
    sheets = pd.read_excel(xlsx, sheet_name=None)
    assert set(sheets) == {"Summary", "Lines", "Blocks", "Settings"} and len(sheets["Lines"]) == 4
    assert "smile_model_bin2.npz" in " ".join(sheets["Settings"]["value"].astype(str))
    w.close()


def test_smile_apply_is_refused_after_the_camera_changed_bin(app, tmp_path):
    from modules.smile import analyze_smile
    from modules.smile.widget import SmileCorrectionWidget
    frame, _, _ = _lamp()
    current = {"s": CameraSetting("genicam", 0, 2, True, frame.shape)}
    store = CalibrationStore(tmp_path / "wl", tmp_path / "smile")
    smile = SmileCorrectionWidget()
    smile.set_calibration_source(store, lambda: current["s"])
    smile.set_frame(frame, setting=current["s"])
    smile._on_analysis_done(analyze_smile(frame))
    current["s"] = CameraSetting("genicam", 0, 1, True, (128, 800))
    assert smile.apply_to_camera() is None and "different setting" in smile.lbl_status.text()
    assert not store.smile_path(CameraSetting("genicam", 0, 2, True)).exists()
    smile.close()


def test_fwhm_without_smile_measures_rows_one_by_one(app, tmp_path):
    from modules.spectral_resolution.widget import SpectralResolutionWidget
    frame, _, _ = _lamp()
    w = SpectralResolutionWidget()
    w.set_frame(frame, 2)                                         # no store: nothing to load
    assert w.smile is None and w.rows_per_block.value() == 1 and "Load" in w.lbl_smile.text()
    w.close()
