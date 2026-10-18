"""FWHM panel: select peaks, then calculate (Gaussian/Lorentzian, baseline removers).

Covers the peak selection (automatic from the wavelength profile or by detection, clicks in
nm that must land on the right pixel and snap to the maximum, no duplicates, remove/clear,
dragging), the end-to-end calculation with the Lorentzian profile and every baseline method,
clearing the result when anything changes, the Excel settings, and the baseline plumbing of
``measure_frame`` / ``fit_spectrum`` on a strongly curved background.
"""
import dataclasses
import time

import numpy as np
import pytest
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QValidator

from modules.contracts import CalibrationStore, CameraSetting
from modules.spectral_resolution import (BASELINE_DEFAULTS, BASELINE_LABELS, BASELINE_METHODS, PROFILES,
                                         baseline_settings, fit_spectrum, measure_frame, measure_spectrum,
                                         row_blocks, usable_columns)

X = np.arange(400.0)
CENTERS = [60.3, 150.3, 240.3, 330.3]
WAVELENGTHS = [430.0, 475.0, 520.0, 565.0]        # m = 0.5 nm/px, c = 399.85 nm: nm and px differ a lot


def lorentz_line(center, fwhm, height, x=X):
    """Pixel-integrated Lorentzian of peak ``height`` (what a detector records)."""
    g = fwhm / 2.0
    return height * g * (np.arctan((x + .5 - center) / g) - np.arctan((x - .5 - center) / g))


def lamp_frame(rows=32, fwhm=3.0, seed=21, border=0):
    """Smile-free lamp frame: Lorentzian lines on a sloped and curved background (+ optional zero border)."""
    background = 200 + 0.8 * X + 3e-3 * (X - 200) ** 2
    spectrum = background + sum(lorentz_line(c, fwhm, 3000) for c in CENTERS)
    frame = spectrum[None, :] + np.random.default_rng(seed).normal(0, 3, (rows, X.size))
    frame[:, :border] = 0
    return frame.astype(np.float32)


def profile_for(frame, pixels=CENTERS, wavelengths=WAVELENGTHS):
    """Wavelength profile measured on ``frame`` (bin 1, reference row = centre row)."""
    from modules.contracts import default_reference_row
    from modules.wavelength_calibration import fit_wavelength
    return fit_wavelength(list(pixels), list(wavelengths), 1, frame.shape, default_reference_row(frame.shape[0]))


def same(a, b):
    """Two lists of LineMeasurement are equal field by field (NaN equal to NaN)."""
    def key(w):
        return tuple("nan" if isinstance(v, float) and np.isnan(v) else v for v in dataclasses.astuple(w))
    return [key(w) for w in a] == [key(w) for w in b]


@pytest.fixture
def app():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def pump(app, condition, timeout=30.0):
    """Process Qt events until ``condition()`` or ``timeout`` seconds."""
    end = time.time() + timeout
    while time.time() < end and not condition():
        app.processEvents()
        time.sleep(0.005)
    return condition()


def settle(app, seconds=0.05):
    """Process events for a moment (deferred line rebuilds use a 0 ms timer)."""
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()


def widget_with_profile(app, tmp_path, frame, profile=None, show=False):
    """A widget that received ``frame`` for a camera setting whose profile is stored."""
    from modules.spectral_resolution.widget import SpectralResolutionWidget
    setting = CameraSetting("genicam", 0, 1, True, frame.shape)
    store = CalibrationStore(tmp_path / "wl", tmp_path / "smile")
    store.save_wavelength(profile or profile_for(frame), setting)
    w = SpectralResolutionWidget()
    w.set_calibration_source(store, lambda: setting)
    if show:
        w.resize(1200, 800)
        w.show()
        settle(app)
    w.set_frame(frame, 1, setting=setting)
    settle(app)
    return w


def calculate(app, w):
    """Press Calculate FWHM and wait for the result."""
    assert w.btn_analyze.isEnabled()
    w.analyze()
    assert w._worker is not None and not w.fit_box.isEnabled() and not w.peaks_panel.isEnabled()
    assert pump(app, lambda: w.result is not None and w._worker is None)
    assert w.fit_box.isEnabled() and w.peaks_panel.isEnabled()
    return w.result


class FakeClick:
    """Just enough of pyqtgraph's MouseClickEvent for the scene-click handler."""

    def __init__(self, scene_pos, button=Qt.LeftButton):
        self._pos, self._button = scene_pos, button

    def button(self):
        return self._button

    def double(self):
        return False

    def scenePos(self):  # noqa: N802 (pyqtgraph API)
        return self._pos


def click_view(w, x_view):
    """Simulate a left click on the spectrum plot at plot x ``x_view`` (mid height)."""
    box = w.spectrum_plot.getPlotItem().getViewBox()
    (_, _), (y0, y1) = box.viewRange()
    w._on_scene_clicked(FakeClick(box.mapViewToScene(QPointF(float(x_view), 0.5 * (y0 + y1)))))


# ───────────────────────────── automatic selection ─────────────────────────────

def test_calibration_lines_inside_the_usable_columns_are_preselected(app, tmp_path):
    from modules.frame_exchange import FrameSnapshot
    frame = lamp_frame(border=30)                                 # zero-filled border like a smile-corrected frame
    profile = profile_for(frame, [12.0] + CENTERS, [405.85] + WAVELENGTHS)   # first point lies in the border
    w = widget_with_profile(app, tmp_path, lamp_frame(), profile)
    w.set_grab_available(True)
    w.begin_grab(5)
    w.accept_snapshot(FrameSnapshot(5, frame, 0.0, 1, True, True, True))
    assert w.profile is not None and w.columns == (30, 399)
    assert w.peaks == CENTERS and w._selection_source == "calibration"
    assert [w.peak_list.item(i).text() for i in range(4)] == [
        "P1 • 430.00 nm (px 60.3)", "P2 • 475.00 nm (px 150.3)", "P3 • 520.00 nm (px 240.3)",
        "P4 • 565.00 nm (px 330.3)"]
    assert "4 peaks × 32 row blocks" in w.plan.text() and "Gaussian" in w.plan.text()
    assert w.btn_calibration.isEnabled() and w.btn_analyze.isEnabled()
    result = calculate(app, w)                                    # no user interaction needed
    assert [l.pixel for l in result.lines] == CENTERS and [l.wavelength_nm for l in result.lines] == WAVELENGTHS
    w.close()


def test_peaks_are_detected_without_a_profile(app):
    from modules.spectral_resolution.widget import SpectralResolutionWidget
    w = SpectralResolutionWidget()
    w.set_frame(lamp_frame(), 1)                                  # no store: no profile, no smile
    assert w.profile is None and w._selection_source == "detected" and not w.btn_calibration.isEnabled()
    assert len(w.peaks) == 4 and np.allclose(w.peaks, CENTERS, atol=0.3)
    assert w.peak_list.item(0).text().startswith("P1 • px 60.")
    assert w.preview_rows == (8, 23) and "reference row 16" in w.lbl_preview.text()   # rows around the centre
    w.set_baseline("arpls")
    assert w.auto_detect_peaks() == 4                             # on the baseline-corrected preview
    w.min_prominence.setValue(1e5)
    assert w.auto_detect_peaks() == 0 and not w.btn_analyze.isEnabled()
    w.min_prominence.setValue(20)
    w.min_distance.setValue(100)                                  # lines are 90 px apart: keep every second
    assert w.auto_detect_peaks() == 2
    w.close()


def test_loading_a_profile_later_replaces_an_automatic_selection_but_keeps_a_manual_one(app, tmp_path):
    from modules.spectral_resolution.widget import SpectralResolutionWidget
    frame = lamp_frame()
    path = tmp_path / "profile.json"
    profile_for(frame, CENTERS[:3], WAVELENGTHS[:3]).save(path)
    w = SpectralResolutionWidget()
    w.set_frame(frame, 1)
    assert len(w.peaks) == 4 and w._selection_source == "detected"
    w.load_wavelength_file(str(path))
    assert w.peaks == CENTERS[:3] and w._selection_source == "calibration"
    assert w.spectrum_plot.getPlotItem().getAxis("bottom").labelText == "Wavelength (nm)"
    w.set_frame(frame, 1)                                         # new frame: profile gone, detected again
    w.remove_selected_peaks()
    w.select_peaks([3])
    w.remove_selected_peaks()                                     # now a manual selection of 3 peaks
    w.load_wavelength_file(str(path))
    assert w._selection_source == "manual" and len(w.peaks) == 3
    assert w.peak_list.item(0).text().startswith("P1 • 430.00 nm")   # relabelled with the profile
    w.close()


def test_preview_is_the_mean_of_all_rows_after_smile_correction(app, tmp_path):
    from modules.smile import analyze_smile, calibration_from_analysis, save_calibration
    from modules.spectral_resolution.widget import SpectralResolutionWidget
    y = np.arange(64)[:, None].astype(float)
    centers = [60.3, 150.3, 240.3, 330.3]
    frame = np.full((64, 400), 100.0)
    for c in centers:                                             # curved (smile) lines
        frame += np.vstack([lorentz_line(c + 4e-4 * (r - 31) ** 2, 3.0, 3000) for r in y[:, 0]])
    frame = (frame + np.random.default_rng(5).normal(0, 3, frame.shape)).astype(np.float32)
    setting = CameraSetting("genicam", 0, 2, True, frame.shape)
    store = CalibrationStore(tmp_path / "wl", tmp_path / "smile")
    store.smile_path(setting).parent.mkdir(parents=True, exist_ok=True)
    save_calibration(str(store.smile_path(setting)), calibration_from_analysis(analyze_smile(frame)))
    w = SpectralResolutionWidget()
    w.set_calibration_source(store, lambda: setting)
    w.set_frame(frame, 2, setting=setting)
    assert w.smile is not None and w.preview_rows == (0, 63) and "all 64 rows" in w.lbl_preview.text()
    assert len(w.peaks) == 4 and np.allclose(w.peaks, centers, atol=0.4)
    w.close()


# ───────────────────────────── editing the selection ─────────────────────────────

def test_click_in_nm_adds_the_right_pixel_snapped_to_the_maximum(app, tmp_path):
    w = widget_with_profile(app, tmp_path, lamp_frame(), show=True)
    m, c = w.profile.m, w.profile.c
    assert m == pytest.approx(0.5) and c == pytest.approx(399.85)
    assert w.spectrum_plot.getPlotItem().getAxis("bottom").labelText == "Wavelength (nm)"
    w.clear_peaks()
    assert w.peaks == [] and not w.btn_analyze.isEnabled()
    click_view(w, m * 151.5 + c)                                  # 1.2 px right of the line at 150.3
    assert len(w.peaks) == 1 and w.peaks[0] == pytest.approx(150.3, abs=0.25)
    assert w.peak_list.item(0).text() == "P1 • 475.00 nm (px 150.3)"   # within 1.5 px: the known wavelength
    assert w.selected_indices() == [0]
    click_view(w, m * 151.8 + c)                                  # 1.5 px from P1: selects it, no duplicate
    assert len(w.peaks) == 1 and w.selected_indices() == [0]
    click_view(w, m * 242.6 + c)                                  # snaps 2.3 px left to 240.3
    click_view(w, m * 61.0 + c)
    assert np.allclose(w.peaks, [60.3, 150.3, 240.3], atol=0.25)  # numbered in ascending wavelength
    assert [w.peak_list.item(i).text()[:2] for i in range(3)] == ["P1", "P2", "P3"]
    assert w.selected_indices() == [0] and w._selection_source == "manual"
    assert [line.label.format for line in w._peak_lines] == ["P1", "P2", "P3"]
    assert [line.value() for line in w._peak_lines] == pytest.approx([m * p + c for p in w.peaks])
    box = w.spectrum_plot.getPlotItem().getViewBox()             # clicks outside the data area are ignored
    w._on_scene_clicked(FakeClick(box.sceneBoundingRect().bottomRight() + QPointF(5, 5)))
    w._on_scene_clicked(FakeClick(box.mapViewToScene(QPointF(m * 330.3 + c, 0)), Qt.RightButton))
    assert len(w.peaks) == 3
    w.add_peak(197.0, snap=False)                                 # off-peak position, kept as given
    assert 197.0 in w.peaks and w.peak_list.item(2).text() == f"P3 • {m * 197 + c:.2f} nm (px 197.0)"
    assert w.add_peak(1.0) is None and len(w.peaks) == 4          # outside the usable columns (2 px margin)
    w.close()


def test_remove_selected_and_clear(app, tmp_path):
    w = widget_with_profile(app, tmp_path, lamp_frame())
    assert len(w.peaks) == 4
    w.select_peaks([0, 2])
    assert w.selected_indices() == [0, 2]
    w.remove_selected_peaks()
    assert w.peaks == [150.3, 330.3] and w.peak_list.count() == 2 and len(w._peak_lines) == 2
    assert w.peak_list.item(0).text().startswith("P1 • 475.00 nm")    # renumbered
    assert w.selected_indices() == []
    w.remove_selected_peaks()                                     # nothing selected: nothing removed
    assert len(w.peaks) == 2
    w.clear_peaks()
    assert w.peaks == [] and w.peak_list.count() == 0 and w._peak_lines == []
    assert not w.btn_analyze.isEnabled() and "No peaks selected" in w.plan.text()
    w.analyze()                                                   # refused without a modal dialog
    assert w._worker is None and "No peaks selected" in w.status.text()
    assert w.use_calibration_lines() == 4 and w.peaks == CENTERS
    w.close()


def test_dragging_a_peak_line_snaps_on_release_and_refuses_duplicates(app, tmp_path):
    w = widget_with_profile(app, tmp_path, lamp_frame())
    w.set_peaks([145.0, 240.3])
    line = w._peak_lines[0]
    line.setValue(float(w.px_to_view(148.6)))
    line.sigPositionChangeFinished.emit(line)
    settle(app)
    assert w.peaks[0] == pytest.approx(150.3, abs=0.25) and w.peaks[1] == 240.3
    assert w._peak_lines[0].value() == pytest.approx(float(w.px_to_view(w.peaks[0])))
    assert w.selected_indices() == [0] and w._selection_source == "manual"
    before = w.peaks
    line = w._peak_lines[1]
    line.setValue(float(w.px_to_view(151.0)))                     # onto P1
    line.sigPositionChangeFinished.emit(line)
    settle(app)
    assert w.peaks == before and "put back" in w.status.text()
    assert w._peak_lines[1].value() == pytest.approx(float(w.px_to_view(240.3)))
    w.close()


# ───────────────────────────── calculation ─────────────────────────────

@pytest.mark.parametrize("method", BASELINE_METHODS)
def test_lorentzian_with_every_baseline_method_end_to_end(app, tmp_path, method):
    w = widget_with_profile(app, tmp_path, lamp_frame())
    w.rows_per_block.setValue(8)                                  # smile-free frame: averaging is fine
    w.set_fit_profile("lorentzian")
    w.set_baseline(method)
    assert w.baseline_settings() == baseline_settings(method)     # the controls start at the defaults
    for name, page in w._baseline_pages.items():
        assert page.isVisibleTo(w) == (name == method)            # only this method's parameters
    if method != "none":
        assert np.ptp(w.preview_baseline[w.columns[0]:w.columns[1] + 1]) > 100   # live preview baseline
    assert f"4 peaks × 4 row blocks · Lorentzian · {'no baseline' if method == 'none' else 'baseline ' + BASELINE_LABELS[method]}" \
        in w.plan.text()
    result = calculate(app, w)
    assert result.settings["profile"] == "lorentzian" and result.settings["window_fwhm"] == 5.0   # auto window
    assert result.settings["baseline"] == method
    assert result.settings["baseline_params"] == baseline_settings(method)[1]
    assert set(result.settings["baseline_params"]) == set(BASELINE_DEFAULTS[method])
    for line in result.lines:
        assert line.n_valid == line.n_blocks == 4 and line.fwhm_px_median == pytest.approx(3.0, rel=0.01)
        assert line.fwhm_nm_median == pytest.approx(1.5, rel=0.01)
    assert w.table.rowCount() == 4 and w.table.item(0, 0).text() == "P1" and "Lorentzian" in w.summary.text()
    fit = w.preview_fit                                           # fitted components drawn on the preview
    assert fit is not None and {win.profile for win in fit.windows} == {"lorentzian"}
    assert sorted(i for win in fit.windows for i in win.lines if i >= 0) == [0, 1, 2, 3]
    assert len(w._fit_items) >= 2 * len(fit.windows) and w.btn_export.isEnabled()
    w.close()


def test_any_change_clears_the_result(app, tmp_path):
    w = widget_with_profile(app, tmp_path, lamp_frame())
    w.rows_per_block.setValue(8)

    def cleared():
        return (w.result is None and w.preview_fit is None and w.table.rowCount() == 0 and w.summary.text() == ""
                and not w.btn_export.isEnabled() and w._fit_items == [] and "Calculate FWHM" in w.status.text())

    changes = [lambda: w.set_fit_profile("lorentzian"),
               lambda: w.set_baseline("asls"),
               lambda: w._baseline_inputs["asls"]["lam"].set_real(1e7),       # a parameter control
               lambda: w._baseline_inputs["asls"]["diff_order"].setValue(1),
               lambda: w.click_at(float(w.px_to_view(197.0))),               # selection
               lambda: w.remove_selected_peaks(),
               lambda: w.rows_per_block.setValue(16),
               lambda: w.window_fwhm.setValue(4.0)]
    for change in changes:
        calculate(app, w)
        assert w.btn_export.isEnabled() and w.table.rowCount() == len(w.peaks)
        change()
        assert cleared()
        with pytest.raises(RuntimeError):
            w.export_excel(tmp_path / "stale")
    calculate(app, w)
    assert w.result.settings["baseline_params"]["lam"] == 1e7 and w.result.settings["window_fwhm"] == 4.0
    w.analyze()                                                   # a change while it runs: result discarded
    w.set_fit_profile("gaussian")
    assert pump(app, lambda: w._worker is None)
    assert w.result is None and "discarded" in w.status.text()
    w.close()


def test_excel_records_profile_baseline_and_selected_peaks(app, tmp_path):
    import pandas as pd
    w = widget_with_profile(app, tmp_path, lamp_frame())
    w.rows_per_block.setValue(8)
    w.add_peak(197.0, snap=False)
    w.set_fit_profile("lorentzian")
    w.set_baseline("arpls", lam=1e4)
    calculate(app, w)
    sheets = pd.read_excel(w.export_excel(tmp_path / "fwhm"), sheet_name=None)
    assert set(sheets) == {"Summary", "Lines", "Blocks", "Settings"}
    settings = dict(zip(sheets["Settings"]["name"], sheets["Settings"]["value"]))
    assert settings["profile"] == "lorentzian" and settings["baseline"] == "arpls"
    assert settings["baseline_label"] == "arPLS"
    assert float(settings["baseline_params.lam"]) == 1e4 and int(settings["baseline_params.diff_order"]) == 2
    assert {"baseline_params.max_iter", "baseline_params.tol"} <= set(settings)
    assert float(settings["selected_peaks.P1.pixel"]) == pytest.approx(60.3)
    assert float(settings["selected_peaks.P1.wavelength_nm"]) == 430.0
    assert settings["selected_peaks.P1.source"] == "calibration line"
    assert float(settings["selected_peaks.P3.pixel"]) == 197.0 and settings["selected_peaks.P3.source"] == "m·px + c"
    assert float(settings["selected_peaks.P3.wavelength_nm"]) == pytest.approx(0.5 * 197 + 399.85)
    lines = sheets["Lines"]
    assert list(lines["peak"]) == ["P1", "P2", "P3", "P4", "P5"] and set(lines["profile"]) == {"lorentzian"}
    assert lines["wavelength_nm"][2] == pytest.approx(0.5 * 197 + 399.85)
    assert set(sheets["Blocks"]["profile"]) == {"lorentzian"}
    summary = dict(zip(sheets["Summary"]["parameter"], sheets["Summary"]["value"]))
    assert summary["profile"] == "lorentzian" and summary["baseline"] == "arpls"
    w.close()


def test_layout_log_spin_box_and_close_while_running(app, tmp_path):
    from modules.spectral_resolution.widget import SpectralResolutionWidget
    w = SpectralResolutionWidget()
    assert w.minimumSizeHint().width() <= 900                     # embeddable: no huge minimum width
    for label in (w.lbl_wavelength, w.lbl_smile, w.plan, w.status, w.lbl_preview, w.summary):
        assert label.wordWrap()
    lam = w._baseline_inputs["arpls"]["lam"]
    assert lam.real_value() == 1e5 and lam.text() == "1e5"
    lam.lineEdit().setText("3e4")
    lam.interpretText()
    assert lam.real_value() == 3e4 and lam.text() == "3e4"
    lam.stepUp()                                                  # half a decade: 3e4 * 10 ** 0.5 = 94868
    assert lam.text() == "9.49e4" and lam.real_value() == 94900.0   # the value used is the value shown
    assert lam.validate("1e", 2)[0] == QValidator.Intermediate and lam.validate("abc", 3)[0] == QValidator.Invalid
    assert w._baseline_inputs["asls"]["p"].text() == "0.01"
    with pytest.raises(ValueError):
        w.set_baseline("whittaker")
    with pytest.raises(ValueError):
        w.set_baseline("arpls", porder=2)
    with pytest.raises(ValueError):
        w.set_fit_profile("voigt")
    w.close()
    w = widget_with_profile(app, tmp_path, lamp_frame(rows=64))
    w.analyze()
    worker = w._worker
    w.close()                                                     # waits for the thread
    assert worker.isFinished()


# ───────────────────────────── core: baseline plumbing ─────────────────────────────

def test_measure_frame_removes_a_strong_curved_background():
    x = np.arange(600.0)
    truth = [(120.3, 3.0), (250.3, 4.0), (380.3, 5.0), (500.3, 3.5)]
    background = 400 + 6000 * np.exp(-0.5 * ((x - 310) / 90) ** 2)    # steep wings under the middle lines
    frame = (background + sum(lorentz_line(c, f, 2500, x) for c, f in truth))[None, :]
    frame = frame + np.random.default_rng(5).normal(0, 5, (48, x.size))
    pixels = [round(c) for c, _ in truth]

    def errors(result):
        return np.array([l.fwhm_px_median / f - 1 for l, (_, f) in zip(result.lines, truth)])

    plain = measure_frame(frame, pixels, rows_per_block=16, profile="lorentzian")
    assert plain.settings["baseline"] == "none" and plain.settings["baseline_params"] == {}
    assert all(l.n_valid == 3 for l in plain.lines) and np.abs(errors(plain)).max() < 0.08   # still works
    for method, params in (("arpls", {"lam": 1e3}), ("imodpoly", {"degree": 8})):
        corrected = measure_frame(frame, pixels, rows_per_block=16, profile="lorentzian", baseline=method,
                                  baseline_params=params)
        assert all(l.n_valid == 3 for l in corrected.lines)
        assert np.abs(errors(corrected)).max() < 0.015 < np.abs(errors(plain)).max()
        assert corrected.settings["baseline"] == method
        assert corrected.settings["baseline_params"] == dict(BASELINE_DEFAULTS[method], **params)
    explicit = measure_frame(frame, pixels, rows_per_block=16, profile="lorentzian", baseline="NONE")
    assert all(same(a, b) for a, b in zip(explicit.measurements, plain.measurements))


def test_baseline_uses_only_usable_columns_and_saturation_uses_raw_counts():
    frame = np.full((20, X.size), 0.0) + (300 + 2e-2 * (X - 200) ** 2)[None, :] + lorentz_line(200.3, 3, 2000)[None, :]
    frame[:, :30] = 0                                             # smile border
    frame[3, 199:202] = 4095                                      # one clipped row
    spectrum = frame[:16].mean(axis=0)
    fit = fit_spectrum(spectrum, [200], valid_columns=usable_columns(frame), baseline="asls",
                       clipped=frame[:16].max(axis=0) >= 0.98 * 4095, profile="lorentzian")
    assert fit.valid_columns == (30, 399) and np.all(fit.baseline[:30] == 0) and np.all(fit.corrected[:30] == 0)
    assert fit.measurements[0].flag == "saturated"                # judged on the raw rows, not after removal
    clean = frame[4:20].mean(axis=0)
    fit = fit_spectrum(clean, [200], valid_columns=(30, 399), baseline="asls", profile="lorentzian")
    w = fit.measurements[0]
    assert w.flag == "ok" and w.fwhm_px == pytest.approx(3.0, rel=0.01)
    window = fit.windows[0]
    x = np.arange(window.first, window.last + 1, dtype=float)
    assert window.lines == (0,) and window.fwhm_px(0) == pytest.approx(w.fwhm_px)
    assert np.abs(window.total(x) - fit.corrected[window.first:window.last + 1]).max() < 1e-3 * 2000
    assert same(measure_spectrum(clean, [200], valid_columns=(30, 399), baseline="asls", profile="lorentzian"),
                fit.measurements)


def test_invalid_baseline_is_rejected_before_any_block():
    steps = []
    frame = lamp_frame()
    for method, params in (("whittaker", None), ("arpls", {"lam": -1}), ("asls", {"porder": 2}), ("none", {"lam": 1}),
                           ("arpls", [("lam", 1)])):
        with pytest.raises(ValueError):
            measure_frame(frame, CENTERS, baseline=method, baseline_params=params, progress=lambda f, m: steps.append(f))
    assert steps == []
    assert baseline_settings("ZhangFit", {"lam": 10}) == ("zhangfit", {"lam": 10.0, "diff_order": 1, "max_iter": 15,
                                                                       "tol": 1e-3})
    assert PROFILES == ("gaussian", "lorentzian") and row_blocks(35, 16) == [(0, 15), (16, 34)]


# ───────────────────────────── review regressions ─────────────────────────────

def smile_frame(curvature=3e-3, rows=64):
    """Lamp frame with curved lines (flat 100 counts) and a smile model of it (right border ~5 px)."""
    from modules.smile import analyze_smile, calibration_from_analysis
    frame = np.full((rows, X.size), 100.0)
    for c in CENTERS:
        frame += np.vstack([lorentz_line(c + curvature * (r - rows // 2 + 1) ** 2, 3.0, 3000) for r in range(rows)])
    frame = (frame + np.random.default_rng(5).normal(0, 3, frame.shape)).astype(np.float32)
    return frame, calibration_from_analysis(analyze_smile(frame))


def test_cancel_and_close_stop_a_long_calculation(app):
    from modules.spectral_resolution.widget import SpectralResolutionWidget
    w = SpectralResolutionWidget()
    w.set_frame(lamp_frame(rows=600), 1)                          # no smile model: 600 single-row blocks
    assert w.rows_per_block.value() == 1 and not w.btn_cancel.isEnabled()
    w.analyze()
    assert w.btn_cancel.isEnabled() and not w.btn_analyze.isEnabled()
    assert pump(app, lambda: w.progress.value() >= 2)             # running
    started = time.perf_counter()
    w.cancel()
    assert pump(app, lambda: w._worker is None, timeout=5)
    assert time.perf_counter() - started < 2.0
    assert w.result is None and "cancelled" in w.status.text()
    assert w.btn_analyze.isEnabled() and not w.btn_cancel.isEnabled() and w.fit_box.isEnabled()
    w.analyze()
    assert pump(app, lambda: w.progress.value() >= 2)
    worker = w._worker
    started = time.perf_counter()
    w.close()                                                     # stops the run instead of waiting for its end
    assert worker.isFinished() and time.perf_counter() - started < 2.0


def test_clicks_on_the_legend_do_not_add_peaks(app, tmp_path):
    from PyQt5.QtTest import QTest
    w = widget_with_profile(app, tmp_path, lamp_frame(), show=True)
    w.set_baseline("arpls")                                       # three legend entries
    settle(app, 0.1)
    before = w.peaks
    viewport = w.spectrum_plot.viewport()

    def click(scene_point):
        pos = w.spectrum_plot.mapFromScene(scene_point)
        QTest.mouseMove(viewport, pos)
        settle(app, 0.02)
        QTest.mouseClick(viewport, Qt.LeftButton, Qt.NoModifier, pos)
        settle(app, 0.1)

    click(w._legend.sceneBoundingRect().center())
    assert w.peaks == before
    sample = w._legend.items[0][0]
    visible = sample.item.isVisible()
    click(sample.sceneBoundingRect().center())                    # toggles that curve, adds nothing
    assert w.peaks == before and sample.item.isVisible() != visible
    box = w.spectrum_plot.getPlotItem().getViewBox()
    (_, _), (y0, y1) = box.viewRange()
    click(box.mapViewToScene(QPointF(float(w.px_to_view(197.0)), y0 + 0.1 * (y1 - y0))))   # on the data
    assert len(w.peaks) == len(before) + 1
    w.close()


def test_rows_per_block_returns_with_a_smile_model_and_the_status_says_why(app, tmp_path):
    from modules.frame_exchange import FrameSnapshot
    from modules.smile import save_calibration
    from modules.spectral_resolution.widget import SpectralResolutionWidget
    frame, model = smile_frame(curvature=4e-4)
    setting = CameraSetting("genicam", 0, 1, True, frame.shape)
    store = CalibrationStore(tmp_path / "wl", tmp_path / "smile")
    path = store.smile_path(setting)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = SpectralResolutionWidget()
    w.set_calibration_source(store, lambda: setting)
    w.set_grab_available(True)

    def grab(request_id):
        w.begin_grab(request_id)
        w.accept_snapshot(FrameSnapshot(request_id, frame, 0.0, 1, True, True, True))
        return w.status.text()

    text = grab(1)
    assert w.smile is None and w.rows_per_block.value() == 1
    assert "no smile model for this setting" in text and "measured one by one" in text
    assert "smile correction applied" not in text and "4 peaks selected" in text
    save_calibration(str(path), model)
    text = grab(2)
    assert w.smile is not None and w.rows_per_block.value() == 16      # was left at 1
    assert "smile model applied" in text and "16 rows per block again" in text
    w.rows_per_block.setValue(8)                                  # the user's choice is what comes back
    path.unlink()
    grab(3)
    assert w.rows_per_block.value() == 1
    save_calibration(str(path), model)
    grab(4)
    assert w.rows_per_block.value() == 8
    w.close()


def test_partly_zero_filled_smile_border_stays_out_of_the_preview_and_the_fit(app, tmp_path):
    from modules.smile import save_calibration
    from modules.spectral_resolution.widget import SpectralResolutionWidget
    frame, model = smile_frame()
    setting = CameraSetting("genicam", 0, 1, True, frame.shape)
    store = CalibrationStore(tmp_path / "wl", tmp_path / "smile")
    store.smile_path(setting).parent.mkdir(parents=True, exist_ok=True)
    save_calibration(str(store.smile_path(setting)), model)
    w = SpectralResolutionWidget()
    w.set_calibration_source(store, lambda: setting)
    w.set_frame(frame, 1, setting=setting)
    left, right = w.smile.margin
    first, last = w.columns
    assert (first, last) == (left, X.size - 1 - right) and last < X.size - 3   # the model's valid range
    assert w.analysis_kwargs()["valid_columns"] == (first, last)
    assert w.preview[first:last + 1].min() > 90                   # no partly zero-filled column inside
    for method in ("arpls", "modpoly", "asls"):
        w.set_baseline(method)
        base = w.preview_baseline[first:last + 1]
        assert base[-5:].min() > 95, method                        # no dive at the border (true ~101)
        assert np.allclose(w.detect_peaks(), CENTERS, atol=0.4)   # no spurious edge peak
    w.close()


def test_a_profile_or_smile_model_needs_a_frame_first(app, tmp_path):
    from modules.spectral_resolution.widget import SpectralResolutionWidget
    path = tmp_path / "profile.json"
    profile_for(lamp_frame()).save(path)
    w = SpectralResolutionWidget()
    w.load_wavelength_file(str(path))                             # used to show ✓ and drop it on the next frame
    assert w.profile is None and "frame first" in w.status.text() and w.lbl_wavelength.text() == "—"
    w.load_smile_file(str(tmp_path / "smile.npz"))
    assert w.smile is None and "frame first" in w.status.text()
    w.set_frame(lamp_frame(), 1)
    w.load_wavelength_file(str(path))
    assert w.profile is not None and w.peaks == CENTERS
    w.close()


def test_log_spin_boxes_show_every_value_and_enter_keeps_the_result(app, tmp_path):
    from PyQt5.QtTest import QTest
    from modules.spectral_resolution.widget import _sci
    values = (1e-4, 5e-4, 3.16e-4, 0.00099999, 0.001, 0.01, 0.5, 250, 9990, 9999.99, 1e5, 316228, 1e-8)
    assert [_sci(v) for v in values] == ["1e-4", "5e-4", "3.16e-4", "0.001", "0.001", "0.01", "0.5", "250",
                                         "9990", "1e4", "1e5", "3.16e5", "1e-8"]
    w = widget_with_profile(app, tmp_path, lamp_frame())
    w.rows_per_block.setValue(8)
    w.set_baseline("arpls")
    tol = w._baseline_inputs["arpls"]["tol"]
    tol.stepDown()                                                # 1e-3 -> 1e-4 used to show an empty box
    assert tol.text() == "1e-4" and w.baseline_settings()[1]["tol"] == 1e-4
    p = w._baseline_inputs["asls"]["p"]
    texts = []
    for _ in range(3):
        p.stepDown()
        texts.append(p.text())
    assert texts == ["0.00316", "0.001", "3.16e-4"] and p.real_value() == 3.16e-4
    lam = w._baseline_inputs["arpls"]["lam"]
    lam.stepUp()                                                  # 1e5 -> 316228, shown as 3.16e5
    assert lam.text() == "3.16e5" and w.baseline_settings()[1]["lam"] == 316000.0
    result = calculate(app, w)
    assert result.settings["baseline_params"]["lam"] == 316000.0  # the value shown is the value used
    for control in (lam, w.snr_min, w.rows_per_block):
        QTest.keyClick(control.lineEdit(), Qt.Key_Return)         # Enter on an unchanged value
        assert w.result is result and w.btn_export.isEnabled()
    assert lam.text() == "3.16e5" and w.baseline_settings()[1]["lam"] == 316000.0
    w.snr_min.setValue(12)                                        # a real change still clears it
    assert w.result is None
    w.close()


def test_peak_lines_are_reused_when_the_selection_changes(app, tmp_path):
    w = widget_with_profile(app, tmp_path, lamp_frame())
    lines = list(w._peak_lines)
    assert len(lines) == 4
    w.add_peak(197.0, snap=False)                                 # becomes P3: only one line is created
    assert w._peak_lines[:4] == lines and len(w._peak_lines) == 5
    assert [line.label.format for line in w._peak_lines] == ["P1", "P2", "P3", "P4", "P5"]
    assert [line.value() for line in w._peak_lines] == pytest.approx([float(w.px_to_view(p)) for p in w.peaks])
    assert [mark.value() for mark in w._image_lines] == pytest.approx(w.peaks)
    w.select_peaks([0])
    w.remove_selected_peaks()
    assert w._peak_lines == lines[:4] and w._peak_lines[0].value() == pytest.approx(float(w.px_to_view(150.3)))
    w.clear_peaks()
    items = w.spectrum_plot.getPlotItem().items
    assert w._peak_lines == [] and w._image_lines == [] and not any(line in items for line in lines)
    w.close()


def test_widget_module_doctests_can_be_collected():
    import doctest
    import modules.spectral_resolution.widget as widget
    doctest.DocTestFinder().find(widget)                          # raised ValueError (directive on a comment line)


def test_save_fwhm_stores_the_median_for_the_frame_setting(app, tmp_path):
    frame = lamp_frame()
    w = widget_with_profile(app, tmp_path, frame)
    assert not w.btn_store.isEnabled()
    result = calculate(app, w)
    assert w.btn_store.isEnabled()
    stored_paths = []
    w.resolutionStored.connect(stored_paths.append)
    path = w.store_resolution()
    assert stored_paths == [path] and path.endswith("resolution_bin1.json")
    setting = CameraSetting("genicam", 0, 1, True, frame.shape)
    stored = w._store.load_resolution(setting)
    assert stored.ok and stored.fwhm_nm == pytest.approx(result.summary()["fwhm_nm_median"])
    assert len(stored.details["lines"]) == len(w.peaks) and stored.details["profile"] == w.fit_profile()
    w._setting_provider = lambda: dataclasses.replace(setting, raw_inverted=False)   # camera changed since the grab
    assert w.store_resolution() is None and "different setting" in w.status.text()
    w.set_fit_profile("lorentzian")                                                 # any change clears the result
    assert not w.btn_store.isEnabled()
    w.close()
