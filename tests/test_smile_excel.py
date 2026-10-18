"""modules.smile.excel_report: the smile result and quality analysis exported as .xlsx."""
import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("openpyxl")

from modules.smile import analyze_smile  # noqa: E402
from modules.smile.excel_report import write_smile_excel  # noqa: E402
from modules.smile.quality import SpectralReference, analyze_quality, estimate_shift_uncertainty  # noqa: E402


@pytest.fixture(scope="module")
def result():
    h, w = 200, 300
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    frame = np.random.default_rng(0).normal(100, 2, (h, w)).astype(np.float32)
    for column in (50, 100, 150, 200, 250):
        frame += 3000 * np.exp(-((x - column - 5e-5 * (y - h / 2) ** 2) ** 2) / 4.5)
    analysis = analyze_smile(frame)
    refs = [SpectralReference(400 + 0.5 * c, c, 12) for c in (50, 100, 150, 200, 250)]
    report = analyze_quality(analysis.frame_orig, analysis.frame_corr, refs, m_nm_per_px=0.5, c_nm=400,
                             row_step=8, y_ref=int(analysis.y_ref), saturation=analysis.sat_level)
    unc = estimate_shift_uncertainty(analysis, rows=report.rows[:, None],
                                     columns=np.array([r.column_px for r in refs])[None, :], repeats=20)
    return analysis, report, unc


def test_smile_only_workbook(result, tmp_path):
    analysis, _, _ = result
    path = write_smile_excel(tmp_path / "smile", analysis, metadata={"source": "lamp.png"})
    assert path.endswith(".xlsx")
    sheets = pd.read_excel(path, sheet_name=None)
    assert {"Summary", "Lines", "Traces", "Measurements", "Surface", "Metadata"} <= set(sheets)
    assert "Quality rows" not in sheets                                   # quality not run
    summary = dict(zip(sheets["Summary"]["parameter"], sheets["Summary"]["value"]))
    assert float(summary["RMS after"]) == pytest.approx(analysis.rms_after)
    assert int(summary["Lines used"]) == len(analysis.used_lines)
    assert len(sheets["Lines"]) == len(analysis.all_lines)
    assert len(sheets["Traces"]) == analysis.frame_orig.shape[0]
    m = sheets["Measurements"]
    assert len(m) == analysis.meas_v.size
    np.testing.assert_allclose(m["residual_px"], analysis.residuals, atol=1e-9)
    assert len(sheets["Surface"]) == analysis.surface.coef.size
    assert "lamp.png" in sheets["Metadata"]["value"].astype(str).tolist()


def test_workbook_with_quality_and_uncertainty(result, tmp_path):
    analysis, report, unc = result
    sheets = pd.read_excel(write_smile_excel(tmp_path / "q.xlsx", analysis, report, unc), sheet_name=None)
    rows = sheets["Quality rows"]
    assert len(rows) == report.rows.size * report.wavelengths_nm.size
    assert {"wavelength_error_nm_before", "wavelength_error_nm_after", "fwhm_nm_after"} <= set(rows.columns)
    got = rows["wavelength_error_nm_after"].to_numpy(dtype=float)
    np.testing.assert_allclose(got, report.after["wavelength_error_nm"].reshape(-1), equal_nan=True)
    q = sheets["Quality summary"].set_index("metric")
    assert q.loc["paired_wavelength_after_nm", "rmse"] == pytest.approx(
        report.summary["paired_wavelength_after_nm"]["rmse"])
    assert len(sheets["Uncertainty"]) == len(rows) and "std_px" in sheets["Uncertainty"]
    assert len(sheets["Quality by line"]) == report.wavelengths_nm.size


def test_widget_exports_after_analysis(result, tmp_path):
    from PyQt5.QtWidgets import QApplication
    from modules.smile.widget import SmileCorrectionWidget
    app = QApplication.instance() or QApplication([])  # noqa: F841
    widget = SmileCorrectionWidget()
    assert not widget.btn_export_excel.isEnabled()
    widget._on_analysis_done(result[0])
    assert widget.btn_export_excel.isEnabled()
    path = widget.export_excel(str(tmp_path / "w"))
    assert "Lines" in pd.read_excel(path, sheet_name=None)
    widget.close()
