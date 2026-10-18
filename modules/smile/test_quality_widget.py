"""GUI integration checks for diagnostic tabs and asynchronous results."""
import json

import numpy as np
import pytest

pytest.importorskip('PyQt5')
pytest.importorskip('pytestqt')
from smile.widget import SmileCorrectionWidget
from smile.analysis import analyze_smile


def test_quality_tabs_placeholders_and_calibration(qtbot, tmp_path):
    widget = SmileCorrectionWidget()
    qtbot.addWidget(widget)
    assert widget.tabs.count() == 8
    assert widget.tabs.tabText(4) == 'Wavelength'
    assert not widget.btn_quality.isEnabled()
    config = tmp_path/'wave.json'
    config.write_text(json.dumps(dict(m_nm_per_px=.5, c_nm=500,
                                      standards=[dict(wavelength_nm=540,x_px=80)])))
    widget.load_wavelength_calibration(str(config))
    assert widget.get_quality_report() is None
    assert not widget.btn_quality.isEnabled()
    for canvas in widget._canvases[4:]:
        assert canvas.fig.axes[0].texts


def test_quality_worker_plots_and_export(qtbot, synthetic_frame, tmp_path):
    widget = SmileCorrectionWidget()
    qtbot.addWidget(widget)
    widget.set_frame(synthetic_frame)
    result = analyze_smile(synthetic_frame, interp='linear')
    widget._on_analysis_done(result)
    reference = result.used_lines[0]
    config = tmp_path/'wave.json'
    config.write_text(json.dumps(dict(m_nm_per_px=.5, c_nm=500,
        standards=[dict(wavelength_nm=500+.5*reference.column,
                        x_px=float(reference.column),half_window_px=20)])))
    widget.load_wavelength_calibration(str(config))
    widget.run_quality_analysis()
    qtbot.waitUntil(lambda: widget._quality_worker is None, timeout=60000)
    assert widget.get_quality_report() is not None, widget.lbl_quality.text()
    assert widget.tabs.currentIndex() == 4
    assert len(widget._canvases[5].fig.axes) == 3
    for index in range(4,8):
        path = tmp_path/f'tab_{index}.png'
        widget._canvases[index].fig.savefig(path)
        assert path.stat().st_size > 1000
    widget.set_frame(synthetic_frame)
    assert widget.get_quality_report() is None
    assert not widget.btn_quality.isEnabled()
