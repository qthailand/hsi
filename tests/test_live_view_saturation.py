"""live_view: the live spectrum and histogram flag saturation against a per-pixel ceiling."""
import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_live_spectrum_marks_points_that_reach_their_ceiling():
    _app()
    from modules.live_view import LiveLineSpectrumWidget
    widget = LiveLineSpectrumWidget()
    widget.resize(400, 260)
    spectrum = np.array([1000, 30000, 50600, 50700, 50650, 30000, 51000], np.float32)
    ceiling = np.array([51000, 51000, 51000, 51000, 51000, 51000, 0], np.float32)
    widget.update_spectrum(spectrum, ceiling=ceiling)
    assert widget._n_saturated == 3 and widget._sat_runs == [(2, 5)]
    widget.grab()                                                # paints without error

    widget.update_spectrum(spectrum)                             # no ceiling: no saturation display
    assert widget._n_saturated == 0 and widget._sat_runs == []
    widget.update_spectrum(spectrum, ceiling=ceiling[:3])        # wrong length is ignored
    assert widget._ceiling is None


def test_histogram_counts_saturation_below_full_scale_when_a_ceiling_is_given():
    _app()
    from modules.live_view import HistogramPlot
    hist = HistogramPlot(bit_depth=16)
    frame = np.full((10, 10), 20000, np.uint16)
    frame[:2] = 50900                                            # clipped after dark subtraction
    hist.update_histogram(frame)
    assert hist._sat_pct == 0.0                                  # 99 % of 65535 is never reached
    hist.update_histogram(frame, ceiling=np.full((10, 10), 50920, np.uint16))
    assert abs(hist._sat_pct - 20.0) < 1e-9 and hist._sat_counts.sum() == 20
    hist.grab()
