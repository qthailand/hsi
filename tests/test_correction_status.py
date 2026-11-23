"""Correction indicators describe processed frames and avoid repeat signals."""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from PyQt5.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.status_card import StatusCard, PALETTE

def reporter():
    """The real engine method, driven with fakes (no camera, no Qt thread)."""
    from modules.acquisition import AcquisitionEngine, AcquisitionSink
    events = []
    class Sink(AcquisitionSink):
        def correction_status(self, status):
            events.append(status)
    worker = AcquisitionEngine.__new__(AcquisitionEngine)
    worker.backend = SimpleNamespace(current_bin=1, _dark_mean=None, _dark_enabled=True)
    worker._smile = SimpleNamespace(calibration=None, path=None)
    worker._last_correction_status = None
    worker.sink = Sink()
    return (lambda w, *a, **k: w._report_corrections(*a, **k)), worker, events

def test_transitions_and_no_duplicate_events():
    report, worker, events = reporter()
    frame = SimpleNamespace(shape=(10, 20))
    report(worker, frame); report(worker, frame)
    assert len(events) == 1 and events[-1]['dark'][0] == 'Missing'
    worker.backend._dark_mean = SimpleNamespace(shape=(10, 20))
    worker._smile.calibration = object(); worker._smile.path = 'smile_model_bin1.npz'
    report(worker, frame)
    assert events[-1]['dark'][0] == events[-1]['smile'][0] == 'Applied'
    worker.backend._dark_enabled = False
    report(worker, frame)
    assert events[-1]['dark'][0] == 'Off'
    worker.backend._dark_enabled = True
    worker.backend._dark_mean.shape = (20, 10)
    report(worker, frame)
    assert events[-1]['dark'] == ('Missing', 'Reference shape mismatch')
    report(worker, frame, dark_capture=True)
    assert events[-1]['dark'][0] == 'Capturing' and events[-1]['smile'][0] == 'Waiting'
    report(worker, frame, smile_error=True)
    assert events[-1]['smile'][0] == 'Missing'

def test_generic_card_accepts_arbitrary_values_and_palette():
    app = QApplication.instance() or QApplication([])
    card = StatusCard('Storage')
    for color in PALETTE:
        card.set_content('42 GB', 'Available capacity', color)
        assert card.value.text() == '42 GB' and card._last[-1] == color
    long_detail = 'x' * 100
    card.set_content('Ready', long_detail, 'green')
    assert long_detail in card.toolTip() and len(card.detail.text()) < 40
    with pytest.raises(ValueError):
        card.set_content('Ready', color='purple')


def test_status_card_value_is_a_switch():
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from modules.status_card import StatusCard
    card = StatusCard("Dark")
    card.set_compact(92)
    card.set_clickable(True)
    card.show()
    app.processEvents()
    clicks, switches = [], []
    card.clicked.connect(lambda: clicks.append(1))
    card.value_clicked.connect(lambda: switches.append(1))
    QTest.mouseClick(card.value, Qt.LeftButton)                    # not a switch yet: a card click
    assert clicks == [1] and switches == []
    card.set_togglable(True, "Click to switch")
    assert card.is_togglable() and card.switch_mark.isVisible()
    QTest.mouseClick(card.value, Qt.LeftButton)                    # the value is the switch now
    assert switches == [1] and clicks == [1]
    QTest.mouseClick(card, Qt.LeftButton, pos=QPoint(4, card.height() - 4))   # elsewhere: still the card
    assert clicks == [1, 1]
    card.set_togglable(False)
    assert not card.switch_mark.isVisible()
    card.close()


def test_engine_geometry_switches_pick_the_model():
    from types import SimpleNamespace
    from modules.acquisition.engine import AcquisitionEngine
    smile_only = object()
    cal = SimpleNamespace(has_keystone=True, smile_only=lambda: smile_only)
    eng = AcquisitionEngine.__new__(AcquisitionEngine)
    import threading
    eng._lock = threading.Lock()
    eng._smile = SimpleNamespace(calibration=cal)
    eng.set_geometry(True, True)
    assert eng._active_calibration() is cal
    eng.set_geometry(True, False)
    assert eng._active_calibration() is smile_only and eng._active_calibration() is smile_only   # cached
    eng.set_geometry(False, True)
    assert eng._active_calibration() is None and eng.geometry() == (False, True)
    eng._smile = SimpleNamespace(calibration=None)
    eng.set_geometry(True, True)
    assert eng._active_calibration() is None
