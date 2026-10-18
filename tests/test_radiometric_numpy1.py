"""BSQ generator jobs under NumPy 1.x type promotion (the Orin), where uint64 // int gives float64.

The crop box comes from the image axes; with NumPy 1.x rules a uint64 crop made
``Create Datacube`` and the temp cubes fail with "'numpy.float64' object cannot be
interpreted as an integer".
"""
import warnings

import numpy as np
import pytest

L, S, B = 12, 16, 5


@pytest.fixture
def numpy1_rules():
    if np.lib.NumpyVersion(np.__version__) < '2.0.0':
        yield                                       # NumPy 1.x: these are the rules already
        return
    if not hasattr(np, '_set_promotion_state'):
        pytest.skip('this NumPy cannot emulate NumPy 1.x promotion')
    old = np._get_promotion_state()
    np._set_promotion_state('legacy')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            yield
    finally:
        np._set_promotion_state(old)


@pytest.fixture
def window(monkeypatch):
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import modules.radiometric.widget as widget
    errors = []
    monkeypatch.setattr(widget.QMessageBox, 'information', staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(widget.QMessageBox, 'warning', staticmethod(lambda *a, **k: errors.append(a[-1])))
    win = widget.CalibrationWindow()
    cube = np.arange(L * S * B, dtype=np.float32).reshape(L, S, B)
    win.load_from_memory(cube_array=cube, wavelengths=np.arange(B) * 2.0 + 400, name='Numpy1Test')
    yield app, win, cube, errors, widget
    win.close()
    from modules.cube_io.envi import release_band_maps
    release_band_maps()
    app.processEvents()


class _Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)


def run_jobs_on_this_thread(win, monkeypatch):
    """The emulated promotion rules are per thread: run the window's jobs here, not on a QThread."""
    def run(func, *args, **kwargs):
        worker = type('W', (), {})()
        worker.finished = _Signal()
        def finish():
            result = func(*args, **kwargs)
            for slot in worker.finished.slots:
                slot(result)
        win._pending_job = finish
        return worker, None
    monkeypatch.setattr(win, '_run_in_thread', run)
    win.thread = None


def test_create_datacube_and_temp_cube_work_with_numpy1_rules(window, numpy1_rules, tmp_path, monkeypatch):
    app, win, cube, errors, widget = window
    run_jobs_on_this_thread(win, monkeypatch)
    out = tmp_path / 'out.bip'
    monkeypatch.setattr(widget.QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), '')))
    win.event_create()
    win._pending_job()
    assert not errors
    written = np.fromfile(out, dtype=np.float32).reshape(L, S, B)
    np.testing.assert_array_equal(written, cube)

    win.ensure_raw_memmap()
    win._pending_job()
    assert win.bsq_raw_shape == (L, S, B)
    win.bsq_raw_mm = None                           # release the temp file
