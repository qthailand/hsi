"""Display frames are pulled by the GUI: newest only, and the GUI keeps idle time for dialogs."""
import threading
import time
from types import SimpleNamespace

import numpy as np
from PyQt5.QtWidgets import QApplication

from hsi_fakes import FakeCamera
from modules.acquisition import CameraPreviewThread
from modules.acquisition.engine import MIN_FRAME_INTERVAL


def _thread(tmp_path):
    config = SimpleNamespace(smile_map_folder=str(tmp_path), preprocessing_backend='cpu', m=.5, c=400., resolution=2)
    return CameraPreviewThread(FakeCamera(), config, profile=False)   # never started: frames are offered by hand


def _pump(app, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.002)


def test_only_the_newest_frame_is_delivered_and_never_twice(tmp_path):
    app = QApplication.instance() or QApplication([])
    thread = _thread(tmp_path)
    got = []
    thread.frame_ready.connect(lambda frame, fps, ts: got.append(int(frame[0, 0])))
    for value in (1, 2, 3):
        thread._offer_display_frame(np.full((2, 2), value, np.uint16), 30.0, time.monotonic())
    _pump(app, 0.01)
    assert got == [3]
    thread._offer_display_frame(np.full((2, 2), 4, np.uint16), 30.0, time.monotonic())
    app.processEvents()
    assert got == [3]                                 # still resting after frame 3
    _pump(app, MIN_FRAME_INTERVAL + 0.05)
    assert got == [3, 4]
    _pump(app, MIN_FRAME_INTERVAL + 0.05)
    assert got == [3, 4]                              # nothing new: nothing repeated
    assert thread.stop() and thread._display_latest is None


def test_a_slow_gui_gets_fewer_frames_and_stays_half_idle(tmp_path):
    app = QApplication.instance() or QApplication([])
    thread = _thread(tmp_path)
    busy = 0.04
    starts, lags = [], []

    def slow_slot(frame, fps, ts):
        starts.append(time.perf_counter())
        lags.append(time.monotonic() - ts)
        time.sleep(busy)                              # a GUI that needs 40 ms per frame

    thread.frame_ready.connect(slow_slot)
    stop = threading.Event()

    def worker():                                     # offers frames far faster than the GUI can show
        while not stop.is_set():
            thread._offer_display_frame(np.zeros((2, 2), np.uint16), 200.0, time.monotonic())
            time.sleep(0.005)

    feeder = threading.Thread(target=worker, daemon=True)
    feeder.start()
    try:
        _pump(app, 1.0)
    finally:
        stop.set()
        feeder.join(2)
    gaps = np.diff(starts)
    assert len(starts) >= 5
    assert gaps.min() >= 2 * busy * 0.9               # at most half of the time goes to the display
    assert max(lags) < 0.2                            # no backlog: every shown frame is recent
    thread.stop()
