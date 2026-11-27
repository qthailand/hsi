"""Phase 2E.3: saturation measurement through the running acquisition engine, and camera restore.

A simulated camera (two bins; frames depend on gain and exposure, 12 bits << 4 clipping at 65504)
runs in a real AcquisitionEngine. The sweep goes through EngineFrameSource (engine requests + a raw
tap that only takes frames read with the requested setting) and the camera is put back afterwards.
Camera state restoration (10.4) is checked after: success, an inconclusive result, cancel, a camera
timeout, a camera disconnect, an exception while changing the bin, and a failed save. The
calibration window (widget) and the Corrections row are exercised on a simulated session.
"""
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules.acquisition import AcquisitionEngine, AcquisitionSink  # noqa: E402
from modules.acquisition.saturation_measure import EngineFrameSource, RawTap  # noqa: E402
from modules.camera import AbstractCameraBackend  # noqa: E402
from modules.saturation_profile import MeasurementCancelled, SaturationProfileStore, SweepPlan, measure  # noqa: E402

from test_saturation_tracking import _wait  # noqa: E402

H, W = 40, 64


class SimCamera(AbstractCameraBackend):
    """Bins 1 and 2; value = radiance x exposure x gain factor, 12 bits << 4, clipped at 65504.

    ``fail_bin``: set_bin raises for it; ``silent_from_ms``: no frames at that exposure or longer
    (a timeout); ``unplug()``: every later call raises (a disconnect).
    """

    def __init__(self):
        super().__init__()
        self.gain, self.exposure_ms, self._bin = 1, 10.0, 1
        self.fail_bin, self.silent_from_ms, self.unplugged = None, None, False
        self.radiance = np.outer(np.linspace(0.05, 1.0, H), np.linspace(0.2, 1.0, W))

    def unplug(self):
        self.unplugged = True

    def _check(self):
        if self.unplugged:
            raise RuntimeError("camera disconnected")

    def _open_device(self): pass
    def _close_device(self): pass
    def _start_device(self): self._check()
    def _stop_device(self): pass

    def _fetch_raw_frame(self, timeout_ms=1000):
        self._check()
        time.sleep(0.002)
        if self.silent_from_ms is not None and self.exposure_ms >= self.silent_from_ms:
            time.sleep(0.01)
            return None
        signal = self.radiance * self.exposure_ms * (1 + 0.02 * (self.gain - 1)) * 40.0
        counts = np.clip(np.round(signal), 0, 4094).astype(np.int64)
        frame = (counts << 4).astype(np.uint16)
        return frame[::self._bin, ::self._bin].copy() if self._bin > 1 else frame

    def set_params(self, gain, exposure_ms):
        self._check()
        self.gain, self.exposure_ms = gain, float(exposure_ms)

    def frame_rate(self): return None

    def set_bin(self, bin_value):
        self._check()
        if bin_value == self.fail_bin:
            raise RuntimeError("SDK refused the bin")
        self._bin = int(bin_value)

    info = property(lambda s: SimpleNamespace(height=H // s._bin, width=W // s._bin, bit_depth=16,
                                              image_format="RAW16"))
    exposure_range_us = property(lambda s: (100, 200_000))
    gain_range = property(lambda s: (0, 400))
    current_bin = property(lambda s: s._bin)
    def get_bins(self): return (1, 2)
    def sensor_bit_depth(self): return 12
    def model_name(self): return "Sim"
    def serial_number(self): return "S1"
    def dropped_images_count(self): return 0
    def get_temperature(self): return 20.0


@pytest.fixture
def running(tmp_path):
    camera = SimCamera()
    config = SimpleNamespace(smile_map_folder=str(tmp_path), preprocessing_backend="cpu", m=0.5, c=400.0,
                             resolution=1.5, saturation_tracking=False, camera_backend="sim")
    seen = []
    sink = AcquisitionSink()
    sink.frame = lambda frame, fps, timestamp: seen.append(frame)
    engine = AcquisitionEngine(camera, config, sink=sink, profile=False)
    thread = threading.Thread(target=engine.run_until_stopped, daemon=True)
    thread.start()
    assert _wait(lambda: seen)
    yield engine, camera, seen
    engine.request_stop()
    thread.join(5)


def plan(**kw):
    base = dict(bins=(1, 2), gains=(1.0, 4.0), start_exposure_ms=5.0, frames_per_step=3, max_exposure_ms=200.0)
    base.update(kw)
    return SweepPlan(**base)


def source_of(engine, **kw):
    return EngineFrameSource(engine, "sim", settle_frames=1, apply_timeout_s=kw.pop("apply_timeout_s", 5.0), **kw)


def test_raw_tap_takes_only_frames_of_its_setting():
    tap = RawTap(2, target=(1, 1.0, 10.0), skip=1)
    camera = SimpleNamespace(current_bin=1, gain=1, exposure_ms=5.0)
    assert not tap.offer(np.zeros(3), camera)                           # old exposure: ignored
    camera.exposure_ms = 10.0
    assert not tap.offer(np.ones(3), camera)                            # settling frame: dropped
    frame = np.full(3, 7)
    assert not tap.offer(frame, camera)
    assert tap.offer(np.full(3, 8), camera)                             # full
    frame[:] = 0                                                        # the tap holds copies
    assert [int(f[0]) for f in tap.wait(0.1)] == [7, 8]


def test_engine_measurement_is_verified_and_the_camera_is_put_back(running):
    engine, camera, seen = running
    source = source_of(engine)
    before = source.snapshot()
    assert before == {"bin": 1, "gain": 1.0, "exposure_ms": 10.0}
    exposures = []
    records = measure(source, plan(), progress=lambda info: exposures.append(
        (info["bin"], info["gain"], info["exposure_ms"], info["frame"].shape)))
    assert [(r.identity.bin, r.status, r.level) for r in records] == [(1, "verified", 65504), (2, "verified", 65504)]
    assert records[1].identity.serial == "S1" and records[1].identity.pixel_format == "RAW16"
    for b, gain, exposure, shape in exposures:                           # every grabbed frame fits its bin
        assert shape == (H // b, W // b)
    assert source.restore(before) == []
    assert (camera.current_bin, camera.gain, camera.exposure_ms) == (1, 1, 10.0)
    n = len(seen)
    assert _wait(lambda: len(seen) > n + 3)                              # the preview runs on


def test_an_inconclusive_result_also_restores(running):
    engine, camera, _ = running
    source = source_of(engine)
    before = source.snapshot()
    (record,) = measure(source, plan(bins=(2,), gains=(1.0,)))            # one gain: not enough
    assert record.status == "inconclusive"
    assert source.restore(before) == [] and camera.current_bin == 1


def test_cancel_restores_and_returns_nothing(running):
    engine, camera, _ = running
    source = source_of(engine)
    before = source.snapshot()
    cancel = threading.Event()
    with pytest.raises(MeasurementCancelled):
        measure(source, plan(), cancel=cancel, progress=lambda info: cancel.set())
    assert source.restore(before) == [] and (camera.current_bin, camera.exposure_ms) == (1, 10.0)


def test_a_camera_timeout_fails_the_bin_and_restores(running):
    engine, camera, _ = running
    camera.silent_from_ms = 20.0                                         # no frames at long exposures
    source = source_of(engine)
    before = source.snapshot()
    records = measure(source, plan(bins=(1,), max_exposure_ms=80.0))
    assert records[0].status == "failed" and "TimeoutError" in records[0].reasons[0]
    camera.silent_from_ms = None
    assert source.restore(before) == [] and camera.exposure_ms == 10.0


def test_an_exception_changing_the_bin_fails_that_bin_and_restores(running):
    engine, camera, _ = running
    camera.fail_bin = 2
    source = source_of(engine, apply_timeout_s=1.0)
    before = source.snapshot()
    records = measure(source, plan())
    assert [r.status for r in records] == ["verified", "failed"]
    assert "not applied" in records[1].reasons[0]
    assert source.restore(before) == [] and camera.current_bin == 1


def test_a_disconnect_reports_that_the_camera_was_not_restored(running):
    engine, camera, _ = running
    source = source_of(engine, apply_timeout_s=1.0)
    before = source.snapshot()

    def unplug(info):
        if info["exposure_ms"] >= 10.0:
            camera.unplug()

    records = measure(source, plan(bins=(1,)), progress=unplug)
    assert records[0].status == "failed"
    problems = source.restore(before, timeout_s=1.0)
    assert problems and "not restored" in problems[0]                    # the host keeps scans locked


# ---------------------------------------------------------------- the window
@pytest.fixture
def qt_app():
    from PyQt5.QtWidgets import QApplication
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _run(widget, app, timeout=30.0):
    done = []
    widget.measurement_finished.connect(lambda records, problems: done.append((records, problems)))
    assert widget.start()
    end = time.time() + timeout
    while not done and time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    assert done
    return done[0]


def test_window_measures_saves_and_reports(qt_app, tmp_path):
    from modules.saturation_profile import SaturationMeasureWidget
    from modules.saturation_profile.demo import SimulatedSession, SimulatedSource
    widget = SaturationMeasureWidget()
    running = []
    widget.running_changed.connect(running.append)
    source = SimulatedSource()
    widget.set_camera(source.bins(), 1, 1.0, source.gain_limits(), 20.0, source.exposure_limits_ms(), 65520, "sim")
    widget.bins_box.setCurrentIndex(1)                                   # all bins
    widget.gain_high.setValue(4.0)
    store = SaturationProfileStore(tmp_path)
    widget.set_store(store)
    assert not widget.measure_btn.isEnabled()                            # no session provider yet
    widget.set_session_provider(lambda: SimulatedSession(SimulatedSource()))
    records, problems = _run(widget, qt_app)
    assert running == [True, False] and problems == []
    assert [(r.identity.bin, r.status, r.level) for r in records] == [(1, "verified", 65504), (2, "verified", 65504)]
    assert widget._rows[1].result.text() == "65504 ✓" and "level 65504, gain 1-4" in widget._rows[1].toolTip()
    assert widget._rows[1].strip.steps and widget._rows[1].strip.newest is None      # the sweep, drawn
    assert widget.histogram.level == 65504 and "verified" in widget.message.text()
    saved = []
    widget.profiles_saved.connect(saved.append)
    results = widget.save()
    assert len(results) == 2 and all(r.active for r in results) and saved == [results]
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["saturation_RAW16_d16s12_bin1.json",
                                                                "saturation_RAW16_d16s12_bin2.json"]
    widget.close()


def test_the_window_does_not_widen_while_it_measures(qt_app):
    """Regression: the per-bin result was text that grew with every step (and the live image
    was scaled to its own label), so the window stretched while measuring. The sweep is now
    drawn as cells and every text that can be long is kept from asking for width."""
    from modules.saturation_profile import SaturationMeasureWidget
    from modules.saturation_profile.demo import SimulatedSession, SimulatedSource
    widget = SaturationMeasureWidget()
    source = SimulatedSource()
    widget.set_camera(source.bins(), 1, 1.0, source.gain_limits(), 20.0, source.exposure_limits_ms(), 65520)
    widget.bins_box.setCurrentIndex(1)
    widget.gain_high.setValue(4.0)
    widget.resize(900, 560)
    widget.show()
    qt_app.processEvents()
    before = (widget.minimumSizeHint().width(), widget.width())
    during = []
    on_progress = widget._on_progress

    def watched(info):                                   # the widths after every step, while measuring
        on_progress(info)
        qt_app.processEvents()
        during.append((widget.minimumSizeHint().width(), widget.width()))

    widget._on_progress = watched
    widget.set_session_provider(lambda: SimulatedSession(SimulatedSource()))
    _run(widget, qt_app)
    assert len(during) >= 8
    assert max(w for w, _ in during) <= before[0] and {w for _, w in during} == {before[1]}, (before, during)
    rows = widget._rows
    assert all(len(row.strip.steps) >= 4 for row in rows.values())               # every step drawn
    assert all(row.result.width() == 84 for row in rows.values())                # fixed result column
    assert "plateau" in rows[1].toolTip()                                         # details: in the tooltip
    widget.close()


def test_window_refusal_restore_problems_and_a_failed_save(qt_app, tmp_path, monkeypatch):
    from modules.saturation_profile import SaturationMeasureWidget
    from modules.saturation_profile.demo import SimulatedSession, SimulatedSource
    widget = SaturationMeasureWidget()
    source = SimulatedSource()
    widget.set_camera(source.bins(), 1, 1.0, source.gain_limits(), 20.0, source.exposure_limits_ms(), 65520)
    widget.gain_high.setValue(4.0)

    def refuse():
        raise RuntimeError("Not now: a scan is running")

    widget.set_session_provider(refuse)
    assert not widget.start() and "scan is running" in widget.message.text()

    class Broken(SimulatedSession):
        def finish(self):
            return ["bin 2 != 1"]

    widget.set_session_provider(lambda: Broken(SimulatedSource()))
    records, problems = _run(widget, qt_app)
    assert problems == ["bin 2 != 1"] and "NOT restored" in widget.message.text()
    store = SaturationProfileStore(tmp_path)
    widget.set_store(store)

    def full(record):
        raise OSError("disk full")

    monkeypatch.setattr(store, "save", full)
    assert widget.save() == [] and "Not saved" in widget.message.text()
    assert not list(tmp_path.glob("*.json"))
    widget.close()


def test_window_cancel_keeps_nothing(qt_app):
    from modules.saturation_profile import SaturationMeasureWidget
    from modules.saturation_profile.demo import SimulatedSession, SimulatedSource
    widget = SaturationMeasureWidget()
    source = SimulatedSource(delay_s=0.02)
    widget.set_camera(source.bins(), 1, 1.0, source.gain_limits(), 20.0, source.exposure_limits_ms(), 65520)
    widget.set_session_provider(lambda: SimulatedSession(SimulatedSource(delay_s=0.02)))
    widget.set_store(SaturationProfileStore(Path(".")))
    done = []
    widget.measurement_finished.connect(lambda r, p: done.append((r, p)))
    assert widget.start()
    widget.cancel()
    end = time.time() + 20
    while not done and time.time() < end:
        qt_app.processEvents()
        time.sleep(0.01)
    assert done == [([], [])] and "Cancelled" in widget.message.text() and not widget.save_btn.isEnabled()
    widget.close()


def test_corrections_row_shows_the_profile_state(qt_app):
    import config as app_config
    from windows.scan_window import CORRECTION_NAMES, ScanWindow
    win = ScanWindow(app_config.AppConfig())
    assert ("saturation", "Saturation level") in CORRECTION_NAMES and win.correction_rows["saturation"] is win.saturation_card
    opened = []
    win.status_card_clicked.connect(opened.append)
    win.set_saturation_profile_status("verified", "bin 2: saturated pixels read 65504 (gain 1-200)")
    assert win.saturation_card.value.text() == "Measured" and "65504" in win.saturation_card.toolTip()
    win.set_saturation_profile_status("missing", "no saturation profile for bin 3")
    assert win.saturation_card.value.text() == "Missing" and "not checked" in win.saturation_card.toolTip()
    win.set_saturation_profile_status("incompatible", "serial number 'A' != 'B'")
    assert win.saturation_card.value.text() == "Incompatible"
    win.saturation_card.clicked.emit()
    assert opened == ["saturation"]
    win.close()


def test_controller_locks_scans_until_the_calibration_state_is_back():
    from controllers.scan_controller import ScanController
    statuses, cards = [], []
    controller = ScanController.__new__(ScanController)
    controller.win = SimpleNamespace(set_status=statuses.append,
                                     set_saturation_profile_status=lambda *a: cards.append(a[0]),
                                     dark_panel=SimpleNamespace(set_error=statuses.append))
    controller.dark_lib = SimpleNamespace(is_collecting=False)
    controller.session = SimpleNamespace(state="PREVIEW")
    controller.PREVIEW, controller.ERROR = "PREVIEW", "ERROR"
    controller.camera_profile = None
    state = {"dark_path": "dark_b1.npz", "bin": 1, "corrections": {"smile_model": "smile_model_bin1.npz"}}
    controller._calibration_state = lambda: dict(state, corrections=dict(state["corrections"]))
    expected = controller._calibration_state()
    state["corrections"]["smile_model"] = "smile_model_bin2.npz"          # the bin-2 model stayed loaded
    controller._verify_saturation_restore(expected, attempts=0)
    assert controller._saturation_restore_problem and "smile_model" in controller._saturation_restore_problem
    assert cards[-1] == "restore_failed" and "NOT restored" in statuses[-1]
    controller.start_scan()
    assert "not restored" in statuses[-1].lower()                        # scans stay locked
    controller._saturation_restore_problem = None
    controller._saturation_measuring = True
    controller.start_scan()
    assert "measurement running" in statuses[-1]
    controller._saturation_measuring = False
    state["corrections"]["smile_model"] = "smile_model_bin1.npz"
    controller._verify_saturation_restore(expected, attempts=0)
    assert controller._saturation_restore_problem is None and "restored" in statuses[-1]
