"""End-to-end smoke test of the application shell with a fake camera and fake stage.

Builds the real MainWindow (so every widget module, the controller, the acquisition thread,
the session and the cube builder are exercised together), starts a scan, and checks that
a cube with *every* recorded frame arrives. No hardware or vendor SDK is needed.
"""
import sys
import time
import types

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence

from hsi_fakes import FakeCamera, H, W


class FakeStage:
    """Stage driver that 'moves' for a moment so frames are recorded during the scan."""
    scan_seconds = 0.6

    def __init__(self, open_on_init=False):
        self._opened = False
        self._axis = types.SimpleNamespace(command_stop=lambda: None)

    def open(self): self._opened = True
    def close(self): self._opened = False
    def set_speed(self, v): pass
    def set_speed_dps(self, v, *a): pass
    def set_speed_mm_s(self, v): pass
    def set_position(self, *a): pass

    def set_degree(self, value):
        time.sleep(self.scan_seconds)

    def move_le_motor_mm(self, value):
        time.sleep(self.scan_seconds)


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    """MainWindow reads a fresh default config, never the developer's live ``config/``, nor the
    project's smile models, darks or camera profiles (a real sensor's model does not fit the fake
    frames)."""
    from dataclasses import asdict
    import config as app_config
    import windows.main_window as main_window_module
    from modules.configuration import ConfigStore
    data = asdict(app_config.AppConfig())
    data.update(m=.29, c=400, smile_map_folder=str(tmp_path / 'smile_maps'),
                dark_frame_folder=str(tmp_path / 'dark_frames'),
                camera_profile_folder=str(tmp_path / 'camera_profiles'))
    directory = tmp_path / 'config'
    ConfigStore(directory).initialize(data)
    monkeypatch.setattr(main_window_module, 'load_app_config', lambda: app_config.load_app_config(str(directory)))
    return directory


@pytest.fixture
def unsaved_answers(monkeypatch):
    """Answers to the Save / Discard / Cancel and the scan-running questions (Discard and Yes
    once the list is used up); ``asked`` collects the titles."""
    from PyQt5.QtWidgets import QMessageBox
    import windows.scan_window as scan_window_module
    answers, asked = [], []
    def ask(default):
        def box(parent, title, text, *args):
            asked.append(title)
            return answers.pop(0) if answers else default
        return box
    monkeypatch.setattr(scan_window_module.QMessageBox, 'warning', ask(QMessageBox.Discard))
    monkeypatch.setattr(scan_window_module.QMessageBox, 'question', ask(QMessageBox.Yes))
    return answers, asked


@pytest.fixture
def main_window(monkeypatch, isolated_config, unsaved_answers):
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import modules.camera.factory as factory
    import modules.camera as camera_package
    monkeypatch.setattr(factory, 'create_backend', lambda config: FakeCamera())
    monkeypatch.setattr(camera_package, 'create_backend', lambda config: FakeCamera())
    smc1 = types.ModuleType('modules.stage.smc1')
    smc1.Stage8SMC1 = smc1.Stage8SMC1Linear = FakeStage
    monkeypatch.setitem(sys.modules, 'modules.stage.smc1', smc1)
    from windows.main_window import MainWindow
    window = MainWindow()
    window.resize(1200, 800)
    yield app, window
    controller = window.scan_window.controller
    controller.cleanup()
    window.close()
    app.processEvents()


def pump(app, condition, timeout=40.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_scan_produces_a_cube_with_every_frame(main_window):
    app, window = main_window
    scan = window.scan_window
    controller = scan.controller
    assert pump(app, lambda: controller.preview_frame_index > 3)          # live preview is running
    assert controller.state == controller.PREVIEW

    camera_before = controller.backend.index
    controller.start_scan()
    assert controller.state == controller.SCANNING
    assert pump(app, lambda: scan._pending_cube is not None)               # cube published after the scan move
    assert controller.state == controller.PREVIEW

    cube = scan._pending_cube
    slit, scans, bins = cube.shape
    assert slit == H and bins > 1
    frames_read = controller.backend.index - camera_before
    assert scans > 0.5 * frames_read                                       # not throttled to the 30 fps display rate
    assert scan.open_cube_viewer_btn.isEnabled()
    column = cube[0, :, 0].astype(np.int64)                                # fake frames count up
    assert np.all(np.diff(column) >= 0) and column[-1] > column[0]          # acquisition order
    wavelengths = scan._pending_wavelengths
    assert wavelengths is not None and len(wavelengths) == bins
    # the real-time preview received the same number of columns as the cube
    assert pump(app, lambda: scan.cube_preview._current_shape[1] == scans, timeout=10)


def test_a_scan_locks_and_folds_the_settings_then_restores_them(main_window):
    app, window = main_window
    window.show()
    scan = window.scan_window
    controller, controls, splitter = scan.controller, scan.camera_controls, scan.settings_splitter
    assert pump(app, lambda: controller.preview_frame_index > 3)
    locked = (controls.gain_spin, controls.exp_spin, controls.bin_combo, controls.keep_spectrogram_check,
              scan.rotation_group, scan.linear_group)
    before = splitter.sizes()
    assert before[0] > 0 and all(w.isEnabled() for w in locked)
    controller.start_scan()
    assert controller.state == controller.SCANNING
    assert splitter.sizes()[0] == 0 and not any(w.isEnabled() for w in locked)
    assert controls.temp_spin.isEnabled()                      # the fan-cooled sensor readout stays readable
    assert pump(app, lambda: '/ ~' in scan.status_info())      # Scan elapsed / ~expected, once the stage moves
    assert pump(app, lambda: scan._pending_cube is not None)
    assert controller.state == controller.PREVIEW
    assert splitter.sizes() == before and all(w.isEnabled() for w in locked)
    assert '/ ~' not in scan.status_info()


def test_f5_and_esc_start_and_stop_a_scan_and_recorded_matches_the_cube(main_window):
    app, window = main_window
    scan = window.scan_window
    controller = scan.controller
    for shortcut, key in ((scan.start_scan_shortcut, Qt.Key_F5), (scan.stop_scan_shortcut, Qt.Key_Escape)):
        assert shortcut.key() == QKeySequence(key)
        assert shortcut.context() == Qt.WidgetWithChildrenShortcut   # only with the focus in this window
    assert pump(app, lambda: controller.preview_frame_index > 3)
    scan.stop_scan_shortcut.activated.emit()                    # Esc in preview: Stop is disabled
    assert controller.state == controller.PREVIEW
    scan.start_scan_shortcut.activated.emit()                   # F5
    assert controller.state == controller.SCANNING and 'Bin ' in scan.cube_preview_group.title()
    assert pump(app, lambda: controller.recording and controller.preview_thread.recorded_count() > 5)
    scan.start_scan_shortcut.activated.emit()                   # F5 again while scanning: nothing
    assert controller.state == controller.SCANNING
    scan.stop_scan_shortcut.activated.emit()                    # Esc: the cube is built from the frames so far
    assert controller.state == controller.PREVIEW and scan._pending_cube is not None
    assert scan._status['recorded'] == scan._pending_cube.shape[1]          # the cube's frames
    assert scan.cube_preview_group.title() == 'Realtime Cube Preview'
    assert pump(app, lambda: controller.stage is None)          # let the stage return home


def test_a_free_scan_records_for_its_time_without_a_stage(main_window, monkeypatch):
    """stage_type "free" (no stage): Start records for the time entered, no stage driver is
    opened, the status never mentions a stage going home, and the cube header says free."""
    app, window = main_window
    scan = window.scan_window
    controller = scan.controller
    monkeypatch.setattr(scan.config, 'stage_type', 'free')
    scan.free_duration_spin.setValue(0.8)
    opened = []
    monkeypatch.setattr(FakeStage, '__init__', lambda self, *a, **k: opened.append(self))
    statuses = []
    show = scan.set_status
    monkeypatch.setattr(scan, 'set_status', lambda text: (statuses.append(text), show(text)))
    assert pump(app, lambda: controller.preview_frame_index > 3)
    started = time.perf_counter()
    controller.start_scan()
    assert controller.state == controller.SCANNING
    countdown = []                                            # the Free Scan box counts down
    assert pump(app, lambda: (countdown.append(scan.free_countdown_label.text())
                              or scan._pending_cube is not None))
    shown = [float(text.split()[0]) for text in countdown if text.endswith(" s")]
    assert shown and shown[0] == 0.8 and min(shown) < 0.5 and shown == sorted(shown, reverse=True)
    assert scan.free_countdown_label.text() == "—"           # back to idle once the cube is built
    assert time.perf_counter() - started >= 0.8               # recorded for the whole time
    assert pump(app, lambda: controller.stage is None, timeout=5)
    assert opened == []                                       # no stage driver at all
    assert not any('home' in str(text) for text in statuses)
    assert controller.state == controller.PREVIEW
    assert controller._raw_metadata['stage_type'] == 'free' and controller._raw_metadata['duration_s'] == 0.8
    header = controller.last_scan_header
    assert header['stage_type'] == 'free' and header['duration_s'] == 0.8
    assert 0.8 <= header['duration_s_recorded'] < 0.8 + 0.25    # the whole time was recorded
    assert 'speed_um_s' not in header and 'degree_per_second' not in header
    column = scan._pending_cube[0, :, 0].astype(np.int64)     # acquisition order, as a linear scan
    assert scan._pending_cube.shape[1] > 1 and np.all(np.diff(column) >= 0)


def test_a_free_scan_stopped_early_keeps_the_frames_so_far(main_window, monkeypatch):
    app, window = main_window
    scan = window.scan_window
    controller = scan.controller
    monkeypatch.setattr(scan.config, 'stage_type', 'free')
    scan.free_duration_spin.setValue(60.0)
    assert pump(app, lambda: controller.preview_frame_index > 3)
    controller.start_scan()
    assert pump(app, lambda: controller.recording)            # the recording started
    assert pump(app, lambda: scan._status['recorded'] > 0 or controller.preview_frame_index > 20, timeout=10)
    started = time.perf_counter()
    scan.stop_scan_shortcut.activated.emit()                  # Esc: the cube is built from the frames so far
    assert controller.state == controller.PREVIEW and scan._pending_cube is not None
    assert pump(app, lambda: controller.stage is None, timeout=5)
    assert time.perf_counter() - started < 5                  # the 60 s wait ended with the stop
    header = controller.last_scan_header
    assert header['duration_s'] == 60.0 and 0 < header['duration_s_recorded'] < 10   # set vs recorded


def test_linear_scan_keeps_acquisition_order_like_the_legacy_swir_app(main_window, monkeypatch):
    """Legacy SWIR_LASTES: linear scans are stacked in frame order, never reversed."""
    app, window = main_window
    scan = window.scan_window
    controller = scan.controller
    monkeypatch.setattr(scan.config, 'stage_type', 'linear')
    assert pump(app, lambda: controller.preview_frame_index > 3)
    controller.start_scan()
    assert pump(app, lambda: scan._pending_cube is not None)
    column = scan._pending_cube[0, :, 0].astype(np.int64)
    assert np.all(np.diff(column) >= 0) and column[-1] > column[0]


def test_a_ticked_scan_is_calibrated_frame_by_frame_and_its_counts_come_back(main_window, monkeypatch):
    """Auto radiometric calibration with a matching reference (the choice is stubbed: the fake
    camera has no serial number or dark reference): every recorded frame is multiplied while the
    scan runs, the cube arrives as reflectance, and its counts are rebuilt exactly."""
    import modules.radiometric.auto as auto
    app, window = main_window
    scan = window.scan_window
    controller = scan.controller
    assert pump(app, lambda: controller.preview_frame_index > 3)
    n_bins = controller.preview_thread.binner_n_bins
    factors = (0.5 + 1e-3 * np.arange(H * n_bins, dtype=np.float32)).reshape(H, n_bins)
    factors.setflags(write=False)
    plan = types.SimpleNamespace(ok=True, wanted=True, reference=types.SimpleNamespace(id="ref_test"))
    final = types.SimpleNamespace(ok=True, header_fields=lambda: {"reflectance_reference_cube": "ref_test"},
                                  summary=lambda: "calibrated with ref_test")
    asked = []
    monkeypatch.setattr(auto, "plan_frame_calibration",
                        lambda *a, **k: asked.append(a) or (plan, auto.FrameCalibration(plan, factors)))
    monkeypatch.setattr(auto, "finish_frame_calibration", lambda frame, meta, **k: (final, []))
    assert scan.auto_radiometric_box.isChecked()
    controller.start_scan()
    assert not scan.auto_radiometric_box.isEnabled()            # chosen at Start for the whole scan
    assert scan.scan_calibration_label.text() == "This scan: calibrated frame by frame with ref_test"
    _store, start_meta, wavelengths, lines = asked[0]
    assert lines == H and len(wavelengths) == n_bins and "sensor_temperature_c_end" not in start_meta
    assert pump(app, lambda: scan.calibrated_cube() is not None)
    reflectance = scan.calibrated_cube()
    assert reflectance.dtype == np.float32 and scan._pending_cube is None
    assert scan.open_reflectance_btn.isVisibleTo(scan)
    assert scan.calibration_header(reflectance)["reflectance_reference_mode"] == "automatic, frame by frame"
    counts = scan.counts_cube()
    assert counts.dtype == np.uint16 and counts.shape == reflectance.shape
    np.testing.assert_array_equal(reflectance, counts.astype(np.float32) * factors[:, None, :])
    column = counts[0, :, 0].astype(np.int64)                    # fake frames count up
    assert np.all(np.diff(column) >= 0) and column[-1] > column[0]
    assert scan.auto_radiometric_box.isEnabled()
    # the next scan, unticked, records counts again
    scan.auto_radiometric_box.setChecked(False)
    assert pump(app, lambda: controller.stage is None)
    controller.start_scan()
    assert pump(app, lambda: scan._pending_cube is not None and scan._pending_cube is not counts)
    assert scan._pending_cube.dtype == np.uint16 and scan.calibrated_cube() is None
    assert len(asked) == 1


def test_a_reflectance_cube_that_does_not_fit_keeps_the_scan_in_counts(main_window, monkeypatch):
    """A frame-by-frame scan whose float32 cube cannot be allocated survives in counts (exact)."""
    import controllers.scan_controller as scan_controller_module
    import modules.radiometric.auto as auto
    app, window = main_window
    scan = window.scan_window
    controller = scan.controller
    assert pump(app, lambda: controller.preview_frame_index > 3)
    n_bins = controller.preview_thread.binner_n_bins
    factors = np.full((H, n_bins), 0.25, np.float32)
    factors.setflags(write=False)
    plan = types.SimpleNamespace(ok=True, wanted=True, reference=types.SimpleNamespace(id="ref_test"))
    final = types.SimpleNamespace(ok=True, header_fields=lambda: {}, summary=lambda: "calibrated with ref_test")
    monkeypatch.setattr(auto, "plan_frame_calibration", lambda *a, **k: (plan, auto.FrameCalibration(plan, factors)))
    monkeypatch.setattr(auto, "finish_frame_calibration", lambda frame, meta, **k: (final, []))
    build = scan_controller_module.create_pushbroom_cube_from_corrected

    def no_room_for_float32(frames):
        if frames and frames[0].dtype == np.float32:
            raise MemoryError("Unable to allocate the float32 cube")
        return build(frames)
    monkeypatch.setattr(scan_controller_module, "create_pushbroom_cube_from_corrected", no_room_for_float32)
    controller.start_scan()
    assert pump(app, lambda: scan._pending_cube is not None)
    cube = scan._pending_cube
    assert cube.dtype == np.uint16 and scan.calibrated_cube() is None
    assert "not enough memory for the reflectance cube" in scan.scan_calibration_label.text()
    column = cube[0, :, 0].astype(np.int64)                     # the counts are intact
    assert np.all(np.diff(column) >= 0) and column[-1] > column[0]


def test_open_calibration_opens_bsq_generator_on_the_scanned_cube(main_window):
    app, window = main_window
    scan = window.scan_window
    controller = scan.controller
    assert pump(app, lambda: controller.preview_frame_index > 3)
    controller.start_scan()
    assert pump(app, lambda: scan._pending_cube is not None)
    assert scan.open_calibration_btn.isEnabled()
    before = len(window.mdi.subWindowList())
    scan.open_calibration_btn.click()
    app.processEvents()
    subs = window.mdi.subWindowList()
    assert len(subs) == before + 1
    from modules.radiometric import CalibrationWindow
    calibration = subs[-1].widget()
    assert isinstance(calibration, CalibrationWindow)
    assert calibration.cubheight == scan._pending_cube.shape[0]
    # a new scan disables the button until its own cube is ready
    assert pump(app, lambda: controller.stage is None)
    controller.start_scan()
    assert not scan.open_calibration_btn.isEnabled()


def test_reopening_scan_reads_the_camera_config_again(main_window, monkeypatch):
    app, window = main_window
    import windows.main_window as main_window_module
    reloaded = []
    original = main_window_module.load_app_config
    monkeypatch.setattr(main_window_module, 'load_app_config', lambda: reloaded.append(1) or original())
    from PyQt5.QtCore import QCoreApplication, QEvent
    window.scan_subwindow.close()
    assert pump(app, lambda: QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete) or window.scan_window is None)
    window.create_scan_window()
    assert reloaded and window.scan_window.controller.config is window.config


def test_cube_viewer_opens_on_the_result(main_window):
    app, window = main_window
    cube = (np.arange(6 * 5 * 4) % 1000).astype(np.uint16).reshape(6, 5, 4)
    window.create_cube_viewer(cube, wavelengths=np.array([450., 500., 550., 600.]), title='Test cube')
    app.processEvents()
    assert len(window.mdi.subWindowList()) >= 2


def _statuses(scan, monkeypatch):
    """Record every status line the controller sets."""
    seen = []
    original = scan.set_status
    monkeypatch.setattr(scan, 'set_status', lambda text: seen.append(text) or original(text))
    return seen


def test_a_camera_read_error_does_not_block_the_next_scan(main_window, monkeypatch):
    """Regression: any worker error used to set ERROR for good, and Start did nothing until a restart."""
    app, window = main_window
    scan = window.scan_window
    controller = scan.controller
    statuses = _statuses(scan, monkeypatch)
    assert pump(app, lambda: controller.preview_frame_index > 3)
    controller.backend.failing_reads = 1
    assert pump(app, lambda: any('Camera error' in s for s in statuses))
    assert controller.state == controller.PREVIEW
    shown = controller.preview_frame_index
    assert pump(app, lambda: controller.preview_frame_index > shown + 3)        # the live preview goes on
    controller.start_scan()
    assert controller.state == controller.SCANNING
    assert pump(app, lambda: scan._pending_cube is not None)


def test_a_read_error_mid_scan_disarms_the_recorder_and_keeps_the_frames(main_window, monkeypatch):
    app, window = main_window
    scan = window.scan_window
    controller = scan.controller
    statuses = _statuses(scan, monkeypatch)
    assert pump(app, lambda: controller.preview_frame_index > 3)
    controller.start_scan()
    assert pump(app, lambda: controller.recording and controller.preview_thread.recorded_count() > 5)
    controller.backend.failing_reads = 1
    assert pump(app, lambda: controller.state == controller.PREVIEW)
    assert not controller.recording and not controller.preview_thread._scanning
    assert scan._pending_cube is not None and scan._pending_cube.shape[1] > 5
    assert any('scan stopped, frames so far kept' in s for s in statuses)
    assert pump(app, lambda: controller.stage is None)          # let the stage return home
    assert 'scan stopped' in statuses[-1] and 'stage returned home' in statuses[-1]   # the reason stays
    controller.start_scan()                                     # and a new scan can start
    assert controller.state == controller.SCANNING


def test_the_worker_ending_on_its_own_is_an_error_state(main_window, monkeypatch):
    """A fatal error ends the acquisition loop: ERROR, and Start says why instead of doing nothing."""
    app, window = main_window
    scan = window.scan_window
    controller = scan.controller
    statuses = _statuses(scan, monkeypatch)
    assert pump(app, lambda: controller.preview_frame_index > 3)
    controller.on_preview_error('preprocessing failed')
    controller.preview_thread.request_stop()                    # the loop ends as after a fatal error
    assert pump(app, lambda: controller.state == controller.ERROR)
    assert 'preprocessing failed' in statuses[-1]
    controller.start_scan()
    assert controller.state == controller.ERROR and 'reopen the Scan window' in statuses[-1]


def test_an_unsaved_scan_is_not_dropped_without_asking(main_window, monkeypatch, unsaved_answers):
    from PyQt5.QtWidgets import QMessageBox
    app, window = main_window
    answers, asked = unsaved_answers
    scan = window.scan_window
    controller = scan.controller
    assert pump(app, lambda: controller.preview_frame_index > 3)
    controller.start_scan()
    assert pump(app, lambda: scan._pending_cube is not None)
    cube = scan._pending_cube
    assert pump(app, lambda: controller.stage is None)
    scan.open_cube_viewer_btn.click()
    viewers_before = len(window.mdi.subWindowList())

    answers.append(QMessageBox.Cancel)                          # Start: Cancel keeps the result
    controller.start_scan()
    assert asked == ['Unsaved Scan'] and controller.state == controller.PREVIEW and scan._pending_cube is cube

    answers.append(QMessageBox.Cancel)                          # closing the app: asked before any window closes
    assert not window.close()
    assert asked[-1] == 'Unsaved Scan' and len(window.mdi.subWindowList()) == viewers_before
    assert window.scan_window is scan

    answers.append(QMessageBox.Save)                            # Save: the cube opens with its export dialog
    controller.start_scan()
    assert controller.state == controller.PREVIEW and len(window.mdi.subWindowList()) == viewers_before + 1
    viewer = window.mdi.subWindowList()[-1].widget()
    assert viewer._export_dlg is not None and viewer._export_dlg.isVisible()
    viewer._export_dlg.close()

    viewer.cube_exported.emit('scan.bip')                       # exported: no more questions
    asked.clear()
    controller.start_scan()
    assert asked == [] and controller.state == controller.SCANNING


def test_every_registered_module_opens_inside_the_shell(main_window, tmp_path, monkeypatch):
    """The menu is built from the registry and each entry opens a working sub-window."""
    import windows.integrations as integrations
    from modules.contracts import CalibrationStore
    store = CalibrationStore(tmp_path / 'wavelength', tmp_path / 'smile')       # never the real config folder
    monkeypatch.setattr(integrations, 'calibration_store', lambda host: store)
    app, window = main_window
    entries = [(label, callback) for integration in window.integrations for label, callback in integration.menu_entries()]
    labels = [label for label, _ in entries]
    assert labels == ['Open BSQ Generator', 'Open Smile Analysis', 'Load Smile Model',
                      'Wavelength Calibration', 'FWHM / Spectral Resolution', 'Reference ROI Selection']
    before = len(window.mdi.subWindowList())
    for label, callback in entries:
        if label == 'Load Smile Model':       # opens a file dialog
            continue
        callback()
        app.processEvents()
    assert len(window.mdi.subWindowList()) == before + 5
    # a wavelength profile applied through the shell reaches the live acquisition worker,
    # and once the worker accepts it, it is stored for the camera setting
    from modules.wavelength_calibration import fit_wavelength
    controller = window.scan_controller
    setting = controller.camera_setting()
    profile = fit_wavelength([0, 100], [400, 458], controller.camera_bin, setting.frame_shape)
    wavelength = next(i for i in window.integrations if isinstance(i, integrations.WavelengthIntegration))
    assert wavelength._apply(profile)
    assert pump(app, lambda: store.wavelength_path(setting).exists(), timeout=20)
    stored = store.load_wavelength(setting)
    assert stored.ok and stored.profile.m == pytest.approx(profile.m)


def test_scan_with_keep_spectrogram_hands_metadata_to_the_viewer_and_file(main_window, tmp_path):
    h5py = pytest.importorskip('h5py')
    app, window = main_window
    scan = window.scan_window
    controller = scan.controller
    assert pump(app, lambda: controller.preview_frame_index > 3)
    scan.camera_controls.keep_spectrogram_check.setChecked(True)
    controller.start_scan()
    assert pump(app, lambda: scan._pending_spectrogram is not None)
    meta = scan._pending_spectrogram_metadata
    assert meta and meta['camera_bin'] == controller.camera_bin and meta['stage_type'] == 'rotation'
    assert meta['frame_height'] == H and 'degree' in meta and 'raw_inverted' in meta
    from modules.cube_io.hdf5 import save_hdf5
    path = save_hdf5(str(tmp_path / 'raw.h5'), scan._pending_spectrogram, metadata=meta)
    with h5py.File(path, 'r') as f:
        assert f['spectrogram'].shape[0] == scan._pending_cube.shape[1]         # one raw frame per cube column
        assert f.attrs['camera_bin'] == controller.camera_bin


def test_missing_camera_shows_an_error_and_keeps_the_main_window(monkeypatch, isolated_config):
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    class MissingCamera(FakeCamera):
        def open(self):
            raise RuntimeError('No camera found')

    import modules.camera as camera_package
    monkeypatch.setattr(camera_package, 'create_backend', lambda config: MissingCamera())
    import windows.main_window as main_window_module
    errors = []
    monkeypatch.setattr(main_window_module.QMessageBox, 'critical', lambda *a: errors.append(a[1:]))
    window = main_window_module.MainWindow()
    assert errors and errors[0][0] == 'Camera Error' and 'No camera found' in errors[0][1]
    assert window.scan_window is None and window.mdi.subWindowList() == []
    assert window.open_envi_action.isEnabled()                            # Open ENVI still reachable
    window.close()
    app.processEvents()


def _config_files():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / 'config'
    return {name: (root / name).read_bytes() for name in ('camera.json', 'scan.json', 'processing.json')}


def test_tests_and_tools_never_write_the_users_settings(main_window):
    """MainWindow() without persist_settings (tests, hardware_smoke) must not touch config/*.json.

    Regression: the fake camera's exposure range (<= 0.1 ms) clamped the user's 1.0 ms, and the
    start-up save wrote 0.1 into the real camera.json.
    """
    app, window = main_window
    before = _config_files()
    scan = window.scan_window
    scan.camera_controls.exp_spin.setValue(0.05)
    scan.degree_spin.setValue(7.0)
    pump(app, lambda: False, 1.6)
    window.scan_window.controller.cleanup()
    assert _config_files() == before


def test_persisted_settings_are_the_users_edits_only(monkeypatch, tmp_path):
    import json
    from dataclasses import asdict
    from PyQt5.QtWidgets import QApplication
    import config as app_config
    import modules.camera as camera_package
    import modules.camera.factory as factory
    import windows.main_window as main_window_module
    from modules.configuration import ConfigStore
    app = QApplication.instance() or QApplication([])
    data = asdict(app_config.AppConfig())
    data.update(m=.29, c=400, default_exposure_ms=1.0, default_bin=2,
                camera_profile_folder=str(tmp_path / 'camera_profiles'))
    directory = tmp_path / 'config'
    ConfigStore(directory).initialize(data)
    monkeypatch.setattr(main_window_module, 'load_app_config', lambda: app_config.load_app_config(str(directory)))
    monkeypatch.setattr(factory, 'create_backend', lambda config: FakeCamera())
    monkeypatch.setattr(camera_package, 'create_backend', lambda config: FakeCamera())
    window = main_window_module.MainWindow(persist_settings=True)
    scan = window.scan_window
    try:
        assert scan.camera_controls.exposure_ms() == pytest.approx(0.1)     # clamped by the fake camera
        pump(app, lambda: False, 1.6)
        camera = json.loads((directory / 'camera.json').read_text())
        assert camera['default_exposure_ms'] == 1.0                          # start-up value not written
        scan.camera_controls.exp_spin.setValue(0.05)
        scan.degree_per_second_spin.setValue(3.5)
        pump(app, lambda: False, 1.6)
        camera = json.loads((directory / 'camera.json').read_text())
        stage = json.loads((directory / 'scan.json').read_text())
        assert camera['default_exposure_ms'] == pytest.approx(0.05) and stage['rotation_speed_dps'] == 3.5
        scan.camera_controls.exp_spin.setValue(0.07)                         # closed before the save delay
    finally:
        scan.controller.cleanup()
        window.close()
        app.processEvents()
    camera = json.loads((directory / 'camera.json').read_text())
    assert camera['default_exposure_ms'] == pytest.approx(0.07)              # flushed on close


def test_two_open_copies_do_not_overwrite_each_others_edits(monkeypatch, tmp_path):
    """Each copy writes only the settings edited in it (regression: 2.00 ms reverted to 1.00 ms)."""
    import json
    from dataclasses import asdict
    from PyQt5.QtWidgets import QApplication
    import config as app_config
    import modules.camera as camera_package
    import modules.camera.factory as factory
    import windows.main_window as main_window_module
    from modules.configuration import ConfigStore

    class WideRangeCamera(FakeCamera):
        exposure_range_us = property(lambda s: (10, 2_000_000))    # 0.01 .. 2000 ms, like the POA

    app = QApplication.instance() or QApplication([])
    data = asdict(app_config.AppConfig())
    data.update(m=.29, c=400, default_exposure_ms=1.0, default_bin=2,
                camera_profile_folder=str(tmp_path / 'camera_profiles'))
    directory = tmp_path / 'config'
    ConfigStore(directory).initialize(data)
    monkeypatch.setattr(main_window_module, 'load_app_config', lambda: app_config.load_app_config(str(directory)))
    monkeypatch.setattr(factory, 'create_backend', lambda config: WideRangeCamera())
    monkeypatch.setattr(camera_package, 'create_backend', lambda config: WideRangeCamera())
    first = main_window_module.MainWindow(persist_settings=True)
    second = main_window_module.MainWindow(persist_settings=True)
    try:
        first.scan_window.camera_controls.exp_spin.setValue(2.0)
        pump(app, lambda: False, 1.4)
        second.scan_window.degree_spin.setValue(12.0)          # this copy still shows 1.0 ms
        pump(app, lambda: False, 1.4)
    finally:
        for window in (first, second):
            window.close()
        app.processEvents()
    camera = json.loads((directory / 'camera.json').read_text())
    stage = json.loads((directory / 'scan.json').read_text())
    assert camera['default_exposure_ms'] == 2.0 and stage['rotation_degree'] == 12.0
    reopened = main_window_module.MainWindow(persist_settings=True)
    try:
        assert reopened.scan_window.camera_controls.exposure_ms() == 2.0
    finally:
        reopened.close()
        app.processEvents()


def test_main_window_asks_for_the_camera_pixel_pitch(main_window):
    """The status bar shows the pixel-pitch chip for the opened camera; File > Camera Pixel Pitch is enabled."""
    app, window = main_window
    chip = window.pixel_pitch_status
    profile = window.scan_window.controller.camera_profile
    assert chip.profile is profile and window.pixel_pitch_action.isEnabled()
    assert chip.state == "shared"                 # FakeCamera reports neither model nor serial
    assert chip.sdk_pitch_um is None              # nor a pixel pitch
    profile.set_pixel_pitch(5.3)
    chip.refresh()
    assert chip.state == "shared" and "5.3" in chip.text()
