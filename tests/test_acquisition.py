"""modules.acquisition: recorder, session and the headless/Qt engine, with a fake camera."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from modules.acquisition import (AcquisitionEngine, AcquisitionSink, ScanRecorder, ScanSession)
from modules.camera import AbstractCameraBackend


# ───────────────────────────── recorder ─────────────────────────────

def test_recorder_records_only_when_armed_and_keeps_raw_aligned():
    r = ScanRecorder()
    assert not r.offer(np.zeros((2, 3)), np.zeros((2, 6)), True)          # not armed
    r.set_recording(True)
    raw = np.zeros((2, 6), np.uint16)
    for i in range(3):
        raw[:] = i
        assert r.offer(np.full((2, 3), i, np.uint16), raw, True)          # raw buffer is reused by the caller
    assert r.count() == 3
    frames, kept = r.take_frames(), r.take_raw()
    assert [int(f[0, 0]) for f in frames] == [int(k[0, 0]) for k in kept] == [0, 1, 2]
    assert not r.enabled and r.count() == 0                                # take disarms and empties


def test_recorder_live_list_and_arming_clears():
    r = ScanRecorder(); r.set_recording(True)
    live = r.live_frames()
    r.offer(np.zeros(1), np.zeros(1), False)
    assert len(live) == 1                                                  # a preview sees it at once
    r.take_frames(); r.set_recording(True)
    assert r.count() == 0 and r.live_frames() is not live                  # arming starts a new list


# ───────────────────────────── session ─────────────────────────────

def test_session_cube_outcome():
    s = ScanSession(); s.begin()
    assert s.state == s.SCANNING and not s.recording
    frames = [np.zeros((4, 3), np.uint16)] * 5
    outcome = s.finish(frames, raw=[])
    assert outcome.kind == 'cube' and outcome.images is frames and s.state == s.PREVIEW
    s.begin(); assert s.finish([], []).kind == 'empty'


def test_session_visual_selection_never_publishes_a_cube():
    s = ScanSession(); s.begin(); s.start_reference_preview(7)
    s.add_reference_column(np.ones((4, 3)))
    outcome = s.finish([np.zeros((4, 3))] * 3, raw=[np.zeros((4, 6))])
    assert outcome.kind == 'reference' and outcome.request_id == 7 and len(outcome.reference_columns) == 1
    assert s.images == [] and s.reference_preview_id is None
    s.begin(); s.start_reference_preview(8)
    out = s.finish([], [])
    assert out.kind == 'reference_failed' and out.request_id == 8 and out.error


def test_session_failure_discards_and_reports_pending_request():
    s = ScanSession(); s.begin(); s.start_reference_preview(3); s.recording = True
    assert s.fail() == 3 and s.state == s.PREVIEW and not s.recording
    s.begin(); assert s.fail() is None


# ───────────────────────────── engine (fake camera) ─────────────────────────────

from hsi_fakes import FakeCamera, W, H  # noqa: E402


class Collector(AcquisitionSink):
    def __init__(self):
        self.frames, self.errors, self.bins, self.failed_bins = [], [], [], []
        self.wavelength, self.rates = [], []

    def frame(self, frame, fps, timestamp): self.frames.append(frame)
    def error(self, message): self.errors.append(message)
    def bin_applied(self, bin_value): self.bins.append(bin_value)
    def bin_failed(self, requested, message): self.failed_bins.append((requested, message))
    def wavelength_failed(self, profile, message): self.wavelength.append(message)
    def frame_rate(self, hz): self.rates.append(hz)


@pytest.fixture
def engine(tmp_path):
    camera = FakeCamera()
    config = SimpleNamespace(smile_map_folder=str(tmp_path), preprocessing_backend='cpu', m=.5, c=400., resolution=2)
    sink = Collector()
    eng = AcquisitionEngine(camera, config, sink=sink, profile=False)
    thread = threading.Thread(target=eng.run_until_stopped, daemon=True)
    thread.start()
    yield eng, camera, sink
    eng.request_stop(); thread.join(5)


def wait_for(condition, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_every_frame_is_recorded_independent_of_the_display_throttle(engine):
    eng, camera, sink = engine
    assert wait_for(lambda: sink.frames)                         # first frame includes the numba warm-up
    eng.set_keep_spectrogram(True); eng.set_recording(True)
    live = eng.live_recorded_frames()
    start = camera.index
    assert wait_for(lambda: camera.index - start > 120)
    frames, raw = eng.take_recorded_frames(), eng.take_recorded_raw()
    assert len(frames) > len(sink.frames) // 4 and len(frames) == len(raw)          # all frames, raw 1:1
    values = [int(r[0, 0]) for r in raw]
    assert all(b - a == 1 for a, b in zip(values, values[1:]))   # none dropped, in order
    assert frames[0].shape[0] == H and frames[0].dtype == np.uint16
    assert frames is live                                        # the live preview read this very list


def test_bin_change_success_and_refusal_keep_the_loop_alive(engine):
    eng, camera, sink = engine
    assert wait_for(lambda: sink.frames)
    camera.refuse = {3}
    eng.request_bin(2); assert wait_for(lambda: sink.bins == [2])
    eng.request_bin(3); assert wait_for(lambda: sink.failed_bins)
    assert sink.failed_bins[0][0] == 3 and 'refused' in sink.failed_bins[0][1]
    assert camera.bin == 2 and not sink.errors                   # previous bin restored, no fatal error
    seen = len(sink.frames)
    assert wait_for(lambda: len(sink.frames) > seen)             # still streaming
    assert sink.frames[-1].shape[0] == H


def test_wavelength_change_is_rejected_while_scanning(engine):
    from modules.wavelength_calibration import fit_wavelength
    eng, camera, sink = engine
    assert wait_for(lambda: sink.frames)
    eng.set_scanning(True)
    eng.request_wavelength(fit_wavelength([0, 100], [400, 458], camera.current_bin))
    assert wait_for(lambda: sink.wavelength) and 'Stop scan' in sink.wavelength[0]


def test_stop_closes_the_camera(engine):
    eng, camera, sink = engine
    assert wait_for(lambda: sink.frames)
    eng.request_stop()
    assert wait_for(lambda: camera.closed)


# ───────────────────────────── Qt wrapper ─────────────────────────────

def test_qt_thread_delivers_signals_and_stops_cleanly(tmp_path):
    from PyQt5.QtWidgets import QApplication
    from modules.acquisition import CameraPreviewThread
    app = QApplication.instance() or QApplication([])
    camera = FakeCamera()
    config = SimpleNamespace(smile_map_folder=str(tmp_path), preprocessing_backend='cpu', m=.5, c=400., resolution=2)
    thread = CameraPreviewThread(camera, config, profile=False)
    frames, errors = [], []
    thread.frame_ready.connect(lambda f, fps, ts: frames.append(f)); thread.error.connect(errors.append)
    thread.start()
    end = time.time() + 20
    while not frames and time.time() < end:
        app.processEvents(); time.sleep(0.01)
    assert frames and not errors
    assert thread.stop(timeout_ms=5000) and camera.closed


# ───────────────────────────── scan metadata stored with the raw spectrogram ─────────────────────────────

def test_raw_metadata_describes_the_scan():
    from modules.acquisition import build_raw_metadata
    camera = SimpleNamespace(info=SimpleNamespace(camera_id=0, bit_depth=16, pixel_size=3.75, width=804, height=552),
                             current_bin=2, gain=1, exposure_ms=1.0)
    config = SimpleNamespace(camera_backend='poa', camera_index=0, raw_inverted=True, m=.2903, c=399.791, resolution=2)
    meta = build_raw_metadata(backend=camera, config=config, scan={'stage_type': 'rotation', 'degree': 10.0, 'degree_per_second': 1.0, 'skipped': None})
    assert meta['camera_bin'] == 2 and meta['gain'] == 1 and meta['exposure_ms'] == 1.0 and meta['raw_inverted'] is True
    assert meta['m_nm_per_px_bin1'] == pytest.approx(.2903) and meta['m_nm_per_px'] == pytest.approx(.5806)
    assert (meta['frame_height'], meta['frame_width']) == (552, 804) and meta['degree'] == 10.0
    assert 'skipped' not in meta and meta['started_at']


@pytest.mark.parametrize('bits,sensor_bits,full_scale', [
    (16, 12, 65520),                 # POA: the SDK scales 12 bits to 16
    (16, 10, 65472),                 # GenICam fxo992: 10 bits, shifted to 16
    (10, 10, 1023),                  # XIMEA: 10 bits, not scaled
])
def test_camera_bit_info_and_raw_metadata_carry_the_real_bits(bits, sensor_bits, full_scale):
    from modules.acquisition import build_raw_metadata, camera_bit_info
    camera = SimpleNamespace(info=SimpleNamespace(camera_id=0, bit_depth=bits, pixel_size=1.0, width=8, height=4),
                             current_bin=1, sensor_bit_depth=lambda: sensor_bits)
    assert camera_bit_info(camera) == {'bit_depth': bits, 'sensor_bit_depth': sensor_bits, 'full_scale': full_scale}
    config = SimpleNamespace(camera_backend='x', camera_index=0, raw_inverted=False, m=.3, c=400., resolution=2)
    meta = build_raw_metadata(backend=camera, config=config)
    assert (meta['bit_depth'], meta['sensor_bit_depth'], meta['full_scale']) == (bits, sensor_bits, full_scale)
    del camera.sensor_bit_depth                                  # a backend without it: no scaling assumed
    assert camera_bit_info(camera)['sensor_bit_depth'] == bits


def test_envi_header_carries_the_camera_bits_only_when_given(tmp_path):
    from modules.cube_io.writers import HypercubeWriter
    cube, wl = np.zeros((3, 4, 5), np.uint16), np.array([400., 402., 404.])
    bits = {'bit_depth': 16, 'sensor_bit_depth': 10, 'full_scale': 65472}
    HypercubeWriter(save_path=str(tmp_path / 'a.bin'), config={'m': .29, 'c': 400, **bits}, hypercube=cube, linear_spectrum=wl).run()
    header = (tmp_path / 'a.bin.hdr').read_text()
    assert 'data bit depth = 16' in header and 'sensor bit depth = 10' in header and 'data full scale = 65472' in header
    HypercubeWriter(save_path=str(tmp_path / 'b.bin'), config={'m': .29, 'c': 400}, hypercube=cube, linear_spectrum=wl).run()
    assert 'bit depth' not in (tmp_path / 'b.bin.hdr').read_text()


def test_hdf5_file_carries_the_metadata_and_data_unchanged(tmp_path):
    h5py = pytest.importorskip('h5py')
    from modules.cube_io.hdf5 import save_hdf5
    data = (np.arange(4 * 6 * 8) % 1000).astype(np.uint16).reshape(4, 6, 8)
    meta = {'camera_bin': 2, 'exposure_ms': 1.5, 'gain': 1, 'raw_inverted': True, 'm_nm_per_px': .5806, 'c_nm': 399.791,
            'camera_backend': 'poa', 'unset': None, 'shape': 'must not override'}
    path = save_hdf5(str(tmp_path / 'raw.h5'), data, metadata=meta)
    with h5py.File(path, 'r') as f:
        np.testing.assert_array_equal(f['spectrogram'][...], data)
        a = f.attrs
        assert (a['camera_bin'], a['exposure_ms'], a['gain'], bool(a['raw_inverted'])) == (2, 1.5, 1, True)
        assert a['m_nm_per_px'] == pytest.approx(.5806) and a['camera_backend'] == 'poa'
        assert 'unset' not in a and tuple(a['shape']) == data.shape and a['dtype'] == 'uint16'
        assert a['format_version'] == 1 and a['axes'] == 'frame,slit,spectral'
    save_hdf5(str(tmp_path / 'plain.h5'), data)                    # still works without metadata
    with h5py.File(tmp_path / 'plain.h5', 'r') as f:
        assert tuple(f.attrs['shape']) == data.shape


def test_bin_is_pending_until_its_outcome_is_sent(engine):
    """Start Scan waits while this is true, else the bin would be applied mid-scan."""
    eng, camera, sink = engine
    assert wait_for(lambda: sink.frames) and not eng.bin_pending
    entered, release = threading.Event(), threading.Event()
    set_bin = camera.set_bin
    def slow_set_bin(value):
        entered.set(); release.wait(5); set_bin(value)
    camera.set_bin = slow_set_bin
    pending_when_sent = []
    sink.bin_applied = lambda value: (pending_when_sent.append(eng.bin_pending), sink.bins.append(value))
    eng.request_bin(2)
    assert eng.bin_pending                                        # queued (or already taken)
    assert entered.wait(5) and eng.bin_pending                    # being applied
    release.set()
    assert wait_for(lambda: sink.bins == [2]) and wait_for(lambda: not eng.bin_pending)
    assert pending_when_sent == [True]                            # cleared only after the outcome
    camera.set_bin = set_bin; camera.refuse = {3}
    eng.request_bin(3)
    assert wait_for(lambda: sink.failed_bins) and wait_for(lambda: not eng.bin_pending)


def test_bin_that_cannot_stream_falls_back_to_the_previous_bin(engine):
    """Real GenICam case: the bin is accepted but AcquisitionStart fails (GC_ERR_IO)."""
    eng, camera, sink = engine
    assert wait_for(lambda: sink.frames)
    camera.no_stream_bins = {3}
    eng.request_bin(3)
    assert wait_for(lambda: sink.failed_bins)
    assert sink.failed_bins[0][0] == 3 and 'GC_ERR_IO' in sink.failed_bins[0][1]
    assert camera.bin == 1 and not sink.errors                    # back on the previous bin, loop alive
    seen = len(sink.frames)
    assert wait_for(lambda: len(sink.frames) > seen)


def test_frame_rate_is_reported_at_start_and_after_every_bin_or_exposure_change(engine):
    eng, camera, sink = engine
    assert wait_for(lambda: sink.frames) and sink.rates == [camera.frame_rate()]
    eng.request_bin(2); assert wait_for(lambda: len(sink.rates) == 2)
    eng.request_params(1, 5.0); assert wait_for(lambda: len(sink.rates) == 3)
    camera.refuse = {3}
    eng.request_bin(3); assert wait_for(lambda: len(sink.rates) == 4)        # also when the bin fails
    assert sink.rates[1] == 1000.0 / 2 and sink.rates[2] == 200.0


# ───────────────────────────── saturation ceiling ─────────────────────────────

class CeilingCollector(Collector):
    def __init__(self):
        super().__init__()
        self.ceilings = []

    def saturation_ceiling(self, ceiling): self.ceilings.append(ceiling)


def test_saturation_ceiling_is_full_scale_minus_dark_in_display_space(tmp_path):
    camera = FakeCamera()                                       # 12-bit: a clipped pixel reads 4095
    config = SimpleNamespace(smile_map_folder=str(tmp_path), preprocessing_backend='cpu', m=.5, c=400., resolution=2)
    sink = CeilingCollector()
    eng = AcquisitionEngine(camera, config, sink=sink, profile=False)
    thread = threading.Thread(target=eng.run_until_stopped, daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: sink.frames and sink.ceilings)
        first = sink.ceilings[0]
        assert first.shape == sink.frames[0].shape and int(first.min()) == int(first.max()) == 4095
        count = len(sink.ceilings)
        time.sleep(0.2)
        assert len(sink.ceilings) == count                      # sent only when something changes

        dark = np.full((H, W), 1000, np.uint16)
        dark[:, :W // 2] = 500
        camera.set_dark_reference(dark, None)
        assert wait_for(lambda: len(sink.ceilings) > count)
        ceiling = sink.ceilings[-1]
        assert ceiling.shape == sink.frames[-1].shape
        assert set(np.unique(ceiling).tolist()) <= set(range(3095, 3596))   # 4095 - dark, binned
        assert int(ceiling[:, 0].max()) == 3595 and int(ceiling[:, -1].min()) == 3095

        count = len(sink.ceilings)
        camera.set_dark_enabled(False)
        assert wait_for(lambda: len(sink.ceilings) > count)
        assert int(sink.ceilings[-1].min()) == 4095
    finally:
        eng.request_stop(); thread.join(5)


def test_saturated_mask_uses_each_pixels_own_ceiling():
    from modules.live_view import saturated_mask
    values = np.array([50600, 50000, 65520, 10, 0])
    ceiling = np.array([51000, 51000, 65520, 0, 0])
    assert saturated_mask(values, ceiling).tolist() == [True, False, True, False, False]


def test_genicam_saturation_level_follows_the_transmitted_bits():
    from modules.camera.genicam import GenICamBackend
    camera = GenICamBackend.__new__(GenICamBackend)
    camera._info = SimpleNamespace(bit_depth=16)
    for bits, level in ((12, 65520), (10, 65472), (14, 65532), (16, 65535)):
        camera._source_bits = bits
        assert camera.saturation_level() == level
