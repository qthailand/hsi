"""Contract every camera backend must satisfy, run against simulated vendor SDKs.

POA, XIMEA and GenICam are driven through small fake SDKs, so the *same* assertions
check that all three behave identically from the application's point of view
(see ``modules/camera/README.md`` for how to add a backend and run this suite on it).
"""
import contextlib
import importlib
import sys
import threading
import types
from types import SimpleNamespace

import numpy as np
import pytest

from modules.camera import AbstractCameraBackend, create_backend

SENSOR_W, SENSOR_H = 640, 480


# ───────────────────────────── fake POA ─────────────────────────────

@pytest.fixture
def poa(monkeypatch):
    from modules.camera.poa import pyPOACamera as sdk
    state = SimpleNamespace(bin=1, counter=0, size=(SENSOR_W, SENSOR_H))
    ok = sdk.POAErrors.POA_OK
    props = SimpleNamespace(cameraID=0, bitDepth=12, pixelSize=3.75, bins=[1, 2, 4],
                            maxWidth=SENSOR_W, maxHeight=SENSOR_H)
    attrs = {sdk.POAConfig.POA_EXPOSURE: SimpleNamespace(minValue=10, maxValue=2_000_000),
             sdk.POAConfig.POA_GAIN: SimpleNamespace(minValue=0, maxValue=300)}
    def size():
        return state.size

    def set_bin(camera_id, b):
        # real SDK quirk: the previous ROI is kept, rescaled to the new bin and 4-aligned
        width = state.size[0] * state.bin // b // 4 * 4
        height = state.size[1] * state.bin // b // 2 * 2
        if b < state.bin:                  # shrinking the bin loses a little alignment each time
            width -= 4
        state.bin, state.size = b, (max(width, 8), max(height, 8))
        return ok

    def set_size(camera_id, width, height):
        state.size = (width // 4 * 4, height // 2 * 2)
        return ok
    def convert(buf, h, w, f):
        state.counter += 1
        return np.full((h, w), state.counter % 4000, np.uint16)

    fakes = dict(
        GetCameraCount=lambda: 1,
        GetCameraProperties=lambda i: (ok, props),
        OpenCamera=lambda i: ok, InitCamera=lambda i: ok, CloseCamera=lambda i: ok,
        GetConfigAttributesByConfigID=lambda i, c: (ok, attrs[c]),
        SetGain=lambda *a: ok, SetExp=lambda *a: ok, SetConfig=lambda *a: ok, SetImageFormat=lambda *a: ok,
        SetImageBin=set_bin, SetImageSize=set_size,
        GetImageSize=lambda i: (ok, *size()),
        GetImageFormat=lambda i: (ok, sdk.POAImgFormat.POA_RAW16),
        ImageCalcSize=lambda h, w, f: h * w * 2,
        StartExposure=lambda *a: ok, StopExposure=lambda i: ok,
        ImageReady=lambda i: (ok, True),
        GetImageData=lambda i, buf, t: ok,
        ImageDataConvert=convert,
        GetDroppedImagesCount=lambda i: (ok, 0), GetCameraTEMP=lambda i: (ok, 31.5),
        GetErrorString=lambda e: str(e),
    )
    for name, fake in fakes.items():
        monkeypatch.setattr(sdk, name, fake)
    from modules.camera.poa_camera import POACameraBackend
    yield lambda: POACameraBackend(gain=1, exposure_ms=1)


# ───────────────────────────── fake XIMEA ─────────────────────────────

class _XiImage:
    def __init__(self):
        self.width = self.height = 0
        self._data = None

    def get_image_data_numpy(self):
        return self._data


class _XiCamera:
    def __init__(self, dev=0):
        self.dwn, self.n, self.acq, self.dtype = 1, 0, False, 'XI_BINNING'

    def open_device(self): pass
    def close_device(self): pass
    def set_imgdataformat(self, f): pass
    def get_sensor_bit_depth(self): return 'XI_BPP_10'       # xiAPI returns enum names, not ints
    def get_image_data_bit_depth(self): return 'XI_BPP_10'
    def get_pixel_size(self): return 3.45
    def get_width(self): return SENSOR_W // self.dwn
    def get_height(self): return SENSOR_H // self.dwn
    def get_exposure_minimum(self): return 15
    def get_exposure_maximum(self): return 1_000_000
    def get_gain_minimum(self): return 0.0
    def get_gain_maximum(self): return 18.0
    def get_downsampling_type(self): return self.dtype
    def set_downsampling_type(self, t): self.dtype = t
    def get_downsampling_minimum(self): return 'XI_DWN_1x1'
    def get_downsampling_maximum(self): return 'XI_DWN_2x2'
    def get_downsampling(self): return 'XI_DWN_%dx%d' % (self.dwn, self.dwn)
    def set_downsampling(self, d): self.dwn = int(d.split('_')[-1].split('x')[0])
    def set_gain(self, g): pass
    def set_exposure(self, e): pass
    def start_acquisition(self): self.acq = True
    def stop_acquisition(self): self.acq = False
    def get_chip_temp(self): return 40.0

    def get_image(self, image, timeout):
        assert self.acq, 'acquisition not started'
        self.n += 1
        h, w = SENSOR_H // self.dwn, SENSOR_W // self.dwn
        image.height, image.width = h, w
        image._data = np.full(h * w, self.n % 4000, np.uint16)   # SDK buffer reused between frames


@pytest.fixture
def ximea(monkeypatch):
    package = types.ModuleType('modules.camera.ximea'); package.__path__ = []
    sdk = types.ModuleType('modules.camera.ximea.xiapi'); sdk.Camera, sdk.Image = _XiCamera, _XiImage
    package.xiapi = sdk
    monkeypatch.setitem(sys.modules, 'modules.camera.ximea', package)
    monkeypatch.setitem(sys.modules, 'modules.camera.ximea.xiapi', sdk)
    monkeypatch.delitem(sys.modules, 'modules.camera.ximea_camera', raising=False)
    cls = importlib.import_module('modules.camera.ximea_camera').XIMEACameraBackend
    yield lambda: cls(camera_id=0, gain=1, exposure_ms=1)
    sys.modules.pop('modules.camera.ximea_camera', None)


# ───────────────────────────── fake GenICam ─────────────────────────────

class _Node:
    def __init__(self, cam, name, value, mn=None, mx=None, inc=1):
        self.cam, self.name, self._v, self.min, self._max, self.inc = cam, name, value, mn, mx, inc

    @property
    def max(self): return self._max() if callable(self._max) else self._max

    @property
    def value(self): return self._v

    @value.setter
    def value(self, v):
        self.cam.write(self.name, v)
        self._v = v

    def execute(self): pass


class _NodeMap:
    def __init__(self):
        self.bin = 1
        n = {}
        for name in ('BinningHorizontal', 'BinningVertical'):
            n[name] = _Node(self, name, 1, 1, 2)
        n['Width'] = _Node(self, 'Width', SENSOR_W, 8, lambda: SENSOR_W // self.bin)
        n['Height'] = _Node(self, 'Height', SENSOR_H, 8, lambda: SENSOR_H // self.bin)
        n['PixelFormat'] = _Node(self, 'PixelFormat', 'Mono12')
        n['ExposureTime'] = _Node(self, 'ExposureTime', 1000.0, 10.0, 1_000_000.0)
        n['Gain'] = _Node(self, 'Gain', 0.0, 0.0, 24.0)
        n['ExposureAuto'] = _Node(self, 'ExposureAuto', 'Off')
        n['AcquisitionMode'] = _Node(self, 'AcquisitionMode', 'Continuous')
        n['TriggerMode'] = _Node(self, 'TriggerMode', 'Off')
        n['DeviceTemperature'] = _Node(self, 'DeviceTemperature', 36.0)
        self.n = n

    def write(self, name, value):
        if name.startswith('Binning'):
            self.bin = int(value)
            for k in ('Width', 'Height'):
                self.n[k]._v = self.n[k].max

    def __getattr__(self, name):
        if name in ('n', 'bin'):
            raise AttributeError(name)
        try:
            return self.n[name]
        except KeyError:
            raise AttributeError(name)


class _Buffer:
    def __init__(self, h, w, value, data_format='Mono12'):
        dtype = np.uint8 if data_format == 'Mono8' else np.uint16
        component = SimpleNamespace(height=h, width=w, data=np.full(h * w, value, dtype), data_format=data_format)
        self.payload = SimpleNamespace(components=[component])
        self.is_incomplete = False

    def __enter__(self): return self
    def __exit__(self, *exc): return False


class _Acquirer:
    def __init__(self):
        self.nodes, self.count, self.running = _NodeMap(), 0, False
        self.remote_device = SimpleNamespace(node_map=self.nodes)

    def start(self): self.running = True
    def stop(self): self.running = False
    def destroy(self): pass

    def fetch(self, timeout=0):
        assert self.running, 'acquisition not started'
        self.last_timeout = timeout
        self.count += 1
        value = getattr(self, 'fixed_value', None)
        return _Buffer(self.nodes.n['Height'].value, self.nodes.n['Width'].value,
                       self.count % 255 if value is None else value, self.nodes.n['PixelFormat'].value)


class _Harvester:
    def __init__(self):
        self.device_info_list = [object()]

    def add_file(self, path): pass
    def update(self): pass
    def create(self, index): return _Acquirer()
    def reset(self): pass


@pytest.fixture
def genicam(monkeypatch):
    harvesters = types.ModuleType('harvesters'); core = types.ModuleType('harvesters.core'); core.Harvester = _Harvester
    monkeypatch.setitem(sys.modules, 'harvesters', harvesters)
    monkeypatch.setitem(sys.modules, 'harvesters.core', core)
    monkeypatch.delitem(sys.modules, 'modules.camera.genicam', raising=False)
    cls = importlib.import_module('modules.camera.genicam').GenICamBackend
    yield lambda: cls(camera_index=0, gain=1, exposure_ms=1, cti_path='fake.cti', default_bin=1)
    sys.modules.pop('modules.camera.genicam', None)


# ───────────────────────────── the contract ─────────────────────────────

@pytest.fixture(params=['poa', 'ximea', 'genicam'])
def backend(request):
    factory = request.getfixturevalue(request.param)
    camera = factory()
    yield camera
    with contextlib.suppress(Exception):
        camera.close()


def test_implements_the_abstract_contract(backend):
    assert isinstance(backend, AbstractCameraBackend)
    assert not type(backend).__abstractmethods__
    # corrections come from the shared pipeline, never from a backend override
    assert type(backend).read_frame is AbstractCameraBackend.read_frame


def test_open_reports_geometry_ranges_and_bins(backend):
    backend.open()
    info = backend.info
    assert (info.width, info.height) == (SENSOR_W // backend.current_bin, SENSOR_H // backend.current_bin)
    assert 8 <= info.bit_depth <= 16
    bins = backend.get_bins()
    assert bins and all(isinstance(b, int) for b in bins) and backend.current_bin in bins
    for low, high in (backend.gain_range, backend.exposure_range_us):
        assert low < high


def test_raw_frames_are_uint16_matching_info_and_owned(backend):
    backend.open(); backend.start_stream()
    first = backend.read_frame_raw(timeout_ms=100)
    assert first.dtype == np.uint16 and first.shape == (backend.info.height, backend.info.width)
    snapshot = first.copy()
    second = backend.read_frame_raw(timeout_ms=100)
    np.testing.assert_array_equal(first, snapshot)          # next frame must not overwrite the previous one
    assert second.shape == first.shape


def test_bin_change_updates_info_and_frames(backend):
    backend.open()
    target = [b for b in backend.get_bins() if b != backend.current_bin][0]
    backend.start_stream(); backend.stop_stream()
    backend.set_bin(target)
    assert backend.current_bin == target
    assert (backend.info.width, backend.info.height) == (SENSOR_W // target, SENSOR_H // target)
    backend.start_stream()
    assert backend.read_frame_raw(timeout_ms=100).shape == (SENSOR_H // target, SENSOR_W // target)


def test_returning_to_a_bin_gives_the_same_frame_size(backend):
    """Frame geometry is a function of the bin only: dark/Smile/wavelength files depend on it."""
    backend.open()
    seen = {}
    for b in list(backend.get_bins()) * 2 + [backend.current_bin]:
        backend.stop_stream(); backend.set_bin(b); backend.start_stream()
        size = (backend.info.width, backend.info.height)
        assert seen.setdefault(b, size) == size, "bin %d changed size from %s to %s" % (b, seen[b], size)
        assert backend.read_frame_raw(timeout_ms=100).shape == (size[1], size[0])


def test_unsupported_bin_is_a_value_error(backend):
    backend.open()
    with pytest.raises(ValueError):
        backend.set_bin(97)


def test_params_dark_pipeline_and_diagnostics(backend):
    backend.open(); backend.start_stream()
    low, high = backend.gain_range
    backend.stop_stream(); backend.set_params((low + high) / 2, 5); backend.start_stream()
    shape = (backend.info.height, backend.info.width)
    backend.set_dark_reference(np.zeros(shape, np.uint16), None)
    assert backend.read_frame(timeout_ms=100).shape == shape      # shared dark/hot-pixel path
    assert isinstance(backend.get_temperature(), float)
    assert isinstance(backend.dropped_images_count(), int)


def test_stop_event_interrupts_reads_and_close_is_idempotent(backend):
    backend.open(); backend.start_stream()
    event = threading.Event(); backend.set_stop_event(event); event.set()
    assert backend.read_frame_raw(timeout_ms=100) is None
    backend.close(); backend.close()


def test_factory_builds_each_backend(poa, ximea, genicam):
    config = SimpleNamespace(camera_backend='x', camera_index=0, cti_path='fake.cti',
                             default_gain=1, default_exposure_ms=1, default_bin=1)
    for name, class_name in (('poa', 'POACameraBackend'), ('XIMEA', 'XIMEACameraBackend'), ('genicam', 'GenICamBackend')):
        config.camera_backend = name
        assert type(create_backend(config)).__name__ == class_name
    config.camera_backend = 'nonsense'
    with pytest.raises(ValueError, match='nonsense'):
        create_backend(config)


def test_genicam_retries_acquisition_start_while_the_camera_is_busy(genicam, monkeypatch):
    camera = genicam()
    camera.open()
    monkeypatch.setattr(type(camera), 'START_RETRY_DELAYS', (0.0, 0.0, 0.0))
    original_start, calls = camera._ia.start, []

    def flaky_start():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError('GenTL exception: GC_ERR_IO')
        original_start()

    camera._ia.start = flaky_start
    camera.start_stream()
    assert len(calls) == 3 and camera.read_frame_raw(timeout_ms=100) is not None
    camera.stop_stream()
    camera._ia.start = lambda: (_ for _ in ()).throw(RuntimeError('GC_ERR_IO'))
    with pytest.raises(RuntimeError, match='GC_ERR_IO'):
        camera.start_stream()
    camera.close()


@pytest.mark.parametrize('fmt,sensor,value,expected,bit_depth', [
    ('Mono8', 'SensorBpp8', 200, 200, 8),              # 8-bit: values unchanged, bit_depth 8
    ('Mono10', 'SensorBpp10', 1000, 1000 << 6, 16),    # 10-bit scaled to 16-bit
    ('Mono12', 'SensorBpp10', 4000, 4000 << 4, 16),    # Mono12: the original behaviour
    ('Mono12p', 'SensorBpp12', 4000, 4000 << 4, 16),
    ('Mono16', 'SensorBpp10', 1000, 1000 << 6, 16),    # 16-bit container: the sensor depth decides
    ('Mono16', 'SensorBpp8', 200, 200, 8),
    ('Mono16', None, 60000, 60000, 16),                # true 16-bit: unchanged
])
def test_genicam_output_follows_the_pixel_format(genicam, fmt, sensor, value, expected, bit_depth):
    camera = genicam()
    camera.open()
    camera._nm.n['PixelFormat']._v = fmt
    if sensor is not None:
        camera._nm.n['SensorPixelSize'] = _Node(camera._nm, 'SensorPixelSize', sensor)
    camera._refresh_info()
    assert camera.info.bit_depth == bit_depth
    camera._ia.fixed_value = value
    camera.start_stream()
    frame = camera.read_frame_raw(timeout_ms=100)
    assert frame.dtype == np.uint16 and int(frame[0, 0]) == expected
    camera.close()


def test_genicam_writes_the_configured_sensor_pixel_size(genicam, monkeypatch):
    from modules.camera.genicam import GenICamBackend
    written = []
    monkeypatch.setattr(GenICamBackend, '_set_node_value', lambda self, name, value: written.append((name, value)))
    camera = GenICamBackend(camera_index=0, gain=1, exposure_ms=1, cti_path='fake.cti', default_bin=1, sensor_pixel_size=8)
    camera.open()
    assert ('SensorPixelSize', 'SensorBpp8') in written
    camera.close()
    with pytest.raises(ValueError):
        GenICamBackend(sensor_pixel_size=9)


def test_pixel_format_bits():
    from modules.camera.genicam import pixel_format_bits
    assert [pixel_format_bits(n) for n in ('Mono8', 'Mono10Packed', 'Mono12p', 'BayerRG12', 'Bpp16', 'Unknown', None)] == \
        [8, 10, 12, 12, 16, None, None]


@pytest.mark.parametrize('fmt,sensor,expected', [
    ('Mono12Packed', 'SensorBpp10', 10),               # fxo992: 10 real bits in a 12-bit container
    ('Mono12', None, 12),
    ('Mono8', 'SensorBpp10', 8),                       # the camera already cut it to 8 bits
])
def test_genicam_sensor_bit_depth_is_the_real_data_bits(genicam, fmt, sensor, expected):
    camera = genicam()
    camera.open()
    camera._nm.n['PixelFormat']._v = fmt
    if sensor is not None:
        camera._nm.n['SensorPixelSize'] = _Node(camera._nm, 'SensorPixelSize', sensor)
    camera._refresh_info()
    assert camera.sensor_bit_depth() == expected
    camera.close()


def test_poa_sensor_bit_depth_is_the_sdk_value_before_scaling(poa):
    camera = poa()
    camera.open()
    assert camera.info.bit_depth == 16 and camera.sensor_bit_depth() == 12
    camera.close()


def test_ximea_bit_depth_is_read_from_the_xiapi_enum_name(ximea):
    camera = ximea()
    camera.open()
    assert camera.info.bit_depth == 10 and camera.sensor_bit_depth() == 10
    assert camera.saturation_level() == 1023
    camera.close()


def test_genicam_fetch_timeout_is_passed_to_harvesters_in_seconds(genicam):
    camera = genicam()
    camera.open()
    camera.start_stream()
    assert camera.read_frame_raw(timeout_ms=100) is not None
    assert camera._ia.last_timeout == 0.1                     # harvesters' fetch(timeout) is in s
    camera.close()
