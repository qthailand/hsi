# modules/camera

Camera acquisition behind one contract. Three backends ship: **POA** (Player One),
**GenICam** (GigE Vision via `harvesters`) and **XIMEA** (xiAPI). Applications pick one
by name and never import a vendor SDK directly; an SDK is imported only when its backend
is selected.

## Overview

* `AbstractCameraBackend` (`base.py`) owns open/close/stream state and the shared
  correction pipeline (dark subtraction, hot-pixel repair via `modules.dark`,
  saturation check). Backends implement only the hardware hooks.
* `create_backend(config)` (`factory.py`) builds the backend named by
  `config.camera_backend` (`"poa"`, `"genicam"`, `"ximea"`, case-insensitive).
* `modules/camera/poa/` and `modules/camera/ximea/` are vendor SDK wrappers; leave them alone.

## Requirements

| Backend | Needs |
|---------|-------|
| POA | `PlayerOneCamera.dll`, shipped in `modules/camera/poa` |
| XIMEA | xiAPI in `modules/camera/ximea`; its `libs/` folder must contain the vendor DLLs |
| GenICam | `pip install harvesters` plus a GenTL producer `.cti` file |

Also numpy, opencv-python (`base.py` imports `cv2`) and `modules.dark`. No PyQt5.

## Stand-alone use

```
python -m modules.camera --backend poa --grab frame.npy --image frame.png
python -m modules.camera --backend genicam --cti C:\path\producer.cti --index 0 --bin 2
python -m modules.camera --backend ximea --gain 1 --exposure-ms 20 --frames 50
```

Options: `--backend {poa,genicam,ximea}`, `--index`, `--cti`, `--gain`, `--exposure-ms`,
`--bin`, `--frames` (frames read for statistics), `--grab FILE.npy` (last raw frame).
It prints geometry, ranges, fps and frame statistics; exit code 1 if no frame arrived.

## Embedding in an application

```python
from types import SimpleNamespace
from modules.camera import create_backend

config = SimpleNamespace(camera_backend="poa", camera_index=0, cti_path=None,
                         default_gain=100, default_exposure_ms=10, default_bin=2)
with create_backend(config) as camera:      # open() / close()
    camera.set_params(gain=100, exposure_ms=20)
    camera.start_stream()
    frame = camera.read_frame_raw(timeout_ms=1000)   # None on timeout
    print(camera.info.width, camera.info.height, camera.current_bin)
```

`AppConfig` already provides the attributes `create_backend` reads.

### Configuration keys

| Key | Used by | Meaning |
|-----|---------|---------|
| `camera_backend` | all | `poa`, `genicam` or `ximea` (default `poa`) |
| `camera_index` | GenICam, XIMEA | device index (default 0); POA ignores it |
| `cti_path` | GenICam | GenTL producer; default `$GENICAM_GENTL64_PATH/libsv_gev_tl_x64.cti` |
| `default_gain` | all | initial gain, vendor units |
| `default_exposure_ms` | all | initial exposure, milliseconds |
| `default_bin` | GenICam, XIMEA | bin applied at `open` (`None` = leave); POA ignores it and picks its own |
| `sensor_pixel_size` | GenICam | `SensorPixelSize` written at `open`: 8, 10 (default, as originally) or 12 |

### Backend quirks

* **POA**: opens the first enumerated camera only. At `open` it forces `RAW16`, offset 20,
  USB bandwidth limit 90 and a default bin of 2 (else 1, else the first supported bin).
  `bit_depth` is reported as 16 for 10/12/16-bit sensors.
* **GenICam**: frames are `uint16`. Bits per pixel come from `PixelFormat` (8/10/12/14), or from
  `SensorPixelSize` when the format is `Mono16`/unknown: 8 bits stay 0..255 (`bit_depth` 8);
  10/12/14 bits are shifted into the 16-bit range (`<< 6`/`<< 4`/`<< 2`, `bit_depth` 16). Mono12
  is exactly the original `<< 4`. `set_bin` writes both Binning nodes; if the camera refuses, it
  retries with a reset (minimal) ROI and then restores the full frame. A `default_bin`
  that fails at `open` prints a warning instead of raising. `dropped_images_count()` is
  always 0.
* **XIMEA**: uses `XI_MONO16` and reports the sensor bit depth in `info.bit_depth`.
  Binning is `XI_BINNING` downsampling (`XI_DWN_NxN`) and is verified after setting.
  SDK calls are serialised by a lock; `get_temperature()` returns the last value
  instead of blocking while a frame is being fetched.

## Data formats / conventions

* A frame is a 2-D `uint16` array `(rows, columns) = (height, width) = (slit, spectral)`
  at the current hardware bin. The caller owns it (always a copy, never an SDK buffer).
* Exposure: `set_params(gain, exposure_ms)` takes **milliseconds**; `exposure_range_us`
  is in **microseconds**. Gain is in vendor units (`gain_range`).
* Temperature in degrees Celsius; `-999.0` means unavailable.
* Binning is N x N; `info.width/height` already reflect the bin.

## API summary

| Member | Purpose |
|--------|---------|
| `open()`, `close()`, context manager | connect / release (idempotent) |
| `start_stream()`, `stop_stream()` | start / stop acquisition |
| `read_frame_raw(timeout_ms)` | uncorrected frame or `None` |
| `read_frame_raw_into(provide, timeout_ms)` | same values; GenICam converts straight into `provide(shape)` (e.g. the CuPy zero-copy buffer, overwritten by the next frame), other backends return `read_frame_raw`'s new array |
| `read_frame(timeout_ms)` | frame with dark + hot-pixel correction |
| `read_frame_profiled(profiler, timeout_ms)` | `read_frame` with per-stage timing |
| `set_params(gain, exposure_ms)`, `set_bin(n)` | configuration |
| `info`, `current_bin`, `get_bins()`, `exposure_range_us`, `gain_range` | state and limits |
| `get_temperature()`, `dropped_images_count()`, `detect_saturation(frame)` | diagnostics |
| `cooling_supported()`, `target_temperature_range()`, `get_target_temperature()`, `set_target_temperature(c)` | TEC cooler target in °C; the default (fan-cooled/uncooled) is unsupported. POA uses `isHasCooler` + `POA_TARGET_TEMP`/`POA_COOLER`, XIMEA `iscooled` + `cooling`/`target_temp`, GenICam the first writable set-point node it finds (`TARGET_TEMPERATURE_NODES`, logged at open) and a cooler switch node if any. Setting a target switches the cooler on |
| `set_stop_event(event)` | let another thread abort a blocked read |
| `set_dark_reference(...)`, `set_dark_enabled(...)` | from `modules.dark` |

## Threading

One acquisition thread should own `open`/`start_stream`/`read_frame*`/`set_params`/
`set_bin`. `info`, `current_bin` and the range properties are plain reads usable from the
GUI thread. To stop a worker blocked in a read, set the event given to `set_stop_event`.
`close()` also sets it.

## Adding a backend

1. Create `modules/camera/<name>_camera.py` with a subclass of `AbstractCameraBackend`.
2. Implement the abstract members: `_open_device`, `_close_device`, `_start_device`,
   `_stop_device`, `_fetch_raw_frame`, `set_params`, `set_bin`, and the
   properties/methods `info`, `exposure_range_us`, `gain_range`, `get_bins`,
   `current_bin`, `dropped_images_count`, `get_temperature`. Their docstrings in
   `base.py` are the contract (units, shapes, ownership, which values mean failure).
   Do not override `read_frame`; override `set_stop_event` to honour a stop event.
3. Register the name in `factory.py`: add it to `BACKENDS` and a branch in
   `create_backend` that imports the class lazily.
4. Add a fake-SDK fixture to `tests/test_camera_backends.py` and include its name in the
   `backend` fixture's `params`; the whole contract suite then runs against it:

```python
@pytest.fixture
def mycam(monkeypatch):
    sdk = types.ModuleType("modules.camera.mycam_sdk")      # the fake vendor SDK
    sdk.Camera = _FakeCamera                                 # open/get_image/set_gain/...
    monkeypatch.setitem(sys.modules, "modules.camera.mycam_sdk", sdk)
    monkeypatch.delitem(sys.modules, "modules.camera.mycam_camera", raising=False)
    cls = importlib.import_module("modules.camera.mycam_camera").MyCamBackend
    yield lambda: cls(gain=1, exposure_ms=1)
    sys.modules.pop("modules.camera.mycam_camera", None)

@pytest.fixture(params=["poa", "ximea", "genicam", "mycam"])
def backend(request): ...
```

The fake must report a 640 x 480 sensor, bins that divide it, ranges with `min < max`,
and frames that shrink with the bin.

## Tests

```
PYTHONUTF8=1 python -m pytest -q tests/test_camera_backends.py tests/test_genicam_bin.py tests/test_fetch_profile.py
```

`test_camera_backends.py` runs the same contract against all backends using fake SDKs,
so no hardware is needed.

## Widgets

`CameraControls` (`modules/camera/widget.py`, a `QGroupBox`; needs PyQt5, loaded lazily so
`import modules.camera` stays headless) holds gain, exposure (ms), bin, a
temperature row and a "Keep Spectrogram" check box. The temperature row is an editable
"Target Temp" after `set_cooling_supported(True, range)` (TEC cameras) and otherwise a
read-only "Sensor Temp" updated by `show_sensor_temperature(t)`. It never touches a camera: the host connects its signals.
There is no GUI launcher; embed it like this:

```python
from modules.camera import CameraControls

controls = CameraControls()                      # GUI thread, no arguments
lo_us, hi_us = camera.exposure_range_us
controls.set_limits(*camera.gain_range, lo_us / 1000, hi_us / 1000)   # exposure in ms
controls.set_bin_options(camera.get_bins(), default_bin=camera.current_bin)
controls.set_values(gain=100, exposure_ms=10.0)  # silent, clamped to the limits
controls.params_changed.connect(lambda: engine.request_params(controls.gain(), controls.exposure_ms()))
controls.bin_changed.connect(lambda: engine.request_bin(controls.selected_bin()))
layout.addWidget(controls)
```

Signals: `params_changed()` (gain/exposure edited), `bin_changed()` and
`target_temperature_changed(float)` (editing finished, only while cooling is supported;
`set_target_temperature` does not emit). `set_values`,
`set_bin_options` and `set_selected_bin` do not emit. `set_bin_enabled(False)` locks the
bin during a scan; `keep_spectrogram()` is read when a scan starts.
