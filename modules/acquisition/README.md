# modules/acquisition

Frame acquisition and scan bookkeeping for a pushbroom spectrograph: one camera is run
through *fetch -> dark/hot-pixel -> orientation -> Smile -> spectral binning -> record* and
the results go to a sink (plain callbacks) or to Qt signals. The engine, recorder and
session are Qt-free.

## Overview

| Class | Qt? | Role |
|-------|-----|------|
| `AcquisitionEngine` | no | the pipeline; run `run_until_stopped` on your own thread |
| `AcquisitionSink` | no | base class with no-op callbacks (`frame`, `cube_frames`, `correction_status`, `sharpness`, `snapshot`, `processed_frame`, `error`, `bin_applied`, `bin_failed`, `params_applied`, `params_failed`, `wavelength_applied`, `wavelength_failed`) |
| `CameraPreviewThread` | yes | the engine on a `QThread`; sink callbacks become signals |
| `ScanRecorder` | no | thread-safe "record every frame" buffers (owned by the engine) |
| `ScanSession` / `ScanOutcome` | no | GUI-independent scan state machine |
| `StageProfiler` | no | running per-stage timing |
| `CorrectionStatusDisplay` | yes | two `StatusCard`s (Dark, Smile) fed by the status dict |

## Requirements

numpy and a camera backend from `modules.camera` (passed in). Declared dependencies
(`__requires__`): `dark`, `frame_exchange`, `preprocessing`, `sharpness`, `smile`,
`spectral_processing`, `status_card`. PyQt5 is needed only for `CameraPreviewThread` and
`CorrectionStatusDisplay`; `import modules.acquisition` does not import it.

## Stand-alone use

Headless check, no GUI:

```
python -m modules.acquisition --backend poa --seconds 5 --gain 1 --exposure-ms 10 --bin 1 --record
```

It runs the full pipeline and prints display frames, fps, output shape, recorded frame
count, number of spectral bins, sharpness and errors; exit code 1 if no frame arrived.
Other options: `--index`, `--m`, `--c`, `--resolution`. There is no GUI launcher.

## Embedding

### Headless engine with a sink

```python
import threading
from types import SimpleNamespace
from modules.camera import create_backend
from modules.acquisition import AcquisitionEngine, AcquisitionSink

config = SimpleNamespace(camera_backend="poa", camera_index=0, cti_path=None,
                         default_gain=1, default_exposure_ms=10, default_bin=1,
                         m=0.2903, c=399.791, resolution=2.0,      # nm/px at bin 1, nm, nm per bin
                         smile_map_folder=".", preprocessing_backend="auto")

class Print(AcquisitionSink):                 # called on the WORKER thread
    def frame(self, frame, fps, timestamp):
        print(frame.shape, round(fps))
    def error(self, message):
        print("error:", message)

engine = AcquisitionEngine(create_backend(config), config, sink=Print())
worker = threading.Thread(target=engine.run_until_stopped, daemon=True)
worker.start()
engine.request_params(100, 20.0)              # gain, exposure in ms (queued)
engine.set_recording(True)                    # record EVERY frame from now on
...
engine.set_recording(False)
frames = engine.take_recorded_frames()        # list of (slit, wavelength_bin) uint16
engine.request_stop(); worker.join()
```

### Qt: `CameraPreviewThread`

```python
from modules.acquisition import CameraPreviewThread, CorrectionStatusDisplay

status_display = CorrectionStatusDisplay(parent_widget)   # .dark / .smile cards
thread = CameraPreviewThread(create_backend(config), config)
thread.frame_ready.connect(viewer.set_frame)            # (frame, fps, timestamp), newest only, <= 30/s
thread.cube_frames_ready.connect(collect)               # list of every recorded frame
thread.sharpness_ready.connect(sharpness_panel.set_value)
thread.correction_status.connect(status_display.update_status)
thread.error.connect(show_error)
thread.start()
...
thread.stop()                                           # True when finished within 2 s
```

Other signals: `snapshot_ready(object)`, `processed_frame_ready(object)` (answer to
`request_processed_frame()`: the preprocessed frame before spectral binning), `bin_applied(int)`, `bin_failed(int, str)`,
`params_applied()`, `params_failed(str)`, `wavelength_applied(object)`, `wavelength_failed(object, str)`,
`saturation_ready(object)` (reserved, not emitted yet). The host owns the thread and must
call `stop()` before closing.

### `ScanSession`

```python
from modules.acquisition import ScanSession

session = ScanSession()
session.begin()                              # PREVIEW -> SCANNING, buffers cleared
thread.set_recording(True)                   # arm the worker, move the stage ...
thread.set_recording(False)
outcome = session.finish(thread.take_recorded_frames(), thread.take_recorded_raw())
if outcome.kind == "cube":                   # also "empty", "reference", "reference_failed"
    build_cube(outcome.images)               # modules.cube_io, shape (slit, scan, wavelength)
# on motion failure: session.fail() discards the scan
```

## Data conventions

* Sensor frame `(slit, spectral_pixel)` `uint16`; binned frame `(slit, wavelength_bin)`;
  a cube is `(slit, scan, wavelength)`.
* Exposure in milliseconds, gain in vendor units, wavelengths in nm.
* With Keep Spectrogram (`set_keep_spectrogram(True)`) the raw frames are recorded too,
  index-aligned with the binned ones; `take_recorded_*` return lists the caller owns.
* Dark capture mode (`set_dark_capture_mode`) skips corrections and delivers raw frames.

## API summary

* `AcquisitionEngine(backend, config=None, sink=None, profile=True)`: `run_until_stopped`,
  `request_stop`, `request_params(gain, exposure_ms)`, `request_bin(n)`,
  `request_wavelength(profile)`, `set_wavelength_lookup(lookup)` (stored calibration per bin, used at
  start and after every bin change), `set_resolution_lookup(lookup)` / `request_resolution_reload()`
  (stored FWHM per bin as the bin width when `config.resolution_source == "fwhm"`: `fwhm_step_factor`
  × FWHM, at least 2 spectral pixels; `config.resolution` when there is none), `binning_info()`
  (bin width in use and its source, stored in the scan metadata), `request_snapshot(id)`, `request_processed_frame()`, `request_smile_model(path)`,
  `set_recording`, `set_scanning`, `set_keep_spectrogram`, `set_raw_inverted`,
  `set_dark_capture_mode`, `recorded_count`, `take_recorded_frames`, `take_recorded_raw`,
  `binner_n_bins`, `binner_lambda_out`.
* `ScanRecorder`: `set_recording`, `offer`, `count`, `take_frames`, `take_raw`, `pop_batch`.
* `ScanSession`: `begin`, `finish`, `fail`, `set_error`, `start_reference_preview`,
  `cancel_reference_preview`, `note_preview_frames`, `add_reference_column`.
* `CorrectionStatusDisplay(parent=None)`: `.dark`, `.smile`, `update_status(status)`;
  a `QObject`, so it is documented here rather than listed in `__widgets__`.

## Threading

One worker thread owns the camera: it opens, streams, reads and closes it, and applies
queued parameter/bin/wavelength changes at frame boundaries. Every `request_*` / `set_*`
method may be called from any other thread. **Every frame is recorded** by the worker
(into `ScanRecorder`) regardless of what the GUI shows; only the *display* stream is
throttled to 30 fps (sharpness is computed about every 0.2 s). Sink callbacks run on the
worker thread; `CameraPreviewThread` converts them into queued Qt signals so slots run on
the GUI thread. Display frames are the exception: the worker only replaces the newest frame
in a one-slot mailbox and the GUI thread emits `frame_ready` when it is ready for one, then
rests at least as long as the frame took to handle and paint (`MAX_GUI_DUTY` = 0.5). A slow
GUI therefore skips frames instead of queueing them, and native dialogs keep getting time. `ScanSession` and `CorrectionStatusDisplay` are GUI-thread only.

## Tests

```
PYTHONUTF8=1 python -m pytest -q tests/test_acquisition.py tests/test_app_smoke.py
```
