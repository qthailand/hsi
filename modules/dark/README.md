# dark

## Overview

Qt-free dark-frame handling for the camera backends:

* `DarkProcessor` subtracts a dark mean frame and repairs hot pixels (mean of valid
  neighbours) on 2-D `(slit, wavelength)` frames, with neighbour tables precomputed once.
* `DarkFrameLibrary` captures N dark frames, computes mean and hot-pixel map, and stores
  one `.npz` per `(bin, exposure, gain, sensor temperature)`; `load`/`find` return the file
  captured nearest the current temperature (60 °C is assumed when the camera reports none).

## Requirements

* numpy and OpenCV (`cv2`, used for saturating subtraction).
* The processing/library classes need no Qt and no other `modules.*` package (`__requires__ = ()`); `DarkFramePanel` additionally needs PyQt5 and `qtawesome`.

## Stand-alone use

The status widget is described under "Widgets and dark selection". `python -m modules.dark INPUT.npy REFERENCE.npz OUTPUT.npy` applies a
saved reference (`mean`, `hot_pixel_map`) to a raw 2-D `.npy` frame and saves the result;
shapes must match.

## Embedding in an application

```python
import numpy as np
from modules.dark import DarkFrameLibrary, DarkProcessor

lib = DarkFrameLibrary("dark_frames", n_frames=50)
lib.start_capture(bin=1, exposure=100.0, gain=5.0)
# in the camera loop, with the shutter closed:
#   if lib.add_frame(frame, temperature=camera.get_temperature()): break   # True once saved
ref = lib.load(1, 100.0, 5.0, temperature=25.0)   # nearest temperature; dict or None

proc = DarkProcessor()
proc.set_dark_reference(ref["mean"], ref["hot_pixel_map"])
# per frame (steps used by the camera backends):
#   out = proc._apply_hot_pixel(proc._apply_dark_subtract(frame))   # modifies the result in place
```

## Data formats / conventions

* Frames: 2-D `(slit, wavelength)`, usually `uint8`/`uint16`; dark and frame must share
  shape and dtype. Subtraction saturates at 0 for unsigned types.
* `hot_pixel_map`: same shape, non-zero = hot (`uint8`, 255 from the library). A hot pixel
  is detected with threshold `max(mean + 4 sigma, 99.9th percentile)` on the
  max-normalised mean frame (falling back to `mean + 3 sigma` / 99.5th percentile).
* File `dark_frames/dark_b{bin}_e{exposure:.1f}_g{gain:.1f}_t{temperature:+d}.npz` (temperature
  = mean over the capture in °C, rounded to whole degrees; a recapture in the same degree
  replaces the file) with arrays `mean`, `hot_pixel_map`, `bin` (int32), `exposure` (ms,
  float32), `gain` (float32), `n_frames` (int32), and `temperature`, `temperature_min`,
  `temperature_max` (°C, float64). Without a temperature (camera reports `-999`, or an
  older file) the name has no `_t` part and the temperature arrays are absent; such a file
  is used only when its `(bin, exposure, gain)` has no file with a temperature.
  Note: no `format_version` key is stored.
* `_apply_hot_pixel` modifies the given frame in place; `_apply_dark_subtract` returns a
  new array (or the same/copy when no usable dark is set).

## API summary

* `DarkProcessor`: `set_dark_reference(mean, hot_map)`, `set_dark_enabled(bool)` (flag
  only), `_apply_dark_subtract(frame)`, `_apply_hot_pixel(frame)`.
  `dark_reference` returns the current immutable `DarkReference` (dark mean and hot-pixel tables).
* `DarkFrameLibrary(folder, n_frames=50)`: `start_capture`, `add_frame(frame, temperature)`,
  `cancel`, `progress`, `is_collecting`, `find`/`load(bin, exposure, gain, temperature)`,
  `exists`, `delete(..., temperature=None)`, `list_keys`. `UNKNOWN_TEMPERATURE_C` (60.0) and
  `valid_temperature(value)` (``None`` for ``-999``/NaN) are exported too.

## Threading

`DarkFrameLibrary` is not thread-safe; use it from one thread. `DarkProcessor.set_dark_reference`
builds a complete `DarkReference` and installs it with one assignment, so the GUI thread may
call it while the camera worker processes frames: a frame loop reads `dark_reference` once per
frame (as `modules.preprocessing` does) and never mixes two references. `add_frame` writes the file synchronously when the last
frame arrives.

## Tests

`python -m pytest -q tests -k dark` and `tests/test_module_contract.py`.

## Widgets and dark selection

`DarkFramePanel` (`modules/dark/widget.py`, a `QGroupBox`; needs PyQt5 and `qtawesome`,
loaded lazily so `import modules.dark` stays headless) shows the dark state and has
Capture / Cancel / Reload buttons and an "Apply Dark Subtraction" check box. It captures
nothing itself: the host reacts to its signals and reports back with `set_*` methods.
There is no GUI launcher (`python -m modules.dark` is the CLI above).

```python
from modules.dark import DarkFrameLibrary, DarkFramePanel, select_dark

lib = DarkFrameLibrary("dark_frames", n_frames=50)
panel = DarkFramePanel()                              # GUI thread, no arguments
panel.capture_requested.connect(lambda: lib.start_capture(bin=2, exposure=10.0, gain=100.0))
panel.cancel_requested.connect(lib.cancel)
panel.dark_toggled.connect(camera.set_dark_enabled)  # bool
panel.set_capturing(0, 50, bin_value=2, exposure_ms=10.0, gain=100)
panel.update_progress(12, 50)                         # per captured frame
```

Signals: `capture_requested()`, `cancel_requested()`, `reload_requested()`,
`dark_toggled(bool)`. States: `IDLE`, `LOADED`, `CAPTURING`, `ERROR` (set with `set_idle`,
`set_loaded(bin, exposure_ms, gain, n_frames, n_hot, temperature_c, current_temperature_c)`,
`set_capturing(...)`, `set_error(msg)`); `set_current_temperature(t)` updates the ΔT shown for
the loaded dark; `set_interactive(False)` disables Capture/Reload during a scan.

### `select_dark`

Picks the stored dark that fits the camera's current frame, without any GUI:

```python
sel = select_dark(lib, bin=2, exposure=10.0, gain=100.0, frame_shape=(552, 804), temperature=25.3)
if sel.kind == "loaded":
    camera.set_dark_reference(sel.result["mean"], sel.result["hot_pixel_map"])
    panel.set_loaded(2, 10.0, 100, n_frames=int(sel.result["n_frames"]), n_hot=sel.n_hot)
elif sel.kind == "mismatch":      # file exists but its shape is another camera's / bin's
    panel.set_error("Dark does not match the camera")
else:                             # "missing"
    panel.set_idle()
```

`DarkSelection` has `kind` (`"loaded"`, `"missing"`, `"mismatch"`), `result`, `dark_shape`,
`frame_shape`, `n_hot`, `dark_temperature` (°C of the chosen file, or `None`) and `temperature`
(the camera temperature it was chosen for, or `None`).
