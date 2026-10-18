# live_view

Live display of a camera stream: raw-to-8-bit mapping, image canvas with a selectable row
and focus ROI, RAW/Enhance toggle, intensity histogram and spectrum plots.

## Overview

| Name | File | Purpose |
|------|------|---------|
| `DisplayMapper`, `DisplayMode` | `mapping.py` | Qt-free raw -> uint8 conversion (RAW linear / ENHANCE min-max) |
| `DisplayControls` | `display_controls.py` | `[Display: RAW v] [10→16 bit]` bar, emits a new mapper |
| `LiveImageCanvas` | `image_canvas.py` | `QLabel` showing the frame, red selected row, yellow focus ROI, overlay buttons |
| `LiveLineSpectrumWidget` | `spectrum.py` | fast QPainter spectrum of one row, Y axis auto-scales to bit-depth ceilings |
| `SpectrumPlot` | `spectrum.py` | static pyqtgraph spectrum with a band marker |
| `HistogramPlot` | `histogram.py` | 256-bin intensity histogram with saturation read-out (QPainter) |

## Requirements

* Python 3.9+, numpy, PyQt5
* `pyqtgraph` (SpectrumPlot), `qtawesome` (DisplayControls icon)
* `modules.image` (`raw_2d_frame`); `__requires__ = ("image",)`

## Stand-alone use

```
python -m modules.live_view
```

shows a synthetic gradient on `LiveImageCanvas` with `DisplayControls` and a
`LiveLineSpectrumWidget` for the selected row (click the image to change it).

## Embedding in an application

```python
import numpy as np
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
from modules.live_view import DisplayControls, LiveImageCanvas, LiveLineSpectrumWidget

app = QApplication([])
panel = QWidget()
layout = QVBoxLayout(panel)
controls, canvas, spectrum = DisplayControls(), LiveImageCanvas(), LiveLineSpectrumWidget()
for w in (controls, canvas, spectrum):
    layout.addWidget(w)

controls.mapper_changed.connect(canvas.set_display_mapper)

def on_frame(frame):                       # call on the GUI thread, frame (slit, wavelength)
    canvas.set_array(frame)
    spectrum.update_spectrum(frame[canvas.selected_y])

canvas.y_changed.connect(lambda y: ...)    # image row selected by the user
panel.show()
```

Headless use of the mapper (no Qt):

```python
from modules.live_view import DisplayMapper, DisplayMode
img8 = DisplayMapper(DisplayMode.RAW, bit_depth=12).to_uint8(frame)
```

## Data formats / conventions

* Frame: 2-D `(rows, columns)` = `(slit, wavelength)`, normally uint16; a 3-D
  `(rows, columns, C)` array uses channel 0.
* Spectrum: 1-D over bands; `wavelengths` in nm (otherwise the x axis is the band index).
* `selected_y`, `focus_roi_range()` are 0-based image rows; the ROI is at least 20 rows.
* `bit_depth` = sensor bits (RAW white = `2**bit_depth - 1`).
* RAW mode: 16 bit -> `>> 8`, 12 bit -> `>> 4`, 10 bit scaled, others float scaled;
  ENHANCE: per-frame min-max (constant frame -> 128).

## API summary

* `DisplayMapper(mode, bit_depth)`: `set_bit_depth`, `to_uint8(frame)`.
* `DisplayControls(bit_depth=16)`: signal `mapper_changed(object)`; `current_mapper()`,
  `set_bit_depth(bit_depth, sensor_bits=None)`. The bit depth is read-only (set by the camera)
  and shown as `10→16 bit` when the camera's real bits are scaled up.
* `LiveImageCanvas()`: signal `y_changed(int)` (user action only); `set_array`, `set_display_mapper`,
  `set_selected_y`, `set_focus_roi_height`, `focus_roi_range`, `full_height`,
  `set_reference_roi((start, end, height) | None)`, `add_overlay_button(text)`.
* `LiveLineSpectrumWidget()`: `update_spectrum(spectrum, wavelengths=None)`, `set_bit_depth`, `clear`.
* `SpectrumPlot()`: `plot(spectrum, wavelengths=None, current_band=None)`, `clear`.
* `HistogramPlot(bit_depth=16)`: `update_histogram(frame)`, `set_bit_depth`.

All five widgets are in `__widgets__` (no-argument constructors).

## Threading

Widgets are GUI-thread only; push camera frames in through a queued signal.
`DisplayMapper` is pure numpy and may run in any thread. `import modules.live_view` does not
import PyQt5 (widgets load lazily; `DisplayMapper`/`DisplayMode` are Qt-free and eager).

## Tests

`tests/test_module_contract.py` (`-k live_view`) checks the contract and builds each widget in a
`QMainWindow`; `tests/test_app_smoke.py` exercises the application wiring.
