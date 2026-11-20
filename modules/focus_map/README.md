# focus_map

## Overview
Focus aid for a pushbroom spectrograph whose sharp region is a curved band across the
`(slit, wavelength)` frame (field curvature plus focus that moves with wavelength). Point
the camera at a bar target (bars along the spectral axis, e.g. the A3 keystone target) and
the window shows, live:

* a map of the local bar contrast (bright = sharp) with the sharpest column of every slit
  row and the parabola fitted through them (the "<" shaped best-focus curve);
* **worst-zone CNR** (contrast-to-noise) - the 5th percentile over the lit frame of the bar
  amplitude (lock-in at the bar period, DN) divided by the single-pixel noise (DN, from second
  differences along the spectrum), with its peak and a trend arrow. Turn the focus, and close
  the aperture, until it is highest: stopping down sharpens the bars but costs light, and the
  CNR falls again once the light lost outweighs the sharpness gained. The light-independent
  worst-zone contrast (bar amplitude / local level) is shown below it, with the signal level
  (DN, % of full scale for integer frames), the noise and a clipping warning;
* the wavelength (or column) of best focus at the slit centre and at the slit ends, and a
  one-line hint (which way the best-focus band should move, whether field curvature is so
  large that stopping down is the remedy).

The single-number `modules.sharpness` score is a mean over the frame and peaks where the
brightest bars are sharp; this module looks for an even focus over the whole frame.

## Requirements
numpy, scipy, matplotlib; PyQt5 for the widget only.

## Stand-alone use
`python -m modules.focus_map [frame.png|frame.npy]`

## Embedding in an application
```python
from modules.focus_map import FocusMapWidget
widget = FocusMapWidget()
widget.set_wavelength_provider(lambda camera_bin: (lambda col: 0.73 * col + 838.0))  # optional
widget.grabRequested.connect(lambda: widget.begin_grab(hub.request()))   # one-shot frame protocol
hub.frame_ready.connect(widget.accept_snapshot)
hub.failed.connect(widget.snapshot_failed)
```
With *Live* on, the widget requests the next frame after each one is processed.

## Data formats / conventions
Frames are 2-D `(slit, wavelength)`, any dtype; the snapshot packet needs `request_id`,
`image` and optionally `camera_bin`. Contrast is unit-less (bar modulation / local level).
`FocusMap.fit` is `col = a0 + a1 r + a2 r^2` with `r = (row - centre_row) / half lit height`.

## API summary
`focus_map(image, period=None) -> FocusMap`, `bar_period(image)`, `focus_advice(fm, to_nm)`,
`linear_wavelength(slope, offset)`, `FocusMapWidget`.

## Threading
Core functions are pure. The widget is GUI-thread only; one 1024 x 1280 frame takes ~50 ms.

## Tests
`tests/test_focus_map.py`, `tests/test_module_contract.py`.
