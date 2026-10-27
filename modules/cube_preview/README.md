# cube_preview

Realtime false-colour preview of a pushbroom scan: the image grows one column per frame, the
user clicks pixels to plot their spectra and drags R/G/B lines to change the composite.
"Check Circle" / "Check Square" (right of the image) help calibrate the stage speed: they
find every round / square target, outline and number them in yellow and report each one's
circularity / squareness and scan/slit ratio W/H, plus the mean W/H over all of them.

## Overview

| Name | File | Purpose |
|------|------|---------|
| `RealtimeCubePreviewWidget` | `widget.py` | image + percentile slider + status + spectrum plot |
| `CenteredPixmapLabel` | `widget.py` | fit/centre image view with wheel zoom, middle-drag pan, click dots |
| `CubeRenderWorker` | `render.py` | worker (own `QThread`, 15 fps) that builds the RGB image and extracts spectra |
| `find_closest_band` | `widget.py` | helper: band index nearest to a wavelength |
| `detect_shapes`, `detect_shape`, `ShapeResult` | `shape_check.py` | find every circle / square target in the RGB image and measure it |
| `summarize_aspects`, `AspectSummary` | `shape_check.py` | combine the targets' W/H into one speed factor, leaving out outliers |

## Requirements

* Python 3.9+, numpy, opencv-python (shape check), matplotlib (tab20 colours), PyQt5, pyqtgraph
* `modules.ui_kit` (`RangeSlider`, `ZoomPanMixin`); `__requires__ = ("ui_kit",)`

## Stand-alone use

```
python -m modules.cube_preview
```

feeds synthetic frames (one every 30 ms) into the preview; click the image to plot spectra.

## Embedding in an application

```python
import numpy as np
from PyQt5.QtWidgets import QApplication
from modules.cube_preview import RealtimeCubePreviewWidget

app = QApplication([])
preview = RealtimeCubePreviewWidget()          # or rgb_target_wavelengths=(1500, 1300, 1100) for SWIR
app.aboutToQuit.connect(preview.cleanup)       # IMPORTANT: stops the render thread
preview.show()

wavelengths = np.linspace(400, 1000, 200)      # nm, or None
preview.reset()                                # start of a scan
# for every camera frame (GUI thread):
#     preview.push_frame(frame, wavelengths, bands=200)   # frame (slit, wavelength)
# preview.push_frames(list_of_frames, wavelengths, 200)   # batch variant
# or show a list that grows elsewhere (read by reference, never copied):
#     preview.set_frame_source(recording_list, wavelengths)
#     preview.set_frame_source(final_list, wavelengths, shift=n)  # same frames, n cut off the start;
#                                                              # points/spectra/bands/zoom stay
# preview.set_saturation_mask(mask)                       # optional (n_frames, slit) bool
```

## Data formats / conventions

* Cube = `(slit, scan, wavelength)`; one frame = 2-D `(slit, wavelength)`;
  `cube[y, x, :] == frames[x][y, :]`.
* Image = `(slit rows, scan columns)`; click coordinates `(x, y)` = `(scan column, slit row)`.
* `wavelengths`: 1-D array in nm, one per band, or `None` (x axis shows band index).
* Frames are stored by reference (no copy): do not modify them after pushing.
* RGB from the worker is float32 `(h, w, 3)` in `0..1`, min-max normalised over the scan so far;
  the widget converts to uint8 and applies a per-channel percentile stretch (slider, default 2..98 %).

## Stage-speed check (Check Circle / Check Square)

Scan one or more round or square targets (a printed dot grid works well), then press the
button. Every target of that kind is outlined and numbered (reading order: top to bottom,
then left to right). The check runs on the unstretched image from the render worker; the %
slider only changes the display (its clipping would fatten small targets and pull W/H
towards 1).

| Value | Meaning |
|-------|---------|
| Circularity | 4πA/P², 1.000 = circle (square ≈ 0.81). Barely changes for a small stretch (1.1:1 ellipse → 0.994). |
| Squareness | (short/long side of the min-area rectangle) × fill, 1.000 = square (circle ≈ 0.79). |
| W/H (scan/slit) | `sqrt(mu20/mu02)` of the target; exact stretch for circles and squares at any angle. |
| Speed × | = W/H (with several targets: their mean). W/H > 1: target too wide along the scan, stage too slow → new speed = speed × W/H. |

With two or more targets the panel shows the mean W/H, its spread (± standard deviation,
min – max) and a table with one row per target; selecting a row highlights that target in
cyan. A target whose W/H is ≥ 1.5 % and ≥ 3.5 robust sigmas off the median of all is drawn
orange, marked "no" and left out of the mean (`summarize_aspects`). "Copy" puts the table and
the summary on the clipboard, tab separated.

How the targets are found (`shape_check.py` docstring has the details): dead slit rows and
dropped / saturated frames are repaired; a row + column background model removes
vignetting, stripes and illumination drift; candidates are thresholded relative to the
noise in both polarities (bright and dark targets), so small and dim targets are kept;
targets touching each other are split (a disc cut off a neighbour is measured by an ellipse
through its free edge); each target is measured at the mid level between its own interior
and the background around it, on an upsampled window (sub-pixel). Circles and squares are
told apart with the stretch undone, so a badly stretched target is still recognised.
Targets touching the image border and holes / spots inside a larger target are ignored.

Accuracy on synthetic scans with known geometry (noise, blur, vignetting, stripes, dead
lines, touching / dim / small targets; 167 + 292 + 309 targets): every target found with no
false detection except four squares under 15 px a side, W/H error median 0.04–0.09 %,
95th percentile 0.4–0.75 %. Limits: a square under ~15 px a side blurs into a disc and is
not reported as a square; a target closer than ~1 px to another one is measured from its
free edge only.

"Clear" removes the outlines, the table and the values; `reset()` (new scan) does too.
"Clear Spectra" does not.

## API summary

* `RealtimeCubePreviewWidget(rgb_target_wavelengths=(645, 555, 465))`: `push_frame(frame, wavelengths, bands)`,
  `push_frames(frames, wavelengths, bands)`, `set_frame_source(frames, wavelengths=None, shift=0)`,
  `reset()` (the next image is shown fitted to the canvas),
  `set_saturation_mask(mask)`, `cleanup()`; attributes `image_label` (white background), `frame_label`,
  `pixel_label`, `shape_label`, `shape_table`, `spec_plot`.
* `CenteredPixmapLabel(parent=None, background="#0d0d0d")`: signals `clicked(int, int)`, `zoom_changed(float)`; `setPixmapScaled`, `getPixmap`,
  `add_click_point`, `clear_click_points`, `click_points`, `set_shape_overlay(outline, text)`,
  `set_shape_overlays([(outline, text), ...], flagged=(), selected=None)`, `select_shape_overlay(index, text=None)`,
  `shape_overlays()`, `clear_shape_overlay`.
* `shape_check.detect_shapes(rgb, "circle" | "square") -> list[ShapeResult]` (pure, any thread; RGB or grey,
  any dtype): `kind`, `score`, `circularity`, `squareness`, `aspect`, `width`, `height`, `center`, `area`,
  `outline`, `touching`. `detect_shape(...) -> ShapeResult | None` returns only the best scoring one.
* `shape_check.summarize_aspects(results) -> AspectSummary | None`: `count`, `aspect` (mean W/H of the used
  targets), `std`, `min`, `max`, `outliers` (indices left out).
* `CubeRenderWorker`: signals `rgb_ready(object, int, int)`, `frames_count_ready(int)`,
  `spectrum_ready(object, object, int, int)`; `start`, `stop`, `push_frame(s)`, `set_frames`, `set_wavelengths`,
  `set_rgb_bands`, `request_spectrum`, `reset`.

`__widgets__ = ("RealtimeCubePreviewWidget", "CenteredPixmapLabel")`.

## Threading

The preview widget is GUI-thread only and starts a `CubeRenderWorker` thread in its constructor.
Frame pushes and `reset` are lock-protected; results return via queued signals. The thread is not
stopped automatically: call `cleanup()` (e.g. from `closeEvent` or `QApplication.aboutToQuit`).
`import modules.cube_preview` does not import PyQt5.

## Tests

`tests/test_module_contract.py` (`-k cube_preview`) checks the contract and builds each widget in a
`QMainWindow`; `tests/test_app_smoke.py` covers the application wiring;
`tests/test_cube_preview_shape_check.py` covers the shape check.
