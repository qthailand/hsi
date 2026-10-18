# cube_viewer

## Overview

Interactive viewer for a hyperspectral cube `(slit, scan, wavelength)`:

* `CubeViewerWindow` - the full viewer: band slider / play-pause sweep, single-band and RGB-composite display, zoom and pan, click-to-spectrum (every left click adds a colour-coded curve and dot), cube info and an "Export Datacube" dialog (ENVI BIP, GeoTIFF, raw binary).
* `CubeImageCanvas` - single-band image canvas with zoom/pan, `pixel_clicked(ix, iy)` and dot overlay.
* `RGBCompositeControls` - single-band / RGB toggle, R/G/B band spin boxes (default 645/555/465 nm) and stretch slider.

## Requirements

* PyQt5, NumPy, `pyqtgraph` (spectrum plot), `matplotlib` (dot colours).
* Other modules: `cube_io` (export workers, ENVI loading), `cube_preview`, `live_view` (spectrum plot, display controls), `ui_kit` (zoom/pan, range slider) - see `__requires__`.
* `import modules.cube_viewer` does not import PyQt5; classes load on first attribute access.

## Stand-alone use

```
python -m modules.cube_viewer               # synthetic 96 x 128 x 60 cube
python -m modules.cube_viewer scan.hdr      # ENVI cube (opened as a memory map)
```

Keys: Left/Right = band, Space = play/pause, R = reset, E = export, `+` `-` `0` `1` = zoom in/out/fit/100 %.

## Embedding in an application

`CubeViewerWindow` takes a cube, so it is not constructible without arguments (it is not in `__widgets__`; `CubeImageCanvas` and `RGBCompositeControls` are).

```python
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from modules.cube_viewer import CubeViewerWindow

app = QApplication.instance() or QApplication([])
cube = np.random.randint(0, 4096, (32, 48, 40), dtype=np.uint16)   # (slit, scan, wavelength)
wavelengths = np.linspace(400, 1000, 40)                            # nm, or None

viewer = CubeViewerWindow(cube, wavelengths, resolution=1,
                          config={"m": 0.2903, "c": 399.791}, camera_bin=1)
host = QMainWindow()
host.setCentralWidget(viewer)       # or viewer.show() as a top-level window
host.show()
print(viewer.status_info())
```

Closing the widget releases its references to the cube; the host keeps ownership of the array it passed in.

## Data formats / conventions

* Cube `(slit, scan, wavelength)` = `(H, W, bands)`, any numeric dtype, held by reference.
* Wavelengths: 1-D, nm, one per band, or `None` (band indices are then shown and exported).
* Pixel `(ix, iy)` = `(column, row)`; the spectrum is `cube[iy, ix, :]`.
* Export transposes (copy) to band-first `(wavelength, slit, scan)` for the `cube_io` writers; `config` carries `m`, `c` and optional `pixel_size`.

## API summary

| Name | Purpose |
|------|---------|
| `CubeViewerWindow(cube, wavelengths=None, resolution=2, display_mapper=None, config=None, camera_bin=None)` | full viewer; `on_band_changed(band)`, `update_band_image()`, `status_info()` |
| `CubeImageCanvas()` | signals `pixel_clicked(int, int)`, `zoom_changed(float)`; `set_array`, `set_display_mapper`, `add_click_point`, `clear_click_points` |
| `RGBCompositeControls()` | signals `mode_changed(str)`, `composite_updated(ndarray)`; `set_cube(cube, wavelengths)`, `is_rgb_mode()` |

## Threading

All widgets live in the GUI thread. The export dialog moves a `cube_io` export worker to its own `QThread`; closing the dialog or the viewer joins that thread (blocks until a running export ends).

## Tests

Contract checks in `tests/test_module_contract.py`; export adapters are exercised by `tests/test_stage2_regression.py`. Run `python -m pytest -q tests`.
