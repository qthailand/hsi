# radiometric

## Overview

Offline radiometric calibration and post-processing of ENVI hyperspectral cubes.
It turns a raw cube (sensor counts) into a calibrated, optionally cropped, spatially
binned and wavelength-resampled ENVI BIP cube. Calibration factors come from a scan
of a white reference (row-wise reflectance factors from a reflectance table such as a
Spectralon Excel sheet, restricted to an ROI) or from a fluorescence reference
(ROI-average spectrum, Savitzky-Golay smoothed, inverted). `core.py` is Qt-free;
`widget.py` holds the interactive `CalibrationWindow` and the `RoiDialog` ROI picker;
`workers.py` is a small `QObject` adapter that runs a core function on a `QThread`.
This processing is offline only; it is never part of live acquisition.

## Requirements

* numpy, scipy (`CubicSpline`, `savgol_filter`), pandas (+ `openpyxl`/`xlrd` to read the
  reflectance Excel sheet).
* `modules.cube_io` (ENVI header parsing and band reading): `__requires__ = ('cube_io',)`.
* PyQt5, matplotlib and `spectral` only for the GUI classes; they are loaded lazily, so
  `import modules.radiometric` does not import PyQt5.

## Stand-alone use

`python -m modules.radiometric` without `--input` opens the calibration window in a
`QMainWindow`. With `--input` it is an offline CLI:

```
# row-wise reflectance factors (ROI mask = boolean .npy with the shape of the reference image)
python -m modules.radiometric --input white.bsq --compute rowwise \
    --roi-mask roi.npy --reference-spectrum spectralon.xlsx --avg-window 20 --output rowwise.npy

# fluorescence factors
python -m modules.radiometric --input fluo_ref.bsq --compute fluorescence \
    --roi-mask roi.npy --output fluo_factors.npy

# calibrated, cropped (X0 X1 Y0 Y1), 2x2-binned cube
python -m modules.radiometric --input scan.bsq --output scan_cal.bip \
    --rowwise rowwise.npy --crop 0 640 0 480 --spatial-bin 2

# same, resampled to 400..800 nm (STOP excluded) every 1.5 nm
python -m modules.radiometric --input scan.bsq --output scan_rs.bip \
    --factors fluo_factors.npy --target-grid 400 800 1.5
```

`--factors` (per band) and `--rowwise` (per row and band) are mutually exclusive. Without
either, the raw counts are cropped/binned/resampled. The CLI prints the output path.

## Embedding in an application

Headless (no Qt):

```python
import numpy as np
from modules.cube_io.envi import parse_envi_header
from modules.radiometric import (compute_rowwise_factors, process_cube, resample_cube)

ref_meta = parse_envi_header('white.bsq')
mask = np.zeros((ref_meta['lines'], ref_meta['samples']), bool); mask[100:200, 50:300] = True
res = compute_rowwise_factors('white.bsq', ref_meta, 'spectralon.xlsx', 20, mask,
                              progress_callback=print)      # prints 0, 1, ..., bands
meta = parse_envi_header('scan.bsq')
process_cube(dict(raw_filepath='scan.bsq', raw_meta=meta, save_path='scan_cal.bip',
                  crop_coords=(0, meta['samples'], 0, meta['lines']),
                  is_wavelength=True, band_count=meta['bands'],
                  factors=None, rowwise_factors=res['factors'], binning_factor=1))
```

GUI, inside another application:

```python
from modules.radiometric import CalibrationWindow

win = CalibrationWindow()                 # no arguments; a QWidget
win.calibration_done.connect(on_done)     # (output_path: str, raw_meta: dict)
win.load_from_memory(cube, wavelengths)   # cube (bands, H, W) or (H, W, bands), float-able; nm
win.show()
```

`load_from_memory` writes the cube to `<tempdir>/scan_<name>.bip` (+ `.hdr`) and loads it
like a file; `load_from_scan(path)` loads an existing ENVI file. Note that the constructor
sets the application-wide Fusion style and a dark palette (`QApplication.setPalette`).

## Data formats

* **Input**: ENVI data file + `.hdr` (any interleave and data type supported by
  `modules.cube_io.envi`). Header fields used: `lines`, `samples`, `bands`, `data type`,
  `wavelength` (nm). Bands are read one at a time as float32 `(lines, samples)` images.
* **Axes**: in memory and in the preview the cube is `(lines/slit, samples/scan, bands/wavelength)`.
  Crop boxes are `(x0, x1, y0, y1)` with `x` on `samples`, `y` on `lines`, half-open.
* **Output cube**: ENVI BIP, `(lines, samples, bands)`, little endian (`byte order = 0`),
  data type 4 (float32) when factors are applied, otherwise the input type (uint16, or
  float32 for type-4 input). The header `<save_path>.hdr` records description, spatial binning,
  `wavelength units = Nanometers` and, if requested, the wavelengths.
* **Profiles**: plain NumPy `.npy` files (no version header): row-wise `(lines, bands)`
  float32, fluorescence `(bands,)`. Apply by multiplication to counts. Saving also writes
  the band centres (nm, float64 `(bands,)`) of the reference cube to `<name>.wavelength.npy`
  beside `<name>.npy` (when the reference header has wavelengths). Factors are applied by band
  index, so they are refused (`ValueError`, message box in the window) for a cube with another
  band count, or with band centres further apart than a quarter of the cube's median band step
  (ENVI headers keep 2 decimals). A profile without a sidecar is checked by band count only.
* **Reflectance table**: Excel sheet without header; column 0 wavelength (nm), column 1
  reflectance. The tables of the reference targets live in `reflectance_profiles/`
  (`REFLECTANCE_PROFILE_DIR`; Spectralon 5/20/40 %, PVC). The window lists them in
  *Reference* (`list_reflectance_profiles()`); *Browse...* selects a table elsewhere. A file
  copied into the folder appears the next time the window is opened.
* **Temporary cubes** (`build_raw_cube_tmp`, `build_cal_cube_tmp`): header-less memmaps of
  shape `(lines * bands, samples)` (band planes stacked), uint16 (raw, clipped) or float32
  (calibrated); pixel `(y, x)` of band `b` is `[y + b * lines, x]`. The GUI writes them
  to the system temp directory.
* **ROI mask**: boolean `(lines, samples)` of the reference scan.

## API summary

* `compute_rowwise_factors(ref_filepath, ref_meta, ref_excel, avg_window_size, mask, progress_callback=None)`
  -> `{'factors': (lines, bands), 'span': (top, bottom)}` (alias of `rowwise_calculation_heavy_logic`).
* `compute_fluorescence_factors(ref_filepath, ref_meta, mask, progress_callback=None)`
  -> `{'factors': (bands,)}` (alias of `fluo_calculation_heavy_logic`).
* `process_cube(params, progress_callback=None)` -> output path (alias of `create_datacube_logic`).
  `params`: `raw_filepath, raw_meta, save_path, crop_coords, is_wavelength, band_count,
  factors, rowwise_factors, binning_factor`.
* `resample_cube(params, progress_callback=None)` -> output path (alias of
  `create_datacube_resampled_logic`); like `process_cube` without `band_count` and with
  `lam_target` (nm).
* `build_raw_cube_tmp(params, ...)` (`folder, raw_meta, save_path, crop`) and
  `build_cal_cube_tmp(params, ...)` (`raw_filepath, raw_meta, save_path, crop, factors,
  rowwise_factors`) -> `{'path', 'lines', 'samples', 'bands', 'dtype'}`.
  These three export functions also take an optional `factor_wavelengths` (nm of the factors'
  bands) and call `check_factor_bands` before writing anything.
* `check_factor_bands(n_bands, factor_wavelengths, cube_meta)`: raises `ValueError` unless
  factors of `n_bands` bands (band centres `factor_wavelengths`, or `None` = unknown) fit the cube.
* `save_factor_profile(path, factors, wavelengths=None)` -> profile path (with the sidecar);
  `load_factor_profile(path)` -> `(factors, wavelengths or None)`.
* `CalibrationWindow`: `load_from_memory`, `load_from_scan`, signal
  `calibration_done(str, dict)`. `RoiDialog(parent, ref_filepath, ref_meta, wl, show_spectrum)`:
  `get_mask()`. `Worker(fn, *args, **kwargs)`: signals `finished(object)`, `error(str)`,
  `progress(int)`; slot `run`.

## Threading

Core functions are synchronous, hold no shared state and may run on any thread; they are
heavy (one pass over all bands), so call them off the GUI thread. `progress_callback(int)`
is invoked from the calling thread with `0` and then the number of bands processed (not a
percentage). GUI classes must be used on the GUI thread; `CalibrationWindow` runs its jobs
on its own `QThread` through `Worker` and updates a progress bar whose maximum is the band count.

## Tests

Run `python -m pytest -q tests/test_module_contract.py -k radiometric` and the radiometric /
calibration tests in `tests/` (e.g. `python -m pytest -q tests -k "radiometric or calibration or bsq"`).

## Known structure limits

`widget.py` is one large class (`CalibrationWindow`, about 1.3k lines) that mixes UI
construction, state, file dialogs and the calibration workflow (profiles, memmap
management, export); it is the next candidate for splitting into panels (input,
calibration, resample/binning, preview/spectra) with the workflow moved to a controller.
The module also depends on the private `_read_envi_band_float32` of `cube_io`, profiles carry
no `format_version`, and the window changes the application style/palette and has no
`closeEvent` that stops its worker thread.
