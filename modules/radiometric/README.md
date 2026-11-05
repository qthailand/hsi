# radiometric

## Overview

Offline radiometric calibration and post-processing of ENVI hyperspectral cubes.
It turns a raw cube (sensor counts) into a calibrated, optionally cropped, spatially
binned and wavelength-resampled ENVI BIP cube. Calibration factors come from a scan
of a white reference (row-wise reflectance factors from a reflectance table such as a
Spectralon Excel sheet, restricted to an ROI; the table is never extrapolated: bands outside its
wavelength range get NaN reflectance and a warning, and the output header records the table,
reference cube and ROI rows as `reflectance reference*` with the warning in `calibration warnings`)
or from a fluorescence reference
(ROI-average spectrum, Savitzky-Golay smoothed, inverted). `core.py` is Qt-free;
`widget.py` holds the interactive `CalibrationWindow` and the `RoiDialog` ROI picker;
`workers.py` is a small `QObject` adapter that runs a core function on a `QThread`.
This processing is offline only; it is never part of live acquisition.

For automatic calibration, `reference.py` (Qt-free) keeps *radiometric references* per camera:
row-wise factors fitted on one Spectralon scan, stored with the camera setting they belong to
(`ReferenceKey`). A later scan is calibrated with an active reference only when its key matches
exactly (camera, bin, slit rows, exposure, gain, band centres, bin width, wavelength m/c, Smile,
data full scale (bit depth), Smile model and its creation time, dark and hot-pixel correction,
stage type), the reference target was scanned at most 4 h before the scan and the sensor
temperature is within 2 °C; scan length and speed may differ. Every key field is also an ENVI
header field, so a scan reopened from its file gives the same key.

A reference is made from the scan just finished with *Save as Radiometric Reference* (Scan
window): `RadiometricReferenceDialog` (`reference_dialog.py`, Qt) shows the cube, offers five ROI
tools - **Rect**, **Ellipse**, **Freehand** (all with handles afterwards: edge handles stretch one
side, e.g. left/right; the freehand polygon's vertices and bounding box are editable), **Detect**
(the largest bright rectangle/ellipse, then editable) and **Spectral pick** (click: the connected
region of similar spectra; Ctrl+click adds, Alt+click removes) - plus arrows / Ctrl+arrows / Ctrl+Z,
the reflectance table, the QC list, and *Create profile*. `roi_tools.py` (Qt-free) holds the
algorithms. Every ROI goes through one clean-up (`clean_mask`): saturated pixels (the scan
controller's saturation ceiling, else 95 % of full scale) out, a 3 px inset (the image border
is not an edge), a spectral gate (pixels unlike the ROI's own per-row spectrum: labels, tape,
holder), a ±5 % row gate (shadow, dust, bevel) and row bands: a step between neighbouring slit
rows - the row median brightness jumping by > 6 % (directly and across one row) or the row
spectrum jumping beyond the row medians' own noise - marks a band. Another material is dropped
(the one holding the spectral-pick seed, else the largest, stays); brightness bands only at the
ROI's ends (holder frame, bevel rows), so a dust line across the slit keeps its rows and a gap
between two picked targets is never cut. Vignetting, even a steep falloff, is continuous. The
dialog fits the factors with `window_fit="linear"` and an automatic averaging window (the fewest
rows giving ~500 ROI pixels per factor, usually 1-4, so slit dust and row patterns get their own
factors): the core's original 20-row window *mean* is biased at the ROI's ends wherever the
vignetting falls off (on a 552-row slit falling to 55 %: 5-9 % at the end rows; the line fit:
0.06-0.35 %). Create refuses an incomplete key (e.g. dark subtraction off), a missing table or an
empty / mostly saturated ROI, and asks when the self-check (factors applied to their own ROI)
misses the table by > 0.5 % RMS overall or > 1 % in any slit row with 20+ pixels (band-averaged).
The saturation ceiling and the bit depth used are the ones recorded with the scan. Factors are fitted and applied on
the cube in memory (`compute_rowwise_factors_array`, `apply_rowwise_factors`), with exactly the
values of the file-based calibration.

*Auto radiometric calibration* (Scan window, ticked by default, locked during a scan) calibrates
the scan **frame by frame**. When Start is pressed, `plan_frame_calibration` (`auto.py`, Qt-free)
builds the scan's key from its start metadata (settings, corrections, start sensor temperature; the
same fields a reference made of it would use) and searches the camera's active references (GUI
thread, a few small files). With a match, its factors are latched with the recording
(`ScanRecorder.set_recording(factors=...)`): the acquisition thread stores every recorded,
lambda-binned frame as `float32(frame) * F` (the level at which the factors were fitted; a key with
other bands, bin width or m/c does not match), the values `apply_rowwise_factors` gives, so the cube
equals a manual calibration with the same factors. Only the recorded copy is multiplied: the live
image, spectrum, histogram and sharpness stay in counts (exposure is still set on counts), while the
live cube preview, which reads the recording, shows reflectance. Cost: one multiply per frame, about
0.17 ms at 552 x 237 including the memory of the kept frame (0.35-1 ms for 1-2 k slit rows), against
a frame interval of 8-14 ms at 70-123 fps, so the frame rate does not drop while the acquisition
thread has that headroom. When the scan ends, `finish_frame_calibration` checks again with the final
metadata (mean of the start and end temperature); if the reference may not stay applied, or a frame
could not be multiplied, the frames are turned back into counts (`counts_from_reflectance`,
exact: the factors are finite and positive) and the scan says why.

The scan then holds only its reflectance (*Open Reflectance Viewer*; its export header drops the
raw-count fields and records the reference: `reflectance reference mode = automatic, frame by
frame`, its scan time, age, temperature difference and ROI tool). Its counts are rebuilt exactly the
first time they are needed (Open Cube Viewer, Calibration, Save as Radiometric Reference: a reference
is always fitted on counts). Untick before scanning a new reference target to record counts.

The after-scan path stays for a scan recorded in counts: ticking the box after a scan calibrates
that scan (`plan_auto_calibration` + `apply_reference` on a background thread, the counts kept), and
so does a reference whose factors have NaN bands (outside the reflectance table; those cannot be
multiplied into the only copy). Unticking after a scan drops its reflectance (counts rebuilt first).
When no reference fits, the scan window says why (e.g. `exposure: scan 12 ms, reference 10 ms`,
`reference is 5.0 h old`) at Start and after the scan, and nothing is calibrated. A reference just
created from the scan, and activated, calibrates that scan too (a direct check: its reflectance is
the table's).

## Requirements

* numpy, scipy (`CubicSpline`, `savgol_filter`; `scipy.ndimage` for the ROI tools), pandas
  (+ `openpyxl`/`xlrd` to read the reflectance Excel sheet).
* `roi_tools` also needs OpenCV (`cv2`: contour and shape fitting, path simplification) and
  `matplotlib.path` (polygon rasterisation; no Qt). It is imported only by the dialog and by
  callers that import `modules.radiometric.roi_tools`.
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
* **Radiometric reference store** (`RadiometricReferenceStore(folder)`; the application uses the
  camera profile's `radiometric_folder`, `camera_profiles/<backend>/<model>_<serial>/radiometric/`):
  one folder per reference, `ref_<YYYYMMDD>_<HHMMSS>[_<n>]/`, written to a hidden staging folder
  and renamed once complete, never modified afterwards:
  `reference.json` (`format_version` 1, `id` (= folder name; a copied folder is refused),
  `created_at` (save time) and `acquired_at` (when the target was scanned; the age counts from
  it), ISO times with offset, `key`
  = `ReferenceKey.to_dict()`, `reflectance_profile` file name, `avg_window_size`,
  `roi` {`span`, `pixels`, `rows`, `samples`}, `sensor_temperature_c`, free `info`),
  `factors.npy` + `factors.wavelength.npy` (`save_factor_profile`), `roi_mask.npy`,
  `roi_mean_spectrum.npy` (optional); the files are fsynced before the folder is renamed.
  `active.json` = `{"format_version": 1, "active": [ids]}`, replaced atomically; at most one
  active reference per key (activating replaces only readable references of the same key; a
  corrupt `active.json` is reported, never overwritten). Malformed ids in `active.json` are
  ignored (no path traversal); a reference with a malformed `reference.json` or unreadable
  factors is skipped and reported by `find` without affecting the others.

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
* `compute_rowwise_factors_array(cube, wavelengths, ref_excel, avg_window_size, mask, progress_callback=None,
  source_name='in-memory cube', window_fit='mean')`: the row-wise fit on a `(lines, samples, bands)` array
  or memmap, bit-identical to the file version. Both return `roi_mean_spectrum` (mean ROI counts per band)
  and `window_fit` too. `window_fit` (also on `compute_rowwise_factors`): `'mean'` = the window's mean
  counts (the original behaviour, still the default); `'linear'` = a pixel-weighted line through the
  window's row means evaluated at the row, unbiased on a vignetting falloff.
* `apply_rowwise_factors(cube, factors, factor_wavelengths=None, wavelengths=None, rows_per_chunk=64,
  progress_callback=None)` -> new float32 cube
  `cube * factors[:, None, :]` (row chunks, no float64 copy); refuses other slit rows / bands.
* `ReferenceKey.from_scan(metadata, wavelengths, lines)` (metadata = raw scan metadata merged with the
  export-header metadata, or `metadata_from_header` of an ENVI file); `.mismatch(scan_key)` -> ASCII
  reasons such as `exposure: scan 12 ms, reference 10 ms`; `.missing()` -> why a key cannot carry a
  reference (exposure, gain, bin, rows or bands unknown, dark subtraction off).
* `RadiometricReferenceStore(folder)`: `save(factors, key, *, reflectance_profile, avg_window_size, mask,
  roi_mean_spectrum=None, sensor_temperature_c=None, info=None, acquired_at=None, created_at=None,
  activate=False)`,
  `load(id)`, `references()`, `active()`, `active_ids()`, `activate(id)` -> ids replaced,
  `deactivate(id)`, `find(scan_key, *, scan_temperature_c=None, now=None, max_age_h=4.0,
  temperature_tolerance_c=2.0)` -> `ReferenceMatch(reference or None, rejected={id: reasons})`
  with `.ok` and `.reason()` (names the closest rejected reference).
* `RadiometricReference`: `factors()`, `roi_mask()`, `roi_mean_spectrum()`, `age_hours(now)`,
  `header_fields()` (the `reflectance_reference*` export-header keys).
* `reference_problems(reference, scan_key, *, scan_temperature_c, now, max_age_h, temperature_tolerance_c)`
  and `scan_temperature_c(metadata)` (mean of the scan's start/end sensor temperature, `None` if unknown).
  `RadiometricReferenceStore.status(now=None)` -> `(state "on"|"expired"|"off", badge text)`.
* `auto` (also exported): `plan_frame_calibration(store, start_metadata, wavelengths, lines)` -> `(plan,
  FrameCalibration or None)` (`.factors` read-only float32, `.reference`); `finish_frame_calibration(frame,
  end_metadata)` -> `(plan, problems)`; `counts_from_reflectance(cube_or_frame, factors)` -> uint16 counts;
  `plan_auto_calibration(store, metadata, wavelengths, lines, *, max_age_h=4.0,
  temperature_tolerance_c=2.0)` -> `AutoCalibrationPlan` (`.ok`, `.reference`, `.wanted` = the camera has an
  active reference at all, `.summary()`, `.header_fields()`, `.age_hours()`, `.delta_t_c()`);
  `apply_reference(cube, reference, wavelengths=None)` -> float32 reflectance; `reflectance_export_metadata(
  metadata, fields)` (scan metadata without the raw-count fields, plus the reference fields);
  `modules.radiometric.auto.memory_problem(shape)` (refuses a reflectance cube larger than 80 % of free RAM).
* `reflectance_at(ref_excel, wavelengths)` -> `(reflectance float32 per band, NaN outside the table; table range)`.
* `roi_tools` (import `modules.radiometric.roi_tools`): `rect_mask(shape, x0, y0, x1, y1)`,
  `ellipse_mask(shape, cx, cy, rx, ry, angle_deg=0)`, `polygon_mask(shape, [(x, y), ...])` (pixel (r, c)
  has its centre at x=c, y=r; inside = centre inside), `simplify_polyline`, `saturation_map(cube, ceiling=None,
  full_scale=None)` -> `(mask, source)`, `brightness_map`, `rgb_preview`, `detect_shapes(brightness, saturated)`
  -> `[ShapeCandidate]` largest first, `largest_shape`, `SpectralIndex(cube, wavelengths, saturated).pick(row, col,
  threshold_deg=None)` -> `SpectralPick`, `robust_threshold`, `spectral_outliers`, `clean_mask(mask, brightness,
  saturated, *, inset_px=3, row_tolerance=0.05, index=None | cube+wavelengths)` -> `CleanupResult`,
  `assess_roi(cleanup, cube, wavelengths, brightness, ...)` -> `RoiQuality(items, numbers)`.
* `RadiometricReferenceDialog(cube, wavelengths, metadata=None, store=None, saturation_ceiling=None, parent=None)`:
  signal `reference_saved(RadiometricReference)`, attribute `reference` after `exec_()`; programmatic
  `set_rect`, `set_polygon`, `spectral_pick`, `nudge`, `undo`, `set_profile`, `create_profile`, `problems()`.
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
`tests/test_radiometric_reference.py` covers the in-memory fit/apply (equal to the file pipeline),
`ReferenceKey` matching and the reference store; `tests/test_radiometric_roi_tools.py` the ROI modes,
clean-up and QC on a synthetic Spectralon scene (vignetting, bevel, shadow, screw, label, hot spot,
white paper, Smile-like drift); `tests/test_radiometric_reference_dialog.py` the dialog, the Scan
window button / badge and the integration (offscreen; message boxes are replaced).
`tests/test_radiometric_auto.py` the automatic calibration: equal to the manual path (T1), every setting
difference refused with its reason (T6), age / temperature limits, the export header, the Scan
window's per-scan box, stale results and failures; frame by frame: the recorder's multiply (equal to the
cube path), latching, the start / end checks, exact counts; `tests/test_app_smoke.py` a frame-by-frame scan
through the real engine thread.

## Known structure limits

`widget.py` is one large class (`CalibrationWindow`, about 1.3k lines) that mixes UI
construction, state, file dialogs and the calibration workflow (profiles, memmap
management, export); it is the next candidate for splitting into panels (input,
calibration, resample/binning, preview/spectra) with the workflow moved to a controller.
The module also depends on the private `_read_envi_band_float32` of `cube_io`, profiles carry
no `format_version`, and the window changes the application style/palette and has no
`closeEvent` that stops its worker thread.
