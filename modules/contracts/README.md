# contracts

## Overview

Shared data layouts and conventions that several modules must agree on (module-contract
rule R7), so that no module depends on another just to exchange data:

* `CubeData`: frozen container for a 3-D cube plus explicit axis names.
* `default_reference_row(height)`: the one rule for the *reference row* used by Smile
  correction, quality diagnostics and wavelength calibration.
* `normalize_wavelength_config(config)`: accepts both wavelength-calibration file layouts
  and returns the "Standards" layout.

## Requirements

numpy only. No Qt and no other `modules.*` package (`__requires__ = ()`). No widget.

## Stand-alone use

There is nothing to launch; this is a library of definitions.

```python
from modules.contracts import default_reference_row
default_reference_row(552)   # 276
```

## Embedding in an application

```python
import numpy as np
from modules.contracts import CubeData, default_reference_row, normalize_wavelength_config

cube = CubeData(np.zeros((552, 300, 640), np.uint16),         # (slit, scan, wavelength)
                wavelengths=np.linspace(400, 900, 640), units="counts")
cube.validate()                      # at an import/export boundary, not in the frame loop

row = max(0, default_reference_row(frame.shape[0]))             # frame: (slit rows, spectral columns)
standards = normalize_wavelength_config(json.load(open("profile.json")))
```

## Data formats / conventions

* **Cube**: `array` with three axes; default `axes = ("slit", "scan", "wavelength")`. Axis names
  must be unique and include `"wavelength"`; `wavelengths` (optional, nm) is 1-D with one value per
  sample of the wavelength axis. `CubeData` never copies, converts or validates on construction
  (the caller keeps ownership of the array); call `validate()` explicitly.
* **Spectral image**: `(slit rows, spectral columns)`, zero-based pixels.
* **Reference row**: centre row `height / 2` for even heights, `floor(height / 2) - 1` for odd
  heights (the Smile Analysis convention); `height == 1` yields `-1` (callers clamp to 0);
  `height <= 0` raises `ValueError`.
* **Wavelength-calibration JSON, two layouts** (both accepted wherever a wavelength calibration is loaded):
  1. *Profile* (written by `modules.wavelength_calibration`): `format_version` 1, `units` `"nm"`,
     `m`, `c`, `pixels`, `wavelengths`, `camera_bin`, `frame_shape`, `selected_row` (`m`/`c` refer
     to the pixels of the image at its own hardware bin).
  2. *Standards* (used by Smile quality analysis): `m_nm_per_px`, `c_nm`,
     `standards = [{"wavelength_nm", "x_px", "half_window_px"?, "blended"?}]`, optional `reference_row_y`.

  `normalize_wavelength_config` returns a Standards dict unchanged if it already has `standards`
  (it must then have `m_nm_per_px` and `c_nm`); a profile becomes
  `{m_nm_per_px: m, c_nm: c, standards: [{wavelength_nm, x_px}...], reference_row_y: selected_row,
  source: "wavelength_calibration_profile"}`. It raises `ValueError` for non-nm units, a profile
  without matched points, or an unrecognised object.

* **Calibrations per camera setting** (`calibration_set.py`): `CameraSetting(camera_backend,
  camera_index, camera_bin, raw_inverted, frame_shape)`; `mismatch(other)` lists what differs
  (frame shape only when both are known). `CalibrationStore(wavelength_folder, smile_folder)`:
  `wavelength_bin<N>.json` (a normal profile plus `camera_setting` and `saved_at`; loadable with
  `WavelengthCalibration.load`) and `smile_model_bin<N>.npz`; `save_wavelength` copies the previous
  file to `history/<name>.<timestamp>.json`; `load_wavelength` returns `StoredWavelength` with
  `ok`/`problems`. `resolution_bin<N>.json` (same folder as the wavelength profiles) holds the
  median FWHM in nm (`fwhm_nm`, plus `camera_setting`, `saved_at` and free-form details such as
  the per-line values): `save_resolution(fwhm_nm, setting, details=None)` (refuses a non-positive
  or non-finite FWHM; previous file to `history/`), `load_resolution(setting)` returns
  `StoredResolution` (`fwhm_nm`, `ok`/`problems`, `details`). The HSI app uses
  `config/calibration/wavelength/<backend>/` and `smile_maps/<backend>/`.

## API summary

`CubeData` (`validate()`), `default_reference_row`, `normalize_wavelength_config`,
`WavelengthCalibration`, `CameraSetting`, `CalibrationStore`, `StoredWavelength`, `StoredResolution`.
Implementation: `geometry.py`, `wavelength.py`, `calibration_set.py`, `__init__.py`.

## Threading

All functions are pure and thread-safe. `CubeData` is frozen, but its array is shared by
reference: do not mutate it while another thread reads it.

## Tests

`python -m pytest -q tests/test_smile_wavelength_profile.py tests/test_module_contract.py`
