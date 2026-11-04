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
  `StoredResolution` (`fwhm_nm`, `ok`/`problems`, `details`). Optional
  `fallback_wavelength_folder` / `fallback_smile_folder` are read only: `find_wavelength`,
  `find_smile`, `find_resolution` (and the loaders) return the camera's own file, else the
  fallback folder's, else `None`; `is_fallback(path)` says which. The `*_path` methods are
  where a calibration is saved (always the camera's folders).
* **Camera profile** (`camera_profile.py`): `CameraProfile.for_camera(profiles_folder, backend,
  model, serial)` is `<profiles_folder>/<backend>/<model>_<serial>/` (`profile_name`: characters
  outside `A-Z a-z 0-9 . _ -` become `_`; `<model>` or `SN<serial>` when only one is known;
  `unknown` when neither) with `dark_folder` (`dark/`), `smile_folder` (`smile/`),
  `wavelength_folder` (`wavelength/`) and `info_path` (`profile.json`: `format_version`,
  `backend`, `model`, `serial`, `created_at`, `last_opened_at`, plus what `write_info(**extra)`
  adds; the HSI app stores `frame_width`/`frame_height`/`camera_bin` at open). The HSI app saves every calibration of the open
  camera there and reads the older per-backend folders (`dark_frames/<backend>/`,
  `smile_maps/<backend>/`, `calibration/wavelength/<backend>/`) only as a fallback.
  `write_info` keeps keys it is not given, so values the user entered survive every camera open.
* **Sensor pixel pitch** (in `profile.json`): `set_pixel_pitch(um, note)` saves the pitch at
  bin 1 in micrometres (0.5-50 µm, else `ValueError`) as the `"sensor"` block
  (`SensorInfo`: `pixel_pitch_um`, `note`, `entered_at`, `model`, `serial`); the previous block
  moves to `"sensor_history"`. `pixel_pitch_um` is `None` until it is entered.
  `pitch_warnings(sdk_pitch_um)` lists reasons to re-check it (profile shared by cameras without
  a serial, entered for another camera, SDK value more than 1 % off). `sdk_pixel_pitch_um(value)`
  turns a backend's `info.pixel_size` into a pitch, or `None` for the `1.0` placeholder.

## API summary

`CubeData` (`validate()`), `default_reference_row`, `normalize_wavelength_config`,
`WavelengthCalibration`, `CameraSetting`, `CalibrationStore`, `StoredWavelength`, `StoredResolution`,
`CameraProfile`, `SensorInfo`, `profile_name`, `sdk_pixel_pitch_um`, `PIXEL_PITCH_RANGE_UM`.
Implementation: `geometry.py`, `wavelength.py`, `calibration_set.py`, `camera_profile.py`, `__init__.py`.

## Threading

All functions are pure and thread-safe. `CubeData` is frozen, but its array is shared by
reference: do not mutate it while another thread reads it.

## Tests

`python -m pytest -q tests/test_smile_wavelength_profile.py tests/test_module_contract.py`
