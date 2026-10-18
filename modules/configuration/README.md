# configuration

## Overview

`ConfigStore` persists the application settings as a **split `config/` directory**: one JSON
file per section plus an `app.json` manifest that is published last, and the wavelength files
under `calibration/wavelength/`. Writes are atomic (unique temp file, then replace), so a failed
save cannot replace the active file. There is no Qt and no frame-loop I/O.

## Requirements

* Standard library only, plus `modules.wavelength_calibration` (`__requires__ = ("wavelength_calibration",)`),
  which `ConfigStore.read` imports to load and validate the active wavelength profile.
* No widget.

## Stand-alone use

There is no launcher (library only). Quick inspection:

```
python -c "from modules.configuration import ConfigStore; print(ConfigStore('config').read())"
```

## Embedding in an application

```python
from modules.configuration import ConfigStore

store = ConfigStore("config")            # project root = parent of "config"
if not store.directory.exists():
    store.initialize(defaults)           # flat dict with the section keys plus "m" and "c"
settings = store.read()                  # flat dict; "m"/"c" from the active profile if one is set
# after the worker applied a calibration successfully:
store.save_wavelength_default(profile, camera_config)
```

`profile` is a `modules.wavelength_calibration.WavelengthCalibration`; `camera_config` needs
`camera_backend`, `camera_id`, `camera_index` and `raw_inverted` attributes.

## Data formats / file layouts

```
<project>/
  config/
    app.json              {"format_version": 1,
                           "sections": {"camera": "camera.json", "scan": "scan.json",
                                        "processing": "processing.json", "dark": "dark.json",
                                        "smile": "smile.json"},
                           "wavelength_base": "calibration/wavelength/base.json",
                           "active_wavelength": null | "calibration/wavelength/profile_<uuid>.json"}
    camera.json           camera_backend, camera_id, camera_index, cti_path, default_gain,
                          default_exposure_ms, default_bin
    scan.json             stage_type, scan_trim_extra_ms (optional, default 0)
    processing.json       resolution, raw_inverted, rgb_target_wavelengths, preprocessing_backend,
                          profile_timing (optional, default true: log [profile] stage timings),
                          resolution_source (optional, "config" | "fwhm", default "config"),
                          fwhm_step_factor (optional, default 1.0)
    dark.json             dark_frame_folder, dark_frame_n_frames
    smile.json            smile_map_folder
    camera_<uuid>.json    legacy only: older versions wrote it on Save as Default; used until camera.json is edited after it
  calibration/wavelength/
    base.json             {"m": nm/px (unbinned), "c": nm, "units": "nm"}
    profile_<uuid>.json   WavelengthCalibration fields (format_version 1, units "nm", stage) plus
                          "compatibility": {camera_backend, camera_id, camera_index, raw_inverted, camera_bin}
    <backend>/wavelength_bin<N>.json   per camera setting (modules.contracts.CalibrationStore);
                          wavelength_folder(backend) is this folder. The scan uses it at start and
                          after every bin change, ahead of the m/c returned by read()
    <backend>/resolution_bin<N>.json   median FWHM (nm) per camera setting; the bin width when
                          resolution_source is "fwhm"
```

* Section files hold only the keys listed (see `SECTIONS`); unknown keys passed to `initialize` are dropped.
* `manifest["sections"]` maps a section to its current file name; `app.json` is rewritten last.
* `read()` merges all sections, sets `m`/`c` from `base.json`, then (if `active_wavelength` is set)
  checks the profile's `compatibility` block against the current settings and replaces `m` with the
  profile's unbinned `base_m` and `c` with its intercept (`_wavelength_profile` holds the object).
  A mismatch raises `ValueError` ("select compatible settings or clear active_wavelength").
* Units: `m` nm per pixel, `c` nm, exposure in ms.
* Old `profile_*.json` / `camera_*.json` files are never deleted by the store.

## API summary

* `SECTIONS`: section name to the keys it stores.
* `ConfigStore(directory)`: `.directory`, `.root`, `.manifest`;
  `initialize(data)`, `read()`, `save_wavelength_default(profile, config)`, `wavelength_folder(camera_backend)`.

## Threading

Not thread-safe: single writer at a time (the manifest update in `save_wavelength_default` is a
read-modify-write). Keep it out of the frame loop.

## Tests

`python -m pytest -q tests/test_configuration.py tests/test_module_contract.py`
