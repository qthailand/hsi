# flatfield

## Overview

Flat-field correction along the slit of a pushbroom scan. A scan of a target that is uniform
along the whole slit (a white strip) gives the relative response of every slit row and band;
dividing a scan by it evens the rows out.

| Piece | What it does |
|---|---|
| `compute_flatfield(cube, mask, ...)` | fits the `(lines, bands)` profile on the ROI of the strip; interpolates joints (`seam_threshold`), reports lasting brightness steps (`step_threshold`) and the saturated share |
| `apply_flatfield(cube, profile)` | divides a cube by the profile, in row chunks |
| `evening_factors(profile)`, `check_profile(profile)` | factors to multiply by; validation of a profile array |
| `FlatFieldStore` | keeps profiles per camera, finds the newest one that fits a scan (`find`), reports the one for a bin (`status`) |
| `FlatFieldProfile`, `FlatFieldMatch`, `flatfield_mismatch` | a stored profile; the result of `find`; why a profile does not fit a scan |
| `dialog.FlatFieldDialog`, `dialog.FlatFieldViewDialog` | *Save as Flat-field Profile* and the profile viewer (Qt, imported on use) |

The reflectance scale is not part of this module: pass the profile to
`modules.radiometric.compute_flatfield_factors_array` with a scan of a reflectance target.

### Joints and the seam window

A row whose broadband response differs from what its neighbours predict by more than
`seam_threshold` is treated as a joint (dirt, gap, shadow) and interpolated. The prediction uses
`seam_window` rows (default 31), capped at a quarter of the measured rows (at least 3, see
`effective_seam_window`): on a short ROI a window reaching across the slit would read the falloff
towards its ends as joints. The window used is returned as `seam_window`.

## Requirements

* NumPy (core and store); PyQt5 and Matplotlib for the dialogs only.
* `modules.radiometric` (`__requires__ = ("radiometric",)`): the `ReferenceKey` a profile
  belongs to, and the reference dialog the flat-field dialog extends.

## Stand-alone use

The module has no widget of its own to launch (`__widgets__ = ()`); use it from Python:

```python
from modules.flatfield import FlatFieldStore, apply_flatfield, compute_flatfield

flat = compute_flatfield(strip_cube, strip_mask, seam_threshold=0.05)
evened = apply_flatfield(scan_cube, flat["profile"])
```

## Embedding in an application

```python
from modules.flatfield import FlatFieldStore

store = FlatFieldStore(camera_profile.flatfield_folder)
match = store.find(scan_key)            # FlatFieldMatch: .profile or .reason
if match.ok:
    profile = match.profile.profile()   # (lines, bands) float32
```

In the HSI application *Save as Flat-field Profile* (scan window) opens `FlatFieldDialog` on the
scan just finished; the Flat-field status card reports and shows the profile of the selected bin.

## Data formats / conventions

* Cubes are `(lines, samples, bands)` dark-subtracted counts; a profile is float32
  `(lines, bands)`, about 1, finite and positive.
* Store layout (one folder per profile, never modified after it is written):

  ```
  <folder>/flat_<YYYYMMDD>_<HHMMSS>[_<n>]/
      flatfield.json            format_version, id, created_at, acquired_at, key, ROI,
                                sensor_temperature_c, info
      profile.npy               (lines, bands) float32
      profile.wavelength.npy    (bands,) float64 band centres, nm
      roi_mask.npy              bool (lines, samples) ROI of the flat-field scan
  ```

* `flatfield.json` carries `format_version` (`FLATFIELD_FORMAT_VERSION`); other versions are
  refused with a clear message.
* A profile belongs to a camera setting (`ReferenceKey`: camera, bin, orientation, slit rows,
  bands, gain, corrections) and to the lamp and target geometry, which no file records: after
  moving a lamp or changing the working distance make a new profile.
