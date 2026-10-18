# reference_roi

## Overview

Selects the spatial rows (along the slit) occupied by a stationary white reference.
`ReferenceROIWidget` shows a preview scan as an RGB image (spatial row on y, scan frame
on x) and lets the user drag a horizontal band; `ReferenceROIStore` persists the result
as an immutable profile plus an atomically updated `active.json`. The widget owns no
hardware: it emits requests and the host application performs the scan.

## Requirements

* numpy; PyQt5 only for `ReferenceROIWidget` (loaded lazily, so
  `import modules.reference_roi` is Qt-free).
* No other `modules.*` package (`__requires__ = ()`).

## Stand-alone use

```
python -m modules.reference_roi
```

opens the widget without hardware (useful for layout checks; scanning needs a host
that answers `previewRequested`).

## Embedding in an application

```python
from modules.reference_roi import ReferenceROIWidget, ReferenceROIStore

store = ReferenceROIStore(project_root)
widget = ReferenceROIWidget()
widget.previewRequested.connect(start_preview_scan)       # host scans, then:
#   widget.accept_preview(rgb_uint8_image, context_dict)   # or widget.fail_preview(msg)
widget.cancelRequested.connect(stop_preview_scan)
widget.saveRequested.connect(store.save_active)            # receives a ReferenceROIProfile
widget.deactivateRequested.connect(store.clear_active)
active = store.load_active()                               # None if nothing is active
```

`start_preview_scan` can build the preview with `select_rgb(frame, wavelengths)` per
scan frame and `compose_preview(columns)` at the end.

## Data formats / conventions

* Preview image: `uint8` RGB `(slit, scan, 3)`; display scaling never changes the row
  mapping. `select_rgb` returns `float32` `(slit, 3)` using the bands nearest
  645/555/465 nm; `compose_preview` scales per channel between the 1st and 99th
  percentiles.
* ROI: half-open rows `[y_start, y_end)`, `0 <= y_start < y_end <= spatial_height`,
  post-orientation spatial rows (`coordinate_space = 'post_orientation_spatial_rows'`).
* `compatibility` dict keys: `camera_backend`, `camera_id`, `camera_index`,
  `camera_bin`, `raw_inverted`, `spatial_height`. `ReferenceROIProfile.matches(ctx)`
  compares them; changes to wavelength bin count do not invalidate spatial rows.
* Files in `<project>/calibration/reference_roi/`: `<profile_id>.json` (never
  rewritten, `format_version` 1) and `active.json`
  (`{"format_version": 1, "active_profile": "<file>.json" | null}`). Unknown versions
  are rejected with `ValueError`. Deactivating keeps profile files.

## API summary

* `ReferenceROIProfile(name, y_start, y_end, compatibility)`: `validate()`,
  `matches(context)`, `to_dict()`, `from_dict(data)`.
* `ReferenceROIStore(project_root)`: `load_active()`, `save_active(profile)`,
  `clear_active()`.
* `select_rgb(frame, wavelengths, targets=(645., 555., 465.))`,
  `compose_preview(columns)`.
* `ReferenceROIWidget`: slots `accept_preview`, `fail_preview`, `set_busy`; signals
  `previewRequested`, `cancelRequested`, `saveRequested(object)`,
  `deactivateRequested`.

## Threading

Widget: GUI thread only, no threads of its own. Core functions and the store are plain
synchronous code with no shared state; the store does blocking file I/O.

## Tests

Run `python -m pytest -q tests -k "reference or roi"` and `tests/test_module_contract.py`.

## Notes from the original integration

Open **Modules → Reference ROI Selection**. Set stage travel/speed in Scan Window,
then press **Preview Scan**. The existing camera/stage controller performs the scan;
this widget does not own hardware. Select a horizontal band through the centre of
the stationary white reference by dragging vertically. Fine tune Y start/end if
needed, then **Apply & Save Profile**. End is exclusive.

The visual image has spatial rows on y and scan frames on x. RGB uses the nearest
645/555/465 nm bands; percentile scaling is display-only. All spatial rows are
preserved, with no crop or vertical resize in stored coordinates. Stop cancels
this preview; use existing stage controls to reposition before another scan.
Preview is capped at 2048 RGB columns and automatically stops at that limit.

Profiles live in `calibration/reference_roi/`; `active.json` selects one immutable
profile file. Files are atomically published. Restarting loads the active profile.
Deactivate clears the selection while retaining saved profile files. Matching
camera backend/id/index, sensor bin, orientation and spatial height is required
before the yellow GUI overlay is shown on the live spectrogram. Changes to
wavelength bin count do not invalidate spatial coordinates.

The overlay does not alter acquired pixels. This module stores reference ROI
selection only; reflectance correction is not applied by this change. Normal
scan recording remains on the existing path. Preview recording extracts H×3
columns instead of storing a full hyperspectral cube, and skips spectrogram and
saturation accumulation. RGB composition occurs once after stage completion.

Validation uses synthetic images and fake stage/controller tests; real camera,
stage operation and 70–123 fps performance require validation on the target PC.
