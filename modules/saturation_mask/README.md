# saturation_mask

## Overview

Finds raw pixels that clipped (or came near full scale) **before** dark subtraction and follows
them through the same stages the intensities go through, so a recorded band that averaged in a
clipped pixel is flagged even when its value stays below full scale.

Two levels:

* **Basic detection** (`count.py`, every acquired frame): `count_saturated` / `RawCounter` count
  clipped and near-saturated raw pixels in one pass (Numba, rows in parallel; no mask, no
  temporary, the frame not modified). Thread count by frame size (`THREADS_BY_PIXELS`);
  `numba.set_num_threads` is thread-local and restored after each call, so the lambda binning's
  setting is untouched. `near` uses `ceil(0.98 * level)`: the same pixels as `detect_raw`.
* **Full mask** (`core.py`, optional): the stages below; `detect_raw(..., counts=)` and
  `SaturationPropagator.frame(..., counts=)` take the basic counts as their fast path.

| Stage | Intensity pipeline | Mask (this module) |
|---|---|---|
| Raw detection | – | `clipped = raw >= level`, `near = raw >= near_ratio * level` (clipped included) |
| Hot-pixel repair | mean of the valid 3×3 neighbours (`_hot_ys/_hot_xs/_hot_valid`) | OR of the same valid neighbours (none valid: False); off: unchanged |
| Orientation + smile/keystone remap | `cv2.remap` with the fixed-point maps of `prepare_maps` (flip folded in); `'nearest'`: the model's `apply_native` | OR over the taps whose OpenCV 1/32 weight is exactly non-zero (`tap_support`: nearest 1, linear 2, cubic 4, Lanczos4 8 taps); `'nearest'`: the sources read through `apply_native` |
| Keystone rows | `crop_keystone_rows` | `crop_rows(mask, keystone_row_margin(...))` |
| Lambda binning | rounded mean over `iL:iR` | any over `iL:iR` (the binner's own boundaries, overlaps included; subsets give their own bands) |

Policy: **any contributing tap**. The smile border is left out by the binner subset, as for the
intensities; it is not cropped again here.

### Fields

| Field | Meaning |
|---|---|
| `clipped_mask` | `bool (rows, bands)`: a clipped raw pixel contributes to the band |
| `near_saturation_mask` | the same for near saturation (clipped included; near-only = `near & ~clipped`) |
| `affected_fraction` | per row and band, share of the band's `w` corrected pixels affected by clipping |
| `near_affected_fraction` | the same for near saturation |
| `raw_clipped_fraction` | share of the raw frame's pixels that clipped (frame level) |
| `raw_near_fraction` | share near saturation |

A frame with nothing near saturation stops after the raw comparison (`sat_detect`).

### Recording (`record.py`)

| Item | Meaning |
|---|---|
| `LevelEvidence` | the level to check against: hosts build it from a **verified saturation profile** (`modules.saturation_profile`, `source='measured'`, `provenance`, `profile_id`, `profile_status`); otherwise `level=None` with the reason (`missing`, `inconclusive`, `incompatible`, `gain_outside`, or `'error'`: `transient`, the lookup failed, ask again later). The nominal full scale of the bit depth is never evidence |
| `SaturationRecord` | one frame, compact: values only at the near-saturated cells (sparse `uint32` indices, or a packed mask when that is smaller), clipped flags as packed bits over them, affected pixel counts (`fraction × w`, exact) |
| `scan_status(records)` | `checked` (every frame has a record), `partial`, `not_checked` (`None` = frame not checked) |
| `scan_summary(records)` | status, frames, checked, near / clipped frames, bytes |

## Requirements

* NumPy, OpenCV.
* Other modules: `preprocessing` (`prepare_maps`). The smile model, dark reference and binner are
  duck-typed (`modules.smile`, `modules.dark`, `modules.spectral_processing`).
* No PyQt5.

## Stand-alone use

No widget and no `__main__.py`; use it from Python.

## Embedding in an application

```python
from modules.saturation_mask import SaturationPropagator

saturation = SaturationPropagator()          # one per processing thread
# per frame, with the same objects the intensity pipeline used for it:
result = saturation.frame(raw, backend.saturation_level(), dark_reference, dark_enabled,
                          calibration, inverted, binner, row_crop=keystone_row_margin(calibration, h),
                          profiler=profiler)
result.clipped_mask            # (rows, bands), the recorded line's layout
```

## Status

Phase 2B: basic detection runs on every acquired frame and feeds the Scan window's Clipped tile.
Phase 2: the full mask is run by `modules.acquisition` (`SaturationTracker`) on every frame when
*Track Saturation* is on, and kept by the recorder with each frame. Not yet written to files (sidecar: Phase 3) or used
by the viewers (Phase 4).
