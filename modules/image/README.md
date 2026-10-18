# image

## Overview

Qt-free helpers for raw sensor frames: extract a 2-D frame, flip the slit axis, subtract a dark frame (saturating) and repair hot pixels with the mean of their neighbours.

## Requirements

* NumPy and OpenCV (`cv2`).
* No other `modules.*` package (`__requires__ = ()`), no PyQt5.

## Stand-alone use

There is no widget and no `__main__.py`; the package is a function library used from Python:

```python
import numpy as np
from modules.image import subtract_dark_frame

frame = np.full((8, 16), 500, np.uint16)
dark = np.full((8, 16), 100, np.uint16)
corrected = subtract_dark_frame(frame, dark)
```

## Embedding in an application

```python
from modules.image import raw_2d_frame, apply_raw_orientation, subtract_dark_frame, correct_hot_pixels, save_png, png_name

img = raw_2d_frame(camera_frame)                       # (slit, wavelength), C-contiguous
img = subtract_dark_frame(img, dark_mean)              # saturates at 0; shape mismatch -> unchanged + warning
img = correct_hot_pixels(img, hot_pixel_map)           # non-zero map entries are replaced
img = apply_raw_orientation(img, raw_inverted=True)    # flips the slit axis
save_png(png_name("scan", camera_bin=2), img)          # 8/16-bit PNG, values unscaled
```

## Data formats / conventions

* Frame / spectral image: `(slit, wavelength)`, usually uint16; a `(slit, wavelength, C)` input uses channel 0.
* Mismatched shapes never raise: the original frame is returned unchanged and a warning is logged on the `"HSI"` logger. `None` inputs are passed through.
* Ownership: `subtract_dark_frame` and `correct_hot_pixels` return a new array when they change something and the *same* object when they do nothing. `raw_2d_frame` / `apply_raw_orientation` may return the input (a view) when it is already contiguous and not flipped.
* `subtract_dark_frame` of two equal-dtype arrays uses `cv2.subtract` (saturating for integers; floats are not clamped). Different dtypes are subtracted in int32 and clamped at 0.
* `correct_hot_pixels` uses the 8 in-bounds neighbours of each flagged pixel, taken from the uncorrected frame, rounded to the frame dtype.

## API summary

| Function | Purpose |
|----------|---------|
| `raw_2d_frame(frame)` | contiguous 2-D frame (channel 0 of 3-D input); `ValueError` for other ndim |
| `apply_raw_orientation(raw_frame, raw_inverted=True)` | optional flip of the slit axis |
| `subtract_dark_frame(frame, dark_frame)` | clamped dark subtraction |
| `correct_hot_pixels(frame, hot_pixel_map)` | neighbour-mean repair of flagged pixels |

## Threading

Pure functions without shared state; callable from any thread. Inputs are not modified.

## Tests

Exercised through `tests/test_package_regression.py`, `tests/test_stage2_regression.py` and `tests/test_preprocessing.py`; contract checks in `tests/test_module_contract.py`. Run `python -m pytest -q tests`.
