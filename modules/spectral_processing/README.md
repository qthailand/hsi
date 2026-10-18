# spectral_processing

## Overview

Qt-free spectral helpers for the HSI pipeline:

* `LambdaBinner` - averages the spectral pixels of a raw spectrogram into uniform wavelength bins (Numba, parallel, cached compilation).
* `wavelength_axis_from_frame_width(width, m, c)` - linear pixel -> nm axis.
* `spectrum_from_corrected_row(frame, y)` - one slit row of a corrected frame.
* `parse_envi_wavelengths(envi_image, cube)` - band wavelengths (nm) from ENVI metadata.

## Requirements

* NumPy and Numba (`binning.py`); OpenCV is pulled in through `modules.image`.
* Other modules: `image` (`__requires__ = ("image",)`; used lazily by `spectrum_from_corrected_row`).
* No PyQt5 - importing the package never loads Qt.

## Stand-alone use

There is no widget; the CLI bins a saved NPY spectrogram:

```
python -m modules.spectral_processing INPUT.npy OUTPUT.npz --m 0.2903 --c 399.791 [--step 2] [--threads 2]
```

`INPUT.npy` must be 3-D `(frames, spatial, spectral)`; `OUTPUT.npz` contains `frames` (uint16 `(frames, spatial, n_bins)`) and `wavelengths` (float32, bin centres in nm). `--step` is the bin width in nm (default 2), `--threads` the Numba thread count (default 2).

## Embedding in an application

```python
import numpy as np
from modules.spectral_processing import LambdaBinner, wavelength_axis_from_frame_width

raw = np.zeros((10, 64, 1024), dtype=np.uint16)          # (frames, spatial, spectral)
binner = LambdaBinner(m=0.2903, c=399.791, step=2.0, threads=2).fit_from(raw)
binner.warmup()                                          # optional: compile/load the Numba cache up front
binned = binner.transform(raw)                           # (10, 64, binner.n_bins) uint16
cube, wavelengths = binner.transform_to_cube(raw)        # (n_bins, 64, 10) view + nm axis
axis = wavelength_axis_from_frame_width(1024, 0.2903, 399.791)   # float32 nm
```

## Data formats / conventions

* Raw spectrogram: `(frames, spatial, spectral)`; single sensor frame / spectral image: `(slit, wavelength)`; cube: `(slit, scan, wavelength)`.
* Wavelength of spectral pixel `j` is `m * j + c` nm (`m` in nm/pixel, `c` in nm); outputs are float32 nm.
* Binning works in uint16: integer mean with rounding, clamped to 65535. Bin edges start at `c` and are `step` nm apart; empty bins are dropped (`n_bins` can be smaller than the nominal count).
* `transform` returns a new array; `transform_to_cube` returns a non-contiguous transposed view `(n_bins, spatial, frames)` (band-first, as used by the writers in `modules.cube_io`).
* `parse_envi_wavelengths` converts micrometres to nm only when the `wavelength units` entry says so, and returns `None` (with a log warning) on missing or mismatching data.

## API summary

| Name | Purpose |
|------|---------|
| `LambdaBinner(m, c, step=2.0, threads=None)` | binner; `fit(v)`, `fit_from(array)`, `transform`, `transform_to_cube`, `warmup`, `lambda_out`, `n_bins` |
| `wavelength_axis_from_frame_width(width, m, c)` | float32 nm axis |
| `spectrum_from_corrected_row(corrected_frame, selected_y)` | contiguous 1-D row (may be a view); `selected_y` is clamped |
| `parse_envi_wavelengths(envi_image, cube)` | float32 nm array or `None` |

## Threading

No Qt; functions may be called from any thread. A `LambdaBinner` must not be re-fitted while another thread transforms with it. `transform` calls Numba's process-wide `set_num_threads` when `threads > 0`, so binners with different thread counts running concurrently affect each other. Call `warmup()` outside the real-time path to avoid first-call compilation latency.

## Tests

Covered by `tests/test_package_regression.py` and `tests/test_stage2_regression.py`, and by `tests/test_module_contract.py` (contract). Run `python -m pytest -q tests`.
