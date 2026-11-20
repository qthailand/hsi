# keystone_check

## Overview
Checks a cube **after** smile + keystone correction: every scene feature should sit on the same
slit row in every band. Scan a target with sharp edges across the slit (a checkerboard, a
printed page, the bar target), open the corrected cube here and press *Check*:

* the horizontal edges of a reference band (1300 nm by default) are found;
* in every third band each edge's row is measured (centroid of the squared row derivative
  within 4 rows, averaged over the scan columns where that edge is strong) - a symmetric blur
  does not move it, so defocus that changes with wavelength does not read as a shift;
* thin features (a printed text line: its two edges merge when blurred; a fine checkerboard at
  the blurred end of the spectrum) are left out of the statistics;
* the verdict is PASS when 95 % of the edge / band shifts are under 0.25 px (the keystone
  re-trace limit), with a table per 100 nm zone and the worst row.

The keystone margin rows (`keystone row margin` in the header) are left out unless the header
says they were cropped already (`keystone margin rows = cropped ...`).

## Requirements
numpy, scipy, matplotlib; PyQt5 for the widget only. `modules.cube_io` to open ENVI files.

## Stand-alone use
`python -m modules.keystone_check [cube.hdr]`

## Embedding in an application
```python
from modules.keystone_check import KeystoneCheckWidget, check_keystone
widget = KeystoneCheckWidget()
widget.set_source_provider(lambda: [(title, cube, wavelengths, metadata), ...])   # open cubes (optional)
widget.set_cube(cube, wavelengths, metadata, "scan 12")
widget.run_check()                                   # worker thread; result in widget.result / checked(object)
res = check_keystone(cube, wavelengths)              # Qt-free
print(res.verdict())
```

## Data formats / conventions
Cube `(slit, scan, wavelength)`, any dtype (a memmap is read band by band). Shifts in pixels
along the slit, positive = lower in the frame than in the reference band. `KeystoneCheck` holds
`rows`, `wavelengths`, `shifts (edges x bands)`, `thin`, `zones`, `median`, `p95`, `max`, `worst`.

## API summary
`check_keystone(cube, wavelengths, ref_nm=1300, band_step=3, half_window=4, row_margin=(0, 0),
zone_nm=100, progress=None) -> KeystoneCheck`, `KeystoneCheck.verdict()`, `.passed`, `PASS_PX`,
`KeystoneCheckWidget` (`set_cube`, `set_source_provider`, `run_check`, `wait`, signal `checked`).

## Threading
`check_keystone` is pure; the widget runs it on a `QThread` and waits for it on close.

## Tests
`tests/test_keystone_check.py`, `tests/test_module_contract.py`.
