# wavelength_calibration

## Overview

Linear pixel-to-wavelength calibration of a hyperspectral spectrometer:
`wavelength_nm = m * pixel + c`.

* **Qt-free core** (`core.py`): least-squares fit (`fit_wavelength`), row spectrum
  extraction with optional Savitzky-Golay smoothing (`row_spectrum`, `smooth_spectrum`),
  peak detection (`detect_peaks`) and sub-pixel refinement (`refine_peaks`),
  **Auto Calibration** (`auto_calibrate`), and the JSON profile
  (`WavelengthCalibration.save` / `.load`).
* **Qt widget** (`widget.py`): `WavelengthCalibrationWidget`, an embeddable panel that
  keeps the legacy interactive workflow (grab/load image, pick row, find peaks, type
  wavelengths, calculate, save/load profile, apply to the host).

### Savitzky-Golay smoothing controls

The widget smooths the selected row before peak picking:

* **Smooth spectrum (Savitzky-Golay)** check box (default on),
* **Smoothing window (pixels, odd)**, 3..201, default 7 (an even value is bumped to the next odd one),
* **Smoothing polynomial order**, 1..9, default 2.

In the core, `smooth_spectrum(spectrum, window=7, polyorder=2)` forces the window odd and
limits it to the spectrum length; the order is clamped to `window - 1`; if no window of at
least 3 remains the input is returned unchanged. Changing the smoothing only redraws the
spectrum: points already entered stay valid.

### Auto Calibration (anchor peaks + Hg-Ar catalog matching)

1. Press *Find Peak*, then type the true wavelength of **at least 3** peaks (the *anchors*).
2. Press *Auto Calibration* (button enabled only with an image and >= 3 complete rows).
3. `auto_calibrate` snaps each anchor to the detected peak within 2.5 px, fits a line
   through the anchors, predicts the wavelength of every other detected peak and accepts a
   peak when exactly one line of the catalog (`HGAR_LINES`, Hg-Ar lamp, nm in air) lies
   within the **Auto match tolerance** (nm, default 1.0) of the prediction (a second line
   within 1.5x of the best distance makes it ambiguous and rejects it). The fit is repeated
   with the accepted lines until the assignment stops changing (max 8 rounds). Operator
   anchors are never altered or dropped.
4. The table is refilled with anchors, auto-matched lines and unmatched peaks (blank
   wavelength); the status line reports how many lines were identified.

`LEGACY_HGAR` (the old positional "Mercury-Argon 400+" preset) is still available through
the *Load Mercury-Argon 400+ preset* button; it only works when the detected peaks line up
with the preset and every assignment must be verified.

### Hardware bin is read-only

The *Hardware bin of image (from camera)* box cannot be edited. It is set by
`set_frame(frame, camera_bin)`, `set_bin(value)` or, for images loaded from disk, by the
callable passed to `set_bin_provider()` (the file is assumed to have been taken at that
bin). A profile stores `camera_bin`; `base_m = m / camera_bin` is the unbinned slope that HSI uses.

## Requirements

* Python 3, numpy, scipy (`find_peaks`, `savgol_filter`).
* Widget only: PyQt5, pyqtgraph, OpenCV (`cv2`, image loading).
* `__requires__ = ("contracts", "image")`: the widget uses `modules.contracts.geometry.default_reference_row`
  so that its default row equals the Smile reference row, and `modules.image.save_png` for *Export PNG…*
  (the current image as an 8/16-bit PNG, values unscaled).
* Importing `modules.wavelength_calibration` does **not** import PyQt5 (the widget is loaded lazily).

## Stand-alone use

```
python -m modules.wavelength_calibration                 # opens the GUI (modules._support.launch)
python -m modules.wavelength_calibration --points pts.csv --output profile.json --bin 1
```

`pts.csv` has two columns `pixel,wavelength_nm`, no header. The CLI fits, writes the profile
JSON and prints `m`, `c`, R^2 and RMS (nm).

## Embedding in an application

```python
from PyQt5.QtWidgets import QApplication, QMainWindow
from modules.wavelength_calibration import WavelengthCalibrationWidget

app = QApplication.instance() or QApplication([])
widget = WavelengthCalibrationWidget()          # zero-argument constructor
widget.set_bin_provider(lambda: camera.hardware_bin)   # bin for images loaded from disk
widget.set_host_available(True)                 # enables Grab / Apply / Save as Default
widget.grabRequested.connect(request_snapshot)  # host: widget.begin_grab(id) ... widget.accept_snapshot(packet)
widget.applyRequested.connect(apply_profile)    # slot receives a WavelengthCalibration
widget.saveDefaultRequested.connect(save_default)
# host tells the widget the result: widget.calibration_applied(profile)
window = QMainWindow(); window.setCentralWidget(widget); window.show()
```

Host inputs (methods): `set_frame(frame, camera_bin)`, `set_bin`, `set_bin_provider`,
`set_host_available`, `begin_grab(token)`, `accept_snapshot(packet)` (`packet.request_id`,
`.image`, `.camera_bin`), `snapshot_failed(token, message)`, `calibration_applied(profile)`.
Outputs (signals): `grabRequested()`, `applyRequested(object)`, `saveDefaultRequested(object)`.
Inputs also include `set_calibration_source(store, setting_provider)` (shows the camera setting and
the stored profile) and `calibration_stored(profile, path, label)` (host confirms the save).
The widget never reaches into the main window.

Headless use:

```python
from modules.wavelength_calibration import fit_wavelength, row_spectrum, detect_peaks, auto_calibrate

spectrum = row_spectrum(frame, row=276, window=7, polyorder=2)      # frame: (slit rows, spectral columns)
peaks = detect_peaks(spectrum, height_fraction=0.05, distance=5)
cal = fit_wavelength([100, 400, 700], [500.0, 600.0, 700.0], camera_bin=1)
cal.save("profile.json")
```

## Data formats / file layouts

Images are 2-D `(slit rows, spectral columns)`; pixel positions are spectral-column indices
(px) at the image's own hardware bin; wavelengths are nm.

**Wavelength profile JSON** (`WavelengthCalibration.save`, written atomically via a `.tmp` file):

| key | meaning |
|-----|---------|
| `format_version` | `1` (readers reject anything else) |
| `units` | `"nm"` (readers reject anything else) |
| `stage` | `"post_dark_hot_pixel_orientation"` (informational) |
| `m`, `c` | slope (nm/px at `camera_bin`) and intercept (nm) |
| `r2`, `rms_nm` | fit quality (dimensionless, nm) |
| `pixels`, `wavelengths` | matched points (px, nm), same length >= 2, distinct pixels |
| `camera_bin` | hardware bin of the image (>= 1) |
| `frame_shape` | `[rows, columns]` or `[]` |
| `selected_row` | slit row of the spectrum |

`modules.contracts.normalize_wavelength_config` converts this layout to the "Standards" layout
used by Smile quality analysis. Load: `WavelengthCalibration.load(path)` (raises `ValueError`
on a wrong version/units or invalid content).

## API summary

* `WavelengthCalibration` (frozen dataclass; `base_m`, `validate()`, `save(path)`, `load(path)`).
* `fit_wavelength(pixels, wavelengths, camera_bin=1, frame_shape=(), selected_row=0)`.
* `row_spectrum(frame, row, window=7, polyorder=2, smooth=True)`, `smooth_spectrum`.
* `detect_peaks(spectrum, height_fraction=.05, distance=5)`, `refine_peaks(spectrum, peaks)`.
* `auto_calibrate(spectrum, anchor_pixels, anchor_wavelengths, catalog=HGAR_LINES, tolerance_nm=1.0, ...)`
  returning `AutoCalibrationResult(pixels, wavelengths, sources, unmatched, calibration)`.
* `LEGACY_HGAR`, `HGAR_LINES`.
* `WavelengthCalibrationWidget` (see above).

## Threading

The Qt-free functions are pure and thread-safe; `WavelengthCalibration` is immutable. The widget
must be used from the GUI thread and starts no threads of its own.

## Tests

`python -m pytest -q tests/test_wavelength_calibration.py tests/test_module_contract.py`

## Apply and the per-setting store

**6 · Profile → Apply & Save for this setting** applies the profile to the live HSI session and,
once the acquisition worker has accepted it, stores it for the image's camera setting (camera,
index, bin, orientation, frame size) in `config/calibration/wavelength/<backend>/wavelength_bin<N>.json`
(`modules.contracts.CalibrationStore`; the previous file is copied to `history/`). Smile Analysis
(Spectral Quality) and *FWHM / Spectral Resolution* load that file automatically for a frame of
the same setting. The scan uses it too: when streaming starts and after every bin change the
acquisition loads the file of the current bin (if camera, index, orientation and frame size
match; otherwise it keeps the current m/c and logs a warning). The *Image* group shows the
setting and what is stored for it. *Save as Default* additionally makes the profile the start-up
default (unchanged).

FWHM / spectral resolution moved to `modules.spectral_resolution` (pixel-integrated Gaussians,
neighbouring lines fitted together, smile-corrected row blocks).

## Export report (tab-separated)

**6 · Profile → Export Report (.txt, tab-separated)** writes everything on screen to one UTF-8
text file (opens in Excel/LibreOffice):

| Section | Columns |
|---|---|
| `[fit]` | key/value: `m_nm_per_px`, `c_nm`, `m_base_nm_per_px`, `r2`, `rms_nm`, `camera_bin`, `frame_shape`, `selected_row`, `n_points`, smoothing/peak settings, camera setting |
| `[points]` | `pixel`, `wavelength_nm`, `fit_nm`, `residual_nm`, `residual_px` |
| `[spectrum]` | `pixel`, `wavelength_nm`, `intensity_raw`, `intensity_smoothed` |

Each section starts with `# [name]`, then a header row; sections are separated by a blank line.
Sections without data (no fit) are omitted. Headless: `write_report(path, ...)`.

## Line lists and automatic calibration without entered wavelengths

**5 · Calibrate → Line list → Load…** reads reference wavelengths from a text file (e.g.
`OceanOptics_AR-2_wavelengths_nm.txt`): one value in nm per line; the first number of each line
is used, so extra columns are fine; blank lines, headers and lines starting with `#` or `;` are
ignored. The built-in default is the Hg-Ar list.

**Auto Calibration** then works in one of two ways:

* **3 or more wavelengths entered in the table** - those rows are anchors; the other peaks are
  matched to the line list within *Auto match tolerance* (nm) (unchanged behaviour).
* **fewer than 3** - full search (`search_calibration`): every pair of the strongest peaks with
  every pair of listed lines proposes `λ = m·pixel + c`; the proposal explaining the most peaks
  within *Search tolerance* (px) wins (ties: smallest residual), is refined by re-matching and
  re-fitting, and points far off the fit (a stray peak that happens to sit near a listed line)
  are dropped. Assumes wavelength increases with the pixel column (m > 0).

The table is refilled with the matched lines (unmatched peaks keep an empty wavelength) and the fit
is shown; check the residual plot. If the search reports no consistent match, the list probably
does not cover this lamp/spectral range, or the peak settings find too few or too many peaks.
