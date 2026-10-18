# smile

## Overview

Smile (spectral-line curvature) analysis and correction for an imaging spectrometer. A
spectral line that should fall on one column drifts with the slit row; this module measures
that drift from a calibration-lamp frame (e.g. Hg-Ar) and removes it by remapping columns.

* **Analysis** (`core`, `surface`, `analysis`): detect emission lines, trace them to sub-pixel
  accuracy, fit a robust 2-D polynomial displacement surface (`Surface2D`), cross-validate
  leave-one-line-out (`SmileAnalysis`, `analyze_smile`).
* **Correction** (`calibration`, `corrector`, `runtime`): `SmileCorrector` (fit / transform /
  inverse_transform), the persistable `SmileCalibration` (remap maps, margin, `.npz` I/O) and
  `SmileRuntime` (per-bin model for the live HSI frame loop, dtype-preserving CPU remap).
* **Quality** (`quality`, `quality_cmd`, `quality_plot`, see `QUALITY.md`): wavelength error,
  FWHM, line straightness and bootstrap uncertainty before/after correction; independent of Qt.
* **Widget** (`widget`): `SmileCorrectionWidget` (load/grab image, analyse, 4 result tabs plus
  quality tabs, save/load model) and `SmileMainWindow`.

Reference row: the zero-displacement row defaults to `modules.contracts.default_reference_row`,
the same row used by wavelength calibration.

**Apply to camera (bin N)**: stores the analysed model for the frame's camera setting as
`smile_maps/<backend>/smile_model_bin<N>.npz` (the file the HSI runtime loads for that bin; the
previous one is copied to `history/`) and emits `applyRequested(path)` so the host loads it into
the live session. It is refused when the camera has moved to another bin/size since the frame was
grabbed. The wavelength calibration for Spectral Quality is loaded automatically from the store
for the same setting; *โหลด Wavelength Calibration…* overrides it.

## Requirements

* numpy, scipy, OpenCV (`opencv-python`); see `requirements.txt`.
* Widget only: PyQt5 and matplotlib (loaded lazily; `import modules.smile` never imports Qt).
* Excel export only: pandas and openpyxl (imported when `write_smile_excel` runs).
* `modules.image` (`save_png`) for the widget's *Export PNG…*: the image as received, 8/16-bit, values unscaled.
* `__requires__ = ("contracts",)`.

## Stand-alone use

```
python -m modules.smile [image]            # GUI (also: python -m modules.smile.smile)
python -m modules.smile.quality_cmd IMAGE --wavelength-cal cal.json [--cal model.npz] [--plots DIR] [--output q.json]
python -m modules.smile.test_cmd           # benchmarks
```

## Embedding in an application

```python
from PyQt5.QtWidgets import QMainWindow
from modules.smile import SmileCorrectionWidget

widget = SmileCorrectionWidget()                 # zero-argument constructor
widget.analysisComplete.connect(on_analysis)     # SmileAnalysis
widget.statusChanged.connect(statusbar.showMessage)
widget.grabRequested.connect(request_snapshot)   # optional capture from the camera
widget.set_grab_available(True)
#   host: widget.begin_grab(id); later widget.accept_snapshot(packet) / widget.snapshot_failed(id, msg)
widget.set_calibration_source(store, current_setting)   # modules.contracts.CalibrationStore + () -> CameraSetting
widget.applyRequested.connect(load_into_session) # path of the model stored for the frame's setting
widget.set_frame(frame, setting=setting)         # (slit rows, spectral columns) array
widget.analyze()                                 # asynchronous (QThread); wait for analysisComplete
window = QMainWindow(); window.setCentralWidget(widget); window.show()
```

Headless:

```python
from modules.smile import SmileCorrector, save_calibration, load_calibration, SmileCalibration

corrector = SmileCorrector(deg_y=2, deg_x=1, interp="cubic").fit(hgar_frame)
corrector.save("smile_model_bin1.npz")
corrected = corrector.transform(raw_frame, crop=True)

# live pipeline (HSI): one model per hardware bin, dtype preserved, no crop
from modules.smile.runtime import SmileRuntime
rt = SmileRuntime("smile_maps"); rt.set_bin(1)   # loads smile_maps/smile_model_bin1.npz if present
out = rt.apply(sensor_frame)                      # same shape and dtype; unchanged if no model
```

## Data formats / file layouts

Images are 2-D `(slit rows, spectral columns)`; rows and columns are zero-based pixels; analysis
uses `float32`, the runtime keeps `uint8`/`uint16`/`float32`.

**Smile model `.npz`** (`save_calibration` compressed, format 2; `save_calibration_fast`
uncompressed, format 3 and ~3-5x faster to load; `load_calibration` reads versions 1-3 and rejects
newer ones; `allow_pickle=False`):

| array | content |
|-------|---------|
| `map_x` | `float32 (H, W)`: source column for each output pixel (px). `map_y` is the identity row index and is not stored |
| `metadata` | `uint8` array holding UTF-8 JSON (below) |

`metadata` JSON keys: `format_version`, `margin` `[left, right]` (px), `image_shape` `[H, W]`,
`deg_y`, `deg_x`, `surface_coef` (flat, y-major), `y_rng`, `x_rng`, `interp`, `backend`, `fit_rms`,
`cv_rms`, `rms_before`, `rms_after` (px), `n_lines_used`, `n_lines_total`, `created`, `notes`, `y_ref`.
The model is only valid for frames of the same orientation, hardware bin and size as the
calibration image. `SmileRuntime` looks for `smile_model_bin<N>.npz` in its folder.

**Wavelength JSON for quality analysis**: either layout accepted by
`modules.contracts.normalize_wavelength_config` (profile from `modules.wavelength_calibration`, or
the "Standards" layout with `m_nm_per_px`, `c_nm`, `standards`, `reference_row_y`). If
`reference_row_y` differs from the Smile reference row, the widget refuses to run the quality analysis.

**Quality report JSON** (`QualityReport.save`): strict JSON (NaN becomes `null`) with `rows`,
`wavelengths_nm`, `before`, `after`, `summary`, `metadata`.

**Excel report `.xlsx`** (`modules.smile.excel_report.write_smile_excel`, widget button
"Export Excel"): sheets `Summary`, `Lines`, `Traces` (column per line, before/after, one row per
slit row), `Measurements` (fit points, fitted value, residual), `Surface` (coefficients `a_ij` with
powers of y and x, stderr), and after a quality run `Quality summary`, `Quality by line`,
`Quality rows` (per row and reference line, `<metric>_before`/`<metric>_after`), `Uncertainty`;
plus `Metadata`. NaN is written as an empty cell; units are px, nm and counts.

## API summary

* Core: `load_frame`, `save_frame`, `robust_sigma`, `auto_sat_level`, `estimate_noise`, `Line`,
  `detect_lines`, `trace_line`, `straightness_rms`.
* Fit: `Surface2D`, `fit_surface_2d`, `cross_validate_lines`.
* Remap: `build_maps`, `margin_from_map`, `correct_smile`, `crop_margin`.
* Analysis: `SmileAnalysis`, `analyze_smile`, `verify_on_frame`.
* Calibration: `SmileCalibration` (`apply`, `apply_native`, `apply_cropped`, ...), `calibration_from_analysis`,
  `save_calibration`, `load_calibration` (+ `_fast` variants in `modules.smile.calibration`).
* sklearn-style: `SmileCorrector` (`fit`, `transform`, `fit_transform`, `inverse_transform`, `save`, `load`).
* Quality: `SpectralReference`, `QualityReport`, `analyze_quality`, `estimate_shift_uncertainty`, `correlation_shift`.
* Excel: `build_smile_tables`, `write_smile_excel` (in `modules.smile.excel_report`).
* Qt (lazy): `SmileCorrectionWidget`, `SmileMainWindow`, `PlotCanvas`, `launch_gui`, `HAVE_QT`.

## Threading

Analysis and remap functions are pure and can run in any thread. `SmileCalibration` caches a remap
engine: do not mutate it from several threads. `SmileRuntime` is single-threaded (use it from the
frame-loop thread). The widget must be used from the GUI thread; it runs analysis and quality
diagnostics in `QThread` workers and waits for them in `closeEvent`.

## Tests

`python -m pytest -q modules/smile tests/test_smile_runtime.py tests/test_smile_wavelength_profile.py tests/test_module_contract.py`
