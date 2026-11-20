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

* **Keystone** (`keystone`): measures the along-slit distortion from a line-target frame and
  fits it jointly with the smile into ONE 2-D geometry map (`map_x` + `map_y`, one
  interpolation); see *Keystone (smile + keystone in one map)* below.

Reference row: the zero-displacement row defaults to `modules.contracts.default_reference_row`,
the same row used by wavelength calibration.

**Apply to camera (bin N)**: stores the analysed model for the frame's camera setting as
`camera_profiles/<backend>/<model>_<serial>/smile/smile_model_bin<N>.npz` (the open camera's
profile; the file the HSI runtime loads for that bin; the previous one is copied to `history/`;
models in the older `smile_maps/<backend>/` are still read when the profile has none) and emits `applyRequested(path)` so the host loads it into
the live session. It is refused when the camera has moved to another bin/size since the frame was
grabbed. The wavelength calibration for Spectral Quality is loaded automatically from the store
for the same setting; *Load wavelength calibration…* overrides it.

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
rt = SmileRuntime("camera_profiles/poa/Apollo-M_MAX_CAMP7161/smile", fallback_folder="smile_maps/poa")
rt.set_bin(1)                                     # smile_model_bin1.npz: own folder first, then the fallback
out = rt.apply(sensor_frame)                      # same shape and dtype; unchanged if no model
```

## Keystone (smile + keystone in one map)

Smile moves a wavelength along the spectral axis as the slit row changes; keystone moves a slit
position along the slit axis as the wavelength changes (the image of one ground point is not on
one row in every band). `modules.smile.keystone` measures both and writes them as one model.

**What to capture** (same camera setting as the frames the model will correct: orientation,
hardware bin, ROI; average several frames of each):

1. **Lamp** frame (Hg-Ar or similar): emission lines along the slit, as for the smile alone.
2. **Line target**: a target whose bars (dark on a bright background, or bright on dark:
   `polarity`) cross the slit, lit by halogen so every bar is seen at every wavelength. The bars
   must span at least half of the slit (better: all of it), and at least three must be followed
   over 25 % of the spectrum (`MIN_BAR_COVERAGE`). Use the same focus and working distance as for scanning: both
   change the keystone. The rotation of the target in its plane does not matter (each bar still
   crosses the slit at one point, which is all the fit uses).
3. **White** (optional, recommended): the same light, distance and exposure with the target
   removed (or a plain white part of it). `target_norm = (target - dark) / (white - dark)` is
   formed in raw sensor coordinates (both frames went through the same optics, so no map is
   needed). It divides out the halogen spectrum, vignetting, pixel response and, most important,
   any light gradient along the slit that changes with wavelength: such a gradient shifts the
   centre of a bar differently in each column and would otherwise look like keystone. Without a
   white frame only a slow trend along the slit is removed (wide running mean), and the result
   carries a warning.
4. **Dark** (optional): subtracted from the target and the white (the CLI also subtracts it
   from the lamp).

**The joint model** (destination / corrected pixels `X, Y`; maps are destination -> source):

    source_x = X + dx(X, Y)     dx(X, y_ref) = 0   the wavelength axis of the reference row is kept,
                                                    so the wavelength calibration stays valid
    source_y = Y + dy(X, Y)     dy(x_ref, Y) = 0   every band is co-registered onto the slit as seen
                                                    at the reference column

Both shifts are low-order 2-D polynomials whose basis enforces the anchors exactly. They are
fitted together (a lamp point belongs to the destination row `Y = y - dy`, a bar point to the
destination column `X = x - dx`) with robust per-point and per-feature weights: a bar or line
whose own shape biases its centre scatters more and counts less. The keystone degree along the
slit is chosen by leave-one-bar-out (0 to 4). Outside the measured span the shifts **hold the
edge value along the spectrum** (as the smile-only model does) and **continue linearly along the
slit** (value + slope at the edge, so a magnification term keeps scaling the rows instead of a
cubic blowing up).

**Usage**:

```python
from modules.smile import SmileCorrector
from modules.smile.keystone import keystone_report

c = SmileCorrector(deg_y=2, deg_x=1).fit(lamp)                 # smile analysis (lamp traces)
k = c.fit_keystone(target, white=white, dark=dark, polarity="dark")
print(keystone_report(k))
c.save("smile_model_bin2.npz", camera_setting)                  # ONE file, format 4
```

After `fit(lamp)` the smile and the keystone are fitted jointly; a new `fit()` drops a previous
keystone. In a worker thread, compute `analyze_keystone(c.analysis, target, ...)` there and hand the
result to `c.attach_keystone(k)`.

**Keystone onto a stored smile (fore-lens off / on).** The smile belongs to the spectrograph and
the detector (slit, grating, camera lens): taking the fore-lens off and on again does not change it,
as long as the camera body and the spectrograph do not move. The keystone must be measured with the
fore-lens on, at the scanning focus and working distance, because the lens's lateral colour is part
of it. So:

1. fore-lens **off**, lamp at the slit: measure the smile of **each hardware bin** and Apply it
   (`smile_model_bin<N>.npz`, the bin's default model);
2. fore-lens **on**: for each bin, take the line target (+ white) and fit the keystone onto the
   stored smile of that bin (no lamp frame needed), then Apply: one combined model replaces the
   bin's default (the previous file goes to `history/`).

```python
c = SmileCorrector.load("smile_model_bin2.npz")                 # the stored smile of this bin
k = c.fit_keystone(target, white=white, dark=dark, lamp=None)  # lamp=... only to CHECK the smile
c.save("smile_model_bin2.npz", camera_setting)                  # format 4; smile part unchanged
```

A stored smile is a function of the **raw** row (that is how the lamp lines were measured), so the
combined map reads it at the row the pixel comes from: `source_y = Y + dy(X, Y)`,
`source_x = X + S(source_y, X)` (`GeometryModel.dx_rows == "source"`). The keystone fit then does
not involve the smile fit at all (a bar point's destination column is `X = x - S(y, X)`), only dy
is fitted, and the stored smile is kept exactly, including `S(y_ref, X) = 0`. Fit numbers, the
camera setting and the smile's surface are carried over; the file gets a new creation time (so
flat-fields and references made with the old one are refused). Without a lamp frame the QC covers
the target bars only, and the result always carries a reminder that the smile was not re-measured:
pass `lamp=` (any lamp frame of that bin) now and then; a warning asks to measure the smile again
when the lamp lines are more than 0.25 px RMS from straight with the stored smile. A keystone can
also be re-fitted onto a combined model (its smile part is used).

```
python -m modules.smile.keystone --lamp L.png --target T1.png T2.png [--white W.png] [--dark D.png] \
       [--polarity dark|bright] [--deg-y 2] [--deg-x 1] [--dy-xdeg 3] [--interp cubic] --out model.npz
python -m modules.smile.keystone --smile-model smile_model_bin2.npz [--check-lamp L.png] \
       --target T1.png T2.png [--white W.png] --out smile_model_bin2_keystone.npz
```

Several files per option are averaged. The CLI prints the report and writes the model without a
camera setting (the HSI app then applies it as "setting unverified"); store it through the
widget or `SmileCorrector.save(path, camera_setting)` to bind it to the camera.

**File and runtime**: a model with keystone is written as **format 4** (`map_x`, `map_y`, and
`metadata` with a `keystone` entry), by either writer. Builds older than this one refuse format 4
instead of silently applying the smile alone; smile-only models are still written as format 2 / 3.
Rows whose source lies outside the frame are zero-filled, like the border columns: `row_margin`
`(top, bottom)` says how many (`apply_cropped` and `transform(crop=True)` remove them; the live
runtime keeps the full frame). The CPU remap applies `map_y` as stored (for an inverted camera the
slit flip is folded into it as `h - 1 - y`). The CuPy and OpenCV-CUDA preprocessing backends apply
it too, with OpenCV's 2-D 1/32 tables (4 x 4 taps for cubic, 2 x 2 for linear), so the live
pipeline stays on the GPU; see `modules/preprocessing/README.md`.

**QC** (`keystone_report`, `KeystoneAnalysis.qc`): fit and leave-one-line-out residuals of both
surfaces, a re-trace of the lamp lines and the target bars on the **corrected** frames (lines must
be vertical, bars horizontal; per line / bar and the median), and the per-feature weights. A
warning is raised when the median line or bar is more than 0.25 px RMS from straight after
correction, when no white frame was used, or when over a quarter of the columns lie outside the
lamp lines.

**Calibration order** (each step is measured on the output of the previous ones):

1. dark
2. smile per hardware bin (fore-lens off, lamp at the slit), then keystone per bin (fore-lens on,
   line target + white) onto that stored smile, joined into one map (this module); or both from a
   lamp + line target taken in one optical state
3. check the wavelength calibration again (Spectral Quality / `quality_cmd`)
4. flat-field (scan of a white strip, `modules/flatfield`)
5. radiometric reference (Spectralon, `modules/radiometric`)

Flat-fields and radiometric references are measured on geometry-corrected cubes; their key holds
the smile model file and its creation time, so those made with another model are refused, and
steps 4-5 must be redone whenever the geometry model changes.

## Data formats / file layouts

Images are 2-D `(slit rows, spectral columns)`; rows and columns are zero-based pixels; analysis
uses `float32`, the runtime keeps `uint8`/`uint16`/`float32`.

**Smile model `.npz`** (`save_calibration` compressed, format 2; `save_calibration_fast`
uncompressed, format 3 and ~3-5x faster to load; a smile + keystone model is format 4 from
either writer; `load_calibration` / `load_calibration_fast` read versions 1-4, reject newer ones
and reject a file whose metadata has `keystone` but no `map_y`; `allow_pickle=False`):

| format | written by | arrays |
|--------|------------|--------|
| 1 | older builds | `map_x`, `metadata` (linear interpolation assumed) |
| 2 | `save_calibration`, smile only | `map_x`, `metadata` |
| 3 | `save_calibration_fast`, smile only | `map_x`, `metadata` |
| 4 | either writer, smile + keystone | `map_x`, `map_y`, `metadata` (with `keystone`) |

| array | content |
|-------|---------|
| `map_x` | `float32 (H, W)`: source column for each output pixel (px) |
| `map_y` | format 4 only: `float32 (H, W)`, source row for each output pixel (px). Formats 1-3 do not store it (identity row index) |
| `metadata` | `uint8` array holding UTF-8 JSON (below) |

`metadata` JSON keys: `format_version`, `margin` `[left, right]` (px), `image_shape` `[H, W]`,
`deg_y`, `deg_x`, `surface_coef` (flat, y-major), `y_rng`, `x_rng`, `interp`, `backend`, `fit_rms`,
`cv_rms`, `rms_before`, `rms_after` (px), `n_lines_used`, `n_lines_total`, `created`, `notes`, `y_ref`,
`keystone` (format 4 only: `KeystoneAnalysis.to_json()`, i.e. the joint `model` (schema
`hsi-smile-keystone/1`: both surfaces with their measured boxes), `margin`, `row_margin`
`[top, bottom]`, `qc`, `summary`, `warnings`, `interp`, `polarity`, `created`, `inputs`,
`target_bars`; with keystone, `margin` and `map_x` are the joint model's, while `surface_coef` and
the fit numbers describe the smile-only lamp fit),
and `camera_setting` (`CameraSetting.to_dict()`: backend, index, bin, orientation, frame size) when
the model was saved with one (Apply to camera, `SmileCorrector.save(path, camera_setting)`; older
files have none). The model is only valid for frames of the same camera, orientation, hardware bin
and size as the calibration image: the HSI app refuses a model whose `camera_setting` differs and
applies one without it as "setting unverified". `SmileRuntime` looks for `smile_model_bin<N>.npz`
in its folder.

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
* Remap: `build_maps`, `margin_from_map`, `correct_smile`, `crop_margin` (`crop_geometry`, which
  also crops `row_margin` rows, is in `modules.smile.calibration`).
* Analysis: `SmileAnalysis`, `analyze_smile`, `verify_on_frame`.
* Calibration: `SmileCalibration` (`apply`, `apply_native`, `apply_cropped`, ...), `calibration_from_analysis`,
  `save_calibration`, `load_calibration` (+ `_fast` variants in `modules.smile.calibration`).
* sklearn-style: `SmileCorrector` (`fit`, `transform`, `fit_transform`, `inverse_transform`, `save`, `load`;
  keystone: `fit_keystone`, `attach_keystone`, `clear_keystone`, `has_keystone`, `keystone`, `row_margin`).
* Keystone (`modules.smile.keystone`): `analyze_keystone` -> `KeystoneAnalysis`, `keystone_report`,
  `trace_target_lines`, `normalize_target`, `fit_geometry`, `GeometryModel` / `GeometrySurface`
  (`maps`, `source`, `destination`), `row_margin_from_map`; `calibration_from_keystone` in
  `modules.smile.calibration`.
* Quality: `SpectralReference`, `QualityReport`, `analyze_quality`, `estimate_shift_uncertainty`, `correlation_shift`.
* Excel: `build_smile_tables`, `write_smile_excel` (in `modules.smile.excel_report`).
* Qt (lazy): `SmileCorrectionWidget`, `SmileMainWindow`, `PlotCanvas`, `launch_gui`, `HAVE_QT`.

## Threading

Analysis and remap functions are pure and can run in any thread. `SmileCalibration` caches a remap
engine: do not mutate it from several threads. `SmileRuntime` is single-threaded (use it from the
frame-loop thread). The widget must be used from the GUI thread; it runs analysis and quality
diagnostics in `QThread` workers and waits for them in `closeEvent`.

## Performance

`fit_surface_2d` solves each IRLS step through the normal equations (SVD least squares only when
ill-conditioned; coefficients agree to ~1e-14), which speeds up the leave-one-line-out
cross-validation and the 100-repeat bootstrap. `analyze_quality` fits and cross-correlates all
sampled rows of a line in one batch (see `QUALITY.md`, *Computation*). On a 1104 × 1608 frame: Smile
analysis ≈ 2.1 s, Spectral Quality ≈ 7 s (was 3.2 s and ≈ 14 s). Line tracing stays row by row
(each row's window follows the previous row; vectorising over lines with different window widths was
slower and could not keep the float32 sums bit-identical; a vectorised all-rows tracer was also
tried: 4-6x faster, but on weak, noisy lines its centroids land differently and moved the bin-2 map
by up to 0.1 px, so it was not adopted). Full-frame maps evaluate the surface separably
(`Surface2D.grid`: two small matrix products instead of one full-frame array per coefficient);
`SmileAnalysis.timings` holds the seconds per stage (detect, trace, fit, cross-validate, remap,
re-trace). On a 2048 × 2560 frame (desktop CPU): analysis ≈ 1.2 s, result plots ≈ 1.2 s.

The remap engine probes a GPU (OpenCL context, CUDA device count) only for a backend that may be
used: never for `"cpu"` (the live runtime), and for `"auto"` only above 8 Mpx, the only size at
which it would pick CUDA. On a Jetson the first OpenCL / CUDA call can build kernels for minutes.

**Slow computer?** `python -m modules.smile.profile_cmd LAMP.png [--gui]` runs the steps of
Analyze / Apply on a saved lamp frame and prints the seconds of each (analysis stages, model
build, save / load, runtime remap, the GPU probes by themselves, and with `--gui` the widget's
plots). Run it on the machine that is slow, with the app closed and again while it previews:
a big difference points at the live preview competing for the CPU / the Python interpreter.

## Tests

`python -m pytest -q modules/smile tests/test_smile_runtime.py tests/test_smile_wavelength_profile.py tests/test_module_contract.py`

Keystone: `python -m pytest -q tests/test_smile_keystone.py` (synthetic lamp and line-target
frames drawn from a known geometry: map recovery within 0.02 px with a white frame, the anchors,
an asymmetric bar, a merged pair of thin lines, a dim stripe, format 4 I/O, `SmileRuntime` and
the preprocessing CPU / GPU paths).
