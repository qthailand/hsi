# spectral_resolution

## Overview

Measures the FWHM (spectral resolution Δλ) and resolving power R = λ/Δλ of calibration-lamp
lines. In the HSI application it is the *FWHM / Spectral Resolution* window. The workflow is
**select, then calculate**:

1. **Grab from Scan** (or *Load Image*): one oriented, dark-corrected lamp frame.
2. The **wavelength profile** and **smile model** stored for the same camera setting
   (backend, camera index, bin, orientation, frame size) are loaded automatically from
   `modules.contracts.CalibrationStore`; *Load…* overrides either with a file. The frame is
   smile corrected.
3. A **preview spectrum** is drawn and the peaks to measure are **selected automatically**
   (see *Peak selection*). The user checks and edits the selection, picks the line
   **profile** (Gaussian / Lorentzian) and the **baseline** remover with its parameters.
4. **Calculate FWHM**: slit rows are averaged in blocks (default 16 rows; 1 without a smile
   model) and every selected peak is fitted in every block, in a worker thread with a
   progress bar; **Cancel** stops it within a fraction of a second (so does closing the
   panel or a new frame). Status line before the run, e.g. *"4 peaks × 35 row blocks · Lorentzian ·
   baseline arPLS. Each block gets its own fitted centres and widths."*
5. Table (median FWHM px / nm, R, valid blocks), plots (FWHM vs wavelength with 16–84 %
   spread, FWHM along the slit), the fitted components drawn over the preview spectrum, and
   **Export Excel**.
6. **Save FWHM for this setting** stores the median over the lines of their median FWHM (nm)
   in `resolution_bin<N>.json` for the frame's camera setting (refused if the camera has
   since changed setting, or without a wavelength profile). With
   `"resolution_source": "fwhm"` in config/processing.json the scan uses it as the bin width
   (`fwhm_step_factor` × FWHM); the host re-reads it once no scan is running.

Changing the selection, the profile, the baseline or any of its parameters (or the fit
window, rows per block, saturation, min SNR, a calibration, the frame) **clears the result**
— table, summary, plots, fit overlay and export — and asks for a new calculation. A result
that arrives from a calculation started before such a change is stopped and discarded. The
result on screen therefore always matches the current settings; a control that only re-emits
its unchanged value (Enter in a spin box) keeps the result.

### Peak selection

* **Preview spectrum**: with a smile model, the mean of *all* rows of the corrected frame;
  without one, the mean of the 16 rows centred on the reference row (the profile's row, else
  the shared default row) — averaging every row of an uncorrected frame would widen the
  curved lines. The panel says which rows were used. The plot shows the raw preview, the
  current baseline (dashed, updated live while the method or a parameter changes) and the
  baseline-corrected spectrum; the x axis is in nm (`m·px + c`) when a profile is loaded,
  else in pixels.
* **Usable columns**: with a smile model, its valid range (`margin`: every row maps from real
  pixels, interpolation reach included), intersected with the columns that hold data in
  *every* row. The zero-filled border is curved (its width changes from row to row), so a
  column that is zero in only some rows would pull the preview, the baselines and the fits
  down at the edges and could even be detected as a peak.
* **Automatic selection** on every new frame: the profile's calibration points (the lines
  identified in Wavelength Calibration) that lie inside the usable columns (2 px margin). Without a profile: `scipy.signal.find_peaks` on the
  baseline-corrected preview with prominence ≥ *Min prominence* × noise (noise = median
  absolute first difference / 0.6745; default 20) and distance ≥ *Min distance* (default
  3 px). *Auto detect* and *Use calibration lines* replace the selection at any time. A
  later *Load…* of a profile re-selects the calibration lines unless the user has edited the
  selection (then the peaks are kept and relabelled).
* **Click** on the spectrum (not on the legend, whose samples toggle their curve): within 2 px of a selected peak it selects that peak in the list;
  otherwise it adds a peak at the maximum of the corrected preview within ±3 px (sub-pixel
  vertex of a 3-point log-parabola). A position within 1.5 px of a selected peak is never
  added twice. Plot coordinates are mapped to pixels with `px = (λ − c) / m`.
* **Drag** a dashed peak line to move it: on release it snaps to the maximum near the new
  position, or is put back if that is another selected peak or outside the usable columns.
* **Selected peak centers** list: `P1 • 546.07 nm (px 412.3)` (`P1 • px 412.3` without a
  profile), numbered in ascending wavelength; multi-select, *Remove selected* (or Delete),
  *Clear selected peaks*. List, lines and table always agree.
* **Wavelength of a peak**: the profile's known wavelength when the peak is within 1.5 px of
  one of its calibration points, else `m·px + c`. Widths in nm use the profile's `m` at the
  frame's bin.

### Method

* **Model**: sum of *pixel-integrated* line profiles of one shape on a linear background,
  `b0 + b1·(x−x0) + Σ_k P(x; A_k, μ_k, w_k)`. A pixel integrates light over its width;
  fitting point samples overestimates narrow lines (+10 % at 1.5 px). `A` is the area; the
  reported amplitude is the peak height of the continuous profile.

  | profile | `P(x)` | FWHM | amplitude | default window |
  |---|---|---|---|---|
  | Gaussian | `A·[Φ((x+½−μ)/σ) − Φ((x−½−μ)/σ)]` | `2√(2 ln 2)·σ = 2.3548·σ` | `A/(σ√(2π))` | ±3 × FWHM |
  | Lorentzian | `(A/π)·[atan((x+½−μ)/γ) − atan((x−½−μ)/γ)]` | `2·γ` (γ = HWHM) | `A/(πγ)` | ±5 × FWHM |

  Pick the instrument's line shape: the wrong one biases the width (a Lorentzian fitted to a
  Gaussian line comes out ≈ 15–28 % narrower, a Gaussian fitted to a Lorentzian ≈ 9–19 %
  wider) and the flags do not reveal it. The Lorentzian window is wider because 10 % of its
  area lies beyond ±3 FWHM.
* **Start**: the given pixel moves to the highest *local maximum* within ±2 px, never past
  the midpoint to another listed line (on the flank of a stronger neighbour the highest
  sample belongs to the neighbour: that used to pin a weak line's centre on its ±2 px bound
  with a width 15–20 % too large). A guess up to 4 px off an isolated line still climbs to
  its peak. The first width comes from the half-maximum crossings, around the line's own
  peak when the segment's maximum is a neighbour's flank (that used to give 2 px and reject
  well resolved lines as `outside`).
* **Window**: `±max(4, window × FWHM)` per line (*Fit window* `auto` = the profile's default).
* **Neighbours**: lines whose windows overlap, and unlisted peaks inside a window, are fitted
  *together* (a neighbour 6 px away narrowed the old single-Gaussian result by 19 %).
  Lorentzian wings fall off only as 1/d²: when the lines fitted in the other windows bend a
  window's background by more than 0.1 × noise beyond a straight line, the window is fitted
  again with them subtracted (`FitWindow.wings`; a strong line 10–12 FWHM away biased a weak
  one by −4 to −6.5 %).
* **Baseline** (optional, default *None*): estimated on each block's averaged spectrum over
  that block's usable columns only (the columns holding data in all of its rows, within
  `valid_columns`; the border keeps a zero baseline) and subtracted before fitting; the fit
  keeps its own linear background on top. Saturation is always judged on the raw,
  unaveraged rows.
* **Rejected, not reported wrong**: `saturated` (any clipped sample in the line core of any row
  of the block), `low_snr` (< *Min SNR*), `fit_failed` (not converged, or the width or the
  centre ended on its bound), `outside` (the window does not reach
  where the profile has fallen to ≈ 4 % of its peak: 1.06 FWHM Gaussian, 2.33 FWHM
  Lorentzian), `border` (in the smile-invalid columns). `undersampled` (FWHM < 2 px) is kept
  but marked.
* **nm**: `FWHM_px × m`, with `m` of the wavelength profile at the frame's bin.
* Without a smile model each row is measured on its own (rows per block = 1): averaging
  curved lines would widen them. The status line says so, and the user's value comes back
  with the next frame that has a smile model.

### Baseline removers

`modules.spectral_resolution.baseline` (pure NumPy/SciPy; no extra dependency). Penalized
methods solve `(W + λ·D'D) z = W y` with a banded Cholesky (`scipy.linalg.solveh_banded`),
falling back to sparse LU (`scipy.sparse.linalg.spsolve`) for extremely stiff systems.

| method | label | reference | parameters (default; range) |
|---|---|---|---|
| `none` | None | – | – |
| `modpoly` | ModPoly | Lieber & Mahadevan-Jansen, *Appl. Spectrosc.* 57 (2003) 1363 | `degree` 2 (≥ 0), `max_iter` 100 refits (≥ 0), `tol` 1e-3, `use_original` False |
| `imodpoly` | IModPoly | Zhao et al., *Appl. Spectrosc.* 61 (2007) 1225 | `degree` 2, `max_iter` 100 fits (≥ 1), `tol` 1e-3 |
| `zhangfit` | ZhangFit (airPLS) | Zhang, Chen & Liang, *Analyst* 135 (2010) 1138 | `lam` 100 (> 0), `diff_order` 1 (porder, 1–3), `max_iter` 15 (≥ 1), `tol` 1e-3 |
| `asls` | AsLS | Eilers & Boelens, Leiden University report (2005) | `lam` 1e6, `p` 0.01 (0 < p < 1), `diff_order` 2, `max_iter` 50, `tol` 1e-3 |
| `arpls` | arPLS | Baek et al., *Analyst* 140 (2015) 250 | `lam` 1e5, `diff_order` 2, `max_iter` 50, `tol` 1e-3 |

* **ModPoly** fits a polynomial, clips the working spectrum to it (`y := min(y, fit)`) and
  repeats; `use_original=True` clips the original spectrum (as `BaselineRemoval.ModPoly`).
* **IModPoly** removes points above `fit + DEV` once, then clips to `fit + DEV`
  (DEV = standard deviation of the residual) until DEV stops changing.
* **ZhangFit / airPLS** reweights a Whittaker smoother exponentially by the negative
  residuals (end points keep their weight, as Zhang's reference).
* **AsLS** weights points above the baseline by `p`, below by `1 − p`. With the default
  `p = 0.01` the baseline climbs into wide Lorentzian wings near the spectrum ends
  (−0.8 % FWHM at 8 px, −2.5 % at 14 px); `p = 1e-3` with `λ = 1e7` keeps them within 0.1 %.
* **arPLS** weights by a logistic function of the residual relative to the mean and standard
  deviation of the negative residuals.

In the panel only the selected method's parameters are shown; `λ` (and `p`, `tol`) are
entered on a log scale (arrows step by ½ or 1 decade, any value such as `3e4` can be typed)
and shown with 3 significant digits; the value used and exported is exactly the one shown.
`λ` scales roughly with (feature width in samples)^(2·diff_order): too small and the baseline
follows the lines' wings (widths come out narrow), too large and it cannot follow the
background (no better than *None*). The fit's own linear background already absorbs a gently
curved background; a baseline helps where the background bends strongly within a fit window
(e.g. Lorentzian lines on a steep continuum: +5 % → < 1 % in the tests with arPLS `λ = 1e3`).
A small `λ` also follows the wings of *wide* lines: arPLS `λ = 1e3` narrows Lorentzian lines by
3 % at 8 px and 12 % at 14 px FWHM, ZhangFit `λ = 10` by 13 % at 14 px. For lines wider than
about 6 px use `λ ≥ 1e5` (arPLS, AsLS) or keep *None*: on smooth backgrounds *None* was as
accurate as any remover in every synthetic test, which is why it is the default.

### Defaults

Profile Gaussian, baseline None, fit window auto (3 × / 5 × FWHM), rows per block 16 (1
without a smile model), saturation auto (from the image's bit depth), Min SNR 10, Min
prominence 20 × noise, Min distance 3 px. With profile Gaussian and baseline None,
`measure_spectrum` / `measure_frame` return the results of the version before profiles and
baselines existed, bit for bit, except where that version was wrong: a start that moved onto a
stronger neighbour's flank, a first width taken around a neighbour (lines rejected as
`outside`), a centre ended on its bound reported as `ok`, and columns zero-filled in only some
rows of a frame. These occur only in crowded spectra or at a curved smile border: 42 of 44
recorded cases, among them every isolated-line, neighbour, saturation, border and frame case,
are unchanged. (The panel no longer lists calibration lines that fall in the smile border;
they used to appear as `border` rows without a width.)

## Requirements

numpy, scipy, pyqtgraph ≥ 0.11 (widget: `addLegend(brush=…)` and `LegendItem.clear()`),
pandas + openpyxl (Excel export). Uses `modules.smile`
(model loading, saturation estimate), `modules.contracts` and `modules.image` (*Export PNG…*: the frame
as grabbed/loaded, 8/16-bit, values unscaled). `__requires__ = ("contracts", "image", "smile")`.
Runs on Python 3.8+ with NumPy 1.x and SciPy ≥ 1.8 (no NumPy-2-only API, no `scipy.sparse`
`*_array` constructors).

## Stand-alone use

`python -m modules.spectral_resolution` — load an image, then the wavelength profile and smile
model with the *Load…* buttons.

```python
from modules.spectral_resolution import measure_frame, fit_spectrum
left, right = smile.margin                  # the smile model's valid columns
result = measure_frame(corrected_frame, profile.pixels, wavelengths=profile.wavelengths,
                       m=profile.m, c=profile.c, rows_per_block=16, saturation=65000,
                       profile="lorentzian", baseline="arpls", baseline_params={"lam": 1e5},
                       valid_columns=(left, corrected_frame.shape[1] - 1 - right))
result.summary()["fwhm_nm_median"]
result.settings["baseline_params"]          # every effective parameter, defaults included

fit = fit_spectrum(spectrum, [412.3], profile="gaussian", baseline="imodpoly")
window = fit.windows[0]                     # draw window.total(x) / window.component(k, x)
```

## Embedding in an application

```python
from modules.spectral_resolution import SpectralResolutionWidget
from modules.contracts import CalibrationStore
w = SpectralResolutionWidget()
w.set_calibration_source(CalibrationStore(wavelength_dir, smile_dir), lambda: current_setting)
w.set_grab_available(True)
w.grabRequested.connect(request_frame)          # answer with w.begin_grab(id) / w.accept_snapshot(packet)
w.analysisComplete.connect(lambda result: print(result.summary()))
w.resolutionStored.connect(lambda path: reload_bin_width())   # after Save FWHM for this setting
```

Scripting the panel: `w.set_frame(frame, camera_bin)`, `w.peaks`, `w.add_peak(px)`,
`w.click_at(x_view)`, `w.set_peaks(pixels)`, `w.select_peaks([i])`,
`w.remove_selected_peaks()`, `w.clear_peaks()`, `w.auto_detect_peaks()`,
`w.use_calibration_lines()`, `w.set_fit_profile("lorentzian")`,
`w.set_baseline("arpls", lam=1e5)`, `w.analyze()`, `w.cancel()`, `w.export_excel(path)`,
`w.store_resolution()` (path written, or `None` when refused).
`w.load_wavelength_file(path)` / `w.load_smile_file(path)` need a frame first.

## Data formats / conventions

Frames `(slit rows, spectral columns)`, pixels at the frame's hardware bin, nm. Excel sheets:
`Summary` (profile, baseline, overall statistics), `Lines` (`P1`… with profile), `Blocks`
(every block × line with flag and profile), `Settings` (parameters including the profile,
the baseline method and each effective parameter as `baseline_params.<name>`, camera
setting, calibration files used, preview rows and the selected peaks as
`selected_peaks.P1.pixel` / `.wavelength_nm` / `.source`). NaN is written as an empty cell.

## API summary

`measure_spectrum(spectrum, pixels, *, m, c, wavelengths, saturation, snr_min, window_fwhm,
valid_columns, find_neighbours, clipped, profile="gaussian", baseline="none",
baseline_params=None, cancel=None)`, `fit_spectrum(...)` → `SpectrumFit` (`measurements`,
`baseline`, `corrected`, `windows` of `FitWindow` with `wings`), `measure_frame(frame, pixels,
*, rows_per_block, ..., progress, settings, profile, baseline, baseline_params, valid_columns,
cancel)`, `MeasurementCancelled`, `LineMeasurement`, `LineSummary`, `ResolutionResult`
(`array`, `flags`, `summary`, `block_columns`), `PROFILES`, `DEFAULT_WINDOW_FWHM`,
`BASELINE_METHODS`, `BASELINE_LABELS`, `BASELINE_DEFAULTS`, `BASELINE_PARAM_HELP`,
`estimate_baseline`, `remove_baseline`, `baseline_settings`, `remove_spectrum_baseline`,
`row_blocks`, `usable_columns`, `excel_report.write_resolution_excel(path, result, info)`,
`SpectralResolutionWidget`.

## Threading

The core is pure; `cancel()` callables are polled between blocks and on every model
evaluation. The widget measures in its own `QThread` (and fits the preview spectrum once
afterwards for the overlay) and locks the selection and fit controls while it runs. *Cancel*,
a new frame, a change of the settings and `closeEvent` ask the thread to stop; `closeEvent`
then waits for it (a fraction of a second).

## Tests

`tests/test_spectral_resolution.py` (method, store, host contract),
`tests/test_spectral_resolution_profiles.py` (Lorentzian, Gaussian unchanged),
`tests/test_spectral_resolution_regressions.py` (start positions next to strong lines, width
estimate, Lorentzian wings, row-dependent smile border, cancel),
`tests/test_baseline.py` (baseline removers against pybaselines / BaselineRemoval),
`tests/test_spectral_resolution_widget.py` (peak selection, calculation with every
baseline, clearing, Excel, baseline plumbing on a curved background, cancel, legend clicks,
rows per block with/without smile model, log spin boxes, smile-model valid columns, Save FWHM),
`tests/test_fwhm_binning.py` (FWHM store, bin width from the FWHM with config fallback).
