# Spectral quality analysis

The quality layer has no Qt dependency. It measures an uncropped before/after
pair on the same zero-based detector grid. Keystone is outside this release.

## Python API

```python
from smile import SpectralReference, analyze_quality
from smile.quality_plot import export_quality_plots

references = [SpectralReference(546.08, 489.416667, half_window_px=32)]
report = analyze_quality(
    raw_frame, corrected_frame, references,
    m_nm_per_px=0.2920349837184692, c_nm=402.8940165100514,
    resolution_nm=5.0, row_step=8, y_ref=reference_row,
    saturation=65535, independent_frame=False,
)
report.save("quality.json")
export_quality_plots(report, "quality_plots")
```

Use coefficients belonging to the same detector grid and reference row as the
remap model. Cropping changes column coordinates and is intentionally rejected
when the input shapes differ. Do not pass a cropped pair without explicitly
adjusting coefficients and reference columns. The default reference row follows
the correction module's center-row rule; no fixed detector height is assumed.

After `analysisComplete`, an embedded widget exposes the same diagnostics:

```python
report = widget.analyze_quality(
    references, m_nm_per_px=m, c_nm=c, row_step=8, resolution_nm=5.0,
)
```

This call is synchronous. Full-height fitting should run in the host's worker
thread; consume/export the returned report outside the acquisition loop.

## CLI

Run from the directory containing the `smile` package:

```bash
python -m smile.quality_cmd image.png --wavelength-cal wavelength_calibration_6line.json --interp linear --row-step 8 --bootstrap 100 --output quality.json --plots quality_plots
```

Independent validation uses a different exposure and a saved model:

```bash
python -m smile.quality_cmd validation.png --cal model_y552.npz --wavelength-cal wavelength_calibration_6line.json --independent-frame --output validation_quality.json
```

`--independent-frame` records the caller's provenance declaration. It cannot
prove that the two exposures differ. A saved remap has no raw fit measurements,
so bootstrap is omitted in saved-model mode. Gaussian peak fits and correlation
are diagnostic work; their runtime is separate from production remapping.

## Measurements and interpretation

- Wavelength error: fitted peak wavelength minus known lamp wavelength (nm).
  Includes bias, RMSE, P95 absolute error and maximum absolute error. Paired
  summaries use only rows with valid fits both before and after.
- FWHM: Gaussian instrumental line width in nm. It is not wavelength sampling
  interval. Also inspect fit residual RMS for departures from a Gaussian shape.
- Intensity: fitted peak height and the numerical area inside a fixed detector
  window after subtracting the fitted linear background. Area units are counts
  times pixels. These are not calibrated radiance or a flux-conservation proof.
  Window contamination, background errors and nonuniform remap scale can change
  the area; inspect individual line/row results before interpreting a summary.
- Cross-correlation: relative subpixel shift against the reference-row profile.
  This independently checks localization methodology, not absolute wavelength
  identity. It uses a separate algorithm but the same input exposure.
- Missing measurements: weak, saturated, boundary, truncated, or poorly fitted
  lines are NaN internally and null in strict JSON. Valid fractions are reported
  for each line. No pass/fail tolerance is imposed automatically.
- Blends: known reference neighbors less than `resolution_nm` apart are excluded
  from single-line diagnostics. Thus 576.96/579.07 nm are excluded at 5 nm.
  Unlisted interfering lines must be marked manually with `blended=True`.
  A *brighter* line inside the ±`half_window_px` window but more than
  `resolution_nm` away is not excluded, and the fit starts at the window's
  maximum: with the GUI's 32 px window, 800.6 nm next to a stronger 794.8 nm
  (5.8 nm ≈ 20 px at 0.297 nm/px) is measured on 794.8 nm and shows a
  wavelength error of about -6 nm. Mark such lines blended or narrow the window.

## Computation

`analyze_quality` fits every sampled row of one reference line in one call
(`_peaks`; the fit window is the same for all rows):

- The pre-checks of `_peak` (saturation, 4-point baseline, MAD noise, SNR,
  peak position) run on the whole `(rows, window)` block.
- Rows whose window holds another peak (prominence above `snr_min` × noise,
  at least 3 px from the start) are fitted by `_peak`
  (`scipy.optimize.least_squares`) exactly as before: with several lines in the
  window the fit has several minima, and another optimiser path from the same
  start may end in a different one.
- The other rows are fitted together by a batched, bounded Levenberg-Marquardt
  with the same model, analytic Jacobian, start values and bounds. A row is
  kept only when that fit converged with full steps, stays clear of every bound,
  has sigma >= 0.55 px (near the 0.35 px bound SciPy's trust region crawls and
  can run out of `max_nfev`) and passes `_peak`'s acceptance tests; every other
  row, i.e. every rejection, is decided by `_peak` itself.
- Result: the accept/reject decision is SciPy's in every row (0 differences in
  about 30 000 synthetic rows: isolated, weak, narrow, wide, edge, two-line,
  pure-noise windows and full Hg-Ar frames); kept values agree within SciPy's
  own tolerance (centres to 2e-4 px, FWHM to 1e-4 relative) and the batched
  cost was never higher than SciPy's.
- Cross-correlation is unchanged (`correlation_shift` per row).
- Bootstrap and Smile fit: each IRLS step of `fit_surface_2d` solves the normal
  equations with one step of iterative refinement (SVD least squares when
  ill-conditioned); coefficients agree with the SVD solution to ~1e-13 and the
  4σ rejections are unchanged.

Timing in the widget on a 1104 × 1608 synthetic Hg-Ar frame (27 lines, 15
references, every 8th row): Smile analysis 3.5 → 2.7 s, *Spectral Quality*
13.0 → 10.1 s (bootstrap 7.2 → 3.5 s). The batched fit helps little on a dense
lamp spectrum: with the 32 px window about 78 % of the rows have a neighbour
line in the window and keep the SciPy fit (≈ 6.8 s); with isolated lines the
fits are 7-15× faster. The rest is the per-row cross-correlation (≈ 1.4 s) and
the 100 bootstrap surface fits.

## Conditional uncertainty

```python
from smile import estimate_shift_uncertainty

uncertainty = estimate_shift_uncertainty(
    analysis, rows=query_rows[:, None], columns=query_columns[None, :],
    repeats=100, seed=0,
)
```

The bootstrap resamples whole slit rows, refits the polynomial displacement,
and reanchors it at `analysis.y_ref`. Outputs include standard deviation,
percentile 95% intervals, successful repeats and a support mask. The mask is
measured-coordinate bounding-box coverage, not a guarantee of local sampling
or predictive accuracy. Clamped positions outside those bounds are unsupported.
Intervals are conditional on the chosen model and line measurements; they
exclude systematic errors, wavelength coefficient uncertainty, reference-line
identity errors and correlation between neighboring slit rows. Twenty repeats
are useful for smoke tests; increase repeats for a stable uncertainty report.

## Dependencies

Processing: NumPy >= 2.0, SciPy >= 1.10, OpenCV >= 4.8.
PNG exports additionally require Matplotlib. Qt is optional.

## Method references

- Bakker et al. (2019), DOI 10.1109/WHISPERS.2019.8921045.
- Leung et al. (2022), DOI 10.1117/12.2630442; arXiv:2208.10610.
- Lenhard and Baumgartner (2010), Estimation of radiometric uncertainty after
  smile correction, https://elib.dlr.de/64147/.

These diagnostics apply selected principles from the literature; they are not
an exact reproduction of those papers' complete calibration methods.

The supplied six-line coefficients were fitted with parabolic local peak
interpolation. This validator uses Gaussian line fitting. Their peak-center
definitions can differ for asymmetric or blended lines; do not interpret every
residual offset as a remap error. The existing coefficients are retained.

## GUI tabs

The widget displays eight tabs. The original four retain their roles; the
additional tabs are Wavelength, Line Quality, Uncertainty, and Cross-correlation.

1. Load a reference image and run Smile analysis.
2. Click **Load Wavelength Calibration** and select a JSON containing
   `m_nm_per_px`, `c_nm`, and `standards` (the supplied six-line JSON is compatible).
3. Click **Spectral Quality**. Diagnostics run in a worker, sampling every
   eight rows with 100 bootstrap repeats. A configured `reference_row_y` must
   agree with the Smile fit's reference row.
4. Use the Export PNG button on each tab for 300 dpi presentation plots.

The Line Quality tab includes FWHM, peak height, and area. The Uncertainty tab
shows conditional shift standard deviation; unsupported coordinates are gaps.
Plots remain blank with instructions until the required input is available.
Changing the image or wavelength calibration clears previous quality plots.
The GUI uses the current fitting exposure, so results are labelled in-sample.
For every-row independent validation, use the quality CLI.

GUI verification: offscreen PyQt5 integration tests cover all eight tabs,
calibration loading, asynchronous computation, populated plots, export, and
clearing stale results. Total suite: 36 passed, 1 legacy GUI test skipped.
