# Wavelength calibration extracted from legacy PlayerOne

Apply this overlay to the latest HSI containing Smile + Grab integration.
Open Modules > Wavelength Calibration.

Workflow:
1. Grab from Scan (post-dark/hot-pixel and orientation, before Smile/binning),
   or Load Image. Offline images must already use the HSI orientation.
2. Select a row by dragging the image line or setting Spectrum row.
3. Find Peak. Defaults preserve legacy Savitzky-Golay window 7/order 2,
   minimum peak height 5% of maximum and separation 5 pixels.
4. Enter known wavelengths. The old Mercury–Argon 400+ positional preset is
   retained exactly, including blank positions. Verify assignments manually:
   missing/extra peaks change positional correspondence. This is NOT automatic
   spectral line identification. Unassigned rows are omitted from the fit.
5. Calculate linear lambda=m*pixel+c, inspect R², RMS and residual plot.
6. Save Profile to JSON for later use. Load Profile restores points and fit.
7. Apply to HSI during normal preview, with matching camera bin/geometry.
   The worker updates config at a frame boundary and rebuilds the binning cache.
   The widget confirms application after the worker signal. No automatic apply
   occurs on fit, load or save. Applying is session-only; reload/apply the saved
   profile after restart. Existing cubes retain their own wavelength arrays.

At hardware bin B, the fitted sensor slope is m_B. HSI uses m_base=m_B/B and
retains c, matching the existing m*camera_bin convention. The module does not
introduce a different subpixel bin-center offset or polynomial wavelength model.
Verify calibration when changing hardware ROI/bin or camera. A matching shape
alone cannot prove a matching optical setup.

Smile compatibility: Grab is before Smile. For new models referenced to y_ref,
select that same row when deriving wavelength coordinates intended for corrected
images. Independently calibrated wavelength/Smile models are not automatically
guaranteed compatible. No hard-coded y=552 is imposed.

Package modules/wavelength_calibration contains core.py (no Qt), widget.py,
__init__.py (lazy GUI exports), and __main__.py.
Standalone GUI: python -m modules.wavelength_calibration
Headless CSV fit (two numeric columns, no header):
python -m modules.wavelength_calibration --points points.csv --output calibration.json --bin 2
Python API: fit_wavelength(pixels, wavelengths, camera_bin=2)

Differences from old tab: real save/load, profile validation, editable/add/remove
points, detection controls, residual/RMS plot, correct image row bounds,
replacement rather than accumulation of peaks, and stale fit invalidation.
Old CameraPO code and acquisition logic were not imported.
Performance: calibration fitting occurs on demand. No calibration analysis runs
in the live frame loop. Apply triggers a one-time binning-cache rebuild/JIT path;
perform it before starting acquisition scans. Existing fast kernels are retained.

Tests: 60 passed, 1 skipped; run
python -m pytest -q tests modules/smile
Physical camera/stage and production-PC speed not verified in this environment.
