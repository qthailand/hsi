# Split configuration — single source of truth

Install over the latest HSI wavelength calibration update. The archive includes
all required new settings with values from the original supplied HSI configuration.
If you customized settings locally, set those values in the new files before
launching. Delete the old backend.json; no runtime code reads or falls back to it.

config/app.json: format version, section filenames, baseline wavelength file,
and the selected persistent wavelength profile.
config/camera.json: backend, IDs, exposure, gain and startup hardware bin.
config/scan.json: stage type.
  scan_trim_extra_ms (optional, default 0): with GenICam, the recorded scan keeps only
  frames taken at constant stage speed (the controller's acceleration/deceleration time is
  cut off each end); this extra time in ms is cut off both ends as well.
  stage_device (optional): the stage controller to open, e.g.
  "/dev/serial/by-id/usb-XIMC_..._if00" (Linux; stays the same across reboots, unlike
  /dev/ttyACM0) or a libximc URI ("xi-com:..."). Missing or "" = the device libximc
  enumerates (Windows), else on Linux the first /dev/serial/by-id link naming XIMC.
config/processing.json: orientation, spectral resolution (bin width) and RGB targets.
  resolution: bin width in nm. resolution_source: "config" (default; always use resolution)
  or "fwhm" (bin width = fwhm_step_factor x the median FWHM saved for the camera setting,
  at least 2 spectral pixels; resolution is used when no matching FWHM is saved).
  fwhm_step_factor: default 1.0 (one bin per FWHM; 0.5 = Nyquist).
  wavelength_range_nm (optional): recorded wavelength range per camera backend, e.g.
  {"genicam": [900.0, 1650.0]}. Only the bins whose every pixel lies inside the range are
  binned, shown and recorded (the same bins as without a range, bit for bit); the raw
  spectrogram stays full sensor and the cube's wavelength list gives the kept bands. Missing,
  {} or null = the full calibrated range. Set it with the Wavelength Range box in the Scan
  window (saved for the next start only by main.py, and only after a change you made).
config/dark.json: reference directory and capture frame count.
config/smile.json: model directory.
calibration/wavelength/base.json: base slope/intercept when no profile is selected.
These are project-relative paths, independent of the shell working directory.

The existing AppConfig API remains the in-memory settings object. File reads
occur at startup; there is no file read/write inside the per-frame processing.
Missing files/fields, malformed values or a missing active calibration raise an
explicit error. If the camera/bin/orientation no longer matches the active profile
(for example the camera was changed in camera.json), the profile is not applied:
a warning is logged and base.json is used, so the camera change always takes effect.
The selected profile's saved dimensions are checked against actual camera geometry.

Per camera setting: calibration/wavelength/<backend>/wavelength_bin<N>.json, written by
"Apply & Save for this setting". When the scan window starts streaming and after every
bin change, the file of the current backend and bin is used if its camera index,
orientation (raw_inverted) and frame size match; it takes precedence over the m/c above.
If there is no such file (or it does not match), the current m/c are kept and a warning
is logged.

Per camera setting: calibration/wavelength/<backend>/resolution_bin<N>.json, written by
"Save FWHM for this setting" in the FWHM / Spectral Resolution window (median FWHM in nm
over the measured lines). Used only when resolution_source is "fwhm": at stream start,
after every bin change and after a new FWHM is saved (deferred while a scan runs).
A missing, mismatching or invalid file falls back to resolution with a warning. The bin
width used is recorded in the scan metadata (resolution_nm, resolution_source, fwhm_nm).

## Calibration actions
Apply to HSI: session-only change at a worker frame boundary.
Save Profile: export a reusable JSON; it does not select a startup calibration.
Save as Default: apply, wait for worker success, then persist the selected profile.
Restart automatically loads its coefficients and hardware bin. Offline profiles
must use matching camera geometry and orientation; fitting a same-size image does
not establish camera identity. The profile source is the currently connected camera
when saving as default, so users must verify any imported profile belongs to it.

Profiles have unique filenames. app.json selects the active one. Save as Default
writes the profile's hardware bin as default_bin into camera.json itself, so
camera.json is always the file to edit. (Older versions wrote a generated
camera_<id>.json and pointed app.json at it; such a file is used only until
camera.json is edited after it, and the next Save as Default points app.json back
at camera.json.) Old profile files remain
available for recovery. Failures are reported; a successful session apply can
remain active even if persistence fails. No fitting or coefficient updates occur
while scanning/dark capture. The existing m_base=m_fitted/hardware_bin convention
is retained, without additional bin-center offsets.

To clear an incompatible default, set active_wavelength to null in app.json;
the explicit base.json calibration is then used. This does not restore a legacy file.

Validation: 66 passed, 1 skipped (including persistence, restart, incompatibility,
manifest-write failure and absence of old configuration references).
Physical camera and production-PC FPS remain to be verified locally.
