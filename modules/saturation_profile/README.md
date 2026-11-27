# saturation_profile

## Overview

Measures what a clipped raw pixel of a camera **actually reads** for each configuration, instead
of computing it from the bit depth. The value comes from real saturated frames. For example, the
Apollo-M MAX ADC tops out at 4094, so it reads 65504, not 4095 << 4 = 65520.

The result is stored in the camera profile as a *saturation profile*. Basic Detection and Full
Saturation Tracking take their confirmed-clipping threshold, and its provenance, from that one
source.

This module is quality data only:

* the intensities are never touched;
* the nominal full scale (`camera_bit_info().full_scale`, the metadata and radiometric keys) is
  never touched;
* no other calibration file (dark, smile, keystone, wavelength, FWHM, radiometric, flat-field)
  is touched.

| Parameter | Purpose |
|---|---|
| Nominal full scale | metadata, intensity representation, radiometric reference |
| Measured saturation level | clipping detection, near-saturation, quality tracking |

## Measurement algorithm (`core.py`)

### 1. Exposure sweep (`measure`, `SweepPlan`)

The sweep runs per hardware bin and per gain:

1. Start at the given exposure.
2. If a plateau is already there, step down (`/ factor`) to see where it forms. Stop stepping
   down when the plateau does not shrink: it does not depend on exposure, so it is hot pixels.
3. Step up (`x factor`) until a plateau absent at the shortest exposure, or one that grew, has
   been followed for `steps_after_plateau` more steps, or until the exposure limit is reached.

`FrameSource.apply` returns once the setting is active, and `grab` returns only frames exposed
with it. Each step reduces its frames to the top of their histograms (`StepFrames`); the frames
themselves are not kept.

### 2. Plateau candidates (`step_candidates`)

A candidate is an exact value in the top `band` (10 %) of the nominal full scale that, in every
frame of the step:

* holds at least `min_pixels` pixels (or `min_fraction` of the frame, whichever is larger); and
* holds `spike_ratio` (×10) times the median count of its neighbouring values.

The raw maximum alone never decides anything.

### 3. Clipping evidence (`analyse`)

A candidate is a clipping plateau when all of these hold:

* it stays once it appeared;
* its pixel count does not fall as the exposure rises;
* either a shorter exposure without it was measured (its *entry*), or its count grew ×1.2.

Several cases give no such evidence and are reported as such, never guessed:

* a constant count from the shortest exposure (hot pixels, fixed pattern);
* a frame saturated everywhere at every exposure;
* a value seen at one exposure only.

Each secondary peak is judged the same way.

### 4. Level

1. Per gain, the level is the lowest clipping plateau. It is accepted as one scalar threshold
   only when the pixels between it and the highest clipping plateau are (almost) all on clipping
   plateaus.
2. Every gain measured (at least 2) must give the same level. The result is then **verified**
   over that gain range, and valid only inside it.
3. Levels that differ between gains are all recorded, and the result is **inconclusive**. No
   gain-dependent profile is made.

### Statuses

* **Verified:** the only status that gives a threshold.
* **Inconclusive:** too little or conflicting evidence.
* **Failed:** the camera timed out or failed during the measurement.

### Limits (stored in every record)

* This is **digital** clipping, not sensor non-linearity or full-well saturation.
* With hardware binning, a binned pixel reads the level only when every sensor pixel of its block
  clipped. A partly clipped block reads less, and the near-saturation count catches it.

## Requirements

numpy. The analysis needs no Qt.

## Stand-alone use

`BackendFrameSource` (`sources.py`) drives an open `modules.camera` backend directly. Use it only
when no other program is acquiring from the camera:

```python
from modules.camera import create_backend
from modules.saturation_profile import SweepPlan, measure, SaturationProfileStore
from modules.saturation_profile.sources import BackendFrameSource
backend = create_backend(config); backend.open()
records = measure(BackendFrameSource(backend, "poa"), SweepPlan((1, 2, 3, 4), (1, 200), 250.0))
store = SaturationProfileStore(camera_profile.saturation_folder)
for record in records:
    store.save(record)
```

## Embedding in an application

* `SaturationMeasureWidget` is the calibration window. It shows the live frame, the upper
  histogram with the plateaus highlighted and the measured threshold as a line, the step it is on
  (bin, gain, exposure), the result per bin (coloured dot, tooltip with the reasons), and
  Measure / Cancel / Save buttons.
* The host gives it three things:
  * `set_session_provider(callable)`: returns `source` + `finish()`. `finish` restores the camera
    and returns the problems, and runs on the measurement thread in every case: done, cancelled
    or failed.
  * `set_camera(...)`: bins, gain and exposure ranges.
  * `set_store(store)`.
* It reports back through `running_changed(bool)`, `measurement_finished(records, problems)` and
  `profiles_saved(results)`.
* In the HSI application, the Scan window's *Saturation level* Corrections row (Measured /
  Missing / Incompatible / Inconclusive / Gain outside) opens it. The controller:
  * measures through `modules.acquisition.saturation_measure.EngineFrameSource`;
  * locks scans, dark capture and the camera settings while the measurement runs;
  * does not save the measurement's bins as the user's;
  * after the restore, compares the calibration state with the one before. If it differs, scans
    stay locked and the row shows *Not restored*.
* `python -m modules.saturation_profile` opens the window on a simulated camera (`demo.py`).


* The calibration window measures through the application's acquisition engine.
* The engine reads `SaturationProfileStore.lookup(identity, gain)` when the camera configuration
  changes. It never reads a profile per frame.

## Data formats / conventions

Defined in `modules/contracts/saturation_profile.py` (`format_version` 1). The files live in
`camera_profiles/<backend>/<model>_<serial>/saturation/`:

* `saturation_<format>_d<delivered>s<sensor>_bin<N>.json`: the active profile. Its identity is
  backend, model, serial, pixel format, delivered and sensor bits, and bin.
* `history/<stem>.<YYYYmmdd-HHMMSS>[_n].json`: the profiles it replaced, and measurements that
  did not become active.

Each record holds:

* status, level, nominal full scale;
* every plateau value, with its role, reason and per-gain counts over exposure;
* the verified gain range, the gains and exposures tested, and the frames;
* the evidence of each step (top values, plateaus);
* reasons, `measured_at`, the git revision, firmware, and the limits.

Saving (`SaturationProfileStore.save`):

* Writes are atomic: a temporary file, then `os.replace`. The replaced file is copied to history
  first.
* A verified record always becomes active.
* Any other record becomes active only when the active one is not verified. It never replaces a
  verified profile.

A lookup returns `verified` only when the identity in the file matches the active configuration
and the gain is inside the verified range. Every other result means *not checked*: `missing`,
`inconclusive`, `failed`, `incompatible`, `invalid` or `gain_outside`.

## API summary

* `measure`, `SweepPlan`, `FrameSource`, `MeasurementCancelled`, `frame_timeout_s`.
* `analyse`, `step_from_frames`, `step_candidates`, `top_histogram`, `fully_saturated`,
  `failed_record`, `PlateauParams`, `StepFrames`.
* `SaturationProfileStore`, `ProfileLookup`, `SaveResult`, `software_revision`, `STATE_*`.
* `sources.BackendFrameSource`, `identity_of`, `nominal_full_scale_of`.

## Threading

* The analysis is pure.
* `measure` blocks; run it on a worker thread and cancel it with a `threading.Event`.
* The store does plain file I/O, with one writer at a time.

## Tests

* `tests/test_saturation_profile.py`: a simulated camera, plus real Apollo-M MAX evidence in
  `tests/fixtures/saturation/` (top histograms and 48×48 raw crops from 2026-11-27).
