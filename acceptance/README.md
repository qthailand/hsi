# HSI automated acceptance test suite

One entry point, `acceptance/run_acceptance.py`, checks the HSI system as the operator left it.
It runs on Windows workstations and on Linux ARM64 (Jetson Orin).

The suite **verifies** the system; it never **prepares** it:

* It does not calibrate, generate, migrate or overwrite any profile.
* It changes no configuration file.
* It moves no stage unless you ask (`--allow-stage-motion`).
* It leaves the camera at the setting it found.

## Workflow on the target (Jetson Orin)

1. **Calibrate with the production application** on the target: dark references for the scan
   settings, Smile / keystone, wavelength (and FWHM if `resolution_source = fwhm`), and the
   **saturation level** (Corrections → *Saturation level* → Measure → Save, at all bins and gains
   you will scan with). Leave the camera profiles where the application saved them.
2. **Close the application.** The suite opens the camera itself.
3. Run the suite:

   ```bash
   python3 acceptance/run_acceptance.py --camera real --expect-backend cupy_cuda
   python3 acceptance/run_acceptance.py --camera real --mode stress --stress-minutes 60 --expect-backend cupy_cuda
   ```

4. Read `summary.html` (or `summary.csv`, `report.json`) in
   `acceptance_reports/<time>_<host>/`. The exit code is 0 when nothing FAILed.

## What it loads and how

| What | How (the application's own mechanism) |
|---|---|
| Configuration | `config.load_app_config(<config dir>)` (`--config-dir`, default the project's `config/`) |
| Camera | `modules.camera.create_backend(config)` (real), or replay / synthetic frames with the camera's identity |
| Camera profile | `CameraProfile.for_camera(config.camera_profile_folder, backend, model, serial)`. No `profile.json` refresh. |
| Dark | `DarkFrameLibrary` + `select_dark` for (bin, exposure, gain, frame size). Only existing folders are opened. |
| Smile, wavelength, FWHM | `config.calibration_store(config)` (the profile, then the older folders) |
| Saturation level | `SaturationProfileStore.lookup` (verified, the same configuration, gain inside the verified range) |

No path is hard-coded: the project root is the parent of `acceptance/`, and every folder comes from the
configuration.

## Profile readiness

Before any test, the suite checks the camera profile for the setting under test:

* camera profile (model / serial);
* dark;
* Smile;
* wavelength;
* FWHM;
* saturation.

Each item is `READY`, `MISSING`, `MISMATCH` (with the reason), or `NOT USED` for an optional item.
A test that needs an item that is not ready is **SKIPPED** with that reason. Only that test is
skipped; everything else still runs.

| Test | Needs |
|---|---|
| `production.detection_on_off`, `production.stress` | camera profile, saturation |
| `production.saturation_profile_in_use` | saturation |
| `production.pixel_regression` | camera profile (records which corrections were active) |
| `production.scan_with_stage` | camera profile, dark, Smile, wavelength, saturation |

## Tests

### Integrity

| Test | Checks |
|---|---|
| `integrity.environment` | OS, CPU, Python and library versions; git revision and local changes |
| `integrity.camera_identity` | Model, serial, bin, gain, exposure, frame size, pixel format, bits |
| `integrity.profile_readiness` | Profile readiness (above) |
| `integrity.profiles_unchanged` | SHA-256 of every file is identical before and after the tests, in: camera profiles, config folder, older dark / smile / calibration folders |
| `integrity.camera_restored` | (real camera) The setting is identical before and after |

### Synthetic unit tests

`synthetic.unit.<file>` runs the existing unit tests, one process per file, from `tests/` (when
present). They use synthetic data only and never production profiles. Use `--unit-tests` for
other files and `--skip-unit` to leave them out.

### Production acceptance tests

**`production.detection_on_off`**: the acquisition loop with basic detection ON, then OFF (full
tracking off). It reports:

* count mean, P95, P99 and max;
* line rate (FPS), dropped frames and frames missing;
* frame latency (read to recorder);
* CPU, GPU, RAM, temperatures and clocks.

It passes when, with detection ON:

* count mean ≤ `--count-mean-ms` (0.5);
* count P99 ≤ `--count-p99-ms` (1.0);
* at least 30 frames were counted.

A paced camera (real, or `--replay-fps`) must also have line rate ON ≥ `--fps-ratio` (0.98) × OFF,
and no more dropped frames ON than OFF. A free-running replay / synthetic camera reports the line
rate for information only.

**`production.saturation_profile_in_use`**: the engine counts against the verified profile that
readiness found (same profile id).

**`production.processing_backend`**: the backend requested and the backend that actually ran. A
**CPU fallback is reported** in the summary banner. With `--expect-backend X` the test FAILs
unless X actually ran.

**`production.pixel_regression`**:

1. The same raw frames go through the acquisition engine twice, on the **same backend** and with
   the production profiles:
   * once with the code under test;
   * once with the code of `--baseline-ref` (default `41f330b`, the last commit before Phase 2E),
     extracted by `git archive` into a temporary folder; the repository is not touched.
2. Every binned frame must be pixel-identical.

The raw frames are: replay frames, frames captured from the real camera at its configured setting
(`--capture-frames`), or deterministic synthetic frames.

**`production.stress`** (`--mode stress`): basic detection runs for `--stress-minutes`, as a live
preview so the recorder does not fill the RAM. It reports per-minute windows. It passes when:

* the count P99 stays ≤ the limit;
* no window falls below 90 % of the median frame rate;
* the count mean does not rise 1.5× from the first window;
* process memory grows ≤ 50 MB + 1 MB per minute;
* there are no errors.

**`production.scan_with_stage`**: only with `--allow-stage-motion` and `--camera real`.

* It runs a real scan through the application's controller with the configured stage. **The stage
  moves.**
* It checks:
  * the cube header's `saturation_*` keys;
  * basic detection vs full tracking;
  * the camera setting before and after.

## Camera sources

| `--camera` | Frames | Identity |
|---|---|---|
| `real` (default) | The camera, at its configured setting | From the camera |
| `replay --replay-dir D` | `D/*.npy` raw frames (2-D or stacks) | `D/identity.json` (written by a real-camera run's capture), else from the camera profile |
| `synthetic` | Deterministic frames at the camera's bit depth and frame size, clipped at the measured level | Camera profile (`--profile` if several) + configured bin / gain / exposure |

A real-camera run keeps its captured frames and `identity.json` in `<report>/work/captured/`. Pass
that folder to `--replay-dir` to replay them later (on the same or another machine).

## Options

```
--camera real|replay|synthetic   --replay-dir DIR   --replay-fps HZ   --profile <model>_<serial>
--mode quick|stress              --seconds S         --stress-minutes M
--backend NAME                   --expect-backend NAME
--baseline-ref REV               --skip-regression   --skip-unit   --unit-tests FILE ...
--allow-stage-motion             --config-dir DIR    --output DIR
--count-mean-ms 0.5  --count-p99-ms 1.0  --fps-ratio 0.98  --capture-frames 16
```

The length of each ON / OFF run:

* quick mode: 5 s;
* stress mode: 30 s;
* a paced camera: long enough for 40 frames (e.g. 20 s at 500 ms exposure).

## Reports

Everything goes to `acceptance_reports/<time>_<host>/` (ignored by git).

| File | Contents |
|---|---|
| `report.json` | Every result and metric; environment, git revision and local changes; camera identity; configuration and its file hashes; profile readiness with revisions (profile ids, dark capture time, Smile model creation); the processing backend; monitor series |
| `summary.csv` | One row per test: status, reason, the main metrics |
| `summary.html` | The verdict, a CPU-fallback banner, the tests, readiness, integrity, camera, configuration, software and platform |
| `work/` | Worker inputs / outputs / logs, captured frames |

Every test is **PASS**, **FAIL** or **SKIPPED**, always with its reason.

## System metrics

| Metric | Source |
|---|---|
| CPU, RAM, process RSS / CPU, CPU clock | `psutil` |
| Temperatures | `psutil` / `/sys/class/thermal` (Linux); usually none on Windows, reported as unavailable |
| GPU load, clock and SoC temperatures (Jetson) | `tegrastats`, else Jetson sysfs |
| GPU (desktop NVIDIA) | `nvidia-smi` |

The report names the source it used for each, or `n/a`.

## Requirements

* The application's own environment (numpy, numba, OpenCV, PyQt5, the camera SDK).
* `psutil`.
* `pytest`, for the unit tests.
* `git`, for the pixel regression.

When an optional tool is missing, the tests that need it are SKIPPED with that reason.

## Layout

| File | Contents |
|---|---|
| `run_acceptance.py` | Entry point, test logic, pass criteria |
| `worker.py` | One task per process: probe, capture, pipeline, perf, stress, scan |
| `profiles.py` | Profile readiness (read only) |
| `monitor.py` | System sampler |
| `frames.py` | Replay / synthetic frames |
| `report.py` | JSON / CSV / HTML |
| `common.py` | Paths, hashing, git, statuses |
