# modules/stage

Motorised stage for pushbroom scanning: Standa/XIMC **8SMC1** rotation and linear stages.

## Overview

* `StageSequence` (`sequence.py`) - Qt-free scan: open, home, scan, return home, close,
  with callbacks. Usable from scripts and any GUI toolkit.
* `StageThread` (`thread.py`) - the same sequence on a `QThread` with signals.
* `Stage8SMC1` / `Stage8SMC1Linear` (`smc1.py`) - the drivers over `libximc`.
* `speed.py` - how a speed is encoded for the controller (no SDK needed; used by the
  drivers and the scan window).

Importing `modules.stage` does not import PyQt5 or `libximc`; both are loaded lazily.

## Requirements

* The vendor library `libximc` (Python bindings `libximc.highlevel` and its native
  library), only on a machine that drives real hardware. A custom `stage_factory`
  needs none.
* PyQt5 only for `StageThread`.

## Stand-alone use

```
python -m modules.stage --type rotation --degree 10 --dps 2
python -m modules.stage --type linear --distance-mm 20 --speed-um-s 1000
```

Runs one headless sequence on the real stage and prints time-stamped events; exit code 0
if it completed, else 1.

## Embedding in an application

Without Qt:

```python
from modules.stage import StageSequence

seq = StageSequence("rotation",
                    on_before=start_recording,    # scan move is about to start
                    on_after=build_cube,          # scan move ended (also after a stop)
                    on_home=lambda: print("home"),
                    on_error=print)
seq.degree, seq.degree_per_second = 20, 2.0
threading.Thread(target=seq.run).start()          # run() blocks
...
seq.request_stop()                                # any thread; still returns home
```

With Qt:

```python
from modules.stage import StageThread

thread = StageThread(parent=window, stage_type="linear")
thread.distance_mm, thread.speed_um_s = 20.0, 1000.0
thread.stagebefore.connect(start_recording)
thread.stageafter.connect(build_cube)
thread.returned_home.connect(on_home)
thread.error.connect(on_error)
thread.start()
```

For tests or simulators pass `stage_factory=` (a zero-argument callable returning an
unopened driver with `open/close/set_position/set_speed...`).

## Data formats / conventions

* Rotation: degrees and degrees/second (full step 0.02 deg; speed limit 20 deg/s).
* Device: `stage_device` (config/scan.json) path or URI; empty = libximc enumeration,
  else (Linux, no `/dev/ximc` udev links) the first `/dev/serial/by-id` link naming XIMC.
* Homing before a scan runs at `HOMING_SPEED_STEPS_S` = 500 full steps/s on both stages
  (linear 10 mm/s, rotation 10 deg/s); the return home after a scan uses the scan speed (v1).
* Linear: millimetres and mm/s; **`speed_um_s` on the sequence is micrometres/second**.
  50 steps per mm.
* `StageSequence.phase`: `idle, opening, homing, scanning, returning, closing`.
* Result flags after `run()`: `failed`, `failed_phase`, `cancelled`, `scan_finished`,
  `completed`.
* `on_after` fires as soon as the scan move ends (or is stopped), so the cube can be
  built before the stage has returned home. A return-home failure does not invalidate
  that data.
* Rotation moves (`set_degree`) round to the nearest full step (2.3 deg = 115 steps).

## Speed resolution

The controller takes a speed as whole full steps/s (`Speed`) plus a fraction in
microsteps/s of the engine's microstep mode (`uSpeed`). The drivers write the nearest
microstep (`speed.encode_speed`; a fraction that rounds up to a whole step carries into
`Speed`), read the move settings back, and return the speed the controller runs.

| Stage | 1 full step | Microstep mode | Speed resolution |
|-------|-------------|----------------|------------------|
| Linear | 20 µm | as configured on the controller, read when the driver opens | 20 µm/s ÷ microsteps per step: 0.078 µm/s at 1/256, 10 µm/s at 1/2 |
| Rotation | 0.02 deg | 1/2, set by the driver | 0.01 deg/s |

* Limits: at most `MAX_SPEED_STEPS_S` = 1000 full steps/s (linear 20 mm/s, rotation
  20 deg/s), or the engine's nominal speed if the controller limits to it
  (`ENGINE_LIMIT_RPM`). A faster request is clamped and a positive speed below one
  microstep/s is raised to one microstep, both with a logged warning. `set_speed_dps`
  rejects more than 20 deg/s (`ValueError`).
* Before this, the fraction was dropped: the linear stage ran 150 µm/s at 140 (-6.7 %),
  250 at 240, and anything below 20 µm/s at 0; rotation ran 0.03 deg/s at 0.02.
* The speed actually run: `set_speed` returns full steps/s (also `speed_steps_s`),
  `set_speed_mm_s` mm/s, `set_speed_dps` deg/s, all from the controller's read-back.
  `StageSequence.speed_um_s_achieved` / `degree_per_second_achieved` hold it for the scan
  move (`None` with a driver that returns nothing, e.g. a test fake).
* The scan window stores it in the raw spectrogram metadata as `speed_um_s_achieved` /
  `degree_per_second_achieved` next to the requested `speed_um_s` / `degree_per_second`,
  times the expected scan duration with it, and shows `(stage runs ...)` in the cube
  preview title when it differs from the entered speed. Its rotation speed box takes
  0.01..20 deg/s with two decimals (what the stage runs); the linear box takes up to
  20000 µm/s.
* Speed calibration (scan/slit stretch W/H from Check Circle / Check Square): correct the
  achieved speed, `new speed = achieved speed × W/H`.

## API summary

| Member | Purpose |
|--------|---------|
| `StageSequence(stage_type, stage_factory, on_before, on_after, on_home, on_error)` | configure |
| `.degree`, `.degree_per_second`, `.distance_mm`, `.speed_um_s` | motion parameters |
| `.run()` | execute (blocking, never raises; failures go to `on_error`) |
| `.request_stop(return_home=True)` | interrupt; `False` aborts without homing (shutdown only) |
| `StageThread` signals | `stagebefore`, `stageafter`, `returned_home`, `error(str)` |
| `.speed_um_s_achieved`, `.degree_per_second_achieved` | scan speed the controller runs (read back) |
| `Stage8SMC1` | `open/close`, `set_speed[_dps]`, `set_degree`, `set_position[_r]`, `zero`, `speed_steps_s` |
| `Stage8SMC1Linear` | `open/close`, `move_to_mm`, `move_by_mm`, `set_speed[_mm_s]`, `get_position_mm`, `set_position[_r]`, `speed_steps_s` |
| `speed.encode_speed` / `decode_speed` | steps/s <-> `(Speed, uSpeed)` for a microstep mode |

## Threading

`run()` must execute on one worker thread (`StageThread` does this). `request_stop()` may
be called from any thread. The drivers are not thread-safe; the only cross-thread call is
the stop command issued by `request_stop`. `StageThread` signals are emitted from the
worker thread and delivered queued to GUI receivers.

## Tests

```
PYTHONUTF8=1 python -m pytest -q tests/test_stage_failures.py
```

The tests use a fake stage through `stage_factory`, and the speed tests import the real
drivers over a fake `libximc` that records the `Speed`/`uSpeed` written, so no hardware
or SDK is needed.
