# modules/stage

Motorised stage for pushbroom scanning: Standa/XIMC **8SMC1** rotation and linear stages.

## Overview

* `StageSequence` (`sequence.py`) - Qt-free scan: open, home, scan, return home, close,
  with callbacks. Usable from scripts and any GUI toolkit.
* `StageThread` (`thread.py`) - the same sequence on a `QThread` with signals.
* `Stage8SMC1` / `Stage8SMC1Linear` (`smc1.py`) - the drivers over `libximc`.

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
* Linear: millimetres and mm/s; **`speed_um_s` on the sequence is micrometres/second**.
  50 steps per mm.
* `StageSequence.phase`: `idle, opening, homing, scanning, returning, closing`.
* Result flags after `run()`: `failed`, `failed_phase`, `cancelled`, `scan_finished`,
  `completed`.
* `on_after` fires as soon as the scan move ends (or is stopped), so the cube can be
  built before the stage has returned home. A return-home failure does not invalidate
  that data.

## API summary

| Member | Purpose |
|--------|---------|
| `StageSequence(stage_type, stage_factory, on_before, on_after, on_home, on_error)` | configure |
| `.degree`, `.degree_per_second`, `.distance_mm`, `.speed_um_s` | motion parameters |
| `.run()` | execute (blocking, never raises; failures go to `on_error`) |
| `.request_stop(return_home=True)` | interrupt; `False` aborts without homing (shutdown only) |
| `StageThread` signals | `stagebefore`, `stageafter`, `returned_home`, `error(str)` |
| `Stage8SMC1` | `open/close`, `set_speed_dps`, `set_degree`, `set_position[_r]`, `zero` |
| `Stage8SMC1Linear` | `open/close`, `move_to_mm`, `move_by_mm`, `set_speed_mm_s`, `get_position_mm`, `set_position[_r]` |

## Threading

`run()` must execute on one worker thread (`StageThread` does this). `request_stop()` may
be called from any thread. The drivers are not thread-safe; the only cross-thread call is
the stop command issued by `request_stop`. `StageThread` signals are emitted from the
worker thread and delivered queued to GUI receivers.

## Tests

```
PYTHONUTF8=1 python -m pytest -q tests/test_stage_failures.py
```

The tests use a fake stage through `stage_factory`, so no hardware is needed.
