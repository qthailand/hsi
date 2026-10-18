# HSI package refactor — stage 1

This archive is an overlay for a COPY of the original hsi directory. Extract it
into the directory containing that copy, replacing matching Python files.
Keep original SDK DLLs, dark_frames, backend.json and other runtime data.
The delivery excludes DLLs, data, Git metadata and Python/Numba caches.
Start as before: python main.py from the hsi directory.

## Implemented
- modules.spectral_processing: unchanged Numba LambdaBinner and wavelength API.
  The name avoids collision with third-party Spectral Python in standalone use.
- modules.dark: original DarkFrameLibrary and cached DarkProcessor methods.
  AbstractCameraBackend inherits these exact methods. No additional per-frame
  proxy call, image copy, queue, thread or Qt signal was introduced.
- modules.image: original orientation and image helper implementation.
- modules.cube_io: original cube assembly and lazy ENVI memmap loader.
- modules.contracts: optional CubeData boundary validation; no per-frame use.
- modules.smile: supplied package copied unchanged; not enabled in acquisition.
- Legacy paths remain import shims. Active processing callers import packages.
- modules root loads legacy BSQ GUI only on explicit access.

Processing order, preview limit, acquisition lifecycle, recording policy,
array axes and buffers remain unchanged. This is structural refactoring.
Legacy create_pushbroom_cube_from_corrected returns (slit, scan, wavelength);
LambdaBinner.transform_to_cube retains (wavelength, slit, scan).

## Commands (from hsi)
python -m modules.spectral_processing input.npy output.npz --m 0.5806 --c 399.791 --threads 2
python -m modules.dark input.npy dark_frames/dark_b2_e10.0_g180.0.npz corrected.npy
python -m modules.cube_io example.hdr
python -m modules.smile
python -m pytest -q tests modules/smile

## Performance validation
The production baseline supplied by the user is 70–123 fps acquisition and
all processing after fetch below 1 ms on their PC. Neither is certified by
this container. Compare original and refactored applications on that PC using
identical camera configuration and enabled features.

For a same-PC relative kernel comparison, supply an untouched extracted original:
python tools/benchmark_packages.py --legacy C:/baseline/hsi --height 1024 --width 2048 --threads 2 --repeat 1000
Use actual delivered camera dimensions and calibration coefficients (--m, --c,
--step). The benchmark warms up JIT, alternates measurement order and checks
exact output equality. It measures dark/hot-pixel/orientation/binning only:
fetch, GUI, disk recording, smile, sharpness and diagnostics are excluded.
BENCHMARK_EXAMPLE.json is a synthetic container run, not production validation.
Moving the Numba function changes its cache path: first startup recompiles it.
Do not copy old Python or Numba caches into the new package.

## Remaining work
Extract export writer implementations from Qt workers; split BSQ/radiometric
GUI and processing; camera/motion lifecycle ownership; integrate new smile
before spectral binning after scientific/performance comparison. No new GUI
or real-time smile pipeline is claimed in this stage.
