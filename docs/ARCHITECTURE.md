# HSI architecture

The application is a **thin shell around self-contained modules**. Each module under
`modules/` follows [MODULE_CONTRACT.md](MODULE_CONTRACT.md), so it can be added to this
application, run stand-alone (`python -m modules.<name>`), or embedded in another PyQt5
program. The list of modules, what each requires and how to start it is generated in
[MODULES.md](MODULES.md).

## Layers

```
 main.py ──► windows/          application shell (MainWindow, ScanWindow, integrations, registry)
                │  uses
                ▼
            controllers/       ScanController: wires widgets <-> acquisition thread <-> stage
                │  uses
                ▼
            modules/*          feature modules (no imports of windows/controllers/workers/widgets)
                │
                ├─ hardware      camera (POA, GenICam, XIMEA), stage (XIMC 8SMC1)
                ├─ acquisition   engine + recorder + session (Qt-free) and the QThread adapter
                ├─ processing    preprocessing, dark, smile, spectral_processing, image, sharpness
                ├─ calibration   wavelength_calibration, spectral_resolution (FWHM), reference_roi, radiometric
                ├─ data          cube_io, contracts (shared formats), configuration
                └─ GUI parts     ui_kit, live_view, cube_preview, cube_viewer, spectrogram_viewer,
                                 status_card, frame_exchange
```

Dependencies only point **downwards**; each module lists the other modules it imports in
`__requires__`, and `tests/test_module_contract.py` fails on undeclared imports, on cycles,
on imports of the application layers, and when `import modules.<name>` loads PyQt5.

## One frame, one scan

```
camera.read_frame_raw()                       modules.camera
  -> dark + hot pixel -> orient -> Smile      modules.preprocessing / dark / smile
  -> spectral binning                         modules.spectral_processing
  -> ScanRecorder.offer()  (EVERY frame)      modules.acquisition  (worker thread)
  -> frame_ready  (<= 30 fps, display only)   CameraPreviewThread -> ScanController -> widgets
  -> cube_frames_ready (batches, every frame) -> real-time cube preview
scan end: ScanSession.finish() -> cube = create_pushbroom_cube_from_corrected(images)
```

The 30 fps limit applies to what the GUI draws. Display frames are pulled: the worker keeps
only the newest one, and the GUI takes it when ready and then rests at least as long as it was
busy, so a slow GUI (e.g. the Orin) shows fewer frames instead of queueing them, and dialogs
stay responsive. The cube always contains every frame the camera delivered while the stage
moved, and Keep Spectrogram stores the matching raw sensor frames 1:1. A Reference ROI
preview scan uses those same recorded frames, one RGB column each.

## Where logic lives

| Concern | Place | Qt? |
|---|---|---|
| Per-frame pipeline, pending camera changes | `modules.acquisition.AcquisitionEngine` | no |
| Recording of all frames | `modules.acquisition.ScanRecorder` | no |
| Scan state machine, visual-selection scan | `modules.acquisition.ScanSession` | no |
| Qt signals for the engine | `modules.acquisition.CameraPreviewThread` | yes |
| Stage motion sequence | `modules.stage.StageSequence` (Qt wrapper `StageThread`) | no / yes |
| Which dark file fits the camera | `modules.dark.select_dark` | no |
| Settings groups | `modules.<name>.config` dataclasses, composed by `config.AppConfig` | no |
| Wavelength file formats | `modules.contracts.wavelength` | no |
| Wiring modules into the window | `windows/integrations.py`, `windows/registry.py` | yes |
| Widgets <-> worker <-> stage | `controllers/scan_controller.py` | yes |

## Adding a module

1. Copy `modules/_template/` (see the contract), fill in `core.py`, `README.md` and the
   `__init__.py` markers.
2. `python tools/module_audit.py` until clean; add tests.
3. To show it in the application add an `Integration` subclass in `windows/integrations.py`
   and list it in `windows/registry.py`.
4. `python tools/module_index.py` to refresh `docs/MODULES.md`.

## Compatibility shims

Old import paths still work: `workers/camera_preview.py`,
`workers/stage_thread.py`, `workers/cube_render.py`, `workers/export_workers.py`,
`workers/hdf5_worker.py`, `workers/cube_worker.py`, `widgets/*.py`, `utils/display.py`,
`windows/cube_viewer.py`, `windows/spectrogram_viewer.py`, `Stage8SMC1.py`,
`ImageSharpness.py`. Each is a few lines that re-export the new module; new code imports
from `modules.*`. They can be deleted once nothing outside this repository imports them.

## Tests

* `tests/test_module_contract.py` - the contract, for every module.
* `tests/test_camera_backends.py` - one contract suite run against POA, GenICam and XIMEA
  with simulated SDKs.
* `tests/test_acquisition.py` - recorder, session, engine (fake camera), Qt thread.
* `tests/test_app_smoke.py` - the real `MainWindow` with a fake camera and stage: preview,
  a full scan producing a cube with every frame, each module opening from the menu.
