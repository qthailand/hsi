"""End-to-end check of the real application with the real camera and stage.

Opens the real ``MainWindow`` (offscreen, so no display is needed), runs the live preview
for a few seconds, performs one short scan and verifies that the cube holds *every* frame
the camera delivered while the stage moved.

Usage (needs the camera and the stage connected; the stage moves ``--degree`` degrees)::

    python tools/hardware_smoke.py --backend poa --degree 2 --dps 1 --bin 2 --gain 1 --exposure-ms 1

Exit code 0 means every check passed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    """Run the scan check; returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="poa", help="poa | genicam | ximea")
    parser.add_argument("--degree", type=float, default=2.0, help="scan angle (rotation stage)")
    parser.add_argument("--dps", type=float, default=1.0, help="scan speed (degrees per second)")
    parser.add_argument("--bin", type=int, default=2)
    parser.add_argument("--gain", type=int, default=1)
    parser.add_argument("--exposure-ms", type=float, default=1.0)
    args = parser.parse_args(argv)

    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    import config as app_config
    import windows.main_window as main_window_module

    original_loader = app_config.load_app_config

    def load_config(*a, **k):
        cfg = original_loader(*a, **k)
        cfg.camera_backend, cfg.default_bin = args.backend, args.bin
        cfg.default_gain, cfg.default_exposure_ms = args.gain, args.exposure_ms
        cfg.stage_type = "rotation"
        return cfg

    main_window_module.load_app_config = load_config
    window = main_window_module.MainWindow()
    window.resize(1200, 800)
    scan = window.scan_window
    controller = scan.controller
    scan.degree_spin.setValue(args.degree)
    scan.degree_per_second_spin.setValue(args.dps)

    def pump(condition, timeout):
        end = time.time() + timeout
        while time.time() < end:
            app.processEvents()
            if condition():
                return True
            time.sleep(0.005)
        return False

    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    try:
        info = controller.backend.info
        check("camera open", info.width > 0, f"{info.width}x{info.height} bin {controller.camera_bin}")
        check("live preview running", pump(lambda: controller.preview_frame_index > 30, 30),
              f"frames shown: {controller.preview_frame_index}")
        pump(lambda: False, 2.0)

        fetched = {"n": 0}
        original_fetch = controller.backend._fetch_raw_frame

        def counting(timeout_ms=1000):
            frame = original_fetch(timeout_ms)
            if frame is not None:
                fetched["n"] += 1
            return frame

        controller.backend._fetch_raw_frame = counting
        scan.camera_controls.keep_spectrogram_check.setChecked(True)
        marks = {}
        recorder = controller.preview_thread.recorder
        original_arm, original_take = recorder.set_recording, recorder.take_frames

        def arm(active):
            if active and "start" not in marks:
                marks["start"] = fetched["n"]          # camera frames fetched when recording is armed
            original_arm(active)

        def take():
            marks.setdefault("end", fetched["n"])      # ... and when it is disarmed (cube is built after)
            return original_take()

        recorder.set_recording, recorder.take_frames = arm, take

        def watch():
            return scan._pending_cube is not None

        controller.start_scan()
        check("scan started", controller.state == controller.SCANNING)
        ok = pump(watch, args.degree / max(args.dps, 0.01) + 60)
        check("cube published", ok)
        stage_done = pump(lambda: controller.stage is None, 60)
        check("stage returned home", stage_done)
        if ok:
            cube = scan._pending_cube
            slit, scans, bins = cube.shape
            raw = scan._pending_spectrogram
            check("cube axes (slit, scan, wavelength)", slit == info.height and bins > 1, f"{cube.shape}")
            check("wavelength axis matches bins", scan._pending_wavelengths is not None
                  and len(scan._pending_wavelengths) == bins)
            check("raw spectrogram kept 1:1", raw is not None and raw.shape[0] == scans,
                  f"raw {None if raw is None else raw.shape}")
            during = marks["end"] - marks["start"]
            check("every camera frame in the cube", abs(scans - during) <= 8,
                  f"cube scans {scans}, camera frames while recording {during}")
            check("preview matches cube", pump(lambda: scan.cube_preview._current_shape[1] == scans, 10),
                  f"preview columns {scan.cube_preview._current_shape[1]}")
        # hardware bin changes while previewing: camera, frame size and the matching dark follow
        for target in (1, 4, 2):
            index = scan.camera_controls.bin_combo.findData(target)
            if index < 0:
                continue
            scan.camera_controls.bin_combo.setCurrentIndex(index)
            switched = pump(lambda: controller.camera_bin == target, 20)
            before = controller.preview_frame_index
            flowing = pump(lambda: controller.preview_frame_index > before + 15, 20)
            info = controller.backend.info
            dark = controller.dark_reference
            dark_ok = dark is None or tuple(dark["mean"].shape) == (info.height, info.width)
            check(f"bin {target}: applied and streaming", switched and flowing, f"{info.width}x{info.height}")
            check(f"bin {target}: dark matches frame", dark_ok,
                  "no dark stored for this bin" if dark is None else f"dark {dark['mean'].shape}")
    finally:
        controller.cleanup()
        window.close()
        app.processEvents()
    print("ALL PASSED" if all(results) else "FAILURES")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
