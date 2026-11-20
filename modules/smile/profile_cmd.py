"""Where does "Analyze Smile" spend its time on this computer? (e.g. a Jetson Orin at bin 1)

Runs the same steps as the smile widget's Analyze / Apply on a saved lamp frame and prints the
seconds per step, so a slow computer shows which part is slow:

* the analysis stages (detect, trace, fit, cross-validate, remap, re-trace);
* building the model the widget keeps (``SmileCorrector.from_analysis``);
* saving / loading the model file and the live runtime's first remap;
* the GPU probes (OpenCL context, CUDA device count) by themselves - on a Jetson the first
  OpenCL / CUDA call can compile kernels for minutes;
* with ``--gui``: the widget drawing its result plots (offscreen).

Usage::

    python -m modules.smile.profile_cmd LAMP.png [--gui] [--repeat 2] [--no-gpu-probe]

Data conventions: the frame is one oriented sensor frame ``(slit rows, spectral columns)``,
read with OpenCV (8 / 16-bit PNG or TIFF), as the widget's "load image" does.

Threading: single-threaded script (the GUI step creates a QApplication).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def _t(label, fn, rows):
    t0 = time.perf_counter()
    out = fn()
    rows.append((label, time.perf_counter() - t0))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Time each step of Analyze Smile on this computer.")
    ap.add_argument("image", help="lamp frame (PNG / TIFF), oriented, the bin to test")
    ap.add_argument("--gui", action="store_true", help="also time the widget's result plots (offscreen Qt)")
    ap.add_argument("--repeat", type=int, default=1, help="run the analysis this many times (1st run includes imports)")
    ap.add_argument("--no-gpu-probe", action="store_true", help="skip the OpenCL / CUDA probes")
    ap.add_argument("--no-log", action="store_true", help="do not write debug logs to modules/smile/logs")
    a = ap.parse_args(argv)
    from . import debug_log
    if not a.no_log:
        debug_log.enable()

    rows = []
    t_start = time.perf_counter()
    import cv2
    from .analysis import analyze_smile
    from .calibration import calibration_from_analysis, load_calibration, save_calibration
    from .corrector import SmileCorrector
    from .runtime import SmileRuntime
    rows.append(("imports", time.perf_counter() - t_start))

    frame = _t("read image", lambda: cv2.imread(a.image, cv2.IMREAD_UNCHANGED), rows)
    if frame is None:
        raise SystemExit(f"cannot read {a.image}")
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"frame {frame.shape[1]} x {frame.shape[0]} {frame.dtype}; Python {sys.version.split()[0]}, "
          f"NumPy {np.__version__}, OpenCV {cv2.__version__}, {os.cpu_count()} CPUs")

    analysis = None
    for i in range(max(1, a.repeat)):
        analysis = _t(f"analyze_smile (run {i + 1})", lambda: analyze_smile(frame.astype(np.float32)), rows)
        for k, v in analysis.timings.items():
            if k != "total":
                rows.append((f"    {k}", v))
    _t("SmileCorrector.from_analysis", lambda: SmileCorrector.from_analysis(analysis), rows)
    cal = _t("calibration_from_analysis (+ CPU remap engine)", lambda: calibration_from_analysis(analysis), rows)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "smile_model_bin1.npz")
        _t("save model", lambda: save_calibration(path, cal), rows)
        _t("load model", lambda: load_calibration(path), rows)
        rt = SmileRuntime(tmp)
        _t("runtime load (set_bin)", lambda: rt.set_bin(1), rows)
        raw = np.ascontiguousarray(frame if frame.dtype in (np.uint8, np.uint16) else frame.astype(np.float32))
        _t("runtime first remap", lambda: rt.apply(raw), rows)
        _t("runtime second remap", lambda: rt.apply(raw), rows)

    if not a.no_gpu_probe:
        def ocl():
            try:
                if not cv2.ocl.haveOpenCL():
                    return "not available"
                cv2.ocl.setUseOpenCL(True)
                u = cv2.UMat(np.ones((64, 64), np.float32))
                cv2.add(u, u).get()                # forces the context (and any kernel build)
                return "available"
            except Exception as exc:          # noqa: BLE001 - report, not fail
                return f"error: {exc}"

        def cuda():
            try:
                return cv2.cuda.getCudaEnabledDeviceCount()
            except Exception as exc:          # noqa: BLE001
                return f"error: {exc}"
        r1 = _t("GPU probe: OpenCL context", ocl, rows)
        r2 = _t("GPU probe: CUDA device count", cuda, rows)
        print(f"OpenCL: {r1}; CUDA devices: {r2}")

    if a.gui:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        from .widget import SmileCorrectionWidget
        w = _t("widget create", SmileCorrectionWidget, rows)
        _t("widget set_frame", lambda: w.set_frame(frame), rows)
        _t("widget result plots (_on_analysis_done)", lambda: (w._on_analysis_done(analysis), app.processEvents()), rows)
        w.close()

    total = time.perf_counter() - t_start
    width = max(len(r[0]) for r in rows)
    print("\nstep".ljust(width + 2) + "seconds")
    for label, sec in rows:
        print(f"{label:<{width}}  {sec:8.2f}")
    print(f"{'total':<{width}}  {total:8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
