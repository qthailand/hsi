"""One acceptance task in its own process (run by ``run_acceptance.py``; not meant to be run by hand).

    python worker.py --task TASK --code-root DIR --args ARGS.json --out OUT.json

``--code-root`` is the tree whose modules are imported: the code under test, or the baseline
tree of the pixel regression (``git archive`` of the baseline revision). The configuration is
always loaded with *that tree's* ``config.load_app_config`` from the production config folder
given in the arguments, and the production camera profiles are read where that configuration
points - never written (no ``profile.json`` refresh, no folder created, no profile generated).

Tasks: ``probe`` (real camera identity and setting), ``capture`` (raw frames of the real camera at
its configured setting, for the pixel regression), ``pipeline`` (replayed raw frames through the
acquisition engine: binned frame hash per raw frame hash), ``perf`` (the acquisition loop with
basic detection on or off: count statistics, line rate, latency, dropped frames), ``stress``
(the same for minutes, per-minute series), ``scan`` (a real scan with the configured stage: only
when the operator allowed stage motion).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--code-root", required=True)
parser.add_argument("--args", required=True)
parser.add_argument("--out", required=True)
cli = parser.parse_args()
CODE = Path(cli.code_root).resolve()
sys.path.insert(0, str(CODE))
sys.path.insert(1, str(Path(__file__).resolve().parent))      # frames.py of the suite
ARGS = json.loads(Path(cli.args).read_text(encoding="utf-8"))

import numpy as np  # noqa: E402

from frames import load_replay, synthetic  # noqa: E402


def sha(a) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes() + str(a.dtype).encode() + str(a.shape).encode()).hexdigest()[:20]


def stats(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)), "max": float(v.max())}


def load_config():
    import config as app_config
    cfg = app_config.load_app_config(ARGS["config_dir"])
    if ARGS.get("backend"):
        cfg.preprocessing_backend = ARGS["backend"]
    return cfg


def camera_profile(cfg, identity):
    from modules.contracts import CameraProfile
    profile = CameraProfile.for_camera(cfg.camera_profile_folder, cfg.camera_backend, identity.get("model"),
                                       identity.get("serial"))
    return profile


def backend_setting(backend) -> dict:
    return {"bin": int(backend.current_bin), "gain": float(getattr(backend, "gain", float("nan"))),
            "exposure_ms": float(getattr(backend, "exposure_ms", float("nan")))}


def identity_of_backend(backend, cfg) -> dict:
    info = backend.info
    sensor = getattr(backend, "sensor_bit_depth", None)
    bits = int(info.bit_depth)
    sensor_bits = int(sensor()) if sensor is not None else bits
    try:
        temperature = float(backend.get_temperature())
    except Exception:                                          # noqa: BLE001
        temperature = None
    low, high = backend.exposure_range_us
    return {"backend": cfg.camera_backend, "model": backend.model_name(), "serial": backend.serial_number(),
            "bins": [int(b) for b in backend.get_bins()], **backend_setting(backend),
            "frame_shape": [int(info.height), int(info.width)], "pixel_format": str(getattr(info, "image_format", "")),
            "delivered_bits": bits, "sensor_bits": sensor_bits,
            "nominal_full_scale": ((1 << min(sensor_bits, bits)) - 1) << (bits - min(sensor_bits, bits)),
            "sensor_temperature_c": temperature if temperature is not None and temperature > -900 else None,
            "gain_range": [float(v) for v in backend.gain_range], "exposure_range_ms": [low / 1000.0, high / 1000.0],
            "source": "real camera"}


def open_real(cfg):
    from modules.camera import create_backend
    backend = create_backend(cfg)
    backend.open()
    return backend


# ── frames: replay / synthetic ───────────────────────────────────────────

def replay_frames(folder: Path):
    return load_replay(folder)


def synthetic_frames(identity: dict, level: int, count: int = 8):
    return synthetic(identity["frame_shape"], identity["delivered_bits"], identity["sensor_bits"], level, count)


def replay_camera_class():
    from modules.camera import AbstractCameraBackend

    class ReplayCamera(AbstractCameraBackend):
        """Replays raw frames with the identity of the camera they came from."""

        def __init__(self, frames, identity, fps=None):
            super().__init__()
            self.frames, self.identity, self.i, self.read = frames, identity, 0, 0
            self.gain, self.exposure_ms = identity["gain"], identity["exposure_ms"]
            self.period = 1.0 / fps if fps else 0.0
            self._next = time.perf_counter()

        def _open_device(self): pass
        def _close_device(self): pass
        def _start_device(self): pass
        def _stop_device(self): pass

        def _fetch_raw_frame(self, timeout_ms=1000):
            if self.period:
                self._next += self.period
                delay = self._next - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    self._next = time.perf_counter()
            self.i += 1
            self.read += 1
            return self.frames[self.i % len(self.frames)]

        def set_params(self, gain, exposure_ms): pass
        def frame_rate(self): return (1.0 / self.period) if self.period else None
        def set_bin(self, bin_value): pass
        info = property(lambda s: SimpleNamespace(height=s.frames[0].shape[0], width=s.frames[0].shape[1],
                                                  bit_depth=int(s.identity["delivered_bits"]),
                                                  image_format=s.identity.get("pixel_format"),
                                                  camera_id=0, pixel_size=1.0))
        exposure_range_us = property(lambda s: (1, 10_000_000))
        gain_range = property(lambda s: tuple(s.identity.get("gain_range") or (0, 1000)))
        current_bin = property(lambda s: int(s.identity["bin"]))
        def get_bins(self): return tuple(self.identity.get("bins") or (int(self.identity["bin"]),))
        def sensor_bit_depth(self): return int(self.identity["sensor_bits"])
        def model_name(self): return self.identity.get("model")
        def serial_number(self): return self.identity.get("serial")
        def dropped_images_count(self): return 0
        def get_temperature(self): return -999.0

    return ReplayCamera


def apply_production_dark(backend, cfg, profile, identity):
    """The dark the application would load for this setting (read only); ``None`` if none."""
    from modules.dark import DarkFrameLibrary, select_dark
    own, legacy = Path(profile.dark_folder), Path(cfg.dark_frame_folder)
    if not own.exists() and not legacy.exists():
        return None
    library = DarkFrameLibrary(folder=str(own if own.exists() else legacy), n_frames=cfg.dark_frame_n_frames,
                               fallback_folder=str(legacy) if own.exists() and legacy.exists() else None,
                               camera_model=profile.model, camera_serial=profile.serial)
    selection = select_dark(library, int(identity["bin"]), float(identity["exposure_ms"]), float(identity["gain"]),
                            tuple(identity["frame_shape"]), temperature=identity.get("sensor_temperature_c"))
    if selection.kind != "loaded":
        return None
    backend.set_dark_reference(selection.result["mean"], selection.result["hot_pixel_map"])
    backend.set_dark_enabled(True)
    return library.find(int(identity["bin"]), float(identity["exposure_ms"]), float(identity["gain"]),
                        temperature=identity.get("sensor_temperature_c"))


def build_engine(cfg, backend, profile, tracking=None):
    from modules.acquisition import AcquisitionEngine, AcquisitionSink
    cfg.camera_profile = profile                               # in memory, as the scan controller does
    if tracking is not None:
        cfg.saturation_tracking = bool(tracking)
    engine = AcquisitionEngine(backend, cfg, sink=AcquisitionSink(), profile=False)
    engine.set_raw_inverted(bool(getattr(cfg, "raw_inverted", True)))
    return engine


def run_engine(engine):
    thread = threading.Thread(target=engine.run_until_stopped, daemon=True)
    thread.start()
    return thread


def camera_for(cfg, identity, mode, level=None):
    """``(backend, frames_source_label)`` for ``mode`` replay / synthetic / real."""
    if mode == "real":
        return open_real(cfg), "real camera"
    if mode == "replay":
        frames = replay_frames(Path(ARGS["replay_dir"]))
    else:
        frames = synthetic_frames(identity, level or int(identity["nominal_full_scale"]))
    return replay_camera_class()(frames, identity, ARGS.get("replay_fps")), f"{mode} frames"


# ── tasks ────────────────────────────────────────────────────────────────

def task_probe():
    cfg = load_config()
    backend = open_real(cfg)
    try:
        return {"identity": identity_of_backend(backend, cfg)}
    finally:
        backend.close()


def task_capture():
    cfg = load_config()
    backend = open_real(cfg)
    out_dir = Path(ARGS["capture_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        before = backend_setting(backend)
        identity = identity_of_backend(backend, cfg)
        backend.start_stream()
        frames, deadline = [], time.time() + 30 + 4 * int(ARGS["frames"]) * identity["exposure_ms"] / 1000.0
        skip = 2
        while len(frames) < int(ARGS["frames"]) and time.time() < deadline:
            f = backend.read_frame_raw(timeout_ms=int(identity["exposure_ms"] * 3 + 2000))
            if f is None:
                continue
            if skip:
                skip -= 1
                continue
            frames.append(np.array(f, copy=True))
        backend.stop_stream()
        np.save(out_dir / "raw_frames.npy", np.stack(frames))
        (out_dir / "identity.json").write_text(json.dumps(identity, indent=1), encoding="utf-8")
        return {"frames": len(frames), "identity": identity, "setting_before": before,
                "setting_after": backend_setting(backend)}
    finally:
        backend.close()


def task_pipeline():
    cfg = load_config()
    identity = ARGS["identity"]
    profile = camera_profile(cfg, identity)
    frames = replay_frames(Path(ARGS["replay_dir"]))
    camera = replay_camera_class()(frames, identity)
    dark = apply_production_dark(camera, cfg, profile, identity)
    engine = build_engine(cfg, camera, profile)
    thread = run_engine(engine)
    try:
        time.sleep(1.5)
        engine.set_keep_spectrogram(True)
        engine.set_recording(True)
        want = max(2 * len(frames), int(ARGS.get("frames", 0)))
        end = time.time() + 60
        while engine.recorded_count() < want and time.time() < end:
            time.sleep(0.02)
        images = engine.take_recorded_frames()
        raws = engine.take_recorded_raw()
        info = engine.correction_info()
        backend_used = engine._preprocessor.backend_name
        fallback = getattr(engine._preprocessor, "fallback_reason", "")
    finally:
        engine.request_stop()
        thread.join(10)
    mapping = {}
    for raw, image in zip(raws, images):
        mapping.setdefault(sha(raw), sha(image))
    return {"hashes": mapping, "frames_recorded": len(images), "distinct_raw": len(mapping),
            "backend_requested": cfg.preprocessing_backend, "backend_used": backend_used,
            "fallback_reason": fallback, "dark_file": dark and str(dark),
            "corrections": {k: v for k, v in info.items() if k != "frame_rate_hz"}}


def _spy_counts(engine):
    calls = []
    tracker = getattr(engine, "_saturation_tracker", None)
    counter = getattr(tracker, "_counter", None)
    if counter is None:
        return calls
    original = counter.count

    def spy(frame, level):
        t = time.perf_counter_ns()
        result = original(frame, level)
        calls.append((time.perf_counter(), (time.perf_counter_ns() - t) / 1e6))
        return result

    counter.count = spy
    return calls


def task_perf():
    cfg = load_config()
    identity = ARGS["identity"]
    profile = camera_profile(cfg, identity)
    backend, source = camera_for(cfg, identity, ARGS["camera"], ARGS.get("level"))
    before = backend_setting(backend)
    dark = apply_production_dark(backend, cfg, profile, identity)
    engine = build_engine(cfg, backend, profile, tracking=False)
    if not ARGS["detection"]:                                   # basic detection OFF (test-only patch)
        engine._saturation_tracker.count = lambda frame, profiler=None: None
    calls = _spy_counts(engine)
    try:
        dropped0 = int(backend.dropped_images_count())
    except Exception:                                          # noqa: BLE001
        dropped0 = None
    thread = run_engine(engine)
    try:
        time.sleep(float(ARGS.get("warmup_s", 2.0)))
        read0 = getattr(backend, "read", None)
        engine.set_recording(True)
        start = len(calls)
        time.sleep(float(ARGS["seconds"]))
        images = engine.take_recorded_frames()
        rate = engine.line_rate_info()
        scan = engine.saturation_scan_info()
        latest = engine.latest_raw_saturation()
        read = (getattr(backend, "read", 0) - read0) if read0 is not None else None
        try:
            dropped = int(backend.dropped_images_count()) - dropped0 if dropped0 is not None else None
        except Exception:                                      # noqa: BLE001
            dropped = None
        camera_hz = backend.frame_rate() if hasattr(backend, "frame_rate") else None
        backend_used = engine._preprocessor.backend_name
    finally:
        engine.request_stop()
        thread.join(15)
    after = backend_setting(backend)
    return {"source": source, "detection": bool(ARGS["detection"]), "recorded": len(images),
            "frames_read": read, "dropped_frames": dropped, "camera_hz": camera_hz,
            "line_rate_hz": rate.get("line_rate_hz"), "frames_missing": rate.get("frames_missing"),
            "line_rate_source": rate.get("line_rate_source"),
            "frame_latency_ms": {"mean": scan.get("busy_ms"), "max": scan.get("busy_max_ms")},
            "count_ms": stats([ms for _, ms in calls[start:]]),
            "raw_status": None if latest is None else latest.status,
            "raw_reason": None if latest is None else latest.reason,
            "level": None if latest is None else latest.level,
            "profile_id": None if latest is None else latest.profile_id,
            "scan_raw": {k: scan.get(k) for k in ("raw_status", "raw_checked", "raw_saturated_frames",
                                                   "raw_peak_clipped_percent")},
            "backend_requested": cfg.preprocessing_backend, "backend_used": backend_used,
            "dark_file": dark and str(dark), "setting_before": before, "setting_after": after}


def task_stress():
    cfg = load_config()
    identity = ARGS["identity"]
    profile = camera_profile(cfg, identity)
    backend, source = camera_for(cfg, identity, ARGS["camera"], ARGS.get("level"))
    before = backend_setting(backend)
    apply_production_dark(backend, cfg, profile, identity)
    engine = build_engine(cfg, backend, profile, tracking=False)
    calls = _spy_counts(engine)
    thread = run_engine(engine)
    errors = []
    try:
        time.sleep(2.0)
        t0, start = time.perf_counter(), len(calls)
        engine.reset_saturation_peak()
        duration = float(ARGS["seconds"])
        while time.perf_counter() - t0 < duration:
            time.sleep(1.0)
        latest = engine.latest_raw_saturation()
        backend_used = engine._preprocessor.backend_name
    except Exception as exc:                                   # noqa: BLE001
        errors.append(str(exc))
        latest, backend_used = None, None
    finally:
        engine.request_stop()
        thread.join(15)
    series, window = [], float(ARGS.get("window_s", 60.0))
    samples = calls[start:]
    if samples:
        first = samples[0][0]
        buckets = {}
        for t, ms in samples:
            buckets.setdefault(int((t - first) // window), []).append(ms)
        full = int((samples[-1][0] - first) // window)          # complete windows only: a partial last
        if full >= 1:                                          # window would read as a frame-rate drop
            buckets = {k: v for k, v in buckets.items() if k < full}
        for k in sorted(buckets):
            st = stats(buckets[k])
            st.update(window=k, fps=len(buckets[k]) / window)
            series.append(st)
    return {"source": source, "count_ms": stats([ms for _, ms in samples]), "series": series,
            "frames": len(samples), "errors": errors,
            "raw_status": None if latest is None else latest.status,
            "checked_frames": None if latest is None else latest.checked_frames,
            "backend_used": backend_used, "setting_before": before, "setting_after": backend_setting(backend)}


def task_scan():
    """A real scan through the application's controller (stage motion allowed by the operator)."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtCore import QEventLoop, QTimer
    from PyQt5.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication([])
    for name in ("critical", "warning", "information", "question"):
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.Yes))
    from controllers.scan_controller import ScanController
    from windows.scan_window import ScanWindow

    def spin(seconds, until=None):
        end = time.time() + seconds
        while time.time() < end:
            loop = QEventLoop()
            QTimer.singleShot(50, loop.quit)
            loop.exec_()
            if until is not None and until():
                return True
        return until is None

    cfg = load_config()
    win = ScanWindow(cfg)
    controller = ScanController(win, cfg, persist_settings=False)
    try:
        spin(60, until=lambda: controller.preview_frame_index >= 3)
        before = backend_setting(controller.backend)
        controller.start_scan()
        started = controller.state == controller.SCANNING
        spin(float(ARGS.get("timeout_s", 600)), until=lambda: controller.state != controller.SCANNING)
        spin(120, until=lambda: controller.stage is None)
        spin(2)
        header = dict(controller.last_scan_header)
        summary = (controller.last_scan_saturation or {}).get("summary", {})
        return {"started": started, "header": {k: v for k, v in header.items() if k.startswith("saturation")
                                               or k in ("camera_bin", "exposure_ms", "gain", "line_rate_hz")},
                "full_tracking": summary, "setting_before": before,
                "setting_after": backend_setting(controller.backend)}
    finally:
        controller.cleanup()
        spin(2)


TASKS = {"probe": task_probe, "capture": task_capture, "pipeline": task_pipeline, "perf": task_perf,
         "stress": task_stress, "scan": task_scan}

if __name__ == "__main__":
    try:
        result = {"ok": True, **TASKS[cli.task]()}
    except Exception as exc:                                   # noqa: BLE001 - reported by the runner
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    Path(cli.out).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
