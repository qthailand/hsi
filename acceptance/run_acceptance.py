"""HSI automated acceptance test suite: the single entry point (Windows and Linux ARM64 / Jetson).

    python acceptance/run_acceptance.py [--camera real|replay|synthetic] [--mode quick|stress] [...]

The suite *checks* the system as the operator left it; it never prepares it. It loads the
production configuration and camera profiles with the application's own mechanisms, uses the
calibration profiles made with the production application on this machine, never creates,
migrates or overwrites a profile, hashes every calibration and configuration file before and after,
moves no stage unless ``--allow-stage-motion`` is given, and leaves the camera as it found it.

Sections of the report:

* **integrity** - environment / identity, profile and configuration hashes before vs after, camera
  setting restored;
* **synthetic** - the existing unit tests (synthetic data only, no production profile);
* **production** - processing backend (and CPU fallback), saturation profile in use, basic detection
  ON / OFF, pixel-wise regression against the code before Phase 2E on the same backend, long stress
  run, and (only when allowed) a real scan with the stage.

See ``acceptance/README.md``. Exit code: 0 when nothing FAILed, 1 otherwise, 2 for a usage error.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ACCEPTANCE_DIR, FAIL, INTEGRITY, PASS, PRODUCTION, PROJECT_ROOT, SKIPPED, SYNTHETIC,  # noqa: E402
                    TestResult, diff_hashes, git, hash_tree, now_iso, platform_identity, read_json,
                    software_identity, subprocess_env, write_json)
from monitor import SystemMonitor  # noqa: E402
from report import write_reports  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))

#: Synthetic unit-test files run when present (``tests/``); others with ``--unit-tests``.
DEFAULT_UNIT_TESTS = ("test_saturation_profile.py", "test_saturation_profile_engine.py",
                      "test_saturation_measurement.py", "test_saturation_tracking.py",
                      "test_basic_saturation_detection.py", "test_saturation_mask.py", "test_acquisition.py",
                      "test_preprocessing.py", "test_preprocessing_binned.py", "test_scan_header_metadata.py",
                      "test_envi_roundtrip.py", "test_module_contract.py")
#: The code the pixel regression compares with by default: the last commit before Phase 2E.
DEFAULT_BASELINE = "41f330b"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", choices=("real", "replay", "synthetic"), default="real",
                   help="frame source of the production tests (default: real)")
    p.add_argument("--replay-dir", help="folder of raw *.npy frames (+ identity.json) for --camera replay")
    p.add_argument("--replay-fps", type=float, help="pace replay/synthetic frames (default: free-running)")
    p.add_argument("--profile", help="camera profile folder (<model>_<serial>) for replay/synthetic without "
                                     "identity.json; default: the only one of the configured backend")
    p.add_argument("--mode", choices=("quick", "stress"), default="quick")
    p.add_argument("--seconds", type=float, help="duration of each ON / OFF run (quick 5 s, stress 30 s)")
    p.add_argument("--stress-minutes", type=float, default=30.0, help="stress-run length (mode stress)")
    p.add_argument("--backend", help="preprocessing backend to request (default: the configuration's)")
    p.add_argument("--expect-backend", help="FAIL unless this backend actually runs (e.g. cupy_cuda on Jetson)")
    p.add_argument("--baseline-ref", default=DEFAULT_BASELINE, help="git revision of the pixel-regression baseline")
    p.add_argument("--skip-unit", action="store_true", help="skip the synthetic unit tests")
    p.add_argument("--skip-regression", action="store_true", help="skip the pixel-wise regression")
    p.add_argument("--unit-tests", nargs="*", help="unit-test files (in tests/) instead of the default list")
    p.add_argument("--allow-stage-motion", action="store_true",
                   help="also run a real scan with the configured stage (it MOVES the stage)")
    p.add_argument("--config-dir", help="configuration folder (default: the application's config/)")
    p.add_argument("--output", help="report folder (default: acceptance_reports/<time>_<host>)")
    p.add_argument("--count-mean-ms", type=float, default=0.5, help="acceptable mean count time (ms)")
    p.add_argument("--count-p99-ms", type=float, default=1.0, help="upper limit of the count P99 (ms)")
    p.add_argument("--fps-ratio", type=float, default=0.98, help="min line rate ON / OFF (paced cameras)")
    p.add_argument("--capture-frames", type=int, default=16, help="raw frames captured for the regression")
    return p.parse_args(argv)


class Runner:
    """Runs the tests in order and collects :class:`TestResult` s (see the module docstring)."""

    def __init__(self, args):
        self.args = args
        self.quick = args.mode == "quick"
        self.seconds = args.seconds or (5.0 if self.quick else 30.0)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        import platform
        self.out = Path(args.output) if args.output else PROJECT_ROOT / "acceptance_reports" / f"{stamp}_{platform.node()}"
        self.out.mkdir(parents=True, exist_ok=True)
        ignore = self.out.parent / ".gitignore"
        if not args.output and not ignore.exists():
            ignore.write_text("*\n", encoding="utf-8")         # reports never end up in git
        self.work = self.out / "work"
        self.work.mkdir(exist_ok=True)
        self.tests = []
        self.config_dir = Path(args.config_dir).resolve() if args.config_dir else PROJECT_ROOT / "config"
        self.identity: Optional[dict] = None
        self.readiness: Dict = {}
        self.backend_info: Dict = {}
        self.workers = 0

    # ── helpers ─────────────────────────────────────────────

    def add(self, test_id, category, status, reason, metrics=None, details=None, duration=0.0):
        result = TestResult(test_id, category, status, reason, metrics or {}, details or {}, duration)
        self.tests.append(result)
        mark = {PASS: "PASS", FAIL: "FAIL", SKIPPED: "SKIP"}[status]
        print(f"  [{mark:4s}] {test_id}: {reason}", flush=True)
        return result

    def worker(self, task, args, code_root=PROJECT_ROOT, timeout=None, monitor=True):
        """Run one worker task; returns ``(result, monitor summary, seconds)``."""
        self.workers += 1
        args = {"config_dir": str(self.config_dir), **args}
        args_path = self.work / f"{self.workers:02d}_{task}_args.json"
        out_path = self.work / f"{self.workers:02d}_{task}_out.json"
        write_json(args_path, args)
        cmd = [sys.executable, str(ACCEPTANCE_DIR / "worker.py"), "--task", task, "--code-root", str(code_root),
               "--args", str(args_path), "--out", str(out_path)]
        t0 = time.time()
        log = open(self.work / f"{self.workers:02d}_{task}.log", "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(code_root), env=subprocess_env(), stdout=log, stderr=subprocess.STDOUT)
        mon = SystemMonitor(1.0).start(proc.pid) if monitor else None
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        finally:
            log.close()
        summary = mon.stop() if mon else {}
        result = read_json(out_path) if out_path.exists() else {"ok": False, "error": f"worker ended with code "
                                                                                    f"{proc.returncode}, no result"}
        return result, summary, time.time() - t0

    def config(self):
        import config as app_config
        return app_config.load_app_config(str(self.config_dir))

    def hashes(self, cfg) -> Dict[str, Dict[str, str]]:
        folders = {"camera_profiles": Path(cfg.camera_profile_folder), "config": self.config_dir,
                   "legacy_dark": Path(cfg.dark_frame_folder), "legacy_smile": Path(cfg.smile_map_folder)}
        legacy_wavelength = PROJECT_ROOT / "calibration"
        folders["legacy_calibration"] = legacy_wavelength
        return {name: hash_tree(path) for name, path in folders.items()}

    # ── identity ────────────────────────────────────────────

    def find_identity(self, cfg):
        args = self.args
        if args.camera == "real":
            result, _, _ = self.worker("probe", {}, timeout=120, monitor=False)
            if not result.get("ok"):
                return None, f"real camera not available: {result.get('error')}"
            return result["identity"], "real camera"
        if args.camera == "replay":
            if not args.replay_dir:
                return None, "--replay-dir is required for --camera replay"
            ident = Path(args.replay_dir) / "identity.json"
            if ident.exists():
                return read_json(ident), f"identity.json of {args.replay_dir}"
        return self.identity_from_profile(cfg)

    def identity_from_profile(self, cfg):
        """The configured camera setting, with the camera named by its profile (replay / synthetic)."""
        backend_root = Path(cfg.camera_profile_folder) / cfg.camera_backend
        folders = sorted(p for p in backend_root.glob("*") if (p / "profile.json").exists()) if backend_root.exists() else []
        if self.args.profile:
            folders = [p for p in folders if p.name == self.args.profile]
        if len(folders) != 1:
            return None, (f"{len(folders)} camera profiles under {backend_root}: name one with --profile"
                          if folders or self.args.profile else f"no camera profile under {backend_root}")
        info = read_json(folders[0] / "profile.json")
        camera_bin = int(cfg.default_bin)
        shape, bits = None, None
        try:                                                   # the frame size the bin's Smile model was made for
            import config as app_config
            from modules.contracts import CameraProfile, CameraSetting
            cfg.camera_profile = CameraProfile.for_camera(cfg.camera_profile_folder, cfg.camera_backend,
                                                          info.get("model"), info.get("serial"))
            smile = app_config.calibration_store(cfg).find_smile(
                CameraSetting(cfg.camera_backend, int(cfg.camera_index), camera_bin, bool(cfg.raw_inverted)))
            if smile:
                from modules.smile.calibration import load_calibration
                shape = list(load_calibration(str(smile)).image_shape)
            for path in sorted((folders[0] / "saturation").glob(f"saturation_*_bin{camera_bin}.json")):
                data = read_json(path)
                bits = data["identity"]
                break
        except Exception as exc:                               # noqa: BLE001 - reported below
            return None, f"camera setting of {folders[0].name} unknown: {exc}"
        if shape is None and int(info.get("camera_bin", -1)) == camera_bin:
            shape = [int(info["frame_height"]), int(info["frame_width"])]
        if shape is None or bits is None:
            return None, (f"frame size / pixel format of bin {camera_bin} unknown for {folders[0].name} "
                          f"(no Smile model / saturation profile for it): use --camera real or "
                          f"--replay-dir with identity.json")
        return {"backend": cfg.camera_backend, "model": info.get("model"), "serial": info.get("serial"),
                "bins": [camera_bin], "bin": camera_bin, "gain": float(cfg.default_gain),
                "exposure_ms": float(cfg.default_exposure_ms), "frame_shape": shape,
                "pixel_format": bits["pixel_format"], "delivered_bits": int(bits["delivered_bits"]),
                "sensor_bits": int(bits["sensor_bits"]),
                "nominal_full_scale": ((1 << int(bits["sensor_bits"])) - 1) << (int(bits["delivered_bits"]) -
                                                                                int(bits["sensor_bits"])),
                "sensor_temperature_c": None, "gain_range": None,
                "source": f"camera profile {folders[0].name} + configured setting"}, f"profile {folders[0].name}"

    # ── tests ───────────────────────────────────────────────

    def unit_tests(self):
        tests_dir = PROJECT_ROOT / "tests"
        if self.args.skip_unit:
            self.add("synthetic.unit", SYNTHETIC, SKIPPED, "skipped (--skip-unit)")
            return
        if not tests_dir.exists():
            self.add("synthetic.unit", SYNTHETIC, SKIPPED, f"no {tests_dir} on this machine")
            return
        try:
            import pytest  # noqa: F401
        except ImportError:
            self.add("synthetic.unit", SYNTHETIC, SKIPPED, "pytest is not installed")
            return
        names = self.args.unit_tests or DEFAULT_UNIT_TESTS
        for name in names:
            path = tests_dir / name
            test_id = f"synthetic.unit.{Path(name).stem}"
            if not path.exists():
                self.add(test_id, SYNTHETIC, SKIPPED, f"{name} not present")
                continue
            t0 = time.time()
            env = subprocess_env()
            env["TEMP"] = env["TMP"] = str(self.work)
            out = subprocess.run([sys.executable, "-m", "pytest", str(path), "-q", "-p", "no:cacheprovider"],
                                 cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True, timeout=1800)
            tail = [line for line in out.stdout.splitlines() if " passed" in line or " failed" in line
                    or " error" in line]
            summary = tail[-1].strip() if tail else (out.stdout.strip().splitlines() or ["no output"])[-1]
            failed = [line for line in out.stdout.splitlines() if line.startswith("FAILED")]
            status = PASS if out.returncode == 0 else (SKIPPED if out.returncode == 5 else FAIL)
            self.add(test_id, SYNTHETIC, status, summary, details={"failed": failed}, duration=time.time() - t0)

    def blocked(self, test_id, needs):
        missing = [f"{k}: {self.readiness[k]['reason']}" for k in needs
                   if k in self.readiness and self.readiness[k]["state"] != "READY"]
        if missing:
            self.add(test_id, PRODUCTION, SKIPPED, "profile not ready - " + "; ".join(missing))
            return True
        return False

    def frames_for_regression(self, cfg):
        """Raw frames both trees replay: given replay frames, frames captured now, or synthetic ones."""
        args = self.args
        if args.camera == "replay":
            return Path(args.replay_dir), "replay frames"
        if args.camera == "real":
            capture = self.work / "captured"
            result, _, _ = self.worker("capture", {"capture_dir": str(capture), "frames": args.capture_frames},
                                       timeout=600, monitor=False)
            if not result.get("ok"):
                return None, f"capture failed: {result.get('error')}"
            return capture, f"{result['frames']} raw frames captured from the camera"
        from frames import save_frames, synthetic
        level = (self.readiness.get("saturation", {}).get("revision") or {}).get("level") \
            or self.identity["nominal_full_scale"]
        folder = self.work / "synthetic"
        save_frames(synthetic(self.identity["frame_shape"], self.identity["delivered_bits"],
                              self.identity["sensor_bits"], level), folder)
        return folder, "synthetic frames"

    def pixel_regression(self, cfg):
        test_id = "production.pixel_regression"
        if self.args.skip_regression:
            self.add(test_id, PRODUCTION, SKIPPED, "skipped (--skip-regression)")
            return
        if self.blocked(test_id, ("camera_profile",)):
            return
        ref = self.args.baseline_ref
        sha = git("rev-parse", "--verify", f"{ref}^{{commit}}")
        if sha is None:
            self.add(test_id, PRODUCTION, SKIPPED, f"baseline {ref} not available (git missing or revision unknown)")
            return
        frames_dir, source = self.frames_for_regression(cfg)
        if frames_dir is None:
            self.add(test_id, PRODUCTION, SKIPPED, source)
            return
        t0 = time.time()
        baseline = Path(tempfile.mkdtemp(prefix="hsi_baseline_"))
        try:
            archive = subprocess.run(["git", "archive", "--format=tar", sha], cwd=str(PROJECT_ROOT),
                                     capture_output=True, timeout=300)
            if archive.returncode != 0:
                self.add(test_id, PRODUCTION, SKIPPED, f"git archive {ref} failed")
                return
            tar_path = baseline / "baseline.tar"
            tar_path.write_bytes(archive.stdout)
            with tarfile.open(tar_path) as tar:
                try:
                    tar.extractall(baseline / "code", filter="data")   # no absolute paths / links outside
                except TypeError:                                   # Python without extraction filters
                    tar.extractall(baseline / "code")
            args = {"replay_dir": str(frames_dir), "identity": self.identity, "backend": self.args.backend}
            new, _, _ = self.worker("pipeline", args, timeout=900)
            old, _, _ = self.worker("pipeline", args, code_root=baseline / "code", timeout=900)
        finally:
            shutil.rmtree(baseline, ignore_errors=True)
        if not new.get("ok") or not old.get("ok"):
            self.add(test_id, PRODUCTION, FAIL, f"pipeline run failed: {new.get('error') or old.get('error')}",
                     details={"new": new, "baseline": old}, duration=time.time() - t0)
            return
        common = sorted(set(new["hashes"]) & set(old["hashes"]))
        same = sum(new["hashes"][k] == old["hashes"][k] for k in common)
        metrics = {"frames_compared": len(common), "frames_identical": same}
        details = {"baseline": f"{ref} ({sha[:12]})", "source": source, "backend_new": new["backend_used"],
                   "backend_baseline": old["backend_used"], "corrections": new.get("corrections"),
                   "dark_file": new.get("dark_file")}
        if new["backend_used"] != old["backend_used"]:
            self.add(test_id, PRODUCTION, FAIL, f"not the same backend: {new['backend_used']} vs "
                                                f"{old['backend_used']}", metrics, details, time.time() - t0)
        elif not common:
            self.add(test_id, PRODUCTION, FAIL, "no frame recorded by both runs", metrics, details, time.time() - t0)
        elif same == len(common):
            self.add(test_id, PRODUCTION, PASS, f"{same}/{len(common)} binned frames pixel-identical to {ref} on "
                                                f"{new['backend_used']} ({source})", metrics, details, time.time() - t0)
        else:
            self.add(test_id, PRODUCTION, FAIL, f"{len(common) - same}/{len(common)} binned frames differ from {ref}",
                     metrics, details, time.time() - t0)

    def run_seconds(self) -> float:
        """Length of each ON / OFF run: ``--seconds``, else the mode's default, long enough for at
        least 40 frames of a paced camera (a real camera at its exposure)."""
        if self.args.seconds:
            return float(self.args.seconds)
        frame_s = 0.0
        if self.args.camera == "real":
            frame_s = float(self.identity.get("exposure_ms") or 0) / 1000.0
        elif self.args.replay_fps:
            frame_s = 1.0 / self.args.replay_fps
        return min(300.0, max(self.seconds, 40 * frame_s))

    def detection_on_off(self):
        test_id = "production.detection_on_off"
        if self.blocked(test_id, ("camera_profile", "saturation")):
            return None
        args = self.args
        level = (self.readiness["saturation"].get("revision") or {}).get("level")
        seconds = self.run_seconds()
        common = {"identity": self.identity, "camera": args.camera, "seconds": seconds, "level": level,
                  "replay_dir": args.replay_dir, "replay_fps": args.replay_fps, "backend": args.backend}
        t0 = time.time()
        on, mon_on, _ = self.worker("perf", {**common, "detection": True}, timeout=seconds * 4 + 300)
        off, mon_off, _ = self.worker("perf", {**common, "detection": False}, timeout=seconds * 4 + 300)
        if not on.get("ok") or not off.get("ok"):
            self.add(test_id, PRODUCTION, FAIL, f"run failed: {on.get('error') or off.get('error')}",
                     details={"on": on, "off": off}, duration=time.time() - t0)
            return on
        self.record_backend(on)
        count = on["count_ms"]
        fps_on, fps_off = on.get("line_rate_hz"), off.get("line_rate_hz")
        metrics = {"count_mean_ms": count.get("mean"), "count_p95_ms": count.get("p95"),
                   "count_p99_ms": count.get("p99"), "count_max_ms": count.get("max"), "fps": fps_on,
                   "fps_off": fps_off, "fps_ratio": (fps_on / fps_off) if fps_on and fps_off else None,
                   "dropped_frames": on.get("dropped_frames"),
                   "frame_latency_mean_ms": on["frame_latency_ms"].get("mean"),
                   "frame_latency_max_ms": on["frame_latency_ms"].get("max"), **self.monitor_metrics(mon_on)}
        details = {"on": on, "off": off, "monitor_on": mon_on, "monitor_off": mon_off}
        problems = []
        if on.get("raw_status") != "checked":
            self.add(test_id, PRODUCTION, SKIPPED, f"frames not checked: {on.get('raw_reason')}", metrics, details,
                     time.time() - t0)
            return on
        if count.get("n", 0) == 0:
            problems.append("no frame counted")
        elif count["n"] < 30:
            problems.append(f"only {count['n']} frames counted (use --seconds for a longer run)")
        if (count.get("mean") or 0) > args.count_mean_ms:
            problems.append(f"count mean {count['mean']:.3f} > {args.count_mean_ms} ms")
        if (count.get("p99") or 0) > args.count_p99_ms:
            problems.append(f"count P99 {count['p99']:.3f} > {args.count_p99_ms} ms")
        paced = args.camera == "real" or args.replay_fps
        if paced:
            if metrics["fps_ratio"] is not None and metrics["fps_ratio"] < args.fps_ratio:
                problems.append(f"line rate ON/OFF {metrics['fps_ratio']:.3f} < {args.fps_ratio}")
            if (on.get("dropped_frames") or 0) > (off.get("dropped_frames") or 0):
                problems.append(f"dropped frames ON {on.get('dropped_frames')} > OFF {off.get('dropped_frames')}")
            if on.get("frames_missing"):
                problems.append(f"{on['frames_missing']} frames missing (camera timestamps)")
        reason = (f"count mean {count['mean']:.3f} / P99 {count['p99']:.3f} ms; line rate ON {fps_on or 0:.1f} Hz, "
                  f"OFF {fps_off or 0:.1f} Hz" + ("" if paced else " (free-running: line rate informational)"))
        self.add(test_id, PRODUCTION, FAIL if problems else PASS, "; ".join(problems) or reason, metrics, details,
                 time.time() - t0)
        return on

    def saturation_in_use(self, on):
        test_id = "production.saturation_profile_in_use"
        if self.blocked(test_id, ("saturation",)):
            return
        expected = (self.readiness["saturation"].get("revision") or {}).get("profile_id")
        if on is None or not on.get("ok"):
            self.add(test_id, PRODUCTION, SKIPPED, "no acquisition run to check")
            return
        got = on.get("profile_id")
        if on.get("raw_status") == "checked" and got == expected:
            self.add(test_id, PRODUCTION, PASS, f"engine counts against level {on.get('level')} of {got}",
                     details={"expected": expected, "engine": got})
        else:
            self.add(test_id, PRODUCTION, FAIL, f"engine status {on.get('raw_status')} ({on.get('raw_reason')}), "
                                                f"profile {got} != {expected}")

    def record_backend(self, run):
        requested = run.get("backend_requested")
        used = run.get("backend_used")
        self.backend_info = {"requested": requested, "used": used,
                             "cpu_fallback": used == "cpu" and requested not in (None, "cpu"),
                             "reason": run.get("fallback_reason")}

    def backend_test(self):
        test_id = "production.processing_backend"
        info = self.backend_info
        if not info:
            self.add(test_id, PRODUCTION, SKIPPED, "no acquisition run reported its backend")
            return
        text = f"requested {info['requested']}, actual {info['used']}"
        if info["cpu_fallback"]:
            text += " - CPU FALLBACK"
        expect = self.args.expect_backend
        if expect and info["used"] != expect:
            self.add(test_id, PRODUCTION, FAIL, f"{text}; expected {expect}", details=info)
        else:
            self.add(test_id, PRODUCTION, PASS, text + ("" if expect else " (no --expect-backend)"), details=info)

    def stress(self):
        test_id = "production.stress"
        if self.quick:
            self.add(test_id, PRODUCTION, SKIPPED, "quick mode (run with --mode stress)")
            return
        if self.blocked(test_id, ("camera_profile", "saturation")):
            return
        args = self.args
        seconds = args.stress_minutes * 60.0
        window = 60.0 if seconds >= 300 else max(10.0, seconds / 5)
        level = (self.readiness["saturation"].get("revision") or {}).get("level")
        t0 = time.time()
        run, mon, _ = self.worker("stress", {"identity": self.identity, "camera": args.camera, "seconds": seconds,
                                             "window_s": window, "level": level, "replay_dir": args.replay_dir,
                                             "replay_fps": args.replay_fps, "backend": args.backend},
                                  timeout=seconds + 600)
        if not run.get("ok"):
            self.add(test_id, PRODUCTION, FAIL, f"run failed: {run.get('error')}", details=run, duration=time.time() - t0)
            return
        series = [w for w in run["series"] if w.get("n")]
        count = run["count_ms"]
        metrics = {"count_mean_ms": count.get("mean"), "count_p95_ms": count.get("p95"),
                   "count_p99_ms": count.get("p99"), "count_max_ms": count.get("max"),
                   "fps": (sum(w["fps"] for w in series) / len(series)) if series else None, **self.monitor_metrics(mon)}
        problems = list(run.get("errors") or [])
        if not series:
            problems.append("no frame counted")
        else:
            if (count.get("p99") or 0) > args.count_p99_ms:
                problems.append(f"count P99 {count['p99']:.3f} > {args.count_p99_ms} ms")
            fps = sorted(w["fps"] for w in series)
            median = fps[len(fps) // 2]
            if median and fps[0] < 0.9 * median:
                problems.append(f"line rate fell to {fps[0]:.1f} Hz (median {median:.1f})")
            first, worst = series[0]["mean"], max(w["mean"] for w in series)
            if first and worst > 1.5 * first and worst > args.count_mean_ms:
                problems.append(f"count mean rose from {first:.3f} to {worst:.3f} ms")
        growth = self.rss_growth(mon, skip_s=max(30.0, 0.1 * seconds))
        metrics["proc_rss_growth_mb"] = growth
        allowed = 50.0 + args.stress_minutes
        if growth is not None and growth > allowed:
            problems.append(f"process memory grew {growth:.0f} MB (> {allowed:.0f})")
        reason = (f"{args.stress_minutes:g} min, {run['frames']} frames, {len(series)} windows: count P99 "
                  f"{count.get('p99', 0):.3f} ms, memory growth {growth if growth is not None else 0:.0f} MB")
        self.add(test_id, PRODUCTION, FAIL if problems else PASS, "; ".join(problems) or reason, metrics,
                 {"run": run, "monitor": mon}, time.time() - t0)

    def stage_scan(self):
        test_id = "production.scan_with_stage"
        if not self.args.allow_stage_motion:
            self.add(test_id, PRODUCTION, SKIPPED, "stage motion not allowed (pass --allow-stage-motion to scan)")
            return
        if self.args.camera != "real":
            self.add(test_id, PRODUCTION, SKIPPED, "needs --camera real")
            return
        if self.blocked(test_id, ("camera_profile", "dark", "smile", "wavelength", "saturation")):
            return
        t0 = time.time()
        run, mon, _ = self.worker("scan", {"timeout_s": 900}, timeout=1500)
        if not run.get("ok"):
            self.add(test_id, PRODUCTION, FAIL, f"scan failed: {run.get('error')}", details=run, duration=time.time() - t0)
            return
        header, tracking = run["header"], run.get("full_tracking") or {}
        problems = []
        if not run.get("started"):
            problems.append("the scan did not start")
        if header.get("saturation_profile_status") != "verified":
            problems.append(f"header saturation status {header.get('saturation_profile_status')}")
        if header.get("saturation_unchecked_frames"):
            problems.append(f"{header['saturation_unchecked_frames']} lines not checked")
        if tracking.get("status") == "checked" and tracking.get("clipped_frames") != header.get("saturation_clipped_frames"):
            problems.append(f"full tracking {tracking.get('clipped_frames')} vs basic "
                            f"{header.get('saturation_clipped_frames')} clipped lines")
        if run["setting_before"] != run["setting_after"]:
            problems.append(f"camera setting changed {run['setting_before']} -> {run['setting_after']}")
        self.add(test_id, PRODUCTION, FAIL if problems else PASS,
                 "; ".join(problems) or f"{header.get('saturation_checked_frames')} lines checked, "
                                        f"{header.get('saturation_clipped_frames')} clipped, level "
                                        f"{header.get('saturation_level')}", self.monitor_metrics(mon),
                 {"run": run}, time.time() - t0)

    @staticmethod
    def rss_growth(mon: Dict, skip_s: float):
        """Process memory growth after the start-up (``skip_s`` seconds: imports, JIT, buffers), MB."""
        series = [s for s in mon.get("series", []) if s.get("proc_rss_mb") is not None]
        if len(series) < 2:
            return None
        start = series[0]["t"] + skip_s
        steady = [s for s in series if s["t"] >= start] or series[-2:]
        return steady[-1]["proc_rss_mb"] - steady[0]["proc_rss_mb"] if len(steady) >= 2 else None

    @staticmethod
    def monitor_metrics(mon: Dict) -> Dict:
        temps = mon.get("temps_max") or {}
        return {"cpu_pct_mean": mon.get("cpu_pct_mean"), "proc_cpu_pct_mean": mon.get("proc_cpu_pct_mean"),
                "gpu_load_pct_mean": mon.get("gpu_load_pct_mean"), "ram_used_mb_max": mon.get("ram_used_mb_max"),
                "proc_rss_mb_max": mon.get("proc_rss_mb_max"), "proc_rss_growth_mb": mon.get("proc_rss_growth_mb"),
                "temp_max_c": max(temps.values()) if temps else None,
                "cpu_freq_mhz_mean": mon.get("cpu_freq_mhz_mean"), "gpu_freq_mhz_mean": mon.get("gpu_freq_mhz_mean")}

    # ── the run ─────────────────────────────────────────────

    def run(self) -> int:
        args = self.args
        started = now_iso()
        print(f"HSI acceptance - {args.mode}, camera {args.camera}; report: {self.out}", flush=True)
        config_before = hash_tree(self.config_dir)            # before loading: a repaired file counts as a change
        try:
            cfg = self.config()
        except Exception as exc:                               # noqa: BLE001 - nothing can run
            print(f"Configuration not loadable from {self.config_dir}: {exc}")
            return 2
        before = self.hashes(cfg)
        before["config"] = config_before
        self.identity, identity_source = self.find_identity(cfg)
        configuration = {k: getattr(cfg, k, None) for k in (
            "camera_backend", "camera_index", "default_bin", "default_gain", "default_exposure_ms",
            "preprocessing_backend", "resolution", "resolution_source", "raw_inverted", "saturation_tracking",
            "stage_type", "camera_profile_folder")}
        configuration["config_dir"] = str(self.config_dir)
        configuration["config_files"] = before["config"]
        print("- integrity", flush=True)
        self.add("integrity.environment", INTEGRITY, PASS, f"{platform_identity()['system']} "
                 f"{platform_identity()['machine']}, git {software_identity()['git_revision'][:12]}")
        if self.identity is None:
            self.add("integrity.camera_identity", INTEGRITY, FAIL if args.camera == "real" else SKIPPED, identity_source)
        else:
            self.add("integrity.camera_identity", INTEGRITY, PASS,
                     f"{self.identity.get('model')} {self.identity.get('serial')} bin {self.identity['bin']}, gain "
                     f"{self.identity['gain']:g}, {self.identity['exposure_ms']:g} ms ({identity_source})")
            from profiles import check_profiles
            self.readiness = {k: v.to_dict() for k, v in check_profiles(cfg, self.identity).items()}
            not_ready = {k: v["reason"] for k, v in self.readiness.items() if v["state"] not in ("READY", "NOT USED")}
            self.add("integrity.profile_readiness", INTEGRITY, PASS if not not_ready else SKIPPED,
                     "all profiles ready" if not not_ready else
                     "not ready (dependent tests skipped): " + "; ".join(f"{k}: {v}" for k, v in not_ready.items()))
        print("- synthetic unit tests", flush=True)
        self.unit_tests()
        print("- production tests", flush=True)
        if self.identity is None:
            for test_id in ("production.processing_backend", "production.saturation_profile_in_use",
                            "production.detection_on_off", "production.pixel_regression", "production.stress",
                            "production.scan_with_stage"):
                self.add(test_id, PRODUCTION, SKIPPED, f"no camera identity: {identity_source}")
        else:
            on = self.detection_on_off()
            self.saturation_in_use(on)
            self.backend_test()
            self.pixel_regression(cfg)
            self.stress()
            self.stage_scan()
        print("- integrity after", flush=True)
        after = self.hashes(cfg)
        changes = {name: diff_hashes(before[name], after[name]) for name in before}
        touched = {name: c for name, c in changes.items() if any(c.values())}
        self.add("integrity.profiles_unchanged", INTEGRITY, FAIL if touched else PASS,
                 "files changed: " + str(touched) if touched else
                 f"{sum(len(v) for v in before.values())} calibration / configuration files byte-identical",
                 details={"before": before, "changes": changes})
        restore_id = "integrity.camera_restored"
        if args.camera == "real" and self.identity is not None:
            probe, _, _ = self.worker("probe", {}, timeout=120, monitor=False)
            keys = ("bin", "gain", "exposure_ms")
            if not probe.get("ok"):
                self.add(restore_id, INTEGRITY, FAIL, f"camera not reachable after the tests: {probe.get('error')}")
            else:
                was = {k: self.identity[k] for k in keys}
                now = {k: probe["identity"][k] for k in keys}
                self.add(restore_id, INTEGRITY, PASS if was == now else FAIL,
                         f"camera setting {now}" + ("" if was == now else f" != before {was}"))
        else:
            self.add(restore_id, INTEGRITY, SKIPPED, "no real camera used")
        verdict = FAIL if any(t.status == FAIL for t in self.tests) else PASS
        report = {"started_at": started, "finished_at": now_iso(), "verdict": verdict,
                  "options": {k: v for k, v in vars(args).items()}, "platform": platform_identity(),
                  "software": software_identity(), "camera_identity": self.identity or {"error": identity_source},
                  "configuration": configuration, "profile_readiness": self.readiness, "backend": self.backend_info,
                  "integrity": {"hashed_files": {k: len(v) for k, v in before.items()},
                                "changed": touched or "none"},
                  "tests": [t.to_dict() for t in self.tests]}
        paths = write_reports(self.out, report)
        counts = {s: sum(1 for t in self.tests if t.status == s) for s in (PASS, FAIL, SKIPPED)}
        print(f"\n{verdict}: {counts[PASS]} passed, {counts[FAIL]} failed, {counts[SKIPPED]} skipped")
        if self.backend_info.get("cpu_fallback"):
            print(f"NOTE: processing backend {self.backend_info['requested']} fell back to CPU")
        for kind, path in paths.items():
            print(f"{kind:5s} {path}")
        return 0 if verdict == PASS else 1


def main(argv=None) -> int:
    """Run the suite; returns the exit code."""
    args = parse_args(argv)
    if args.camera == "replay" and not args.replay_dir:
        print("--camera replay needs --replay-dir")
        return 2
    return Runner(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
