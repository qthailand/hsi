# debug_controller.py
"""Standalone console test for ScanController — no GUI needed."""

import sys
import os
import time
import logging

# ★ Force headless — no display needed
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt5.QtCore import QTimer, QCoreApplication
from PyQt5.QtWidgets import QApplication

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DEBUG")


def main():
    print("=" * 60)
    print("  ScanController Console Debug")
    print("=" * 60)

    # ── 1. QApplication (offscreen — ไม่ต้อง display) ──
    print("\n[STEP 1] Creating QApplication...")
    app = QApplication(sys.argv)
    print("[STEP 1] OK")

    # ── 2. Load config ──
    print("\n[STEP 2] Loading config...")
    try:
        from config import load_app_config
        config = load_app_config("backend.json")
        print(f"[STEP 2] OK — config loaded")
        print(f"  m={config.m}, c={config.c}, resolution={config.resolution}")
        print(f"  camera_backend={config.camera_backend}")
        print(f"  default_bin={config.default_bin}")
        print(f"  default_gain={config.default_gain}")
        print(f"  default_exposure_ms={config.default_exposure_ms}")
    except Exception as e:
        print(f"[STEP 2] FAIL — {e}")
        return

    # ── 3. Test backend directly ──
    print("\n[STEP 3] Testing camera backend...")

    backend_type = str(getattr(config, 'camera_backend', 'poa')).lower()

    try:
        if backend_type == "genicam":
            from workers.backends.genicam import GenICamBackend
            backend = GenICamBackend(
                camera_index=getattr(config, 'camera_index', 0),
                gain=config.default_gain,
                exposure_ms=config.default_exposure_ms,
                cti_path=getattr(config, 'cti_path', None),
                default_bin=config.default_bin,
            )
        else:
            from workers.backends.poa_camera import POACameraBackend
            backend = POACameraBackend(
                gain=config.default_gain,
                exposure_ms=config.default_exposure_ms,
            )

        print(f"[STEP 3a] Backend class: {backend.__class__.__name__}")

        print("[STEP 3b] backend.open()...")
        t0 = time.time()
        backend.open()
        print(f"[STEP 3b] OK — {time.time()-t0:.2f}s")

        print(f"  gain_range={backend.gain_range}")
        print(f"  exposure_range_us={backend.exposure_range_us}")
        print(f"  bit_depth={backend.bit_depth}")
        print(f"  bins={backend.get_bins()}")
        print(f"  width={backend.info.width}, height={backend.info.height}")

    except Exception as e:
        print(f"[STEP 3] FAIL — {e}")
        import traceback
        traceback.print_exc()
        return

    # ── 4. Test single frame read ──
    print("\n[STEP 4] Reading single frame...")

    try:
        backend.start_stream()
        print("[STEP 4a] start_stream() OK")

        t0 = time.time()
        frame = backend.read_frame(timeout_ms=5000)
        elapsed = time.time() - t0

        if frame is None:
            print(f"[STEP 4b] FAIL — frame is None ({elapsed:.2f}s)")
        else:
            print(f"[STEP 4b] OK — {elapsed:.4f}s")
            print(f"  shape={frame.shape} dtype={frame.dtype}")
            print(f"  min={frame.min()} max={frame.max()}")

        backend.stop_stream()

    except Exception as e:
        print(f"[STEP 4] FAIL — {e}")
        import traceback
        traceback.print_exc()
        backend.close()
        return

    # ── 5. FPS benchmark (100 frames) ──
    print("\n[STEP 5] FPS benchmark (100 frames)...")

    try:
        backend.start_stream()

        n_frames = 100
        times = []
        fps_list = []

        for i in range(n_frames):
            t0 = time.time()
            frame = backend.read_frame(timeout_ms=5000)
            elapsed = time.time() - t0
            times.append(elapsed)

            if frame is not None:
                fps = 1.0 / max(elapsed, 1e-6)
                fps_list.append(fps)

            if (i + 1) % 10 == 0:
                avg = sum(times[-10:]) / 10
                print(f"  frame {i+1:3d}/{n_frames}  "
                      f"instant={1/max(elapsed,1e-6):.1f}fps  "
                      f"avg_10={1/max(avg,1e-6):.1f}fps  "
                      f"shape={frame.shape if frame is not None else 'None'}")

        backend.stop_stream()

        if fps_list:
            avg_fps = sum(fps_list) / len(fps_list)
            min_fps = min(fps_list)
            max_fps = max(fps_list)
            print(f"\n  RESULT: avg={avg_fps:.1f}fps  "
                  f"min={min_fps:.1f}fps  max={max_fps:.1f}fps")
        else:
            print("\n  RESULT: no valid frames!")

    except Exception as e:
        print(f"[STEP 5] FAIL — {e}")
        import traceback
        traceback.print_exc()

    # ── 6. Test dark frame library ──
    print("\n[STEP 6] Testing DarkFrameLibrary...")

    try:
        from workers import DarkFrameLibrary
        dark_lib = DarkFrameLibrary(
            folder=config.dark_frame_folder,
            n_frames=config.dark_frame_n_frames,
        )
        print(f"[STEP 6a] OK — folder={config.dark_frame_folder}")
        print(f"  is_collecting={dark_lib.is_collecting}")

        result = dark_lib.load(
            bin=config.default_bin,
            exposure=config.default_exposure_ms,
            gain=config.default_gain,
        )
        if result is None:
            print("[STEP 6b] No dark frame found (expected if first run)")
        else:
            print(f"[STEP 6b] Dark loaded: n_frames={result['n_frames']}")

    except Exception as e:
        print(f"[STEP 6] FAIL — {e}")
        import traceback
        traceback.print_exc()

    # ── 7. Test LambdaBinner ──
    print("\n[STEP 7] Testing LambdaBinner...")

    try:
        from BiningTools import LambdaBinner

        bin_factor = int(getattr(config, 'default_bin', 1) or 1)
        mm = float(config.m) * bin_factor
        cc = float(config.c)
        step = float(config.resolution)

        binner = LambdaBinner(m=mm, c=cc, step=step, method="numba")
        test_width = backend.info.width or 804
        binner.fit(test_width)
        binner.warmup()

        print(f"[STEP 7] OK")
        print(f"  m={mm:.4f} (native={config.m} × bin={bin_factor})")
        print(f"  step={step:.2f}")
        print(f"  input={test_width} pixels → {binner.n_bins} bins")
        print(f"  λ range: {binner.lambda_out[0]:.1f} – {binner.lambda_out[-1]:.1f} nm")

        # test transform
        dummy = np.random.randint(0, 1000, (1, 50, test_width), dtype=np.uint16)
        t0 = time.time()
        out = binner.transform(dummy)
        print(f"  transform: {dummy.shape} → {out.shape} in {time.time()-t0:.4f}s")

    except Exception as e:
        print(f"[STEP 7] FAIL — {e}")
        import traceback
        traceback.print_exc()

    # ── 8. Cleanup ──
    print("\n[STEP 8] Cleanup...")
    try:
        backend.close()
        print("[STEP 8] backend.close() OK")
    except Exception as e:
        print(f"[STEP 8] FAIL — {e}")

    print("\n" + "=" * 60)
    print("  Done — all steps above show where the issue is")
    print("=" * 60)


if __name__ == "__main__":
    import numpy as np
    main()