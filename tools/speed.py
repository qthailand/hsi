#hsi/tools/speed.py
"""Standalone GenICam camera test script — single file, no dependencies beyond harvesters + numpy."""

import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from BiningTools import LambdaBinner
# ── config ────────────────────────────────────────────────
CTI_PATH = os.path.join(
    os.getenv("GENICAM_GENTL64_PATH", ""),
    "libsv_gev_tl_x64.cti",
)
DEVICE_INDEX = 0
EXPOSURE_US = 1000.0
GAIN = 42.0
BUFFER_COUNT = 1
FETCH_TIMEOUT_MS = 5000
NUM_FRAMES = 100
BINNING = 2

# ── spectral binning config ───────────────────────────────
ENABLE_SPECTRAL_BIN = True   # ★ เปิด/ปิด spectral binning
BIN_M = 0.2903               # λ = m * pixel + c
BIN_C = 399.7910
BIN_STEP = 2.0               # nm per bin

def main():
    from harvesters.core import Harvester

    print(f"CTI: {CTI_PATH}")
    print(f"Device index: {DEVICE_INDEX}")
    print()

    # ── 1. connect ────────────────────────────────────────
    print("[1/7] Connecting...")
    h = Harvester()
    h.add_file(CTI_PATH)
    h.update()

    if len(h.device_info_list) == 0:
        print("ERROR: No device found. Check IP/subnet/firewall.")
        h.reset()
        sys.exit(1)

    for i, info in enumerate(h.device_info_list):
        print(f"  [{i}] {info}")

    ia = h.create(DEVICE_INDEX)
    ia.buffer_handling_mode = "NewestOnly"
    ia.num_buffers = BUFFER_COUNT
    nm = ia.remote_device.node_map
    print("  Connected OK\n")

    # ── 2. node helpers ───────────────────────────────────
    def read(name):
        try:
            return getattr(nm, name).value
        except Exception:
            return None

    def write(name, value):
        try:
            getattr(nm, name).value = value
            return True
        except Exception:
            return False

    def write_max(name):
        try:
            node = getattr(nm, name)
            node.value = node.max
            return True
        except Exception:
            return False

    def has(name):
        return hasattr(nm, name)

    # ── 3. tune ───────────────────────────────────────────
    print("[2/7] Tuning GigE...")
    if has("GevSCPSPacketSize"):
        write_max("GevSCPSPacketSize")
        print(f"  Packet size: {read('GevSCPSPacketSize')}")
    if has("DeviceLinkThroughputLimit"):
        write_max("DeviceLinkThroughputLimit")
    write("GevSCPD", 0)

    print("[3/7] Tuning framerate...")
    if has("AcquisitionFrameRate"):
        write_max("AcquisitionFrameRate")
    if has("AcquisitionFrameRateEnable"):
        write("AcquisitionFrameRateEnable", False)

    print("[4/7] Tuning image...")
    write("ExposureAuto", "Off")
    write("ExposureTime", EXPOSURE_US)
    write("Gain", GAIN)
    write("AcquisitionMode", "Continuous")
    write("TriggerMode", "Off")

    if BINNING and has("BinningHorizontal"):
        write("BinningHorizontal", BINNING)
        write("BinningVertical", BINNING)
        print(f"  Binning: {BINNING}x{BINNING}")

    # ── 4. diagnostics ────────────────────────────────────
    print("\n[5/7] Camera info:")
    for name in ("Width", "Height", "PixelFormat", "ExposureTime",
                 "Gain", "AcquisitionFrameRate", "AcquisitionFrameRateResult",
                 "GevSCPSPacketSize", "GevSCPD"):
        val = read(name)
        if val is not None:
            print(f"  {name}: {val}")

    w = int(read("Width") or 0)
    h_px = int(read("Height") or 0)
    print(f"  Resolution: {w}x{h_px}")
    print()

    # ── 5. spectral binner ────────────────────────────────
    binner = None
    if ENABLE_SPECTRAL_BIN:

        m = BIN_M * BINNING
        binner = LambdaBinner(m=m, c=BIN_C, step=BIN_STEP)
        binner.fit(w)
        binner.warmup()
        print(f"[5/7] LambdaBinner: {w} pixels → {binner.n_bins} bins "
              f"(m={m:.4f}, c={BIN_C}, step={BIN_STEP}nm, "
              f"λ {binner.lambda_out[0]:.1f}-{binner.lambda_out[-1]:.1f} nm)")
    else:
        print("[5/7] Spectral binning: DISABLED")
    print()

    # ── 6. warmup ─────────────────────────────────────────
    ia.start()

    print("[6/7] Warming up (10 frames)...")
    for _ in range(10):
        try:
            with ia.fetch(timeout=FETCH_TIMEOUT_MS) as buffer:
                pass
        except Exception:
            pass
    print("  Warmup done\n")

    # ── 7. acquire ────────────────────────────────────────
    print(f"[7/7] Acquiring {NUM_FRAMES} frames...")
    times_fetch = []
    times_bin = []
    errors = 0
    incomplete = 0

    try:
        for i in range(NUM_FRAMES):
            t0 = time.perf_counter()
            try:
                with ia.fetch(timeout=FETCH_TIMEOUT_MS) as buffer:
                    if getattr(buffer, "is_incomplete", False):
                        incomplete += 1
                        continue

                    comp = buffer.payload.components[0]
                    data = comp.data
                    dt_fetch = (time.perf_counter() - t0) * 1000
                    times_fetch.append(dt_fetch)

                    if i == 0:
                        print(f"  Frame 0: {comp.width}x{comp.height}  "
                              f"dtype={data.dtype}  shape={data.shape}  "
                              f"contiguous={data.flags['C_CONTIGUOUS']}")

                    # ★ spectral binning
                    if binner is not None:
                        t1 = time.perf_counter()
                        frame_2d = data.reshape(comp.height, comp.width)
                        binned = binner.transform(frame_2d[np.newaxis])
                        dt_bin = (time.perf_counter() - t1) * 1000
                        times_bin.append(dt_bin)

                    if (i + 1) % 20 == 0:
                        parts = [f"fetch={dt_fetch:.2f}ms"]
                        if times_bin:
                            parts.append(f"bin={times_bin[-1]:.2f}ms")
                        print(f"  Frame {i+1}: {'  '.join(parts)}")

            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  Frame {i} error: {type(e).__name__}: {e}")

    except KeyboardInterrupt:
        print("\n  Interrupted")

    ia.stop()

    # ── summary ───────────────────────────────────────────
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    if times_fetch:
        arr = np.array(times_fetch)
        print(f"  Frames OK:       {len(times_fetch)}")
        print(f"  Errors:          {errors}")
        print(f"  Incomplete:      {incomplete}")
        print(f"  Fetch mean:      {arr.mean():.2f} ms")
        print(f"  Fetch median:    {np.median(arr):.2f} ms")
        print(f"  Fetch min:       {arr.min():.2f} ms")
        print(f"  Fetch max:       {arr.max():.2f} ms")
        print(f"  FPS (from fetch):{1000.0 / arr.mean():.1f}")

    if times_bin:
        arr_bin = np.array(times_bin)
        print(f"\n  Spectral binning ({binner.n_bins} bins):")
        print(f"  Bin mean:        {arr_bin.mean():.2f} ms")
        print(f"  Bin median:      {np.median(arr_bin):.2f} ms")
        print(f"  Bin min:         {arr_bin.min():.2f} ms")
        print(f"  Bin max:         {arr_bin.max():.2f} ms")

        total = np.array(times_fetch) + np.array(times_bin[:len(times_fetch)])
        print(f"\n  Total (fetch+bin):")
        print(f"  Total mean:      {total.mean():.2f} ms")
        print(f"  Total FPS:       {1000.0 / total.mean():.1f}")
    else:
        print(f"\n  Spectral binning: DISABLED")

    if not times_fetch:
        print("  No frames captured!")

    # cleanup
    ia.destroy()
    h.reset()
    print("\nDone.")


if __name__ == "__main__":
    main()