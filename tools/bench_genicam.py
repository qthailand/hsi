# tools/bench_genicam.py
"""GenICam benchmark — 1000 frames ผ่าน pipeline จริง
ดึงค่าทั้งหมดจาก AppConfig

Usage:
    python tools/bench_genicam.py
    python tools/bench_genicam.py -n 2000
    python tools/bench_genicam.py --no-dark --no-bin
"""

import os
import sys
import time
import argparse
import numpy as np

# ── path setup ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import AppConfig, load_app_config


# ── pipeline steps (จาก project จริง) ──────────────────────

def apply_raw_orientation(raw: np.ndarray, inverted: bool = False) -> np.ndarray:
    if inverted:
        return np.flipud(raw).copy()
    return raw


def apply_dark(frame: np.ndarray, dark_mean: np.ndarray | None,
               hot_map: np.ndarray | None) -> np.ndarray:
    if dark_mean is not None and dark_mean.shape == frame.shape:
        result = frame.astype(np.int32) - dark_mean.astype(np.int32)
        frame = np.clip(result, 0, 65535).astype(np.uint16)

    if hot_map is not None and hot_map.shape == frame.shape:
        result = frame.copy()
        hot_ys, hot_xs = np.where(hot_map)
        for y, x in zip(hot_ys, hot_xs):
            neighbors = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < frame.shape[0] and 0 <= nx < frame.shape[1]:
                        if not hot_map[ny, nx]:
                            neighbors.append(int(frame[ny, nx]))
            if neighbors:
                result[y, x] = int(np.mean(neighbors))
        frame = result

    return frame


def create_binner(width: int, camera_bin: int, m: float, c: float,
                  resolution: float, method: str = "numba"):
    from modules.spectral_processing import LambdaBinner
    mm = m * camera_bin
    binner = LambdaBinner(m=mm, c=c, step=resolution, method=method)
    binner.fit(width)
    binner.warmup()
    return binner


def load_dark_reference(folder: str, bin_value: int,
                        exposure_ms: float, gain: float) -> tuple:
    """โหลด dark จาก DarkFrameLibrary — เหมือน ScanController"""
    try:
        from modules.dark import DarkFrameLibrary
        lib = DarkFrameLibrary(folder=folder, n_frames=1)
        result = lib.load(bin=bin_value, exposure=exposure_ms, gain=gain)
        if result is None:
            return None, None
        return result["mean"], result["hot_pixel_map"]
    except Exception as e:
        print(f"  WARNING: Could not load dark: {e}")
        return None, None


# ── benchmark ───────────────────────────────────────────────

def bench(config: AppConfig, n_frames: int = 1000, do_dark: bool = True,
          do_bin: bool = True, do_orient: bool = True):
    from harvesters.core import Harvester

    # ── ดึงค่าจาก config ──────────────────────────────────
    cti_path = getattr(config, 'cti_path', None) or os.path.join(
        os.getenv("GENICAM_GENTL64_PATH", ""), "libsv_gev_tl_x64.cti"
    )
    camera_index = getattr(config, 'camera_index', 0)
    gain = config.default_gain
    exposure_ms = config.default_exposure_ms
    bin_factor = config.default_bin
    raw_inverted = config.raw_inverted
    dark_folder = config.dark_frame_folder

    # LambdaBinner params
    lam_m = float(config.m)/float(config.default_bin)
    lam_c = float(config.c)
    lam_resolution = float(config.resolution)

    print("═" * 60)
    print("  GenICam Pipeline Benchmark")
    print("═" * 60)
    print(f"  CTI:           {cti_path}")
    print(f"  Camera index:  {camera_index}")
    print(f"  Frames:        {n_frames}")
    print(f"  Gain:          {gain}")
    print(f"  Exposure:      {exposure_ms} ms")
    print(f"  Bin:           {bin_factor}x{bin_factor}")
    print(f"  Dark subtract: {do_dark}")
    print(f"  LambdaBin:     {do_bin}")
    print(f"  Orientation:   {do_orient}")
    print(f"  Raw inverted:  {raw_inverted}")
    print(f"  Dark folder:   {dark_folder}")
    if do_bin:
        print(f"  λ params:      m={lam_m} c={lam_c} step={lam_resolution}")
    print()

    # ── load dark ─────────────────────────────────────────
    dark_mean = None
    hot_map = None

    if do_dark:
        print(f"  Loading dark reference...")
        dark_mean, hot_map = load_dark_reference(
            dark_folder, bin_factor, exposure_ms, gain
        )
        if dark_mean is not None:
            print(f"  Dark mean:     loaded ({dark_mean.shape})")
        else:
            print(f"  Dark mean:     None (no matching dark)")
        if hot_map is not None:
            n_hot = int(np.count_nonzero(hot_map))
            print(f"  Hot pixels:    {n_hot}")
        else:
            print(f"  Hot pixels:    None")
        print()

    # ── connect camera ────────────────────────────────────
    h = Harvester()
    h.add_file(cti_path)
    h.update()

    if len(h.device_info_list) == 0:
        print("ERROR: No device found")
        return

    #print(f"  Device: {h.device_info_list[0].get('model', '?')}")
    ia = h.create(camera_index)
    ia.buffer_handling_mode = "NewestOnly"
    ia.num_buffers = 64
    nm = ia.remote_device.node_map
    print(nm.PixelFormat.value)
    # ── tune camera (เหมือน GenICamBackend.tune_*) ────────
    try:
        nm.BinningHorizontal.value = bin_factor
        nm.BinningVertical.value = bin_factor
    except Exception:
        print("  WARNING: Could not set binning")

    try:
        nm.ExposureAuto.value = "Off"
        nm.ExposureTime.value = exposure_ms * 1000.0
    except Exception:
        pass

    try:
        nm.Gain.value = float(gain)
    except Exception:
        pass

    try:
        nm.AcquisitionMode.value = "Continuous"
        nm.TriggerMode.value = "Off"
    except Exception:
        pass

    # tune framerate

    keyword = 'pixel'
    avaiable_nodes = [node for node in dir(nm) if keyword.lower() in node.lower()]
    for node_name in avaiable_nodes:
        print(node_name)

    # tune framerate — ต้องเหมือน GenICamBackend.tune_framerate()
    try:
        if hasattr(nm, 'AcquisitionFrameRate'):
            nm.AcquisitionFrameRate.value = nm.AcquisitionFrameRate.max
            print(f"  AcquisitionFrameRate: {nm.AcquisitionFrameRate.value} (max)")
    except Exception:
        pass

    try:
        nm.AcquisitionFrameRateEnable.value = False
    except Exception:
        try:
            nm.AcquisitionFrameRateEnable.value = "Off"
        except Exception:
            pass

    # tune gige
    try:
        if hasattr(nm, 'GevSCPSFireTest'):
            nm.GevSCPSFireTest.execute()
            print("Ok")
        
        if hasattr(nm, 'GevSCPSPacketSize'):
            nm.GevSCPSPacketSize.value = nm.GevSCPSPacketSize.max
            print('GevSCPSPacketSize.Max', nm.GevSCPSPacketSize.max)
            print('GevSCPSPacketSize', nm.GevSCPSPacketSize.value)

        if hasattr(nm, 'DeviceLinkThroughputLimit'):
            nm.DeviceLinkThroughputLimit.value = nm.DeviceLinkThroughputLimit.max
            print(f"Throughput: {nm.DeviceLinkThroughputLimit.value}")

        if hasattr(nm, 'GevSCPD'):
            nm.GevSCPD.value = 0
            print("GevSCPD:", nm.GevSCPD.value)

    except Exception:
        pass

    # ── read info ─────────────────────────────────────────
    try:
        w = int(nm.Width.value)
        h_px = int(nm.Height.value)
    except Exception:
        w, h_px = 0, 0

    cam_fps = None
    for node_name in ("AcquisitionFrameRateResult", "AcquisitionFrameRate"):
        try:
            cam_fps = float(getattr(nm, node_name).value)
            break
        except Exception:
            pass

    bit_depth = 12
    pixel_format = None
    try:
        pixel_format = str(nm.PixelFormat.value)
    except Exception:
        pass

    print(f"  Resolution:    {w}x{h_px}")
    print(f"  Pixel format:  {pixel_format}")
    print(f"  Bit depth:     {bit_depth}")
    print(f"  Camera FPS:    {cam_fps}")
    print()

    # ── bandwidth analysis ───────────────────────────────────
    print()
    print("─" * 60)
    print("  BANDWIDTH ANALYSIS")
    print("─" * 60)

    # ข้อมูลจาก camera
    bytes_per_frame = w * h_px * 2  # uint16
    target_fps = 136.0
    current_fps = cam_fps or 0.0

    data_needed = bytes_per_frame * target_fps / (1024 * 1024)
    data_current = bytes_per_frame * current_fps / (1024 * 1024)

    # GigE theoretical max
    packet_size = nm.GevSCPSPacketSize.value if hasattr(nm, 'GevSCPSPacketSize') else 1500
    throughput_limit = nm.DeviceLinkThroughputLimit.value if hasattr(nm, 'DeviceLinkThroughputLimit') else 0

    print(f"  Frame size:         {bytes_per_frame/1024:.1f} KB")
    print(f"  Packet size:        {packet_size} bytes")
    print(f"  Throughput limit:   {throughput_limit/1024/1024:.1f} MB/s  ({throughput_limit*8/1e6:.0f} Mbps)")
    print(f"  Current FPS:        {current_fps:.1f}")
    print(f"  Current data rate:  {data_current:.1f} MB/s  ({data_current*8:.0f} Mbps)")
    print()
    print(f"  Target FPS:         {target_fps:.0f}")
    print(f"  Target data rate:   {data_needed:.1f} MB/s  ({data_needed*8:.0f} Mbps)")
    print(f"  GigE theoretical:   118.0 MB/s  (944 Mbps)")

    if data_needed > 118:
        print(f"  ⚠ Target EXCEEDS GigE capacity!")
    elif throughput_limit > 0 and data_needed > throughput_limit / (1024*1024):
        print(f"  ⚠ Target EXCEEDS DeviceLinkThroughputLimit!")
    elif throughput_limit > 0 and throughput_limit / (1024*1024) < 118:
        print(f"  ⚠ DeviceLinkThroughputLimit is BELOW GigE max!")
    else:
        print(f"  ✓ Bandwidth OK")

    # GevSCPD
    scpd = nm.GevSCPD.value if hasattr(nm, 'GevSCPD') else 'N/A'
    print(f"  GevSCPD:            {scpd}")
    print("─" * 60)


    # ── create binner ─────────────────────────────────────
    binner = None
    if do_bin and w > 0:
        print("  Fitting LambdaBinner...")
        t0 = time.monotonic()
        binner = create_binner(w, bin_factor, lam_m, lam_c, lam_resolution)
        t_fit = time.monotonic() - t0
        print(f"  Binner fitted: {w} → {binner.n_bins} bins ({t_fit:.3f}s)")
        print(f"  λ range:       {binner.lambda_out[0]:.1f} – {binner.lambda_out[-1]:.1f} nm")
        print()

    # ── benchmark pipeline ────────────────────────────────
    ia.start()
    print(f"  Fetching {n_frames} frames through pipeline...")
    print()

    t_fetch_total = 0.0
    t_orient_total = 0.0
    t_dark_total = 0.0
    t_bin_total = 0.0
    timestamps = []
    errors = 0
    incomplete = 0

    # warmup
    print("  Warmup (10 frames)...")
    for _ in range(10):
        try:
            with ia.fetch(timeout=5000) as buf:
                pass
        except Exception:
            pass

    t_start = time.monotonic()

    for i in range(n_frames):
        try:
            # step 1: fetch
            t0 = time.monotonic()
            with ia.fetch(timeout=5000) as buffer:
                t_fetch = time.monotonic()

                comp = buffer.payload.components[0]
                raw = comp.data.reshape(comp.height, comp.width).astype(np.uint16)

                shift = 16 - bit_depth
                raw16 = np.ascontiguousarray(raw << shift) if shift > 0 else np.ascontiguousarray(raw)

                if getattr(buffer, 'is_incomplete', False):
                    incomplete += 1

            t_fetch_total += (t_fetch - t0)

            # step 2: orientation
            t0 = time.monotonic()
            if do_orient:
                frame = apply_raw_orientation(raw16, raw_inverted)
            else:
                frame = raw16
            t_orient_total += (time.monotonic() - t0)

            # step 3: dark subtract
            t0 = time.monotonic()
            if do_dark:
                frame = apply_dark(frame, dark_mean, hot_map)
            t_dark_total += (time.monotonic() - t0)

            # step 4: lambda bin
            t0 = time.monotonic()
            if binner is not None:
                binned = binner.transform(frame[np.newaxis, :, :])
                frame = np.ascontiguousarray(binned[0])
            t_bin_total += (time.monotonic() - t0)

            timestamps.append(time.monotonic())

            # progress
            if (i + 1) % 100 == 0:
                elapsed = time.monotonic() - t_start
                current_fps = (i + 1) / elapsed
                out_shape = f"{frame.shape[1]}x{frame.shape[0]}" if frame.ndim == 2 else str(frame.shape)
                print(f"    {i+1:>5}/{n_frames}  {current_fps:.1f} fps  "
                      f"out={out_shape}  elapsed={elapsed:.2f}s")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"    ERROR #{errors}: {type(e).__name__}: {e}")

    t_end = time.monotonic()

    # ── stop ──────────────────────────────────────────────
    ia.stop()
    ia.destroy()
    h.reset()

    # ── results ───────────────────────────────────────────
    total_time = t_end - t_start
    n_ok = len(timestamps)

    if n_ok < 2:
        print(f"\n  ERROR: Only {n_ok} frames captured")
        return

    overall_fps = n_ok / total_time
    intervals = np.diff(timestamps)

    avg_interval = np.mean(intervals)
    min_interval = np.min(intervals)
    max_interval = np.max(intervals)
    std_interval = np.std(intervals)

    p50 = np.percentile(intervals, 50)
    p95 = np.percentile(intervals, 95)
    p99 = np.percentile(intervals, 99)

    bytes_per_frame = w * h_px * 2
    data_rate = (bytes_per_frame * n_ok) / total_time / (1024 * 1024)

    print()
    print("═" * 60)
    print("  RESULTS")
    print("═" * 60)
    print(f"  Total time:      {total_time:.3f} s")
    print(f"  Frames OK:       {n_ok}")
    print(f"  Errors:          {errors}")
    print(f"  Incomplete:      {incomplete}")
    print()
    print(f"  Overall FPS:     {overall_fps:.2f}")
    print(f"  Avg interval:    {avg_interval*1000:.2f} ms")
    print(f"  Min interval:    {min_interval*1000:.2f} ms")
    print(f"  Max interval:    {max_interval*1000:.2f} ms")
    print(f"  Std interval:    {std_interval*1000:.2f} ms")
    print()
    print(f"  P50 interval:    {p50*1000:.2f} ms  ({1/p50:.1f} fps)")
    print(f"  P95 interval:    {p95*1000:.2f} ms  ({1/p95:.1f} fps)")
    print(f"  P99 interval:    {p99*1000:.2f} ms  ({1/p99:.1f} fps)")
    print()

    # ── per-step timing ───────────────────────────────────
    print("─" * 60)
    print("  PIPELINE TIMING (avg per frame)")
    print("─" * 60)

    t_avg = total_time / n_ok * 1000

    t_fetch_avg = t_fetch_total / n_ok * 1000
    t_orient_avg = t_orient_total / n_ok * 1000
    t_dark_avg = t_dark_total / n_ok * 1000
    t_bin_avg = t_bin_total / n_ok * 1000
    t_overhead = t_avg - t_fetch_avg - t_orient_avg - t_dark_avg - t_bin_avg

    steps = [
        ("Fetch (GenICam)", t_fetch_avg),
        ("Orientation", t_orient_avg),
        ("Dark subtract", t_dark_avg),
        ("LambdaBinning", t_bin_avg),
        ("Other overhead", max(0, t_overhead)),
    ]

    for name, ms in steps:
        bar = "█" * int(ms / t_avg * 30) if t_avg > 0 else ""
        pct = ms / t_avg * 100 if t_avg > 0 else 0
        print(f"  {name:<20} {ms:>8.3f} ms  {pct:>5.1f}%  {bar}")

    print(f"  {'─'*44}")
    print(f"  {'TOTAL':<20} {t_avg:>8.3f} ms  {100:>5.1f}%")
    print(f"  {'Pipeline FPS':<20} {1000/t_avg:>8.2f}")
    print()

    # ── data summary ──────────────────────────────────────
    print("─" * 60)
    print("  DATA SUMMARY")
    print("─" * 60)
    print(f"  Input size:      {w}x{h_px}  ({bytes_per_frame/1024:.1f} KB/frame)")
    if binner is not None:
        out_bytes = binner.n_bins * h_px * 4
        print(f"  Output size:     {binner.n_bins}x{h_px}  ({out_bytes/1024:.1f} KB/frame)")
    print(f"  Data rate (in):  {data_rate:.1f} MB/s")
    print(f"  Bandwidth:       {data_rate * 8:.1f} Mbps")
    print("═" * 60)

    # ── config summary ────────────────────────────────────
    print()
    print("─" * 60)
    print("  CONFIG USED")
    print("─" * 60)
    print(f"  config.default_gain:      {config.default_gain}")
    print(f"  config.default_exposure:  {config.default_exposure_ms} ms")
    print(f"  config.default_bin:       {config.default_bin}")
    print(f"  config.raw_inverted:      {config.raw_inverted}")
    print(f"  config.m:                 {config.m}")
    print(f"  config.c:                 {config.c}")
    print(f"  config.resolution:        {config.resolution}")
    print(f"  config.dark_frame_folder: {config.dark_frame_folder}")
    print(f"  config.camera_backend:    {getattr(config, 'camera_backend', 'poa')}")
    print(f"  config.camera_index:      {getattr(config, 'camera_index', 0)}")
    print(f"  config.cti_path:          {getattr(config, 'cti_path', None)}")
    print("═" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GenICam Pipeline Benchmark")
    parser.add_argument("-n", "--frames", type=int, default=1000, help="Number of frames")
    parser.add_argument("--no-dark", action="store_true", help="Skip dark subtraction")
    parser.add_argument("--no-bin", action="store_true", help="Skip LambdaBinning")
    parser.add_argument("--no-orient", action="store_true", help="Skip orientation")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    args = parser.parse_args()

    # ── load config (เหมือน app จริง) ────────────────────
    if args.config:
        config = load_app_config(args.config)
    else:
        config = AppConfig()

    print(config)

    bench(
        config=config,
        n_frames=args.frames,
        do_dark=not args.no_dark,
        do_bin=not args.no_bin,
        do_orient=not args.no_orient,
    )