"""Check where the GenICam saturation level comes from (sensor, bit shift or dark).

Point a bright light at the slit so part of the frame saturates, then run::

    python tools/genicam_bit_check.py --gain 1 --exposure-ms 50

It prints the camera's pixel-format nodes, the raw buffer values *before* the
``<< (16 - bits)`` shift, the converted values after it, and (if a dark file exists for
the same bin/exposure/gain) the saturated level after dark subtraction.
Nothing is written to the camera except what ``open`` normally writes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _describe(name, frame, shift):
    flat = frame.ravel()
    values, counts = np.unique(flat, return_counts=True)
    top = values[-5:]
    print(f"{name}: dtype {frame.dtype} min {flat.min()} max {flat.max()} "
          f"pixels at max {counts[-1]} ({counts[-1] / flat.size:.2%})")
    print(f"    top 5 values: {top.tolist()}")
    if shift:
        print(f"    values that are multiples of 2**{shift}: {np.mean(flat % (1 << shift) == 0):.1%}")


def main(argv=None) -> int:
    """Grab one frame and print the bit-depth diagnostics; returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gain", type=float, default=1)
    parser.add_argument("--exposure-ms", type=float, default=50.0)
    parser.add_argument("--bin", type=int, default=None, help="default: config default_bin")
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args(argv)

    from config import load_app_config
    from modules.camera.factory import create_backend
    from modules.camera.genicam import GenICamBackend
    from modules.dark import DarkFrameLibrary, DarkProcessor

    config = load_app_config()
    config.camera_backend = "genicam"
    config.default_gain = args.gain
    config.default_exposure_ms = args.exposure_ms
    if args.bin is not None:
        config.default_bin = args.bin

    captured = {}
    original_convert = GenICamBackend._convert

    def spy_convert(data, bits):
        captured["raw"] = np.array(data, copy=True)
        captured["bits"] = bits
        return original_convert(data, bits)

    GenICamBackend._convert = staticmethod(spy_convert)
    camera = create_backend(config)
    try:
        camera.open()
        print("PixelFormat      :", camera._read_node("PixelFormat"))
        print("PixelSize        :", camera._read_node("PixelSize"))
        print("SensorPixelSize  :", camera._read_node("SensorPixelSize"))
        for node in ("Gain", "BlackLevel", "BlackLevelRaw", "Offset", "DigitalGain", "Gamma"):
            if camera._has_node(node):
                print(f"{node:<17}:", camera._read_node(node))
        print("info.bit_depth   :", camera.info.bit_depth, " bin", camera.current_bin)

        camera.set_params(args.gain, args.exposure_ms)
        camera.start_stream()
        frame = None
        for _ in range(args.frames):
            image = camera.read_frame_raw(timeout_ms=3000)
            if image is not None:
                frame = image
        if frame is None:
            print("no frame received")
            return 1

        bits = int(captured["bits"])
        shift = 16 - bits if 8 < bits < 16 else 0
        print(f"\nbits used by _convert: {bits}  -> shift << {shift}")
        print(f"expected full scale after shift: {((1 << bits) - 1) << shift}")
        _describe("raw buffer (before shift)", captured["raw"], 0)
        _describe("converted frame (after shift)", frame, shift)

        library = DarkFrameLibrary(config.dark_frame_folder)
        dark = library.load(camera.current_bin, args.exposure_ms, args.gain)
        if dark is None or dark["mean"].shape != frame.shape:
            print("\nno matching dark file for this bin/exposure/gain")
        else:
            _describe("dark mean", dark["mean"], 0)
            processor = DarkProcessor()
            processor.set_dark_reference(dark["mean"], None)
            _describe("after dark subtract", processor._apply_dark_subtract(frame), 0)
        return 0
    finally:
        camera.close()
        GenICamBackend._convert = staticmethod(original_convert)


if __name__ == "__main__":
    raise SystemExit(main())
