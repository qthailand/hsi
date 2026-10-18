"""Headless acquisition check: ``python -m modules.acquisition --backend poa --seconds 5``.

Runs the full pipeline (no GUI) for a few seconds and prints the frame rate, the number of
frames recorded and the spectral bins produced.
"""
from __future__ import annotations
# Allow running this file directly (IDE 'run file', runfile): make the project importable.
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "modules.acquisition"

import argparse
import threading
import time
from types import SimpleNamespace

from modules.camera import create_backend

from .engine import AcquisitionEngine, AcquisitionSink


class _Counter(AcquisitionSink):
    """Count what the engine delivers."""

    def __init__(self):
        self.frames = 0
        self.fps = 0.0
        self.shape = None
        self.errors = []
        self.sharpness_score = None

    def frame(self, frame, fps, timestamp):
        self.frames += 1
        self.fps, self.shape = fps, frame.shape

    def sharpness(self, score):
        self.sharpness_score = score

    def error(self, message):
        self.errors.append(message)


def main(argv=None) -> int:
    """Run the engine for ``--seconds`` and print a summary; returns 0 if frames arrived."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="poa", help="poa | genicam | ximea")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--gain", type=float, default=1)
    parser.add_argument("--exposure-ms", type=float, default=10.0)
    parser.add_argument("--bin", type=int, default=1)
    parser.add_argument("--m", type=float, default=0.2903, help="nm/pixel at bin 1")
    parser.add_argument("--c", type=float, default=399.791, help="wavelength (nm) of pixel 0")
    parser.add_argument("--resolution", type=float, default=2.0, help="nm per output bin")
    parser.add_argument("--record", action="store_true", help="record every frame during the run")
    args = parser.parse_args(argv)

    config = SimpleNamespace(camera_backend=args.backend, camera_index=args.index, cti_path=None,
                             default_gain=args.gain, default_exposure_ms=args.exposure_ms, default_bin=args.bin,
                             m=args.m, c=args.c, resolution=args.resolution, smile_map_folder=".",
                             preprocessing_backend="auto")
    camera = create_backend(config)
    camera.open()
    if args.bin in camera.get_bins():
        camera.set_bin(args.bin)
    camera.set_params(args.gain, args.exposure_ms)

    sink = _Counter()
    engine = AcquisitionEngine(camera, config, sink=sink)
    if args.record:
        engine.set_recording(True)
    worker = threading.Thread(target=engine.run_until_stopped, daemon=True)
    worker.start()
    time.sleep(args.seconds)
    engine.request_stop()
    worker.join(5)
    recorded = len(engine.take_recorded_frames())
    print(f"{sink.frames} display frames, {sink.fps:.1f} fps, output {sink.shape}, recorded {recorded}, "
          f"bins {engine.binner_n_bins}, sharpness {sink.sharpness_score}, errors {sink.errors or 'none'}")
    return 0 if sink.frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
