"""Command line: ``python -m modules.stage --type rotation --degree 10 --dps 2``.

Runs one headless sequence (home, scan, return home) on the real stage and prints
every event. Needs the stage connected and ``libximc`` installed.
"""
# Allow running this file directly (IDE 'run file', runfile): make the project importable.
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "modules.stage"
import argparse
import time

from .sequence import StageSequence


def main(argv=None) -> int:
    """Run the scan sequence once; returns 0 if it completed, else 1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", choices=("rotation", "linear"), default="rotation")
    parser.add_argument("--degree", type=float, default=10.0, help="rotation: scan angle")
    parser.add_argument("--dps", type=float, default=1.0, help="rotation: degrees per second")
    parser.add_argument("--distance-mm", type=float, default=10.0, help="linear: travel")
    parser.add_argument("--speed-um-s", type=float, default=1000.0, help="linear: speed")
    args = parser.parse_args(argv)

    start = time.perf_counter()

    def stamp(text):
        print(f"{time.perf_counter() - start:7.2f}s  {text}")

    sequence = StageSequence(
        args.type,
        on_before=lambda: stamp("scan starts"),
        on_after=lambda: stamp("scan ended"),
        on_home=lambda: stamp("returned home"),
        on_error=lambda message: stamp(f"ERROR: {message}"),
    )
    sequence.degree, sequence.degree_per_second = args.degree, args.dps
    sequence.distance_mm, sequence.speed_um_s = args.distance_mm, args.speed_um_s
    sequence.run()
    stamp("completed" if sequence.completed else f"not completed (failed={sequence.failed})")
    return 0 if sequence.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
