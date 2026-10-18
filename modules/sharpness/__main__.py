"""Stand-alone use.

``python -m modules.sharpness image.npy [--axis 0]`` prints the score of a saved
image; ``python -m modules.sharpness --gui`` opens the sidebar widget.
"""
# Allow running this file directly (IDE 'run file', runfile): make the project importable.
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "modules.sharpness"
import argparse


def main(argv=None) -> int:
    """Score an ``.npy`` image, or open the demo widget with ``--gui``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help=".npy image (slit, wavelength)")
    parser.add_argument("--axis", type=int, choices=(0, 1), default=0)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args(argv)
    if args.gui or not args.image:
        from modules._support import launch
        from .widget import SharpnessPanel

        def build():
            panel = SharpnessPanel()
            panel.set_value(274.0)
            return panel
        return launch(build, "Sharpness", (120, 320))
    import numpy as np
    from .core import sharpness_score
    print(f"sharpness (axis={args.axis}): {sharpness_score(np.load(args.image), args.axis):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
