"""Stand-alone use: ``python -m modules.focus_map [frame.png|frame.npy]`` opens the focus map window."""
# Allow running this file directly (IDE 'run file', runfile): make the project importable.
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "modules.focus_map"
import sys

from modules._support import launch


def main(argv=None) -> int:
    """Open the window, optionally with a frame file; returns the Qt exit code."""
    argv = sys.argv[1:] if argv is None else argv
    from .widget import FocusMapWidget

    def build():
        widget = FocusMapWidget()
        widget.set_grab_available(False)
        if argv:
            import numpy as np
            path = argv[0]
            if path.lower().endswith(".npy"):
                img = np.load(path)
            else:
                import matplotlib.image as mpimg
                img = mpimg.imread(path)
            img = np.asarray(img, np.float32)
            widget.set_frame(img[..., :3].mean(axis=2) if img.ndim == 3 else img)
        return widget
    return launch(build, "Focus Map", (1250, 760))


if __name__ == "__main__":
    raise SystemExit(main())
