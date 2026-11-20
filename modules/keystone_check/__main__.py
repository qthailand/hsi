"""Stand-alone use: ``python -m modules.keystone_check [cube.hdr]`` opens the Keystone Check window."""
# Allow running this file directly (IDE 'run file', runfile): make the project importable.
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "modules.keystone_check"
import sys

from modules._support import launch


def main(argv=None) -> int:
    """Open the window, optionally with an ENVI cube; returns the Qt exit code."""
    argv = sys.argv[1:] if argv is None else argv
    from .widget import KeystoneCheckWidget

    def build():
        widget = KeystoneCheckWidget()
        if argv:
            from modules.cube_io import envi_scan_metadata, load_envi_cube_for_viewer
            hdr = argv[0] if argv[0].lower().endswith(".hdr") else argv[0] + ".hdr"
            cube, wl = load_envi_cube_for_viewer(hdr)
            widget.set_cube(cube, wl, envi_scan_metadata(hdr[:-4]), hdr)
        return widget
    return launch(build, "Keystone Check", (1400, 700))


if __name__ == "__main__":
    raise SystemExit(main())
