# smile/test_gui.py
"""Launch the standalone Smile main window for manual inspection.

Usage:
    cd c:\\modules
    python -m smile.test_gui [image_path]
"""
import sys

from smile.widget import HAVE_QT, launch_gui


def main() -> None:
    """Launch the widget in a standalone window."""
    if not HAVE_QT:
        sys.exit(
            "PyQt5 not available. Install with:\n"
            "  pip install PyQt5 matplotlib"
        )

    sys.exit(launch_gui(sys.argv[1:]))


if __name__ == "__main__":
    main()
