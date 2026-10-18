"""Stand-alone use: ``python -m modules._template`` opens the widget in its own window."""
# Allow running this file directly (IDE 'run file', runfile): make the project importable.
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "modules._template"
from modules._support import launch


def main() -> int:
    """Open the demo window; returns the Qt exit code."""
    from .widget import TemplateWidget

    def build():
        widget = TemplateWidget()
        widget.set_value(21, 2)
        return widget
    return launch(build, "Template", (300, 100))


if __name__ == "__main__":
    raise SystemExit(main())
