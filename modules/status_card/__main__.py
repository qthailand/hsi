"""Stand-alone demo: ``python -m modules.status_card`` shows a sample card."""
# Allow running this file directly (IDE 'run file', runfile): make the project importable.
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "modules.status_card"
from modules._support import launch


def main():
    """Open a window containing a demo ``StatusCard``."""
    def factory():
        from .widget import StatusCard
        card = StatusCard("Demo")
        card.set_content("42 GB", "Available capacity", color="green")
        return card
    return launch(factory, "Status Card", (320, 160))


if __name__ == '__main__':
    main()
