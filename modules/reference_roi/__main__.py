"""Stand-alone launcher: ``python -m modules.reference_roi`` opens the ROI selector."""
# Allow running this file directly (IDE 'run file', runfile): make the project importable.
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "modules.reference_roi"
from modules._support import launch


def main():
    """Open ``ReferenceROIWidget`` in its own window (no hardware attached)."""
    def factory():
        from .widget import ReferenceROIWidget
        return ReferenceROIWidget()
    return launch(factory, "Reference ROI Selection", (900, 700))


if __name__ == '__main__':
    main()
