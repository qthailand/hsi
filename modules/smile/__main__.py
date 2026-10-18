"""Run the standalone Smile GUI with ``python -m smile``."""
# Allow running this file directly (IDE 'run file', runfile): make the project importable.
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "modules.smile"

from .smile import main


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        raise SystemExit(
            f"{exc}\nInstall GUI dependencies with: "
            "pip install PyQt5 matplotlib"
        ) from exc
