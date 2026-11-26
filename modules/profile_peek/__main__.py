"""Stand-alone demo: ``python -m modules.profile_peek`` shows a peek of a synthetic flat-field profile."""
# Allow running this file directly (IDE 'run file', runfile): make the project importable.
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "modules.profile_peek"
from modules._support import launch


def main():
    """Open a window containing a ``ProfilePeek`` with a synthetic flat-field profile."""
    def factory():
        from types import SimpleNamespace

        import numpy as np

        from .core import flatfield_peek
        from .widget import ProfilePeek
        rows = np.linspace(-1, 1, 400)
        table = np.outer(1.0 - 0.25 * rows ** 4, np.ones(60))
        profile = SimpleNamespace(profile=lambda: table, key=SimpleNamespace(label=lambda: "bin 1 · 400-1000 nm"),
                                  acquired_at="2026-11-20 10:12", coverage=0.98, info={})
        peek = ProfilePeek()
        peek.set_data(flatfield_peek(profile))
        return peek
    return launch(factory, "Profile Peek", (340, 320))


if __name__ == '__main__':
    main()
