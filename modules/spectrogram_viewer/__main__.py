"""Stand-alone demo: ``python -m modules.spectrogram_viewer``.

Shows the viewer with a synthetic ``(n, h, v)`` uint16 spectrogram.
"""
from __future__ import annotations


def synthetic_spectrogram(n: int = 40, h: int = 128, v: int = 160):
    """Build a demo spectrogram.

    Returns:
        numpy.ndarray: uint16 array ``(n, h, v)`` = ``(frames, spatial, spectral)``.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:h, 0:v]
    base = 500.0 + 2000.0 * np.exp(-((x - v / 2) / (v / 6)) ** 2)
    frames = [base * (0.6 + 0.4 * np.sin(2 * np.pi * i / n + y / 20.0)) + rng.normal(0, 30, (h, v))
              for i in range(n)]
    return np.clip(np.stack(frames), 0, 65535).astype(np.uint16)


def build_viewer():
    """Create a ``SpectrogramViewer`` loaded with the synthetic spectrogram (needs a ``QApplication``)."""
    from modules.spectrogram_viewer import SpectrogramViewer

    viewer = SpectrogramViewer()
    viewer.load(synthetic_spectrogram())
    return viewer


def main(argv=None) -> int:
    """Show the demo viewer and run the Qt event loop; returns the exit code."""
    from modules._support import launch

    return launch(build_viewer, "Spectrogram Viewer", (1100, 700))


if __name__ == "__main__":
    raise SystemExit(main())
