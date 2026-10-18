"""Stand-alone demo: ``python -m modules.cube_preview`` (synthetic scan fed frame by frame)."""
from __future__ import annotations

import sys

from modules._support import launch


def _demo():
    """Preview widget that receives one synthetic frame every 30 ms."""
    import numpy as np
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication

    from modules.cube_preview import RealtimeCubePreviewWidget

    slit, bands, n_frames = 120, 100, 200
    wavelengths = np.linspace(400.0, 1000.0, bands)
    widget = RealtimeCubePreviewWidget()
    state = {"i": 0}

    def push():
        i = state["i"]
        if i >= n_frames:
            timer.stop()
            return
        rows = np.linspace(0.2, 1.0, slit)[:, None]
        spectral = 1000 + 3000 * np.exp(-((wavelengths - 400 - 3 * i) / 150.0) ** 2)[None, :]
        widget.push_frame((rows * spectral).astype(np.uint16), wavelengths, bands)
        state["i"] += 1

    timer = QTimer(widget)
    timer.timeout.connect(push)
    timer.start(30)
    QApplication.instance().aboutToQuit.connect(widget.cleanup)
    return widget


def main() -> int:
    """Open the demo window and run the Qt event loop."""
    return launch(_demo, "cube_preview demo", size=(900, 800))


if __name__ == "__main__":
    sys.exit(main())
