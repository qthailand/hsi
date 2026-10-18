"""Stand-alone demo: ``python -m modules.live_view`` (canvas + display controls + live spectrum)."""
from __future__ import annotations

import sys

from modules._support import launch


def _demo():
    """Synthetic horizontal-gradient frame shown on the canvas, with its selected-row spectrum."""
    import numpy as np
    from PyQt5.QtWidgets import QVBoxLayout, QWidget

    from modules.live_view import DisplayControls, LiveImageCanvas, LiveLineSpectrumWidget

    rows, cols = 240, 320
    wavelengths = np.linspace(400.0, 1000.0, cols)
    frame = np.tile(np.linspace(0, 60000, cols), (rows, 1))
    frame += np.linspace(0, 5000, rows)[:, None]
    frame = frame.astype(np.uint16)

    panel = QWidget()
    layout = QVBoxLayout(panel)
    controls = DisplayControls()
    canvas = LiveImageCanvas()
    spectrum = LiveLineSpectrumWidget()
    layout.addWidget(controls)
    layout.addWidget(canvas, stretch=2)
    layout.addWidget(spectrum, stretch=1)

    controls.mapper_changed.connect(lambda mapper: (canvas.set_display_mapper(mapper), canvas.set_array(frame)))
    canvas.y_changed.connect(lambda y: spectrum.update_spectrum(frame[y], wavelengths))
    canvas.set_array(frame)
    spectrum.update_spectrum(frame[canvas.selected_y], wavelengths)
    return panel


def main() -> int:
    """Open the demo window and run the Qt event loop."""
    return launch(_demo, "live_view demo", size=(800, 800))


if __name__ == "__main__":
    sys.exit(main())
