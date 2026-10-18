"""Stand-alone demo: ``python -m modules.ui_kit`` (RangeSlider with a live read-out)."""
from __future__ import annotations

import sys

from modules._support import launch


def _demo():
    """Build a small panel with a RangeSlider and a label showing its range."""
    from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

    from modules.ui_kit import RangeSlider

    panel = QWidget()
    layout = QVBoxLayout(panel)
    slider = RangeSlider(low=0.02, high=0.98)
    label = QLabel("range: 2.0% - 98.0%")
    slider.range_changed.connect(lambda lo, hi: label.setText(f"range: {lo * 100:.1f}% - {hi * 100:.1f}%"))
    layout.addWidget(slider)
    layout.addWidget(label)
    layout.addStretch()
    return panel


def main() -> int:
    """Open the demo window and run the Qt event loop."""
    return launch(_demo, "ui_kit demo", size=(480, 160))


if __name__ == "__main__":
    sys.exit(main())
