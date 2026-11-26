# windows/image_viewer.py
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from modules.live_view.image_canvas import LiveImageCanvas


class ImageViewer(QWidget):
    start_scan_requested = pyqtSignal()
    stop_scan_requested = pyqtSignal()
    line_y_changed = pyqtSignal(int)
    #: :meth:`set_scanning_ui` was called: True = scanning (Start off, Stop on). Lets a host
    #: that shows its own Start / Stop buttons follow the same state.
    scanning_ui_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()

        self.image = LiveImageCanvas()
        self.image.y_changed.connect(self.line_y_changed.emit)

        self.start_btn = self.image.add_overlay_button(
            "\u25b6 Start Scan",
        )
        self.stop_btn = self.image.add_overlay_button(
            "\u25a0 Stop Scan",
        )
        self.stop_btn.setEnabled(False)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.image)
        self.setLayout(layout)

        self.start_btn.clicked.connect(self.start_scan_requested.emit)
        self.stop_btn.clicked.connect(self.stop_scan_requested.emit)

    def use_scan_buttons(self, start_btn, stop_btn):
        """Let ``start_btn`` / ``stop_btn`` (a host's own, e.g. pinned under its settings) be the
        Start / Stop buttons: they emit :attr:`start_scan_requested` / :attr:`stop_scan_requested`
        and :meth:`set_scanning_ui` enables them. The buttons on the image are removed."""
        start_enabled, stop_enabled = self.start_btn.isEnabled(), self.stop_btn.isEnabled()
        for button in (self.start_btn, self.stop_btn):
            self.image.remove_overlay_button(button)
        self.start_btn, self.stop_btn = start_btn, stop_btn
        start_btn.setEnabled(start_enabled)
        stop_btn.setEnabled(stop_enabled)
        start_btn.clicked.connect(self.start_scan_requested.emit)
        stop_btn.clicked.connect(self.stop_scan_requested.emit)

    def set_scanning_ui(self, scanning: bool):
        self.start_btn.setEnabled(not scanning)
        self.stop_btn.setEnabled(scanning)
        self.scanning_ui_changed.emit(bool(scanning))

    def set_image(self, frame: np.ndarray):
        self.image.set_array(frame)

    def set_selected_y(self, y: int):
        self.image.set_selected_y(y)

    def selected_y(self) -> int:
        return int(self.image.selected_y)

    def set_focus_roi_height(self, height: int):
        self.image.set_focus_roi_height(height)

    def focus_roi_range(self) -> tuple[int, int]:
        return self.image.focus_roi_range()

    def full_focus_roi_height(self):
        self.image.full_height()
