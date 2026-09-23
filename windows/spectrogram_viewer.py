# windows/spectrogram_viewer.py
"""Spectrogram Viewer window — frame-by-frame full-res preview.

Opens as an independent top-level window.
Loads spectrogram data from RAM after scan completes.

Keyboard: ◄ ► = ±1 frame, Shift+◄ ► = ±10, Home/End = first/last.
"""

from __future__ import annotations

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QGroupBox,
    QScrollArea,
)


class SpectrogramViewer(QWidget):
    """Standalone window to browse full-res spectrogram frames.

    Usage:
        viewer = SpectrogramViewer()
        viewer.load(spectrogram_array)   # (n, h, v) uint16
        viewer.show()
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrogram Viewer")
        self.resize(1100, 700)
        self.setFocusPolicy(Qt.StrongFocus)

        self._frames: np.ndarray | None = None
        self._index = 0

        # ── image ────────────────────────────────────────
        self._image_label = QLabel("No data")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(500, 400)
        self._image_label.setStyleSheet("background: #0d1117;")

        # ── slider ───────────────────────────────────────
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)

        # ── info bar ─────────────────────────────────────
        self._info_label = QLabel()
        self._info_label.setStyleSheet("font-size: 12px; color: #aaa;")
        self._info_label.setAlignment(Qt.AlignCenter)

        # ── layout ───────────────────────────────────────
        layout = QVBoxLayout()
        layout.addWidget(self._info_label)
        layout.addWidget(self._image_label, stretch=1)
        layout.addWidget(self._slider)
        self.setLayout(layout)

    # ── public API ──────────────────────────────────────

    def load(self, spectrogram: np.ndarray) -> None:
        """Load (n, h, v) array for browsing."""
        self._frames = spectrogram
        self._index = 0

        n = spectrogram.shape[0]
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)

        self._render()

    # ── rendering ───────────────────────────────────────

    def _render(self) -> None:
        if self._frames is None:
            return

        frame = self._frames[self._index]
        h, w = frame.shape

        norm = frame.astype(np.float32)
        lo, hi = norm.min(), norm.max()
        if hi > lo:
            norm = (norm - lo) / (hi - lo)
        gray8 = (norm * 255).astype(np.uint8)

        qi = QImage(gray8.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qi)
        self._image_label.setPixmap(pixmap.scaled(
            self._image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        ))

        self._info_label.setText(
            f"Frame {self._index + 1} / {self._frames.shape[0]}"
            f"    ·    {h}×{w}"
            f"    ·    {frame.dtype}"
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    # ── navigation ──────────────────────────────────────

    def _on_slider(self, value: int) -> None:
        self._index = int(value)
        self._render()

    def _go(self, delta: int) -> None:
        if self._frames is None:
            return
        n = self._frames.shape[0]
        new = max(0, min(n - 1, self._index + delta))
        self._slider.setValue(new)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        step = 10 if event.modifiers() & Qt.ShiftModifier else 1

        if key == Qt.Key_Left:
            self._go(-step)
        elif key == Qt.Key_Right:
            self._go(step)
        elif key == Qt.Key_Home:
            self._go(-self._index)
        elif key == Qt.Key_End:
            self._go(self._frames.shape[0] - 1 - self._index
                     if self._frames is not None else 0)
        else:
            super().keyPressEvent(event)