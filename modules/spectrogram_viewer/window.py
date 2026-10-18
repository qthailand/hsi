# windows/spectrogram_viewer.py
"""Spectrogram Viewer window — frame-by-frame full-res preview + save.

Opens as an independent top-level window.
Loads spectrogram data from RAM after scan completes.

Keyboard: ◄ ► = ±1 frame, Shift+◄ ► = ±10, Home/End = first/last, Ctrl+S = save.

Data conventions:
    A spectrogram is ``(n, h, v)`` = ``(frames, spatial, spectral)`` of any numeric
    dtype (typically uint16 raw camera frames; a ``numpy.memmap`` is fine).  Each
    frame is min-max normalised to 8 bit for display.  The viewer keeps a
    reference to the array (no copy).  Saving writes an HDF5 file via
    ``modules.cube_io.hdf5.save_hdf5`` (dataset ``"spectrogram"``).

Threading:
    The widget lives in the GUI thread.  Saving runs in an ``HDF5SaveWorker``
    ``QThread`` owned by the viewer; the array must not be modified while a save
    is running.

Example:
    >>> import numpy as np
    >>> from modules.spectrogram_viewer import SpectrogramViewer   # doctest: +SKIP
    >>> viewer = SpectrogramViewer()                                # doctest: +SKIP
    >>> viewer.load(np.random.randint(0, 4096, (20, 64, 96), dtype=np.uint16))  # doctest: +SKIP
    >>> viewer.show()                                               # doctest: +SKIP
"""

from __future__ import annotations

import logging
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QProgressDialog
)
from modules.cube_io.hdf5_worker import HDF5SaveWorker
logger = logging.getLogger(__name__)


class SpectrogramViewer(QWidget):
    """Standalone window to browse + export full-res spectrogram frames.

    A ``QWidget`` that can be shown as its own top-level window or placed in any
    layout.  Construct with no arguments, then call ``load``.  The "Save to Disk"
    button (or Ctrl+S) asks for an ``.h5`` path and writes the whole spectrogram
    in the background with a progress dialog; success and failure are reported
    with message boxes.  The widget emits no custom signals.

    Args:
        parent: Optional parent widget.

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
        self._metadata: dict | None = None
        self._index = 0
        self._save_worker: HDF5SaveWorker | None = None
        self._save_dialog: QProgressDialog | None = None
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

        # ── save button ──────────────────────────────────
        self._save_btn = QPushButton("Save to Disk")
        self._save_btn.setEnabled(False)
        self._save_btn.setStyleSheet("""
            QPushButton {
                background-color: #1565c0; color: white;
                border-radius: 6px; padding: 8px 16px;
                font-weight: bold; font-size: 13px;
            }
            QPushButton:hover { background-color: #1e88e5; }
            QPushButton:pressed { background-color: #0d47a1; }
            QPushButton:disabled { background-color: #1a2a40; color: #555; }
        """)
        self._save_btn.clicked.connect(self._on_save)

        # ── bottom row: slider + save ────────────────────
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self._slider, stretch=1)
        bottom_row.addWidget(self._save_btn)

        # ── layout ───────────────────────────────────────
        layout = QVBoxLayout()
        layout.addWidget(self._info_label)
        layout.addWidget(self._image_label, stretch=1)
        layout.addLayout(bottom_row)
        self.setLayout(layout)

    # ── public API ──────────────────────────────────────

    def load(self, spectrogram: np.ndarray, metadata: dict | None = None) -> None:
        """Load a spectrogram for browsing and jump to the first frame.

        Args:
            spectrogram: ``(n, h, v)`` = ``(frames, spatial, spectral)`` array of
                any numeric dtype.  Held by reference, not copied; every frame
                must be 2-D.  Enables the save button.
        """
        self._frames = spectrogram
        self._metadata = metadata
        self._index = 0

        n = spectrogram.shape[0]
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)

        self._save_btn.setEnabled(True)
        self._render()

    # ── save ────────────────────────────────────────────
    def _on_save(self) -> None:
        if self._frames is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Spectrogram",
            "spectrogram.h5", "HDF5 (*.h5)",
        )
        if not path:
            return

        if not path.endswith(".h5"):
            path += ".h5"

        self._save_btn.setEnabled(False)

        # ── progress dialog ──────────────────────────
        self._save_dialog = QProgressDialog(
            "Saving spectrogram...", "Cancel", 0, 100, self
        )
        self._save_dialog.setWindowTitle("Save")
        self._save_dialog.setMinimumDuration(500)
        self._save_dialog.setWindowModality(Qt.WindowModal)
        self._save_dialog.canceled.connect(self._on_save_cancel)

        # ── background worker ────────────────────────
        self._save_worker = HDF5SaveWorker(path, self._frames, self, metadata=self._metadata)
        self._save_worker.progress.connect(self._save_dialog.setValue)
        self._save_worker.finished.connect(self._on_save_done)
        self._save_worker.failed.connect(self._on_save_failed)
        self._save_worker.start()

    def _on_save_done(self, path: str):
        self._save_btn.setEnabled(True)
        self._save_dialog = None
        self._save_worker = None
        QMessageBox.information(
            self, "Saved",
            f"Saved to\n{path}\n\n"
            f"Shape: {self._frames.shape}",
        )

    def _on_save_failed(self, msg: str):
        self._save_btn.setEnabled(True)
        self._save_dialog = None
        self._save_worker = None
        QMessageBox.critical(self, "Save failed", msg)

    def _on_save_cancel(self):
        if self._save_worker is not None:
            self._save_worker.requestInterruption()
            self._save_worker.wait(3000)
        self._save_worker = None
        self._save_dialog = None
        self._save_btn.setEnabled(True)
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
        """Re-render the current frame to fit the new size (Qt override)."""
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
        """Keyboard navigation (Qt override).

        Left/Right step one frame (ten with Shift), Home/End jump to the first/last
        frame and Ctrl+S starts a save; other keys go to the base class.
        """
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
        elif key == Qt.Key_S and event.modifiers() & Qt.ControlModifier:
            self._on_save()
        else:
            super().keyPressEvent(event)