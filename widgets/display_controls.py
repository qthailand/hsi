"""Minimal two-mode display toggle for the preview bar."""

from __future__ import annotations

import qtawesome as qta
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QPushButton,
)

from utils.display import DisplayMode, DisplayMapper


class DisplayControls(QWidget):
    """Compact inline bar: [Display: RAW ▾] [16 bit] [⟳]

    Signals
    -------
    mapper_changed(DisplayMapper) — emitted on every change.
    """

    mapper_changed = pyqtSignal(object)  # DisplayMapper

    def __init__(self, bit_depth: int = 16, parent=None):
        super().__init__(parent)

        self._suppress = False

        # ── widgets ───────────────────────────────────────────────
        lbl = QLabel("Display:")
        lbl.setStyleSheet("font-size: 11px; color: #9e9e9e;")

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["RAW", "Enhance"])
        self._mode_combo.setToolTip(
            "RAW: linear map to 8-bit (preserves true brightness)\n"
            "Enhance: min-max stretch (maximises contrast)"
        )
        self._mode_combo.setFixedWidth(90)

        self._bit_spin = QSpinBox()
        self._bit_spin.setRange(1, 32)
        self._bit_spin.setValue(bit_depth)
        self._bit_spin.setSuffix(" bit")
        self._bit_spin.setFixedWidth(70)
        self._bit_spin.setToolTip("Sensor bit depth (RAW mode only)")

        self._reset_btn = QPushButton()
        self._reset_btn.setIcon(qta.icon("fa5s.undo", color="#bdbdbd"))
        self._reset_btn.setFixedSize(28, 28)
        self._reset_btn.setToolTip("Reset")

        # ── layout ────────────────────────────────────────────────
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(lbl)
        row.addWidget(self._mode_combo)
        row.addWidget(self._bit_spin)
        row.addWidget(self._reset_btn)
        row.addStretch()
        self.setLayout(row)

        # ── connections ───────────────────────────────────────────
        self._mode_combo.currentIndexChanged.connect(self._on_change)
        self._bit_spin.valueChanged.connect(self._on_change)
        self._reset_btn.clicked.connect(self._reset)

        # ── state ─────────────────────────────────────────────────
        self._on_change()

    # ── public ───────────────────────────────────────────────────

    def current_mapper(self) -> DisplayMapper:
        idx = self._mode_combo.currentIndex()
        mode = DisplayMode.RAW if idx == 0 else DisplayMode.ENHANCE
        return DisplayMapper(mode=mode, bit_depth=self._bit_spin.value())

    def set_bit_depth(self, bit_depth: int):
        self._suppress = True
        self._bit_spin.setValue(int(bit_depth))
        self._suppress = False
        self._on_change()

    # ── internal ─────────────────────────────────────────────────

    def _on_change(self, *_):
        if self._suppress:
            return
        is_raw = self._mode_combo.currentIndex() == 0
        self._bit_spin.setVisible(is_raw)
        self.mapper_changed.emit(self.current_mapper())

    def _reset(self):
        self._suppress = True
        self._mode_combo.setCurrentIndex(0)
        self._bit_spin.setValue(16)
        self._suppress = False
        self._on_change()