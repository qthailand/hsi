# widgets/camera_controls.py
"""Camera parameter controls widget — gain, exposure, binning, keep_spectrogram.

Self-contained widget that emits signals when user changes parameters.
No direct backend coupling — the controller connects these signals
to the appropriate backend calls.

Usage:
    controls = CameraControls()
    controls.set_limits(0, 255, 1.0, 1000.0)
    controls.set_bin_options((1, 2, 4), default_bin=2)
    controls.params_changed.connect(controller.apply_params)
    controls.bin_changed.connect(controller.apply_bin)
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QGroupBox,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
)
from PyQt5.QtCore import pyqtSignal


class CameraControls(QGroupBox):
    """Gain / exposure / binning / keep_spectrogram controls with signals.

    Signals:
        params_changed: Emitted when gain or exposure value changes.
        bin_changed:    Emitted when the bin combo selection changes.
    """

    params_changed = pyqtSignal()
    bin_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Camera", parent)

        # ── gain ────────────────────────────────────────────────
        self.gain_spin = QSpinBox()
        self.gain_spin.setRange(1, 300)

        # ── exposure (ms) ───────────────────────────────────────
        self.exp_spin = QDoubleSpinBox()
        self.exp_spin.setDecimals(3)
        self.exp_spin.setRange(1, 1000)
        self.exp_spin.setSuffix(" ms")

        # ── binning factor ──────────────────────────────────────
        self.bin_combo = QComboBox()

        # ── keep full-resolution spectrogram during scan ────────
        self.keep_spectrogram_check = QCheckBox("Keep Spectrogram")
        self.keep_spectrogram_check.setToolTip(
            "เก็บ full frame จาก sensor ขณะแสกน\n"
            "(ไม่ผ่าน spectral binning)"
        )

        # ── layout ──────────────────────────────────────────────
        layout = QFormLayout()
        layout.addRow("Gain", self.gain_spin)
        layout.addRow("Exposure", self.exp_spin)
        layout.addRow("Bin", self.bin_combo)
        layout.addRow("", self.keep_spectrogram_check)
        self.setLayout(layout)

        # ── forward changes as signals ──────────────────────────
        self.gain_spin.valueChanged.connect(self.params_changed.emit)
        self.exp_spin.valueChanged.connect(self.params_changed.emit)
        self.bin_combo.currentIndexChanged.connect(self.bin_changed.emit)

    # ── value getters ───────────────────────────────────────────

    def gain(self) -> int:
        """Current gain value."""
        return self.gain_spin.value()

    def exposure_ms(self) -> float:
        """Current exposure time in milliseconds."""
        return self.exp_spin.value()

    def selected_bin(self) -> int:
        """Current binning factor from the combo (1 if unset)."""
        value = self.bin_combo.currentData()
        return int(value) if value is not None else 1

    def keep_spectrogram(self) -> bool:
        """True if full-resolution frames should be stored during scan."""
        return self.keep_spectrogram_check.isChecked()

    # ── configuration ──────────────────────────────────────────

    def set_limits(
        self,
        gain_min: int,
        gain_max: int,
        exp_min_ms: float,
        exp_max_ms: float,
    ) -> None:
        """Apply hardware limits to the spin boxes.

        Called after camera open so that UI ranges match
        the actual sensor capability.
        """
        self.gain_spin.setRange(gain_min, gain_max)
        self.exp_spin.setRange(exp_min_ms, exp_max_ms)

    def set_bin_options(
        self, bins: tuple[int, ...], default_bin: int
    ) -> None:
        """Populate the bin combo with supported binning factors.

        Signals are blocked during population to avoid triggering
        ``bin_changed`` before the caller is ready.
        """
        bins = tuple(int(b) for b in bins) if bins else (1,)
        self.bin_combo.blockSignals(True)
        self.bin_combo.clear()
        for b in bins:
            self.bin_combo.addItem(str(b), b)
        idx = list(bins).index(default_bin) if default_bin in bins else 0
        self.bin_combo.setCurrentIndex(idx)
        self.bin_combo.blockSignals(False)

    # ── state ──────────────────────────────────────────────────

    def set_bin_enabled(self, enabled: bool) -> None:
        """Enable / disable bin combo (disabled during scan)."""
        self.bin_combo.setEnabled(enabled)