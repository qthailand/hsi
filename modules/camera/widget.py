# widgets/camera_controls.py
"""Camera parameter controls widget: gain, exposure, binning, temperature, keep_spectrogram.

Self-contained widget that emits signals when the user changes parameters.
It has no backend coupling; the host connects the signals to its camera calls
(for example ``AcquisitionEngine.request_params`` / ``request_bin`` from
:mod:`modules.acquisition`).

Data conventions:
    Exposure is in **milliseconds** (``exposure_ms``), gain is an integer in vendor
    units, binning is an integer N for N x N hardware binning, temperatures are °C.

Threading:
    GUI thread only. The widget never touches a camera; read values with the getters
    in a slot and forward them to the acquisition worker.

Example:
    >>> # controls = CameraControls()
    >>> # controls.set_limits(0, 255, 1.0, 1000.0)
    >>> # controls.set_bin_options((1, 2, 4), default_bin=2)
    >>> # controls.params_changed.connect(lambda: apply(controls.gain(), controls.exposure_ms()))
    >>> # controls.bin_changed.connect(lambda: apply_bin(controls.selected_bin()))
    >>> # controls.set_cooling_supported(camera.cooling_supported(), camera.target_temperature_range())
    >>> # controls.target_temperature_changed.connect(engine.request_target_temperature)
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QGroupBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
)
from PyQt5.QtCore import pyqtSignal

_TEMPERATURE_RANGE_C = (-80.0, 100.0)    # widest range shown when the camera reports none


class CameraControls(QGroupBox):
    """Gain / exposure / binning / keep_spectrogram controls with signals.

    A ``QGroupBox`` titled "Camera" holding a gain spin box, an exposure spin box
    (ms, 3 decimals), a bin combo box and a "Keep Spectrogram" check box. It is
    constructible with no arguments; call :meth:`set_limits` and
    :meth:`set_bin_options` after the camera is open so the ranges match the sensor.

    Signals:
        params_changed: Emitted (no arguments) when gain or exposure changes through
            the user or a call that does not block signals. Not emitted by
            :meth:`set_values`.
        bin_changed: Emitted (no arguments) when the bin combo selection changes. Not
            emitted by :meth:`set_bin_options` or :meth:`set_selected_bin`.
            Toggling "Keep Spectrogram" emits nothing; poll :meth:`keep_spectrogram`.
        target_temperature_changed(float): The user finished editing the target temperature
            (°C); only while cooling is supported. Not emitted by :meth:`set_target_temperature`.

    Temperature row: with :meth:`set_cooling_supported` ``True`` it is an editable
    "Target Temp"; otherwise (the default, fan-cooled cameras) a read-only "Sensor Temp"
    that :meth:`show_sensor_temperature` updates.

    The host must provide: the actual camera calls (connect the signals) and the
    hardware limits and bin list. Use from the GUI thread only.
    """

    params_changed = pyqtSignal()
    bin_changed = pyqtSignal()
    target_temperature_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        """Build the controls with default ranges (gain 1-300, exposure 1-1000 ms).

        Args:
            parent: Optional parent widget.
        """
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

        # ── temperature: target (TEC cooling) or read-only sensor value ──
        self.temp_label = QLabel()
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setDecimals(1)
        self.temp_spin.setSingleStep(1.0)
        self.temp_spin.setSuffix(" °C")
        self.temp_spin.setKeyboardTracking(False)
        self._cooling_supported = False
        self._last_target = None

        # ── keep full-resolution spectrogram during scan ────────
        self.keep_spectrogram_check = QCheckBox("Keep Spectrogram")
        self.keep_spectrogram_check.setToolTip(
            "เก็บภาพ raw จาก sensor ขณะสแกน ทุกเฟรม\n"
            "(ก่อน dark/hot-pixel/orient/smile และก่อน spectral binning)\n"
            "จำนวนและลำดับเฟรมตรงกับ cube 1:1"
        )

        # ── layout ──────────────────────────────────────────────
        layout = QFormLayout()
        layout.addRow("Gain", self.gain_spin)
        layout.addRow("Exposure", self.exp_spin)
        layout.addRow("Bin", self.bin_combo)
        layout.addRow(self.temp_label, self.temp_spin)
        layout.addRow("", self.keep_spectrogram_check)
        self.setLayout(layout)
        self.set_cooling_supported(False)

        # ── forward changes as signals ──────────────────────────
        self.gain_spin.valueChanged.connect(self.params_changed.emit)
        self.exp_spin.valueChanged.connect(self.params_changed.emit)
        self.bin_combo.currentIndexChanged.connect(self.bin_changed.emit)
        self.temp_spin.editingFinished.connect(self._on_target_edited)

    # ── value getters ───────────────────────────────────────────

    def gain(self) -> int:
        """Current gain in vendor units (integer)."""
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

    def cooling_supported(self) -> bool:
        """True while the temperature row is an editable target."""
        return self._cooling_supported

    def target_temperature(self) -> float | None:
        """Target temperature in °C, or ``None`` when the camera has no cooling target."""
        return float(self.temp_spin.value()) if self._cooling_supported else None

    # ── temperature ────────────────────────────────────────────

    def set_cooling_supported(self, supported: bool, value_range=None) -> None:
        """Make the temperature row an editable target (TEC) or a read-only sensor value.

        Args:
            supported: True if the camera regulates to a target temperature.
            value_range: ``(min, max)`` °C settable target, or ``None`` for a wide default.
        """
        self._cooling_supported = bool(supported)
        low, high = value_range if value_range else _TEMPERATURE_RANGE_C
        self.temp_spin.blockSignals(True)
        self.temp_spin.setRange(float(low), float(high))
        self.temp_spin.setReadOnly(not supported)
        self.temp_spin.setButtonSymbols(
            QAbstractSpinBox.UpDownArrows if supported else QAbstractSpinBox.NoButtons)
        if not supported:
            self.temp_spin.setSpecialValueText("—")      # shown at the minimum = unknown
            self.temp_spin.setValue(self.temp_spin.minimum())
        else:
            self.temp_spin.setSpecialValueText("")
        self.temp_spin.blockSignals(False)
        self.temp_label.setText("Target Temp" if supported else "Sensor Temp")
        self.temp_spin.setToolTip(
            "Cooler target temperature (the cooler is switched on when set)" if supported
            else "This camera has no cooling target (fan-cooled); current sensor temperature")

    def set_target_temperature(self, temp_c: float) -> None:
        """Show ``temp_c`` as the target without emitting ``target_temperature_changed``."""
        self.temp_spin.blockSignals(True)
        self.temp_spin.setValue(float(temp_c))
        self.temp_spin.blockSignals(False)
        self._last_target = float(self.temp_spin.value())

    def show_sensor_temperature(self, temp_c: float) -> None:
        """Show the current sensor temperature in the read-only row (ignored while editable).

        Args:
            temp_c: °C; below ``-900`` (``-999``) shows "—".
        """
        if self._cooling_supported:
            return
        self.temp_spin.blockSignals(True)
        self.temp_spin.setValue(float(temp_c) if temp_c > -900 else self.temp_spin.minimum())
        self.temp_spin.blockSignals(False)

    def _on_target_edited(self) -> None:
        if not self._cooling_supported:
            return
        value = float(self.temp_spin.value())
        if value == self._last_target:
            return
        self._last_target = value
        self.target_temperature_changed.emit(value)

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
        the actual sensor capability. Values outside the new range are clamped by
        the spin boxes, which emits ``params_changed``.

        Args:
            gain_min: Smallest gain (vendor units).
            gain_max: Largest gain (vendor units).
            exp_min_ms: Smallest exposure in milliseconds.
            exp_max_ms: Largest exposure in milliseconds (convert from the camera's
                ``exposure_range_us`` by dividing by 1000).
        """
        self.gain_spin.setRange(gain_min, gain_max)
        self.exp_spin.setRange(exp_min_ms, exp_max_ms)

    def set_values(self, gain: int, exposure_ms: float) -> None:
        """Set gain/exposure (clamped to the current limits) without emitting signals.

        Args:
            gain: Gain in vendor units.
            exposure_ms: Exposure in milliseconds.
        """
        for spin, value in ((self.gain_spin, gain), (self.exp_spin, exposure_ms)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def set_bin_options(
        self, bins: tuple[int, ...], default_bin: int
    ) -> None:
        """Populate the bin combo with supported binning factors.

        Signals are blocked during population to avoid triggering
        ``bin_changed`` before the caller is ready.

        Args:
            bins: Supported binning factors (N for N x N); an empty value gives ``(1,)``.
            default_bin: Factor to select; the first entry is used if it is not in ``bins``.
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

    def set_selected_bin(self, bin_value: int) -> None:
        """Show ``bin_value`` in the combo without emitting ``bin_changed``.

        Args:
            bin_value: Binning factor; ignored if it is not one of the listed options.
        """
        index = self.bin_combo.findData(int(bin_value))
        if index < 0 or index == self.bin_combo.currentIndex():
            return
        self.bin_combo.blockSignals(True)
        self.bin_combo.setCurrentIndex(index)
        self.bin_combo.blockSignals(False)

    def set_bin_enabled(self, enabled: bool) -> None:
        """Enable / disable bin combo (disabled during scan).

        Args:
            enabled: True to let the user change the bin.
        """
        self.bin_combo.setEnabled(enabled)