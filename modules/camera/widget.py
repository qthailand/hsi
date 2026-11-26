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
    QButtonGroup,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QWidget,
)
from PyQt5.QtCore import Qt, pyqtSignal

_TEMPERATURE_RANGE_C = (-80.0, 100.0)    # widest range shown when the camera reports none
#: Colour of the cooling-state label per state (others grey).
_STATE_COLORS = {"stable": "#66bb6a", "low": "#ffa726", "high": "#ffa726", "alert": "#ef5350",
                 "off": "#9e9e9e"}


class CameraControls(QGroupBox):
    """Gain / exposure / binning / keep_spectrogram controls with signals.

    A ``QGroupBox`` titled "Camera" (or the model, see :meth:`set_camera_name`) holding, for a
    narrow settings column: Exposure (ms, 3 decimals) and Gain side by side, the bin as a bar of
    buttons (``1 × 1``, ``2 × 2``, ...; :attr:`bin_segments`), the temperature on one row and a
    "Keep Spectrogram" check box. It is constructible with no arguments; call
    :meth:`set_limits` and :meth:`set_bin_options` after the camera is open so the ranges match
    the sensor.

    Signals:
        params_changed: Emitted (no arguments) when gain or exposure changes through
            the user or a call that does not block signals. Not emitted by
            :meth:`set_values`.
        bin_changed: Emitted (no arguments) when the user picks another bin button. Not
            emitted by :meth:`set_bin_options` or :meth:`set_selected_bin`.
            Toggling "Keep Spectrogram" emits nothing; poll :meth:`keep_spectrogram`.
        target_temperature_changed(float): The user finished editing the target temperature
            (°C); only while cooling is supported. Not emitted by :meth:`set_target_temperature`.

    Temperature row: with :meth:`set_cooling_supported` ``True`` it is an editable
    "Target Temp"; otherwise (the default, fan-cooled cameras) a read-only "Sensor Temp"
    that :meth:`show_sensor_temperature` updates. Next to it, :meth:`set_cooling_state` shows
    the cooler's regulation state (e.g. "stable"); the label is hidden while it is ``None``.

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

        # ── binning factor: a bar of buttons, one per supported N x N (it holds the value) ──
        self.bin_segments = QWidget()
        self.bin_segments.setObjectName("binSegments")
        segments = QHBoxLayout(self.bin_segments)
        segments.setContentsMargins(0, 0, 0, 0)
        segments.setSpacing(0)
        self._bin_group = QButtonGroup(self)
        self._bin_group.setExclusive(True)
        self._bin_group.idClicked.connect(self._on_bin_segment)
        self._bins: tuple = ()
        self._bin_index = -1

        # ── temperature: target (TEC cooling) or read-only sensor value ──
        self.temp_label = QLabel()
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setDecimals(1)
        self.temp_spin.setSingleStep(1.0)
        self.temp_spin.setSuffix(" °C")
        self.temp_spin.setKeyboardTracking(False)
        self.cooling_state_label = QLabel()
        self.cooling_state_label.setToolTip("Cooler regulation state reported by the camera")
        self.cooling_state_label.hide()
        self._cooling_supported = False
        self._last_target = None

        # ── keep full-resolution spectrogram during scan ────────
        self.keep_spectrogram_check = QCheckBox("Keep Spectrogram")
        self.keep_spectrogram_check.setToolTip(
            "เก็บภาพ raw จาก sensor ขณะสแกน ทุกเฟรม\n"
            "(ก่อน dark/hot-pixel/orient/smile และก่อน spectral binning)\n"
            "จำนวนและลำดับเฟรมตรงกับ cube 1:1"
        )

        # ── layout: Exposure | Gain, the bin bar, the temperature row, Keep Spectrogram ──
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        grid.addWidget(QLabel("Exposure"), 0, 0)
        grid.addWidget(QLabel("Gain"), 0, 1)
        grid.addWidget(self.exp_spin, 1, 0)
        grid.addWidget(self.gain_spin, 1, 1)
        grid.addWidget(QLabel("Bin"), 2, 0, 1, 2)
        grid.addWidget(self.bin_segments, 3, 0, 1, 2)
        temp_row = QHBoxLayout()
        temp_row.setContentsMargins(0, 0, 0, 0)
        temp_row.addWidget(self.temp_label)
        temp_row.addWidget(self.temp_spin, stretch=1)
        temp_row.addWidget(self.cooling_state_label)
        grid.addLayout(temp_row, 4, 0, 1, 2)
        grid.addWidget(self.keep_spectrogram_check, 5, 0, 1, 2)
        self.setLayout(grid)
        self.set_cooling_supported(False)

        # ── forward changes as signals ──────────────────────────
        self.gain_spin.valueChanged.connect(self.params_changed.emit)
        self.exp_spin.valueChanged.connect(self.params_changed.emit)
        self.temp_spin.editingFinished.connect(self._on_target_edited)

    # ── the bin bar ─────────────────────────────────────────────

    def bin_options(self) -> tuple:
        """The binning factors offered, in order (``()`` before :meth:`set_bin_options`)."""
        return self._bins

    def _build_bin_buttons(self) -> None:
        layout = self.bin_segments.layout()
        for button in self._bin_group.buttons():
            self._bin_group.removeButton(button)
            layout.removeWidget(button)
            button.deleteLater()
        n = len(self._bins)
        for i, b in enumerate(self._bins):
            button = QPushButton(f"{b} \u00d7 {b}")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setToolTip(f"Bin {b} \u00d7 {b} on the sensor")
            button.setProperty("segment", "only" if n == 1 else "first" if i == 0
                               else "last" if i == n - 1 else "middle")
            self._bin_group.addButton(button, i)
            layout.addWidget(button, 1)

    def _show_bin_index(self, index: int) -> None:
        """Check button ``index`` without emitting :attr:`bin_changed`."""
        self._bin_index = int(index)
        button = self._bin_group.button(self._bin_index)
        if button is not None and not button.isChecked():
            button.setChecked(True)

    def choose_bin(self, bin_value: int) -> bool:
        """Pick ``bin_value`` as the user would (press its button): emits :attr:`bin_changed` when
        it differs from the current bin. For scripts and tests; :meth:`set_selected_bin` only shows.

        Returns:
            bool: False when ``bin_value`` is not offered (nothing happens).
        """
        if int(bin_value) not in self._bins:
            return False
        index = self._bins.index(int(bin_value))
        button = self._bin_group.button(index)
        if button is not None:
            button.setChecked(True)
        self._on_bin_segment(index)
        return True

    def _on_bin_segment(self, index: int) -> None:
        if index != self._bin_index:
            self._bin_index = int(index)
            self.bin_changed.emit()

    # ── value getters ───────────────────────────────────────────

    def gain(self) -> int:
        """Current gain in vendor units (integer)."""
        return self.gain_spin.value()

    def exposure_ms(self) -> float:
        """Current exposure time in milliseconds."""
        return self.exp_spin.value()

    def selected_bin(self) -> int:
        """Current binning factor (1 before :meth:`set_bin_options`)."""
        if 0 <= self._bin_index < len(self._bins):
            return int(self._bins[self._bin_index])
        return 1

    def keep_spectrogram(self) -> bool:
        """True if full-resolution frames should be stored during scan."""
        return self.keep_spectrogram_check.isChecked()

    def cooling_supported(self) -> bool:
        """True while the temperature row is an editable target."""
        return self._cooling_supported

    def target_temperature(self) -> float | None:
        """Target temperature in °C, or ``None`` when the camera has no cooling target."""
        return float(self.temp_spin.value()) if self._cooling_supported else None

    # ── title ──────────────────────────────────────────────────

    def set_camera_name(self, name) -> None:
        """Title the box with the camera model, or "Camera" when ``name`` is empty / ``None``."""
        name = str(name).strip() if name else ""
        self.setTitle(name or "Camera")
        self.setToolTip(f"Camera: {name}" if name else "")

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

    def set_cooling_state(self, state) -> None:
        """Show the cooler's regulation state next to the temperature, or hide it.

        Args:
            state: Short text from the camera (``"stable"``, ``"low"``, ``"high"``, ``"alert"``),
                or ``None`` when the camera reports none.
        """
        if not state:
            self.cooling_state_label.hide()
            return
        text = str(state)
        color = _STATE_COLORS.get(text.lower(), "#9e9e9e")
        self.cooling_state_label.setText(text)
        self.cooling_state_label.setStyleSheet(f"color: {color}; font-weight: 600;")
        self.cooling_state_label.show()

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
        """Offer the supported binning factors (one button each).

        Emits nothing (``bin_changed`` is for the user's picks).

        Args:
            bins: Supported binning factors (N for N x N); an empty value gives ``(1,)``.
            default_bin: Factor to select; the first entry is used if it is not in ``bins``.
        """
        bins = tuple(int(b) for b in bins) if bins else (1,)
        self._bins = bins
        self._build_bin_buttons()
        self._show_bin_index(list(bins).index(default_bin) if default_bin in bins else 0)

    # ── state ──────────────────────────────────────────────────

    def set_selected_bin(self, bin_value: int) -> None:
        """Show ``bin_value`` as the selected bin without emitting ``bin_changed``.

        Args:
            bin_value: Binning factor; ignored if it is not one of the listed options.
        """
        if int(bin_value) in self._bins:
            self._show_bin_index(self._bins.index(int(bin_value)))

    def set_bin_enabled(self, enabled: bool) -> None:
        """Enable / disable the bin buttons (disabled during scan).

        Args:
            enabled: True to let the user change the bin.
        """
        self.bin_segments.setEnabled(bool(enabled))

    def set_settings_enabled(self, enabled: bool) -> None:
        """Enable / disable gain, exposure, cooling target and Keep Spectrogram (disabled during scan).

        The bin has its own :meth:`set_bin_enabled`. A read-only sensor temperature stays
        enabled, so it remains readable.

        Args:
            enabled: True to let the user change the settings.
        """
        for widget in (self.gain_spin, self.exp_spin, self.keep_spectrogram_check):
            widget.setEnabled(enabled)
        self.temp_spin.setEnabled(enabled or not self._cooling_supported)