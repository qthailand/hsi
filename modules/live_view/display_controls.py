"""Minimal two-mode display toggle (RAW / Enhance) for the preview bar.

Data conventions:
    The bit depth is an integer number of bits (1..32), set by the camera and shown
    read-only (``"10→16 bit"`` when the camera scales 10 real bits up to 16).

Threading:
    GUI thread only.

Example:
    >>> from modules.live_view import DisplayControls, LiveImageCanvas  # doctest: +SKIP
    >>> controls, canvas = DisplayControls(), LiveImageCanvas()          # doctest: +SKIP
    >>> controls.mapper_changed.connect(canvas.set_display_mapper)       # doctest: +SKIP
"""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QComboBox,
)

from modules.live_view.mapping import DisplayMode, DisplayMapper


class DisplayControls(QWidget):
    """Compact inline bar: [Display: RAW v] [10→16 bit].

    Args:
        bit_depth: Initial bit depth of the frames (1..32).
        parent: Optional parent widget.

    Signals:
        mapper_changed(object): a fresh :class:`DisplayMapper` (mode + bit depth)
            emitted whenever the mode changes and after :meth:`set_bit_depth`.  The
            constructor also emits it once, but before anyone can be connected.

    The bit depth comes from the camera through :meth:`set_bit_depth` and is shown
    read-only (a wrong value would mis-scale RAW display); it is hidden in Enhance
    mode.
    """

    mapper_changed = pyqtSignal(object)  # DisplayMapper

    def __init__(self, bit_depth: int = 16, parent=None):
        super().__init__(parent)

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

        self._bit_depth = int(bit_depth)
        self._sensor_bits = int(bit_depth)
        self._bit_label = QLabel()
        self._bit_label.setMinimumWidth(70)
        self._bit_label.setStyleSheet("font-size: 11px; color: #bdbdbd;")
        self._show_bit_depth()

        # ── layout ────────────────────────────────────────────────
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(lbl)
        row.addWidget(self._mode_combo)
        row.addWidget(self._bit_label)
        row.addStretch()
        self.setLayout(row)

        # ── connections ───────────────────────────────────────────
        self._mode_combo.currentIndexChanged.connect(self._on_change)

        # ── state ─────────────────────────────────────────────────
        self._on_change()

    # ── public ───────────────────────────────────────────────────

    def current_mapper(self) -> DisplayMapper:
        """Build a mapper for the current UI state.

        Returns:
            DisplayMapper: New instance (RAW or ENHANCE, with the camera's bit depth).
        """
        idx = self._mode_combo.currentIndex()
        mode = DisplayMode.RAW if idx == 0 else DisplayMode.ENHANCE
        return DisplayMapper(mode=mode, bit_depth=self._bit_depth)

    def set_bit_depth(self, bit_depth: int, sensor_bits: int | None = None):
        """Show the camera's bit depth and emit ``mapper_changed`` once.

        Args:
            bit_depth: Bit depth of the delivered frames, used for RAW mapping (1..32).
            sensor_bits: Real data bits before the camera/backend scaled them up to
                ``bit_depth`` (e.g. 10 for a 10-bit sensor in 16-bit frames); shown as
                ``"10→16 bit"``.  ``None`` means the same as ``bit_depth``.
        """
        self._bit_depth = max(1, min(int(bit_depth), 32))
        self._sensor_bits = self._bit_depth if sensor_bits is None else int(sensor_bits)
        self._show_bit_depth()
        self._on_change()

    # ── internal ─────────────────────────────────────────────────

    def _show_bit_depth(self):
        if self._sensor_bits != self._bit_depth:
            self._bit_label.setText(f"{self._sensor_bits}→{self._bit_depth} bit")
            self._bit_label.setToolTip(
                f"Camera: {self._sensor_bits}-bit data scaled up to {self._bit_depth} bits "
                f"(set by the camera, RAW mode only)")
        else:
            self._bit_label.setText(f"{self._bit_depth} bit")
            self._bit_label.setToolTip(
                f"Camera: {self._bit_depth}-bit data (set by the camera, RAW mode only)")

    def _on_change(self, *_):
        is_raw = self._mode_combo.currentIndex() == 0
        self._bit_label.setVisible(is_raw)
        self.mapper_changed.emit(self.current_mapper())
