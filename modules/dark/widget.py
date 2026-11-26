"""Dark Frame status panel with colored indicator and qtawesome icons.

``DarkFramePanel`` shows whether a dark reference is active and offers Capture / Cancel /
Reload buttons plus an "Apply Dark Subtraction" toggle. It performs no capture itself:
the host (see :class:`modules.dark.DarkFrameLibrary` and
:func:`modules.dark.select_dark`) reacts to the signals and pushes state back with the
``set_*`` methods. It is a view of a :class:`~modules.dark.state.DarkFrameState`
(:attr:`DarkFramePanel.model`), which holds the state; a host that shows the dark elsewhere
(a status row, a peek) can use that object alone.

Data conventions:
    Exposure is in milliseconds, gain in vendor units, bin is the hardware bin factor.

Threading:
    GUI thread only. Requires PyQt5 and ``qtawesome`` (icons).

Example:
    >>> # panel = DarkFramePanel()
    >>> # panel.capture_requested.connect(start_capture)   # host calls library.start_capture
    >>> # panel.set_capturing(0, 50, bin_value=2, exposure_ms=10.0, gain=100)
    >>> # panel.update_progress(25, 50)
    >>> # panel.set_loaded(2, 10.0, 100, n_frames=50, n_hot=12, temperature_c=25.3)
    >>> # panel.set_current_temperature(26.5)              # shows "ΔT +1.2"
"""

from __future__ import annotations

import qtawesome as qta
from PyQt5.QtCore import Qt, pyqtSignal, QSize

from .state import DarkFrameState
from PyQt5.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QCheckBox,
)


_DOT_RED = "#ef5350"
_DOT_GREEN = "#66bb6a"
_DOT_AMBER = "#ffa726"
_DOT_GRAY = "#616161"
_DOT_SIZE = 16


class DarkFramePanel(QGroupBox):
    """Self-contained dark-frame status + control panel.

    A ``QGroupBox`` titled "Dark Frame", constructible with no arguments. It starts in the
    ``IDLE`` state. The host provides the actual capture/load logic and the camera
    settings; the panel only displays state and emits requests. GUI thread only.

    Signals:
        capture_requested: Capture button clicked (no arguments).
        cancel_requested: Cancel button clicked (only visible while capturing).
        reload_requested: Reload button clicked (load the dark from disk again).
        dark_toggled(bool): "Apply Dark Subtraction" check box toggled; the argument is
            the new state. The box starts checked and no signal is emitted at start-up.

    Class attributes ``IDLE``, ``LOADED``, ``CAPTURING`` and ``ERROR`` are the state names.

    Attributes:
        model (DarkFrameState): The state it shows; its ``set_*`` methods are the panel's.
    """

    capture_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    reload_requested = pyqtSignal()
    dark_toggled = pyqtSignal(bool)  # ★ เพิ่ม signal

    IDLE = "idle"
    LOADED = "loaded"
    CAPTURING = "capturing"
    ERROR = "error"

    def __init__(self, parent=None):
        """Build the panel in the idle state with subtraction enabled.

        Args:
            parent: Optional parent widget.
        """
        super().__init__("Dark Frame", parent)
        self.model = DarkFrameState(self)

        # --- status dot ---
        self._dot_label = QLabel()
        self._dot_label.setFixedSize(QSize(_DOT_SIZE + 4, _DOT_SIZE + 4))
        self._dot_label.setAlignment(Qt.AlignCenter)

        # --- text labels ---
        self._title_label = QLabel("No dark frame")
        self._title_label.setStyleSheet("font-weight: 600;")

        self._detail_label = QLabel()
        self._detail_label.setStyleSheet("font-size: 11px; color: #9e9e9e;")

        self._extra_label = QLabel()
        self._extra_label.setStyleSheet("font-size: 11px; color: #9e9e9e;")

        text_col = QVBoxLayout()
        text_col.setSpacing(1)
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.addWidget(self._title_label)
        text_col.addWidget(self._detail_label)
        text_col.addWidget(self._extra_label)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        top_row.addWidget(self._dot_label, alignment=Qt.AlignTop)
        top_row.addLayout(text_col, stretch=1)

        # ★ dark toggle checkbox
        self._dark_toggle = QCheckBox("Apply Dark Subtraction")
        self._dark_toggle.setChecked(True)
        self._dark_toggle.setStyleSheet(
            "font-size: 11px; color: #e0e0e0;"
        )
        self._dark_toggle.toggled.connect(self.model.set_dark_enabled)

        # --- progress ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setStyleSheet(
            """
            QProgressBar {
                background: #2a2a2a;
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: #ffa726;
                border-radius: 3px;
            }
            """
        )

        self._progress_label = QLabel()
        self._progress_label.setAlignment(Qt.AlignCenter)
        self._progress_label.setStyleSheet(
            "font-size: 11px; color: #ffa726; font-weight: 600;"
        )

        # --- buttons ---
        self._capture_btn = QPushButton("  Capture")
        self._capture_btn.setIcon(
            qta.icon("fa5s.camera", color="#e0e0e0"),
        )
        self._capture_btn.setToolTip(
            "Capture dark frames with current camera settings"
        )

        self._cancel_btn = QPushButton("  Cancel")
        self._cancel_btn.setIcon(
            qta.icon("fa5s.stop-circle", color=_DOT_RED),
        )
        self._cancel_btn.setToolTip("Cancel dark frame capture")

        self._reload_btn = QPushButton("  Reload")
        self._reload_btn.setIcon(
            qta.icon("fa5s.sync-alt", color="#90caf9"),
        )
        self._reload_btn.setToolTip("Reload dark frame from disk")

        for btn in (self._capture_btn, self._cancel_btn, self._reload_btn):
            btn.setIconSize(QSize(14, 14))

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.addWidget(self._capture_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._reload_btn)
        btn_row.addStretch()

        # --- assemble ---
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addLayout(top_row)
        layout.addWidget(self._dark_toggle)  # ★ เพิ่มตรงนี้
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._progress_label)
        layout.addLayout(btn_row)
        self.setLayout(layout)

        # --- connections ---
        self._capture_btn.clicked.connect(self.model.capture_requested.emit)
        self._cancel_btn.clicked.connect(self.model.cancel_requested.emit)
        self._reload_btn.clicked.connect(self.model.reload_requested.emit)
        self.model.capture_requested.connect(self.capture_requested.emit)
        self.model.cancel_requested.connect(self.cancel_requested.emit)
        self.model.reload_requested.connect(self.reload_requested.emit)
        self.model.dark_toggled.connect(self.dark_toggled.emit)
        self.model.changed.connect(self._render)

        # --- initial ---
        self._render()

    # ── public API (the model's) ────────────────────────────────────

    def is_dark_enabled(self) -> bool:
        """Whether *Apply Dark Subtraction* is on."""
        return self.model.is_dark_enabled()

    def set_dark_enabled(self, enabled: bool) -> None:
        """Tick or untick *Apply Dark Subtraction* (emits ``dark_toggled`` when it changes)."""
        self.model.set_dark_enabled(enabled)

    def set_idle(self):
        """Show "No dark frame" (grey): nothing loaded and nothing capturing."""
        self.model.set_idle()

    def set_loaded(self, bin_value: int, exposure_ms: float, gain: float, n_frames: int, n_hot: int,
                   temperature_c: float | None = None, current_temperature_c: float | None = None):
        """Show that a dark reference is active (green); see :meth:`DarkFrameState.set_loaded`."""
        self.model.set_loaded(bin_value, exposure_ms, gain, n_frames, n_hot, temperature_c,
                              current_temperature_c)

    def set_current_temperature(self, temperature_c: float):
        """Update the current sensor temperature (°C) used for the ΔT of the loaded dark."""
        self.model.set_current_temperature(temperature_c)

    def set_capturing(self, collected: int, total: int, bin_value: int | None = None,
                      exposure_ms: float | None = None, gain: float | None = None):
        """Show capture progress (amber) with a progress bar; the Cancel button appears."""
        self.model.set_capturing(collected, total, bin_value, exposure_ms, gain)

    def update_progress(self, collected: int, total: int):
        """Frames collected so far (ignored unless capturing)."""
        self.model.update_progress(collected, total)

    def set_error(self, message: str = "Load failed"):
        """Show an error title (red), e.g. when the stored dark does not fit the camera."""
        self.model.set_error(message)

    def is_capturing(self) -> bool:
        """True while the panel is in the capturing state."""
        return self.model.is_capturing()

    def state(self) -> str:
        """The state: :attr:`IDLE`, :attr:`LOADED`, :attr:`CAPTURING` or :attr:`ERROR`."""
        return self.model.state()

    def is_interactive(self) -> bool:
        """Whether Capture and Reload may be used now (:meth:`set_interactive`)."""
        return self.model.is_interactive()

    def capture_progress(self) -> tuple:
        """``(collected, total)`` frames of the capture in progress, ``(0, 0)`` when none runs."""
        return self.model.capture_progress()

    def set_interactive(self, enabled: bool):
        """Enable/disable Capture and Reload (e.g. during a scan); Cancel stays enabled."""
        self.model.set_interactive(enabled)

    # ── drawing ─────────────────────────────────────────────────────

    def _set_dot(self, color: str):
        self._dot_label.setPixmap(qta.icon("fa5s.circle", color=color).pixmap(_DOT_SIZE, _DOT_SIZE))

    def _render(self):
        """Show the model: dot, texts, progress, the toggle and the buttons."""
        model = self.model
        state = model.state()
        color = {"grey": _DOT_GRAY, "green": _DOT_GREEN, "amber": _DOT_AMBER, "red": _DOT_RED}[model.color_name()]
        self._set_dot(color)
        self._title_label.setText(model.title())
        self._title_label.setStyleSheet(f"font-weight: 600; color: {'#9e9e9e' if state == model.IDLE else color};")
        self._detail_label.setText(model.detail())
        capturing = state == model.CAPTURING
        self._extra_label.setText(model.extra() if state == model.LOADED else "")
        self._extra_label.setVisible(state == model.LOADED)
        done, total = model.capture_progress()
        self._progress_bar.setRange(0, max(1, total))
        self._progress_bar.setValue(done)
        self._progress_label.setText(f"{done} / {total}")
        self._progress_bar.setVisible(capturing)
        self._progress_label.setVisible(capturing)
        self._dark_toggle.blockSignals(True)
        self._dark_toggle.setChecked(model.is_dark_enabled())
        self._dark_toggle.blockSignals(False)
        self._capture_btn.setVisible(not capturing)
        self._capture_btn.setEnabled(model.is_interactive())
        self._cancel_btn.setVisible(capturing)
        self._cancel_btn.setEnabled(True)
        self._reload_btn.setEnabled(model.is_interactive() and not capturing)
