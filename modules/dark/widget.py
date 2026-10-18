"""Dark Frame status panel with colored indicator and qtawesome icons.

``DarkFramePanel`` shows whether a dark reference is active and offers Capture / Cancel /
Reload buttons plus an "Apply Dark Subtraction" toggle. It performs no capture itself:
the host (see :class:`modules.dark.DarkFrameLibrary` and
:func:`modules.dark.select_dark`) reacts to the signals and pushes state back with the
``set_*`` methods.

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

        self._state = self.IDLE
        self._interactive = True
        self._loaded_info = {}
        self._dark_temperature = None      # °C of the loaded dark, None if unknown
        self._current_temperature = None   # °C reported by the camera, None/-999 if unknown

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
        self._dark_toggle.toggled.connect(self.dark_toggled.emit)

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
        self._capture_btn.clicked.connect(self.capture_requested.emit)
        self._cancel_btn.clicked.connect(self.cancel_requested.emit)
        self._reload_btn.clicked.connect(self.reload_requested.emit)

        # --- initial ---
        self._apply_state(self.IDLE)

    # ── public API ──────────────────────────────────────────────────

    def is_dark_enabled(self) -> bool:
        """★ Return current toggle state.

        Returns:
            bool: True if the "Apply Dark Subtraction" check box is checked.
        """
        return self._dark_toggle.isChecked()

    def set_idle(self):
        """Show "No dark frame" (grey): nothing loaded and nothing capturing."""
        self._apply_state(self.IDLE)

    def set_loaded(
        self,
        bin_value: int,
        exposure_ms: float,
        gain: float,
        n_frames: int,
        n_hot: int,
        temperature_c: float | None = None,
        current_temperature_c: float | None = None,
    ):
        """Show that a dark reference is active (green).

        Args:
            bin_value: Hardware bin of the dark.
            exposure_ms: Exposure of the dark in milliseconds.
            gain: Gain of the dark (vendor units).
            n_frames: Number of frames averaged into the dark.
            n_hot: Number of hot pixels in its hot-pixel map.
            temperature_c: Sensor temperature (°C) the dark was captured at, ``None`` if unknown.
            current_temperature_c: Current sensor temperature (°C) for the ΔT shown next to
                it; ``None``/``-999`` hides ΔT. Update it later with :meth:`set_current_temperature`.
        """
        self._dark_temperature = temperature_c
        if current_temperature_c is not None:
            self._current_temperature = current_temperature_c
        self._apply_state(
            self.LOADED,
            bin_value=bin_value,
            exposure_ms=exposure_ms,
            gain=gain,
            n_frames=n_frames,
            n_hot=n_hot,
        )

    def set_current_temperature(self, temperature_c: float):
        """Update the current sensor temperature (°C) used for the ΔT of the loaded dark.

        Args:
            temperature_c: Current temperature; below ``-900`` (``-999``) means unknown.
        """
        self._current_temperature = temperature_c
        if self._state == self.LOADED:
            self._extra_label.setText(self._loaded_extra_text())

    def set_capturing(
        self,
        collected: int,
        total: int,
        bin_value: int | None = None,
        exposure_ms: float | None = None,
        gain: float | None = None,
    ):
        """Show capture progress (amber) with a progress bar; the Cancel button appears.

        Args:
            collected: Frames collected so far.
            total: Frames to collect.
            bin_value: Bin being captured (the settings line is blank if None).
            exposure_ms: Exposure in milliseconds, shown with ``bin_value``.
            gain: Gain in vendor units, shown with ``bin_value``.
        """
        self._apply_state(
            self.CAPTURING,
            collected=collected,
            total=total,
            bin_value=bin_value,
            exposure_ms=exposure_ms,
            gain=gain,
        )

    def update_progress(self, collected: int, total: int):
        """Fast progress update – no layout rebuild.

        Ignored unless the panel is in the capturing state (call :meth:`set_capturing` first).

        Args:
            collected: Frames collected so far.
            total: Frames to collect.
        """
        if self._state != self.CAPTURING:
            return
        self._progress_bar.setRange(0, max(1, total))
        self._progress_bar.setValue(collected)
        self._progress_label.setText(f"{collected} / {total}")

    def set_error(self, message: str = "Load failed"):
        """Show an error title (red), e.g. when the stored dark does not fit the camera.

        Args:
            message: Text for the title line.
        """
        self._apply_state(self.ERROR, error_message=message)

    def is_capturing(self) -> bool:
        """True while the panel is in the capturing state."""
        return self._state == self.CAPTURING

    def set_interactive(self, enabled: bool):
        """Enable/disable user actions (e.g. during scan).

        Args:
            enabled: False disables Capture and Reload. Cancel stays enabled while
                capturing.
        """
        self._interactive = enabled
        self._refresh_buttons()

    # ── internal ────────────────────────────────────────────────────

    def _dot_icon(self, color: str):
        return qta.icon("fa5s.circle", color=color)

    def _set_dot(self, color: str):
        self._dot_label.setPixmap(
            self._dot_icon(color).pixmap(_DOT_SIZE, _DOT_SIZE)
        )

    def _refresh_buttons(self):
        """Re-apply enable/visibility based on current state + interactive flag."""
        enabled = self._interactive

        if self._state == self.CAPTURING:
            self._capture_btn.hide()
            self._cancel_btn.show()
            self._cancel_btn.setEnabled(True)
            self._reload_btn.show()
            self._reload_btn.setEnabled(False)
        else:
            self._capture_btn.show()
            self._capture_btn.setEnabled(enabled)
            self._cancel_btn.hide()
            self._reload_btn.show()
            self._reload_btn.setEnabled(enabled)

    def _loaded_extra_text(self) -> str:
        """``"50 frames · 3 hot pixels · 25.3°C (ΔT +1.2)"`` for the loaded dark."""
        kw = self._loaded_info
        text = f"{kw.get('n_frames', 0)} frames · {kw.get('n_hot', 0)} hot pixels"
        dark_t = self._dark_temperature
        if dark_t is None:
            return text + " · temp unknown"
        text += f" · {dark_t:.1f}°C"
        current = self._current_temperature
        if current is not None and current > -900:
            text += f" (ΔT {current - dark_t:+.1f})"
        return text

    def _apply_state(self, state: str, **kw):
        self._state = state

        # ---- idle ----
        if state == self.IDLE:
            self._set_dot(_DOT_GRAY)
            self._title_label.setText("No dark frame")
            self._title_label.setStyleSheet(
                "font-weight: 600; color: #9e9e9e;"
            )
            self._detail_label.setText(
                "Capture or reload to enable correction"
            )
            self._extra_label.hide()
            self._progress_bar.hide()
            self._progress_label.hide()

        # ---- loaded ----
        elif state == self.LOADED:
            self._set_dot(_DOT_GREEN)
            self._title_label.setText("Dark frame active")
            self._title_label.setStyleSheet(
                f"font-weight: 600; color: {_DOT_GREEN};"
            )
            self._detail_label.setText(
                f"bin={kw.get('bin_value', '?')} · "
                f"exp={kw.get('exposure_ms', 0):.1f}ms · "
                f"gain={kw.get('gain', 0):.0f}"
            )
            self._loaded_info = kw
            self._extra_label.setText(self._loaded_extra_text())
            self._extra_label.show()
            self._progress_bar.hide()
            self._progress_label.hide()

        # ---- capturing ----
        elif state == self.CAPTURING:
            self._set_dot(_DOT_AMBER)
            self._title_label.setText("Capturing dark frames\u2026")
            self._title_label.setStyleSheet(
                f"font-weight: 600; color: {_DOT_AMBER};"
            )
            bv = kw.get("bin_value")
            if bv is not None:
                self._detail_label.setText(
                    f"bin={bv} · "
                    f"exp={kw.get('exposure_ms', 0):.1f}ms · "
                    f"gain={kw.get('gain', 0):.0f}"
                )
            else:
                self._detail_label.setText("")

            collected = kw.get("collected", 0)
            total = kw.get("total", 1)
            self._extra_label.hide()
            self._progress_bar.show()
            self._progress_bar.setRange(0, max(1, total))
            self._progress_bar.setValue(collected)
            self._progress_label.setText(f"{collected} / {total}")
            self._progress_label.show()

        # ---- error / stale ----
        elif state == self.ERROR:
            self._set_dot(_DOT_RED)
            self._title_label.setText(kw.get("error_message", "Error"))
            self._title_label.setStyleSheet(
                f"font-weight: 600; color: {_DOT_RED};"
            )
            self._detail_label.setText("Capture or reload to restore")
            self._extra_label.hide()
            self._progress_bar.hide()
            self._progress_label.hide()

        self._refresh_buttons()