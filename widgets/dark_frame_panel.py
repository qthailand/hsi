"""Dark Frame status panel with colored indicator and qtawesome icons."""

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
    """Self-contained dark-frame status + control panel."""

    capture_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    reload_requested = pyqtSignal()
    dark_toggled = pyqtSignal(bool)  # ★ เพิ่ม signal

    IDLE = "idle"
    LOADED = "loaded"
    CAPTURING = "capturing"
    ERROR = "error"

    def __init__(self, parent=None):
        super().__init__("Dark Frame", parent)

        self._state = self.IDLE
        self._interactive = True

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
        """★ Return current toggle state."""
        return self._dark_toggle.isChecked()

    def set_idle(self):
        self._apply_state(self.IDLE)

    def set_loaded(
        self,
        bin_value: int,
        exposure_ms: float,
        gain: float,
        n_frames: int,
        n_hot: int,
    ):
        self._apply_state(
            self.LOADED,
            bin_value=bin_value,
            exposure_ms=exposure_ms,
            gain=gain,
            n_frames=n_frames,
            n_hot=n_hot,
        )

    def set_capturing(
        self,
        collected: int,
        total: int,
        bin_value: int | None = None,
        exposure_ms: float | None = None,
        gain: float | None = None,
    ):
        self._apply_state(
            self.CAPTURING,
            collected=collected,
            total=total,
            bin_value=bin_value,
            exposure_ms=exposure_ms,
            gain=gain,
        )

    def update_progress(self, collected: int, total: int):
        """Fast progress update – no layout rebuild."""
        if self._state != self.CAPTURING:
            return
        self._progress_bar.setRange(0, max(1, total))
        self._progress_bar.setValue(collected)
        self._progress_label.setText(f"{collected} / {total}")

    def set_error(self, message: str = "Load failed"):
        self._apply_state(self.ERROR, error_message=message)

    def is_capturing(self) -> bool:
        return self._state == self.CAPTURING

    def set_interactive(self, enabled: bool):
        """Enable/disable user actions (e.g. during scan)."""
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
            self._extra_label.setText(
                f"{kw.get('n_frames', 0)} frames · "
                f"{kw.get('n_hot', 0)} hot pixels"
            )
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