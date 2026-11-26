"""``DarkFrameState``: the dark reference's state and requests, with no widget of its own.

What the dark panel showed and asked for, as an object: the state (``idle``, ``loaded``,
``capturing``, ``error``), *Apply Dark Subtraction*, the capture progress, the loaded dark's
setting and temperature, and the Capture / Cancel / Reload requests. A host updates it with the
``set_*`` methods (as it updated the panel) and draws it where it likes: a status row, a peek,
or :class:`~modules.dark.widget.DarkFramePanel`, which is a view of one.

Data conventions:
    Exposure in milliseconds, gain in vendor units, bin the hardware factor, temperatures in °C
    (below -900, e.g. -999, means unknown).

Threading:
    A ``QObject`` (signals): GUI thread only. Needs PyQt5 (QtCore), not QtWidgets.

Example:
    >>> # state = DarkFrameState()
    >>> # state.capture_requested.connect(start_capture)       # host calls library.start_capture
    >>> # state.set_capturing(0, 50, bin_value=2, exposure_ms=10.0, gain=100)
    >>> # state.update_progress(25, 50); state.capture_progress()
    >>> # (25, 50)
"""
from __future__ import annotations

from PyQt5.QtCore import QObject, pyqtSignal


class DarkFrameState(QObject):
    """State, texts and requests of the dark reference (see the module docstring).

    Signals:
        capture_requested, cancel_requested, reload_requested: the user asked (via a view's
            button, or :meth:`request_capture` / :meth:`request_cancel` / :meth:`request_reload`).
        dark_toggled(bool): *Apply Dark Subtraction* changed (:meth:`set_dark_enabled`); not at start.
        changed(): anything shown changed (state, texts, progress, the toggle, interactivity).

    Class attributes ``IDLE``, ``LOADED``, ``CAPTURING`` and ``ERROR`` are the state names.
    """

    capture_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    reload_requested = pyqtSignal()
    dark_toggled = pyqtSignal(bool)
    changed = pyqtSignal()

    IDLE = "idle"
    LOADED = "loaded"
    CAPTURING = "capturing"
    ERROR = "error"

    def __init__(self, parent=None):
        """Idle, subtraction on, interactive."""
        super().__init__(parent)
        self._state = self.IDLE
        self._interactive = True
        self._enabled = True
        self._info: dict = {}                 # setting of the loaded / captured dark
        self._dark_temperature = None         # °C of the loaded dark, None if unknown
        self._current_temperature = None      # °C reported by the camera, None / -999 if unknown
        self._error = ""
        self._progress = (0, 0)

    # ── the toggle ───────────────────────────────────────────────

    def is_dark_enabled(self) -> bool:
        """Whether *Apply Dark Subtraction* is on."""
        return self._enabled

    def set_dark_enabled(self, enabled: bool) -> None:
        """Switch *Apply Dark Subtraction* (emits ``dark_toggled`` and ``changed`` when it changes)."""
        enabled = bool(enabled)
        if enabled != self._enabled:
            self._enabled = enabled
            self.dark_toggled.emit(enabled)
            self.changed.emit()

    # ── states ───────────────────────────────────────────────────

    def set_idle(self):
        """Nothing loaded and nothing capturing."""
        self._set(self.IDLE)

    def set_loaded(self, bin_value: int, exposure_ms: float, gain: float, n_frames: int, n_hot: int,
                   temperature_c: float | None = None, current_temperature_c: float | None = None):
        """A dark reference is active.

        Args:
            bin_value: Hardware bin of the dark.
            exposure_ms: Exposure of the dark in milliseconds.
            gain: Gain of the dark (vendor units).
            n_frames: Number of frames averaged into the dark.
            n_hot: Number of hot pixels in its hot-pixel map.
            temperature_c: Sensor temperature (°C) the dark was captured at, ``None`` if unknown.
            current_temperature_c: Current sensor temperature for the ΔT; ``None`` keeps the last.
        """
        self._dark_temperature = temperature_c
        if current_temperature_c is not None:
            self._current_temperature = current_temperature_c
        self._set(self.LOADED, bin_value=bin_value, exposure_ms=exposure_ms, gain=gain,
                  n_frames=n_frames, n_hot=n_hot)

    def set_current_temperature(self, temperature_c: float):
        """The current sensor temperature (°C) for the ΔT of the loaded dark (below -900: unknown)."""
        if temperature_c != self._current_temperature:
            self._current_temperature = temperature_c
            if self._state == self.LOADED:
                self.changed.emit()

    def set_capturing(self, collected: int, total: int, bin_value: int | None = None,
                      exposure_ms: float | None = None, gain: float | None = None):
        """A capture runs: ``collected`` of ``total`` frames, for the given setting (if known)."""
        self._progress = (int(collected), int(total))
        self._set(self.CAPTURING, bin_value=bin_value, exposure_ms=exposure_ms, gain=gain)

    def update_progress(self, collected: int, total: int):
        """Frames collected so far of the running capture (ignored when none runs)."""
        if self._state != self.CAPTURING:
            return
        progress = (int(collected), int(total))
        if progress != self._progress:
            self._progress = progress
            self.changed.emit()

    def set_error(self, message: str = "Load failed"):
        """An error (e.g. the stored dark does not fit the camera); ``message`` is the title."""
        self._error = str(message)
        self._set(self.ERROR)

    def set_interactive(self, enabled: bool):
        """Allow Capture and Reload (False during a scan); Cancel stays allowed while capturing."""
        if bool(enabled) != self._interactive:
            self._interactive = bool(enabled)
            self.changed.emit()

    def _set(self, state: str, **info):
        self._state = state
        if state in (self.LOADED, self.CAPTURING):
            self._info = info
        if state != self.CAPTURING:
            self._progress = (0, 0)
        self.changed.emit()

    # ── getters ──────────────────────────────────────────────────

    def state(self) -> str:
        """:attr:`IDLE`, :attr:`LOADED`, :attr:`CAPTURING` or :attr:`ERROR`."""
        return self._state

    def is_capturing(self) -> bool:
        """Whether a capture runs."""
        return self._state == self.CAPTURING

    def is_interactive(self) -> bool:
        """Whether Capture and Reload may be used now."""
        return self._interactive

    def capture_progress(self) -> tuple:
        """``(collected, total)`` of the running capture, ``(0, 0)`` when none runs."""
        return self._progress if self._state == self.CAPTURING else (0, 0)

    def can_capture(self) -> bool:
        """Capture (and Reload) may be asked now: interactive and not capturing."""
        return self._interactive and self._state != self.CAPTURING

    # ── requests (what a view's buttons do) ──────────────────────

    def request_capture(self):
        """Ask the host to capture a dark (ignored while not allowed)."""
        if self.can_capture():
            self.capture_requested.emit()

    def request_reload(self):
        """Ask the host to load the stored dark again (ignored while not allowed)."""
        if self.can_capture():
            self.reload_requested.emit()

    def request_cancel(self):
        """Ask the host to cancel the running capture (ignored when none runs)."""
        if self.is_capturing():
            self.cancel_requested.emit()

    # ── texts (as the panel shows them) ──────────────────────────

    def color_name(self) -> str:
        """``'grey'`` (idle), ``'green'`` (loaded), ``'amber'`` (capturing) or ``'red'`` (error)."""
        return {self.IDLE: "grey", self.LOADED: "green", self.CAPTURING: "amber", self.ERROR: "red"}[self._state]

    def title(self) -> str:
        """``No dark frame`` / ``Dark frame active`` / ``Capturing dark frames…`` / the error."""
        return {self.IDLE: "No dark frame", self.LOADED: "Dark frame active",
                self.CAPTURING: "Capturing dark frames…", self.ERROR: self._error or "Error"}[self._state]

    def detail(self) -> str:
        """The setting line (``bin=2 · exp=10.0ms · gain=100``) or what to do."""
        if self._state == self.IDLE:
            return "Capture or reload to enable correction"
        if self._state == self.ERROR:
            return "Capture or reload to restore"
        info = self._info
        if info.get("bin_value") is None:
            return ""
        return (f"bin={info.get('bin_value', '?')} · exp={float(info.get('exposure_ms') or 0):.1f}ms "
                f"· gain={float(info.get('gain') or 0):.0f}")

    def extra(self) -> str:
        """Loaded: ``50 frames · 3 hot pixels · 25.3°C (ΔT +1.2)``; capturing: ``12 / 50``; else ``''``."""
        if self._state == self.CAPTURING:
            done, total = self._progress
            return f"{done} / {total}"
        if self._state != self.LOADED:
            return ""
        info = self._info
        text = f"{info.get('n_frames', 0)} frames · {info.get('n_hot', 0)} hot pixels"
        dark_t = self._dark_temperature
        if dark_t is None:
            return text + " · temp unknown"
        text += f" · {dark_t:.1f}°C"
        current = self._current_temperature
        if current is not None and current > -900:
            text += f" (ΔT {current - dark_t:+.1f})"
        return text

    def summary(self) -> str:
        """Title, detail and extra on separate lines (a tooltip)."""
        return "\n".join(part for part in (self.title(), self.detail(), self.extra()) if part)
