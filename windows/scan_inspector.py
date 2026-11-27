# windows/scan_inspector.py
"""The look of the scan window's settings column (the *inspector*) and its two summary widgets.

* :func:`inspector_stylesheet` - sections without frames, fields with the unit inside, toggle
  switches for check boxes, segmented bin buttons; scoped to ``#inspector`` so no other window
  changes.
* :class:`ReadyList` - *Ready to scan?*: only what needs attention, one row each with its fix,
  then one line for the checks that passed.

GUI thread only.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import qtawesome as qta

from PyQt5.QtCore import QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton, QToolButton, QVBoxLayout,
                             QWidget)

from modules.ui_kit.theme import ACCENT, BORDER, PAGE

ASSETS = Path(__file__).resolve().parent / "assets"
#: Numbers (fields, temperatures) in a monospaced face, as in the design.
MONO_FAMILY = "JetBrains Mono, Consolas, Cascadia Mono, monospace"
#: Text colours: primary, secondary, dim.
TEXT, MUTED, DIM = "#e6e9ee", "#9aa4b2", "#7d8794"
FIELD_BG, FIELD_BORDER, LINE = "#16191d", "#343a42", "#2c3138"
#: Ready-list severities: dot colour and row background.
SEVERITY = {"error": ("#ff6b6b", "#2a1618"), "warning": ("#e3b341", "#2a2312"),
            "info": ("#8ab8ff", "#16223a")}


def _url(name: str) -> str:
    return (ASSETS / name).as_posix()


def inspector_stylesheet() -> str:
    """Style sheet for the settings column; apply it to the widget named ``inspector``."""
    on, off = _url("switch_on.svg"), _url("switch_off.svg")
    on_dis, off_dis = _url("switch_on_disabled.svg"), _url("switch_off_disabled.svg")
    return f"""
#inspector {{ background: {PAGE}; }}
#inspector QScrollArea {{ background: transparent; border: none; }}
#inspector #inspectorBody {{ background: transparent; }}
#inspector QLabel {{ color: {MUTED}; font-size: 12px; background: transparent; }}
#inspector QLabel[role="section"] {{ color: {TEXT}; font-size: 13px; font-weight: 600; }}

/* sections: a hairline above, a bold title, no frame */
#inspector QGroupBox {{
    border: none; border-top: 1px solid {LINE}; border-radius: 0;
    margin-top: 6px; padding: 30px 0 2px 0; background: transparent;
    font-size: 13px; font-weight: 600; color: {TEXT};
}}
#inspector QGroupBox::title {{
    subcontrol-origin: padding; subcontrol-position: top left; left: 0; top: 9px; padding: 0 6px 0 0;
    color: {TEXT};
}}
#inspector QGroupBox:disabled, #inspector QGroupBox::title:disabled {{ color: {DIM}; }}
#inspector QGroupBox::indicator {{ width: 30px; height: 17px; }}
#inspector QGroupBox::indicator:checked {{ image: url({on}); }}
#inspector QGroupBox::indicator:unchecked {{ image: url({off}); }}
#inspector QGroupBox::indicator:checked:disabled {{ image: url({on_dis}); }}
#inspector QGroupBox::indicator:unchecked:disabled {{ image: url({off_dis}); }}

/* fields: dark, rounded, the unit inside (suffix), numbers monospaced, no arrows */
#inspector QSpinBox, #inspector QDoubleSpinBox, #inspector QLineEdit {{
    background: {FIELD_BG}; border: 1px solid {FIELD_BORDER}; border-radius: 7px;
    padding: 4px 8px; min-height: 22px; color: {TEXT};
    font-family: {MONO_FAMILY}; font-size: 12px; selection-background-color: {ACCENT};
}}
#inspector QSpinBox:focus, #inspector QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
#inspector QSpinBox:disabled, #inspector QDoubleSpinBox:disabled {{
    color: {DIM}; background: #1a1d21; border-color: #2a2f35;
}}
#inspector QComboBox {{
    background: {FIELD_BG}; border: 1px solid {FIELD_BORDER}; border-radius: 7px; padding: 4px 8px;
    min-height: 22px; color: {TEXT};
}}

/* check boxes are switches */
#inspector QCheckBox {{ color: {MUTED}; font-size: 12px; spacing: 8px; background: transparent; }}
#inspector QCheckBox:disabled {{ color: {DIM}; }}
#inspector QCheckBox::indicator {{ width: 30px; height: 17px; }}
#inspector QCheckBox::indicator:checked {{ image: url({on}); }}
#inspector QCheckBox::indicator:unchecked {{ image: url({off}); }}
#inspector QCheckBox::indicator:checked:disabled {{ image: url({on_dis}); }}
#inspector QCheckBox::indicator:unchecked:disabled {{ image: url({off_dis}); }}

/* buttons: quiet */
#inspector QPushButton {{
    background: #1f2328; border: 1px solid {FIELD_BORDER}; border-radius: 7px; padding: 6px 12px;
    color: {TEXT}; font-size: 12px;
}}
#inspector QPushButton:hover {{ background: #262b31; }}
#inspector QPushButton:disabled {{ color: {DIM}; background: #1a1d21; border-color: #2a2f35; }}
#inspector QPushButton[link="true"] {{
    background: transparent; border: none; padding: 0 2px; color: #8ab8ff; font-weight: 600;
}}

/* the bin: one bar of buttons */
#inspector #binSegments QPushButton {{
    border-radius: 0; border: 1px solid {FIELD_BORDER}; border-left-width: 0; background: {FIELD_BG};
    color: {MUTED}; padding: 5px 0; min-height: 18px;
}}
#inspector #binSegments QPushButton[segment="first"] {{
    border-left-width: 1px; border-top-left-radius: 7px; border-bottom-left-radius: 7px;
}}
#inspector #binSegments QPushButton[segment="last"] {{
    border-top-right-radius: 7px; border-bottom-right-radius: 7px;
}}
#inspector #binSegments QPushButton[segment="only"] {{
    border-left-width: 1px; border-radius: 7px;
}}
#inspector #binSegments QPushButton:checked {{ background: #1d3a2a; color: #ffffff; font-weight: 600; }}
#inspector #binSegments QPushButton:disabled {{ color: {DIM}; }}
#inspector #binSegments QPushButton:checked:disabled {{ background: #1f2a24; color: {MUTED}; }}

/* rows: a correction, a ready-list item */

/* the pinned Start / Stop area */
#inspector #scanActionBar {{ border-top: 1px solid {LINE}; background: #1b1e22; }}
#inspector QProgressBar {{ background: #262b31; border: none; border-radius: 2px; max-height: 4px; }}
#inspector QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}
"""


def _dot(color: str, size: int = 8) -> QLabel:
    dot = QLabel()
    dot.setFixedSize(size, size)
    dot.setStyleSheet(f"background: {color}; border-radius: {size // 2}px;")
    return dot


def section_label(text: str) -> QLabel:
    """A section title as in the design (bold, primary colour)."""
    label = QLabel(text)
    label.setProperty("role", "section")
    return label


class ReadyList(QWidget):
    """*Ready to scan?*: the items that need attention, then ``✓ N checks passed``.

    Call :meth:`set_items` whenever the state may have changed. The rows are rebuilt only when
    their kinds change (severity, button, whether the button may act); a new text or callback is
    put into the rows in place, so a button never vanishes under the cursor between press and
    release, and a button never outlives the callback that locked it (``None``: no button).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(5)
        self._structure = None
        self._texts = []
        self._labels = []
        self._callbacks = []
        self._passed = None
        self.rows = []
        self.passed_label = QLabel()
        self.passed_label.setStyleSheet(f"color: {DIM}; font-size: 11px; padding-left: 8px;")

    def set_items(self, items: Sequence[tuple], passed: Sequence[str] = ()):
        """Show ``items`` and the passed checks.

        Args:
            items: ``(severity, text, action_text, callback)`` each; severity is ``"error"``,
                ``"warning"`` or ``"info"``; ``action_text`` ``""`` / callback ``None`` for none.
            passed: Names of the checks that passed (in the tooltip of the summary line).
        """
        items = list(items)
        structure = tuple((s, a if callback is not None else "") for s, _t, a, callback in items)
        self._callbacks = [callback for *_rest, callback in items]
        self._texts = [text for _s, text, _a, _c in items]
        if structure == self._structure:
            for label, text in zip(self._labels, self._texts):
                if label.text() != text:
                    label.setText(text)
            self._set_passed(passed)
            return
        self._structure = structure
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None and widget is not self.passed_label:
                widget.deleteLater()
        self.rows = []
        self._labels = []
        for index, (severity, text, action, callback) in enumerate(items):
            dot_color, background = SEVERITY.get(severity, SEVERITY["info"])
            row = QFrame()
            row.setProperty("role", "readyItem")
            row.setStyleSheet(f"QFrame {{ background: {background}; border-radius: 7px; }}")
            line = QHBoxLayout(row)
            line.setContentsMargins(8, 6, 8, 6)
            line.setSpacing(8)
            line.addWidget(_dot(dot_color))
            label = QLabel(text)
            label.setStyleSheet(f"color: {TEXT}; font-size: 12px; background: transparent;")
            label.setWordWrap(True)
            line.addWidget(label, 1)
            self._labels.append(label)
            if action and callback is not None:
                button = QPushButton(action)
                button.setProperty("link", True)
                button.setCursor(Qt.PointingHandCursor)
                button.clicked.connect(lambda _=False, i=index: self._run(i))
                line.addWidget(button)
            self._layout.addWidget(row)
            self.rows.append(row)
        self._passed = None
        self._set_passed(passed)
        self._layout.addWidget(self.passed_label)

    def _run(self, index: int):
        """A row's button: the callback of the latest :meth:`set_items` (``None``: locked now)."""
        callback = self._callbacks[index] if index < len(self._callbacks) else None
        if callback is not None:
            callback()

    def _set_passed(self, passed):
        passed = tuple(passed)
        if passed == self._passed:
            return
        self._passed = passed
        n = len(passed)
        self.passed_label.setText(f"✓ {n} check{'s' if n != 1 else ''} passed" if n else "")
        self.passed_label.setToolTip("\n".join(passed))
        self.passed_label.setVisible(bool(n))

    def texts(self):
        """The texts of the shown items, in order (for tests and the status line)."""
        return list(self._texts)


def mono_font(widget: QWidget, size_pt: Optional[float] = None) -> None:
    """Give ``widget`` the monospaced number face (fields set it in the style sheet)."""
    font = QFont(widget.font())
    font.setFamily(MONO_FAMILY.split(",")[0])
    if size_pt:
        font.setPointSizeF(size_pt)
    widget.setFont(font)


__all__ = ["inspector_stylesheet", "ReadyList", "section_label", "mono_font", "BORDER"]


# ── live column tiles (left of the scan window) ────────────────────────────

#: Sharpness readings kept for the tile's sparkline (about 30 s at the preview's ~5 Hz).
SHARPNESS_HISTORY = 150
#: Bars of the Clipped tile's mini histogram (the 256 display bins summed in groups).
CLIPPED_BARS = 32
_GREEN, _BLUE_BAR, _RED_BAR, _AMBER = "#56D364", "#3A6EA5", "#F85149", "#E3B341"


def live_column_stylesheet() -> str:
    """Style sheet for the live column: the settings column's fields and switches, plus cards."""
    return inspector_stylesheet().replace("#inspector", "#liveColumn") + f"""
#liveColumn {{ background: {PAGE}; }}
#liveColumn QFrame[role="card"] {{ background: #1b1e22; border: 1px solid {LINE}; border-radius: 10px; }}
#liveColumn QFrame[role="card"] QLabel {{ background: transparent; border: none; }}
#liveColumn QToolButton {{ background: transparent; border: none; border-radius: 5px; padding: 2px; }}
#liveColumn QToolButton:hover {{ background: #262b31; }}
"""


def card_frame() -> QFrame:
    """An empty card of the live column (rounded, a hairline border)."""
    frame = QFrame()
    frame.setProperty("role", "card")
    frame.setAttribute(Qt.WA_StyledBackground, True)
    return frame


def format_sharpness(score: float) -> str:
    """The sharpness number as the sidebar showed it: no decimals from 100, one below."""
    return f"{score:.0f}" if score >= 100 else f"{score:.1f}"


class _Sparkline(QWidget):
    """The sharpness history: a green line, the best so far dashed above it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = []
        self.best = None
        self.setMinimumHeight(28)

    def paintEvent(self, event):
        """Qt handler: draw the dashed best and the history line."""
        if len(self.values) < 2 or not self.best:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        top = max(self.best, max(self.values)) or 1.0
        lo = min(self.values)
        span = max(top - lo, abs(top) * 0.05, 1e-9)

        def y_of(v):
            return 3 + (h - 6) * (1.0 - (v - lo) / span)

        best = QColor(_GREEN)
        best.setAlpha(120)
        p.setPen(QPen(best, 1, Qt.DashLine))
        p.drawLine(0, int(y_of(self.best)), w, int(y_of(self.best)))
        n = len(self.values)
        step = w / (SHARPNESS_HISTORY - 1) if n >= SHARPNESS_HISTORY else w / (n - 1)
        p.setPen(QPen(QColor(_GREEN), 2))
        p.drawPolyline(QPolygonF([QPointF(i * step, y_of(v)) for i, v in enumerate(self.values)]))
        p.end()


class SharpnessTile(QFrame):
    """Sharpness card: the score, a reset button, the last ~30 s as a line with the best dashed.

    Feed it with :meth:`set_value` like the sidebar it replaces; :attr:`reset_requested` is its
    reset button (the host resets every sharpness display).
    """

    reset_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("role", "card")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._history = deque(maxlen=SHARPNESS_HISTORY)
        self._best = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        head = QHBoxLayout()
        title = QLabel("Sharpness")
        title.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self.reset_btn = QToolButton()
        self.reset_btn.setIcon(qta.icon("fa5s.redo-alt", color=DIM))
        self.reset_btn.setToolTip("Reset the best sharpness and the history")
        self.reset_btn.clicked.connect(self.reset_requested.emit)
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.reset_btn)
        layout.addLayout(head)
        self.value_label = QLabel("—")
        self._value_style(False)
        layout.addWidget(self.value_label)
        self.spark = _Sparkline()
        layout.addWidget(self.spark)

    def _value_style(self, scanning: bool):
        color = "#ffa726" if scanning else TEXT
        self.value_label.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: 500; "
                                       f"font-family: {MONO_FAMILY};")

    def set_value(self, score: float):
        """Show a new score and add it to the history (the best so far is kept)."""
        score = float(score)
        self._history.append(score)
        self._best = score if self._best is None else max(self._best, score)
        self.value_label.setText(format_sharpness(score))
        self.value_label.setToolTip(f"best {format_sharpness(self._best)}")
        self.spark.values, self.spark.best = list(self._history), self._best
        self.spark.update()

    def set_scanning(self, scanning: bool):
        """Amber number while a scan runs (as the sidebar)."""
        self._value_style(bool(scanning))

    def reset(self):
        """Forget the history and the best; show an em dash until the next score."""
        self._history.clear()
        self._best = None
        self.value_label.setText("—")
        self.value_label.setToolTip("")
        self.spark.values, self.spark.best = [], None
        self.spark.update()

    def best(self):
        """The best score since the last reset, or ``None``."""
        return self._best


class _MiniHistogram(QWidget):
    """Exposure histogram in a few bars; the saturated share of each bar drawn red."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.counts = None
        self.saturated = None
        self.setMinimumHeight(28)

    def paintEvent(self, event):
        """Qt handler: bars (square-root scale, so the tails stay visible), saturated part red."""
        if self.counts is None or not np.any(self.counts):
            return
        p = QPainter(self)
        w, h = self.width(), self.height()
        n = len(self.counts)
        top = float(np.sqrt(self.counts.max())) or 1.0
        gap = 1
        bar = max(1.0, (w - gap * (n - 1)) / n)
        for i, count in enumerate(self.counts):
            x = int(i * (bar + gap))
            full = max(1 if count else 0, int(round((h - 1) * np.sqrt(count) / top)))
            sat = 0
            if self.saturated is not None and count:
                sat = int(round(full * min(1.0, self.saturated[i] / count)))
            if full - sat > 0:
                p.fillRect(x, h - full + sat, int(bar), full - sat, QColor(_BLUE_BAR))
            if sat > 0:
                p.fillRect(x, h - full, int(bar), sat, QColor(_RED_BAR))
        p.end()


class ClippedTile(QFrame):
    """Clipped card: confirmed clipping of the raw frames, and the exposure histogram.

    The numbers come from the acquisition engine's basic detection (every acquired raw frame
    counted before any correction against the camera's stated full scale), handed over with
    :meth:`set_raw_saturation` from its mailbox
    (:class:`modules.acquisition.saturation_tracking.RawSaturationStats`):

    * the big number: clipped share of the newest frame (``raw >= level``); red above
      ``warn_percent``, amber when only near full scale (``raw >= 98 %`` of it: the exposure
      warning), *not checked* (grey) when the camera states no level;
    * the small number: the highest clipped share and the frames that clipped since the last
      reset (the engine keeps it over every frame, not only the displayed ones; its button and
      every Start scan reset it), ``partial`` when some of those frames were not checked.

    The bars are the histogram of the displayed frame (:meth:`update_histogram`, the maths is
    :func:`modules.live_view.exposure_histogram`); they no longer set the numbers.

    Signals:
        peak_reset(): the max was reset (button or :meth:`reset_peak`); the host resets the
            engine's peak.
    """

    peak_reset = pyqtSignal()

    def __init__(self, warn_percent: float = 0.1, parent=None):
        super().__init__(parent)
        self.setProperty("role", "card")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.warn_percent = float(warn_percent)
        self._bit_depth = 16
        self._stats = None
        self._percent = None
        self._near_percent = None
        self._max_percent = None
        self.counts = None
        self.saturated_counts = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        head = QHBoxLayout()
        title = QLabel("Clipped")
        title.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self.reset_btn = QToolButton()
        self.reset_btn.setIcon(qta.icon("fa5s.redo-alt", color=DIM))
        self.reset_btn.setToolTip("Reset the max (also done at every Start scan)")
        self.reset_btn.clicked.connect(self.reset_peak)
        self.frames_label = QLabel()          # frames that clipped since the reset: in the title
        head.addWidget(title)                 # row, so the value row fits a narrow tile
        head.addStretch(1)
        head.addWidget(self.frames_label)
        head.addWidget(self.reset_btn)
        layout.addLayout(head)
        value_row = QHBoxLayout()
        value_row.setSpacing(6)
        self.value_label = QLabel("\u2014")
        self.max_label = QLabel()
        value_row.addWidget(self.value_label)
        value_row.addStretch(1)
        value_row.addWidget(self.max_label, 0, Qt.AlignBottom)
        layout.addLayout(value_row)
        self.bars = _MiniHistogram()
        layout.addWidget(self.bars)
        self._shown = None
        self._show_value()
        self._show_max()
        self.bars.setToolTip("Histogram of the displayed frame; pixels at the display ceiling red")

    # ── the numbers (engine's basic detection) ──────────────────

    def set_raw_saturation(self, stats) -> None:
        """Show the engine's basic detection (``RawSaturationStats``, or ``None``: no frame yet)."""
        self._stats = stats
        checked = stats is not None and stats.checked
        self._percent = stats.clipped_percent if checked else None
        self._near_percent = stats.near_percent if checked else None
        self._max_percent = stats.peak_clipped_percent if stats is not None else None
        self._show_value()
        self._show_max()

    def _show_value(self):
        stats = self._stats
        if stats is None:
            text, color, size, tip = "\u2014", TEXT, 22, "Clipped pixels of the newest raw frame (no frame yet)"
        elif not stats.checked:
            text, color, size = "not checked", DIM, 13
            tip = f"Clipping not checked: {stats.reason}"
        else:
            percent, near, size = self._percent, self._near_percent, 22
            text = f"{percent:.2f} %"
            if percent > self.warn_percent:
                color = _RED_BAR                     # confirmed clipping
            elif near > self.warn_percent:
                color = _AMBER                       # near full scale: exposure warning
            else:
                color = TEXT
            tip = (f"Clipped raw pixels (\u2265 {stats.level}) in the newest frame: {stats.clipped:,} of "
                   f"{stats.pixels:,}\nNear full scale (\u2265 {stats.near_level}): {near:.2f} %"
                   f"{' - lower the exposure' if near > self.warn_percent else ''}\nLevel: {stats.provenance}")
        shown = (text, color, size)
        if shown != self._shown:
            self._shown = shown
            self.value_label.setText(text)
            self.value_label.setStyleSheet(f"color: {color}; font-size: {size}px; font-weight: 500; "
                                           f"font-family: {MONO_FAMILY};")
        self.value_label.setToolTip(tip)

    def _show_max(self):
        """Peak since the reset and the frames that clipped: grey, red once the slit clipped on
        the way (above ``warn_percent``); prefixed ``\u2265`` when some frames were not checked
        (partial: the true values are at least these)."""
        stats, peak = self._stats, self._max_percent
        clipped = peak is not None and peak > self.warn_percent
        if peak is None:
            text, frames = "\u2014", ""
            tip = "Highest share of clipped pixels since the last reset (reset at every Start scan)"
        else:
            partial = stats.window_status == "partial"
            at_least = "\u2265" if partial else ""
            text = f"max {at_least}{peak:.2f} %"
            frames = f"{at_least}{stats.saturated_frames} fr"
            tip = (f"Since the last reset (every Start scan): highest clipped share {peak:.3f} %, "
                   f"{stats.saturated_frames} of {stats.checked_frames} checked frames clipped, "
                   f"{stats.near_frames} near full scale"
                   + (f"\nPartial: {stats.frames - stats.checked_frames} of {stats.frames} frames not checked "
                      f"(no stated level), so the true values are at least these" if partial else ""))
        style = (f"color: {_RED_BAR if clipped else DIM}; font-size: 11px; "
                 f"font-family: {MONO_FAMILY}; font-weight: {600 if clipped else 400};")
        for label, value in ((self.max_label, text), (self.frames_label, frames)):
            label.setText(value)
            label.setToolTip(tip)
            label.setStyleSheet(style)

    def saturated_percent(self):
        """Clipped share of the newest raw frame (%), or ``None`` (no frame, or not checked)."""
        return self._percent

    def near_percent(self):
        """Near-full-scale share of the newest raw frame (%, clipped included), or ``None``."""
        return self._near_percent

    def check_status(self):
        """``'checked'`` / ``'not_checked'`` of the newest frame, or ``None`` before one."""
        return None if self._stats is None else self._stats.status

    def max_saturated_percent(self):
        """Highest clipped share (%) since the last reset, or ``None`` (no checked frame since)."""
        return self._max_percent

    def saturated_frames(self):
        """Frames that clipped since the last reset, or ``None`` before the first frame."""
        return None if self._stats is None else self._stats.saturated_frames

    def reset_peak(self):
        """Forget the max (the button, and every Start scan) and ask the engine to do so."""
        self._max_percent = None
        stats = self._stats
        if stats is not None:
            from dataclasses import replace
            self._stats = replace(stats, frames=0, checked_frames=0, saturated_frames=0, near_frames=0,
                                  peak_clipped_percent=None, peak_near_percent=None)
        self._show_max()
        self.peak_reset.emit()

    # ── the bars (displayed frame) ──────────────────────────────

    def set_values(self, percent, counts=None, saturated=None):
        """Draw a 256-bin histogram summed into bars (``percent``: share at the ceiling, used to
        colour the top bar when ``saturated`` is not given; the numbers are not changed)."""
        if counts is None or not len(counts):
            return
        groups = np.array_split(np.asarray(counts, dtype=np.float64), CLIPPED_BARS)
        self.bars.counts = np.array([g.sum() for g in groups])
        if saturated is not None and len(saturated) == len(counts):
            sat_groups = np.array_split(np.asarray(saturated, dtype=np.float64), CLIPPED_BARS)
            self.bars.saturated = np.array([g.sum() for g in sat_groups])
        else:
            sat = np.zeros(CLIPPED_BARS)
            if percent:
                sat[-1] = self.bars.counts[-1]
            self.bars.saturated = sat
        self.bars.update()

    def set_bit_depth(self, bit_depth: int):
        """Sensor bit depth used to scale frames to the 256 display bins (next frame on)."""
        self._bit_depth = int(bit_depth)

    def update_histogram(self, frame, ceiling=None):
        """Bars of the displayed frame (per-pixel ``ceiling`` colours the top red when given)."""
        from modules.live_view import exposure_histogram
        self.counts, self.saturated_counts, ceiling_percent = exposure_histogram(frame, self._bit_depth, ceiling)
        self.set_values(ceiling_percent, self.counts, self.saturated_counts)


__all__ += ["SharpnessTile", "ClippedTile", "live_column_stylesheet", "card_frame", "format_sharpness"]
