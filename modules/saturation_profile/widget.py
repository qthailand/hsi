"""Saturation level calibration window: measure, watch the plateau form, save the profiles.

:class:`SaturationMeasureWidget` runs :func:`modules.saturation_profile.measure` on its own thread
and shows, as it goes:

* the newest frame;
* the top of its histogram, with the plateau values highlighted and, once known, the measured
  threshold as a line;
* the step it is on (bin, gain, exposure, and a progress bar);
* the result of every hardware bin: a coloured dot (verified / inconclusive / failed), the level,
  and the reasons in its tooltip.

It saves the results with its store (:meth:`set_store`).

What the host provides:

* ``set_session_provider(callable)``: called on the GUI thread when *Measure* is pressed. It
  returns a *session*, an object with ``source`` (a :class:`~modules.saturation_profile.FrameSource`)
  and ``finish() -> list[str]``. ``finish`` restores the camera and calibration state and returns
  the problems; it runs on the measurement thread after the measurement ends, is cancelled, or
  fails. It may raise ``RuntimeError`` to refuse (e.g. a scan is running).
* ``set_camera(...)``: the bins, gains and exposures to offer.
* ``set_store(store)``: where *Save* writes (a ``SaturationProfileStore``).

Signals:
    running_changed(bool): a measurement started / ended (the host locks scans meanwhile).
    measurement_finished(object, object): the records and the restore problems (``[]``: restored).
    profiles_saved(object): the ``SaveResult`` list of a *Save*.

Threading:
    GUI thread only; the measurement thread is owned by the widget and stopped (cancelled, then
    waited for so the camera is restored) in ``closeEvent``.
"""
from __future__ import annotations

import threading
from typing import Callable, List, Optional

import numpy as np
from PyQt5.QtCore import QRectF, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel,
                             QProgressBar, QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from .core import DEFAULT_PARAMS, MeasurementCancelled, SweepPlan, measure

GREEN, AMBER, RED, GREY = QColor(86, 211, 100), QColor(227, 179, 65), QColor(248, 81, 73), QColor(125, 135, 148)
STATUS_COLORS = {"verified": GREEN, "inconclusive": AMBER, "failed": RED}


class _MeasureThread(QThread):
    progress = pyqtSignal(object)
    done = pyqtSignal(object, object, str, bool)          # records, problems, error, cancelled

    def __init__(self, session, plan, cancel, parent=None):
        super().__init__(parent)
        self.session, self.plan, self.cancel = session, plan, cancel

    def run(self):
        records, error, cancelled = [], "", False
        try:
            records = measure(self.session.source, self.plan, cancel=self.cancel, progress=self._progress,
                              software=getattr(self.session, "software", "unknown"))
        except MeasurementCancelled:
            cancelled = True
        except Exception as exc:                               # noqa: BLE001 - shown in the window
            error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                problems = list(self.session.finish())
            except Exception as exc:                           # noqa: BLE001 - shown, scans stay locked
                problems = [f"restore failed: {exc}"]
        self.done.emit(records, problems, error, cancelled)

    def _progress(self, info):
        frame = info.get("frame")
        if frame is not None:                                  # a small copy for the preview
            step = max(1, int(max(frame.shape) // 480))
            info = dict(info, frame=np.array(frame[::step, ::step], copy=True))
        self.progress.emit(info)


class _TopHistogram(QWidget):
    """The top of a step's histogram: bars, plateau values highlighted, the threshold as a line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.hist, self.plateaus, self.floor, self.top, self.level, self.nominal = {}, {}, 0, 1, None, None

    def set_step(self, hist, plateaus, floor, nominal, level=None):
        """Show ``hist`` (``{value: pixels}``) from ``floor`` to the nominal full scale."""
        self.hist, self.plateaus = dict(hist or {}), dict(plateaus or {})
        self.floor, self.nominal, self.level = int(floor), nominal, level
        self.top = max([int(nominal or 0)] + list(self.hist)) + 1
        self.update()

    def set_level(self, level):
        """Draw the measured threshold (``None``: none)."""
        self.level = level
        self.update()

    def _x(self, value, width):
        return (float(value) - self.floor) / max(1.0, float(self.top - self.floor)) * (width - 1)

    def paintEvent(self, event):
        """Qt handler: square-root bars, plateaus amber, threshold green, nominal dashed."""
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), self.palette().base())
        if self.hist:
            n = min(160, max(10, w // 3))
            edges = np.linspace(self.floor, self.top, n + 1)
            values = np.fromiter(self.hist.keys(), dtype=np.float64)
            counts = np.fromiter(self.hist.values(), dtype=np.float64)
            plateau = np.isin(values, list(self.plateaus))
            bars, _ = np.histogram(values[~plateau], bins=edges, weights=counts[~plateau])
            peak = max(1.0, float(np.sqrt(max(bars.max(), counts.max()))))
            bar_w = (w - 1) / n
            for i, c in enumerate(bars):
                if c > 0:
                    bh = max(1, int((h - 14) * np.sqrt(c) / peak))
                    p.fillRect(QRectF(i * bar_w, h - bh, max(1.0, bar_w - 1), bh), self.palette().mid())
            for value, c in zip(values[plateau], counts[plateau]):
                bh = max(2, int((h - 14) * np.sqrt(c) / peak))
                p.fillRect(QRectF(self._x(value, w) - 2, h - bh, 4, bh), AMBER)
        if self.nominal:
            p.setPen(QPen(GREY, 1, Qt.DashLine))
            x = self._x(self.nominal, w)
            p.drawLine(int(x), 0, int(x), h)
        if self.level is not None:
            p.setPen(QPen(GREEN, 2))
            x = self._x(self.level, w)
            p.drawLine(int(x), 0, int(x), h)
            p.drawText(int(min(x + 4, w - 60)), 12, f"{self.level}")
        p.end()


class _StepStrip(QWidget):
    """The sweep of one bin drawn as cells: one row per gain, one cell per exposure step (short
    to long), grey without a plateau, amber once a plateau value is there; the newest cell is
    outlined. It never asks for more width than it is given (it does not grow the window)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.steps = []                                   # (gain, exposure_ms, plateau found)
        self.newest = None
        self.setFixedHeight(22)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

    def add_step(self, gain: float, exposure_ms: float, plateau: bool):
        """One step measured."""
        self.steps = [s for s in self.steps if (s[0], s[1]) != (gain, exposure_ms)]
        self.steps.append((float(gain), float(exposure_ms), bool(plateau)))
        self.newest = (float(gain), float(exposure_ms))
        self.update()

    def finish(self):
        """The sweep ended: no cell is the newest any more."""
        self.newest = None
        self.update()

    def paintEvent(self, event):
        """Qt handler: the cells."""
        if not self.steps:
            return
        p = QPainter(self)
        gains = sorted({g for g, _, _ in self.steps})
        rows = {g: sorted((e, found) for gg, e, found in self.steps if gg == g) for g in gains}
        columns = max(len(r) for r in rows.values())
        size = max(3.0, min(9.0, (self.width() - 2) / max(1, columns) - 2))
        row_h = max(3.0, min(size, (self.height() - 2) / max(1, len(gains)) - 2))
        for r, gain in enumerate(gains):
            for c, (exposure, found) in enumerate(rows[gain]):
                rect = QRectF(1 + c * (size + 2), 1 + r * (row_h + 2), size, row_h)
                p.fillRect(rect, AMBER if found else self.palette().mid())
                if self.newest == (gain, exposure):
                    p.setPen(QPen(self.palette().text().color(), 1))
                    p.drawRect(rect.adjusted(-1, -1, 1, 1))
        p.end()


class _ResultRow(QWidget):
    """One hardware bin: a coloured dot, ``bin N``, the sweep's cells while it runs, and a short
    result (level or status); every detail is in the tooltip. Fixed text: the row never widens."""

    def __init__(self, camera_bin: int, parent=None):
        super().__init__(parent)
        self.camera_bin = camera_bin
        self._details: list = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.dot = QLabel("●")
        self.name = QLabel(f"bin {camera_bin}")
        self.strip = _StepStrip()
        self.result = QLabel()
        self.result.setFixedWidth(84)
        self.result.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.dot)
        layout.addWidget(self.name)
        layout.addWidget(self.strip, 1)
        layout.addWidget(self.result)
        self.set_state("waiting")

    def add_step(self, gain: float, exposure_ms: float, plateaus) -> None:
        """A step of this bin's sweep was measured (``plateaus``: the plateau values found)."""
        self.strip.add_step(gain, exposure_ms, bool(plateaus))
        values = ", ".join(str(v) for v in plateaus) or "none"
        self._details.append(f"gain {gain:g}, {exposure_ms:g} ms: plateau {values}")
        self.set_state("measuring", tip="\n".join(self._details[-24:]))

    def set_state(self, state: str, text: str = "", tip: str = ""):
        """``state``: waiting / measuring / verified / inconclusive / failed; ``text``: the short
        result (a level), ``tip``: the details."""
        color = STATUS_COLORS.get(state, GREY)
        self.dot.setStyleSheet(f"color: {color.name()};")
        self.result.setText(text or {"waiting": "", "measuring": "…"}.get(state, state))
        self.result.setStyleSheet(f"color: {color.name()};" if state in STATUS_COLORS else "")
        if state not in ("waiting", "measuring"):
            self.strip.finish()
        self.setToolTip(tip)


class SaturationMeasureWidget(QWidget):
    """Measure, watch and save saturation profiles (see the module docstring)."""

    running_changed = pyqtSignal(bool)
    measurement_finished = pyqtSignal(object, object)
    profiles_saved = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._provider: Optional[Callable] = None
        self._store = None
        self._thread: Optional[_MeasureThread] = None
        self._cancel = threading.Event()
        self._records: List = []
        self._bins = (1,)
        self._current_bin = 1
        self._nominal = None
        self._rows = {}

        self.image = QLabel()
        self.image.setMinimumSize(240, 160)
        self.image.setAlignment(Qt.AlignCenter)
        # Ignored: the pixmap (scaled to the label) must never make the label - and the window - grow
        self.image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.histogram = _TopHistogram()
        self.histogram.setToolTip("Top of the raw histogram of the newest frame: plateau values amber, the measured "
                                  "threshold green, the nominal full scale dashed")
        self.step_label = QLabel("Point the camera at a bright, steady light, then Measure")
        self.step_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # long text never widens
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        self.bins_box = QComboBox()
        self.bins_box.addItems(["Current bin", "All bins"])
        self.gain_low, self.gain_high = QDoubleSpinBox(), QDoubleSpinBox()
        self.start_exposure, self.max_exposure = QDoubleSpinBox(), QDoubleSpinBox()
        for spin, suffix in ((self.gain_low, ""), (self.gain_high, ""), (self.start_exposure, " ms"),
                             (self.max_exposure, " ms")):
            spin.setDecimals(2 if suffix else 0)
            spin.setRange(0.0, 1e6)
            spin.setSuffix(suffix)
        self.gain_low.setToolTip("Gains measured: the level must be the same at both to be verified, and is "
                                 "valid only between them")
        self.gain_high.setToolTip(self.gain_low.toolTip())
        self.start_exposure.setToolTip("First exposure of every sweep (steps x2 up, /2 down to see the plateau form)")
        self.max_exposure.setToolTip("Longest exposure used")
        form = QFormLayout()
        gains = QHBoxLayout()
        gains.addWidget(self.gain_low)
        gains.addWidget(self.gain_high)
        form.addRow("Bins", self.bins_box)
        form.addRow("Gains", gains)
        form.addRow("Start exposure", self.start_exposure)
        form.addRow("Max exposure", self.max_exposure)

        self.results = QVBoxLayout()
        self.results.setSpacing(2)
        self.measure_btn = QPushButton("Measure")
        self.cancel_btn = QPushButton("Cancel")
        self.save_btn = QPushButton("Save")
        self.measure_btn.clicked.connect(self.start)
        self.cancel_btn.clicked.connect(self.cancel)
        self.save_btn.clicked.connect(self.save)
        buttons = QHBoxLayout()
        for button in (self.measure_btn, self.cancel_btn, self.save_btn):
            buttons.addWidget(button)

        grid = QGridLayout(self)
        grid.addWidget(self.image, 0, 0)
        side = QVBoxLayout()
        side.addLayout(form)
        side.addWidget(QLabel("Result per bin"))
        side.addLayout(self.results)
        side.addStretch(1)
        side.addLayout(buttons)
        grid.addLayout(side, 0, 1, 3, 1)
        grid.addWidget(self.histogram, 1, 0)
        status = QVBoxLayout()
        status.addWidget(self.step_label)
        status.addWidget(self.progress)
        status.addWidget(self.message)
        grid.addLayout(status, 2, 0)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        self._update_buttons()

    # ── host interface ──────────────────────────────────────

    def set_session_provider(self, provider: Optional[Callable]) -> None:
        """``provider() -> session`` (see the module docstring); ``None`` disables Measure."""
        self._provider = provider
        self._update_buttons()

    def set_store(self, store) -> None:
        """Where Save writes (``SaturationProfileStore``); ``None`` disables Save."""
        self._store = store
        self._update_buttons()

    def set_camera(self, bins, current_bin: int, gain: float, gain_limits, exposure_ms: float, exposure_limits,
                   nominal_full_scale: Optional[int] = None, label: str = "") -> None:
        """Offer this camera's bins, gain and exposure ranges, and default the plan from its setting."""
        self._bins = tuple(int(b) for b in bins) or (int(current_bin),)
        self._current_bin = int(current_bin)
        self._nominal = nominal_full_scale
        low, high = (float(v) for v in gain_limits)
        for spin in (self.gain_low, self.gain_high):
            spin.setRange(low, high)
        self.gain_low.setValue(float(gain))
        self.gain_high.setValue(min(high, max(float(gain) + 100.0, 4.0 * float(gain))))
        e_low, e_high = (float(v) for v in exposure_limits)
        for spin in (self.start_exposure, self.max_exposure):
            spin.setRange(e_low, e_high)
        self.start_exposure.setValue(max(e_low, min(e_high, float(exposure_ms) / 2.0)))
        self.max_exposure.setValue(min(e_high, max(8000.0, float(exposure_ms) * 8.0)))
        self.step_label.setText(label or self.step_label.text())
        self._rebuild_rows(self._selected_bins())

    def plan(self) -> SweepPlan:
        """The sweep the current settings describe."""
        gains = tuple(sorted({self.gain_low.value(), self.gain_high.value()}))
        return SweepPlan(bins=self._selected_bins(), gains=gains, start_exposure_ms=self.start_exposure.value(),
                         max_exposure_ms=self.max_exposure.value())

    def running(self) -> bool:
        """Whether a measurement is running."""
        return self._thread is not None

    def records(self) -> List:
        """The records of the last measurement."""
        return list(self._records)

    # ── actions ─────────────────────────────────────────────

    def start(self) -> bool:
        """Start a measurement (False when it cannot: no provider, running, or the host refused)."""
        if self._provider is None or self._thread is not None:
            return False
        try:
            session = self._provider()
        except RuntimeError as exc:
            self._show_message(str(exc), RED)
            return False
        plan = self.plan()
        self._records = []
        self._rebuild_rows(plan.bins)
        self._cancel = threading.Event()
        self._thread = _MeasureThread(session, plan, self._cancel, self)
        self._thread.progress.connect(self._on_progress)
        self._thread.done.connect(self._on_done)
        self._show_message("Measuring - the camera changes bin, gain and exposure; scans are locked", GREY)
        self.progress.setRange(0, 0)
        self._thread.start()
        self._update_buttons()
        self.running_changed.emit(True)
        return True

    def cancel(self) -> None:
        """Ask the running measurement to stop (the camera is restored, nothing is saved)."""
        if self._thread is not None:
            self._cancel.set()
            self._show_message("Cancelling - restoring the camera", GREY)

    def save(self) -> list:
        """Save the last records to the store; returns the ``SaveResult`` list."""
        if self._store is None or not self._records or self._thread is not None:
            return []
        results, errors = [], []
        for record in self._records:
            try:
                results.append(self._store.save(record))
            except OSError as exc:
                errors.append(f"bin {record.identity.bin}: {exc}")
        verified = sum(1 for r in self._records if r.verified)
        if errors:
            self._show_message("Not saved: " + "; ".join(errors), RED)
        else:
            kept = sum(1 for r in results if not r.active)
            self._show_message(f"Saved {len(results)} profile(s), {verified} verified"
                               + (f"; {kept} kept in history (a verified profile stays active)" if kept else ""),
                               GREEN)
        self.profiles_saved.emit(results)
        return results

    # ── internals ───────────────────────────────────────────

    def _selected_bins(self):
        return self._bins if self.bins_box.currentIndex() == 1 else (self._current_bin,)

    def _rebuild_rows(self, bins):
        while self.results.count():
            item = self.results.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._rows = {}
        for b in bins:
            row = _ResultRow(b)
            self._rows[b] = row
            self.results.addWidget(row)

    def _show_message(self, text, color):
        self.message.setText(text)
        self.message.setStyleSheet(f"color: {color.name()};")

    def _update_buttons(self):
        running = self._thread is not None
        self.measure_btn.setEnabled(self._provider is not None and not running)
        self.cancel_btn.setEnabled(running)
        self.save_btn.setEnabled(self._store is not None and bool(self._records) and not running)
        for widget in (self.bins_box, self.gain_low, self.gain_high, self.start_exposure, self.max_exposure):
            widget.setEnabled(not running)

    def _on_progress(self, info):
        b, step = info["bin"], info["step"]
        self.step_label.setText(f"bin {b}  ·  gain {info['gain']:g}  ·  {info['exposure_ms']:g} ms  "
                                f"·  max {max(step.maxima)}")
        row = self._rows.get(b)
        if row is not None:
            row.add_step(info["gain"], info["exposure_ms"], list(info["plateaus"]))
        for other, other_row in self._rows.items():           # the newest cell: only the bin measuring now
            if other != b and other_row.strip.newest is not None:
                other_row.strip.finish()
        nominal = self._nominal or (max(step.maxima) if step.maxima else 1)
        self.histogram.set_step(step.histograms[-1] if step.histograms else {}, info["plateaus"],
                                DEFAULT_PARAMS.floor(nominal), nominal)
        frame = info.get("frame")
        if frame is not None:
            self._show_frame(frame)

    def _show_frame(self, frame):
        f = np.asarray(frame, dtype=np.float32)
        top = float(f.max()) or 1.0
        img = np.ascontiguousarray((f / top * 255.0).astype(np.uint8))
        qimage = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8).copy()
        self.image.setPixmap(QPixmap.fromImage(qimage).scaled(self.image.size(), Qt.KeepAspectRatio))

    def _on_done(self, records, problems, error, cancelled):
        self._thread.wait()
        self._thread = None
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self._records = [] if cancelled else list(records)
        for record in self._records:
            row = self._rows.get(record.identity.bin)
            if row is None:
                continue
            head = f"bin {record.identity.bin}: {record.status}"
            if record.verified:
                low, high = record.gain_range
                head += f", level {record.level}, gain {low:g}-{high:g}"
            tip = "\n".join([head] + [f"{p.value}: {p.role} - {p.reason}" for p in record.plateaus]
                            + list(record.reasons) + ["", *row._details[-24:]])
            row.set_state(record.status, f"{record.level} ✓" if record.verified else record.status, tip)
        if cancelled:
            for row in self._rows.values():
                row.set_state("waiting", "cancelled", row.toolTip())
        if self._records and self._records[-1].verified:
            self.histogram.set_level(self._records[-1].level)
        if problems:
            self._show_message("Camera NOT restored: " + "; ".join(problems) + " - scans stay locked", RED)
        elif cancelled:
            self._show_message("Cancelled; the camera is back to its setting, nothing saved", GREY)
        elif error:
            self._show_message(f"Measurement failed: {error}; the camera is back to its setting", RED)
        else:
            verified = sum(1 for r in self._records if r.verified)
            self._show_message(f"{verified} of {len(self._records)} bin(s) verified; the camera is back to its "
                               f"setting. Save to use them.", GREEN if verified else AMBER)
        self._update_buttons()
        self.running_changed.emit(False)
        self.measurement_finished.emit(self._records, list(problems))

    def closeEvent(self, event):
        """Qt handler: a running measurement is cancelled and waited for (the camera is restored)."""
        if self._thread is not None:
            self._cancel.set()
            self._thread.wait(60000)
        super().closeEvent(event)


__all__ = ["SaturationMeasureWidget"]
