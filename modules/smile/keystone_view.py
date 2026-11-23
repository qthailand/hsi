# smile/keystone_view.py
"""Keystone page of the smile calibration widget (Qt).

Workflow (fore-lens off for the smile, on for the keystone):

1. *Smile model* - by default the smile model stored for the camera bin of the target frame
   (measured with the fore-lens off and applied on the Smile tab). When this session also has a
   smile analysis of a lamp frame, the user may pick it instead (joint smile + keystone fit).
2. *Line target* - grabbed or loaded; the bars are traced at once in a small worker thread and
   drawn on the *Preview* / *Deviation* plots so the keystone can be checked by eye before any fit.
3. *White frame* - optional (recommended): divides out the illumination.
4. *Analyze* - :meth:`SmileCorrector.fit_keystone` on the stored model (keystone only), or
   :func:`modules.smile.keystone.analyze_keystone` on the session analysis (joint fit).
5. *Apply* - the owning :class:`~modules.smile.widget.SmileCorrectionWidget` writes ONE smile +
   keystone model for the target's camera setting (:attr:`KeystonePage.applyKeystoneRequested`).

Data conventions:
    Frames are 2-D ``(slit rows, spectral columns)`` with the same orientation, bin and size as
    the smile model. Grabbed frames are oriented and dark-subtracted by the host; files are used
    as stored.

Threading:
    Widgets live on the GUI thread; only :meth:`KeystoneWorker.run` and :meth:`PreviewWorker.run`
    run in worker threads and touch no widget.
"""
from __future__ import annotations

import html
import logging
import math
import os
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QThread, Qt, QRectF, pyqtSignal
from PyQt5.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QToolButton,
    QScrollArea,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QColor, QFont, QPainter, QPen

from .keystone_spacing import spacing_variation

from .core import auto_sat_level, load_frame
from .ui_theme import colors
from .keystone import (
    RETRACE_WARN_PX,
    ZONE_SPAN_MIN,
    analyze_keystone,
    bar_coverage,
    coverage_warnings,
    detect_polarity,
    frames_same_scene,
    keystone_report,
    normalize_target,
    trace_target_lines,
)

logger = logging.getLogger(__name__)

__all__ = ["KeystonePage", "KeystoneWorker", "PreviewWorker", "target_preview", "nice_gain"]

_IMAGE_FILTER = ("Images or a model with frames (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.npz);;"
                 "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;Model with frames (*.npz);;All (*.*)")
_POLARITIES = [("Auto (from the image)", "auto"), ("Dark bars on bright", "dark"), ("Bright bars on dark", "bright")]
# Canvas index -> (tab name, export file). Indices 0-2 are the original check plots (kept).
_PLOTS = [
    ("Target  Before / After", "Keystone_Target_Before_After.png"),
    ("Row Shift Map", "Keystone_Row_Shift_Map.png"),
    ("Re-trace Residual", "Keystone_Retrace_Residual.png"),
    ("Preview", "Keystone_Preview.png"),
    ("Deviation (magnified)", "Keystone_Deviation.png"),
    ("Distortion Grid", "Keystone_Distortion_Grid.png"),
    ("White", "Keystone_White.png"),
    ("Spacing Uniformity", "Keystone_Spacing_Uniformity.png"),
]
BEFORE_AFTER, SHIFT_MAP, RETRACE, PREVIEW, DEVIATION, GRID, WHITE, SPACING = range(8)
#: Workflow tabs of the plot area: two before Analyze (open with a target), two after it.
TAB_SPECTROGRAM, TAB_TARGET, TAB_CORRECTION, TAB_GEOMETRY = range(4)
_ANALYSIS_TABS = (TAB_CORRECTION, TAB_GEOMETRY)
# canvas index -> its workflow tab
_GROUP = {PREVIEW: TAB_TARGET, DEVIATION: TAB_TARGET, WHITE: TAB_TARGET,
          BEFORE_AFTER: TAB_CORRECTION, RETRACE: TAB_CORRECTION, SPACING: TAB_CORRECTION,
          SHIFT_MAP: TAB_GEOMETRY, GRID: TAB_GEOMETRY}

# workflow states, step rows and dashboard parts shared with the Smile page (modules.smile.ui_parts)
from .ui_parts import (BAD as _BAD, BUSY as _BUSY, CHIP as _CHIP, GLYPH as _GLYPH, GOOD as _GOOD,  # noqa: E402
                       OPTIONAL as _OPTIONAL, TODO as _TODO, WARN as _WARN, DashboardMixin, Step as _Step)


def _c(key: str) -> str:
    """Colour *key* of the active application theme (light or dark)."""
    return colors()[key]


def model_setting(camera_setting: Optional[dict], shape, fallback: Optional[Callable] = None):
    """Camera setting for a frame read from a model file: the model's own recorded setting
    (more trustworthy than the camera's current one), else ``fallback(shape)``, else ``None``."""
    if camera_setting:
        try:
            from modules.contracts import CameraSetting
            setting = CameraSetting.from_dict(camera_setting)
            return setting if setting.frame_shape else setting.with_frame(tuple(shape))
        except Exception:  # noqa: BLE001 - an odd record falls back to the current camera
            pass
    if fallback is not None:
        try:
            return fallback(tuple(shape))
        except Exception:  # noqa: BLE001
            return None
    return None


def _ascii(text: str) -> str:
    """User-visible mismatch texts stay ASCII (``CameraSetting.mismatch`` uses a not-equal sign)."""
    return text.replace("≠", "!=").replace("·", "-").replace("×", "x")


def _setting_problems(reference, other) -> List[str]:
    """Differences between two camera settings (empty when one is unknown or they agree)."""
    if reference is None or other is None:
        return []
    try:
        return [_ascii(p) for p in reference.mismatch(other)]
    except Exception:  # noqa: BLE001
        return []


def _num(v) -> float:
    """JSON value -> float (``None`` = not measured -> NaN)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def nice_gain(ideal: float, minimum: float = 1.0) -> float:
    """Largest of 1, 2, 3, 5 x 10^k not above *ideal* (at least *minimum*): a readable magnification."""
    if not np.isfinite(ideal) or ideal <= minimum:
        return float(minimum)
    exp = 10.0 ** math.floor(math.log10(ideal))
    for m in (5, 3, 2, 1):
        if m * exp <= ideal:
            return float(max(minimum, m * exp))
    return float(minimum)


def _nice_floor(value: float) -> float:
    """Largest of 1, 2, 5 x 10^k not above *value* (> 0)."""
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exp = 10.0 ** math.floor(math.log10(value))
    for m in (5, 2, 1):
        if m * exp <= value:
            return float(m * exp)
    return float(exp)


def _draw(canvas) -> None:
    """Redraw a canvas with constrained layout (kept on resize, unlike a one-off tight layout)."""
    try:
        canvas.fig.set_layout_engine("constrained")
    except Exception:  # noqa: BLE001
        pass
    canvas.draw_idle()


def _mtime(path) -> Optional[float]:
    """Modification time of a file, or ``None`` when it does not exist."""
    try:
        return os.path.getmtime(str(path))
    except OSError:
        return None


def _short_date(created: str) -> str:
    """``2026-11-19T14:05:33`` -> ``2026-11-19 14:05``."""
    text = (created or "").replace("T", " ")
    return text[:16] if len(text) >= 16 else (text or "?")


def _stretch(img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    vals = img[mask] if mask is not None else img[np.isfinite(img)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, [1, 99.5])
    return float(lo), float(max(hi, lo + 1e-6))


# ------------------------------------------------------------------------------------- preview maths
def _bar_fan(bars, w: int, h: int) -> dict:
    """Per bar: row change across a common width (right - left); a slope over the rows = keystone,
    a constant = tilt.

    Returns:
        ``rows``, ``fan`` (px), ``xl``, ``xr`` (common columns), ``tilt_deg`` (fan at the frame centre
        row over the width), ``top``, ``bottom`` (fit at the first / last bar), ``coef`` (``fan = c0 + c1 *
        (row - (h-1)/2)``).
    """
    if not bars:
        return {}
    xl = float(np.median([b.x.min() for b in bars]))
    xr = float(np.median([b.x.max() for b in bars]))
    width = max(1.0, xr - xl)
    rows, fan = [], []
    for b in bars:
        a, c = max(float(b.x.min()), xl), min(float(b.x.max()), xr)
        if c - a < 0.3 * width:
            continue
        ya, yc = b.y_at(np.array([a, c]), w)
        rows.append(float(b.y_at(np.array([0.5 * (a + c)]), w)[0]))
        fan.append(float((yc - ya) / (c - a) * width))
    out = dict(rows=np.array(rows), fan=np.array(fan), xl=xl, xr=xr)
    if len(rows) >= 2:
        rc = (h - 1) / 2.0
        coef = np.polynomial.polynomial.polyfit(np.array(rows) - rc, np.array(fan), 1)
        top, bottom = (float(np.polynomial.polynomial.polyval(r - rc, coef)) for r in (min(rows), max(rows)))
        out.update(coef=coef, tilt_deg=float(np.degrees(np.arctan2(coef[0], width))), top=top, bottom=bottom,
                   top_row=float(min(rows)), bottom_row=float(max(rows)))
    return out


def target_preview(target: np.ndarray, white: Optional[np.ndarray] = None, polarity: str = "dark") -> dict:
    """Bars of a line-target frame for the preview plots (no smile model needed).

    Args:
        target: Line-target frame ``(H, W)`` (raw counts).
        white: Optional white frame of the same size.
        polarity: ``"dark"``, ``"bright"`` or ``"auto"`` (decided from the image).

    Returns:
        dict: ``shape``, ``polarity`` (the one used), ``polarity_auto`` (detection details or ``None``),
        ``norm`` (normalised target), ``bars`` (:class:`TargetLine`),
        ``info`` (tracing info or ``None``), ``error`` (text when no bars could be followed),
        ``saturated_pct``, ``coverage`` (median, 0-1), ``contrast`` (median bar depth, 0-1),
        ``x_ref`` (reference column), ``fan`` (:func:`_bar_fan`), ``white``, ``lit_pct`` (with white),
        ``alt`` (``(other polarity, bars found)`` when few bars were found), ``span_coverage``
        (:func:`~modules.smile.keystone.bar_coverage`: rows followed per spectral zone) and
        ``span_warnings`` (zones / slit ends where the keystone would be extrapolated).
    """
    t = np.asarray(target, np.float32)
    h, w = t.shape
    sat = auto_sat_level(t)
    saturated = float(np.mean(t >= 0.98 * sat)) * 100.0 if np.isfinite(sat) else 0.0
    norm = normalize_target(t, white)
    polarity_auto = None
    if polarity == "auto":
        polarity, polarity_auto = detect_polarity(t, normalized=norm)
    out = dict(shape=(h, w), polarity=polarity, polarity_auto=polarity_auto, norm=norm, bars=[], info=None, error="",
               saturated_pct=saturated,
               coverage=float("nan"), contrast=float("nan"), x_ref=(w - 1) / 2.0, fan={}, white=white is not None,
               lit_pct=float(np.mean(norm > 0) * 100.0) if white is not None else float("nan"), alt=None)
    try:
        bars, info = trace_target_lines(t, polarity=polarity, normalized=norm)
        out.update(bars=bars, info=info)
    except ValueError as exc:
        out["error"] = str(exc)
        bars, info = [], None
    if len(bars) < 5:
        other = "bright" if polarity == "dark" else "dark"
        try:
            n_other = len(trace_target_lines(t, polarity=other, normalized=norm)[0])
        except ValueError:
            n_other = 0
        if n_other > len(bars):
            out["alt"] = (other, n_other)
    if not bars:
        return out
    lo = float(np.median([b.x.min() for b in bars]))
    hi = float(np.median([b.x.max() for b in bars]))
    out["x_ref"] = float(np.clip((w - 1) / 2.0, lo, hi))
    out["coverage"] = float(np.median([b.coverage for b in bars]))
    # bar depth at the seed column against the local background
    seed = int(info["seed_column"])
    c0, c1 = max(0, seed - 2), min(w, seed + 3)
    prof = np.mean(norm[:, c0:c1], axis=1)
    period = max(4.0, float(info.get("period") or 8.0))
    depth = []
    for b in bars:
        r = int(round(float(b.y_at(np.array([seed]), w)[0])))
        a, z = max(0, r - int(period)), min(h, r + int(period) + 1)
        win = prof[a:z]
        win = win[win > 0]
        if win.size < 3 or not 0 <= r < h or prof[r] <= 0:
            continue
        if polarity == "dark":
            bg = float(np.percentile(win, 90))
            depth.append((bg - float(prof[r])) / max(bg, 1e-6))
        else:
            bg = float(np.percentile(win, 10))
            depth.append((float(prof[r]) - bg) / max(float(prof[r]), 1e-6))
    if depth:
        out["contrast"] = float(np.clip(np.median(depth), 0.0, 1.0))
    out["fan"] = _bar_fan(bars, w, h)
    span = bar_coverage(bars, (h, w), normalized=norm, step=int(info.get("step") or 5))
    out["span_coverage"] = span
    out["span_warnings"] = coverage_warnings(span)
    return out


# ------------------------------------------------------------------------------------- workers
class KeystoneWorker(QThread):
    """Run the keystone analysis off the GUI thread.

    Session mode (``stored_path`` None): :func:`analyze_keystone` on the lamp's smile analysis
    (joint fit). Stored mode: a private :class:`~modules.smile.corrector.SmileCorrector` is loaded
    from ``stored_path`` here and :meth:`~SmileCorrector.fit_keystone` fits only the keystone.

    Signals:
        done: ``("session", smile_analysis, KeystoneAnalysis)`` or ``("stored", corrector, KeystoneAnalysis)``.
        failed: Error text.
    """

    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, analysis, target: np.ndarray, kwargs: dict, stored_path: Optional[str] = None):
        """Args:
            analysis: The lamp's :class:`~modules.smile.analysis.SmileAnalysis` (read only), or
                ``None`` in stored mode.
            target: Line-target frame (a private copy).
            kwargs: Passed to :func:`analyze_keystone` / ``fit_keystone`` (``white``, ``polarity``,
                ``inputs``, stored mode also ``lamp``).
            stored_path: Stored smile model to fit the keystone onto (stored mode).
        """
        super().__init__()
        self.analysis = analysis
        self._target = target
        self._kwargs = kwargs
        self._stored_path = stored_path

    def run(self) -> None:
        """Compute the geometry; emit ``done`` or ``failed``."""
        try:
            if self._stored_path:
                from .corrector import SmileCorrector
                corrector = SmileCorrector.load(self._stored_path)
                self.done.emit(("stored", corrector, corrector.fit_keystone(self._target, **self._kwargs)))
            else:
                self.done.emit(("session", self.analysis, analyze_keystone(self.analysis, self._target,
                                                                          **self._kwargs)))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Keystone analysis failed")
            self.failed.emit(str(exc))


class PreviewWorker(QThread):
    """Trace the bars of a target frame for the preview (:func:`target_preview`).

    Signals:
        done: ``(token, preview dict)``.
        failed: ``(token, error text)``.
    """

    done = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(self, token: int, target: np.ndarray, white: Optional[np.ndarray], polarity: str):
        super().__init__()
        self.token = token
        self._args = (target, white, polarity)

    def run(self) -> None:
        try:
            self.done.emit((self.token, target_preview(*self._args)))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Keystone preview failed")
            self.failed.emit((self.token, str(exc)))


# ------------------------------------------------------------------------------------- step rows
def _chip(text: str, kind: str, tip: str = "") -> str:
    bg, fg, border = colors()[_CHIP.get(kind, "chip_info")]
    return text, f"QLabel {{ background:{bg}; color:{fg}; border:1px solid {border}; border-radius:9px; " \
                 f"padding:2px 9px; }}", tip


class _TiltIndicator(QWidget):
    """Sensor tilt pictogram: numerical angle is exact; visual angle is exaggerated.

    A positive detector slope (row increases toward the right) looks clockwise
    in Qt/image coordinates, never interpreted as a physical mounting command.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = float("nan")
        self.setMinimumSize(110, 55)
        self.setToolTip("Detector image-coordinate estimate, not a physical inclinometer. "
                        "The icon's angle is exaggerated for legibility.")

    def set_angle(self, value) -> None:
        self.angle = _num(value)
        self.update()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(QPen(QColor("#8eb7c4"), 1, Qt.DashLine))
        p.drawLine(int(w * .1), h // 2, int(w * .9), h // 2)
        p.save()
        p.translate(w / 2, h / 2)
        if np.isfinite(self.angle):
            p.rotate(float(np.clip(self.angle * 12, -15, 15)))
        p.setPen(QPen(QColor("#225074"), 2))
        p.setBrush(QColor("#f6fbff"))
        rect_w = min(104, w - 28)
        rect_h = min(46, h - 15)
        p.drawRoundedRect(QRectF(-rect_w / 2, -rect_h / 2, rect_w, rect_h), 5, 5)
        p.setPen(QColor("#078d81"))
        f = QFont(self.font())
        f.setBold(True)
        f.setPointSize(13)
        p.setFont(f)
        label = f"{self.angle:+.3f}°" if np.isfinite(self.angle) else "—"
        p.drawText(QRectF(-rect_w / 2, -rect_h / 2, rect_w, rect_h), Qt.AlignCenter, label)
        p.restore()



# ------------------------------------------------------------------------------------- the page
class KeystonePage(DashboardMixin, QWidget):
    """Smile model (stored by default) + line target (+ white) -> smile + keystone model, with a
    live preview of the bars and check plots.

    Signals:
        grabRequested: ``"target"`` or ``"white"``; the owner asks the host for one frame and
            hands it back through :meth:`set_frame`.
        keystoneReady: The :class:`~modules.smile.keystone.KeystoneAnalysis` just computed
            (:attr:`result_mode` says onto which smile).
        applyKeystoneRequested: *Apply* pressed (the owner stores the model for the camera).
        saveKeystoneRequested: *Save Matrix* pressed.
        statusChanged: Message for the owner's status bar.

    Attributes:
        target, white: Frames ``float32 (H, W)`` or ``None``.
        target_setting, white_setting: ``CameraSetting`` they were taken with (``None`` = unknown).
        result: Latest :class:`KeystoneAnalysis` of this session, or ``None``.
        result_mode: ``"stored"`` (keystone onto the stored smile) or ``"session"`` (joint fit).
        result_corrector: Stored mode: the :class:`SmileCorrector` (stored smile + this keystone).
        result_setting: Camera setting of the target used for :attr:`result`.
        stored: The stored smile model found for the target's setting (``dict`` with ``path``,
            ``corrector``, ``problems``, ``setting``, ``fallback``) or ``None`` (no store / setting).
        preview: Latest :func:`target_preview` of the current target, or ``None``.
    """

    grabRequested = pyqtSignal(str)
    keystoneReady = pyqtSignal(object)
    keystoneCleared = pyqtSignal()
    applyKeystoneRequested = pyqtSignal()
    saveKeystoneRequested = pyqtSignal()
    statusChanged = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None,
                 setting_for_file: Optional[Callable[[Tuple[int, int]], object]] = None):
        """Args:
            parent: Parent widget.
            setting_for_file: ``shape -> CameraSetting or None`` for frames loaded from disk.
        """
        super().__init__(parent)
        self._setting_for_file = setting_for_file
        self._store = None
        self._current_setting: Optional[Callable[[], object]] = None
        self._smile = None                  # SmileAnalysis of the lamp (this session)
        self._lamp_shape = None
        self._lamp_setting = None
        self.target = self.white = None
        self.target_setting = self.white_setting = None
        self._target_source = self._white_source = ""
        self.result = None
        self.result_mode = None
        self.result_corrector = None
        self.result_setting = None
        self.result_source = None           # stored mode: (smile file, its mtime) the keystone was fitted onto
        self.stored = None
        self._stored_cache = {}
        self._smile_choice = None           # None = automatic (stored when it exists)
        self._pending_setting = None
        self._pending_mode = None
        self._pending_source = None
        self._worker: Optional[KeystoneWorker] = None
        self._grab_enabled = False
        self._applied = ""
        self.preview = None
        self._preview_token = 0
        self._frame_gen = 0                 # bumped on every target / white change
        self._failure = ""                  # message of the last failed analysis (until the next frame / run)
        self._pending_gen = 0               # _frame_gen when the running analysis started
        self._preview_worker: Optional[PreviewWorker] = None
        self._preview_again = False
        self._preview_error = ""
        self._build_ui()
        self._clear_plots("Run Analyze Keystone to see the check plots", (BEFORE_AFTER, SHIFT_MAP, RETRACE, GRID, SPACING))
        self._clear_plots("Load or grab a line-target frame: its bars are drawn here at once", (PREVIEW, DEVIATION))
        self._clear_plots("No white frame (optional, recommended)", (WHITE,))
        self._refresh()

    # ------------------------------------------------------------------ UI
    def _panel(self, index: int, *, minimum: int = 200) -> QWidget:
        """Dashboard panel of canvas *index* (the grid's panel carries the magnify switch)."""
        extras = ()
        if index == GRID:
            self.chk_exaggerate = QCheckBox("Magnified (visual only)")
            self.chk_exaggerate.setChecked(True)   # a ~1 px keystone is invisible at true scale; the title gives xG
            self.chk_exaggerate.setToolTip("Enlarge the DRAWN distortion only; the calibration model, values and "
                                           "corrected image do not change. The plot title shows actual display ×G.")
            self.chk_exaggerate.toggled.connect(lambda _on: self._plot_grid(self.result))
            extras = (self.chk_exaggerate,)
        return self._make_plot_panel(index, _PLOTS[index][0], minimum=minimum, extras=extras)

    def _metric(self, title: str, *, tilt: bool = False):
        frame = QFrame()
        frame.setObjectName("ksMetric")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(9, 7, 9, 6)
        lay.setSpacing(2)
        heading = QLabel(title)
        heading.setObjectName("ksMetricHeading")
        lay.addWidget(heading)
        if tilt:
            self.tilt_indicator = _TiltIndicator(frame)
            lay.addWidget(self.tilt_indicator, stretch=1)
            value = QLabel("Detector image coordinates · icon exaggerated")
            value.setObjectName("ksMetricHint")
        else:
            value = QLabel("—")
            value.setObjectName("ksMetricValue")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(value)
        return frame, value

    #: Warnings that are standing reminders, not a verdict on this analysis (kept in the report).
    _REMINDERS = ("Smile taken from the stored model",)

    def _update_dashboard(self) -> None:
        """Read-only UI values; never changes calibration or host Apply rules."""
        if not hasattr(self, "metric_rms"):
            return
        p = self.preview or {}
        result = self.result
        bars = len(p.get("bars") or []) if p else 0
        if result is not None:
            bars = len(result.target_lines)
        self.metric_bars.setText(f"{bars} detected" if bars else "—")
        self.metric_rms.setText("—")
        self.metric_shift.setText("—")
        self.metric_fitter.setText("—")
        self.metric_status.setText("Before analysis")
        for lab in (self.metric_bars, self.metric_rms, self.metric_shift, self.metric_fitter, self.metric_status):
            self._paint_metric(lab, "text")
        if bars:
            self._paint_metric(self.metric_bars, "good" if bars >= 3 else "bad")
        self._paint_banner("info")
        preview_tilt = _num((p.get("fan") or {}).get("tilt_deg"))
        self.tilt_indicator.set_angle(preview_tilt)
        self.metric_tilt_note.setText(
            ("Right side lower" if preview_tilt > 0 else "Right side higher" if preview_tilt < 0 else "Level")
            + " · displayed angle exaggerated" if np.isfinite(preview_tilt) else "Tilt pending")
        cov = _num(p.get("coverage"))
        self.coverage_value.setValue(round(100 * np.clip(cov, 0, 1)) if np.isfinite(cov) else 0)
        self.coverage_note.setText(
            f"Median traced bar width: {100*cov:.1f}%" if np.isfinite(cov) else
            "Median traced bar width: not available")
        if result is None:
            self.lbl_spacing_status.setText("Spacing variation: after analysis (diagnostic only)")
            self.lbl_ready.setText("TARGET INSPECTION  ·  Load / grab a line target, then Analyze Keystone")
            return
        q, s = result.qc, result.summary
        rms = _num(q.get("retrace_target_median_bar_px"))
        if not np.isfinite(rms):
            rms = _num(q.get("retrace_target_rms_px"))
        self.metric_rms.setText(f"{rms:.3f} px" if np.isfinite(rms) else "N/A")
        self._paint_metric(self.metric_rms, "bad" if not np.isfinite(rms) or rms > 2 * RETRACE_WARN_PX else
                           "warn" if rms > RETRACE_WARN_PX else "good")
        shift = _num(s.get("keystone_max_px"))
        self.metric_shift.setText(f"{shift:.2f} px" if np.isfinite(shift) else "N/A")
        self.metric_fitter.setText(str(q.get("dy_fitter") or q.get("dy_model") or "Auto"))
        self.metric_fitter.setToolTip(next((n for n in q.get("notes") or [] if n.startswith("keystone fitter")), ""))
        angle = _num(s.get("tilt_deg"))
        self.tilt_indicator.set_angle(angle)
        self.metric_tilt_note.setText(
            ("Right side lower" if angle > 0 else "Right side higher" if angle < 0 else "Level")
            + " · detector image coordinates" if np.isfinite(angle) else "Tilt unavailable")
        # Warnings are surfaced, not translated into a new Apply gate.
        valid_rms = bool(np.isfinite(rms) and rms <= RETRACE_WARN_PX)
        warnings = [w for w in (result.warnings or []) if not w.startswith(self._REMINDERS)]
        n_rem = len(result.warnings or []) - len(warnings)
        if not valid_rms:
            status, state = "Review QC", ("bad" if not np.isfinite(rms) or rms > 2 * RETRACE_WARN_PX else _WARN)
        elif warnings:
            status, state = f"{len(warnings)} warning" + ("s" if len(warnings) > 1 else ""), _WARN
        else:
            status, state = "Re-trace PASS", _GOOD
        self.metric_status.setText(status)
        self._paint_metric(self.metric_status, {_GOOD: "good", _WARN: "warn"}.get(state, "bad"))
        if state == _GOOD:
            self.lbl_ready.setText("ANALYSIS COMPLETE  ·  Review the quality plots; Apply is controlled by the host"
                                   + (f"  ·  {n_rem} reminder(s) in the technical report" if n_rem else ""))
        else:
            self.lbl_ready.setText("REVIEW REQUIRED  ·  " + (warnings[0] if warnings else
                                                            "Re-trace residual above %.2f px: check the re-trace "
                                                            "and coverage plots" % RETRACE_WARN_PX)
                                   + (f"  (+{len(warnings) - 1} more in the technical report)" if len(warnings) > 1 else ""))
        self._paint_banner(state if state in (_GOOD, _WARN) else _BAD)
        if self._spacing_stats and np.isfinite(self._spacing_stats.get("after_rms", float("nan"))):
            v = self._spacing_stats
            self.lbl_spacing_status.setText(f"Spacing variation: {v['before_rms']:.3f} → {v['after_rms']:.3f} px RMS"
                                            "  (relative to the reference column; diagnostic only)")
        else:
            self.lbl_spacing_status.setText("Spacing variation: not available")

    def _build_ui(self) -> None:
        from .widget import PlotCanvas  # imported here intentionally: widget.py references KeystonePage
        from .spectrogram_view import SpectrogramView
        self.setObjectName("keystonePage")
        palette = colors(self)
        page_bg = palette["page"]
        card_bg = palette["card"]
        fg = palette["text"]
        border = palette["border"]
        self.setStyleSheet(f"""
            #keystonePage {{ background: {page_bg}; color: {fg}; }}
            #ksTitleBar {{ background: #193d5c; border-radius: 7px; }}
            #ksTitleBar QLabel {{ color: #ffffff; font-weight: 700; font-size: 12pt; }}
            #ksMetric, #ksPlotPanel {{ background: {card_bg}; border: 1px solid {border}; border-radius: 8px; }}
            #ksMetricHeading {{ color: {palette['muted']}; font-size: 9pt; }}
            #ksMetricValue {{ color: {palette['good']}; font-size: 15pt; font-weight: bold; }}
            #ksMetricHint {{ color: {palette['muted']}; font-size: 8pt; }}
            #ksPlotTitle {{ color: {fg}; font-size: 10pt; font-weight: bold; border: none; }}
            #ksReady {{ background: {palette['chip_warn'][0]}; color: {palette['chip_warn'][1]};
                        border: 1px solid {palette['chip_warn'][2]}; border-radius: 6px; padding: 8px; }}
            QTabWidget::pane {{ border: 1px solid {border}; border-radius: 8px; top: -1px; }}
            QTabBar::tab {{ padding: 6px 14px; margin-right: 3px; }}
            QTabBar::tab:selected {{ border-bottom: 3px solid #0b9398; }}
        """)
        root = QVBoxLayout(self)
        root.setContentsMargins(5, 5, 5, 5)
        root.setSpacing(6)

        self.btn_load_target = QPushButton("Load target...")
        self.btn_load_target.setToolTip("Line-target frame(s). Several files are averaged only if they show the "
                                        "same scene (a static target); otherwise they are refused.")
        self.btn_load_target.clicked.connect(self._on_load_target)
        self.btn_grab_target = QPushButton("Grab target")
        self.btn_grab_target.setToolTip("One oriented frame of the line target (lens on, same bin as the smile "
                                        "model); dark-subtracted when the host's dark correction is on")
        self.btn_grab_target.clicked.connect(lambda: self.grabRequested.emit("target"))
        self.btn_load_white = QPushButton("Load white...")
        self.btn_load_white.setToolTip("Optional white reference: same illumination, working distance and exposure "
                                       "with the target removed; normalises the illumination profile along the slit")
        self.btn_load_white.clicked.connect(self._on_load_white)
        self.btn_grab_white = QPushButton("Grab white")
        self.btn_grab_white.clicked.connect(lambda: self.grabRequested.emit("white"))
        self.btn_clear_white = QPushButton("Clear white")
        self.btn_clear_white.clicked.connect(self.clear_white)
        self.combo_polarity = QComboBox()
        for label, value in _POLARITIES:
            self.combo_polarity.addItem(label, value)
        self.combo_polarity.setToolTip("Bar polarity. Auto: decided from the image (the bars are the narrower "
                                       "features; when bars and gaps are equally wide both are traced and the "
                                       "better kept). Or set dark / bright bars.")
        self.combo_polarity.currentIndexChanged.connect(lambda _i: self._polarity_changed())
        self.combo_smile = QComboBox()
        self.combo_smile.addItem("Stored model (default)", "stored")
        self.combo_smile.addItem("This session's lamp analysis (joint fit)", "session")
        self.combo_smile.setToolTip("Stored: the smile measured with the fore-lens off and applied for this bin; "
                                    "only the keystone is fitted.\nSession: the lamp frame analysed on the Smile tab "
                                    "now; smile and keystone are fitted together.")
        self.combo_smile.activated.connect(self._on_smile_choice)
        self.btn_analyze = QPushButton("Analyze Keystone")
        self.btn_analyze.setStyleSheet("font-weight: bold;")
        self.btn_analyze.clicked.connect(lambda: self.analyze())
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setFixedWidth(65)
        self.busy.setVisible(False)
        self.btn_apply = QPushButton("Apply smile + keystone")
        self.btn_apply.setToolTip("Store ONE smile + keystone model for the target's camera setting (the previous "
                                  "file is kept in history/) and load it into the live HSI session.")
        self.btn_apply.clicked.connect(self.applyKeystoneRequested.emit)
        self.btn_save_matrix = QPushButton("Save model...")
        self.btn_save_matrix.setToolTip("Save the smile + keystone model to a file of your choice.")
        self.btn_save_matrix.clicked.connect(self.saveKeystoneRequested.emit)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        lay = QVBoxLayout(left)
        lay.setContentsMargins(1, 1, 5, 1)
        lay.setSpacing(6)
        workflow_title = QLabel("WORKFLOW")
        workflow_title.setObjectName("ksTitleBar")
        workflow_title.setStyleSheet("QLabel { background:#193d5c;color:white;padding:9px;font-weight:bold;"
                                     "border-radius:6px; }")
        lay.addWidget(workflow_title)
        self.step_smile = _Step(1, "Smile model", "")
        self.step_smile.head.addWidget(self.combo_smile)
        self.step_target = _Step(2, "Line target (fore-optics on)",
                                 "Image a bar target across the slit at the scanning working distance and focus")
        self.step_target.add(self.btn_load_target, self.btn_grab_target, QLabel("Bars:"), self.combo_polarity)
        self.step_white = _Step(3, "White reference (optional)",
                                "Recommended: same illumination with the target removed, to normalise the "
                                "illumination profile")
        self.step_white.add(self.btn_load_white, self.btn_grab_white, self.btn_clear_white)
        self.step_analyze = _Step(4, "Analyze",
                                  "Fit the keystone (spatial misregistration) onto the selected smile model")
        self.step_analyze.add(self.btn_analyze, self.busy)
        self.step_apply = _Step(5, "Apply",
                                "Store one combined smile + keystone correction model for this camera setting")
        self.step_apply.add(self.btn_apply, self.btn_save_matrix)
        for step in (self.step_smile, self.step_target, self.step_white, self.step_analyze, self.step_apply):
            step.finish()
            lay.addWidget(step)
        self.lbl_frames = QLabel()
        self.lbl_frames.setVisible(False)
        lay.addWidget(self.lbl_frames)
        self.btn_report = QToolButton()
        self.btn_report.setText("Technical report ▸")
        self.btn_report.setCheckable(True)
        self.btn_report.setChecked(False)
        self.btn_report.setToolTip("Show detailed fitting and QC diagnostics (kept for advanced users)")
        self.btn_report.toggled.connect(self._toggle_report)
        lay.addWidget(self.btn_report)
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setLineWrapMode(QTextEdit.WidgetWidth)
        self.report.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 8.5pt;")
        self.report.setMinimumHeight(180)
        self.report.setVisible(False)
        lay.addWidget(self.report, stretch=4)
        lay.addStretch(1)
        self.btn_report.setChecked(True)                # the QC numbers stay in view; collapse when not needed
        split.addWidget(left)

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(3, 0, 0, 0)
        rlay.setSpacing(6)
        title = QLabel("CALIBRATION QUALITY SUMMARY")
        title.setStyleSheet("QLabel { background:#193d5c;color:white;padding:8px;font-weight:bold;"
                            "border-radius:6px; }")
        rlay.addWidget(title)
        metrics = QHBoxLayout()
        metrics.setSpacing(5)
        a, self.metric_rms = self._metric("Re-trace RMS")
        b, self.metric_bars = self._metric("Target bars")
        c, self.metric_shift = self._metric("Max Keystone")
        d, self.metric_fitter = self._metric("Fitter")
        e, self.metric_tilt_note = self._metric("Sensor tilt", tilt=True)
        f, self.metric_status = self._metric("Quality status")
        for frame, stretch in ((a, 1), (b, 1), (c, 1), (d, 2), (e, 2), (f, 2)):
            metrics.addWidget(frame, stretch)
        rlay.addLayout(metrics)
        # Retain all original preview/QC chips; the preview worker updates them.
        chips = QHBoxLayout()
        chips.setSpacing(5)
        self._chips: List[QLabel] = []
        for _ in range(6):
            chip = QLabel()
            chip.setVisible(False)
            chips.addWidget(chip)
            self._chips.append(chip)
        chips.addStretch(1)
        rlay.addLayout(chips)
        # Six original preview/analysis plots, the separate SpectrogramView,
        # and one new read-only spacing diagnostic are grouped in THREE tabs.
        self._canvases: List[PlotCanvas] = [None] * len(_PLOTS)
        self._tab_pages = [None] * len(_PLOTS)
        self.plot_tabs = QTabWidget()
        self.plot_tabs.setDocumentMode(True)
        self.plot_tabs.tabBar().setElideMode(Qt.ElideNone)     # never cut a tab name

        # Spectrogram: the interactive bar view over the whole tab (bar selection, dy(lambda)).
        spectro = QWidget()
        slay = QVBoxLayout(spectro)
        slay.setContentsMargins(4, 4, 4, 4)
        self.spectrogram = SpectrogramView(spectro)
        slay.addWidget(self.spectrogram)
        self.plot_tabs.addTab(spectro, "1  Spectrogram")

        # Target check: the preview over the full width; deviation and white below it, side by side.
        target = QWidget()
        tlay = QVBoxLayout(target)
        tlay.setContentsMargins(4, 4, 4, 4)
        target_split = QSplitter(Qt.Vertical)
        target_split.addWidget(self._panel(PREVIEW, minimum=260))
        lower = QSplitter(Qt.Horizontal)
        lower.addWidget(self._panel(DEVIATION, minimum=220))
        lower.addWidget(self._panel(WHITE, minimum=220))
        lower.setStretchFactor(0, 3)
        lower.setStretchFactor(1, 2)
        lower.setSizes([600, 400])
        target_split.addWidget(lower)
        target_split.setStretchFactor(0, 1)
        target_split.setStretchFactor(1, 1)
        tlay.addWidget(target_split)
        self.plot_tabs.addTab(target, "2  Target check")

        # Correction: original Before/After and Re-trace are retained; spacing added.
        correction = QWidget()
        clay = QHBoxLayout(correction)
        clay.setContentsMargins(4, 4, 4, 4)
        correction_split = QSplitter(Qt.Horizontal)
        correction_split.addWidget(self._panel(BEFORE_AFTER, minimum=400))
        spacing_panel = self._panel(SPACING, minimum=250)
        retrace_panel = self._panel(RETRACE, minimum=440)    # two stacked plots (bars, lamp lines)
        correction_split.addWidget(self._scroll_panels(spacing_panel, retrace_panel))
        correction_split.setStretchFactor(0, 3)
        correction_split.setStretchFactor(1, 2)
        correction_split.setSizes([560, 440])
        clay.addWidget(correction_split)
        self.plot_tabs.addTab(correction, "3  Correction result")

        # Geometry: displacement heatmap / original grid together, not separate tabs.
        geometry = QWidget()
        glay = QHBoxLayout(geometry)
        glay.setContentsMargins(4, 4, 4, 4)
        geometry_split = QSplitter(Qt.Horizontal)
        geometry_split.addWidget(self._panel(SHIFT_MAP, minimum=400))
        grid_panel = self._panel(GRID, minimum=310)
        coverage_panel = QFrame()
        coverage_panel.setObjectName("ksPlotPanel")
        cov_lay = QVBoxLayout(coverage_panel)
        self.coverage_note = QLabel("Median traced bar width: not available")
        self.coverage_value = QProgressBar()
        self.coverage_value.setRange(0, 100)
        self.coverage_value.setValue(0)
        self.coverage_value.setFormat("%p%")
        self.coverage_value.setToolTip("Median fraction of spectral width traced by each bar; "
                                       "not a measured 2D area fraction")
        cov_lay.addWidget(self.coverage_note)
        cov_lay.addWidget(self.coverage_value)
        tilt_note = QLabel("Detector tilt is an optical/image-coordinate estimate, not an inclinometer. "
                           "The pictogram above is exaggerated for readability.")
        tilt_note.setWordWrap(True)
        tilt_note.setStyleSheet(f"color:{palette['muted']}; padding:6px;")
        geometry_split.addWidget(self._scroll_panels(grid_panel, coverage_panel, tilt_note))
        geometry_split.setStretchFactor(0, 3)
        geometry_split.setStretchFactor(1, 2)
        geometry_split.setSizes([560, 440])
        glay.addWidget(geometry_split)
        self.plot_tabs.addTab(geometry, "4  Geometry analysis")
        self._group_pages = (spectro, target, correction, geometry)     # where an expanded plot panel stops
        self.plot_tabs.setCurrentIndex(TAB_SPECTROGRAM)
        for tab in _ANALYSIS_TABS:
            self.plot_tabs.setTabEnabled(tab, False)
        rlay.addWidget(self.plot_tabs, stretch=1)

        self.lbl_spacing_status = QLabel("Spacing variation: after analysis (diagnostic only)")
        self.lbl_spacing_status.setStyleSheet(f"color:{palette['muted']};")
        rlay.addWidget(self.lbl_spacing_status)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 1140])
        root.addWidget(split, stretch=1)
        self.lbl_ready = QLabel("TARGET INSPECTION  ·  Load / grab a target, then Analyze Keystone")
        self.lbl_ready.setObjectName("ksReady")
        self.lbl_ready.setWordWrap(True)
        root.addWidget(self.lbl_ready)
        self.lbl_state = QLabel()
        self.lbl_state.setWordWrap(True)
        self.lbl_state.setStyleSheet(f"color:{palette['muted']};")
        root.addWidget(self.lbl_state)
        self._spacing_stats = None
        self.report.setPlainText("Keystone: with the fore-lens ON, grab or load a line target (bars across the "
                                 "slit). The smile model stored for this bin (measured with the lens OFF and "
                                 "applied on the Smile tab) is used as it is; Analyze fits only the keystone, "
                                 "Apply writes ONE smile + keystone model.")

    def _toggle_report(self, on: bool) -> None:
        self.btn_report.setText("Technical report ▾" if on else "Technical report ▸")
        self.report.setVisible(on)

    def show_plot(self, index: int) -> None:
        """Navigate to the group containing an existing canvas index."""
        self.plot_tabs.setCurrentIndex(_GROUP.get(index, TAB_SPECTROGRAM))

    def _sync_analysis_tabs(self) -> None:
        """Correction / geometry tabs open only with an analysis result (a loaded model opens geometry)."""
        enabled = self.result is not None
        for tab in _ANALYSIS_TABS:
            self.plot_tabs.setTabEnabled(tab, enabled)
        if not enabled and self.plot_tabs.currentIndex() in _ANALYSIS_TABS:
            self.plot_tabs.setCurrentIndex(TAB_SPECTROGRAM)

    # ------------------------------------------------------------- context
    def set_calibration_source(self, store, current_setting: Optional[Callable[[], object]] = None) -> None:
        """Where stored smile models live and how to read the camera's current setting."""
        self._store = store
        self._current_setting = current_setting
        self._stored_cache.clear()
        self._refresh()

    def set_context(self, analysis, lamp_shape=None, lamp_setting=None, reason: str = "") -> None:
        """New lamp frame or new smile analysis of this session.

        A keystone fitted jointly with the previous analysis no longer applies and is dropped; a
        keystone fitted onto the stored smile is kept. Target / white frames are kept while they
        still match the lamp frame (size and, if known, camera setting); otherwise they are dropped.

        Args:
            analysis: Smile analysis of the lamp (``None`` before analysis).
            lamp_shape: ``(H, W)`` of the lamp frame (``None`` = no frame).
            lamp_setting: Its ``CameraSetting`` (``None`` = unknown).
            reason: Shown in the report when a result was dropped.
        """
        had_result = self.result is not None and self.result_mode == "session"
        self._smile = analysis
        self._lamp_shape = tuple(lamp_shape) if lamp_shape is not None else None
        self._lamp_setting = lamp_setting
        dropped = []
        for kind in ("target", "white"):
            frame = getattr(self, kind)
            if frame is None or self._lamp_shape is None:
                continue
            if tuple(frame.shape) != self._lamp_shape or _setting_problems(self._lamp_setting,
                                                                          getattr(self, kind + "_setting")):
                self._set_frame_attr(kind, None, None, "")
                dropped.append(kind)
        if dropped:
            self._frame_gen += 1                     # a keystone running on the dropped frames is discarded
            if self.result is not None:              # any keystone (stored mode too) was fitted on them
                had_result = True
                reason = reason or ("The keystone was fitted on frames that do not match the new lamp frame: "
                                    "it was dropped. Grab a new target and run Analyze Keystone.")
        if had_result or self.result is None:
            self.clear_result(reason if had_result else "")
        else:
            self._refresh()
        if dropped:
            self._start_preview()
            self.lbl_state.setText(self.lbl_state.text() + "  " + " and ".join(dropped).capitalize()
                                   + " frame dropped: it does not match the new lamp frame.")

    def clear_result(self, reason: str = "") -> None:
        """Forget the computed keystone (report and result plots return to the placeholder).

        Emits :attr:`keystoneCleared` when a result existed, so the owner detaches a joint-fit
        keystone from its corrector (the smile stays).
        """
        had_result = self.result is not None
        self.result = None
        self.result_mode = None
        self.result_corrector = None
        self.result_setting = None
        self.result_source = None
        self._applied = ""
        self._clear_plots("Run Analyze Keystone to see the check plots", (BEFORE_AFTER, SHIFT_MAP, RETRACE, GRID, SPACING))
        self._spacing_stats = None
        self._sync_analysis_tabs()
        if reason:
            self.report.setPlainText(reason)
        self._refresh(reason)
        if had_result:
            self.keystoneCleared.emit()

    def set_grab_enabled(self, enabled: bool) -> None:
        """Grab buttons follow the owner's grab availability / pending request."""
        self._grab_enabled = bool(enabled)
        busy = self._worker is not None             # same rule as _refresh: no grabs while analysing
        self.btn_grab_target.setEnabled(self._grab_enabled and not busy)
        self.btn_grab_white.setEnabled(self._grab_enabled and not busy)

    def set_actions(self, apply_enabled: bool, save_enabled: bool, apply_text: str = "") -> None:
        """The owner decides whether Apply / Save Matrix are possible for the current result."""
        self.btn_apply.setEnabled(bool(apply_enabled) and self.result is not None)
        self.btn_save_matrix.setEnabled(bool(save_enabled) and self.result is not None)
        if apply_text:
            self.btn_apply.setText(apply_text)
        self._refresh_apply_step()

    def mark_applied(self, path: str) -> None:
        """The owner stored the model: step 5 turns green."""
        self._applied = f"Applied {time.strftime('%H:%M')}: {os.path.basename(path)}"
        if self.result_mode == "stored":            # the file now holds this result's smile + keystone
            self.result_source = (str(path), _mtime(path))
        self._stored_cache.clear()
        self._refresh()

    def refresh_stored(self) -> None:
        """A model was written to the store (re-read it; a stored-mode result may now be stale)."""
        self._stored_cache.clear()
        self._refresh()

    def stored_result_problem(self) -> str:
        """Why the keystone fitted onto the stored smile must not be applied any more (``""`` = fine).

        Applying it writes the smile it was fitted onto; when that file was replaced meanwhile (for
        example a new smile applied on the Smile tab), Apply would silently restore the old smile.
        """
        if self.result is None or self.result_mode != "stored" or not self.result_source:
            return ""
        path, mtime = self.result_source
        name = os.path.basename(path)
        if _mtime(path) != mtime:
            return f"the stored smile model {name} changed after Analyze Keystone"
        if self._store is not None and self.result_setting is not None:
            try:
                now = self._store.find_smile(self.result_setting)
            except Exception:  # noqa: BLE001
                now = None
            if now is not None and os.path.normcase(os.path.abspath(str(now))) != os.path.normcase(
                    os.path.abspath(path)):
                return f"another smile model ({os.path.basename(str(now))}) is now stored for this bin"
        return ""

    # -------------------------------------------------------------- frames
    def _set_frame_attr(self, kind, frame, setting, source, dark_applied: Optional[bool] = None) -> None:
        setattr(self, kind, None if frame is None else np.asarray(frame, dtype=np.float32))
        setattr(self, kind + "_setting", setting if frame is not None else None)
        setattr(self, "_%s_source" % kind, source if frame is not None else "")
        setattr(self, "_%s_dark" % kind, dark_applied if frame is not None else None)

    _DARK_TIP = ("No dark subtracted: the host's dark correction was off or had no dark for this bin. "
                 "Load a dark in the main window and grab again.")

    def _dark_note(self, kind: str) -> str:
        """Short warning line for a grabbed frame the host did not dark-subtract ("" otherwise);
        the full advice is ``_DARK_TIP`` (the step's tooltip)."""
        if getattr(self, "_%s_dark" % kind, None) is not False:
            return ""
        return f"<br><span style='color:{_c('warn')}'>No dark subtracted (hover for details)</span>"

    def set_frame(self, kind: str, frame: Optional[np.ndarray], setting=None, source: str = "",
                  dark_applied: Optional[bool] = None) -> None:
        """Set the line-target (``kind="target"``) or white (``"white"``) frame; the bars are traced
        for the preview at once (worker thread).

        A keystone already computed was fitted on the previous frames, so it is dropped (Apply /
        Save are disabled and the owner detaches it from its corrector; the smile stays) until the
        next *Analyze Keystone*. A keystone still being computed is discarded when it arrives.
        """
        if kind not in ("target", "white"):
            raise ValueError(f"Unknown keystone frame kind {kind!r}")
        if frame is not None and np.asarray(frame).ndim != 2:
            raise ValueError("A keystone frame must be 2-D (rows, columns)")
        self._set_frame_attr(kind, frame, setting, source, dark_applied)
        self._frame_gen += 1
        self._failure = ""
        if self.result is not None:
            self.clear_result(f"{kind.capitalize()} frame changed after Analyze Keystone: the keystone fitted on "
                              "the previous frames was dropped. Run Analyze Keystone again.")
        if frame is not None:
            h, w = self.target.shape if kind == "target" else self.white.shape
            self._status(f"{kind.capitalize()} frame {w}x{h} ({source or 'array'})")
        if kind == "white":
            self._plot_white()
        self._start_preview()
        if frame is not None:
            if self.target is not None:
                self.plot_tabs.setCurrentIndex(TAB_SPECTROGRAM)
            else:
                self.show_plot(WHITE)
        self._refresh()

    def set_target(self, frame, setting=None, source: str = "") -> None:
        """Shortcut for :meth:`set_frame` ``("target", ...)``."""
        self.set_frame("target", frame, setting, source)

    def set_white(self, frame, setting=None, source: str = "") -> None:
        """Shortcut for :meth:`set_frame` ``("white", ...)``."""
        self.set_frame("white", frame, setting, source)

    def clear_white(self) -> None:
        """Analyse without a white frame (the illumination is flattened along the slit only)."""
        self.set_frame("white", None)
        self._status("White frame cleared")

    def load_files(self, kind: str, paths: List[str]) -> None:
        """Load (and average) image files as the target or white frame.

        A model file (``.npz``) saved with its frames gives its stored target (and, for
        ``kind="target"``, its white too), with the camera setting recorded in the model.

        Raises:
            ValueError: Files of different sizes, or a model file without that frame.
        """
        if any(str(p).lower().endswith(".npz") for p in paths):
            if len(paths) != 1:
                raise ValueError("Load one model file at a time (its frames are not averaged)")
            self.load_model_frames(paths[0], kind)
            return
        frames = [load_frame(p) for p in paths]
        if any(f.shape != frames[0].shape for f in frames):
            raise ValueError("All files must have the same size to be averaged")
        if len(frames) > 1:
            same, details = frames_same_scene(frames)
            if not same:
                bad = "; ".join("%s: shift %s rows, correlation %s" % (
                    os.path.basename(paths[d["index"]]),
                    "?" if d.get("shift_rows") is None else "%.1f" % d["shift_rows"],
                    "?" if d.get("correlation") is None else "%.2f" % d["correlation"]) for d in details)
                raise ValueError("These files do not show the same scene (the target moved, or another part or "
                                 "distance of it was imaged) and are not averaged: %s. Load one file, or several "
                                 "frames of a static target." % bad)
        frame = frames[0] if len(frames) == 1 else np.mean(np.stack(frames), axis=0).astype(np.float32)
        setting = None
        if self._setting_for_file is not None:
            try:
                setting = self._setting_for_file(frame.shape)
            except Exception:  # noqa: BLE001
                setting = None
        names = ", ".join(os.path.basename(p) for p in paths)
        source = names if len(paths) == 1 else f"mean of {len(paths)}: {names}"
        self.set_frame(kind, frame, setting, source)

    def load_model_frames(self, path: str, kind: str = "target") -> None:
        """Use the frames stored in a model file: the target and its white (``kind="target"``) or
        the white alone (``kind="white"``).

        Raises:
            ValueError: The file holds no such frame (saved before frames were stored).
        """
        from .calibration import load_frames
        frames, info, recorded = load_frames(path)
        if kind not in frames:
            raise ValueError(f"{os.path.basename(path)} holds no {kind} frame (saved without frames, or a "
                             "smile-only model)")
        setting = model_setting(recorded, frames[kind].shape, self._setting_for_file)
        name = os.path.basename(path)
        for k in ((kind,) if kind == "white" else ("target", "white")):
            if k in frames:
                original = (info.get(k) or {}).get("source") or k
                self.set_frame(k, frames[k], setting, f"{original} [from {name}]",
                               (info.get(k) or {}).get("dark_applied"))

    def _on_load(self, kind: str) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, f"Keystone {kind} frame(s)", "", _IMAGE_FILTER)
        if paths:
            try:
                self.load_files(kind, paths)
            except Exception as exc:  # noqa: BLE001
                self._status(f"Load {kind} failed: {exc}")

    def _on_load_target(self) -> None:
        self._on_load("target")

    def _on_load_white(self) -> None:
        self._on_load("white")

    # ------------------------------------------------------------- preview
    @property
    def preview_running(self) -> bool:
        """True while the bar preview worker is alive."""
        return self._preview_worker is not None

    def _polarity_changed(self) -> None:
        self._start_preview()
        self._refresh()

    def _start_preview(self) -> None:
        """Trace the current target's bars in a worker (one at a time; stale results are ignored)."""
        self._preview_token += 1
        self.preview = None
        self._preview_error = ""
        if self.target is None:
            self._clear_chips()
            self._clear_plots("Load or grab a line-target frame: its bars are drawn here at once"
                              + ("\n(the white frame is on the White tab)" if self.white is not None else ""),
                              (PREVIEW, DEVIATION))
            return
        if self._preview_worker is not None:
            self._preview_again = True              # restarted with the newest frames when it ends
            return
        self._launch_preview()

    def _launch_preview(self) -> None:
        white = None
        if self.white is not None and self.white.shape == self.target.shape:
            white = self.white.copy()
        worker = PreviewWorker(self._preview_token, self.target.copy(), white, self.polarity)
        worker.done.connect(self._on_preview_done)
        worker.failed.connect(self._on_preview_failed)
        worker.finished.connect(self._on_preview_finished)
        self._preview_worker = worker
        self._preview_again = False
        self._clear_plots("Detecting bars...", (PREVIEW, DEVIATION))
        self._set_chips([_chip("Detecting bars...", "info")])
        self._refresh()
        worker.start()

    def update_preview(self, synchronous: bool = False) -> None:
        """Re-trace the preview (``synchronous``: here, for scripts and tests)."""
        if not synchronous:
            self._start_preview()
            return
        self._preview_token += 1
        if self.target is None:
            self._start_preview()
            return
        white = self.white if self.white is not None and self.white.shape == self.target.shape else None
        self._on_preview_done((self._preview_token, target_preview(self.target, white, self.polarity)))

    def wait_preview(self, timeout_s: float = 60.0) -> bool:
        """Process events until no preview worker runs (scripts / tests). Returns ``True`` when idle."""
        from PyQt5.QtWidgets import QApplication
        deadline = time.time() + timeout_s
        while self._preview_worker is not None and time.time() < deadline:
            QApplication.processEvents()
            time.sleep(0.005)
        QApplication.processEvents()
        return self._preview_worker is None

    def _wavelength_fn(self):
        """``column -> nm`` of the wavelength calibration stored for the target's setting, or ``None``."""
        setting = self._lookup_setting()
        if self._store is None or setting is None:
            return None
        try:
            stored = self._store.load_wavelength(setting)
        except Exception:  # noqa: BLE001 - a missing / odd profile only drops the nm labels
            return None
        if stored is None or not getattr(stored, "ok", False):
            return None
        m, c = float(stored.profile.m), float(stored.profile.c)
        return lambda x: m * float(x) + c

    def _on_preview_done(self, payload) -> None:
        token, preview = payload
        if token != self._preview_token:
            return                                   # a newer frame arrived meanwhile
        wavelength = self._wavelength_fn()
        if wavelength is not None and preview.get("span_coverage"):
            preview["span_warnings"] = coverage_warnings(preview["span_coverage"], wavelength)
        self.preview = preview
        self._preview_error = preview.get("error", "")
        self._plot_preview(preview)
        self._plot_deviation(preview)
        if preview.get("bars"):
            self.spectrogram.set_data(preview, wavelength)
        else:
            self.spectrogram.clear("No bars to show: " + (preview.get("error") or "none could be followed"))
        self._update_chips(preview)
        self._refresh()

    def _on_preview_failed(self, payload) -> None:
        token, message = payload
        if token != self._preview_token:
            return
        self.preview = None
        self._preview_error = message
        self._clear_plots("Bar detection failed: " + message, (PREVIEW, DEVIATION))
        self._set_chips([_chip("Bar detection failed", _BAD, message)])
        self._refresh()

    def _on_preview_finished(self) -> None:
        if self._preview_worker is not None:
            self._preview_worker.wait()
            self._preview_worker.deleteLater()
        self._preview_worker = None
        if self._preview_again and self.target is not None:
            self._launch_preview()
        else:
            self._preview_again = False
            self._refresh()

    # ----------------------------------------------------------- smile source
    def _camera_now(self):
        try:
            return self._current_setting() if self._current_setting else None
        except Exception:  # noqa: BLE001
            return None

    def _lookup_setting(self):
        """Camera setting whose stored smile applies: the target's, else the camera's current one."""
        if self.target is not None:
            if self.target_setting is not None:
                return self.target_setting
            now = self._camera_now()
            return now.with_frame(self.target.shape) if now is not None else None
        return self._camera_now()

    def _load_stored(self, path):
        from .corrector import SmileCorrector
        key = str(path)
        try:
            mtime = os.path.getmtime(key)
        except OSError:
            mtime = None
        cached = self._stored_cache.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        corrector = SmileCorrector.load(key)
        self._stored_cache[key] = (mtime, corrector)
        return corrector

    def _update_stored(self) -> None:
        """Find, load and check the stored smile model for :meth:`_lookup_setting`."""
        self.stored = None
        setting = self._lookup_setting()
        if self._store is None or setting is None:
            return
        info = dict(path=None, corrector=None, problems=[], setting=setting, fallback=False)
        self.stored = info
        try:
            path = self._store.find_smile(setting)
        except Exception as exc:  # noqa: BLE001
            info["problems"] = [f"store unreadable: {exc}"]
            return
        if path is None:
            return
        info["path"] = str(path)
        try:
            info["fallback"] = bool(self._store.is_fallback(path))
        except Exception:  # noqa: BLE001
            pass
        try:
            corrector = self._load_stored(path)
        except Exception as exc:  # noqa: BLE001
            info["problems"] = [_ascii(f"unreadable ({exc})")]
            return
        info["corrector"] = corrector
        cal = getattr(corrector, "_source_cal", None)
        problems = []
        saved = getattr(cal, "camera_setting", None)
        if saved:
            try:
                from modules.contracts import CameraSetting
                problems += [_ascii(p) for p in setting.mismatch(CameraSetting.from_dict(saved))]
            except Exception as exc:  # noqa: BLE001
                problems.append(_ascii(f"camera setting in the file unreadable ({exc})"))
        shape = tuple(int(v) for v in getattr(corrector, "_image_shape", ()) or ())
        frame_shape = tuple(self.target.shape) if self.target is not None else tuple(setting.frame_shape or ())
        if shape and frame_shape and shape != frame_shape and not any(p.startswith("frame") for p in problems):
            problems.append(f"model frame {shape[1]}x{shape[0]} != target {frame_shape[1]}x{frame_shape[0]}")
        info["problems"] = problems

    @property
    def stored_ok(self) -> bool:
        """A stored smile model usable for the current target exists."""
        return bool(self.stored and self.stored.get("corrector") is not None and not self.stored["problems"])

    def smile_mode(self) -> Optional[str]:
        """``"stored"`` (default when a valid stored model exists), ``"session"`` or ``None``."""
        session = self._smile is not None
        if self._smile_choice == "session" and session:
            return "session"
        if self.stored_ok:
            return "stored"
        return "session" if session else None

    def set_smile_source(self, source: Optional[str]) -> None:
        """Pick the smile: ``"stored"``, ``"session"`` or ``None`` (automatic: stored when it exists)."""
        if source not in (None, "stored", "session"):
            raise ValueError(f"Unknown smile source {source!r}")
        self._smile_choice = source
        self._refresh()

    def _on_smile_choice(self, index: int) -> None:
        self.set_smile_source(self.combo_smile.itemData(index))

    # ------------------------------------------------------------ analysis
    @property
    def running(self) -> bool:
        """True while a keystone worker is alive."""
        return self._worker is not None

    @property
    def polarity(self) -> str:
        """``"dark"`` or ``"bright"`` bars."""
        return self.combo_polarity.currentData()

    def set_polarity(self, polarity: str) -> None:
        """Select ``"dark"`` or ``"bright"`` bars."""
        index = self.combo_polarity.findData(polarity)
        if index < 0:
            raise ValueError(f"Unknown polarity {polarity!r}")
        self.combo_polarity.setCurrentIndex(index)

    def _no_smile_text(self) -> str:
        setting = self._lookup_setting()
        bin_text = f"bin {setting.camera_bin}" if setting is not None else "this camera setting"
        if self.stored and self.stored.get("path") and self.stored["problems"]:
            return (f"Stored smile model {os.path.basename(self.stored['path'])} does not fit this target ("
                    + "; ".join(self.stored["problems"]) + "): measure the smile for this setting (lens off) "
                    "and Apply it, or analyse a lamp frame on the Smile tab.")
        if self._store is None or setting is None:
            return ("No smile model: analyse a lamp frame on the Smile tab (or connect the camera so the stored "
                    "model of its bin is used).")
        return (f"No smile model for {bin_text}: measure the smile (lens off) on the Smile tab and Apply it first.")

    def check(self) -> Tuple[bool, str]:
        """Whether *Analyze Keystone* can run now, and why not."""
        if self._worker is not None:
            return False, "Keystone analysis running..."
        self._update_stored()
        mode = self.smile_mode()
        if self.target is None:
            if mode is None:
                return False, self._no_smile_text()
            return False, "Load or grab a line-target frame (bars across the slit)."
        if mode is None:
            return False, self._no_smile_text()
        if mode == "session":
            lamp_shape = tuple(self._smile.frame_orig.shape)
            for kind, frame, setting in (("Target", self.target, self.target_setting),
                                         ("White", self.white, self.white_setting)):
                if frame is None:
                    continue
                if tuple(frame.shape) != lamp_shape:
                    return False, (f"{kind} frame {frame.shape[1]}x{frame.shape[0]} differs from the lamp frame "
                                   f"{lamp_shape[1]}x{lamp_shape[0]}: use the same camera setting.")
                problems = _setting_problems(self._lamp_setting, setting)
                if problems:
                    return False, (f"{kind} frame was taken at another camera setting than the lamp: "
                                   + "; ".join(problems))
            return True, ""
        if self.white is not None:
            if self.white.shape != self.target.shape:
                return False, (f"White frame {self.white.shape[1]}x{self.white.shape[0]} differs from the target "
                               f"{self.target.shape[1]}x{self.target.shape[0]}: use the same camera setting.")
            problems = _setting_problems(self.target_setting, self.white_setting)
            if problems:
                return False, "White frame was taken at another camera setting than the target: " + "; ".join(problems)
        return True, ""

    def _check_lamp(self):
        """Session lamp frame usable to CHECK a stored smile (same size and setting as the target)."""
        if self._smile is None or self.target is None:
            return None
        lamp = getattr(self._smile, "frame_orig", None)
        if lamp is None or tuple(lamp.shape) != tuple(self.target.shape):
            return None
        if _setting_problems(self._lamp_setting, self.target_setting):
            return None
        return np.asarray(lamp, np.float32).copy()

    def analyze(self, synchronous: bool = False) -> bool:
        """Start the keystone analysis (worker thread, or here when ``synchronous``).

        Returns:
            ``False`` when it cannot run (reason in the state line / status).
        """
        ok, why = self.check()
        if not ok:
            self._status("Keystone not analysed: " + why)
            return False
        mode = self.smile_mode()
        inputs = dict(target=self._target_source or "array", polarity=self.polarity)
        if self.white is not None:
            inputs["white"] = self._white_source or "array"
        kwargs = dict(white=None if self.white is None else self.white.copy(), polarity=self.polarity, inputs=inputs)
        wavelength = self._wavelength_fn()
        if wavelength is not None:
            kwargs["wavelength"] = wavelength
        self._pending_setting = self.target_setting
        self._pending_gen = self._frame_gen
        self._failure = ""
        self._pending_mode = mode
        stored_path = None
        analysis = self._smile if mode == "session" else None
        self._pending_source = None
        if mode == "stored":
            stored_path = self.stored["path"]
            self._pending_source = (str(stored_path), _mtime(stored_path))
            inputs["smile"] = "stored " + os.path.basename(stored_path)
            lamp = self._check_lamp()
            if lamp is not None:
                kwargs["lamp"] = lamp
                inputs["lamp_check"] = "session lamp frame"
        if synchronous:
            try:
                if mode == "stored":
                    from .corrector import SmileCorrector
                    corrector = SmileCorrector.load(stored_path)
                    payload = ("stored", corrector, corrector.fit_keystone(self.target.copy(), **kwargs))
                else:
                    payload = ("session", analysis, analyze_keystone(analysis, self.target.copy(), **kwargs))
            except Exception as exc:  # noqa: BLE001
                self._on_failed(str(exc))
                return False
            self._on_done(payload)
            return True
        worker = KeystoneWorker(analysis, self.target.copy(), kwargs, stored_path=stored_path)
        worker.done.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        self.busy.setVisible(True)
        self._status("Analyzing keystone (bar tracing + fit)..." if mode == "stored"
                     else "Analyzing keystone (bar tracing + joint fit)...")
        self._refresh()
        worker.start()
        return True

    def _on_done(self, payload) -> None:
        mode, source, k = payload
        if mode == "session" and source is not self._smile:
            self._status("Smile analysis changed while the keystone was computed: run Analyze Keystone again")
            return
        if self._pending_gen != self._frame_gen:
            message = ("Target or white frame changed while the keystone was computed: result discarded, "
                       "run Analyze Keystone again")
            self.report.setPlainText(message)          # stays after the preview of the new frame refreshes
            self._status(message)
            return
        self.result = k
        self.result_mode = mode
        self.result_corrector = source if mode == "stored" else None
        self.result_setting = self._pending_setting
        self.result_source = self._pending_source if mode == "stored" else None
        self._applied = ""
        self.show_report(keystone_report(k), k.warnings)
        self._plot_target(k)
        self._plot_shift_map(k.map_y, k.model.dy.box, k.row_margin, k.margin)
        self._plot_retrace(k.qc)
        self._plot_grid(k)
        self._plot_spacing(k)
        self._sync_analysis_tabs()
        self.show_plot(BEFORE_AFTER)
        s = k.summary
        self._status(f"Keystone: max {s['keystone_max_px']:.2f} px, tilt {s['tilt_deg']:.3f} deg, "
                     f"{len(k.warnings)} warning(s)")
        self._refresh()
        self.keystoneReady.emit(k)

    def _on_failed(self, message: str) -> None:
        self._failure = message                  # kept until the next frame / Analyze (see _refresh)
        self.report.setPlainText("Keystone analysis failed:\n" + message)
        self._status("Keystone analysis failed: " + message)
        self.step_analyze.set_state(_BAD, html.escape("Failed: " + message))

    def _on_finished(self) -> None:
        if self._worker is not None:
            self._worker.wait()
            self._worker.deleteLater()
        self._worker = None
        self.busy.setVisible(False)
        # keep the worker's outcome (result / failure / discarded) on the state line, not "Ready"
        self._refresh(self.lbl_state.text())

    def wait_worker(self) -> None:
        """Block until running keystone / preview workers have finished (owner's ``closeEvent``)."""
        for worker in (self._worker, self._preview_worker):
            if worker is not None and worker.isRunning():
                worker.wait()

    def closeEvent(self, event):  # noqa: N802
        """Never destroy the page under a running worker."""
        self.wait_worker()
        super().closeEvent(event)

    # ------------------------------------------------------------- display
    def show_report(self, text: str, warnings: Optional[List[str]] = None) -> None:
        """Show a keystone report; ``Warning:`` lines are highlighted."""
        rows = []
        for line in text.splitlines():
            escaped = html.escape(line)
            if line.startswith("Warning:"):
                rows.append(f"<span style='color:{_c('bad')};font-weight:bold'>{escaped}</span>")
            else:
                rows.append(escaped)
        if warnings is not None and not warnings:
            rows.append(f"<span style='color:{_c('good')}'>No warnings.</span>")
        self.report.setHtml("<pre style='margin:0; white-space:pre-wrap'>" + "<br>".join(rows) + "</pre>")

    def show_stored(self, keystone: Optional[dict], map_y: Optional[np.ndarray] = None, source: str = "") -> None:
        """Show the keystone stored in a loaded model file (``None``: the file is smile only)."""
        self.result = None
        self.result_mode = None
        self.result_corrector = None
        self.result_setting = None
        self.result_source = None
        self._applied = ""
        self._clear_plots("Loaded model: target and spacing plots need Analyze Keystone", (BEFORE_AFTER, SHIFT_MAP,
                                                                                    RETRACE, GRID, SPACING))
        self._spacing_stats = None
        self._sync_analysis_tabs()
        if not keystone:
            self.report.setPlainText(f"Loaded model {source}: smile only (no keystone).")
            self._refresh()
            return
        try:
            text = keystone_report(keystone)
        except Exception as exc:  # noqa: BLE001
            text = f"Stored keystone report unreadable: {exc}"
        self.show_report(f"Loaded model {source}\n" + text, keystone.get("warnings"))
        model = keystone.get("model") or {}
        dy = (model.get("dy") or {}).get("box")
        if map_y is not None:
            self.plot_tabs.setTabEnabled(TAB_GEOMETRY, True)
            self._plot_shift_map(np.asarray(map_y), dy, keystone.get("row_margin") or (0, 0),
                                 keystone.get("margin") or (0, 0))
        self._plot_retrace(keystone.get("qc") or {})
        self._refresh("Showing the keystone of the loaded model.")

    def _clear_plots(self, message: str, which=None) -> None:
        spectrogram = getattr(self, "spectrogram", None)
        if spectrogram is not None and (which is None or PREVIEW in which):
            spectrogram.clear(message)
        for index in (range(len(self._canvases)) if which is None else which):
            canvas = self._canvases[index]
            canvas.clear()
            ax = canvas.fig.add_subplot(111)
            ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes, color="#5d6d7e",
                    fontsize=10, wrap=True)
            ax.set_axis_off()
            _draw(canvas)

    # ----------------------------------------------------------- chips
    def _clear_chips(self) -> None:
        self._set_chips([])

    def _set_chips(self, chips) -> None:
        for label, spec in zip(self._chips, list(chips) + [None] * (len(self._chips) - len(chips))):
            if spec is None:
                label.setVisible(False)
                continue
            text, style, tip = spec
            label.setText(text)
            label.setStyleSheet(style)
            label.setToolTip(tip)
            label.setVisible(True)

    def _update_chips(self, p: dict) -> None:
        bars = p["bars"]
        info = p.get("info") or {}
        rejected = len(info.get("rejected") or [])
        n = len(bars)
        chips = [_chip(f"Bars: {n} found" + (f", {rejected} rejected" if rejected else ""),
                       _GOOD if n >= 5 else (_WARN if n >= 3 else _BAD),
                       p.get("error") or "Bars followed across the spectrum")]
        cov = p.get("coverage", float("nan"))
        if np.isfinite(cov):
            chips.append(_chip(f"Coverage {100 * cov:.0f} %", _GOOD if cov >= .7 else (_WARN if cov >= .5 else _BAD),
                               "Median share of the frame width each bar was followed over"))
        warns = p.get("span_warnings") or []
        if p.get("span_coverage"):
            zones_short = [w_ for w_ in warns if w_.startswith("Columns")]
            ends = [w_ for w_ in warns if w_.startswith("The lit slit")]
            if zones_short:
                chips.append(_chip("Slit ends missing in part of the spectrum", _WARN, "\n".join(zones_short)))
            else:
                chips.append(_chip("Bars span the slit at every wavelength", _GOOD,
                                   "In every spectral zone the outermost bars were followed"))
            if ends:
                chips.append(_chip("No bars near the slit ends", _WARN, "\n".join(ends)))
        sat = p.get("saturated_pct", 0.0)
        chips.append(_chip(f"Saturated {sat:.2f} %", _GOOD if sat < .1 else (_WARN if sat < 1 else _BAD),
                           "Pixels at the sensor full scale (lower the exposure if not ~0)"))
        con = p.get("contrast", float("nan"))
        if np.isfinite(con):
            chips.append(_chip(f"Bar contrast {100 * con:.0f} %", _GOOD if con >= .2 else (_WARN if con >= .08 else _BAD),
                               "Median bar depth against the background at the brightest column"))
        pol = "dark bars" if p["polarity"] == "dark" else "bright bars"
        auto = p.get("polarity_auto")
        how = ""
        if auto:
            df = auto.get("dark_fraction")
            how = " (auto" + ("" if df is None else f", dark area {100 * df:.0f} %") + ")"
        alt = p.get("alt")
        if alt:
            chips.append(_chip(f"Polarity: {pol}{how} - try {alt[0]} ({alt[1]} bars)", _WARN,
                               "Few bars with this polarity: set 'Bars:' in step 2"))
        else:
            chips.append(_chip(f"Polarity: {pol}{how}", _GOOD if n >= 5 else "info",
                               "Auto: the bars are the narrower features of the target" if auto else ""))
        if p.get("white"):
            chips.append(_chip(f"White: yes (lit {p.get('lit_pct', 0):.0f} %)", _GOOD))
        else:
            chips.append(_chip("White: no (optional)", "info", "A white frame divides out the illumination"))
        self._set_chips(chips)

    # ----------------------------------------------------------- refresh
    def _status(self, message: str) -> None:
        self.lbl_state.setText(message)
        self.statusChanged.emit(message)

    def _describe(self, kind: str) -> str:
        frame = getattr(self, kind)
        if frame is None:
            return f"{kind.capitalize()}: none"
        h, w = frame.shape
        return f"{kind.capitalize()}: {w}x{h} ({getattr(self, '_%s_source' % kind) or 'array'})"

    def _smile_text(self) -> Tuple[str, str, str]:
        """(state, detail, hint) of step 1."""
        mode = self.smile_mode()
        th_stored = "Spectral smile stored for this bin (measured with the fore-optics removed); no lamp frame needed"
        th_none = "Measure the spectral smile with the fore-optics removed and Apply it on the Smile tab first"
        th_session = "Calibration-lamp frame of this session: smile and keystone fitted jointly"
        if mode == "stored":
            corrector = self.stored["corrector"]
            cal = getattr(corrector, "_source_cal", None)
            setting = self.stored["setting"]
            parts = [f"<b>Stored</b>, bin {setting.camera_bin}"]
            if cal is not None:
                marker = "stored smile model of "          # a keystone was applied onto it later
                notes = cal.notes or ""
                parts.append(_short_date(notes.split(marker, 1)[1].strip() if marker in notes else cal.created))
                if cal.n_lines_used:
                    parts.append(f"{cal.n_lines_used} lines")
                rms = cal.rms_after if np.isfinite(_num(cal.rms_after)) else cal.fit_rms
                if np.isfinite(_num(rms)):
                    parts.append(f"RMS {float(rms):.2f} px")
            name = os.path.basename(self.stored["path"])
            text = " · ".join(parts) + f"<br><span style='color:{_c('muted')}'>{html.escape(name)}"
            if cal is not None and cal.map_y is not None:
                text += f" (has a keystone of {html.escape(_short_date(cal.created))}: it is re-fitted)"
            if self.stored.get("fallback"):
                text += " - older file of unknown camera"
            text += "</span>"
            return _GOOD, text, th_stored
        if mode == "session":
            r = self._smile
            text = (f"<b>This session's lamp analysis</b> (joint fit): {len(r.used_lines)} lines, "
                    f"RMS {r.rms_after:.2f} px")
            if self.stored and self.stored.get("path") and self.stored["problems"]:
                text += (f"<br><span style='color:{_c('bad')}'>Stored {html.escape(os.path.basename(self.stored['path']))}"
                         f" refused: {html.escape('; '.join(self.stored['problems']))}</span>")
            elif self.stored_ok:
                text += f"<br><span style='color:{_c('muted')}'>stored model available (choose above)</span>"
            return _GOOD, text, th_session
        return _BAD, html.escape(self._no_smile_text()), th_none

    def _refresh_apply_step(self) -> None:
        stale = self.stored_result_problem()
        if stale:
            self.step_apply.set_state(_WARN, html.escape("Not applicable any more: " + stale
                                                         + ". Run Analyze Keystone again."))
        elif self._applied:
            self.step_apply.set_state(_GOOD, html.escape(self._applied))
        elif self.result is None:
            self.step_apply.set_state(_TODO, "After Analyze.")
        elif self.btn_apply.isEnabled():
            setting = self.result_setting
            where = f" for bin {setting.camera_bin}" if setting is not None else ""
            self.step_apply.set_state(_TODO, f"Ready{where}",
                                      tip="Stores the smile + keystone model; the previous file is kept in history/.")
        else:
            self.step_apply.set_state(_WARN, "Apply needs a known camera setting (grab the target from the "
                                             "camera) - or use Save Matrix.")

    def _refresh(self, message: str = "") -> None:
        """Enable buttons and update the steps / state line."""
        ok, why = self.check()
        mode = self.smile_mode()
        self.btn_analyze.setEnabled(ok)
        self.btn_clear_white.setEnabled(self.white is not None)
        busy = self._worker is not None
        for b in (self.btn_load_target, self.btn_load_white):
            b.setEnabled(not busy)
        self.btn_grab_target.setEnabled(self._grab_enabled and not busy)
        self.btn_grab_white.setEnabled(self._grab_enabled and not busy)
        self.combo_polarity.setEnabled(not busy)
        # step 1: smile source
        self.combo_smile.blockSignals(True)
        model = self.combo_smile.model()
        model.item(0).setEnabled(self.stored_ok)
        model.item(1).setEnabled(self._smile is not None)
        self.combo_smile.setCurrentIndex(1 if mode == "session" else 0)
        self.combo_smile.blockSignals(False)
        self.combo_smile.setVisible(self._smile is not None)
        self.combo_smile.setEnabled(not busy and self._smile is not None and self.stored_ok)
        state, detail, hint = self._smile_text()
        self.step_smile.set_state(state, detail, hint)
        # step 2: target
        muted = lambda s: f"<span style='color:{_c('muted')}'>{s}</span>"
        if self.target is None:
            self.step_target.set_state(_TODO, "None yet.")
        else:
            h, w = self.target.shape
            src = html.escape(self._target_source or "array")
            source = muted(f"{w}x{h} &middot; {src}")
            dark = self._dark_note("target")
            if self._preview_worker is not None:
                self.step_target.set_state(_BUSY, f"Detecting bars...<br>{source}")
            elif self.preview is not None:
                p = self.preview
                n = len(p["bars"])
                cov = p.get("coverage", float("nan"))
                extra = f" &middot; coverage {100 * cov:.0f} %" if np.isfinite(cov) else ""
                sat = p.get("saturated_pct", 0)
                extra += f" &middot; saturated {sat:.2f} %" if sat > 0 else ""
                state = _GOOD if n >= 5 else (_WARN if n >= 3 else _BAD)
                text = f"<b>{n} bars found</b>{extra}<br>{source}"
                tips = []
                if p.get("error"):
                    text += f"<br><span style='color:{_c('bad')}'>{html.escape(p['error'])}</span>"
                if p.get("alt"):
                    text += (f"<br><span style='color:{_c('warn')}'>Try 'Bars: {p['alt'][0]}' "
                             f"({p['alt'][1]} bars found that way)</span>")
                span_warns = p.get("span_warnings") or []
                if span_warns:                           # long advice: one line here, the text as tooltip
                    state = _WARN if state == _GOOD else state
                    tips += span_warns
                    text += (f"<br><span style='color:{_c('warn')}'>Bars missing near the slit ends in part "
                             f"of the spectrum (hover for details)</span>")
                if dark:
                    state = _WARN if state == _GOOD else state
                    tips.append(self._DARK_TIP)
                self.step_target.set_state(state, text + dark, tip="\n\n".join(tips))
            elif self._preview_error:
                self.step_target.set_state(_BAD, html.escape(self._preview_error) + f"<br>{source}")
            else:
                self.step_target.set_state(_WARN if dark else _GOOD, source + dark,
                                           tip=self._DARK_TIP if dark else "")
        # step 3: white
        if self.white is None:
            self.step_white.set_state(_OPTIONAL, "None (optional, recommended).")
        else:
            h, w = self.white.shape
            lit = ""
            if self.preview is not None and np.isfinite(self.preview.get("lit_pct", float("nan"))):
                lit = f"lit {self.preview['lit_pct']:.0f} % of the frame<br>"
            mismatch = self.target is not None and self.white.shape != self.target.shape
            dark = self._dark_note("white")
            self.step_white.set_state(_WARN if mismatch or dark else _GOOD,
                                      lit + muted(f"{w}x{h} &middot; {html.escape(self._white_source or 'array')}")
                                      + (f"<br><span style='color:{_c('warn')}'>size differs from the target</span>"
                                         if mismatch else "") + dark, tip=self._DARK_TIP if dark else "")
        # step 4: analyze
        if busy:
            self.step_analyze.set_state(_BUSY, "Running" + (" (keystone onto the stored smile)..."
                                                           if self._pending_mode == "stored" else " (joint fit)..."))
        elif self.result is not None:
            s = self.result.summary
            onto = "stored smile" if self.result_mode == "stored" else "joint fit"
            n_warn = len(self.result.warnings)
            real = [x for x in self.result.warnings if not x.startswith("Smile taken from the stored model")]
            self.step_analyze.set_state(
                _GOOD if not real or (len(real) == 1 and real[0].startswith("No white frame")) else _WARN,
                f"Done ({onto}): keystone max <b>{s['keystone_max_px']:.2f} px</b>, tilt {s['tilt_deg']:.3f}&deg;",
                tip="%d warning(s); the quality summary, banner and technical report give them" % n_warn)
        elif self._failure and ok:
            self.step_analyze.set_state(_BAD, html.escape("Failed: " + self._failure))
        elif ok:
            self.step_analyze.set_state(_TODO, "Ready" + (" (keystone onto the stored smile)" if mode == "stored"
                                                         else " (smile + keystone joint fit)"))
        else:
            self.step_analyze.set_state(_TODO, html.escape(why))
        if self.result is None:
            self.btn_apply.setEnabled(False)
            self.btn_save_matrix.setEnabled(False)
        self._refresh_apply_step()
        frames = self._describe("target") + "   |   " + self._describe("white")
        if self.white is None:
            frames += " (optional, recommended)"
        self.lbl_frames.setText(frames)
        if message:
            self.lbl_state.setText(message)
        elif self.result is not None:
            pass                                        # keep the result status
        elif not ok:
            self.lbl_state.setText(why)
        elif self._failure:
            self.lbl_state.setText("Keystone analysis failed: " + self._failure)
        else:
            self.lbl_state.setText("Ready: Analyze Keystone")
        self._update_dashboard()

    # ------------------------------------------------------------- plots
    @staticmethod
    def _bar_colours(n: int):
        import matplotlib
        cmap = matplotlib.colormaps["turbo"]
        return [cmap(0.08 + 0.84 * i / max(1, n - 1)) for i in range(n)]

    @staticmethod
    def _shade_missing(ax, cov: dict, w: int) -> None:
        """Hatch where no bars were followed: per spectral zone the rows beyond its outermost bars
        (red: the keystone there is extrapolated along the slit) and the lit slit ends (orange)."""
        from matplotlib.patches import Rectangle
        full, lit = cov.get("bar_rows"), cov.get("lit_rows")
        if not full:
            return
        shown = False
        for z in cov.get("zones", []):
            if z["share"] >= ZONE_SPAN_MIN:
                continue
            spans = [(full[0], full[1])] if not z["rows"] else [(full[0], z["rows"][0]), (z["rows"][1], full[1])]
            for y0, y1 in spans:
                if y1 - y0 < 1:
                    continue
                ax.add_patch(Rectangle((z["c0"] - .5, y0), z["c1"] - z["c0"] + 1, y1 - y0, facecolor=(1, .3, .25, .22),
                                       edgecolor=(1, .3, .25, .95), hatch="///", lw=1.0, zorder=4))
                shown = True
        if lit:
            c0, c1 = cov.get("lit_columns") or (0, w - 1)
            for y0, y1 in ((lit[0], full[0]), (full[1], lit[1])):
                if y1 - y0 > 2:
                    ax.add_patch(Rectangle((c0 - .5, y0), c1 - c0 + 1, y1 - y0, facecolor=(1, .62, .1, .10),
                                           edgecolor=(1, .62, .1, .7), hatch="..", lw=.6, zorder=3))
                    shown = True
        if shown:
            ax.text(0.01, 0.995, "hatched: no bars measured there - the keystone would be extrapolated",
                    transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=(1, .45, .35),
                    bbox=dict(boxstyle="round,pad=0.25", fc=(0, 0, 0, .55), ec="none"), zorder=6)

    def _plot_preview(self, p: dict) -> None:
        """Normalised target with every traced bar (curve) and its straight reference (dashed)."""
        canvas = self._canvases[PREVIEW]
        canvas.clear()
        fig = canvas.fig
        norm = np.asarray(p["norm"], np.float32)
        h, w = norm.shape
        lit = norm > 0
        vmin, vmax = _stretch(norm, lit if lit.any() else None)
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.04)
        ax = fig.add_subplot(gs[0])
        ax.imshow(norm, cmap="gray", aspect="auto", vmin=vmin - 0.35 * (vmax - vmin), vmax=vmax,
                  extent=(-.5, w - .5, h - .5, -.5), interpolation="nearest")
        bars = p["bars"]
        colours = self._bar_colours(len(bars))
        x_ref = p["x_ref"]
        for b, colour in zip(bars, colours):
            xs = np.linspace(float(b.x.min()), float(b.x.max()), 160)
            ys = b.y_at(xs, w)
            y0 = float(b.y_at(np.array([np.clip(x_ref, b.x.min(), b.x.max())]), w)[0])
            ax.plot([xs[0], xs[-1]], [y0, y0], ls=(0, (4, 3)), lw=.8, color="#1b2631", alpha=.85)
            ax.plot(xs, ys, lw=1.1, color=colour, alpha=.95)
        if bars:
            ax.axvline(x_ref, color="#f4d03f", lw=.8, ls=":", alpha=.9)
        info = p.get("info") or {}
        for row, why in info.get("rejected") or []:
            ax.plot([info.get("seed_column", x_ref)], [row], marker="x", ms=7, mew=1.6, color="#808b96")
            ax.annotate("rejected", (info.get("seed_column", x_ref), row), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=7, color="#5d6d7e",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.75))
        for group in info.get("merged") or []:
            centre = 0.5 * (group[0] + group[-1])
            near = [b for b in bars if abs(b.y_seed - centre) < 3]
            if near:
                b = near[0]
                ax.annotate("merged pair", (float(b.x.min()), float(b.y_at(np.array([b.x.min()]), w)[0])),
                            xytext=(3, -7), textcoords="offset points", ha="left", va="bottom", fontsize=7,
                            color="#5d6d7e",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.75))
        self._shade_missing(ax, p.get("span_coverage") or {}, w)
        ax.set_xlim(-.5, w - .5)
        ax.set_ylim(h - .5, -.5)
        ax.set_xlabel("Column (spectral)")
        ax.set_ylabel("Row (slit)")
        if bars:
            ax.set_title(f"Line target: {len(bars)} bars traced  -  curves = measured bars, dashed = straight "
                         f"(at column {x_ref:.0f}); the gap is the keystone / tilt", fontsize=9)
        else:
            alt = p.get("alt")
            ax.set_title("Line target: no bars could be followed (reason in step 2)"
                         + (f" - try 'Bars: {alt[0]}'" if alt else ""), fontsize=9, color="#a93226")
        # (c) keystone fan
        axf = fig.add_subplot(gs[1], sharey=ax)
        fan = p.get("fan") or {}
        if len(fan.get("rows", [])):
            rows, values = fan["rows"], fan["fan"]
            order = np.argsort(rows)
            cols = self._bar_colours(len(bars))
            row_of = [float(b.y_at(np.array([0.5 * (max(b.x.min(), fan['xl']) + min(b.x.max(), fan['xr']))]), w)[0])
                      for b in bars]
            col_for = {round(r, 3): c for r, c in zip(row_of, cols)}
            axf.axvline(0, color="#95a5a6", lw=.8)
            axf.plot(values[order], rows[order], color="#aab7b8", lw=.8)
            axf.scatter(values, rows, s=14, c=[col_for.get(round(r, 3), "#34495e") for r in rows], zorder=3,
                        edgecolors="#1b2631", linewidths=.3)
            if "coef" in fan:
                rr = np.linspace(fan["top_row"], fan["bottom_row"], 20)
                axf.plot(np.polynomial.polynomial.polyval(rr - (h - 1) / 2.0, fan["coef"]), rr, ls="--", lw=1,
                         color="#c0392b")
                axf.text(0.03, 0.01, f"tilt {fan['tilt_deg']:+.2f} deg\nfan: top {fan['top']:+.1f} px,\n"
                                     f"bottom {fan['bottom']:+.1f} px", transform=axf.transAxes, fontsize=7.5,
                         va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d5dbe1"))
            lim = max(1.0, float(np.max(np.abs(values))) * 1.15)
            axf.set_xlim(-lim, lim)
        axf.set_title("Keystone fan", fontsize=9)
        axf.set_xlabel("right - left (px)", fontsize=8)
        axf.tick_params(labelleft=False, labelsize=7)
        axf.grid(alpha=.25)
        _draw(canvas)

    def _plot_deviation(self, p: dict) -> None:
        """Bars' deviation from straight, magnified x G around their rows: as measured (tilt +
        keystone) and with the common tilt removed (keystone only)."""
        canvas = self._canvases[DEVIATION]
        canvas.clear()
        fig = canvas.fig
        h, w = p["shape"]
        bars = list(p["bars"])
        if not bars:
            ax = fig.add_subplot(111)
            ax.text(.5, .5, "No bars to show: " + (p.get("error") or ""), ha="center", va="center",
                    transform=ax.transAxes, color="#a93226", wrap=True)
            ax.set_axis_off()
            _draw(canvas)
            return
        step = max(1, int(math.ceil(len(bars) / 12.0)))
        shown = bars[::step]
        if bars[-1] is not shown[-1] and len(bars) > 1:
            shown.append(bars[-1])
        x_ref = p["x_ref"]
        fan = p.get("fan") or {}
        slope = (float(fan["coef"][0]) / max(1.0, fan["xr"] - fan["xl"])) if "coef" in fan else 0.0
        rows = [float(b.y_at(np.array([np.clip(x_ref, b.x.min(), b.x.max())]), w)[0]) for b in shown]
        spacing = float(np.median(np.diff(rows))) if len(rows) > 1 else 0.25 * h
        panels = [("As measured (tilt + keystone)", 0.0),
                  ("Common tilt removed (%+.2f deg): keystone only" % math.degrees(math.atan(slope)), slope)]
        axes = []
        for col, (title, remove) in enumerate(panels):
            ax = fig.add_subplot(1, 2, col + 1, sharey=axes[0] if axes else None)
            axes.append(ax)
            curves = []
            for b, y0 in zip(shown, rows):
                xs = np.linspace(float(b.x.min()), float(b.x.max()), 200)
                curves.append((xs, b.y_at(xs, w) - y0 - remove * (xs - x_ref), y0))
            biggest = max(float(np.max(np.abs(d))) for _x, d, _y in curves)
            gain = nice_gain(0.45 * spacing / max(biggest, 1e-6))
            for xs, dev, y0 in curves:
                ax.plot([0, w - 1], [y0, y0], color="#b3b6b7", lw=.7, ls=(0, (4, 3)), zorder=1)
                yy = y0 + gain * dev
                ax.fill_between(xs, y0, yy, where=dev >= 0, color="#e74c3c", alpha=.5, lw=0, interpolate=True,
                                zorder=2)
                ax.fill_between(xs, y0, yy, where=dev < 0, color="#2e86c1", alpha=.5, lw=0, interpolate=True,
                                zorder=2)
                ax.plot(xs, yy, color="#1b2631", lw=.8, zorder=3)
            ax.axvline(x_ref, color="#7f8c8d", lw=.8, ls=":")
            unit = _nice_floor(biggest / 2.0) if biggest >= 2.0 else _nice_floor(biggest)
            x_bar = 0.025 * w
            y_bar = rows[0] - 0.15 * spacing
            ax.plot([x_bar, x_bar], [y_bar, y_bar - gain * unit], color="#1b2631", lw=3, solid_capstyle="butt",
                    zorder=4)
            ax.text(x_bar + 0.012 * w, y_bar - 0.5 * gain * unit, f"{unit:g} px", va="center", fontsize=8,
                    zorder=4, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.8))
            ax.set_xlim(-.5, w - .5)
            ax.set_xlabel("Column (spectral)")
            ax.set_title(f"{title}\nmagnified x{gain:g}, largest {biggest:.2f} px", fontsize=9)
            ax.grid(axis="x", alpha=.2)
        axes[0].set_ylim(h - .5, -.5)
        axes[0].set_ylabel("Row (slit)")
        axes[1].tick_params(labelleft=False)
        note = "every %d%s bar shown; " % (step, {2: "nd", 3: "rd"}.get(step, "th")) if step > 1 else ""
        fig.suptitle(f"Bar deviation from a straight line through column {x_ref:.0f} ({note}red = bar lower in "
                     "the frame, blue = higher; dashed = straight)", fontsize=9)
        _draw(canvas)

    def _plot_white(self) -> None:
        canvas = self._canvases[WHITE]
        if self.white is None:
            self._clear_plots("No white frame (optional, recommended)", (WHITE,))
            return
        canvas.clear()
        ax = canvas.fig.add_subplot(111)
        wf = np.asarray(self.white, np.float32)
        h, w = wf.shape
        lit = wf > 0.08 * float(np.percentile(wf, 99))
        vmin, vmax = _stretch(wf)
        ax.imshow(wf, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax, extent=(-.5, w - .5, h - .5, -.5))
        try:
            ax.contour(np.arange(w), np.arange(h), lit.astype(np.float32), levels=[0.5], colors="#00bcd4",
                       linewidths=1.2)
        except Exception:  # noqa: BLE001  (all lit / none lit)
            pass
        ax.set_xlim(-.5, w - .5)
        ax.set_ylim(h - .5, -.5)
        ax.set_xlabel("Column (spectral)")
        ax.set_ylabel("Row (slit)")
        ax.set_title(f"White frame: lit region outlined ({100 * lit.mean():.0f} % of the frame); outside it the "
                     "target is not used", fontsize=9)
        _draw(canvas)

    def _plot_target(self, k) -> None:
        canvas = self._canvases[BEFORE_AFTER]
        canvas.clear()
        fig = canvas.fig
        before = np.asarray(k.target_norm, dtype=np.float32)
        after = np.asarray(k.target_corr, dtype=np.float32)
        finite = before[np.isfinite(before) & (before > 0)]
        vmin, vmax = (np.percentile(finite, [1, 99.5]) if finite.size else (0, 1))
        h, w = before.shape
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1, sharey=ax1)
        band = self._worst_band(k, h, w)
        where = f"rows {band[0]:.0f}-{band[1]:.0f}, where the keystone is largest; " if band else ""
        ax1.imshow(before, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax1.set_title(f"Before (raw, {where}toolbar Home / zoom / pan): the bars cross the dashed straight lines",
                      fontsize=9)
        ax2.imshow(after, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")
        med = _num(k.qc.get("retrace_target_median_bar_px"))
        ax2.set_title("After smile + keystone remap: the bars sit on the dashed straight lines"
                      + (f" (re-traced median {med:.2f} px)" if np.isfinite(med) else ""), fontsize=9)
        for Y in k.model.target_Y:
            for ax in (ax1, ax2):
                ax.axhline(Y, color="#ff8c1a", lw=.8, ls=(0, (6, 4)), alpha=.9)
        if band:
            ax1.set_ylim(band[1], band[0])
        left, right = k.margin
        top, bottom = k.row_margin
        for x in (left - .5, w - right - .5):
            ax2.axvline(x, color="#2980b9", lw=.8, ls="--")
        for y in (top - .5, h - bottom - .5):
            ax2.axhline(y, color="#2980b9", lw=.8, ls="--")
        ax1.set_ylabel("Row (slit)")
        ax2.set_ylabel("Row (slit)")
        ax2.set_xlabel("Column (spectral)")
        _draw(canvas)

    @staticmethod
    def _worst_band(k, h: int, w: int):
        """Rows ``(top, bottom)`` around the bar that bends most (about seven bar periods), or ``None``."""
        lines = list(k.target_lines or [])
        if len(lines) < 2:
            return None
        x_ref = float(k.model.x_ref)
        bend = []
        for b in lines:
            x0 = float(np.clip(x_ref, b.x.min(), b.x.max()))
            ends = b.y_at(np.array([b.x.min(), b.x.max()]), w) - b.y_at(np.array([x0]), w)[0]
            bend.append(float(np.max(np.abs(ends))))
        centre = float(lines[int(np.argmax(bend))].y_seed)
        seeds = sorted(float(b.y_seed) for b in lines)
        period = float(np.median(np.diff(seeds)))
        half = max(3.5 * period, 30.0)
        if 2 * half >= 0.6 * h:
            return None
        # stay over the traced bars: rows outside them have no dashed guide line to compare with
        lo = max(-.5, seeds[0] - 0.75 * period)
        hi = min(h - .5, seeds[-1] + 0.75 * period)
        if hi - lo <= 2 * half:
            return None
        top = float(np.clip(centre - half, lo, hi - 2 * half))
        return top, top + 2 * half

    def _grid_lines(self, k):
        """Destination grid rows / columns for the distortion plot."""
        h, w = k.model.shape
        box = k.model.dy.box or [0, w - 1, 0, h - 1]
        r0, r1 = float(box[2]), float(box[3])
        rows = list(k.model.target_Y)
        if not rows or len(rows) > 16:
            rows = list(np.linspace(r0, r1, 13))
        cols = list(np.linspace(0.04 * (w - 1), 0.96 * (w - 1), 13))
        return rows, cols, [float(v) for v in (k.model.lamp_X or [])]

    def _plot_grid(self, k) -> None:
        """Where a straight destination grid is read from in the raw frame (``model.source``)."""
        canvas = self._canvases[GRID]
        if k is None:
            self._clear_plots("Run Analyze Keystone to see the check plots", (GRID,))
            return
        canvas.clear()
        ax = canvas.fig.add_subplot(111)
        model = k.model
        h, w = model.shape
        raw = np.asarray(k.target_norm, np.float32)
        lit = raw > 0
        vmin, vmax = _stretch(raw, lit if lit.any() else None)
        ax.imshow(raw, cmap="gray", aspect="auto", vmin=vmin - 0.8 * (vmax - vmin), vmax=vmax, alpha=.6,
                  extent=(-.5, w - .5, h - .5, -.5), interpolation="nearest")
        rows, cols, lamp_cols = self._grid_lines(k)
        xs = np.linspace(0, w - 1, 240)
        ys = np.linspace(0, h - 1, 240)
        row_curves = [(np.full_like(xs, Y), xs) for Y in rows]
        col_curves = [(ys, np.full_like(ys, X)) for X in cols + lamp_cols]
        disp = []
        for Yv, Xv in row_curves + col_curves:
            sx, sy = model.source(Xv, Yv)
            disp.append(np.hypot(sx - Xv, sy - Yv).max())
        biggest = float(max(disp)) if disp else 0.0
        spacing = min((h - 1) / max(1, len(rows)), (w - 1) / 13.0)
        gain = nice_gain(0.6 * spacing / max(biggest, 1e-6)) if self.chk_exaggerate.isChecked() else 1.0
        for Y in rows:
            ax.plot([0, w - 1], [Y, Y], color="#34495e", lw=.7, ls=(0, (2, 3)), alpha=.7, zorder=2)
        for X in cols:
            ax.plot([X, X], [0, h - 1], color="#34495e", lw=.7, ls=(0, (2, 3)), alpha=.7, zorder=2)
        for i, (Yv, Xv) in enumerate(row_curves):
            sx, sy = model.source(Xv, Yv)
            ax.plot(Xv + gain * (sx - Xv), Yv + gain * (sy - Yv), color="#ff8c1a", lw=1.5, zorder=3,
                    label="rows: keystone + tilt" if i == 0 else None)
        for i, (Yv, Xv) in enumerate(col_curves):
            sx, sy = model.source(Xv, Yv)
            lamp = i >= len(cols)
            ax.plot(Xv + gain * (sx - Xv), Yv + gain * (sy - Yv), color="#b388ff" if lamp else "#1fb5d6",
                    lw=1.2 if lamp else 1.5, zorder=3,
                    label=("lamp lines" if lamp and i == len(cols) else ("columns: smile" if i == 0 else None)))
        box = model.dy.box
        if box is not None and len(box) == 4:
            c0, c1, r0, r1 = (float(v) for v in box)
            ax.plot([c0, c1, c1, c0, c0], [r0, r0, r1, r1, r0], color="#2ecc71", lw=1, ls="--", zorder=4,
                    label="measured bars")
        ax.set_xlim(-.5, w - .5)
        ax.set_ylim(h - .5, -.5)
        ax.set_xlabel("Column (spectral)")
        ax.set_ylabel("Row (slit)")
        scale = f"displacement x{gain:g}" if gain != 1 else "true scale"
        onto = "stored smile" if k.qc.get("smile_source") == "stored" else "joint fit"
        ax.set_title(f"Distortion grid ({scale}, largest {biggest:.1f} px; {onto}): the corrected image reads each "
                     "straight dotted grid line from the curve drawn here", fontsize=9)
        leg = ax.legend(fontsize=7.5, loc="upper right", framealpha=.85)
        leg.set_zorder(5)
        _draw(canvas)

    def _plot_shift_map(self, map_y, box, row_margin, margin) -> None:
        canvas = self._canvases[SHIFT_MAP]
        canvas.clear()
        ax = canvas.fig.add_subplot(111)
        map_y = np.asarray(map_y, dtype=np.float32)
        h, w = map_y.shape
        dy = map_y - np.arange(h, dtype=np.float32)[:, None]
        lim = float(np.nanmax(np.abs(dy))) if np.isfinite(dy).any() else 1.0
        lim = max(lim, 1e-3)
        im = ax.imshow(dy, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto",
                       extent=(-.5, w - .5, h - .5, -.5))
        canvas.fig.colorbar(im, ax=ax, label="Row shift map_y - Y (px)")
        try:
            levels = np.linspace(-lim, lim, 11)
            cs = ax.contour(np.arange(w), np.arange(h), dy, levels=levels, colors="k", linewidths=.4)
            ax.clabel(cs, fmt="%.1f", fontsize=6)
        except Exception:  # noqa: BLE001  (flat map: no contour)
            pass
        if box is not None and len(box) == 4:
            c0, c1, r0, r1 = (float(v) for v in box)
            ax.plot([c0, c1, c1, c0, c0], [r0, r0, r1, r1, r0], color="#27ae60", lw=1, ls="--",
                    label="measured bars")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(-.5, w - .5)
        ax.set_ylim(h - .5, -.5)
        ax.set_xlabel("Column (spectral)")
        ax.set_ylabel("Row (slit)")
        ax.set_title(f"Keystone row shift (source row - destination row); border rows {tuple(row_margin)}, "
                     f"columns {tuple(margin)}", fontsize=9)
        _draw(canvas)

    def _plot_retrace(self, qc: dict) -> None:
        canvas = self._canvases[RETRACE]
        canvas.clear()
        fig = canvas.fig
        bars = qc.get("retrace_target_per_bar") or []
        lamps = qc.get("retrace_lamp_per_line") or []
        ax1 = fig.add_subplot(211)
        if bars:
            y = np.array([_num(p.get("Y")) for p in bars])
            ax1.plot(y, [_num(p.get("rms_px")) for p in bars], "o-", ms=3, label="RMS")
            ax1.plot(y, [_num(p.get("max_px")) for p in bars], "x", ms=3, alpha=.5, label="max |dev|")
        ax1.axhline(RETRACE_WARN_PX, color="#e67e22", lw=.8, ls="--", label=f"{RETRACE_WARN_PX:g} px")
        ax1.axhline(2 * RETRACE_WARN_PX, color="#c0392b", lw=.8, ls=":", label=f"{2 * RETRACE_WARN_PX:g} px")
        ax1.set(xlabel="Bar row (corrected)", ylabel="Re-trace residual (px)",
                title="Target bars re-traced on the corrected frame (0 = straight)")
        ax1.grid(alpha=.2)
        ax1.legend(fontsize=7)
        ax2 = fig.add_subplot(212)
        if lamps:
            x = np.array([_num(p.get("X")) for p in lamps])
            ax2.plot(x, [_num(p.get("rms_px")) for p in lamps], "s-", ms=3, color="#8e44ad", label="RMS")
            ax2.plot(x, [_num(p.get("max_px")) for p in lamps], "x", ms=3, color="#8e44ad", alpha=.5,
                     label="max |dev|")
        ax2.axhline(RETRACE_WARN_PX, color="#e67e22", lw=.8, ls="--")
        title = "Lamp lines re-traced on the corrected frame"
        if not lamps and qc.get("smile_source") == "stored":
            title += " (stored smile: no lamp frame in this session)"
        ax2.set(xlabel="Lamp line column (corrected)", ylabel="Re-trace residual (px)", title=title)
        ax2.grid(alpha=.2)
        if lamps:
            ax2.legend(fontsize=7)
        _draw(canvas)

    def _plot_spacing(self, k) -> None:
        """Spacing change of adjacent bars against the reference column, from the MEASURED centres.

        An inspection aid only (:func:`modules.smile.keystone_spacing.spacing_variation`): it does
        not assert the physical target's bar pitch, and never alters the model or marks a result
        safe to Apply.
        """
        canvas = self._canvases[SPACING]
        canvas.clear()
        self._spacing_stats = None
        if k is None or len(k.target_lines) < 2:
            self._clear_plots("Need at least two traced target bars for spacing variation", (SPACING,))
            return
        try:
            stats = spacing_variation(k.target_lines, k.model)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            self._clear_plots("Spacing variation unavailable: " + str(exc), (SPACING,))
            return
        self._spacing_stats = stats
        if not stats["points"]:
            self._clear_plots("Too few overlapping bar positions", (SPACING,))
            return
        ax = canvas.fig.add_subplot(111)
        ax.plot(stats["x"], stats["before"], lw=1.6, color="#e67e22",
                label=f"Before ({stats['before_rms']:.3f} px RMS)")
        ax.plot(stats["x"], stats["after"], lw=1.6, color="#088e82",
                label=f"After ({stats['after_rms']:.3f} px RMS)")
        ax.axhline(0, color="#6b7d8c", lw=.8, ls="--")
        ax.set(xlabel="Spectral column", ylabel="Spacing change (px)",
               title="Bar spacing vs reference column, measured centres (%d pairs)" % stats["pairs"])
        ax.grid(alpha=.18)
        ax.legend(loc="best", fontsize=7)
        _draw(canvas)

    def _on_export_plot(self, index: int) -> None:
        """Save one plot as a 300-dpi PNG."""
        path, _ = QFileDialog.getSaveFileName(self, "Export plot as PNG", _PLOTS[index][1], "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            self._canvases[index].fig.savefig(path, format="png", dpi=300, bbox_inches="tight",
                                              facecolor=self._canvases[index].fig.get_facecolor())
            self._status(f"Exported plot PNG: {path} (300 dpi)")
        except Exception as exc:  # noqa: BLE001
            self._status(f"Plot export failed: {exc}")
