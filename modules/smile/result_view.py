# smile/result_view.py
"""Result page: what a smile + keystone model does to the frames it was measured from.

:func:`correction_result` (any thread) remaps the lamp / target / white frames with a model
(:class:`~modules.smile.corrector.SmileCorrector`), traces the lamp lines and the target bars on
the frames before and after, and returns their deviation from straight. :class:`ResultView`
(GUI thread) shows it: a verdict strip, a before / after wipe of both frames with the traced
lines on top, and the residual maps.

The *after* numbers are measured on the corrected frames (lines and bars traced again), never
drawn as straight lines. Displacement can be drawn magnified (``xG``); the frames are warped by
the same measured field so lines and image stay together. Every number shown is true pixels.

Data conventions:
    Frames ``(rows = slit, columns = spectral)``, raw orientation as analysed (never corrected).
    ``dx`` (smile) of a lamp line at row ``y`` = its column there - its column at the reference
    row; ``dy`` (keystone) of a bar at column ``x`` = its row there - its row at the reference
    column. The *keystone* figure has the common tilt of the bars removed (the tilt is reported
    on its own); the drawn curves keep it.

Threading:
    :func:`correction_result` is pure (worker thread); widgets live on the GUI thread.
"""
from __future__ import annotations

import math
import os
from typing import Callable, List, Optional

import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .ui_theme import colors, is_dark

__all__ = ["correction_result", "ResultView", "PASS_PX"]

PASS_PX = 0.25                       #: residual limit (px), same as the keystone re-trace check
_GAINS = [("Auto", 0), ("x1", 1), ("x10", 10), ("x25", 25), ("x50", 50)]
_GRID = 4                            # field grid step (px)
_LAMP, _BAR = "#c58cff", "#2fd3e6"


# ------------------------------------------------------------------------------------- compute
def _lamp_traces(frame: np.ndarray, starts=None, rows_ok=None):
    """Traced lamp lines: list of ``(column_start, halfwin, trace (H,))``; *starts* reuses columns,
    *rows_ok* ``(first, last)`` blanks the rows outside (border rows a remap leaves invalid)."""
    from .core import auto_sat_level, detect_lines, estimate_noise, trace_line
    sat = auto_sat_level(frame)
    noise = estimate_noise(frame)
    if starts is None:
        starts = [(float(ln.column), max(3, int(round(1.2 * ln.fwhm))))
                  for ln in detect_lines(frame, sat_level=sat) if not ln.saturated]
    out = []
    h = frame.shape[0]
    for x0, halfwin in starts:
        t = trace_line(frame, x0, halfwin, sat, noise)
        if rows_ok is not None:
            t[:rows_ok[0]] = np.nan
            t[rows_ok[1] + 1:] = np.nan
        if np.isfinite(t).sum() >= 0.5 * h:
            out.append((x0, halfwin, t))
    return out


def _line_dev(trace: np.ndarray, y_ref: float):
    """``(rows, x, dev, fit)`` of one lamp trace: deviation from its column at *y_ref*."""
    rows = np.flatnonzero(np.isfinite(trace)).astype(float)
    x = trace[rows.astype(int)]
    deg = 3 if rows.size > 40 else 2
    coef = np.polyfit(rows - y_ref, x, deg)
    x_ref = float(np.polyval(coef, 0.0))
    return rows, x, x - x_ref, coef, x_ref


def _model_lines(corrector, shape, y_ref: float, columns, n: int = 10) -> list:
    """Lamp lines as the model predicts them (no usable lamp frame).

    The column map sends output column ``c`` at row ``y`` to raw column ``map_x[y, c]``, so a line
    that ends straight at ``c`` runs through those raw columns. The model's own ``map_x`` is used
    (a joint smile + keystone map included); a smile surface ``S`` (``map_x = x + S``) otherwise.
    """
    h, _w = shape
    c0, c1 = columns
    cols = np.linspace(c0 + 0.04 * (c1 - c0), c1 - 0.04 * (c1 - c0), n)
    rows = np.arange(h, dtype=float)
    map_x = getattr(corrector, "_map_x", None)
    if map_x is not None and tuple(np.shape(map_x)) == tuple(shape):
        X = np.asarray(map_x, float)[:, np.round(cols).astype(int)]
    else:
        surface = corrector._surface
        X = cols[None, :] + np.asarray(surface.grid(rows.astype(np.float32), cols.astype(np.float32)), float)
    ref = X[int(np.clip(round(y_ref), 0, h - 1))]
    lines = []
    for j in range(cols.size):
        dev = X[:, j] - ref[j]
        lines.append(dict(rows=rows, x=X[:, j], dev=dev, xfit=X[:, j], devfit=dev))
    return lines


def _bars(target, white):
    from .keystone import detect_polarity, normalize_target, trace_target_lines
    norm = normalize_target(target, white)
    polarity, _info = detect_polarity(target, normalized=norm)
    try:
        bars, _ = trace_target_lines(target, polarity=polarity, normalized=norm)
    except ValueError:
        bars = []
    return bars, norm, polarity


def _field_from_bars(bars, shape, x_ref: float, slope: float = 0.0):
    """Dense ``dy`` (rows, ``(H//g+1, W//g+1)``) between bars, plus per-bar curves on the grid."""
    h, w = shape
    xs = np.arange(0, w, _GRID, dtype=float)
    ys = np.arange(0, h, _GRID, dtype=float)
    if not bars:
        return np.zeros((ys.size, xs.size), np.float32), []
    curves = []
    for b in bars:
        lo, hi = float(np.min(b.x)), float(np.max(b.x))
        r0 = float(b.y_at(np.array([np.clip(x_ref, lo, hi)]), w)[0])
        y = b.y_at(xs, w).astype(float)
        dev = y - r0 - slope * (xs - x_ref)
        out = (xs < lo) | (xs > hi)
        y[out], dev[out] = np.nan, np.nan
        curves.append((r0, y, dev))
    field = np.zeros((ys.size, xs.size), np.float32)
    for j in range(xs.size):
        pos = np.array([c[1][j] for c in curves])
        dev = np.array([c[2][j] for c in curves])
        ok = np.isfinite(pos) & np.isfinite(dev)
        if ok.any():
            o = np.argsort(pos[ok])
            field[:, j] = np.interp(ys, pos[ok][o], dev[ok][o])
    return field, curves


def _field_from_lines(lines, shape):
    """Dense ``dx`` between lamp lines on the field grid (``lines``: ``(rows, fit_x(y), dev(y))``)."""
    h, w = shape
    xs = np.arange(0, w, _GRID, dtype=float)
    ys = np.arange(0, h, _GRID, dtype=float)
    field = np.zeros((ys.size, xs.size), np.float32)
    if not lines:
        return field
    for i, y in enumerate(ys):
        pos = np.array([ln["xfit"][int(y)] for ln in lines])
        dev = np.array([ln["devfit"][int(y)] for ln in lines])
        ok = np.isfinite(pos) & np.isfinite(dev)
        if ok.any():
            o = np.argsort(pos[ok])
            field[i] = np.interp(xs, pos[ok][o], dev[ok][o])
    return field


def _stats(values) -> dict:
    v = np.concatenate([np.ravel(a) for a in values]) if values else np.zeros(0)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(max=float("nan"), rms=float("nan"), n=0)
    return dict(max=float(np.max(np.abs(v))), rms=float(np.sqrt(np.mean(v ** 2))), n=int(v.size))


def correction_result(corrector, frames: dict) -> dict:
    """Measure a model on its own frames, before and after correction.

    Args:
        corrector: Fitted :class:`~modules.smile.corrector.SmileCorrector` (smile or smile + keystone).
        frames: ``{"lamp": ..., "target": ..., "white": ...}`` (any may be missing), raw frames
            of the model's size.

    Returns:
        dict with ``shape``, ``y_ref``, ``x_ref``, ``has_keystone``, per mode (``"before"``,
        ``"after"``): ``lamp`` / ``target_norm`` images (or ``None``), ``lines`` (lamp traces:
        dicts ``rows, x, dev, xfit, devfit``), ``bars`` (``(row_ref, y_on_grid, dev_on_grid)``),
        ``dx`` / ``dy`` fields on the ``grid`` (``(xs, ys)``) and the summary ``stats``:
        ``smile`` / ``keystone`` (``{"before": {max, rms, n}, "after": ...}``), ``tilt_deg``.

    Raises:
        ValueError: No lamp and no target frame, or frames of another size than the model.
    """
    lamp, target, white = (frames.get(k) for k in ("lamp", "target", "white"))
    if lamp is None and target is None:
        raise ValueError("No lamp or target frame to measure the model on")
    shape = getattr(corrector, "_image_shape", None)
    shape = tuple(shape) if shape is not None else None
    for name, f in (("lamp", lamp), ("target", target), ("white", white)):
        if f is not None and shape is not None and tuple(np.shape(f)) != shape:
            raise ValueError(f"The {name} frame {np.shape(f)} differs from the model {shape}")
    some = lamp if lamp is not None else target
    h, w = np.shape(some)
    shape = (h, w)
    y_ref = float(corrector.y_ref) if getattr(corrector, "y_ref", None) is not None else (h - 1) / 2.0
    x_ref = (w - 1) / 2.0

    def fix(f):
        return None if f is None else corrector.transform(np.asarray(f, np.float32), crop=False)

    raw = dict(lamp=None if lamp is None else np.asarray(lamp, np.float32),
               target=None if target is None else np.asarray(target, np.float32),
               white=None if white is None else np.asarray(white, np.float32))
    out = dict(shape=shape, y_ref=y_ref, x_ref=x_ref, has_keystone=bool(getattr(corrector, "has_keystone", False)),
               grid=(np.arange(0, w, _GRID, dtype=float), np.arange(0, h, _GRID, dtype=float)))
    smile, key = {}, {}
    starts = None
    # after the remap the outermost rows / columns hold no data (cropped from corrected cubes)
    rm = tuple(getattr(corrector, "row_margin", (0, 0)) or (0, 0))
    cm = tuple(getattr(corrector, "_margin", (0, 0)) or (0, 0))
    valid_rows = (int(rm[0]), h - 1 - int(rm[1]))
    valid_cols = (int(cm[0]), w - 1 - int(cm[1]))
    out["valid"] = dict(rows=valid_rows, columns=valid_cols)
    tilt_deg = 0.0
    for mode in ("before", "after"):
        lp = raw["lamp"] if mode == "before" else fix(raw["lamp"])
        tg = raw["target"] if mode == "before" else fix(raw["target"])
        wh = raw["white"] if mode == "before" else fix(raw["white"])
        rec = dict(lamp=lp, target_norm=None, lines=[], bars=[], dx=None, dy=None)
        # lamp lines (smile)
        if lp is not None:
            traces = _lamp_traces(lp, starts, valid_rows if mode == "after" else None)
            if mode == "before":
                starts = [(float(np.nanmedian(t)), hw) for _x0, hw, t in traces]
            rows_all = np.arange(h, dtype=float)
            for _x0, _hw, t in traces:
                rows, x, dev, coef, xr = _line_dev(t, y_ref)
                xfit = np.polyval(coef, rows_all - y_ref)
                xfit[(rows_all < rows.min()) | (rows_all > rows.max())] = np.nan
                rec["lines"].append(dict(rows=rows, x=x, dev=dev, xfit=xfit, devfit=xfit - xr))
            smile[mode] = _stats([ln["dev"] for ln in rec["lines"]])
        if (mode == "before" and not rec["lines"]
                and (getattr(corrector, "_map_x", None) is not None or getattr(corrector, "_surface", None) is not None)):
            rec["model_lines"] = _model_lines(corrector, shape, y_ref, valid_cols)
            smile["model"] = _stats([ln["dev"] for ln in rec["model_lines"]])
        rec["dx"] = _field_from_lines(rec["lines"] or rec.get("model_lines") or [], shape)
        # target bars (keystone)
        if tg is not None:
            bars, norm, _pol = _bars(tg, wh if wh is not None and np.any(wh > 0) else None)
            rec["target_norm"] = norm
            slope = 0.0
            if mode == "before" and len(bars) >= 2:     # common tilt: median end-to-end slope of the bars
                sl = [(float(b.y_at(np.array([b.x.max()]), w)[0]) - float(b.y_at(np.array([b.x.min()]), w)[0]))
                      / max(1.0, float(b.x.max() - b.x.min())) for b in bars]
                slope = float(np.median(sl))
                tilt_deg = math.degrees(math.atan(slope))
            rec["dy"], rec["bars"] = _field_from_bars(bars, shape, x_ref)
            meas = []
            for b in bars:
                bx, by = np.asarray(b.x, float), np.asarray(b.y, float)
                if mode == "after":
                    keep = (bx >= valid_cols[0]) & (bx <= valid_cols[1]) & (by >= valid_rows[0]) & (by <= valid_rows[1])
                    bx, by = bx[keep], by[keep]
                    if bx.size == 0:
                        continue
                lo, hi = float(bx.min()), float(bx.max())
                r0 = float(b.y_at(np.array([np.clip(x_ref, lo, hi)]), w)[0])
                meas.append(by - r0 - slope * (bx - x_ref))
            key[mode] = _stats(meas)
        else:
            rec["dy"] = np.zeros_like(rec["dx"])
        out[mode] = rec
    out["stats"] = dict(smile=smile, keystone=key, tilt_deg=tilt_deg)
    if smile.get("before", {}).get("n"):
        out["smile_source"] = "measured"
    elif lamp is not None:
        out["smile_source"] = "untraced"            # a lamp frame, but no line could be followed
    else:
        out["smile_source"] = "model" if "model" in smile else "none"
    return out


class ResultWorker(QThread):
    """Run :func:`correction_result` off the GUI thread (``done(token, result)`` / ``failed(token, msg)``)."""

    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, token: int, corrector, frames: dict):
        super().__init__()
        self._token, self._corrector, self._frames = token, corrector, frames

    def run(self) -> None:
        try:
            self.done.emit(self._token, correction_result(self._corrector, self._frames))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self._token, str(exc))


# ------------------------------------------------------------------------------------------ view
def _nice_gain(value: float) -> float:
    if not np.isfinite(value) or value <= 1:
        return 1.0
    exp = 10 ** math.floor(math.log10(value))
    for m in (5, 2, 1):
        if m * exp <= value:
            return float(m * exp)
    return float(exp)


def _norm01(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.8) -> np.ndarray:
    lit = img[img > 0]
    if lit.size < 16:
        return np.zeros_like(img, np.float32)
    lo, hi = np.percentile(lit, [lo_pct, hi_pct])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def _composite(rec: dict) -> np.ndarray:
    """Target (grey, dark bars) with the lamp lines (bright) on top, 0..1."""
    parts = []
    if rec.get("target_norm") is not None:
        t = np.asarray(rec["target_norm"], np.float32)
        top = float(np.percentile(t[t > 0], 99)) if np.any(t > 0) else 1.0
        parts.append(0.55 * np.clip(t / max(top, 1e-6), 0, 1.1))
    if rec.get("lamp") is not None:
        parts.append(0.8 * _norm01(rec["lamp"], 50, 99.9))
    img = np.sum(parts, axis=0) if parts else None
    return np.clip(img, 0, 1).astype(np.float32)


class _Tile(QFrame):
    """One verdict tile: label, value line, sub line."""

    def __init__(self, label: str, accent: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setObjectName("resultTile")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 9, 12, 9)
        lay.setSpacing(3)
        self.lbl = QLabel(label.upper())
        self.val = QLabel("-")
        self.val.setTextFormat(Qt.RichText)
        self.sub = QLabel("")
        self.sub.setWordWrap(True)
        for wdg in (self.lbl, self.val, self.sub):
            lay.addWidget(wdg)
        self._accent = accent

    def restyle(self, pal: dict) -> None:
        top = f"border-top: 2px solid {self._accent};" if self._accent else ""
        self.setStyleSheet(f"#resultTile {{ background: {pal['card']}; border: 1px solid {pal['border']}; "
                           f"border-radius: 8px; {top} }}")
        clear = "background: transparent; border: none;"
        self.lbl.setStyleSheet(f"{clear} color: {pal['muted']}; font-size: 10px; letter-spacing: 1px;")
        self.val.setStyleSheet(f"{clear} color: {pal['text']}; font-family: Consolas, 'JetBrains Mono', monospace; "
                               "font-size: 20px; font-weight: 700;")
        self.sub.setStyleSheet(f"{clear} color: {pal['muted']}; font-size: 11px;")


class ResultView(QWidget):
    """Smile + keystone result page (see the module docstring).

    Args:
        session_model: ``() -> (corrector, frames, frames_info, label)`` of this session's model
            (``corrector`` ``None`` when there is none), used by *Use this session's model*.
    """

    statusChanged = pyqtSignal(str)

    def __init__(self, session_model: Optional[Callable] = None, parent: Optional[QWidget] = None,
                 session_frames_key: Optional[Callable] = None):
        super().__init__(parent)
        from .widget import PlotCanvas          # lazy: widget.py imports this module
        self._session_model = session_model
        self._session_frames_key = session_frames_key or (lambda: None)
        self._result: Optional[dict] = None
        self._worker: Optional[ResultWorker] = None
        self._token = 0
        self._split = 0.5
        self._drag = False
        self._gain = 1.0
        self._session_key = None
        self._imgs = {}
        self._clip = {}
        self._ax = None
        self._label = ""

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)
        top = QHBoxLayout()
        self.btn_session = QPushButton("Use this session's model")
        self.btn_session.setToolTip("Measure the model of this session (Smile / Keystone tabs) on the frames it "
                                    "was measured from")
        self.btn_session.clicked.connect(self.show_session_model)
        self.btn_open = QPushButton("Open model file...")
        self.btn_open.setToolTip("Measure a stored model (.npz saved with its frames)")
        self.btn_open.clicked.connect(self._on_open)
        self.lbl_source = QLabel("No model yet")
        self.lbl_source.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.btn_export = QPushButton("Export PNG")
        self.btn_export.setToolTip("Export this page's figure as a presentation-quality PNG (300 dpi).")
        self.btn_export.clicked.connect(self._on_export)
        for wdg in (self.btn_session, self.btn_open):
            top.addWidget(wdg)
        top.addWidget(self.lbl_source, stretch=1)
        top.addWidget(self.btn_export)
        lay.addLayout(top)

        tiles = QGridLayout()
        tiles.setSpacing(8)
        self.tile_smile = _Tile("Smile, max |dx|", _LAMP)
        self.tile_key = _Tile("Keystone, max |dy|", _BAR)
        self.tile_rms = _Tile("Residual RMS (re-traced)")
        self.tile_verdict = _Tile("Verdict")
        for i, t in enumerate((self.tile_smile, self.tile_key, self.tile_rms, self.tile_verdict)):
            tiles.addWidget(t, 0, i)
        lay.addLayout(tiles)

        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("Displacement"))
        self._gain_group = QButtonGroup(self)
        for i, (text, _g) in enumerate(_GAINS):
            b = QToolButton()
            b.setText(text)
            b.setCheckable(True)
            b.setToolTip("Draw the measured displacement magnified (the frames are warped the same way). "
                         "x1 = true scale. Numbers are always true pixels.")
            self._gain_group.addButton(b, i)
            row.addWidget(b)
        self._gain_group.button(0).setChecked(True)
        self._gain_group.buttonClicked.connect(lambda _b: self._redraw())
        self.chk_lamp = QCheckBox("Lamp lines")
        self.chk_bars = QCheckBox("Target bars")
        self.chk_ref = QCheckBox("Straight reference")
        for c, on in ((self.chk_lamp, True), (self.chk_bars, True), (self.chk_ref, False)):
            c.setChecked(on)
            c.toggled.connect(lambda _on: self._redraw())
            row.addWidget(c)
        row.addSpacing(10)
        row.addWidget(QLabel("Maps"))
        self._map_group = QButtonGroup(self)
        for i, text in enumerate(("After", "Before")):
            b = QToolButton()
            b.setText(text)
            b.setCheckable(True)
            self._map_group.addButton(b, i)
            row.addWidget(b)
        self._map_group.button(0).setChecked(True)
        self._map_group.buttonClicked.connect(lambda _b: self._redraw())
        row.addStretch()
        hint = QLabel("Drag the divider to compare before / after")
        row.addWidget(hint)
        lay.addLayout(row)

        self.canvas = PlotCanvas(self, figsize=(11, 8))
        lay.addWidget(self.canvas, stretch=1)
        self.lbl_hover = QLabel(" ")
        self.lbl_hover.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_hover.setStyleSheet("font-family: Consolas, 'JetBrains Mono', monospace;")
        lay.addWidget(self.lbl_hover)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self._restyle()
        self._message("Apply or analyse a model, then press 'Use this session's model' - or open a model file "
                      "saved with its frames.")

    # ------------------------------------------------------------------ public
    @property
    def running(self) -> bool:
        return self._worker is not None

    @property
    def result(self) -> Optional[dict]:
        return self._result

    def show_session_model(self) -> bool:
        """Measure this session's model (see *session_model*). Returns ``False`` when there is none."""
        if self._session_model is None:
            return False
        corrector, frames, _info, label = self._session_model()
        if corrector is None:
            self._message("No model in this session yet: analyse the smile (Smile tab) and, for the keystone, "
                          "run Analyze Keystone - or open a model file.")
            return False
        self._session_key = self._key(corrector)
        return self.measure(corrector, frames, label)

    def _key(self, corrector):
        return (id(corrector), id(getattr(corrector, "keystone", None)), self._session_frames_key())

    def session_changed(self) -> bool:
        """``True`` when the session's model differs from the one last measured from it."""
        if self._session_model is None:
            return False
        corrector = self._session_model()[0]
        if corrector is None:
            return False
        return self._key(corrector) != self._session_key

    def open_model(self, path: str) -> bool:
        """Measure a model file with the frames stored in it."""
        from .calibration import load_frames
        from .corrector import SmileCorrector
        try:
            corrector = SmileCorrector.load(path)
            frames, _info, _setting = load_frames(path)
        except Exception as exc:  # noqa: BLE001
            self._message(f"Cannot open {os.path.basename(path)}: {exc}")
            return False
        if "lamp" not in frames and "target" not in frames:
            self._message(f"{os.path.basename(path)} holds no frames (saved before frames were stored): Apply the "
                          "model again from this app to store them.")
            return False
        self._session_key = None
        return self.measure(corrector, frames, os.path.basename(path))

    def measure(self, corrector, frames: dict, label: str = "", synchronous: bool = False) -> bool:
        """Start measuring *corrector* on *frames* (worker thread, or here when ``synchronous``)."""
        frames = {k: v for k, v in (frames or {}).items() if v is not None}
        if "lamp" not in frames and "target" not in frames:
            self._message("This model has no lamp or target frame to measure it on.")
            return False
        self._token += 1
        self.lbl_source.setText(f"Measuring {label}..." if label else "Measuring...")
        self._label = label
        if synchronous:
            try:
                self._on_done(self._token, correction_result(corrector, frames))
            except Exception as exc:  # noqa: BLE001
                self._on_failed(self._token, str(exc))
            return self._result is not None
        if self._worker is not None:
            self._worker.wait()
        worker = ResultWorker(self._token, corrector, frames)
        worker.done.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        worker.start()
        return True

    def wait(self, timeout_s: float = 120.0) -> bool:
        """Block (processing events) until the worker is done; for scripts and tests."""
        import time
        from PyQt5.QtWidgets import QApplication
        deadline = time.time() + timeout_s
        while self._worker is not None and time.time() < deadline:
            QApplication.processEvents()
            time.sleep(0.01)
        QApplication.processEvents()
        return self._worker is None

    def set_split(self, fraction: float) -> None:
        """Divider position (0 = all after, 1 = all before)."""
        self._split = float(np.clip(fraction, 0.0, 1.0))
        self._update_split()

    # ------------------------------------------------------------------ worker
    def _on_done(self, token: int, result: dict) -> None:
        if token != self._token:
            return
        self._result = result
        self.lbl_source.setText(self._label or "model")
        self._update_tiles()
        self._redraw()
        self.statusChanged.emit("Result: " + self._verdict_text())

    def _on_failed(self, token: int, message: str) -> None:
        if token != self._token:
            return
        self._result = None
        self._message("Measuring the model failed: " + message)

    def _on_finished(self) -> None:
        if self._worker is not None:
            self._worker.wait()
            self._worker.deleteLater()
        self._worker = None

    # ------------------------------------------------------------------ tiles
    def _restyle(self) -> None:
        pal = colors(self)
        for t in (self.tile_smile, self.tile_key, self.tile_rms, self.tile_verdict):
            t.restyle(pal)

    def _residual(self):
        s = self._result["stats"]
        parts = [v["after"] for v in (s["smile"], s["keystone"]) if "after" in v and v["after"]["n"]]
        if not parts:
            return float("nan"), float("nan")
        rms = math.sqrt(sum(p["rms"] ** 2 * p["n"] for p in parts) / sum(p["n"] for p in parts))
        return rms, max(p["max"] for p in parts)

    def _scope(self) -> str:
        """What the verdict covers: the parts measured after correction."""
        s = self._result["stats"]
        smile = bool(s["smile"].get("after", {}).get("n"))
        key = bool(s["keystone"].get("after", {}).get("n"))
        if smile and key:
            return "smile + keystone"
        if key:
            why = ("no lines traced on the lamp frame" if self._result.get("smile_source") == "untraced"
                   else "no lamp frame")
            return f"keystone only (smile not measured: {why})"
        if smile:
            return "smile only (no target frame)"
        return ""

    def _verdict_text(self) -> str:
        rms, worst = self._residual()
        if not np.isfinite(rms):
            return "no lines or bars measured"
        ok = worst < PASS_PX
        return ("PASS" if ok else "CHECK") + f" - {self._scope()} (largest residual {worst:.3f} px)"

    def _update_tiles(self) -> None:
        pal = colors(self)
        s = self._result["stats"]

        def pair(tile, d, what, sub):
            if "before" in d and d["before"]["n"]:
                b, a = d["before"]["max"], d["after"]["max"] if "after" in d else float("nan")
                tile.val.setText(f"<span style='color:{pal['hint']}; font-size:15px; text-decoration:line-through'>"
                                 f"{b:.2f}</span> <span style='color:{pal['hint']}; font-size:14px'>&rarr;</span> "
                                 f"{a:.3f} <span style='font-size:12px; color:{pal['muted']}'>px</span>")
                tile.sub.setText(sub)
            else:
                tile.val.setText("-")
                tile.sub.setText(f"no {what} frame")

        pair(self.tile_smile, s["smile"], "lamp", "spectral line curvature along the slit, measured on "
                                                  f"{s['smile'].get('after', {}).get('n', 0)} points after")
        if self._result.get("smile_source") == "untraced":
            m = s["smile"].get("model")
            self.tile_smile.val.setText((f"<span style='color:{pal['hint']}'>&asymp; {m['max']:.2f}</span> "
                                         f"<span style='font-size:12px; color:{pal['muted']}'>px, model</span>")
                                        if m else "-")
            self.tile_smile.sub.setText("the lamp frame shows no line that could be traced (saturated, too faint "
                                        "or not a lamp frame): smile not measured")
        elif self._result.get("smile_source") == "model":
            m = s["smile"]["model"]
            self.tile_smile.val.setText(f"<span style='color:{pal['hint']}'>&asymp; {m['max']:.2f}</span> "
                                        f"<span style='font-size:12px; color:{pal['muted']}'>px, model</span>")
            self.tile_smile.sub.setText("from the stored smile model; no lamp frame, so not measured after "
                                        "correction. Load or grab a lamp frame on the Smile tab to check it.")
        pair(self.tile_key, s["keystone"], "target", f"row drift across wavelength (tilt {s['tilt_deg']:+.3f} deg "
                                                     "removed; the model corrects it too)")
        rms, worst = self._residual()
        if np.isfinite(rms):
            self.tile_rms.val.setText(f"{rms:.3f} <span style='font-size:12px; color:{pal['muted']}'>px</span>")
            sx, sk = s["smile"].get("after"), s["keystone"].get("after")
            self.tile_rms.sub.setText("  ".join(t for t in (
                f"dx {sx['rms']:.3f}" if sx and sx["n"] else "", f"dy {sk['rms']:.3f}" if sk and sk["n"] else "")
                if t) + " px")
            ok = worst < PASS_PX
            col = pal["good"] if ok else pal["warn"]
            partial = self._scope() != "smile + keystone"
            word = ("PASS" if ok else "CHECK") + (" *" if partial else "")
            self.tile_verdict.val.setText(f"<span style='color:{col}'>&#9679; {word}</span>")
            self.tile_verdict.sub.setText(f"{'* ' if partial else ''}{self._scope()}; largest residual "
                                          f"{worst:.3f} px (limit {PASS_PX} px)")
        else:
            self.tile_rms.val.setText("-")
            self.tile_verdict.val.setText("-")
            self.tile_verdict.sub.setText("no lines or bars could be traced")

    # ------------------------------------------------------------------ drawing
    def _message(self, text: str) -> None:
        self.canvas.clear()
        ax = self.canvas.fig.add_subplot(111)
        ax.text(.5, .5, text, ha="center", va="center", transform=ax.transAxes, color="#5d6d7e", fontsize=10,
                wrap=True)
        ax.set_axis_off()
        self._ax = None
        self.lbl_source.setText("No model" if self._result is None else self.lbl_source.text())
        self.canvas.draw_idle()

    def _pick_gain(self) -> float:
        choice = _GAINS[max(0, self._gain_group.checkedId())][1]
        if choice:
            return float(choice)
        r = self._result
        h = r["shape"][0]
        dev = [r["before"]["dx"], r["before"]["dy"]]
        biggest = max(float(np.max(np.abs(d))) for d in dev)
        spacing = []
        if r["before"]["bars"]:
            rows = sorted(c[0] for c in r["before"]["bars"])
            if len(rows) > 1:
                spacing.append(float(np.median(np.diff(rows))))
        xs = sorted(float(np.nanmedian(ln["x"])) for ln in r["before"]["lines"])
        if len(xs) > 1:
            spacing.append(float(np.median(np.diff(xs))))
        gap = min(spacing) if spacing else 0.1 * h
        return _nice_gain(0.3 * gap / max(biggest, 1e-3))

    def _warped(self, mode: str, gain: float) -> np.ndarray:
        """The mode's composite frame warped by ``(gain - 1) x`` its measured displacement field."""
        import cv2
        rec = self._result[mode]
        img = _composite(rec)
        if gain == 1.0:
            return img
        h, w = img.shape
        dx = cv2.resize(rec["dx"], (w, h), interpolation=cv2.INTER_LINEAR)
        dy = cv2.resize(rec["dy"], (w, h), interpolation=cv2.INTER_LINEAR)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        return cv2.remap(img, xs - (gain - 1) * dx, ys - (gain - 1) * dy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _redraw(self) -> None:
        if self._result is None:
            return
        import matplotlib.patheffects as pe
        from matplotlib.patches import Rectangle
        r = self._result
        h, w = r["shape"]
        gain = self._pick_gain()
        self._gain = gain
        dark = is_dark(self)
        pal = colors(self)
        fig = self.canvas.fig
        self.canvas.clear()
        try:
            fig.set_layout_engine("none")           # fixed margins: the wipe redraws fast and never re-flows
        except Exception:  # noqa: BLE001
            pass
        gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.25], left=.065, right=.975, top=.95, bottom=.06,
                              hspace=.32, wspace=.16)
        ax = fig.add_subplot(gs[0, :])
        self._ax = ax
        self._clip = {}
        self._clipped = {"before": [], "after": []}
        for mode in ("before", "after"):
            rect = Rectangle((0, -.5), 1, h, transform=ax.transData, fill=False, visible=False)
            ax.add_patch(rect)
            self._clip[mode] = rect
            img = ax.imshow(self._warped(mode, gain), cmap="gray", vmin=0, vmax=1, aspect="auto",
                            interpolation="bilinear", extent=(-.5, w - .5, h - .5, -.5))
            arts = [img]
            rec = r[mode]
            if self.chk_ref.isChecked():
                for row0, _y, _d in rec["bars"]:
                    arts.append(ax.axhline(row0, color=pal["muted"], lw=.7, ls=(0, (4, 4)), alpha=.7))
                for ln in rec["lines"]:
                    arts.append(ax.axvline(float(np.nanmedian(ln["x"] - ln["dev"])), color=pal["muted"], lw=.7,
                                           ls=(0, (4, 4)), alpha=.7))
            if self.chk_bars.isChecked():
                xs = r["grid"][0]
                for row0, y, d in rec["bars"]:
                    line, = ax.plot(xs, y + (gain - 1) * d, color=_BAR, lw=1.2)
                    line.set_path_effects([pe.Stroke(linewidth=3.2, foreground=_BAR, alpha=.25), pe.Normal()])
                    arts.append(line)
            if self.chk_lamp.isChecked():
                rows_all = np.arange(h, dtype=float)
                for ln in rec["lines"]:
                    line, = ax.plot(ln["xfit"] + (gain - 1) * ln["devfit"], rows_all, color=_LAMP, lw=1.2)
                    line.set_path_effects([pe.Stroke(linewidth=3.2, foreground=_LAMP, alpha=.25), pe.Normal()])
                    arts.append(line)
                for ln in rec.get("model_lines") or []:          # no lamp frame: what the model predicts
                    line, = ax.plot(ln["xfit"] + (gain - 1) * ln["devfit"], rows_all, color=_LAMP, lw=1.1,
                                    ls=(0, (5, 4)), alpha=.9)
                    arts.append(line)
            self._clipped[mode] = arts
        bbox = dict(boxstyle="round,pad=0.35", fc=(0.03, 0.06, 0.1, .85), ec="none")
        self._chip_before = ax.text(0.01, 0.985, "BEFORE", transform=ax.transAxes, ha="left", va="top",
                                    fontsize=8.5, fontweight="bold", color=pal["warn"], family="monospace", bbox=bbox,
                                    zorder=8)
        self._chip_after = ax.text(0.99, 0.985, "AFTER", transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                                   fontweight="bold", color=pal["good"], family="monospace", bbox=bbox, zorder=8)
        ax.text(0.01, 0.015, "TRUE SCALE" if gain == 1 else "displacement x%g" % gain, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8, fontweight="bold", family="monospace",
                color=pal["good"] if gain == 1 else "#f5a524", bbox=bbox, zorder=8)
        self._divider = ax.axvline(0, color="#ffffff", lw=2, zorder=9)
        self._knob, = ax.plot([0], [h / 2.0], marker="o", ms=17, mfc="#f5a524", mec="#0b1520", mew=2, zorder=10,
                              clip_on=False)
        ax.set_xlim(-.5, w - .5)
        ax.set_ylim(h - .5, -.5)
        ax.set_xlabel("Column (spectral)")
        ax.set_ylabel("Slit position (row)")
        title = "Before / after - lamp lines (purple) and target bars (cyan) traced on each frame"
        if r.get("smile_source") in ("model", "untraced"):
            title = ("Before / after - target bars (cyan) traced on each frame; dashed purple = smile predicted by "
                     "the model (no lamp frame)")
        ax.set_title(title, fontsize=10, loc="left")
        # residual maps
        after = self._map_group.checkedId() != 1
        mode = "after" if after else "before"
        lim = PASS_PX if after else max(1.0, math.ceil(max(float(np.max(np.abs(r["before"]["dx"]))),
                                                             float(np.max(np.abs(r["before"]["dy"])))) * 2) / 2)
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "resid", ["#4696ff", pal["card"] if dark else "#f4f6f8", "#ff9632"])
        for col, (key, title, guide) in enumerate((("dx", "Residual dx (smile)", "lines"),
                                                   ("dy", "Residual dy (keystone)", "bars"))):
            axm = fig.add_subplot(gs[1, col])
            field = r[mode][key]
            im = axm.imshow(field, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto", interpolation="bilinear",
                            extent=(-.5, w - .5, h - .5, -.5))
            s = r["stats"]["smile" if key == "dx" else "keystone"].get(mode)
            text = (f"max {s['max']:.3f} px, RMS {s['rms']:.3f} px" if s and s["n"] else "not measured")
            if key == "dx" and r.get("smile_source") in ("model", "untraced", "none"):
                text = "model prediction" if mode == "before" else "not measured (no lamp frame)"
                if mode == "after":
                    axm.text(.5, .5, "No lamp frame: smile not measured after correction", ha="center",
                             va="center", transform=axm.transAxes, fontsize=8.5, color=pal["muted"])
            axm.set_title(f"{title}, {mode}: {text}", fontsize=9, loc="left")
            axm.tick_params(labelsize=7)
            cb = fig.colorbar(im, ax=axm, fraction=.04, pad=.01)
            cb.ax.tick_params(labelsize=7)
            cb.set_label("px", fontsize=7)
        self._update_split(draw=False)
        self.canvas.draw_idle()

    def _update_split(self, draw: bool = True) -> None:
        if self._ax is None or self._result is None:
            return
        h, w = self._result["shape"]
        x = -.5 + self._split * w
        self._clip["before"].set_bounds(-.5, -.5, x + .5, h)
        self._clip["after"].set_bounds(x, -.5, w - .5 - x, h)
        for mode, rect in self._clip.items():   # a Rectangle clip is frozen when set: set it again
            for a in self._clipped[mode]:
                a.set_clip_path(rect)
        self._divider.set_xdata([x, x])
        self._knob.set_xdata([x])
        self._chip_before.set_visible(self._split > .08)
        self._chip_after.set_visible(self._split < .92)
        if draw:
            self.canvas.draw_idle()

    # ------------------------------------------------------------------ mouse
    def _on_press(self, event) -> None:
        if self._ax is None or event.inaxes is not self._ax or event.xdata is None or event.button != 1:
            return
        self._drag = True
        self.set_split((event.xdata + .5) / self._result["shape"][1])

    def _on_motion(self, event) -> None:
        if self._ax is None or self._result is None:
            return
        h, w = self._result["shape"]
        if self._drag and event.xdata is not None:
            self.set_split((event.xdata + .5) / w)
            return
        if event.inaxes is not self._ax or event.xdata is None:
            self.lbl_hover.setText(" ")
            return
        mode = "before" if (event.xdata + .5) / w < self._split else "after"
        xs, ys = self._result["grid"]
        i = int(np.clip(round(event.ydata / _GRID), 0, ys.size - 1))
        j = int(np.clip(round(event.xdata / _GRID), 0, xs.size - 1))
        rec = self._result[mode]
        src = self._result.get("smile_source")
        if src == "measured":
            dx = f"dx {rec['dx'][i, j]:+.3f}"
        elif src in ("model", "untraced") and mode == "before" and rec.get("model_lines"):
            dx = f"dx {rec['dx'][i, j]:+.3f} (model)"
        else:
            dx = "dx n/a"
        dy = f"dy {rec['dy'][i, j]:+.3f}" if self._result["stats"]["keystone"].get(mode, {}).get("n") else "dy n/a"
        self.lbl_hover.setText(f"{mode.upper():6s} col {event.xdata:.0f}  row {event.ydata:.0f}   "
                               f"{dx}  {dy} px (true scale)")

    def _on_release(self, _event) -> None:
        self._drag = False

    # ------------------------------------------------------------------ files
    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open model file", "", "Smile model (*.npz);;All (*.*)")
        if path:
            self.open_model(path)

    def _on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export result as PNG", "Smile_Keystone_Result.png",
                                              "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        self.canvas.fig.savefig(path, format="png", dpi=300, facecolor=self.canvas.fig.get_facecolor())

    def closeEvent(self, event):  # noqa: N802
        if self._worker is not None:
            self._worker.wait()
        super().closeEvent(event)
