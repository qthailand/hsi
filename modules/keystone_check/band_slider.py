"""Band-window slider: the cube's spectrum with a centre band and two handles that open out from it.

Looks like the levels slider under a histogram: the bars are the band levels (one per band), and
under them two triangle handles sit side by side at the centre band. Drag the left handle to the
left / the right handle to the right to take in more bands on that side (cumulative); drag the
centre marker, click a bar, use the arrow keys or the mouse wheel to move the centre band (the
two side counts are kept). The bands taken in are tinted blue (left side) and red (right side).

Signals:
    changed(int, int, int): ``(centre band, bands on the left, bands on the right)`` whenever the
    window changes (by the user or :meth:`BandWindowSlider.set_window`).

Threading:
    GUI thread only.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import QSizePolicy, QWidget

_BG = QColor("#2b2d31")
_BAR = QColor("#8a8d92")
_BAR_CENTRE = QColor("#ffffff")
_LEFT = QColor("#4fc3f7")
_RIGHT = QColor("#ef5350")
_TEXT = QColor("#9aa0a6")
_TRACK = QColor("#45484e")
_HANDLE_W = 7            # half width of a triangle handle (px)


class BandWindowSlider(QWidget):
    """Spectrum + centre / left / right handles choosing the bands of the fusion preview."""

    changed = pyqtSignal(int, int, int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(118)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setToolTip("Drag the blue ▲ left / red ▲ right to fuse more bands onto the centre band;\n"
                        "drag the white marker, click the spectrum, arrow keys or wheel to move the centre band")
        self._wl = np.zeros(0)
        self._spec = np.zeros(0)
        self._c = 0
        self._nl = 0
        self._nr = 0
        self._max_side = 10 ** 9
        self._drag: Optional[str] = None
        self._press_x = 0.0
        self._hover: Optional[str] = None

    # ── state ──────────────────────────────────────────────────
    def set_bands(self, wavelengths) -> None:
        """New cube: its band *wavelengths* (nm); the spectrum is flat until :meth:`set_spectrum`."""
        self._wl = np.asarray(wavelengths, float)
        self._spec = np.ones(self._wl.size)
        self._c = min(self._c, max(0, self._wl.size - 1))
        self._clip()
        self.update()

    def set_spectrum(self, levels) -> None:
        """Bar heights (one per band, any scale)."""
        lv = np.nan_to_num(np.asarray(levels, float))
        if lv.size == self._wl.size and lv.max() > 0:
            self._spec = lv / lv.max()
            self.update()

    def set_max_side(self, n: int) -> None:
        """Most bands allowed on one side."""
        self._max_side = max(0, int(n))
        self._clip()
        self.update()

    def window(self) -> Tuple[int, int, int]:
        """``(centre band, bands on the left, bands on the right)``."""
        return self._c, self._nl, self._nr

    def set_window(self, centre: Optional[int] = None, n_left: Optional[int] = None,
                   n_right: Optional[int] = None, emit: bool = True) -> None:
        """Set any of the centre band / side counts (clipped to the cube); emits ``changed`` if it moved."""
        old = self.window()
        if centre is not None:
            self._c = int(centre)
        if n_left is not None:
            self._nl = int(n_left)
        if n_right is not None:
            self._nr = int(n_right)
        self._clip()
        self.update()
        if emit and self.window() != old:
            self.changed.emit(*self.window())

    def _clip(self) -> None:
        B = self._wl.size
        if B == 0:
            self._c = self._nl = self._nr = 0
            return
        self._c = int(np.clip(self._c, 0, B - 1))
        self._nl = int(np.clip(self._nl, 0, min(self._c, self._max_side)))
        self._nr = int(np.clip(self._nr, 0, min(B - 1 - self._c, self._max_side)))

    # ── geometry ───────────────────────────────────────────────
    def _areas(self) -> Tuple[QRectF, float, float]:
        """(bar area, track y, handle top y)."""
        m = 2.0 * _HANDLE_W + 4
        h = float(self.height())
        bars = QRectF(m, 4.0, max(10.0, self.width() - 2 * m), h - 4.0 - 44.0)
        track_y = bars.bottom() + 3.0
        return bars, track_y, track_y + 7.0

    def _x(self, edge: float) -> float:
        """x of band boundary *edge* (band b spans edges b .. b + 1)."""
        bars, _, _ = self._areas()
        return bars.left() + edge * bars.width() / max(1, self._wl.size)

    def _band_at(self, x: float) -> int:
        bars, _, _ = self._areas()
        return int(np.floor((x - bars.left()) / bars.width() * max(1, self._wl.size)))

    def _handle_x(self) -> dict:
        return {"left": self._x(self._c - self._nl) - _HANDLE_W,
                "centre": self._x(self._c + 0.5),
                "right": self._x(self._c + 1 + self._nr) + _HANDLE_W}

    # ── painting ───────────────────────────────────────────────
    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        p = QPainter(self)
        p.fillRect(self.rect(), _BG)
        B = self._wl.size
        if B == 0:
            p.setPen(_TEXT)
            p.drawText(self.rect(), Qt.AlignCenter, "open a cube")
            return
        bars, track_y, hy = self._areas()
        lo, hi = self._c - self._nl, self._c + self._nr
        for b in range(B):
            x0, x1 = self._x(b), self._x(b + 1)
            hgt = max(1.0, float(self._spec[b]) * bars.height())
            if b == self._c:
                col = _BAR_CENTRE
            elif lo <= b < self._c:
                col = _LEFT
            elif self._c < b <= hi:
                col = _RIGHT
            else:
                col = _BAR
            p.fillRect(QRectF(x0, bars.bottom() - hgt, max(1.0, x1 - x0), hgt), col)
        # wavelength grid + labels every 100 nm
        p.setFont(QFont(self.font().family(), 8))
        step = 100.0 if (self._wl[-1] - self._wl[0]) > 250 else 50.0
        for nm in np.arange(np.ceil(self._wl.min() / step) * step, self._wl.max() + 1e-6, step):
            b = float(np.interp(nm, self._wl, np.arange(B))) + 0.5
            x = self._x(b)
            p.setPen(QPen(QColor(255, 255, 255, 40), 1))
            p.drawLine(QPointF(x, bars.top()), QPointF(x, bars.bottom()))
            p.setPen(_TEXT)
            p.drawText(QRectF(x - 30, hy + 15, 60, 14), Qt.AlignHCenter | Qt.AlignTop, f"{nm:.0f}")
        # track with the fused range
        p.fillRect(QRectF(bars.left(), track_y, bars.width(), 4), _TRACK)
        p.fillRect(QRectF(self._x(lo), track_y, self._x(self._c) - self._x(lo), 4), _LEFT)
        p.fillRect(QRectF(self._x(self._c + 1), track_y, self._x(hi + 1) - self._x(self._c + 1), 4), _RIGHT)
        p.fillRect(QRectF(self._x(self._c), track_y - 1, max(2.0, self._x(self._c + 1) - self._x(self._c)), 6),
                   _BAR_CENTRE)
        # handles: ▲ left (blue), ▲ right (red), small white centre marker between / above them
        p.setRenderHint(QPainter.Antialiasing, True)
        hx = self._handle_x()
        for name, col in (("left", _LEFT), ("right", _RIGHT)):
            x = hx[name]
            tri = QPolygonF([QPointF(x, hy), QPointF(x - _HANDLE_W, hy + 13), QPointF(x + _HANDLE_W, hy + 13)])
            p.setPen(QPen(QColor("#ffffff") if self._hover == name or self._drag == name else QColor("#111"), 1))
            p.setBrush(col)
            p.drawPolygon(tri)
        x = hx["centre"]
        dia = QPolygonF([QPointF(x, track_y - 6), QPointF(x + 4, track_y - 1), QPointF(x, track_y + 4),
                         QPointF(x - 4, track_y - 1)])
        p.setPen(QPen(QColor("#111"), 1))
        p.setBrush(_BAR_CENTRE)
        p.drawPolygon(dia)
        # read-out
        p.setPen(_TEXT)
        txt = f"{self._wl[self._c]:.1f} nm"
        if self._nl:
            txt = f"{self._wl[lo]:.0f} ◀ " + txt
        if self._nr:
            txt += f" ▶ {self._wl[hi]:.0f}"
        tw = p.fontMetrics().horizontalAdvance(txt) + 10
        box = QRectF(bars.right() - tw, bars.top(), tw, 16)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(30, 31, 34, 220))
        p.drawRoundedRect(box, 3, 3)
        p.setPen(QColor("#e0e0e0"))
        p.drawText(box, Qt.AlignCenter, txt)
        p.end()

    # ── mouse / keys ───────────────────────────────────────────
    def _pick(self, x: float, y: float) -> Optional[str]:
        bars, track_y, hy = self._areas()
        hx = self._handle_x()
        if y >= hy - 2:
            d = {k: abs(x - hx[k]) for k in ("left", "right")}
            k = min(d, key=d.get)
            return k if d[k] <= _HANDLE_W + 3 else None
        if y >= track_y - 7 and abs(x - hx["centre"]) <= 6:
            return "centre"
        if bars.contains(QPointF(x, y)):
            return "bars"
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.button() != Qt.LeftButton or self._wl.size == 0:
            return
        hit = self._pick(event.x(), event.y())
        if hit == "bars":
            self.set_window(centre=self._band_at(event.x()))
            hit = "centre"
        self._drag = hit
        self._press_x = event.x()
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        x = event.x()
        if self._drag is None:
            hov = self._pick(x, event.y())
            if hov != self._hover:
                self._hover = hov
                self.setCursor(Qt.SizeHorCursor if hov in ("left", "right", "centre") else Qt.ArrowCursor)
                self.update()
            return
        if self._drag == "left":
            self.set_window(n_left=self._c - self._band_at(x + _HANDLE_W + 0.5))
        elif self._drag == "right":
            self.set_window(n_right=self._band_at(x - _HANDLE_W - 0.5) - self._c)
        elif self._drag == "centre":
            self.set_window(centre=self._band_at(x))

    def mouseReleaseEvent(self, _event) -> None:  # noqa: N802 - Qt override
        self._drag = None
        self.update()

    def leaveEvent(self, _event) -> None:  # noqa: N802 - Qt override
        self._hover = None
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        step = 1 if event.angleDelta().y() > 0 else -1
        self.set_window(centre=self._c + step)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Left / Right: centre band; with Shift: open / close the right side; with Ctrl: the left side."""
        k = event.key()
        if k not in (Qt.Key_Left, Qt.Key_Right):
            super().keyPressEvent(event)
            return
        d = 1 if k == Qt.Key_Right else -1
        mods = event.modifiers()
        if mods & Qt.ShiftModifier:
            self.set_window(n_right=self._nr + d)
        elif mods & Qt.ControlModifier:
            self.set_window(n_left=self._nl - d)
        else:
            self.set_window(centre=self._c + d)
