# widgets/zoom_pan.py
"""Zoom and pan mixin for image canvas widgets.

Usage:
    class MyCanvas(QWidget, ZoomPanMixin):
        def __init__(self):
            super().__init__()
            self._init_zoom_pan()

        def _zp_image_size(self):
            return self._img_w, self._img_h

        def paintEvent(self, event):
            ox, oy, sx, sy = self._zp_transform()
            # draw image at (ox, oy) with scale (sx, sy)

        def wheelEvent(self, event):
            self._zp_wheel_event(event)

        def mousePressEvent(self, event):
            if not self._zp_mouse_press(event):
                # handle left click ...
                pass

        def mouseMoveEvent(self, event):
            if not self._zp_mouse_move(event):
                pass

        def mouseReleaseEvent(self, event):
            if not self._zp_mouse_release(event):
                pass
"""

from __future__ import annotations

from PyQt5.QtCore import Qt


class ZoomPanMixin:
    """Mixin that adds zoom (wheel) and pan (middle-drag) to any widget.

    Coordinates model:
        base_scale = min(widget_w / img_w, widget_h / img_h)
        scale = base_scale * zoom_level
        ox = (widget_w - img_w * scale) / 2 + pan_x
        oy = (widget_h - img_h * scale) / 2 + pan_y

        widget (wx, wy) → image (ix, iy):
            ix = (wx - ox) / scale
            iy = (wy - oy) / scale

        image (ix, iy) → widget (wx, wy):
            wx = ox + ix * scale
            wy = oy + iy * scale
    """

    # ── init ──────────────────────────────────────────────

    def _init_zoom_pan(self):
        """Call in __init__ after super().__init__()."""
        self._zp_zoom: float = 1.0       # 1.0 = fit-to-window
        self._zp_pan_x: float = 0.0      # offset in widget pixels
        self._zp_pan_y: float = 0.0
        self._zp_zoom_min: float = 0.05   # 5%
        self._zp_zoom_max: float = 100.0  # 10000%
        self._zp_dragging: bool = False
        self._zp_drag_start: tuple[float, float] | None = None
        self._zp_drag_pan_start: tuple[float, float] | None = None

    # ── override point ────────────────────────────────────

    def _zp_image_size(self) -> tuple[int, int]:
        """Return (img_w, img_h). Override in subclass."""
        raise NotImplementedError

    def _zp_on_zoom_changed(self):
        """Called after zoom changes. Override to update UI."""
        pass

    # ── public API ────────────────────────────────────────

    @property
    def zoom_percent(self) -> float:
        return self._zp_zoom * 100.0

    def zoom_fit(self):
        """Reset to fit-to-window."""
        self._zp_zoom = 1.0
        self._zp_pan_x = 0.0
        self._zp_pan_y = 0.0
        self.update()
        self._zp_on_zoom_changed()

    def zoom_actual(self):
        """Zoom to 1:1 actual pixels."""
        iw, ih = self._zp_image_size()
        if iw <= 0 or ih <= 0:
            return
        ww, wh = float(self.width()), float(self.height())
        base = min(ww / iw, wh / ih)
        if base > 0:
            self._zp_zoom = 1.0 / base
            self._zp_pan_x = 0.0
            self._zp_pan_y = 0.0
            self.update()
            self._zp_on_zoom_changed()

    def zoom_in(self):
        """Zoom in 1.25x centered on widget center."""
        self._zp_zoom_at(1.25, self.width() / 2.0, self.height() / 2.0)

    def zoom_out(self):
        """Zoom out 1.25x centered on widget center."""
        self._zp_zoom_at(1.0 / 1.25, self.width() / 2.0, self.height() / 2.0)

    # ── transform ─────────────────────────────────────────

    def _zp_transform(self) -> tuple[float, float, float, float]:
        """Compute (ox, oy, sx, sy) for current zoom/pan state."""
        iw, ih = self._zp_image_size()
        if iw <= 0 or ih <= 0:
            return 0.0, 0.0, 1.0, 1.0

        ww, wh = float(self.width()), float(self.height())
        base = min(ww / iw, wh / ih)
        scale = base * self._zp_zoom

        ox = (ww - iw * scale) / 2.0 + self._zp_pan_x
        oy = (wh - ih * scale) / 2.0 + self._zp_pan_y

        return ox, oy, scale, scale

    # ── coordinate mapping ────────────────────────────────

    def _zp_map_widget_to_image(self, wx: int, wy: int) -> tuple[int | None, int | None]:
        """Map widget coords to image coords (zoom/pan aware)."""
        iw, ih = self._zp_image_size()
        ox, oy, sx, sy = self._zp_transform()
        ix = (wx - ox) / sx
        iy = (wy - oy) / sy
        if 0 <= ix < iw and 0 <= iy < ih:
            return int(ix), int(iy)
        return None, None

    def _zp_map_image_to_widget(self, ix: int, iy: int) -> tuple[float, float]:
        """Map image coords to widget coords (zoom/pan aware)."""
        ox, oy, sx, sy = self._zp_transform()
        return ox + ix * sx, oy + iy * sy

    # ── event handlers ────────────────────────────────────

    def _zp_wheel_event(self, event):
        """Handle mouse wheel → zoom centered on cursor."""
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self._zp_zoom_at(factor, float(event.x()), float(event.y()))

    def _zp_mouse_press(self, event) -> bool:
        """Handle middle-button press → start pan. Returns True if consumed."""
        if event.button() == Qt.MiddleButton:
            self._zp_dragging = True
            self._zp_drag_start = (float(event.x()), float(event.y()))
            self._zp_drag_pan_start = (self._zp_pan_x, self._zp_pan_y)
            self.setCursor(Qt.ClosedHandCursor)
            return True
        return False

    def _zp_mouse_move(self, event) -> bool:
        """Handle mouse move during pan. Returns True if consumed."""
        if self._zp_dragging and self._zp_drag_start is not None:
            dx = event.x() - self._zp_drag_start[0]
            dy = event.y() - self._zp_drag_start[1]
            self._zp_pan_x = self._zp_drag_pan_start[0] + dx
            self._zp_pan_y = self._zp_drag_pan_start[1] + dy
            self.update()
            return True
        return False

    def _zp_mouse_release(self, event) -> bool:
        """Handle middle-button release → end pan. Returns True if consumed."""
        if event.button() == Qt.MiddleButton:
            self._zp_dragging = False
            self.setCursor(Qt.ArrowCursor)
            return True
        return False

    def _zp_mouse_double_click(self, event) -> bool:
        """Handle double-click → zoom fit. Returns True if consumed."""
        if event.button() == Qt.LeftButton:
            self.zoom_fit()
            return True
        return False

    # ── internal ──────────────────────────────────────────

    def _zp_zoom_at(self, factor: float, cx: float, cy: float):
        """Zoom by factor, keeping widget point (cx, cy) fixed."""
        old_zoom = self._zp_zoom
        new_zoom = max(
            self._zp_zoom_min,
            min(self._zp_zoom_max, old_zoom * factor),
        )
        if abs(new_zoom - old_zoom) < 1e-12:
            return

        iw, ih = self._zp_image_size()
        if iw <= 0 or ih <= 0:
            return

        ww, wh = float(self.width()), float(self.height())
        base = min(ww / iw, wh / ih)

        old_scale = base * old_zoom
        new_scale = base * new_zoom

        # current origin
        old_ox = (ww - iw * old_scale) / 2.0 + self._zp_pan_x
        old_oy = (wh - ih * old_scale) / 2.0 + self._zp_pan_y

        # image point under cursor
        ix = (cx - old_ox) / old_scale
        iy = (cy - old_oy) / old_scale

        # new origin so same image point stays under cursor
        new_ox = cx - ix * new_scale
        new_oy = cy - iy * new_scale

        self._zp_pan_x = new_ox - (ww - iw * new_scale) / 2.0
        self._zp_pan_y = new_oy - (wh - ih * new_scale) / 2.0
        self._zp_zoom = new_zoom

        self.update()
        self._zp_on_zoom_changed()