# widgets/cube_canvas.py
"""Single-band image canvas with click-to-spectrum and zoom/pan support."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush
from PyQt5.QtWidgets import QWidget, QSizePolicy

from widgets.zoom_pan import ZoomPanMixin


class CubeImageCanvas(QWidget, ZoomPanMixin):
    """Display a 2D band image with click-to-spectrum and zoom/pan.

    Emits:
        pixel_clicked(int, int) — image pixel coordinates on left click
        zoom_changed(float)     — zoom percent after zoom change
    """

    pixel_clicked = pyqtSignal(int, int)
    zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_zoom_pan()

        self._array: np.ndarray | None = None
        self._display_mapper = None
        self._pixmap: QPixmap | None = None

        # click point history
        self._click_points: list[tuple[int, int]] = []

        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("background-color: #0d0d0d;")

    # ── ZoomPanMixin overrides ────────────────────────────

    def _zp_image_size(self) -> tuple[int, int]:
        if self._array is None:
            return 0, 0
        h, w = self._array.shape[:2]
        return w, h

    def _zp_on_zoom_changed(self):
        self.zoom_changed.emit(self.zoom_percent)

    # ── public API ────────────────────────────────────────

    def set_array(self, arr: np.ndarray):
        self._array = arr
        self._update_pixmap()
        self.update()

    def set_display_mapper(self, mapper):
        self._display_mapper = mapper
        self._update_pixmap()
        self.update()

    def add_click_point(self, ix: int, iy: int):
        self._click_points.append((ix, iy))
        self.update()

    def clear_click_points(self):
        self._click_points.clear()
        self.update()

    # ── pixmap ────────────────────────────────────────────

    def _update_pixmap(self):
        if self._array is None:
            self._pixmap = None
            return

        arr = self._array.astype(np.float32)
        lo = np.percentile(arr, 1.0)
        hi = np.percentile(arr, 99.0)
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
        self._pixmap = QPixmap.fromImage(qimg)

    # ── tab20 dot colors ──────────────────────────────────

    _DOT_COLORS = None

    @classmethod
    def _get_dot_colors(cls):
        if cls._DOT_COLORS is None:
            from matplotlib import colormaps
            cmap = colormaps["tab20"]
            cls._DOT_COLORS = []
            for i in range(20):
                r, g, b, a = cmap(i / 20)
                cls._DOT_COLORS.append(
                    QColor(int(r * 255), int(g * 255), int(b * 255))
                )
        return cls._DOT_COLORS

    # ── painting ──────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(13, 13, 13))

        if self._pixmap is None or self._pixmap.isNull():
            painter.end()
            return

        # ★ zoom/pan transform
        ox, oy, sx, sy = self._zp_transform()

        # draw pixmap at zoomed position
        img_w = self._pixmap.width()
        img_h = self._pixmap.height()
        scaled_w = int(img_w * sx)
        scaled_h = int(img_h * sy)

        if scaled_w > 0 and scaled_h > 0:
            scaled = self._pixmap.scaled(
                scaled_w, scaled_h,
                Qt.IgnoreAspectRatio, Qt.SmoothTransformation,
            )
            painter.drawPixmap(int(ox), int(oy), scaled)

        # draw click dots + labels
        colors = self._get_dot_colors()
        for i, (ix, iy) in enumerate(self._click_points):
            color = colors[i % len(colors)]
            cx, cy = self._zp_map_image_to_widget(ix, iy)

            # skip if outside widget
            if cx < -20 or cy < -20 or cx > self.width() + 20 or cy > self.height() + 20:
                continue

            radius = 3
            painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QRectF(
                cx - radius, cy - radius,
                2 * radius, 2 * radius,
            ))

            label = f"({ix},{iy})"
            font = painter.font()
            font.setPixelSize(10)
            painter.setFont(font)
            metrics = painter.fontMetrics()

            tw = metrics.horizontalAdvance(label) + 8
            th = metrics.height() + 4
            lx = cx + radius + 4
            ly = cy - th // 2

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 160)))
            painter.drawRoundedRect(QRectF(lx, ly, tw, th), 3, 3)

            painter.setPen(QPen(QColor(255, 255, 255, 230)))
            painter.drawText(QRectF(lx, ly, tw, th), Qt.AlignCenter, label)

        painter.end()

    # ── mouse events ──────────────────────────────────────

    def wheelEvent(self, event):
        self._zp_wheel_event(event)

    def mouseDoubleClickEvent(self, event):
        if self._zp_mouse_double_click(event):
            return

    def mousePressEvent(self, event):
        # ★ middle button → pan
        if self._zp_mouse_press(event):
            return

        # ★ left button → pixel click
        if event.button() == Qt.LeftButton and self._pixmap is not None:
            ix, iy = self._zp_map_widget_to_image(event.x(), event.y())
            if ix is not None and iy is not None:
                self.pixel_clicked.emit(ix, iy)

    def mouseMoveEvent(self, event):
        if self._zp_mouse_move(event):
            return

    def mouseReleaseEvent(self, event):
        if self._zp_mouse_release(event):
            return