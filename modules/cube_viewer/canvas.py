# widgets/cube_canvas.py
"""Single-band image canvas with click-to-spectrum and zoom/pan support.

Data conventions:
    The canvas shows one 2-D band image ``(rows, cols)`` = ``(slit, scan)`` of any
    numeric dtype.  It is auto-stretched to 8 bit between the 1st and 99th
    percentile.  Pixel coordinates are ``(ix, iy)`` = ``(column, row)`` of that array.

Threading:
    GUI thread only.

Example:
    >>> import numpy as np
    >>> from modules.cube_viewer.canvas import CubeImageCanvas   # doctest: +SKIP
    >>> canvas = CubeImageCanvas()                                # doctest: +SKIP
    >>> canvas.set_array(np.random.rand(64, 96))                  # doctest: +SKIP
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush
from PyQt5.QtWidgets import QWidget, QSizePolicy

from modules.ui_kit.zoom_pan import ZoomPanMixin


class CubeImageCanvas(QWidget, ZoomPanMixin):
    """Display a 2D band image with click-to-spectrum and zoom/pan.

    Zoom and pan come from ``modules.ui_kit.zoom_pan.ZoomPanMixin`` (mouse wheel
    zooms, middle button pans, double click is handled by the mixin).  Clicked
    points can be marked with coloured dots (tab20 palette, labelled ``(ix,iy)``);
    the host decides which points are shown via ``add_click_point`` /
    ``clear_click_points``.  Construct with no arguments; GUI thread only.

    Signals:
        pixel_clicked(int, int): ``(ix, iy)`` image pixel (column, row) under the
            cursor, emitted on a left click inside the image.
        zoom_changed(float): zoom in percent, emitted whenever the zoom changes.
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
        """Show a new band image.

        Args:
            arr: 2-D ``(rows, cols)`` array of any numeric dtype.  Held by
                reference (the caller keeps ownership); the 8-bit display copy
                is stretched between the 1st and 99th percentile.
        """
        self._array = arr
        self._update_pixmap()
        self.update()

    def set_display_mapper(self, mapper):
        """Store a display mapper and refresh the pixmap.

        Args:
            mapper: A ``modules.live_view.mapping.DisplayMapper`` (or ``None``).
                Note: the current rendering still uses the percentile stretch
                and does not consult the mapper.
        """
        self._display_mapper = mapper
        self._update_pixmap()
        self.update()

    def add_click_point(self, ix: int, iy: int):
        """Mark an image pixel with a coloured dot and its ``(ix,iy)`` label.

        Args:
            ix: Column of the image array (pixels).
            iy: Row of the image array (pixels).
        """
        self._click_points.append((ix, iy))
        self.update()

    def clear_click_points(self):
        """Remove all dots added with ``add_click_point``."""
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
        """Paint the zoomed image and the click dots (Qt override)."""
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
        """Zoom with the mouse wheel (Qt override, delegates to the zoom mixin)."""
        self._zp_wheel_event(event)

    def mouseDoubleClickEvent(self, event):
        """Forward double clicks to the zoom mixin (Qt override)."""
        if self._zp_mouse_double_click(event):
            return

    def mousePressEvent(self, event):
        """Start a pan (middle button) or emit ``pixel_clicked`` (left button); Qt override."""
        # ★ middle button → pan
        if self._zp_mouse_press(event):
            return

        # ★ left button → pixel click
        if event.button() == Qt.LeftButton and self._pixmap is not None:
            ix, iy = self._zp_map_widget_to_image(event.x(), event.y())
            if ix is not None and iy is not None:
                self.pixel_clicked.emit(ix, iy)

    def mouseMoveEvent(self, event):
        """Continue a pan drag (Qt override, delegates to the zoom mixin)."""
        if self._zp_mouse_move(event):
            return

    def mouseReleaseEvent(self, event):
        """End a pan drag (Qt override, delegates to the zoom mixin)."""
        if self._zp_mouse_release(event):
            return