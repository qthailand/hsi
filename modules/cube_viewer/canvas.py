# widgets/cube_canvas.py
"""Single-band image canvas with click-to-spectrum, ROI drawing and zoom/pan support.

Data conventions:
    The canvas shows one 2-D band image ``(rows, cols)`` = ``(slit, scan)`` of any
    numeric dtype.  It is auto-stretched to 8 bit between the 1st and 99th
    percentile.  Pixel coordinates are ``(ix, iy)`` = ``(column, row)`` of that array.
    Rectangle / lasso tools and record overlays come from
    ``modules.cube_viewer.roi_draw.RoiDrawMixin`` (float image-corner coordinates).

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
from matplotlib import colormaps
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush
from PyQt5.QtWidgets import QWidget, QSizePolicy

from modules.cube_preview.widget import TAB20_QCOLORS
from modules.cube_viewer.roi_draw import RoiDrawMixin
from modules.live_view.mapping import DisplayMode

#: The colormaps of matplotlib's "Choosing Colormaps" page, by its sections (names this
#: matplotlib does not have are dropped).  ``None`` (no colormap) is the original grey image.
_COLORMAP_PAGE = (
    ("Perceptually Uniform Sequential", ("viridis", "plasma", "inferno", "magma", "cividis")),
    ("Sequential", ("Greys", "Purples", "Blues", "Greens", "Oranges", "Reds", "YlOrBr",
                    "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu", "PuBu", "YlGnBu",
                    "PuBuGn", "BuGn", "YlGn")),
    ("Sequential (2)", ("binary", "gist_yarg", "gist_gray", "gray", "bone", "pink", "spring",
                        "summer", "autumn", "winter", "cool", "Wistia", "hot", "afmhot",
                        "gist_heat", "copper")),
    ("Diverging", ("PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu", "RdYlGn",
                   "Spectral", "coolwarm", "bwr", "seismic", "berlin", "managua", "vanimo")),
    ("Cyclic", ("twilight", "twilight_shifted", "hsv")),
    ("Qualitative", ("Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3",
                     "tab10", "tab20", "tab20b", "tab20c")),
    ("Miscellaneous", ("flag", "prism", "ocean", "gist_earth", "terrain", "gist_stern",
                       "gnuplot", "gnuplot2", "CMRmap", "cubehelix", "brg", "gist_rainbow",
                       "rainbow", "jet", "turbo", "nipy_spectral", "gist_ncar")),
)
COLORMAP_GROUPS = tuple((title, tuple(n for n in names if n in colormaps))
                        for title, names in _COLORMAP_PAGE)


def colormap_lut(name: str, reverse: bool = False) -> np.ndarray:
    """``(256, 3)`` uint8 RGB table of a matplotlib colormap (``reverse``: the ``_r`` order)."""
    rgba = colormaps[name](np.linspace(0.0, 1.0, 256))
    lut = np.round(rgba[:, :3] * 255.0).astype(np.uint8)
    return np.ascontiguousarray(lut[::-1] if reverse else lut)
from modules.ui_kit.zoom_pan import ZoomPanMixin


class CubeImageCanvas(QWidget, ZoomPanMixin, RoiDrawMixin):
    """Display a 2D band image with click-to-spectrum, ROI drawing and zoom/pan.

    Zoom and pan come from ``modules.ui_kit.zoom_pan.ZoomPanMixin`` (mouse wheel
    zooms, middle button pans, double click is handled by the mixin).  Clicked
    points can be marked with coloured dots (tab20 palette, labelled ``(ix,iy)``);
    the host decides which points are shown via ``add_click_point`` /
    ``clear_click_points``.  ``set_tool('rect' | 'lasso')`` switches the left button
    to drawing an area (``roi_drawn``), ``set_roi_overlays`` draws spectrum records
    and ``set_class_overlay`` a class map (see ``modules.cube_viewer.roi_draw``).
    Construct with no arguments; GUI thread only.

    Signals:
        pixel_clicked(int, int): ``(ix, iy)`` image pixel (column, row) under the
            cursor, emitted on a left click inside the image (point tool only).
        zoom_changed(float): zoom in percent, emitted whenever the zoom changes.
        roi_drawn(str, object): ``(kind, geometry)`` of a finished rectangle / lasso.
        roi_discarded(str): a drawing was dropped; the reason, for a status bar.
    """

    pixel_clicked = pyqtSignal(int, int)
    zoom_changed = pyqtSignal(float)

    # point tags keep this canvas's legacy placement: right of the dot, never flipped or
    # clamped (the RGB view keeps its own flip / clamp rule)
    _roi_point_tag_inside = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_zoom_pan()
        self._init_roi_draw()

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
                RAW maps integer cubes linearly from ``0 .. 2**bit_depth - 1`` to
                0..255 (true brightness); ENHANCE, ``None`` and float cubes use the
                1-99 % percentile stretch.
        """
        self._display_mapper = mapper
        self._update_pixmap()
        self.update()

    def set_colormap(self, name=None, reverse: bool = False) -> None:
        """Colour the band image with a matplotlib colormap (after the RAW/Enhance mapping).

        Args:
            name: A matplotlib colormap name (see ``COLORMAP_GROUPS``), or ``None`` for no
                colormap: the original grey image, bit for bit (``reverse`` is ignored).
            reverse: Use the reversed map (``<name>_r``).

        Raises:
            KeyError: ``name`` is not a matplotlib colormap.
        """
        if name is not None and name not in colormaps:
            raise KeyError(f"unknown colormap {name!r}")
        self._colormap = (name, bool(reverse) and name is not None)
        self._lut = None if name is None else colormap_lut(name, reverse)
        self._update_pixmap()
        self.update()

    @property
    def colormap(self) -> tuple:
        """``(name or None, reverse)`` of the current colormap; ``(None, False)`` by default."""
        return getattr(self, "_colormap", (None, False))

    def _to_pixmap(self, arr8: np.ndarray) -> QPixmap:
        """QPixmap of a 2-D uint8 image: grey, or through the colormap table."""
        h, w = arr8.shape
        lut = getattr(self, "_lut", None)
        if lut is None:
            arr8 = np.ascontiguousarray(arr8)
            return QPixmap.fromImage(QImage(arr8.data, w, h, w, QImage.Format_Grayscale8).copy())
        rgb = np.ascontiguousarray(lut[arr8])
        return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())

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

        mapper = self._display_mapper
        if (mapper is not None and mapper.mode is DisplayMode.RAW
                and np.issubdtype(self._array.dtype, np.integer)):
            self._pixmap = self._to_pixmap(mapper.to_uint8(self._array))
            return

        arr = self._array.astype(np.float32)
        lo = np.percentile(arr, 1.0)
        hi = np.percentile(arr, 99.0)
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

        self._pixmap = self._to_pixmap(arr)

    # ── painting ──────────────────────────────────────────

    def paintEvent(self, event):
        """Paint the zoomed image, class overlay, click dots, records and drawing (Qt override)."""
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
            if sx > 1.0 or sy > 1.0:
                # Zoomed in: the painter scales only what lands on screen (a full
                # scaled() copy cost ~60 ms and hundreds of MB per paint at 1000 %).
                painter.drawPixmap(QRectF(int(ox), int(oy), scaled_w, scaled_h),
                                   self._pixmap, QRectF(0, 0, img_w, img_h))
            else:
                # Fit / zoomed out: unchanged (scaled()'s area filter for downscaling).
                scaled = self._pixmap.scaled(
                    scaled_w, scaled_h,
                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation,
                )
                painter.drawPixmap(int(ox), int(oy), scaled)

        self._roi_paint_class_overlay(painter)

        # draw click dots + labels
        colors = TAB20_QCOLORS
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

        self._roi_paint_records(painter)
        self._roi_paint_drawing(painter)
        painter.end()

    # ── mouse events ──────────────────────────────────────

    def wheelEvent(self, event):
        """Zoom with the mouse wheel (Qt override, delegates to the zoom mixin)."""
        self._zp_wheel_event(event)

    def mouseDoubleClickEvent(self, event):
        """A double click closes a polygon being drawn, else zooms to fit (Qt override)."""
        if self._roi_mouse_double_click(event):
            return
        if self._zp_mouse_double_click(event):
            return

    def mousePressEvent(self, event):
        """Start a pan (middle button), a drawing (left, rect / lasso tool) or emit
        ``pixel_clicked`` (left, point tool); Qt override."""
        # ★ middle button → pan
        if self._zp_mouse_press(event):
            return

        # ★ left button, rect / lasso tool → start drawing
        if self._roi_mouse_press(event):
            return

        # ★ left button → pixel click
        if event.button() == Qt.LeftButton and self._pixmap is not None:
            ix, iy = self._zp_map_widget_to_image(event.x(), event.y())
            if ix is not None and iy is not None:
                self.pixel_clicked.emit(ix, iy)

    def mouseMoveEvent(self, event):
        """Continue a pan drag, else a drawing (Qt override)."""
        if self._zp_mouse_move(event):
            return
        self._roi_mouse_move(event)

    def mouseReleaseEvent(self, event):
        """End a pan drag or finish a drawing (Qt override)."""
        if self._zp_mouse_release(event):
            self._roi_restore_cursor()
            return
        self._roi_mouse_release(event)

    def keyPressEvent(self, event):
        """Esc cancels a drawing in progress (Qt override)."""
        if self._roi_key_press(event):
            return
        super().keyPressEvent(event)