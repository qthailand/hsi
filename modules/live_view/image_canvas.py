#widgets/live_image_canvas.py
"""Live image canvas with a selectable row, focus-ROI band and floating buttons.

Data conventions:
    Frames are 2-D ``(rows, columns)`` = ``(slit, wavelength)`` arrays (a 3-D
    ``(rows, columns, C)`` array uses channel 0), shown after conversion to
    uint8 by a :class:`~modules.live_view.mapping.DisplayMapper`.  ``selected_y``
    and the ROI are image ROW indices (0-based), not widget pixels.

Threading:
    GUI thread only.  Call ``set_array`` from the thread that owns the widget
    (use a queued signal when frames arrive from a camera thread).

Example:
    >>> import numpy as np
    >>> from modules.live_view import LiveImageCanvas               # doctest: +SKIP
    >>> canvas = LiveImageCanvas()                                  # doctest: +SKIP
    >>> canvas.set_array(np.zeros((512, 640), np.uint16))           # doctest: +SKIP
    >>> canvas.y_changed.connect(print)                             # doctest: +SKIP
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal, QRect, QRectF, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import QLabel, QSizePolicy, QPushButton, QGraphicsDropShadowEffect

from modules.image import raw_2d_frame
from modules.live_view.mapping import DisplayMapper
import numpy as np
FOCUS_ROI_DEFAULT_HEIGHT = 32
FOCUS_ROI_MIN_HEIGHT = 20

_OVERLAY_BTN_STYLE = """
QPushButton {
    background: rgba(0, 0, 0, 160);
    color: #e0e0e0;
    border: 1px solid rgba(255, 255, 255, 40);
    border-radius: 4px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 600;
}
QPushButton:hover {
    background: rgba(0, 0, 0, 200);
    border: 1px solid rgba(255, 255, 255, 80);
}
QPushButton:disabled {
    color: #666;
    background: rgba(0, 0, 0, 100);
    border: 1px solid rgba(255, 255, 255, 20);
}
"""

_OVERLAY_BTN_OFFSET_X = 8
_OVERLAY_BTN_OFFSET_Y = 8
_OVERLAY_BTN_SPACING = 4


class LiveImageCanvas(QLabel):
    """``QLabel`` that shows the live sensor frame and lets the user pick a row.

    The image is stretched to fill the widget (aspect ratio is not kept).  A red
    line marks ``selected_y`` and a yellow rectangle marks the focus ROI (a band of
    rows ``focus_roi_height`` high centred on ``selected_y``).  Left-click or drag
    selects the row.  Optionally a gold reference band can be drawn
    (:meth:`set_reference_roi`) and floating buttons placed on the image
    (:meth:`add_overlay_button`).  Takes no constructor arguments.

    Signals:
        y_changed(int): new ``selected_y`` (image row) when the USER changes it by
            clicking/dragging.  Not emitted by :meth:`set_selected_y`.

    Attributes:
        selected_y: Currently selected image row (int).
        focus_roi_height: Height of the focus ROI in rows (min 20, default 32).
    """

    y_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()

        self.setText("No Image")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self._pixmap_original: QPixmap | None = None
        self._image_shape: tuple[int, int] | None = None
        self._draw_rect = QRect()

        self.selected_y = 0
        self.focus_roi_height = FOCUS_ROI_DEFAULT_HEIGHT
        self._dragging = False

        self._mapper = DisplayMapper()

        # Overlay buttons
        self._overlay_buttons: list[QPushButton] = []

    # ── overlay buttons ──────────────────────────────────────────

    def add_overlay_button(self, text: str) -> QPushButton:
        """Create a floating button stacked at the top-left of the image.

        Args:
            text: Button caption.

        Returns:
            QPushButton: The new button (child of the canvas); connect its ``clicked``.
        """
        btn = QPushButton(text, self)
        btn.setStyleSheet(_OVERLAY_BTN_STYLE)
        btn.setCursor(Qt.ArrowCursor)
        btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn.adjustSize()
        btn.show()
        self._overlay_buttons.append(btn)
        self._position_overlay_buttons()
        return btn

    def _position_overlay_buttons(self):
        """Stack overlay buttons vertically at the top-left of the draw rect."""
        if not self._overlay_buttons:
            return

        y = self._draw_rect.top() + _OVERLAY_BTN_OFFSET_Y

        for btn in self._overlay_buttons:
            if not btn.isVisible():
                continue
            x = self._draw_rect.left() + _OVERLAY_BTN_OFFSET_X
            btn.move(x, y)
            y += btn.sizeHint().height() + _OVERLAY_BTN_SPACING

    # ── display mapper ───────────────────────────────────────────

    def set_display_mapper(self, mapper: DisplayMapper):
        """Choose how raw frames are converted to 8 bit.

        Args:
            mapper: A ``DisplayMapper`` (for example from ``DisplayControls.mapper_changed``).
                Applies to the next :meth:`set_array`; the current image is not redrawn.
        """
        self._mapper = mapper

    # ── frame rendering ──────────────────────────────────────────

    def set_array(self, frame: np.ndarray):
        """Display a new frame.

        Args:
            frame: 2-D ``(rows, columns)`` array (3-D uses channel 0), any integer
                dtype; ``None`` is ignored.  Not modified or kept (an 8-bit copy is made).

        Raises:
            ValueError: if ``frame`` is neither 2-D nor 3-D.

        Note:
            Clamps ``selected_y`` and ``focus_roi_height`` to the new image height.
        """
        if frame is None:
            return

        frame2d = raw_2d_frame(frame)
        img8 = self._mapper.to_uint8(frame2d)

        h, w = img8.shape
        self._image_shape = (h, w)

        if self.selected_y >= h:
            self.selected_y = max(0, h - 1)

        self.focus_roi_height = max(
            FOCUS_ROI_MIN_HEIGHT,
            min(int(self.focus_roi_height), h),
        )

        qimg = QImage(img8.data, w, h, w, QImage.Format_Grayscale8).copy()
        self._pixmap_original = QPixmap.fromImage(qimg)
        self._update_scaled_pixmap()
        self.update()

    # ── resize / scale ───────────────────────────────────────────

    def resizeEvent(self, event):
        """Qt handler: rescale the pixmap and re-place the overlay buttons."""
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._pixmap_original is None:
            return

        scaled = self._pixmap_original.scaled(
            self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation,
        )

        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2

        self._draw_rect = QRect(x, y, scaled.width(), scaled.height())
        self.setPixmap(scaled)

        self._position_overlay_buttons()

    # ── paint ────────────────────────────────────────────────────

    def set_reference_roi(self, band):
        """Highlight a band of rows as the reference region.

        Args:
            band: ``(start_row, end_row, image_height)`` or ``None`` to remove it.  The
                band is drawn only while the shown image has exactly ``image_height`` rows.
        """
        self._reference_roi = band
        self.update()

    def paintEvent(self, event):
        """Qt handler: draw the image plus reference band, focus ROI and selected-row line."""
        super().paintEvent(event)

        if self._image_shape is None or self._draw_rect.isNull():
            return

        painter = QPainter(self)

        band = getattr(self, '_reference_roi', None)
        if band is not None and band[2] == self._image_shape[0]:
            start, end, height = band
            top = self._draw_rect.top() + start * self._draw_rect.height() / height
            bottom = self._draw_rect.top() + end * self._draw_rect.height() / height
            rect = QRectF(self._draw_rect.left(), top, self._draw_rect.width(), bottom-top)
            painter.fillRect(rect, QColor(255, 215, 0, 65))
            painter.setPen(QPen(QColor('#FFD700'), 2, Qt.DashLine))
            painter.drawRect(rect)

        # ROI rectangle
        y1, y2 = self.focus_roi_range()
        widget_y1 = self._image_y_to_widget_y(y1)
        widget_y2 = self._image_y_to_widget_y(y2)

        roi_pen = QPen(Qt.yellow)
        roi_pen.setWidth(2)
        painter.setPen(roi_pen)
        painter.drawRect(
            self._draw_rect.left(),
            widget_y1,
            self._draw_rect.width(),
            max(1, widget_y2 - widget_y1),
        )

        # Selected Y line
        line_pen = QPen(Qt.red)
        line_pen.setWidth(2)
        painter.setPen(line_pen)
        line_y = self._image_y_to_widget_y(self.selected_y)
        painter.drawLine(
            self._draw_rect.left(), line_y,
            self._draw_rect.right(), line_y,
        )

        painter.end()

    # ── mouse ────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        """Qt handler: left press starts row selection."""
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._set_y_from_mouse(event.pos().y())

    def mouseMoveEvent(self, event):
        """Qt handler: while the left button is held, follow the mouse row."""
        if self._dragging:
            self._set_y_from_mouse(event.pos().y())

    def mouseReleaseEvent(self, event):
        """Qt handler: left release ends row selection."""
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self._set_y_from_mouse(event.pos().y())

    def _set_y_from_mouse(self, widget_y: int):
        if self._image_shape is None or self._draw_rect.isNull():
            return

        widget_y = max(
            self._draw_rect.top(),
            min(self._draw_rect.bottom(), widget_y),
        )

        img_y = self._widget_y_to_image_y(widget_y)
        h, _ = self._image_shape
        img_y = max(0, min(h - 1, img_y))

        if img_y != self.selected_y:
            self.selected_y = img_y
            self.y_changed.emit(self.selected_y)
            self.update()

    # ── coordinate mapping ───────────────────────────────────────

    def _widget_y_to_image_y(self, widget_y: int) -> int:
        h, _ = self._image_shape
        rel_y = widget_y - self._draw_rect.top()
        scale_y = h / max(1, self._draw_rect.height())
        return int(rel_y * scale_y)

    def _image_y_to_widget_y(self, image_y: int) -> int:
        h, _ = self._image_shape
        scale_y = self._draw_rect.height() / max(1, h)
        return int(self._draw_rect.top() + image_y * scale_y)

    # ── public setters ───────────────────────────────────────────

    def set_selected_y(self, y: int):
        """Select an image row programmatically (no ``y_changed`` signal).

        Args:
            y: Row index; clamped to ``0 .. rows-1`` once an image is shown.
        """
        if self._image_shape is None:
            self.selected_y = int(y)
            self.update()
            return

        h, _ = self._image_shape
        self.selected_y = max(0, min(h - 1, int(y)))
        self.update()

    def set_focus_roi_height(self, height: int):
        """Set the focus-ROI height.

        Args:
            height: Rows; clamped to ``[20, image_rows]`` (only the lower bound before
                the first frame).
        """
        height = int(height)
        if self._image_shape is None:
            self.focus_roi_height = max(FOCUS_ROI_MIN_HEIGHT, height)
            self.update()
            return

        h, _ = self._image_shape
        self.focus_roi_height = max(FOCUS_ROI_MIN_HEIGHT, min(height, h))
        self.update()

    def focus_roi_range(self) -> tuple[int, int]:
        """Rows covered by the focus ROI.

        Returns:
            tuple[int, int]: ``(y1, y2)``, start inclusive, end exclusive, kept inside
            the image; ``(0, 1)`` before the first frame.
        """
        if self._image_shape is None:
            return 0, 1

        h, _ = self._image_shape
        roi_h = max(FOCUS_ROI_MIN_HEIGHT, min(int(self.focus_roi_height), h))

        half = roi_h // 2
        y1 = self.selected_y - half
        y2 = y1 + roi_h

        if y1 < 0:
            y1, y2 = 0, roi_h
        if y2 > h:
            y2, y1 = h, h - roi_h

        y1 = max(0, y1)
        y2 = max(y1 + 1, y2)
        return int(y1), int(y2)

    def full_height(self):
        """Make the focus ROI span the whole image height (no-op before the first frame)."""
        if self._image_shape is None:
            return
        h, _ = self._image_shape
        self.focus_roi_height = h
        self.update()
