"""Visual spatial-row selector; hardware requests are signals to the host app.

Data conventions:
    The preview image given to ``accept_preview`` is ``uint8`` RGB ``(slit, scan, 3)``
    (see ``modules.reference_roi.core.compose_preview``). Selections are row ranges
    ``[start, end)`` in image rows, with ``end`` exclusive. The ``context`` dict must
    contain every ``COMPATIBILITY_KEYS`` entry.

Threading:
    GUI thread only. The widget owns no threads and no hardware: the host answers
    ``previewRequested`` by scanning and then calling ``accept_preview`` or
    ``fail_preview``.

Example:
    >>> from PyQt5.QtWidgets import QApplication
    >>> app = QApplication.instance() or QApplication([])
    >>> from modules.reference_roi import ReferenceROIWidget
    >>> w = ReferenceROIWidget()
    >>> _ = w.previewRequested.connect(lambda: print('scan please'))
    >>> w.scan_button.click()
    scan please
"""
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSpinBox, QLineEdit
from .core import ReferenceROIProfile

class SpatialSelectionCanvas(QLabel):
    """Image canvas on which a vertical row range is dragged with the left mouse button.

    Signals:
        rangeChanged(int, int): ``(start, end)`` image rows, ``end`` exclusive.
    """
    rangeChanged = pyqtSignal(int, int)
    def __init__(self):
        """Create an empty canvas showing a hint text."""
        super().__init__()
        self.setMinimumSize(420, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setText('Run Preview Scan to display the reference and sample')
        self._source = None; self._pixmap = None; self._rect = QRectF(); self._range = None; self._anchor = None

    def set_image(self, image):
        """Show ``image`` and clear the selection.

        Args:
            image: ``uint8`` array ``(rows, cols, 3)``; copied into a pixmap.

        Raises:
            ValueError: image is not ``uint8`` RGB.
        """
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8: raise ValueError('Expected uint8 RGB preview')
        self._source = image
        h,w,_ = image.shape
        self._pixmap = QPixmap.fromImage(QImage(image.data,w,h,image.strides[0],QImage.Format_RGB888).copy())
        self._range = None; self.update()

    def set_range(self, start, end):
        """Highlight rows ``[start, end)`` (does not emit ``rangeChanged``)."""
        self._range = (start, end); self.update()

    def paintEvent(self, event):
        """Draw the aspect-preserving image and the highlighted band."""
        if self._pixmap is None:
            super().paintEvent(event); return
        scaled = self._pixmap.scaled(self.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self._rect = QRectF((self.width()-scaled.width())/2,(self.height()-scaled.height())/2,scaled.width(),scaled.height())
        painter = QPainter(self); painter.drawPixmap(self._rect.topLeft().toPoint(),scaled)
        if self._range:
            start,end = self._range; scale = self._rect.height()/self._source.shape[0]
            band = QRectF(self._rect.left(),self._rect.top()+start*scale,self._rect.width(),(end-start)*scale)
            painter.fillRect(band,QColor(255,215,0,65));painter.setPen(QPen(QColor('#FFD700'),2));painter.drawRect(band)
        painter.end()

    def _row(self, position):
        h = self._source.shape[0]
        return min(h-1,max(0,int((position.y()-self._rect.top())*h/max(1,self._rect.height()))))

    def mousePressEvent(self, event):
        """Start a drag selection inside the image."""
        if event.button()==Qt.LeftButton and self._source is not None and self._rect.contains(event.pos()):
            self._anchor = self._row(event.pos()); self._set_drag(event.pos())

    def mouseMoveEvent(self, event):
        """Extend the drag selection."""
        if self._anchor is not None: self._set_drag(event.pos())

    def mouseReleaseEvent(self, event):
        """Finish the drag selection."""
        if self._anchor is not None: self._set_drag(event.pos()); self._anchor = None

    def _set_drag(self, position):
        row = self._row(position); start,end = min(self._anchor,row),max(self._anchor,row)+1
        self.set_range(start,end); self.rangeChanged.emit(start,end)

class ReferenceROIWidget(QWidget):
    """Select the spatial rows of a white reference on a scanned preview image.

    Inputs (methods): ``accept_preview(image, context)``, ``fail_preview(message)``,
    ``set_busy(bool)``. Outputs (signals): ``previewRequested()``,
    ``cancelRequested()``, ``saveRequested(ReferenceROIProfile)``,
    ``deactivateRequested()``. The host performs the scan and persists the profile
    (e.g. with ``ReferenceROIStore``); the widget never touches hardware or disk.
    Constructible with no arguments; it applies its own dark stylesheet.

    Attributes:
        context (dict | None): camera context of the current preview.
        name, start, end: ``QLineEdit`` / ``QSpinBox`` inputs; ``scan_button``,
            ``cancel_button``, ``clear_button``, ``save_button``: ``QPushButton``.
    """
    previewRequested = pyqtSignal()
    cancelRequested = pyqtSignal()
    saveRequested = pyqtSignal(object)
    deactivateRequested = pyqtSignal()

    def __init__(self):
        """Create the widget with no preview loaded."""
        super().__init__()
        self.setStyleSheet("""
            QWidget { background: #20252d; color: #e6edf3; }
            QLineEdit, QSpinBox { background: #121820; border: 1px solid #455366; border-radius: 5px; padding: 6px; }
            QPushButton { background: #354255; border: 1px solid #53657c; border-radius: 6px; padding: 9px 14px; }
            QPushButton:hover { background: #465a74; }
            QPushButton:disabled { color: #728093; background: #28313d; }
            QLabel { padding: 4px; }
        """)
        self.context = None; self._image = None; self._has_selection = False
        layout = QVBoxLayout(self)
        self.name = QLineEdit('Reference ROI')
        self.name.setPlaceholderText('Profile name')
        layout.addWidget(self.name)
        commands = QHBoxLayout()
        self.scan_button = QPushButton('Preview Scan'); self.scan_button.clicked.connect(self._request_preview)
        self.cancel_button = QPushButton('Stop Preview Scan'); self.cancel_button.setEnabled(False); self.cancel_button.clicked.connect(self.cancelRequested)
        self.clear_button = QPushButton('Deactivate Profile'); self.clear_button.clicked.connect(self.deactivateRequested)
        for button in (self.scan_button,self.cancel_button,self.clear_button): commands.addWidget(button)
        layout.addLayout(commands)
        self.canvas = SpatialSelectionCanvas(); layout.addWidget(self.canvas,stretch=1)
        coordinates = QHBoxLayout(); self.start = QSpinBox(); self.end = QSpinBox()
        for spin in (self.start,self.end): spin.setEnabled(False)
        coordinates.addWidget(QLabel('Y start')); coordinates.addWidget(self.start)
        coordinates.addWidget(QLabel('Y end (exclusive)')); coordinates.addWidget(self.end)
        self.save_button = QPushButton('Apply && Save Profile'); self.save_button.setEnabled(False); self.save_button.clicked.connect(self._save)
        coordinates.addWidget(self.save_button); layout.addLayout(coordinates)
        self.status = QLabel('Preview Scan uses the stage travel/speed configured in Scan Window. Drag vertically over the centre of the reference.');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.canvas.rangeChanged.connect(self._selection)
        self.start.valueChanged.connect(self._numeric); self.end.valueChanged.connect(self._numeric)

    def _request_preview(self):
        self.set_busy(True); self.previewRequested.emit()

    def set_busy(self, busy):
        """Enable/disable controls while the host runs a preview scan.

        Args:
            busy: True during a scan (disables editing, enables the stop button).
        """
        self.canvas.setEnabled(not busy)
        for spin in (self.start, self.end): spin.setEnabled(not busy and self.context is not None)
        self.clear_button.setEnabled(not busy)
        self.scan_button.setEnabled(not busy); self.cancel_button.setEnabled(busy)
        self.save_button.setEnabled(not busy and self._has_selection)

    def accept_preview(self, image, context):
        """Display a finished preview scan and enable range selection.

        Args:
            image: ``uint8`` RGB ``(slit, scan, 3)``; kept by reference.
            context: Dict with the ``COMPATIBILITY_KEYS`` entries; copied.
        """
        self.context = dict(context); self._image = image; self._has_selection = False
        self.canvas.set_image(image); h = image.shape[0]
        for spin in (self.start,self.end): spin.blockSignals(True);spin.setEnabled(True);spin.setRange(0,h)
        self.start.setValue(0);self.end.setValue(h)
        for spin in (self.start,self.end): spin.blockSignals(False)
        self.set_busy(False);self.save_button.setEnabled(False)
        self.status.setText(f'Preview: {h} spatial rows × {image.shape[1]} scan frames. Select the centre of the reference; yellow band spans the scan direction.')

    def fail_preview(self, message):
        """Leave busy state and show ``message`` as status text."""
        self.set_busy(False);self.status.setText(message)

    def _selection(self, start, end):
        for spin,value in ((self.start,start),(self.end,end)):
            spin.blockSignals(True);spin.setValue(value);spin.blockSignals(False)
        self._has_selection = True;self.save_button.setEnabled(self.scan_button.isEnabled())
        self.status.setText(f'Reference rows [{start}, {end}) · {end-start} pixels. The entire yellow horizontal band is used in every frame.')

    def _numeric(self):
        if self.context is None: return
        start,end = self.start.value(),self.end.value()
        valid = 0 <= start < end <= self.context['spatial_height']
        self._has_selection = valid;self.save_button.setEnabled(valid and self.scan_button.isEnabled())
        if valid: self.canvas.set_range(start,end)

    def _save(self):
        if self.context is None or not self._has_selection: return
        try:
            profile=ReferenceROIProfile(self.name.text().strip(),self.start.value(),self.end.value(),dict(self.context)).validate()
            self.saveRequested.emit(profile)
        except Exception as exc: self.status.setText(str(exc))
