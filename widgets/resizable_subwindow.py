from PyQt5.QtCore import Qt, QRect, QEvent, QPoint
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QMdiSubWindow, QWidget


class ResizableMdiSubWindow(QMdiSubWindow):
    MARGIN = 8
    MIN_W = 200
    MIN_H = 100

    _CURSORS = {
        "left":        Qt.SizeHorCursor,
        "right":       Qt.SizeHorCursor,
        "top":         Qt.SizeVerCursor,
        "bottom":      Qt.SizeVerCursor,
        "topleft":     Qt.SizeFDiagCursor,
        "bottomright": Qt.SizeFDiagCursor,
        "topright":    Qt.SizeBDiagCursor,
        "bottomleft":  Qt.SizeBDiagCursor,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._rdir = None
        self._drag_gpos = None
        self._start_geom = None

    def setWidget(self, widget):
        super().setWidget(widget)
        if widget:
            self._install_tracking(widget)

    def _install_tracking(self, widget):
        widget.setMouseTracking(True)
        widget.installEventFilter(self)
        for child in widget.findChildren(QWidget):
            child.setMouseTracking(True)
            child.installEventFilter(self)

    def _direction(self, pos):
        r, m = self.rect(), self.MARGIN
        L = pos.x() < m
        R = pos.x() > r.width() - m
        T = pos.y() < m
        B = pos.y() > r.height() - m
        if L and T:
            return "topleft"
        if R and T:
            return "topright"
        if L and B:
            return "bottomleft"
        if R and B:
            return "bottomright"
        if L:
            return "left"
        if R:
            return "right"
        if T:
            return "top"
        if B:
            return "bottom"
        return None

    def _apply_resize(self, gpos: QPoint):
        delta = gpos - self._drag_gpos
        g = QRect(self._start_geom)
        dx, dy = delta.x(), delta.y()
        if "right" in self._rdir:
            g.setRight(g.right() + dx)
        if "bottom" in self._rdir:
            g.setBottom(g.bottom() + dy)
        if "left" in self._rdir:
            g.setLeft(g.left() + dx)
            if g.width() < self.MIN_W:
                g.setLeft(g.right() - self.MIN_W)
        if "top" in self._rdir:
            g.setTop(g.top() + dy)
            if g.height() < self.MIN_H:
                g.setTop(g.bottom() - self.MIN_H)
        self.setGeometry(g)

    # ── event filter (จัดการทุกอย่างที่นี่) ──────────────────────

    def eventFilter(self, obj, event):
        t = event.type()

        if t == QEvent.MouseMove:
            gpos = obj.mapToGlobal(event.pos())

            # กำลัง resize → ขยาย geometry
            if self._rdir and (event.buttons() & Qt.LeftButton):
                self._apply_resize(gpos)
                return True  # กัน event ไม่ให้ไปถึง child

            # hover → เปลี่ยน cursor บน obj โดยตรง ← จุดแก้หลัก
            pos = self.mapFromGlobal(gpos)
            d = self._direction(pos)
            if d:
                obj.setCursor(QCursor(self._CURSORS[d]))  # ✅ ตั้งบน obj ไม่ใช่ self
            else:
                obj.unsetCursor()

        elif t == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                gpos = obj.mapToGlobal(event.pos())
                pos = self.mapFromGlobal(gpos)
                d = self._direction(pos)
                if d:
                    self._rdir = d
                    self._drag_gpos = gpos
                    self._start_geom = self.geometry()
                    return True  # กัน event ไม่ให้ไปถึง child

        elif t == QEvent.MouseButtonRelease:
            if self._rdir:
                self._rdir = self._drag_gpos = self._start_geom = None
                return True

        return super().eventFilter(obj, event)

    # ── SubWindow เอง (กรณี mouse อยู่นอก child เช่น title bar ขอบ) ──

    def mouseMoveEvent(self, event):
        if self._rdir and (event.buttons() & Qt.LeftButton):
            self._apply_resize(event.globalPos())
            return
        d = self._direction(event.pos())
        self.setCursor(QCursor(self._CURSORS[d])) if d else self.unsetCursor()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            d = self._direction(event.pos())
            if d:
                self._rdir = d
                self._drag_gpos = event.globalPos()
                self._start_geom = self.geometry()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._rdir = self._drag_gpos = self._start_geom = None
        super().mouseReleaseEvent(event)
