"""Reusable PyQt5 card for arbitrary values, independent of HSI services.

Use set_content(value, detail, color) with green, red, orange or cyan.
Updates are idempotent; long footer text is shortened with a full tooltip.
set_compact(size) turns the card into a small square with title and value only
(the detail stays in the tooltip), for a grid of many cards.
set_clickable(True) makes the card a button: it emits ``clicked`` on a left click or
Enter / Space, shows a pointing hand and a small hint ("View") in its corner.

Data conventions:
    ``value`` and ``detail`` are converted with ``str()``. ``color`` is a key of
    ``PALETTE`` (name -> ``'#RRGGBB'`` hex string). The footer ``detail`` is
    truncated to 35 characters plus an ellipsis when longer than 38; the tooltip
    always holds the full text.

Threading:
    A normal QWidget: create it and call ``set_content`` on the GUI thread only.

Example:
    >>> from PyQt5.QtWidgets import QApplication
    >>> app = QApplication.instance() or QApplication([])
    >>> from modules.status_card import StatusCard
    >>> card = StatusCard("Storage")
    >>> card.set_content("42 GB", "Available capacity", color="green")
    >>> card.value.text()
    '42 GB'
"""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout
from PyQt5.QtGui import QColor

#: Card colour name -> ``'#RRGGBB'`` gradient start colour.
PALETTE = {'green': '#00C98D', 'red': '#FF637C', 'orange': '#FF9868', 'cyan': '#00BFC7'}

class StatusCard(QFrame):
    """Compact card with selectable palette and a full-detail tooltip.

    Attributes:
        title (QLabel): Upper-cased title label.
        value (QLabel): Large value label.
        detail (QLabel): Small, word-wrapped footer label.

    Signals:
        clicked(): the card was activated (only while :meth:`set_clickable` is on).

    Not constructible with no arguments (``title`` is required), hence it is not
    listed in the package ``__widgets__``.
    """

    clicked = pyqtSignal()

    def __init__(self, title, parent=None):
        """Create the card showing ``'—'`` / ``'Waiting'`` in cyan.

        Args:
            title: Card title; displayed upper-cased.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setObjectName('statusCard')
        self.setMinimumWidth(140)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 9)
        layout.setSpacing(4)
        self.title = QLabel(title.upper())
        self.title.setStyleSheet('color: white; font-size: 10px; font-weight: 600; background: transparent;')
        self.value = QLabel()
        self.value.setStyleSheet('color: white; font-size: 20px; font-weight: 600; background: transparent;')
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet('background: rgba(255,255,255,100);')
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet('color: white; font-size: 10px; background: transparent;')
        for widget in (self.title, self.value, line, self.detail):
            layout.addWidget(widget)
        self._line = line
        self._compact = None          # side of the square in compact mode
        self._clickable = False
        self.hint = QLabel(self)      # corner hint of a clickable card (no layout: it overlays)
        self.hint.setStyleSheet('color: rgba(255,255,255,210); font-size: 9px; background: transparent;')
        self.hint.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hint.hide()
        self._last = None
        self.set_content('—', 'Waiting', color='cyan')

    def set_compact(self, size=92):
        """Make the card a ``size`` x ``size`` px square showing the title and the value only.

        The detail text and the separator are hidden (the tooltip still holds the detail), the
        fonts and margins are smaller, and a value too long for the square is shown in a smaller
        font (14 px down to 9 px). For a grid of cards where a full card each would not fit.

        Args:
            size: Side of the square in pixels.
        """
        self.layout().setContentsMargins(7, 7, 7, 7)
        self.layout().setSpacing(2)
        self.title.setStyleSheet('color: white; font-size: 9px; font-weight: 600; background: transparent;')
        self.value.setStyleSheet('color: white; font-size: 14px; font-weight: 600; background: transparent;')
        self._line.hide()
        self.detail.hide()
        self.layout().addStretch(1)
        self.setMinimumWidth(0)
        self.setFixedSize(int(size), int(size))
        self._compact = int(size)
        self._fit_value()

    def _fit_value(self):
        """Compact mode: the largest value font (14-9 px) with which the value fits the square."""
        margins = self.layout().contentsMargins()
        room = self._compact - margins.left() - margins.right()
        for px in range(14, 8, -1):         # measured on the label itself: the style sheet sets its font
            self.value.setStyleSheet(f'color: white; font-size: {px}px; font-weight: 600; background: transparent;')
            self.value.ensurePolished()
            if self.value.sizeHint().width() <= room:
                break

    def set_clickable(self, on=True, hint='View ▸'):
        """Make the card a button (or a plain card again).

        Args:
            on: While true a left click, Enter or Space emits ``clicked``; the cursor is a
                pointing hand and ``hint`` is shown in the bottom-right corner.
            hint: Corner text, e.g. ``'View ▸'``; ``''`` for none.
        """
        self._clickable = bool(on)
        self.setCursor(Qt.PointingHandCursor if on else Qt.ArrowCursor)
        self.setFocusPolicy(Qt.TabFocus if on else Qt.NoFocus)
        self.hint.setText(str(hint))
        self.hint.setVisible(bool(on and hint))
        self._place_hint()

    def is_clickable(self):
        """Whether the card emits ``clicked`` now."""
        return self._clickable

    def _place_hint(self):
        self.hint.adjustSize()
        self.hint.move(self.width() - self.hint.width() - 7, self.height() - self.hint.height() - 5)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_hint()

    def mouseReleaseEvent(self, event):
        if self._clickable and event.button() == Qt.LeftButton and self.rect().contains(event.pos()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if self._clickable and event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit()
            return
        super().keyPressEvent(event)

    def set_content(self, value, detail='', color='cyan'):
        """Update the card; does nothing if the arguments equal the last update.

        Args:
            value: Main value; shown via ``str(value)``.
            detail: Footer text; shown via ``str(detail)``, shortened if long,
                full text in the tooltip.
            color: One of the ``PALETTE`` keys (``'green'``, ``'red'``,
                ``'orange'``, ``'cyan'``).

        Raises:
            ValueError: ``color`` is not a ``PALETTE`` key.
        """
        value, detail = str(value), str(detail)
        if color not in PALETTE:
            raise ValueError(f'Unknown palette color: {color}')
        key = (value, detail, color)
        if key == self._last:
            return
        self._last = key
        self.setStyleSheet(f'QFrame#statusCard {{background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {PALETTE[color]}, stop:1 {QColor(PALETTE[color]).lighter(118).name()}); border: none; border-radius: 6px;}}')
        self.value.setText(value)
        if self._compact:
            self._fit_value()
        self.detail.setText(detail if len(detail) <= 38 else detail[:35] + '…')
        self.setToolTip(f'{self.title.text()}: {value}\n{detail}')
        self.setAccessibleName(f'{self.title.text()} {value}')

