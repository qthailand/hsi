"""Reusable PyQt5 card for arbitrary values, independent of HSI services.

Use set_content(value, detail, color) with green, red, orange or cyan.
Updates are idempotent; long footer text is shortened with a full tooltip.
set_compact(size) turns the card into a small square with title and value only
(the detail stays in the tooltip), for a grid of many cards.
set_clickable(True) makes the card a button: it emits ``clicked`` on a left click or
Enter / Space, shows a pointing hand and a small hint ("View") in its corner.
set_togglable(True) shows an ON / OFF switch button in the bottom-left corner (a click on it, or
on the value, emits ``value_clicked``, not ``clicked``); set_switch_state(on, label) sets what it shows.

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
from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout
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
        value_clicked(): the value was clicked (only while :meth:`set_togglable` is on): the
            owner switches what the card stands for on or off and updates the card.

    Not constructible with no arguments (``title`` is required), hence it is not
    listed in the package ``__widgets__``.
    """

    clicked = pyqtSignal()
    value_clicked = pyqtSignal()

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
        self._togglable = False
        self._switch_on = True
        # bottom-left switch of a card whose state the user can turn on / off (no layout: it overlays)
        self.switch_btn = QPushButton(self)
        self.switch_btn.setCursor(Qt.PointingHandCursor)
        self.switch_btn.setFocusPolicy(Qt.NoFocus)
        self.switch_btn.clicked.connect(self.value_clicked.emit)
        self.switch_btn.hide()
        self.switch_mark = self.switch_btn        # kept name for callers / tests
        self.set_switch_state(True)
        self.value.installEventFilter(self)
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

    def set_togglable(self, on=True, tip=''):
        """Make the value a switch (``value_clicked`` on a left click on it) or plain text again.

        Args:
            on: While true the value shows a pointing hand and the power mark is shown.
            tip: Extra tooltip line for the value, e.g. ``'Click to switch dark subtraction off'``.
        """
        self._togglable = bool(on)
        self.value.setCursor(Qt.PointingHandCursor if on else Qt.ArrowCursor)
        self.value.setToolTip(str(tip) if on and tip else '')
        self.switch_btn.setToolTip(str(tip) if tip else '')
        self.switch_btn.setVisible(self._togglable)
        self._place_hint()

    def set_switch_state(self, on, label=None):
        """Show the switch as on (filled) or off (outlined), with ``label`` (default ``ON`` / ``OFF``)."""
        self._switch_on = bool(on)
        text = str(label) if label else ('ON' if on else 'OFF')
        self.switch_btn.setText(f'⏻ {text}')
        if on:
            style = ('QPushButton {color: #1b5e20; background: rgba(255,255,255,235); border: none; '
                     'border-radius: 8px; font-size: 10px; font-weight: 700; padding: 1px 7px;}'
                     'QPushButton:hover {background: white;}')
        else:
            style = ('QPushButton {color: white; background: rgba(0,0,0,70); border: 1px solid rgba(255,255,255,200); '
                     'border-radius: 8px; font-size: 10px; font-weight: 700; padding: 1px 7px;}'
                     'QPushButton:hover {background: rgba(0,0,0,110);}')
        self.switch_btn.setStyleSheet(style)
        self._place_hint()

    def switch_state(self):
        """Whether the switch shows on."""
        return self._switch_on

    def is_togglable(self):
        """Whether a click on the value emits ``value_clicked`` now."""
        return self._togglable

    def eventFilter(self, obj, event):
        """A left click on the value of a togglable card is the switch (not a card click)."""
        if obj is self.value and self._togglable and event.type() in (QEvent.MouseButtonPress,
                                                                        QEvent.MouseButtonRelease):
            if event.button() == Qt.LeftButton:
                if event.type() == QEvent.MouseButtonRelease and self.value.rect().contains(event.pos()):
                    self.value_clicked.emit()
                return True
        return super().eventFilter(obj, event)

    def _place_hint(self):
        self.hint.adjustSize()
        btn = getattr(self, 'switch_btn', None)
        bottom = self.height() - self.hint.height() - 5
        if btn is not None:
            btn.adjustSize()
            btn.move(6, self.height() - btn.height() - 4)
            if btn.isVisibleTo(self):            # the switch takes the bottom row: the hint goes above it
                bottom = btn.y() - self.hint.height() - 1
        self.hint.move(self.width() - self.hint.width() - 7, bottom)

    def resizeEvent(self, event):
        """Keep the corner hint and the switch button in their corners."""
        super().resizeEvent(event)
        self._place_hint()

    def mouseReleaseEvent(self, event):
        """A left click on a clickable card emits ``clicked``."""
        if self._clickable and event.button() == Qt.LeftButton and self.rect().contains(event.pos()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        """Enter / Space on a clickable card emits ``clicked``."""
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

