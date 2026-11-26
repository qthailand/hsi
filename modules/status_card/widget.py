"""Reusable PyQt5 card for arbitrary values, independent of HSI services.

Use set_content(value, detail, color) with green, red, orange or cyan.
Updates are idempotent; long footer text is shortened with a full tooltip.
set_compact(size) turns the card into a small square with title and value only
(the detail stays in the tooltip), for a grid of many cards.
set_clickable(True) makes the card a button: it emits ``clicked`` on a left click or
Enter / Space, shows a pointing hand and a small hint ("View") in its corner.
set_togglable(True) shows an ON / OFF switch button in the bottom-left corner (a click on it, or
on the value, emits ``value_clicked``, not ``clicked``); set_switch_state(on, label) sets what it shows.
set_row(name) draws the card as one row instead (dot in the status colour, name, state, ``›``
while clickable, a toggle switch while togglable), for a list of corrections; the API is the same.

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
from PyQt5.QtCore import QEvent, QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap

#: Card colour name -> ``'#RRGGBB'`` gradient start colour.
PALETTE = {'green': '#00C98D', 'red': '#FF637C', 'orange': '#FF9868', 'cyan': '#00BFC7'}
#: Row presentation: switch track colour when on / off, and the size of the switch (px).
ROW_SWITCH_ON, ROW_SWITCH_OFF, ROW_SWITCH_SIZE = '#22c06b', '#343a42', (30, 17)


def switch_icon(on: bool) -> QIcon:
    """A toggle switch drawn as an icon (track + knob); Qt greys it out when disabled."""
    w, h = ROW_SWITCH_SIZE
    pixmap = QPixmap(w * 2, h * 2)                  # drawn at 2x, sharp on high-DPI screens
    pixmap.setDevicePixelRatio(2.0)
    pixmap.fill(Qt.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(ROW_SWITCH_ON if on else ROW_SWITCH_OFF))
    p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), (h - 1) / 2, (h - 1) / 2)
    p.setBrush(QColor('#ffffff' if on else '#aeb7c4'))
    r = h / 2 - 2.3
    p.drawEllipse(QPointF(w - h / 2 if on else h / 2, h / 2), r, r)
    p.end()
    return QIcon(pixmap)

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
    #: The value, colour, tooltip, clickable / togglable state or switch face changed, so a
    #: mirror of the card (e.g. a compact row) can redraw. Not emitted by a no-op update.
    changed = pyqtSignal()

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
        self._row = False               # set_row: drawn as one row
        self._row_note = None           # set_row_note: shown instead of the value in the row
        self._last = None
        self.set_content('—', 'Waiting', color='cyan')

    # ── row presentation ──────────────────────────────────────────

    def set_row(self, name):
        """Draw the card as one row: dot (status colour), ``name``, state, ``›``, switch.

        For a list of corrections. Clicks, signals and setters stay the same: a click on the row
        emits ``clicked`` (while clickable), a click on the switch ``value_clicked``. The title,
        the big value and the detail are kept (read by hosts) but not shown.

        Args:
            name: Name shown in the row (the title is upper case).
        """
        from PyQt5 import sip
        old = self.layout()
        while old.count():                         # the card's widgets stay, its layout goes
            old.takeAt(0)
        sip.delete(old)
        for widget in (self.title, self.value, self._line, self.detail, self.hint):
            widget.hide()
        self._row = True
        self._compact = None
        self.setMinimumSize(0, 30)
        self.setMaximumSize(16777215, 16777215)
        self.row_dot = QLabel()
        self.row_dot.setFixedSize(8, 8)
        self.row_name = QLabel(str(name))
        self.row_name.setStyleSheet('color: #e6e9ee; font-size: 12px; background: transparent;')
        self.row_state = QLabel()
        self.row_state.setStyleSheet('color: #7d8794; font-size: 11px; background: transparent;')
        self.row_chevron = QLabel('\u203a')
        self.row_chevron.setStyleSheet('color: #8ab8ff; font-size: 16px; font-weight: 600; background: transparent;')
        self.switch_btn.setText('')
        self.switch_btn.setFlat(True)
        self.switch_btn.setIconSize(QSize(*ROW_SWITCH_SIZE))
        self.switch_btn.setFixedSize(QSize(ROW_SWITCH_SIZE[0] + 4, ROW_SWITCH_SIZE[1] + 4))
        self.switch_btn.setStyleSheet('QPushButton {background: transparent; border: none; padding: 0;}')
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 3, 4, 3)
        row.setSpacing(8)
        row.addWidget(self.row_dot)
        row.addWidget(self.row_name, 1)
        row.addWidget(self.row_state)
        row.addWidget(self.row_chevron)
        row.addWidget(self.switch_btn)
        self.setStyleSheet('QFrame#statusCard {background: transparent; border: none; border-radius: 6px;}'
                           'QFrame#statusCard:hover {background: #20242a;}')
        self._refresh_row()

    def is_row(self):
        """Whether :meth:`set_row` made the card a row."""
        return self._row

    def set_row_note(self, text):
        """Show ``text`` instead of the value in the row (e.g. ``12 / 50`` while a dark is captured);
        ``None`` shows the value again. No effect before :meth:`set_row`."""
        text = None if text is None else str(text)
        if text != self._row_note:
            self._row_note = text
            if self._row:
                self._refresh_row()

    def row_state_text(self):
        """What the row shows as the state: the note, else the value."""
        return self._row_note if self._row_note is not None else self.value.text()

    def _refresh_row(self):
        if not getattr(self, '_row', False):
            return
        self.row_dot.setStyleSheet(f'background: {PALETTE.get(self.color_name(), "#5b626c")}; border-radius: 4px;')
        self.row_state.setText(self.row_state_text())
        self.row_chevron.setVisible(self._clickable)
        self.switch_btn.setVisible(self._togglable)
        self.switch_btn.setIcon(switch_icon(self._switch_on))

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
        self.hint.setVisible(bool(on and hint) and not getattr(self, '_row', False))
        self._place_hint()
        self._refresh_row()
        self.changed.emit()

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
        self._refresh_row()
        self.changed.emit()

    def set_switch_state(self, on, label=None):
        """Show the switch as on (filled) or off (outlined), with ``label`` (default ``ON`` / ``OFF``)."""
        self._switch_on = bool(on)
        if getattr(self, '_row', False):           # the row's switch is an icon
            self.switch_btn.setToolTip(str(label) if label else self.switch_btn.toolTip())
            self._refresh_row()
            self.changed.emit()
            return
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
        self.changed.emit()

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
        if getattr(self, '_row', False):           # a row lays itself out
            return
        self.hint.adjustSize()
        btn = getattr(self, 'switch_btn', None)
        bottom = self.height() - self.hint.height() - 5
        if btn is not None:
            btn.adjustSize()
            btn.move(6, self.height() - btn.height() - 4)
            if btn.isVisibleTo(self):            # the switch takes the bottom row: the hint goes above it
                bottom = btn.y() - self.hint.height() - 1
        self.hint.move(self.width() - self.hint.width() - 7, bottom)

    def setToolTip(self, text):
        """Set the tooltip (the card's full detail) and emit :attr:`changed`."""
        super().setToolTip(text)
        self.changed.emit()

    def color_name(self):
        """The palette key of the last :meth:`set_content` (``'cyan'`` before the first)."""
        return self._last[2] if getattr(self, '_last', None) else 'cyan'

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
        if not self._row:
            self.setStyleSheet(f'QFrame#statusCard {{background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {PALETTE[color]}, stop:1 {QColor(PALETTE[color]).lighter(118).name()}); border: none; border-radius: 6px;}}')
        self.value.setText(value)
        if self._compact:
            self._fit_value()
        self.detail.setText(detail if len(detail) <= 38 else detail[:35] + '…')
        self.setToolTip(f'{self.title.text()}: {value}\n{detail}')
        self.setAccessibleName(f'{self.title.text()} {value}')
        self._refresh_row()

