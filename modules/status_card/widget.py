"""Reusable PyQt5 card for arbitrary values, independent of HSI services.

Use set_content(value, detail, color) with green, red, orange or cyan.
Updates are idempotent; long footer text is shortened with a full tooltip.

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

    Not constructible with no arguments (``title`` is required), hence it is not
    listed in the package ``__widgets__``.
    """
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
        self._last = None
        self.set_content('—', 'Waiting', color='cyan')

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
        self.detail.setText(detail if len(detail) <= 38 else detail[:35] + '…')
        self.setToolTip(f'{self.title.text()}: {value}\n{detail}')
        self.setAccessibleName(f'{self.title.text()} {value}')

