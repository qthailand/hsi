"""macOS-style traffic light buttons and custom title bar for dark theme.

Data conventions:
    Colours are hex strings; sizes are device-independent pixels.

Threading:
    GUI thread only.  Needs the ``qtawesome`` package for the hover icons.

Example:
    >>> bar = IOSTitleBar(main_window, title="My App")       # doctest: +SKIP
    >>> main_window.setMenuWidget(bar)                       # doctest: +SKIP
    >>> bar.inject_menubar(main_window.menuBar())            # doctest: +SKIP
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
)

import qtawesome as qta


# ── dark theme palette ────────────────────────────────────────────
_BG_TITLE = "#1e1e1e"
_BG_MENU = "#252525"
_BORDER = "#3a3a3a"
_TEXT = "#cccccc"
_TEXT_DIM = "#888888"


class TrafficLightButton(QPushButton):
    """Round 14 px window button that shows an icon while hovered.

    Args:
        color: Fill colour (any ``QColor`` string) when idle.
        hover_color: Fill colour while the mouse is over the button.
        icon: Optional ``QIcon`` (e.g. from ``qtawesome``) painted at 10 px on hover.
        parent: Optional parent widget.

    Signals:
        clicked: Inherited from ``QPushButton``.
    """

    def __init__(
        self,
        color: str,
        hover_color: str,
        icon=None,
        parent=None,
    ):
        super().__init__(parent)
        self._color = QColor(color)
        self._hover_color = QColor(hover_color)
        self._icon = icon
        self._hovered = False
        self.setFixedSize(14, 14)
        self.setCursor(Qt.PointingHandCursor)

    def enterEvent(self, event):
        """Qt handler: switch to the hover colour/icon."""
        self._hovered = True
        self.update()

    def leaveEvent(self, event):
        """Qt handler: switch back to the idle colour."""
        self._hovered = False
        self.update()

    def paintEvent(self, event):
        """Qt handler: paint the coloured dot and, when hovered, its icon."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        color = self._hover_color if self._hovered else self._color
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, self.width(), self.height())

        if self._hovered and self._icon:
            pixmap = self._icon.pixmap(10, 10)
            x = (self.width() - pixmap.width()) // 2
            y = (self.height() - pixmap.height()) // 2
            painter.drawPixmap(x, y, pixmap)

        painter.end()


class IOSTitleBar(QWidget):
    """Two-row title bar: traffic lights + title (row 1), menu bar (row 2).

    Row 1 (title_row):  three round buttons and a centred title
    Row 2 (menu_row):   host menu bar, see :meth:`inject_menubar`

    The bar controls ``window``: the red button closes it, yellow minimises,
    green toggles maximise, and dragging the bar moves it.

    Usage::

        title_bar = IOSTitleBar(main_window, title="HSI System")
        main_window.setMenuWidget(title_bar)
        title_bar.inject_menubar(main_window.menuBar())

    Args:
        window: The top-level widget this bar closes/minimises/maximises/moves.
            Required, so this class is not in the package ``__widgets__``.
        title: Text shown in the centre of row 1.
        parent: Optional parent widget.

    Attributes:
        btn_close, btn_minimize, btn_maximize: The three ``TrafficLightButton``.
        title_label: ``QLabel`` with the title; use ``setText`` to change it.
    """

    _BTN_BOX_WIDTH = 62  # 3 buttons × 14 + 2 gaps × 8 + padding

    def __init__(
        self,
        window: QWidget,
        title: str = "HSI System",
        parent=None,
    ):
        super().__init__(parent)
        self._window = window
        self._dragging = False
        self._drag_offset = QPoint(0, 0)
        self.setMouseTracking(True)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Row 1: title row ─────────────────────────────────────
        title_row = QWidget()
        title_row.setObjectName("title_row")
        title_row.setFixedHeight(36)
        title_row.setStyleSheet(f"""
            QWidget#title_row {{
                background-color: {_BG_TITLE};
                border-bottom: 1px solid {_BORDER};
            }}
        """)

        row_layout = QHBoxLayout(title_row)
        row_layout.setContentsMargins(12, 0, 12, 0)
        row_layout.setSpacing(8)

        # Traffic light buttons
        icon_close = qta.icon("fa5s.times", color="#8b0000")
        icon_minimize = qta.icon("fa5s.minus", color="#8b6914")
        icon_maximize = qta.icon("fa5s.plus", color="#1a6b24")

        self.btn_close = TrafficLightButton("#ff5f57", "#e0443e", icon_close)
        self.btn_minimize = TrafficLightButton("#febc2e", "#dea123", icon_minimize)
        self.btn_maximize = TrafficLightButton("#28c840", "#1aab29", icon_maximize)

        self.btn_close.clicked.connect(lambda: self._window.close())
        self.btn_minimize.clicked.connect(lambda: self._window.showMinimized())
        self.btn_maximize.clicked.connect(self._toggle_maximize)

        btn_box = QWidget()
        btn_box.setStyleSheet("background: transparent;")
        btn_box_layout = QHBoxLayout(btn_box)
        btn_box_layout.setContentsMargins(0, 0, 0, 0)
        btn_box_layout.setSpacing(8)
        btn_box_layout.addWidget(self.btn_close)
        btn_box_layout.addWidget(self.btn_minimize)
        btn_box_layout.addWidget(self.btn_maximize)

        # Centered title
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(
            f"color: {_TEXT}; font-size: 13px; font-weight: 500;"
            f"font-family: 'Segoe UI', sans-serif; background: transparent;"
        )

        # Right placeholder for symmetry
        right_spacer = QWidget()
        right_spacer.setFixedWidth(self._BTN_BOX_WIDTH)
        right_spacer.setStyleSheet("background: transparent;")

        row_layout.addWidget(btn_box)
        row_layout.addStretch()
        row_layout.addWidget(self.title_label)
        row_layout.addStretch()
        row_layout.addWidget(right_spacer)

        main_layout.addWidget(title_row)

        # ── Row 2: menu bar area ─────────────────────────────────
        self._menu_row = QWidget()
        self._menu_row.setObjectName("menu_row")
        self._menu_row.setStyleSheet(f"""
            QWidget#menu_row {{
                background-color: {_BG_MENU};
                border-bottom: 1px solid {_BORDER};
            }}
        """)
        self._menu_row_layout = QHBoxLayout(self._menu_row)
        self._menu_row_layout.setContentsMargins(0, 0, 0, 0)
        self._menu_row_layout.setSpacing(0)
        main_layout.addWidget(self._menu_row)

    def inject_menubar(self, menubar):
        """Move a ``QMenuBar`` into row 2 of this bar and give it the dark style.

        Args:
            menubar: Usually ``main_window.menuBar()``.  It is re-parented.
        """
        menubar.setParent(self._menu_row)
        menubar.setStyleSheet(f"""
            QMenuBar {{
                background-color: transparent;
                color: {_TEXT};
                border: none;
                padding: 2px 4px;
                font-size: 12px;
            }}
            QMenuBar::item {{
                padding: 4px 10px;
                background: transparent;
            }}
            QMenuBar::item:selected {{
                background-color: {_BORDER};
                color: #e0e0e0;
            }}
            QMenu {{
                background-color: {_BG_MENU};
                color: {_TEXT};
                border: 1px solid {_BORDER};
            }}
            QMenu::item:selected {{
                background-color: {_BORDER};
            }}
        """)
        self._menu_row_layout.addWidget(menubar)
        menubar.show()          # ← force show after reparent
        self._menu_row.show()   # ← force show container

    def _toggle_maximize(self):
        if self._window.isMaximized():
            self._window.showNormal()
        else:
            self._window.showMaximized()

    # ── drag to move window ──────────────────────────────────────

    def mousePressEvent(self, event):
        """Qt handler: start dragging the window on left press."""
        if event.button() == Qt.LeftButton:
            self._drag_offset = (
                event.globalPos() - self._window.frameGeometry().topLeft()
            )
            self._dragging = True
            event.accept()

    def mouseMoveEvent(self, event):
        """Qt handler: move the controlled window while the left button is held."""
        if self._dragging and (event.buttons() & Qt.LeftButton):
            self._window.move(event.globalPos() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event):
        """Qt handler: stop dragging."""
        self._dragging = False
        event.accept()
