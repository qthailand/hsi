"""Shared building blocks of the Smile and Keystone calibration pages (one look for both).

* :class:`Step` - one row of the workflow checklist (status glyph, title, detail, controls).
* :class:`DashboardMixin` - plot panels (title, Matplotlib toolbar, Expand, Export PNG), the
  scroll column of secondary panels, summary metric cards, the status banner and the page
  style sheet. The page provides ``_canvases`` / ``_tab_pages`` (lists indexed by canvas),
  ``_group_pages`` (its workflow tab pages) and ``_on_export_plot(index)``; the banner is
  ``lbl_ready``.

Pure Qt widgets, GUI thread only.
"""
from __future__ import annotations

import html
from typing import Optional, Sequence

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QBoxLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .ui_theme import colors

GOOD, WARN, BAD, TODO, BUSY, OPTIONAL = "ok", "warn", "bad", "todo", "busy", "optional"
#: state -> (glyph, colour key of :func:`modules.smile.ui_theme.colors`)
GLYPH = {GOOD: ("✓", "good"), WARN: ("!", "warn"), BAD: ("✗", "bad"),
         TODO: ("●", "todo"), BUSY: ("…", "busy"), OPTIONAL: ("○", "todo")}
#: state -> chip / banner colour triple of the theme
CHIP = {GOOD: "chip_good", WARN: "chip_warn", BAD: "chip_bad", "info": "chip_info"}
#: Dark blue of the section title bars (WORKFLOW, CALIBRATION QUALITY SUMMARY).
TITLE_BG = "#193d5c"


class Step(QFrame):
    """One row of the step checklist: status glyph, title, detail and its controls.

    The step's explanation (*hint*) is its tooltip; the line under the detail shows it only when
    the step needs action (state ``BAD``). :meth:`set_state` takes an optional *tip*: the full
    notes behind a shortened detail.
    """

    def __init__(self, number: int, title: str, hint: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ksStep")
        c = colors(self)
        self.setStyleSheet(f"#ksStep {{ border: 1px solid {c['border']}; border-radius: 6px; background: {c['card']}; }}"
                           f" #ksStep QLabel {{ color: {c['text']}; background: transparent; }}")
        grid = QGridLayout(self)
        grid.setContentsMargins(6, 4, 6, 4)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        self._grid = grid
        self.glyph = QLabel()
        self.glyph.setFixedWidth(26)
        self.glyph.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        grid.addWidget(self.glyph, 0, 0, 6, 1)
        self.title = QLabel(f"<b>{number}&nbsp;&nbsp;{html.escape(title)}</b>")
        head = QHBoxLayout()
        head.setSpacing(6)
        head.addWidget(self.title)
        head.addStretch()
        self.head = head
        grid.addLayout(head, 0, 1)
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.RichText)
        self.detail.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        grid.addWidget(self.detail, 1, 1)
        self.hint = QLabel(hint)
        self.hint.setWordWrap(True)
        self.hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.hint.setStyleSheet(f"color:{c['hint']}; font-size:8.5pt;")
        self.hint.setVisible(False)
        self.setToolTip(hint)
        grid.addWidget(self.hint, 2, 1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(4)
        grid.addLayout(self.actions, 3, 1)
        self._next_row = 4
        grid.setColumnStretch(1, 1)
        self.state = TODO
        self.set_state(TODO, "")

    def set_state(self, state: str, detail: str, hint: Optional[str] = None, tip: str = "") -> None:
        """Status glyph + detail text (rich text); *tip*: the full notes behind a shortened detail."""
        self.state = state
        glyph, key = GLYPH[state]
        colour = colors(self)[key]
        self.glyph.setText(f"<span style='font-size:15pt;font-weight:bold;color:{colour}'>{glyph}</span>")
        self.detail.setText(detail)
        self.detail.setToolTip(tip)
        if hint is not None:
            self.hint.setText(hint)
            self.setToolTip(hint)
        self.hint.setVisible(state == BAD and bool(self.hint.text()))

    def add(self, *widgets) -> None:
        """Controls on the step's action row."""
        for wdg in widgets:
            self.actions.addWidget(wdg)

    def add_row(self, *widgets) -> QHBoxLayout:
        """One more row of controls (or one full-width widget) under the action row."""
        row = QHBoxLayout()
        row.setSpacing(4)
        for wdg in widgets:
            row.addWidget(wdg)
        if not (len(widgets) == 1 and isinstance(widgets[0], QLabel)):
            row.addStretch()
        self._grid.addLayout(row, self._next_row, 1)
        self._next_row += 1
        return row

    def finish(self) -> None:
        self.actions.addStretch()


def title_bar(text: str) -> QLabel:
    """Dark section title (WORKFLOW, CALIBRATION QUALITY SUMMARY)."""
    label = QLabel(text)
    label.setStyleSheet(f"QLabel {{ background:{TITLE_BG}; color:white; padding:8px; font-weight:bold; "
                        "border-radius:6px; }")
    return label


def page_style(page: QWidget) -> str:
    """Style sheet of a calibration page (cards, metric values, plot panel titles, workflow tabs)."""
    p = colors(page)
    return f"""
        #ksMetric, #ksPlotPanel {{ background: {p['card']}; border: 1px solid {p['border']}; border-radius: 8px; }}
        #ksMetricHeading {{ color: {p['muted']}; font-size: 9pt; }}
        #ksMetricValue {{ color: {p['text']}; font-size: 15pt; font-weight: bold; }}
        #ksMetricHint {{ color: {p['muted']}; font-size: 8pt; }}
        #ksPlotTitle {{ color: {p['text']}; font-size: 10pt; font-weight: bold; border: none; }}
        QTabWidget::pane {{ border: 1px solid {p['border']}; border-radius: 8px; top: -1px; }}
        QTabBar::tab {{ padding: 6px 14px; margin-right: 3px; }}
        QTabBar::tab:selected {{ border-bottom: 3px solid #0b9398; }}
    """


class DashboardMixin:
    """Plot panels, metric cards and the banner of a calibration page (see the module docstring)."""

    def _make_plot_panel(self, index: int, name: str, *, minimum: int = 200,
                         extras: Sequence[QWidget] = ()) -> QWidget:
        """A Matplotlib canvas in a dashboard panel: title, *extras*, toolbar, Expand, Export PNG.

        The canvas is ``self._canvases[index]`` (indices stay stable for the host and the tests).
        """
        from .widget import PlotCanvas
        panel = QFrame()
        panel.setObjectName("ksPlotPanel")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(7, 7, 7, 7)
        lay.setSpacing(3)
        bar = QHBoxLayout()
        bar.setSpacing(5)
        label = QLabel(name)
        label.setObjectName("ksPlotTitle")
        bar.addWidget(label)
        for wdg in extras:
            bar.addWidget(wdg)
        bar.addStretch()
        canvas = PlotCanvas(panel)
        try:
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
            toolbar = NavigationToolbar2QT(canvas, panel, coordinates=True)
            toolbar.setIconSize(toolbar.iconSize() * 0.75)
            toolbar.setStyleSheet("QToolBar { border: 0px; }")
            bar.addWidget(toolbar)
        except Exception:  # noqa: BLE001  (no toolbar: the plot still works)
            pass
        btn_expand = QToolButton()
        btn_expand.setText("Expand")
        btn_expand.setCheckable(True)
        btn_expand.setToolTip("Show this plot over the whole tab; press again to return to the dashboard.")
        btn_expand.toggled.connect(lambda on, p=panel, b=btn_expand: self._expand_panel(p, b, on))
        bar.addWidget(btn_expand)
        btn_export = QPushButton("Export PNG")
        btn_export.setToolTip("Export this plot at 300 dpi (not the entire dashboard).")
        btn_export.clicked.connect(lambda _checked=False, i=index: self._on_export_plot(i))
        bar.addWidget(btn_export)
        lay.addLayout(bar)
        canvas.setMinimumHeight(minimum)
        lay.addWidget(canvas, stretch=1)
        self._canvases[index] = canvas
        self._tab_pages[index] = panel
        return panel

    def _expand_panel(self, panel: QWidget, button: QToolButton, on: bool) -> None:
        """Let one plot panel fill its workflow tab (on) or put the dashboard back (off).

        Every widget beside the panel's branch, up to the tab page, is hidden and given back; nothing
        is rebuilt, so the canvas, its toolbar state and the canvas indices are untouched.
        """
        button.setText("Restore" if on else "Expand")
        if not on:
            hidden, stretched = getattr(panel, "_ks_expanded", ([], []))
            for widget in hidden:
                widget.setVisible(True)
            for layout, widget in stretched:
                layout.setStretchFactor(widget, 0)
            panel._ks_expanded = ([], [])
            return
        hidden, stretched = [], []
        w = panel
        while w is not None and w not in getattr(self, "_group_pages", ()):
            parent = w.parentWidget()
            if parent is None:
                break
            if isinstance(parent, QSplitter):
                layout, siblings = None, [parent.widget(i) for i in range(parent.count())]
            else:
                layout = parent.layout()
                siblings = [layout.itemAt(i).widget() for i in range(layout.count())] if layout is not None else []
            for s in siblings:
                if s is not None and s is not w and not s.isHidden():   # not hidden by us or anyone else
                    s.setVisible(False)
                    hidden.append(s)
            if isinstance(layout, QBoxLayout) and layout.indexOf(w) >= 0:
                layout.setStretchFactor(w, 100)         # beats the trailing stretch of a scroll column
                stretched.append((layout, w))
            w = parent
        panel._ks_expanded = (hidden, stretched)

    @staticmethod
    def _scroll_panels(*panels) -> QScrollArea:
        """Secondary panels in one scrolling column (short displays keep every plot reachable)."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(2, 1, 2, 2)
        layout.setSpacing(8)
        for panel in panels:
            layout.addWidget(panel)
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    @staticmethod
    def _metric_card(title: str):
        """``(frame, value label)`` of one summary card."""
        frame = QFrame()
        frame.setObjectName("ksMetric")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(9, 7, 9, 6)
        lay.setSpacing(2)
        heading = QLabel(title)
        heading.setObjectName("ksMetricHeading")
        lay.addWidget(heading)
        value = QLabel("—")
        value.setObjectName("ksMetricValue")
        value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(value)
        return frame, value

    def _paint_metric(self, label: QLabel, key: str) -> None:
        """Colour of a summary value: ``good`` / ``warn`` / ``bad`` for a verdict, ``text`` for a plain number."""
        label.setStyleSheet(f"color:{colors(self)[key]};")

    def _paint_banner(self, state: str) -> None:
        """Banner (``self.lbl_ready``) in the chip colours of *state* (``info`` before any verdict)."""
        bg, fg, border = colors(self)[CHIP.get(state, "chip_info")]
        self.lbl_ready.setStyleSheet(f"QLabel {{ background:{bg}; color:{fg}; border:1px solid {border}; "
                                     f"border-radius:6px; padding:8px; }}")
