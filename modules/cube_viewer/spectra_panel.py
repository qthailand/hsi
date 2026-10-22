"""SPECTRA panel of the cube viewer: drawing-tool buttons, the spectra tree, Export and Clear.

``SpectraPanel`` (no constructor arguments) is the sidebar group that takes the slot of
the old 'Clear All Points' button.  It shows the window's
:class:`~modules.cube_viewer.spectra.SpectrumRecord` objects but owns none of them: the
window owns the ``SpectrumSet`` and calls :meth:`SpectraPanel.add_record`,
:meth:`~SpectraPanel.update_record`, :meth:`~SpectraPanel.remove_record` and
:meth:`~SpectraPanel.clear`; the panel reports what the user does with signals and never
changes a record itself.  After every request it redraws the row from the record, so a
request the window refuses (a duplicate name, say) simply reverts the row.

Layout::

    [Point] [Rect] [Lasso]             exclusive, checkable tool buttons
    +- Name ------------------ N px -+
    | v Spectra (3)                  |
    |   [x] # (120,48)             1 |
    |   [x] # leaf-mean-0      9,801 |
    |   [ ] # leaf-mean-1  computing |
    +--------------------------------+
    [Export...]  [Clear All Points]

Tree:
    * The check box is the record's ``visible`` flag; the swatch is its colour.
    * F2 or a double click renames in place; ``rename_requested(rid, text)``.
    * The Class column (double click there, or :meth:`SpectraPanel.edit_class`) sets the record's class;
      ``class_requested(rid, text)``.  Records with the same class are pooled by the
      segmentation; 'Segment...' is enabled when >= 2 classes have ready spectra.
    * A 'Segmentation' root (set with :meth:`SpectraPanel.set_segmentation`) lists the
      predicted pixels per class and 'unclassified'; its context menu shows / hides the
      overlay or clears the segmentation.
    * Del deletes the selected records; ``delete_requested(rids)``.
    * Context menu: Rename, Set colour... (``QColorDialog``), Show +-sigma (areas only), Copy
      as text, Delete.
    * The tooltip shows kind, N px, bbox and mean +- sigma at the current band
      (:meth:`SpectraPanel.set_band`), or 'computing...' while the record is pending.
    * While the tree has the focus it takes the viewer's single-key shortcuts
      (:data:`TREE_KEYS`: arrows, Space, Home, End, F2, Delete, E, R, 0, 1, +, =, -) through
      ``ShortcutOverride``, so they move in, check or search the tree instead of stepping
      bands, playing the sweep or opening the export dialog.

Data conventions:
    Rows show :class:`~modules.cube_viewer.spectra.SpectrumRecord` objects held by
    reference (the window owns them and their read-only statistics); records are keyed by
    their int ``rid``, colours are ``'#rrggbb'``.  Tooltip values are the record's float64
    statistics in the cube's units at the 0-based band set with :meth:`SpectraPanel.set_band`;
    bboxes are half-open pixel boxes ``(c0, r0, c1, r1)`` (x = column = scan, y = row =
    slit).  Names are always shown as plain text, never as markup.

Threading:
    GUI thread only.

Example:
    >>> from modules.cube_viewer.spectra_panel import SpectraPanel     # doctest: +SKIP
    >>> panel = SpectraPanel()                                          # doctest: +SKIP
    >>> panel.tool_changed.connect(print)                               # doctest: +SKIP
    >>> panel.add_record(spectrum_set.add("point", {"x": 3, "y": 4}))   # doctest: +SKIP
"""

from __future__ import annotations

import math

import numpy as np
import qtawesome as qta
from PyQt5.QtCore import QEvent, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QButtonGroup, QColorDialog, QHBoxLayout, QHeaderView, QMenu,
    QPushButton, QStyledItemDelegate, QToolTip, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
    QWidget,
)

from modules.cube_viewer.roi_stats import snap_rect
from modules.ui_kit.theme import ACCENT, ACCENT_DARK, BORDER, CARD, FIELD_HOVER, PLOT_FOREGROUND

__all__ = ("SpectraPanel", "TREE_KEYS", "TOOL_BUTTONS")

#: (tool, name, qtawesome icon, tooltip) of the image tools; the buttons show the icon only.
TOOL_BUTTONS = (
    ("point", "Point", "mdi6.crosshairs-gps",
     "Point: a left click adds the spectrum of that pixel"),
    ("rect", "Rect", "mdi6.vector-square",
     "Rectangle: drag over the image; the mean and std of the pixels whose "
     "centre is inside are added (Esc cancels)"),
    ("lasso", "Lasso", "mdi6.lasso",
     "Lasso: hold the left button and draw around an area; release closes "
     "it (Esc cancels)"),
    ("polygon", "Polygon", "mdi6.vector-polygon",
     "Polygon: click the corners; Enter, right-click, double-click or a click on the "
     "first corner closes it (Backspace removes the last corner, Esc cancels)"),
)

#: Joined (segmented) look of the tool buttons: one strip, outer corners rounded only.
_TOOL_GROUP_STYLE = f"""
QPushButton {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 0px;
    padding: 0px; margin: 0px;
}}
QPushButton:hover {{ background: {FIELD_HOVER}; }}
QPushButton:checked {{ background: {ACCENT_DARK}; border-color: {ACCENT}; }}
QPushButton[segment="first"] {{ border-top-left-radius: 4px; border-bottom-left-radius: 4px; }}
QPushButton[segment="last"] {{ border-top-right-radius: 4px; border-bottom-right-radius: 4px; }}
QPushButton[segment="middle"], QPushButton[segment="last"] {{ border-left: none; }}
"""

TREE_KEYS = frozenset({
    Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down, Qt.Key_Space, Qt.Key_Home,
    Qt.Key_End, Qt.Key_F2, Qt.Key_Delete, Qt.Key_E, Qt.Key_R, Qt.Key_0, Qt.Key_1,
    Qt.Key_Plus, Qt.Key_Equal, Qt.Key_Minus,
})
_COMMAND_MODIFIERS = Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier

_RID_ROLE = Qt.UserRole
_PENDING_TEXT = "computing…"
_FAILED_TEXT = "failed"
_FAILED_COLOR = QColor("#ef5350")
_SWATCH_PX = 12
_SEGMENT_TIP = ("Classify every pixel from the classes of the spectra (needs at least two "
                "classes with computed spectra; set a row's class in the Class column)")
_SEG_ROLE = Qt.UserRole + 1          # class index of a Segmentation child (-1: unclassified)
_VIEW_ROLE = Qt.UserRole + 2         # 'single' / 'rgb' of a Cubes > <cube> view child
_STALE_COLOR = QColor("#ffb74d")
_BIG = 1 << 40           # stands in for an unknown image size when snapping a lasso bbox


def _is_tree_key(event) -> bool:
    """True for a key in :data:`TREE_KEYS` pressed without Ctrl / Alt / Meta."""
    return event.key() in TREE_KEYS and not (int(event.modifiers()) & int(_COMMAND_MODIFIERS))


def _tooltip_markup(text: str) -> str:
    """Tooltip text Qt shows literally: plain text Qt would take for rich text (a name such
    as ``'leaf <A>'`` or ``'x <b>y</b>'``) is escaped, keeping its lines and spaces."""
    if Qt.mightBeRichText(text):
        return Qt.convertFromPlainText(text, Qt.WhiteSpacePre)
    return text


def _fmt(value: float) -> str:
    """Compact text of one statistic (``'nan'`` for a missing value)."""
    value = float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6g}"


class _NoEditDelegate(QStyledItemDelegate):
    """Delegate of the 'N px' column: never opens an editor there."""

    def createEditor(self, parent, option, index):
        """Qt override: no editor (the count is not editable)."""
        return None


class _SpectraTree(QTreeWidget):
    """The SPECTRA tree (columns Name | N px | Class); see the module docstring.

    Signals:
        delete_pressed(): Del was pressed (the panel deletes the selection).
        rename_pressed(): F2 was pressed (the panel edits the current record's name).
    """

    delete_pressed = pyqtSignal()
    rename_pressed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tooltip_provider = None
        self.setColumnCount(3)
        self.setHeaderLabels(["Name", "N px", "Class"])
        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setItemDelegateForColumn(1, _NoEditDelegate(self))
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # a double click edits the name (a double click on the check box only toggles it);
        # F2 is handled in keyPressEvent so that it always edits the Name column
        self.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.setUniformRowHeights(True)
        self.setIconSize(QSize(_SWATCH_PX, _SWATCH_PX))
        self.setMinimumHeight(140)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_tooltip_provider(self, provider) -> None:
        """Set ``provider(item) -> str`` that builds an item's tooltip when it is shown."""
        self._tooltip_provider = provider

    def event(self, event):
        """Qt override: take the viewer's single-key shortcuts while the tree has the focus."""
        if event.type() == QEvent.ShortcutOverride and _is_tree_key(event):
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event):
        """Qt override: Del deletes the selection, F2 renames; other keys as in a tree."""
        if not (int(event.modifiers()) & int(_COMMAND_MODIFIERS)):
            if event.key() == Qt.Key_Delete:
                self.delete_pressed.emit()
                event.accept()
                return
            if event.key() == Qt.Key_F2 and self.state() != QAbstractItemView.EditingState:
                self.rename_pressed.emit()
                event.accept()
                return
        super().keyPressEvent(event)

    def viewportEvent(self, event):
        """Qt override: build the tooltip of the hovered row when it is asked for."""
        if event.type() == QEvent.ToolTip and self._tooltip_provider is not None:
            item = self.itemAt(event.pos())
            text = self._tooltip_provider(item) if item is not None else ""
            if text:
                QToolTip.showText(event.globalPos(), _tooltip_markup(text), self.viewport())
            else:
                QToolTip.hideText()
                event.ignore()
            return True
        return super().viewportEvent(event)


class SpectraPanel(QWidget):
    """Tool buttons, spectra tree and Export / Clear buttons of the cube viewer sidebar.

    Construct with no arguments; GUI thread only.  See the module docstring for the
    layout, the tree and the keys.

    Attributes:
        tree: The ``QTreeWidget`` (columns Name | N px).
        btn_export: 'Export...' button (emits ``export_requested``).
        btn_clear_points: 'Clear All Points' button; the window connects it.

    Signals:
        tool_changed(str): ``'point'``, ``'rect'`` or ``'lasso'`` was chosen.
        visibility_changed(int, bool): ``(rid, visible)`` from the check box.
        rename_requested(int, str): ``(rid, text)`` typed in the tree (not yet validated).
        color_requested(int, str): ``(rid, '#rrggbb')`` chosen with 'Set colour...'.
        show_std_changed(int, bool): ``(rid, show)`` from 'Show +-sigma'.
        copy_requested(list): rids for 'Copy as text'.
        delete_requested(list): rids to delete (Del key or the context menu).
        selection_changed(object): the highlighted rid (current selected row) or None.
        export_requested(): the 'Export...' button was clicked.
        class_requested(int, str): ``(rid, class name)`` typed in the Class column
            (already stripped and non-empty).
        segment_requested(): the 'Segment...' button was clicked.
        segmentation_overlay_requested(bool): Show / Hide overlay from the Segmentation
            node's context menu.
        clear_segmentation_requested(): 'Clear segmentation' from that menu.
    """

    tool_changed = pyqtSignal(str)
    visibility_changed = pyqtSignal(int, bool)
    rename_requested = pyqtSignal(int, str)
    color_requested = pyqtSignal(int, str)
    show_std_changed = pyqtSignal(int, bool)
    show_median_changed = pyqtSignal(int, bool)
    copy_requested = pyqtSignal(list)
    delete_requested = pyqtSignal(list)
    selection_changed = pyqtSignal(object)
    export_requested = pyqtSignal()
    class_requested = pyqtSignal(int, str)
    segment_requested = pyqtSignal()
    segmentation_overlay_requested = pyqtSignal(bool)
    clear_segmentation_requested = pyqtSignal()
    segment_ready_changed = pyqtSignal(bool)
    view_mode_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._records: dict = {}             # rid -> record (held by reference)
        self._items: dict = {}               # rid -> QTreeWidgetItem
        self._notes: dict = {}               # rid -> failure reason
        self._icons: dict = {}               # '#rrggbb' -> QIcon
        self._updating = False
        self._band = 0
        self._band_text = "band 0"
        self._image_hw = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        tool_strip = QWidget()
        tool_strip.setStyleSheet(_TOOL_GROUP_STYLE)
        tool_row = QHBoxLayout(tool_strip)
        tool_row.setContentsMargins(0, 0, 0, 0)
        tool_row.setSpacing(0)
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_buttons: dict = {}
        segments = ["first"] + ["middle"] * (len(TOOL_BUTTONS) - 2) + ["last"]
        for (name, text, icon, tip), segment in zip(TOOL_BUTTONS, segments):
            button = QPushButton()
            button.setIcon(qta.icon(icon, color=PLOT_FOREGROUND, color_on="#ffffff"))
            button.setIconSize(QSize(18, 18))
            button.setFixedSize(32, 26)
            button.setCheckable(True)
            button.setToolTip(f"{text} - {tip}")
            button.setAccessibleName(text)
            button.setProperty("segment", segment)
            button.setFocusPolicy(Qt.TabFocus)
            self._tool_group.addButton(button)
            self._tool_buttons[name] = button
            tool_row.addWidget(button)
        tool_row.addStretch()
        self._tool_buttons["point"].setChecked(True)
        self._tool_group.buttonToggled.connect(self._on_tool_toggled)
        lay.addWidget(tool_strip)

        self.tree = _SpectraTree()
        self.tree.set_tooltip_provider(self._item_tooltip)
        # Cubes > <cube> > Band / RGB (Spectronon style): a click switches the view
        self._cubes_root = QTreeWidgetItem(self.tree, ["Cubes", ""])
        self._cubes_root.setFlags(Qt.ItemIsEnabled)
        self._cube_item = QTreeWidgetItem(self._cubes_root, ["cube", ""])
        self._cube_item.setFlags(Qt.ItemIsEnabled)
        self._cube_item.setIcon(0, qta.icon("mdi6.cube-outline", color=PLOT_FOREGROUND))
        self._view_items = {}
        for mode, text, icon in (("single", "Band", "mdi6.image-filter-black-white"),
                                 ("rgb", "RGB", "mdi6.palette")):
            item = QTreeWidgetItem(self._cube_item, [text, ""])
            item.setFlags(Qt.ItemIsEnabled)
            item.setIcon(0, qta.icon(icon, color=PLOT_FOREGROUND))
            item.setData(0, _VIEW_ROLE, mode)
            self._view_items[mode] = item
        self._cubes_root.setExpanded(True)
        self._cube_item.setExpanded(True)
        self._root = QTreeWidgetItem(self.tree, ["Spectra (0)", ""])
        self._root.setFlags(Qt.ItemIsEnabled)
        font = self._root.font(0)
        font.setBold(True)
        self._root.setFont(0, font)
        self._root.setExpanded(True)
        self._cubes_root.setFont(0, font)
        self.set_view_mode("single")
        self._seg_root = QTreeWidgetItem(self.tree, ["Segmentation", "", ""])
        self._seg_root.setFlags(Qt.ItemIsEnabled)
        self._seg_root.setFont(0, font)
        self._seg_root.setHidden(True)
        self._seg_summary = None
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.delete_pressed.connect(self._delete_selected)
        self.tree.rename_pressed.connect(self._rename_current)
        lay.addWidget(self.tree, stretch=1)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(6)
        self.btn_export = QPushButton("Export…")
        self.btn_export.setMinimumHeight(32)
        self.btn_export.setMinimumWidth(64)
        self.btn_export.setToolTip("Save the checked, computed spectra as CSV or Excel (XLSX)")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_requested)
        self.btn_clear_points = QPushButton("Clear All Points")
        self.btn_clear_points.setMinimumHeight(32)
        button_row.addWidget(self.btn_export)
        button_row.addWidget(self.btn_clear_points)
        lay.addLayout(button_row)

        self.btn_segment = QPushButton("Segment…")
        self.btn_segment.setMinimumHeight(32)
        self.btn_segment.setToolTip(_SEGMENT_TIP)
        self.btn_segment.setEnabled(False)
        self.btn_segment.clicked.connect(self.segment_requested)
        lay.addWidget(self.btn_segment)

    # ── tools ─────────────────────────────────────────────

    def tool(self) -> str:
        """The checked tool: ``'point'``, ``'rect'`` or ``'lasso'``."""
        for name, button in self._tool_buttons.items():
            if button.isChecked():
                return name
        return "point"

    def set_tool(self, name: str) -> None:
        """Check the button of tool ``name`` (emits ``tool_changed`` when it changes).

        Raises:
            ValueError: Unknown tool name.
        """
        if name not in self._tool_buttons:
            raise ValueError(f"tool must be one of {tuple(self._tool_buttons)}, got {name!r}")
        self._tool_buttons[name].setChecked(True)

    def tool_button(self, name: str) -> QPushButton:
        """The button of tool ``name`` (``KeyError`` for an unknown name)."""
        return self._tool_buttons[name]

    def _on_tool_toggled(self, button, checked: bool) -> None:
        """Emit ``tool_changed`` for the newly checked button."""
        if not checked:
            return
        for name, candidate in self._tool_buttons.items():
            if candidate is button:
                self.tool_changed.emit(name)
                return

    # ── records ───────────────────────────────────────────

    def add_record(self, record) -> None:
        """Append a row for ``record`` (a ``SpectrumRecord``, held by reference)."""
        if record.rid in self._items:
            self.update_record(record)
            return
        item = QTreeWidgetItem()
        item.setData(0, _RID_ROLE, int(record.rid))
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
                      | Qt.ItemIsEditable | Qt.ItemNeverHasChildren)
        item.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
        self._records[record.rid] = record
        self._items[record.rid] = item
        self._updating = True
        try:
            self._root.addChild(item)
        finally:
            self._updating = False
        self._root.setExpanded(True)
        self.update_record(record)
        self._update_root()

    def update_record(self, record, note=None) -> None:
        """Redraw the row of ``record`` from the record (name, check box, colour, count).

        Args:
            record: The record (its row must exist; unknown records are ignored).
            note: Optional reason shown in the tooltip of a failed record; ``None`` keeps
                the previous note.
        """
        item = self._items.get(record.rid)
        if item is None:
            return
        self._records[record.rid] = record
        if note is not None:
            self._notes[record.rid] = str(note)
        self._updating = True
        try:
            if item.text(0) != record.name:
                item.setText(0, record.name)
            check = Qt.Checked if record.visible else Qt.Unchecked
            if item.checkState(0) != check:
                item.setCheckState(0, check)
            item.setIcon(0, self._swatch(record.color))
            if item.text(2) != record.class_name:
                item.setText(2, record.class_name)
            item.setText(1, self._count_text(record))
            item.setForeground(1, _FAILED_COLOR if record.state == "failed" else
                               self.tree.palette().text())
            font = item.font(1)
            font.setItalic(record.state == "pending")
            item.setFont(1, font)
        finally:
            self._updating = False
        self._update_segment_button()

    def remove_record(self, rid: int) -> None:
        """Remove the row of ``rid`` (nothing happens for an unknown rid)."""
        item = self._items.pop(rid, None)
        self._records.pop(rid, None)
        self._notes.pop(rid, None)
        if item is None:
            return
        self._updating = True
        try:
            self._root.removeChild(item)
        finally:
            self._updating = False
        self._update_root()
        self._update_segment_button()
        self._on_selection_changed()

    def clear(self) -> None:
        """Remove every row."""
        self._updating = True
        try:
            self._root.takeChildren()
        finally:
            self._updating = False
        self._items.clear()
        self._records.clear()
        self._notes.clear()
        self._update_root()
        self._update_segment_button()
        self._on_selection_changed()

    def record_ids(self) -> list:
        """rids of the rows, top to bottom (creation order)."""
        return [self._root.child(i).data(0, _RID_ROLE) for i in range(self._root.childCount())]

    def item(self, rid: int):
        """The ``QTreeWidgetItem`` of ``rid``, or None."""
        return self._items.get(rid)

    def selected_rids(self) -> list:
        """rids of the selected rows, top to bottom."""
        rids = []
        for i in range(self._root.childCount()):
            child = self._root.child(i)
            rid = child.data(0, _RID_ROLE)
            if child.isSelected() and rid in self._items:
                rids.append(rid)
        return rids

    def current_rid(self):
        """The highlighted rid: the current row if it is selected, else the first selected
        row, else None."""
        current = self.tree.currentItem()
        if current is not None and current.isSelected():
            rid = current.data(0, _RID_ROLE)
            if rid in self._items:
                return rid
        selected = self.selected_rids()
        return selected[0] if selected else None

    def select(self, rid) -> None:
        """Make ``rid`` the only selected and current row (None clears the selection)."""
        self.tree.clearSelection()
        item = self._items.get(rid) if rid is not None else None
        if item is not None:
            self.tree.setCurrentItem(item, 0)
            item.setSelected(True)

    def edit_name(self, rid: int) -> bool:
        """Open the in-place name editor of ``rid``; True if it opened."""
        item = self._items.get(rid)
        if item is None:
            return False
        self.tree.setCurrentItem(item, 0)
        self.tree.editItem(item, 0)
        return True

    def edit_class(self, rid: int) -> bool:
        """Open the in-place Class editor of ``rid``; True if it opened."""
        item = self._items.get(rid)
        if item is None:
            return False
        self.tree.setCurrentItem(item, 2)
        self.tree.editItem(item, 2)
        return True

    def ready_class_names(self) -> list:
        """Distinct class names of the ready records, in row order."""
        names: dict = {}
        for rid in self.record_ids():
            record = self._records.get(rid)
            if record is not None and record.is_ready:
                names.setdefault(record.class_name, None)
        return list(names)

    def _update_segment_button(self) -> None:
        """Enable 'Segment...' when >= 2 classes have ready spectra."""
        ready = len(self.ready_class_names()) >= 2
        self.btn_segment.setEnabled(ready)
        if ready != getattr(self, "_segment_ready", None):
            self._segment_ready = ready
            self.segment_ready_changed.emit(ready)

    # ── segmentation node ─────────────────────────────────

    def set_segmentation(self, summary) -> None:
        """Show (or with None remove) the 'Segmentation' node.

        Args:
            summary: ``SegmentSession.summary()`` dict (``label``, ``names``, ``colors``,
                ``counts``, ``n_unclassified``, ``total``, ``stale``, ``overlay_visible``)
                or None.  One child per class (swatch, predicted px, %) plus
                'unclassified'; a stale result gets a '(stale)' badge.
        """
        self._seg_summary = summary
        root = self._seg_root
        self._updating = True
        try:
            root.takeChildren()
            if summary is None:
                root.setText(0, "Segmentation")
                root.setHidden(True)
                return
            total = max(int(summary["total"]), 1)
            badge = "  (stale)" if summary.get("stale") else ""
            hidden = "" if summary.get("overlay_visible", True) else "  [hidden]"
            root.setText(0, f"Segmentation – {summary['label']}{badge}{hidden}")
            root.setForeground(0, _STALE_COLOR if summary.get("stale") else
                               self.tree.palette().text())
            rows = list(zip(summary["names"], summary["colors"], summary["counts"]))
            for k, (name, color, count) in enumerate(rows):
                child = QTreeWidgetItem([str(name), f"{int(count):,}",
                                         f"{100.0 * int(count) / total:.1f} %"])
                child.setIcon(0, self._swatch(color))
                child.setData(0, _SEG_ROLE, k)
                child.setFlags(Qt.ItemIsEnabled | Qt.ItemNeverHasChildren)
                child.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
                root.addChild(child)
            n_unc = int(summary["n_unclassified"])
            child = QTreeWidgetItem(["unclassified", f"{n_unc:,}", f"{100.0 * n_unc / total:.1f} %"])
            child.setData(0, _SEG_ROLE, -1)
            child.setFlags(Qt.ItemIsEnabled | Qt.ItemNeverHasChildren)
            child.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
            root.addChild(child)
            root.setHidden(False)
            root.setExpanded(True)
        finally:
            self._updating = False

    def segmentation_item(self):
        """The 'Segmentation' root item (hidden when there is no result)."""
        return self._seg_root

    def segmentation_rows(self) -> list:
        """``[(name, predicted px, percent text)]`` of the Segmentation node's children."""
        root = self._seg_root
        return [(root.child(i).text(0), int(root.child(i).text(1).replace(",", "")),
                 root.child(i).text(2)) for i in range(root.childCount())]

    def segmentation_menu(self):
        """Build (not show) the Segmentation node's menu, or None without a result.

        Actions: Show overlay (checkable), Clear segmentation.
        """
        summary = self._seg_summary
        if summary is None:
            return None
        menu = QMenu(self)
        show = menu.addAction("Show overlay")
        show.setCheckable(True)
        show.setChecked(bool(summary.get("overlay_visible", True)))
        show.toggled.connect(lambda checked: self.segmentation_overlay_requested.emit(bool(checked)))
        menu.addSeparator()
        menu.addAction("Clear segmentation", self.clear_segmentation_requested.emit)
        return menu

    # ── band / tooltip ────────────────────────────────────

    def set_band(self, band: int, text: str | None = None) -> None:
        """Set the band the tooltips show (``text`` labels it, e.g. ``'700.0 nm (band 45)'``)."""
        self._band = int(band)
        self._band_text = text or f"band {self._band}"

    def set_image_size(self, height: int, width: int) -> None:
        """Image size used to clip the lasso bboxes shown in the tooltips."""
        self._image_hw = (int(height), int(width))

    def tooltip_text(self, rid: int) -> str:
        """Tooltip of ``rid``: name, kind, N px, bbox and mean +- std at the current band."""
        record = self._records.get(rid)
        if record is None:
            return ""
        lines = [record.name]
        if record.class_name != record.name:
            lines.append(f"class: {record.class_name}")
        geometry = record.geometry
        stats = record.stats
        if record.kind == "point":
            lines.append(f"point  ·  pixel x={geometry['x']}, y={geometry['y']}  ·  1 px")
        else:
            kind = ("rectangle" if record.kind == "rect"
                    else f"lasso, {len(geometry.get('vertices') or ())} vertices")
            count = (f"{int(stats.n_px):,} px" if stats is not None
                     else (_PENDING_TEXT if record.state == "pending" else "? px"))
            lines.append(f"{kind}  ·  {count}")
            c0, r0, c1, r1 = self._bbox(record)
            if c1 > c0 and r1 > r0:
                lines.append(f"bbox x {c0}–{c1 - 1}, y {r0}–{r1 - 1}  "
                             f"({c1 - c0}×{r1 - r0})")
        if record.state == "pending":
            if record.kind == "point":           # an area's count line already says so
                lines.append(_PENDING_TEXT)
        elif record.state == "failed" or stats is None:
            lines.append(f"failed: {self._notes.get(rid, 'no valid pixel in any band')}")
        else:
            lines.append(self._value_line(record))
        return "\n".join(lines)

    def _value_line(self, record) -> str:
        """``'<band>: mean +- std'`` of a ready record at the current band."""
        stats = record.stats
        band = self._band
        if not 0 <= band < stats.mean.shape[0]:
            return f"{self._band_text}: -"
        n_valid = int(stats.n_valid[band])
        if n_valid == 0:
            return f"{self._band_text}: no valid pixel"
        if record.kind == "point":
            text = f"{self._band_text}: {_fmt(stats.mean[band])}"
        else:
            text = f"{self._band_text}: {_fmt(stats.mean[band])} ± {_fmt(stats.std[band])}"
        extras = []
        if n_valid < int(stats.n_px):
            extras.append(f"{n_valid:,} valid px")
        n_sat = int(stats.n_saturated[band])
        if n_sat:
            extras.append(f"{n_sat:,} saturated")
        return text + (f"  ({', '.join(extras)})" if extras else "")

    def _bbox(self, record) -> tuple:
        """Half-open pixel bbox ``(c0, r0, c1, r1)`` of an area record."""
        geometry = record.geometry
        if record.kind == "rect":
            return tuple(int(geometry[k]) for k in ("c0", "r0", "c1", "r1"))
        vertices = np.asarray(geometry.get("vertices") or (), dtype=np.float64).reshape(-1, 2)
        if vertices.shape[0] == 0:
            return 0, 0, 0, 0
        height, width = self._image_hw or (_BIG, _BIG)
        return snap_rect(float(vertices[:, 0].min()), float(vertices[:, 1].min()),
                         float(vertices[:, 0].max()), float(vertices[:, 1].max()), height, width)

    def _item_tooltip(self, item) -> str:
        """Tooltip provider of the tree."""
        if item is self._root:
            return "Check = shown and exported.  F2 / double click: rename.  Del: delete.  " \
                   "Right click: colour, ±σ, copy."
        if item is self._cubes_root or item is self._cube_item:
            return "The open cube; click Band or RGB to switch the view."
        if item.data(0, _VIEW_ROLE) is not None:
            return f"Show the {item.text(0)} view"
        if item is self._seg_root:
            return ("Right click: show / hide the overlay or clear the segmentation.  "
                    "'(stale)': spectra or classes changed after the run.")
        if item.parent() is self._seg_root:
            return f"{item.text(0)}: {item.text(1)} px ({item.text(2)})"
        rid = item.data(0, _RID_ROLE)
        return self.tooltip_text(rid) if rid is not None else ""

    # ── context menu ──────────────────────────────────────

    def context_menu(self, rid: int):
        """Build (not show) the context menu of ``rid``; None for an unknown rid.

        Actions: Rename, Set colour..., Show +-sigma (checkable, areas only), Copy as text,
        Delete.  Copy and Delete act on the whole selection when ``rid`` is selected.
        """
        record = self._records.get(rid)
        if record is None:
            return None
        targets = self._targets(rid)
        menu = QMenu(self)
        menu.addAction("Rename", lambda: self.edit_name(rid))
        menu.addAction("Set colour…", lambda: self._choose_colour(rid))
        std = menu.addAction("Show ±σ")
        std.setCheckable(True)
        std.setChecked(bool(record.show_std) and record.is_area)
        std.setEnabled(record.is_area)
        std.toggled.connect(lambda checked: self._request_show_std(rid, checked))
        median = menu.addAction("Show median")
        median.setCheckable(True)
        median.setChecked(bool(getattr(record, "show_median", False)) and record.is_area)
        median.setEnabled(record.is_area)
        median.setToolTip("Dashed curve: the per-band median of the area's pixels "
                          "(computed once, needs every pixel)")
        median.toggled.connect(lambda checked: self.show_median_changed.emit(int(rid), bool(checked)))
        menu.addSeparator()
        menu.addAction("Copy as text", lambda: self.copy_requested.emit(list(targets)))
        menu.addSeparator()
        menu.addAction("Delete", lambda: self.delete_requested.emit(list(targets)))
        return menu

    def _targets(self, rid: int) -> list:
        """The selection when it contains ``rid``, else just ``rid``."""
        selected = self.selected_rids()
        return selected if rid in selected else [rid]

    def _on_context_menu(self, pos) -> None:
        """Show the context menu of the row under ``pos``."""
        item = self.tree.itemAt(pos)
        if item is not None and (item is self._seg_root or item.parent() is self._seg_root):
            menu = self.segmentation_menu()
        else:
            rid = item.data(0, _RID_ROLE) if item is not None else None
            menu = self.context_menu(rid) if rid is not None else None
        if menu is not None:
            menu.exec_(self.tree.viewport().mapToGlobal(pos))
            menu.deleteLater()

    def _choose_colour(self, rid: int) -> None:
        """Ask for a colour (``QColorDialog``) and emit ``color_requested``."""
        record = self._records.get(rid)
        if record is None:
            return
        color = QColorDialog.getColor(QColor(record.color), self, f"Colour of {record.name}")
        if color.isValid():
            self.color_requested.emit(int(rid), color.name())
        self._refresh(rid)

    def _request_show_std(self, rid: int, show: bool) -> None:
        """Emit ``show_std_changed`` and redraw the row."""
        self.show_std_changed.emit(int(rid), bool(show))
        self._refresh(rid)

    # ── tree signals ──────────────────────────────────────

    def _on_item_changed(self, item, column: int) -> None:
        """Turn a check-box toggle or an edited name into a request, then redraw the row."""
        if self._updating or column not in (0, 2):
            return
        rid = item.data(0, _RID_ROLE)
        record = self._records.get(rid)
        if record is None:
            return
        if column == 2:
            text = item.text(2).strip()
            if text and text != record.class_name:
                self.class_requested.emit(int(rid), text)
            self._refresh(rid)
            return
        visible = item.checkState(0) == Qt.Checked
        if visible != bool(record.visible):
            self.visibility_changed.emit(int(rid), visible)
        if item.text(0) != record.name:
            self.rename_requested.emit(int(rid), item.text(0))
        self._refresh(rid)

    def set_cube_name(self, name: str) -> None:
        """Name of the cube under the Cubes node (the file stem, or 'cube')."""
        self._cube_item.setText(0, str(name) or "cube")

    def set_view_mode(self, mode: str) -> None:
        """Mark ``'single'`` (Band) or ``'rgb'`` as the shown view (bold); emits nothing."""
        for key, item in self._view_items.items():
            font = item.font(0)
            font.setBold(key == mode)
            item.setFont(0, font)

    def view_mode(self) -> str:
        """The view marked as shown: ``'single'`` or ``'rgb'``."""
        for key, item in self._view_items.items():
            if item.font(0).bold():
                return key
        return "single"

    def _on_item_clicked(self, item, column: int) -> None:
        """A click on Cubes > <cube> > Band / RGB asks for that view."""
        mode = item.data(0, _VIEW_ROLE)
        if mode is not None:
            self.view_mode_requested.emit(mode)

    def _on_item_double_clicked(self, item, column: int) -> None:
        """A double click on the count column edits the name too."""
        if column == 1 and item.data(0, _RID_ROLE) in self._items:
            self.tree.editItem(item, 0)

    def _on_selection_changed(self) -> None:
        """Emit ``selection_changed`` with the highlighted rid (once a row change is done)."""
        if not self._updating:
            self.selection_changed.emit(self.current_rid())

    def _delete_selected(self) -> None:
        """Del key: request deletion of the selected rows."""
        rids = self.selected_rids()
        if rids:
            self.delete_requested.emit(rids)

    def _rename_current(self) -> None:
        """F2: edit the name of the current row."""
        rid = self.current_rid()
        if rid is not None:
            self.edit_name(rid)

    # ── helpers ───────────────────────────────────────────

    def _refresh(self, rid: int) -> None:
        """Redraw the row of ``rid`` from its record (reverts refused edits)."""
        record = self._records.get(rid)
        if record is not None:
            self.update_record(record)

    def _update_root(self) -> None:
        """Show the record count on the root row."""
        self._updating = True
        try:
            self._root.setText(0, f"Spectra ({len(self._items)})")
        finally:
            self._updating = False

    def _swatch(self, color: str) -> QIcon:
        """A small square icon in ``color`` (cached)."""
        icon = self._icons.get(color)
        if icon is None:
            pixmap = QPixmap(_SWATCH_PX, _SWATCH_PX)
            pixmap.fill(QColor(color))
            icon = QIcon(pixmap)
            self._icons[color] = icon
        return icon

    @staticmethod
    def _count_text(record) -> str:
        """'N px' column text: the pixel count, 'computing...' or 'failed'."""
        if record.state == "pending":
            return _PENDING_TEXT
        if record.state == "failed":
            return _FAILED_TEXT
        stats = record.stats
        return f"{int(stats.n_px):,}" if stats is not None else ""
