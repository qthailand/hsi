"""``RadiometricReferenceDialog``: save the scan just finished as a radiometric reference.

The user scanned a reference target (Spectralon), presses *Save as Radiometric Reference* and
gets this dialog: an RGB preview of the cube, five ways to choose the target's pixels, the
reflectance table, the quality checks and *Create profile*. Creating fits the row-wise factors
on the cube in memory (:func:`~modules.radiometric.core.compute_rowwise_factors_array`) and
stores them with the scan's camera setting (:class:`~modules.radiometric.reference.ReferenceKey`)
in a :class:`~modules.radiometric.reference.RadiometricReferenceStore`, optionally activating it
for automatic calibration of later scans.

ROI tools (see :mod:`modules.radiometric.roi_tools`):

* **Rect / Ellipse** - drag; afterwards the handles resize (edge handles stretch one side only,
  e.g. left/right), dragging inside moves; *Shift* while drawing keeps a square / circle;
* **Freehand** - draw a path; it becomes a polygon whose vertices can be dragged (right-click
  deletes one, double-click on an edge adds one) inside a bounding box whose handles stretch
  the whole shape; *Esc* starts a new path;
* **Detect** - the largest bright rectangle or ellipse, then editable like a drawn shape; other
  candidates are in the list;
* **Spectral pick** - click: the connected region of similar spectra; *Ctrl+click* adds a region,
  *Alt+click* removes one; the spectral-angle threshold is automatic or set by hand.

Keyboard (image focused): arrows move the shape 1 px (*Shift*: 10 px); *Ctrl+arrows* stretch
the right / bottom edge; *Ctrl+Z* undoes the last change; *Delete* clears.

Every mode then goes through :func:`~modules.radiometric.roi_tools.clean_mask` (saturation,
inset, spectral gate for other materials, row gate, row profile); the image shows the effective
ROI in teal and the removed pixels in red.

Data conventions:
    ``cube`` is ``(lines, samples, bands)`` dark-subtracted counts (not copied: do not change it
    while the dialog is open); ``wavelengths`` in nm; ``metadata`` is the scan's metadata (raw
    metadata merged with the export-header metadata: camera, exposure, corrections,
    ``started_at``, temperatures, ``full_scale``); ``saturation_ceiling`` the scan controller's
    ``(lines, bands)`` saturated-pixel value, if known.

Threading:
    GUI thread only. The cube-wide maps and the spectral index are computed when the dialog
    opens (a few seconds for a 552 x 3000 x 300 cube, with a wait cursor; the index holds about
    350 MB of float32 unit spectra); the factor fit runs when *Create profile* is pressed.

Example:
    >>> from modules.radiometric.reference_dialog import RadiometricReferenceDialog  # doctest: +SKIP
    >>> dlg = RadiometricReferenceDialog(cube, wavelengths, metadata, store, ceiling)   # doctest: +SKIP
    >>> if dlg.exec_():                                                                 # doctest: +SKIP
    ...     print(dlg.reference.id)
"""
from __future__ import annotations

import logging
import math
import os
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import EllipseSelector, LassoSelector, PolygonSelector, RectangleSelector
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                             QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QSpinBox, QToolButton, QVBoxLayout, QWidget)

from . import roi_tools as rt
from .core import compute_rowwise_factors_array, list_reflectance_profiles, reflectance_at
from .reference import ReferenceKey, scan_temperature_c

logger = logging.getLogger(__name__)

__all__ = ["RadiometricReferenceDialog", "TOOLS"]

#: ROI tools, in toolbar order.
TOOLS = ("rect", "ellipse", "freehand", "detect", "spectral")
#: Factor fit of the averaging window (see ``rowwise_calculation_heavy_logic``): a line through the
#: window's rows, unbiased where the vignetting falls off along the slit.
WINDOW_FIT = "linear"
#: Self-check limits (%) before *Create profile* asks: overall RMS, and the worst single slit row.
SELF_CHECK_RMS_PERCENT = 0.5
SELF_CHECK_ROW_PERCENT = 1.0
#: Rows with fewer ROI pixels do not decide the worst-row self-check (their own noise would).
SELF_CHECK_MIN_ROW_PX = 20
#: Automatic averaging window: the fewest rows giving about this many ROI pixels per factor
#: (pixel scatter 2 % -> 0.09 % factor noise).
AUTO_WINDOW_PIXELS = 500
_TOOL_LABELS = {"rect": "▭ Rect", "ellipse": "◯ Ellipse", "freehand": "✎ Freehand",
                "detect": "◎ Detect", "spectral": "✦ Spectral pick"}
_SHAPE = dict(edgecolor="#E9A23B", facecolor="none", linewidth=1.6)
_HANDLES = dict(markersize=7, markerfacecolor="white", markeredgecolor="#1B2430")
_NO_PROFILE = "— choose the reference target's table —"
_KEEP_RGBA = (0.16, 0.62, 0.56, 0.38)        # effective ROI (teal)
_DROP_RGBA = (0.82, 0.29, 0.36, 0.55)        # drawn but removed by the clean-up (red)


class RadiometricReferenceDialog(QDialog):
    """Choose the reference-target pixels of a scan and store the factors as a reference.

    Args:
        cube: ``(lines, samples, bands)`` counts of the reference scan.
        wavelengths: Band centres, nm.
        metadata: The scan's metadata (see the module docstring); its camera setting becomes
            the reference's key.
        store: ``RadiometricReferenceStore`` to save into; ``None`` disables *Create profile*.
        saturation_ceiling: Optional ``(lines, bands)`` saturated-pixel value of the scan.
        parent: Parent widget.
        profiles: Reflectance tables to offer (default: ``list_reflectance_profiles()``).

    Signals:
        reference_saved(object): the stored ``RadiometricReference``.
    """

    reference_saved = pyqtSignal(object)

    def __init__(self, cube, wavelengths, metadata=None, store=None, saturation_ceiling=None,
                 parent=None, profiles=None):
        super().__init__(parent)
        self.setWindowTitle("Save as Radiometric Reference")
        self.cube = cube
        lines, samples, bands = cube.shape
        self.shape = (lines, samples)
        self.wavelengths = (np.asarray(wavelengths, dtype=np.float64).ravel() if wavelengths is not None
                            else np.arange(bands, dtype=np.float64))
        self.metadata = dict(metadata or {})
        self.store = store
        self.key = ReferenceKey.from_scan(self.metadata, wavelengths, lines)
        self.reference = None
        ceiling = saturation_ceiling if np.shape(saturation_ceiling) == (lines, bands) else None
        self.ceiling = None if ceiling is None else np.array(ceiling, dtype=np.float32)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.saturated, self.saturation_source = rt.saturation_map(
                cube, ceiling=self.ceiling, full_scale=self.metadata.get("full_scale"))
            # unit spectra: Spectral pick, and the spectral gate of every mode's clean-up
            self._index: Optional[rt.SpectralIndex] = rt.SpectralIndex(cube, self.wavelengths, self.saturated)
            self.brightness = self._index.brightness
            self.rgb = rt.rgb_preview(cube, self.wavelengths)
        except BaseException:
            # e.g. MemoryError on a large cube: do not leave a half-built dialog parented to the
            # main window that keeps the cube alive for the rest of the session
            self.cube = self.saturated = self._index = self.brightness = self.rgb = None
            self.setParent(None)
            self.deleteLater()
            raise
        finally:
            QApplication.restoreOverrideCursor()
        self._candidates: Optional[List[rt.ShapeCandidate]] = None
        self._tool = "rect"
        self._selector = None
        self._selector_kind = None          # "rect" | "ellipse" | "polygon" | "lasso" | None
        self._has_shape = False             # the selector holds a shape (not just an empty tool)
        self._committed = None              # undo entry (tool, state, extras) after the last update
        self._pending_undo = None           # the entry before a change in progress (see _mark)
        self._spectral = np.zeros(self.shape, dtype=bool)
        self._seeds: list = []              # [(row, col, "+"|"-", threshold_deg, auto)]
        self._undo: list = []
        self._restoring = False
        self._table_check = None            # (path, out-of-range bands, table range, error)
        self._roi_note = None               # detected candidate behind the ROI (Detect tool)
        self.drawn = np.zeros(self.shape, dtype=bool)
        self.cleanup: Optional[rt.CleanupResult] = None
        self.quality: Optional[rt.RoiQuality] = None
        self._build_ui(profiles)
        self._set_tool("rect")
        self._update_roi()

    # ── layout ──────────────────────────────────────────────────────────
    def _build_ui(self, profiles):
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        root.addLayout(left, 3)

        tools = QHBoxLayout()
        self.tool_group = QButtonGroup(self)
        self.tool_buttons = {}
        for name in TOOLS:
            button = QToolButton()
            button.setText(_TOOL_LABELS[name])
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            button.clicked.connect(lambda _=False, n=name: self._set_tool(n))
            self.tool_group.addButton(button)
            self.tool_buttons[name] = button
            tools.addWidget(button)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_roi)
        tools.addWidget(self.clear_btn)
        tools.addStretch(1)
        left.addLayout(tools)

        context = QHBoxLayout()
        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        context.addWidget(self.hint_label, 1)
        self.shape_combo = QComboBox()
        self.shape_combo.setMinimumWidth(220)
        self.shape_combo.activated.connect(self._on_candidate_chosen)
        context.addWidget(self.shape_combo)
        self.auto_threshold_box = QCheckBox("Auto threshold")
        self.auto_threshold_box.setChecked(True)
        self.auto_threshold_box.toggled.connect(self._on_threshold_mode)
        context.addWidget(self.auto_threshold_box)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.2, 10.0)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSuffix(" °")
        self.threshold_spin.setValue(rt.DEFAULT_SAM_THRESHOLD_DEG)
        self.threshold_spin.setEnabled(False)
        self.threshold_spin.editingFinished.connect(self._redo_spectral)
        context.addWidget(self.threshold_spin)
        left.addLayout(context)

        self.figure = Figure(figsize=(7.5, 5.5), layout="tight")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.ax = self.figure.add_subplot(111)
        self.ax.imshow(self.rgb, interpolation="nearest", aspect="auto")
        self._overlay = self.ax.imshow(np.zeros(self.shape + (4,), np.float32), interpolation="nearest",
                                       aspect="auto")
        self.ax.set_xlabel("scan frame (sample)")
        self.ax.set_ylabel("slit row (line)")
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        left.addWidget(self.toolbar)
        left.addWidget(self.canvas, 1)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("key_press_event", self._on_key)

        right = QVBoxLayout()
        root.addLayout(right, 1)
        form = QFormLayout()
        profile_row = QHBoxLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.addItem(_NO_PROFILE, None)
        for path in (profiles if profiles is not None else list_reflectance_profiles()):
            self.profile_combo.addItem(os.path.splitext(os.path.basename(str(path)))[0], str(path))
        self.profile_combo.currentIndexChanged.connect(lambda *_: self._update_roi(redraw=False))
        profile_row.addWidget(self.profile_combo, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_profile)
        profile_row.addWidget(browse)
        form.addRow("Reflectance profile", profile_row)
        window_row = QHBoxLayout()
        self.avg_window_spin = QSpinBox()
        self.avg_window_spin.setRange(1, 1000)
        self.avg_window_spin.setValue(1)
        self.avg_window_spin.setSuffix(" rows")
        self.avg_window_spin.setEnabled(False)
        self.auto_window_box = QCheckBox("Auto")
        self.auto_window_box.setChecked(True)
        self.auto_window_box.setToolTip(
            f"The fewest slit rows that give about {AUTO_WINDOW_PIXELS} ROI pixels per factor: each row keeps "
            "its own factor where the ROI is wide (slit dust and fixed row patterns are corrected), more rows "
            "only where it is narrow.")
        self.auto_window_box.toggled.connect(lambda on: (self.avg_window_spin.setEnabled(not on),
                                                         self._update_auto_window()))
        window_row.addWidget(self.avg_window_spin, 1)
        window_row.addWidget(self.auto_window_box)
        form.addRow("Row averaging window", window_row)
        self.inset_spin = QSpinBox()
        self.inset_spin.setRange(0, 50)
        self.inset_spin.setValue(rt.DEFAULT_INSET_PX)
        self.inset_spin.setSuffix(" px")
        self.inset_spin.valueChanged.connect(lambda *_: self._update_roi())
        form.addRow("Edge inset", self.inset_spin)
        self.gate_spin = QDoubleSpinBox()
        self.gate_spin.setRange(0.0, 50.0)
        self.gate_spin.setDecimals(1)
        self.gate_spin.setValue(rt.DEFAULT_ROW_TOLERANCE * 100)
        self.gate_spin.setSuffix(" %")
        self.gate_spin.setSpecialValueText("off")
        self.gate_spin.setToolTip("Pixels whose brightness differs more than this from the median of "
                                  "their slit row's ROI pixels are left out (shadow, bevel, dust).")
        self.gate_spin.valueChanged.connect(lambda *_: self._update_roi())
        form.addRow("Row gate", self.gate_spin)
        self.spectral_gate_box = QCheckBox("Leave out other materials")
        self.spectral_gate_box.setChecked(True)
        self.spectral_gate_box.setToolTip("Pixels whose spectrum departs from the ROI's own spectrum of their "
                                          "slit row (a label, tape, the holder edge: as bright as the target) "
                                          "are left out.")
        self.spectral_gate_box.toggled.connect(lambda *_: self._update_roi())
        form.addRow("Spectral gate", self.spectral_gate_box)
        right.addLayout(form)

        qc_group = QGroupBox("Quality checks")
        qc_layout = QVBoxLayout(qc_group)
        self.qc_label = QLabel()
        self.qc_label.setWordWrap(True)
        self.qc_label.setTextFormat(Qt.RichText)
        qc_layout.addWidget(self.qc_label)
        right.addWidget(qc_group)

        self.spectrum_figure = Figure(figsize=(3.6, 1.9), layout="tight")
        self.spectrum_canvas = FigureCanvas(self.spectrum_figure)
        self.spectrum_ax = self.spectrum_figure.add_subplot(111)
        right.addWidget(self.spectrum_canvas)

        key_group = QGroupBox("Camera setting (reference key)")
        key_layout = QVBoxLayout(key_group)
        self.key_label = QLabel(self._key_text())
        self.key_label.setWordWrap(True)
        key_layout.addWidget(self.key_label)
        right.addWidget(key_group)

        self.activate_box = QCheckBox("Activate automatic radiometric calibration")
        self.activate_box.setChecked(True)
        self.activate_box.setToolTip("Later scans with exactly this camera setting (scan length and "
                                     "speed may differ) are calibrated with this reference.")
        right.addWidget(self.activate_box)
        right.addStretch(1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.create_btn = QPushButton("Create profile")
        self.create_btn.clicked.connect(self.create_profile)
        # Enter in a spin box must not store a reference: no default button at all
        for button in self.findChildren(QPushButton) + [self.cancel_btn, self.create_btn]:
            button.setAutoDefault(False)
            button.setDefault(False)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.create_btn)
        right.addLayout(buttons)
        self.resize(1400, 820)

    def keyPressEvent(self, event):
        """Enter / Return never accept the dialog (only *Create profile* does); Esc still cancels."""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)

    def done(self, result):
        """Close, and drop the large arrays (cube reference, spectral index, maps): a closed
        dialog that is not deleted at once must not keep hundreds of MB alive."""
        super().done(result)
        self._remove_selector()
        self._index = None
        self.cube = None
        self.rgb = self.brightness = self.saturated = None
        self._undo.clear()
        self._committed = self._pending_undo = None

    def _key_text(self) -> str:
        k = self.key
        temp = scan_temperature_c(self.metadata)
        parts = [k.label(), f"Smile {'on' if k.smile_corrected else 'off'}",
                 f"dark {'on' if k.dark_corrected else 'off' if k.dark_corrected is False else 'unknown'}"]
        if temp is not None:
            parts.append(f"sensor {temp:.1f} °C")
        started = self.metadata.get("started_at")
        if started:
            parts.append(f"scanned {started}")
        return " · ".join(parts)

    # ── tools ───────────────────────────────────────────────────────────
    def _set_tool(self, tool: str):
        """Switch the ROI tool (one of :data:`TOOLS`). A drawn shape stays editable when switching
        between Rect / Ellipse / Freehand of the same kind; Spectral pick starts without a shape."""
        if tool not in TOOLS:
            raise ValueError(f"unknown tool {tool!r}")
        hints = {
            "rect": "Drag a rectangle. Handles resize (edge handles stretch one side, e.g. left/right), "
                    "drag inside to move. Arrows move, Ctrl+arrows stretch, Ctrl+Z undo.",
            "ellipse": "Drag an ellipse (Shift: circle in pixels). Handles stretch left/right or up/down; "
                       "the pixel aspect along the scan depends on the stage speed.",
            "freehand": "Draw around the target. Then drag vertices, right-click deletes one, double-click "
                        "on an edge adds one; the box handles stretch the whole shape. Esc: new path.",
            "detect": "The largest bright rectangle / ellipse; pick another candidate in the list, "
                      "then adjust it with its handles.",
            "spectral": "Click the target: its connected region of similar spectra. Ctrl+click adds, "
                        "Alt+click removes a region.",
        }
        self._hints = hints
        if self._selector_kind is not None or self._seeds:
            self._mark()                    # the shape or pick about to be replaced can be undone
        self._apply_tool_ui(tool)
        if tool in ("rect", "ellipse"):
            if self._selector_kind != tool:
                self._make_selector(tool)
        elif tool == "freehand":
            if self._selector_kind != "polygon":
                self._make_selector("lasso")
        elif tool == "detect":
            self._run_detection()
        else:
            self._remove_selector()
        self.canvas.setFocus()
        if self.cleanup is not None:
            self._update_roi()

    def _apply_tool_ui(self, tool: str):
        """Tool button, hint and the tool's own controls (no selector change)."""
        self._tool = tool
        self.tool_buttons[tool].setChecked(True)
        hints = getattr(self, "_hints", None)
        if hints:
            self.hint_label.setText(hints[tool])
        self.shape_combo.setVisible(tool == "detect")
        self.auto_threshold_box.setVisible(tool == "spectral")
        self.threshold_spin.setVisible(tool == "spectral")

    def _remove_selector(self):
        if self._selector is not None:
            box = getattr(self._selector, "_box", None)          # a polygon's bounding-box selector
            if box is not None:
                box.set_active(False)
                box.set_visible(False)
                box.disconnect_events()
                self._selector._box = None
            self._selector.set_active(False)
            self._selector.set_visible(False)
            self._selector.disconnect_events()
            self._selector = None
            self._selector_kind = None
            self.canvas.draw_idle()
        self._has_shape = False

    def _make_selector(self, kind: str, geometry=None):
        """Create the editable selector ``kind`` (``rect``, ``ellipse``, ``polygon`` or ``lasso``),
        optionally already holding ``geometry`` (extents or vertices)."""
        self._remove_selector()
        common = dict(useblit=True, interactive=True, drag_from_anywhere=True, ignore_event_outside=False,
                      use_data_coordinates=True, props=dict(_SHAPE), handle_props=dict(_HANDLES),
                      minspanx=2, minspany=2, spancoords="data")
        if kind == "rect":
            self._selector = RectangleSelector(self.ax, self._on_shape_changed, **common)
        elif kind == "ellipse":
            self._selector = EllipseSelector(self.ax, self._on_shape_changed, **common)
        elif kind == "polygon":
            self._selector = PolygonSelector(self.ax, self._on_polygon_changed, useblit=True,
                                             props=dict(color=_SHAPE["edgecolor"], linewidth=1.6),
                                             handle_props=dict(_HANDLES), draw_bounding_box=True,
                                             box_props=dict(edgecolor="#1B2430", facecolor="none",
                                                            linestyle=":", linewidth=0.9),
                                             box_handle_props=dict(_HANDLES))
        elif kind == "lasso":
            self._selector = LassoSelector(self.ax, self._on_lasso, useblit=True,
                                           props=dict(color=_SHAPE["edgecolor"], linewidth=1.6))
        else:
            raise ValueError(kind)
        self._selector_kind = kind
        self._has_shape = geometry is not None and kind != "lasso"
        if geometry is not None:
            if kind == "polygon":
                self._selector.verts = [tuple(map(float, p)) for p in geometry]
            else:
                self._selector.extents = tuple(map(float, geometry))
                self._selector.set_visible(True)
        self.canvas.draw_idle()

    # ── geometry state (undo, saving) ────────────────────────────────────
    def _state(self):
        """What the ROI is made of now: ``(kind, geometry, seeds, spectral mask)``."""
        kind = self._selector_kind
        geometry = None
        if kind in ("rect", "ellipse") and self._has_shape:
            geometry = tuple(float(v) for v in self._selector.extents)
        elif kind == "polygon":
            geometry = [tuple(map(float, p)) for p in self._selector.verts]
        if self._seeds or self._tool == "spectral":
            # a spectral pick stays the ROI until a shape is drawn, whichever tool is selected
            return ("spectral", None, list(self._seeds), self._spectral.copy())
        return (kind, geometry, [], None)

    def _snapshot(self):
        """Undo entry of the ROI now: ``(tool, state, extras)`` with the Detect note and the
        threshold mode / value, which undo restores too."""
        extras = (self._roi_note, bool(self.auto_threshold_box.isChecked()), float(self.threshold_spin.value()))
        return (self._tool, self._state(), extras)

    def _mark(self):
        """Start a change: the ROI as it is now becomes undoable once the next update shows that
        something really changed (no undo step for changes that change nothing). Selector callbacks
        come after the change; then the last committed ROI is the one before it."""
        if not self._restoring and self._pending_undo is None and self._committed is not None:
            self._pending_undo = self._committed

    def _commit(self):
        """After an ROI update: record the marked state if the ROI differs from it."""
        now = self._snapshot()
        before, self._pending_undo = self._pending_undo, None
        if before is not None and not self._restoring and not _same_state(before, now):
            if not (self._undo and _same_state(self._undo[-1], before)):
                self._undo.append(before)
                del self._undo[:-20]
        self._committed = now

    def undo(self):
        """Go back to the ROI before the last change (with its tool, controls and threshold mode)."""
        if not self._undo:
            return
        tool, (kind, geometry, seeds, spectral), (note, auto, threshold) = self._undo.pop()
        self._restoring = True
        self._pending_undo = None
        try:
            self._apply_tool_ui(tool)
            for widget in (self.auto_threshold_box, self.threshold_spin):
                widget.blockSignals(True)
            self.auto_threshold_box.setChecked(auto)
            self.threshold_spin.setValue(threshold)
            self.threshold_spin.setEnabled(not auto)
            for widget in (self.auto_threshold_box, self.threshold_spin):
                widget.blockSignals(False)
            self._roi_note = note
            if kind == "spectral":
                self._remove_selector()
                self._seeds, self._spectral = list(seeds), spectral.copy()
            elif kind in ("rect", "ellipse", "polygon") and geometry is not None:
                self._seeds, self._spectral = [], np.zeros(self.shape, dtype=bool)
                self._make_selector(kind, geometry)
            else:
                self._seeds, self._spectral = [], np.zeros(self.shape, dtype=bool)
                self._make_selector(kind if kind in ("rect", "ellipse", "lasso") else "rect")
            self._update_roi()
        finally:
            self._restoring = False
        self._committed = self._snapshot()

    def clear_roi(self):
        """Remove the ROI (the current tool stays selected)."""
        self._mark()
        self._seeds, self._spectral = [], np.zeros(self.shape, dtype=bool)
        self._roi_note = None
        tool = self._tool
        self._remove_selector()
        if tool in ("rect", "ellipse"):
            self._make_selector(tool)
        elif tool in ("freehand", "detect"):
            self._make_selector("lasso" if tool == "freehand" else "rect")
        self._update_roi()

    # ── selector callbacks ──────────────────────────────────────────────
    def _on_shape_changed(self, *_):
        self._mark()
        x0, x1, y0, y1 = self._selector.extents
        self._has_shape = (x1 - x0) > 2 and (y1 - y0) > 2      # a click without a drag clears it
        self._seeds, self._spectral = [], np.zeros(self.shape, dtype=bool)
        self._update_roi()

    def _on_polygon_changed(self, *_):
        self._mark()
        self._has_shape = len(self._selector.verts) >= 3
        self._seeds, self._spectral = [], np.zeros(self.shape, dtype=bool)
        self._update_roi()

    def _on_lasso(self, vertices):
        points = rt.simplify_polyline(vertices, 0.5)
        if len(points) < 3:
            return
        self._mark()
        self._make_selector("polygon", points)
        self._seeds, self._spectral = [], np.zeros(self.shape, dtype=bool)
        self._update_roi()

    def set_polygon(self, vertices):
        """Use ``[(x, y), ...]`` as an editable freehand polygon (as if drawn)."""
        self._mark()
        self._set_tool("freehand")
        self._on_lasso(vertices)

    def set_rect(self, x0, x1, y0, y1, tool="rect"):
        """Use the rectangle (``tool="ellipse"``: the ellipse inscribed in it) with extents ``x0..x1``,
        ``y0..y1`` (pixel-centre coordinates) as the ROI, editable with its handles."""
        self._mark()
        self._set_tool(tool)
        self._make_selector(tool, (x0, x1, y0, y1))
        self._seeds, self._spectral = [], np.zeros(self.shape, dtype=bool)
        self._update_roi()

    # ── detection ───────────────────────────────────────────────────────
    def _run_detection(self):
        if self._candidates is None:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self._candidates = rt.detect_shapes(self.brightness, self.saturated, inset_px=0)
            finally:
                QApplication.restoreOverrideCursor()
            self.shape_combo.clear()
            for c in self._candidates:
                self.shape_combo.addItem(c.label())
        shapes = [i for i, c in enumerate(self._candidates) if c.kind in ("rectangle", "ellipse")]
        if not shapes:
            self.hint_label.setText("No rectangle or ellipse found: draw the ROI, or use Spectral pick.")
            self._remove_selector()
            self._update_roi()
            return
        best = max(shapes, key=lambda i: self._candidates[i].area_px)
        self.shape_combo.setCurrentIndex(best)
        self._apply_candidate(self._candidates[best])

    def _on_candidate_chosen(self, index: int):
        if self._candidates and 0 <= index < len(self._candidates):
            candidate = self._candidates[index]
            if candidate.kind == "irregular":
                self.hint_label.setText("That candidate is irregular; use Freehand or Spectral pick for it.")
                return
            self._apply_candidate(candidate)

    def _apply_candidate(self, candidate: rt.ShapeCandidate):
        """Make a detected shape the ROI: a rectangle / ellipse selector when it is axis-aligned,
        else a polygon along its outline, editable either way."""
        self._mark()
        outline = candidate.outline
        x0, y0 = outline.min(axis=0)
        x1, y1 = outline.max(axis=0)
        if candidate.axis_aligned and candidate.kind == "rectangle":
            self._make_selector("rect", (x0, x1, y0, y1))
        elif candidate.axis_aligned and candidate.kind == "ellipse":
            self._make_selector("ellipse", (x0, x1, y0, y1))
        else:
            self._make_selector("polygon", outline)
        self._seeds, self._spectral = [], np.zeros(self.shape, dtype=bool)
        self._roi_note = {"candidate": candidate.label(), "touches_border": candidate.touches_border}
        self._update_roi()

    # ── spectral pick ───────────────────────────────────────────────────
    def _spectral_index(self) -> rt.SpectralIndex:
        if self._index is None:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self._index = rt.SpectralIndex(self.cube, self.wavelengths, self.saturated)
            finally:
                QApplication.restoreOverrideCursor()
        return self._index

    def spectral_pick(self, row: int, col: int, mode: str = "new"):
        """Spectral pick at ``(row, col)``: ``mode`` ``"new"`` replaces the ROI, ``"add"`` adds the
        region, ``"remove"`` takes it out (as a click, Ctrl+click, Alt+click)."""
        self._mark()
        if self._tool != "spectral":
            self._set_tool("spectral")
        auto = self.auto_threshold_box.isChecked()
        pick = self._spectral_index().pick(row, col, None if auto else self.threshold_spin.value())
        if auto:
            self.threshold_spin.blockSignals(True)
            self.threshold_spin.setValue(pick.threshold_deg)
            self.threshold_spin.blockSignals(False)
        entry = (int(row), int(col), "-" if mode == "remove" else "+", float(pick.threshold_deg), bool(auto))
        if mode == "new":
            self._seeds, self._spectral = [entry], pick.mask.copy()
        elif mode == "add":
            self._seeds.append(entry)
            self._spectral |= pick.mask
        else:
            self._seeds.append(entry)
            self._spectral &= ~pick.mask
        if not pick.mask.any():
            self.hint_label.setText("No region at that pixel (saturated or no valid spectrum).")
        self._update_roi()

    def _on_threshold_mode(self, auto: bool):
        self.threshold_spin.setEnabled(not auto)
        self._redo_spectral()

    def _redo_spectral(self):
        """Recompute the spectral regions of every seed with the current threshold setting."""
        if not self._seeds:
            return
        auto = self.auto_threshold_box.isChecked()
        value = float(self.threshold_spin.value())
        if all(s[4] == auto and (auto or abs(s[3] - value) < 1e-9) for s in self._seeds):
            return                          # nothing changed (e.g. focus left the spin box)
        index = self._spectral_index()
        self._mark()
        mask = np.zeros(self.shape, dtype=bool)
        seeds = []
        for row, col, sign, _, _ in self._seeds:
            pick = index.pick(row, col, None if auto else self.threshold_spin.value())
            mask = (mask | pick.mask) if sign == "+" else (mask & ~pick.mask)
            seeds.append((row, col, sign, float(pick.threshold_deg), bool(auto)))
        self._seeds, self._spectral = seeds, mask
        self._update_roi()

    # ── mouse / keyboard ────────────────────────────────────────────────
    def _on_press(self, event):
        if event.inaxes is not self.ax or event.xdata is None or self.toolbar.mode:
            return
        self.canvas.setFocus()
        col, row = int(round(event.xdata)), int(round(event.ydata))
        if self._tool == "spectral" and event.button == 1:
            mods = set(getattr(event, "modifiers", None) or ())
            key = (event.key or "").lower()
            ctrl = "ctrl" in mods or "control" in key or "ctrl" in key
            alt = "alt" in mods or "alt" in key
            self.spectral_pick(row, col, "add" if ctrl else "remove" if alt else "new")
        elif self._selector_kind == "polygon" and event.dblclick and event.button == 1:
            self._insert_vertex(event.xdata, event.ydata)

    def _insert_vertex(self, x: float, y: float):
        verts = list(self._selector.verts)
        if len(verts) < 3:
            return
        pts = np.asarray(verts + [verts[0]], dtype=np.float64)
        a, b = pts[:-1], pts[1:]
        ab = b - a
        t = np.clip(((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1]) / np.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
        d = np.hypot(a[:, 0] + t * ab[:, 0] - x, a[:, 1] + t * ab[:, 1] - y)
        i = int(np.argmin(d))
        self._mark()
        verts.insert(i + 1, (float(x), float(y)))
        self._selector.verts = verts
        self._update_roi()

    def _on_key(self, event):
        key = (event.key or "").lower()
        if key in ("ctrl+z", "control+z"):
            self.undo()
            return
        if key in ("delete", "backspace"):
            self.clear_roi()
            return
        if key == "escape":
            # the selectors clear themselves on Esc without telling us: clear the ROI with them,
            # so the mask in use is never one that is no longer shown
            if self._tool == "freehand":
                self._mark()
                self._seeds, self._spectral = [], np.zeros(self.shape, dtype=bool)
                self._make_selector("lasso")
                self._update_roi()
            elif self._selector_kind is not None:
                self.clear_roi()
            return
        parts = key.split("+")
        arrow = parts[-1]
        if arrow not in ("left", "right", "up", "down"):
            return
        stretch = "ctrl" in parts or "control" in parts
        step = 10.0 if "shift" in parts else 1.0
        dx = {"left": -step, "right": step}.get(arrow, 0.0)
        dy = {"up": -step, "down": step}.get(arrow, 0.0)
        self.nudge(dx, dy, stretch)

    def nudge(self, dx: float, dy: float, stretch: bool = False):
        """Move the shape by ``(dx, dy)`` pixels, or with ``stretch`` move only its right / bottom edge."""
        kind = self._selector_kind
        if kind in ("rect", "ellipse") and self._has_shape:
            x0, x1, y0, y1 = self._selector.extents
            if stretch:
                x1, y1 = max(x0 + 1, x1 + dx), max(y0 + 1, y1 + dy)
            else:
                x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
            self._mark()
            self._selector.extents = (x0, x1, y0, y1)
        elif kind == "polygon":
            pts = np.asarray(self._selector.verts, dtype=np.float64)
            if stretch:
                for axis, d in ((0, dx), (1, dy)):
                    lo, hi = pts[:, axis].min(), pts[:, axis].max()
                    if d and hi > lo:
                        pts[:, axis] = lo + (pts[:, axis] - lo) * max(hi - lo + d, 1.0) / (hi - lo)
            else:
                pts += (dx, dy)
            self._mark()
            self._selector.verts = [tuple(p) for p in pts]
        else:
            return
        self._update_roi()

    # ── ROI → mask, clean-up, QC ────────────────────────────────────────
    def _drawn_mask(self) -> np.ndarray:
        kind = self._selector_kind
        if self._seeds or self._tool == "spectral":
            return self._spectral.copy()
        if kind == "rect" and self._has_shape:
            x0, x1, y0, y1 = self._selector.extents
            return rt.rect_mask(self.shape, x0, y0, x1, y1)
        if kind == "ellipse" and self._has_shape:
            x0, x1, y0, y1 = self._selector.extents
            return rt.ellipse_mask(self.shape, (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2)
        if kind == "polygon":
            return rt.polygon_mask(self.shape, self._selector.verts)
        return np.zeros(self.shape, dtype=bool)

    def _update_roi(self, redraw: bool = True):
        if redraw or self.cleanup is None:
            self.drawn = self._drawn_mask()
            gate = self.gate_spin.value() / 100.0
            seed_rows = [s[0] for s in self._seeds if s[2] == "+"]
            self.cleanup = rt.clean_mask(self.drawn, self.brightness, self.saturated,
                                         inset_px=self.inset_spin.value(), row_tolerance=gate if gate > 0 else None,
                                         index=self._index, spectral=self.spectral_gate_box.isChecked(),
                                         seed_rows=seed_rows)
            self.quality = rt.assess_roi(self.cleanup, self.cube, self.wavelengths, self.brightness,
                                         saturation_source=self.saturation_source, ceiling=self.ceiling,
                                         full_scale=self.metadata.get("full_scale"))
            self._update_auto_window()
            self._draw_overlay()
            self._draw_spectrum()
            self._commit()
        self._show_quality()

    def recommended_window(self) -> int:
        """Averaging window (rows) for the current ROI: ``ceil(AUTO_WINDOW_PIXELS / median pixels
        per ROI row)``, 1-50. Slit-row patterns (dust, a fixed-pattern row) need small windows; a
        window only adds noise reduction where the ROI is narrow along the scan."""
        mask = self.cleanup.mask if self.cleanup is not None else None
        if mask is None or not mask.any():
            return 1
        per_row = mask.sum(axis=1)
        per_row = per_row[per_row > 0]
        return int(np.clip(math.ceil(AUTO_WINDOW_PIXELS / max(float(np.median(per_row)), 1.0)), 1, 50))

    def _update_auto_window(self):
        if self.auto_window_box.isChecked():
            self.avg_window_spin.blockSignals(True)
            self.avg_window_spin.setValue(self.recommended_window())
            self.avg_window_spin.blockSignals(False)

    def _draw_overlay(self):
        rgba = np.zeros(self.shape + (4,), np.float32)
        rgba[self.cleanup.mask] = _KEEP_RGBA
        rgba[self.drawn & ~self.cleanup.mask] = _DROP_RGBA
        self._overlay.set_data(rgba)
        self.canvas.draw_idle()

    def _draw_spectrum(self):
        ax = self.spectrum_ax
        ax.clear()
        m = self.cleanup.mask
        if m.any():
            rr, cc = np.nonzero(m)
            if rr.size > 20000:
                pick = np.linspace(0, rr.size - 1, 20000).astype(int)
                rr, cc = rr[pick], cc[pick]
            spec = np.nanmean(np.asarray(self.cube[rr, cc, :], dtype=np.float64), axis=0)
            ax.plot(self.wavelengths, spec, color="#E9A23B", lw=1.4)
        ax.set_title("ROI mean counts", fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        self.spectrum_canvas.draw_idle()

    def _table_items(self) -> List[rt.QcItem]:
        path = self.profile_combo.currentData()
        if not path:
            return [rt.QcItem("error", "Choose the reflectance table of the reference target")]
        if self._table_check is None or self._table_check[0] != path:
            try:
                reflectance, table_range = reflectance_at(path, self.wavelengths)
                self._table_check = (path, int(np.count_nonzero(~np.isfinite(reflectance))), table_range, None)
            except Exception as exc:
                self._table_check = (path, 0, None, str(exc))
        _, outside, table_range, error = self._table_check
        if error:
            return [rt.QcItem("error", f"Cannot read the table: {error}")]
        if outside:
            return [rt.QcItem("warn", f"{outside} bands outside the table ({table_range[0]:g}-{table_range[1]:g} nm) "
                                      "get no reflectance (NaN)")]
        return [rt.QcItem("ok", f"Table covers all bands ({table_range[0]:g}-{table_range[1]:g} nm)")]

    def problems(self) -> List[str]:
        """Why *Create profile* is disabled now (empty = it can be pressed)."""
        problems = list(self.key.missing())
        if self.store is None:
            problems.append("No camera profile is open: the reference cannot be stored")
        problems += [i.text for i in self._table_items() if i.level == "error"]
        if self.quality is not None:
            problems += self.quality.errors()
        return problems

    def _show_quality(self):
        colors = {"ok": "#2A9D8F", "warn": "#C77D00", "error": "#D1495B"}
        marks = {"ok": "✓", "warn": "!", "error": "✕"}
        items = [rt.QcItem("error", t) for t in self.key.missing()]
        if self.store is None:
            items.append(rt.QcItem("error", "No camera profile is open: the reference cannot be stored"))
        items += self._table_items() + list(self.quality.items if self.quality else ())
        html = "".join(f"<div style='margin:2px 0'><b style='color:{colors[i.level]}'>{marks[i.level]}</b>"
                       f"&nbsp;{_escape(i.text)}</div>" for i in items)
        numbers = self.quality.numbers if self.quality else {}
        html += (f"<div style='color:#5E6A7D;margin-top:6px'>ROI {numbers.get('final_px', 0)} px of "
                 f"{numbers.get('drawn_px', 0)} drawn · saturated {numbers.get('saturated_px', 0)} · "
                 f"inset {numbers.get('inset_removed_px', 0)} · other material "
                 f"{numbers.get('spectral_removed_px', 0)} · row gate {numbers.get('gate_removed_px', 0)}</div>")
        self.qc_label.setText(html)
        self.create_btn.setEnabled(not self.problems())

    # ── profile ─────────────────────────────────────────────────────────
    def _browse_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, "Reflectance table", "", "Excel (*.xlsx *.xls)")
        if path:
            self.profile_combo.addItem(os.path.splitext(os.path.basename(path))[0], path)
            self.profile_combo.setCurrentIndex(self.profile_combo.count() - 1)

    def set_profile(self, path) -> None:
        """Select the reflectance table ``path`` (added to the list when it is not in it)."""
        path = str(path)
        for i in range(self.profile_combo.count()):
            if self.profile_combo.itemData(i) and os.path.normcase(str(self.profile_combo.itemData(i))) == os.path.normcase(path):
                self.profile_combo.setCurrentIndex(i)
                return
        self.profile_combo.addItem(os.path.splitext(os.path.basename(path))[0], path)
        self.profile_combo.setCurrentIndex(self.profile_combo.count() - 1)

    # ── create ──────────────────────────────────────────────────────────
    def _confirm(self, title: str, text: str) -> bool:
        """Ask the user (tests replace this)."""
        return QMessageBox.question(self, title, text, QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No) == QMessageBox.Yes

    def _report_error(self, title: str, text: str) -> None:
        """Show an error (tests replace this)."""
        QMessageBox.critical(self, title, text)

    def roi_description(self) -> dict:
        """JSON description of how the ROI was made (stored in the reference's ``info``)."""
        kind, geometry, seeds, _ = self._state()
        info = {"tool": self._tool, "shape": kind, "inset_px": int(self.inset_spin.value()),
                "row_tolerance": (self.gate_spin.value() / 100.0) or None,
                "spectral_gate": bool(self.spectral_gate_box.isChecked()),
                "saturation_source": self.saturation_source}
        if geometry is not None:
            info["geometry"] = [list(map(float, p)) for p in geometry] if kind == "polygon" else list(geometry)
        if seeds:
            info["seeds"] = [{"row": r, "col": c, "sign": s, "threshold_deg": round(t, 3), "auto": a}
                             for r, c, s, t, a in seeds]
        note = getattr(self, "_roi_note", None)
        if self._tool == "detect" and note:
            info.update(note)
        return info

    def create_profile(self):
        """Fit the factors on the effective ROI, check them and store the reference.

        Returns:
            The stored reference, or ``None`` when nothing was stored.
        """
        if self.cube is None:                       # closed: its arrays are released
            return None
        problems = self.problems()
        if problems:
            self._report_error("Create profile", "\n".join(problems))
            return None
        table = self.profile_combo.currentData()
        mask = self.cleanup.mask.copy()
        started = self.metadata.get("started_at")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = compute_rowwise_factors_array(self.cube, self.wavelengths, table,
                                                   self.avg_window_spin.value(), mask,
                                                   source_name=f"scan {started}" if started else "scan",
                                                   window_fit=WINDOW_FIT)
            check, worst_row = self._self_check(result["factors"], mask, table)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            logger.exception("Radiometric reference fit failed")
            self._report_error("Create profile", f"The factors could not be fitted: {exc}")
            return None
        QApplication.restoreOverrideCursor()
        if (check is not None and check > SELF_CHECK_RMS_PERCENT) or \
                (worst_row is not None and worst_row[1] > SELF_CHECK_ROW_PERCENT):
            row_text = (f"; the worst slit row (row {worst_row[0]}) is off by {worst_row[1]:.2f} %"
                        if worst_row is not None else "")
            if not self._confirm("Create profile",
                                 f"Applied to its own ROI, the reference gives the table within {check:.2f} % RMS"
                                 f"{row_text} (expected below {SELF_CHECK_RMS_PERCENT:g} % overall and "
                                 f"{SELF_CHECK_ROW_PERCENT:g} % per row). The ROI may hold uneven or non-target "
                                 "pixels, or the averaging window is too long for the vignetting.\n\n"
                                 "Store it anyway?"):
                return None
        info = {"source": result["reference"]["reflectance_reference_cube"],
                "roi": self.roi_description(), "qc": dict(self.quality.numbers),
                "window_fit": result["window_fit"],
                "self_check_rms_percent": None if check is None else round(float(check), 4),
                "self_check_worst_row": None if worst_row is None else
                [int(worst_row[0]), round(float(worst_row[1]), 4)],
                "table_range_nm": list(result["table_range_nm"]),
                "out_of_range_bands": int(result["out_of_range_bands"])}
        try:
            reference = self.store.save(result["factors"], self.key, reflectance_profile=table,
                                        avg_window_size=self.avg_window_spin.value(), mask=mask,
                                        roi_mean_spectrum=result["roi_mean_spectrum"],
                                        sensor_temperature_c=scan_temperature_c(self.metadata), info=info,
                                        acquired_at=started, activate=False)
        except Exception as exc:
            logger.exception("Radiometric reference could not be stored")
            self._report_error("Create profile", f"The reference could not be stored: {exc}")
            return None
        if self.activate_box.isChecked():
            # stored first, activated second: an activation failure (locked or broken active.json)
            # must not look like "not stored" and invite a second, orphan copy
            try:
                self.store.activate(reference.id)
            except Exception as exc:
                logger.exception("Radiometric reference %s stored but not activated", reference.id)
                self._report_error("Create profile", f"Stored as {reference.id}, but it could not be activated "
                                                     f"for automatic calibration: {exc}")
        self.reference = reference
        self.reference_saved.emit(reference)
        self.accept()
        return reference

    def _self_check(self, factors, mask, table):
        """The factors applied to the pixels they were fitted on must give the table back.

        Returns:
            ``(overall, worst_row)``: ``overall`` = RMS (%) over bands of (mean calibrated ROI
            reflectance / table - 1); ``worst_row`` = ``(row, %)`` of the slit row whose mean
            calibrated reflectance, averaged over the bands, is furthest from the table (a row
            biased by the averaging window shows here, the overall mean hides it). A window bias is
            a gain error common to all bands, so averaging over bands keeps it while the pixel
            noise averages out; rows with fewer than :data:`SELF_CHECK_MIN_ROW_PX` pixels do not
            decide it. ``None`` where nothing can be compared.
        """
        expected, _ = reflectance_at(table, self.wavelengths)
        good = np.isfinite(expected) & (expected > 0)
        f = np.asarray(factors, np.float64)
        rows = np.where(mask.any(axis=1))[0]
        if not good.any() or not rows.size:
            return None, None
        overall_sum = np.zeros(int(good.sum()))
        overall_n = 0
        worst, worst_small = None, None
        for y in rows:
            cols = np.nonzero(mask[y])[0]
            n_px = cols.size
            if cols.size > 512:
                cols = cols[np.linspace(0, cols.size - 1, 512).astype(int)]
            values = np.asarray(self.cube[y, cols, :], dtype=np.float64)[:, good] * f[y, good]
            row_mean = np.nanmean(values, axis=0)
            overall_sum += np.nansum(values, axis=0)
            overall_n += cols.size
            err = float(abs(np.nanmean(row_mean / expected[good]) - 1.0) * 100.0)
            if not np.isfinite(err):
                continue
            if n_px >= SELF_CHECK_MIN_ROW_PX:
                if worst is None or err > worst[1]:
                    worst = (int(y), err)
            elif worst_small is None or err > worst_small[1]:
                worst_small = (int(y), err)
        worst = worst if worst is not None else worst_small
        overall = overall_sum / max(overall_n, 1)
        rms = float(math.sqrt(np.nanmean((overall / expected[good] - 1.0) ** 2)) * 100.0)
        return (rms if np.isfinite(rms) else None), worst


def _same_state(a, b) -> bool:
    """Whether two undo entries ``(tool, (kind, geometry, seeds, spectral mask), extras)`` hold the same ROI."""
    (tool_a, (kind_a, geo_a, seeds_a, mask_a)), (tool_b, (kind_b, geo_b, seeds_b, mask_b)) = a[:2], b[:2]
    if (tool_a, kind_a, seeds_a) != (tool_b, kind_b, seeds_b):
        return False
    if (geo_a is None) != (geo_b is None) or (mask_a is None) != (mask_b is None):
        return False
    if geo_a is not None and (np.shape(geo_a) != np.shape(geo_b) or not np.allclose(geo_a, geo_b)):
        return False
    return mask_a is None or np.array_equal(mask_a, mask_b)


def _escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
