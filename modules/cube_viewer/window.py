# windows/cube_viewer.py
"""Cube viewer window with single-band and RGB composite modes.
Both modes use 1-click-1-spectrum with shared tab20 colored dots and legend.

Data conventions:
    The cube is ``(slit, scan, wavelength)`` = ``(H, W, bands)`` of any numeric dtype
    (typically uint16; a ``numpy.memmap`` from ``modules.cube_io.load_envi_cube_for_viewer``
    works).  ``wavelengths`` is a 1-D array in nm with one value per band, or ``None``.
    The viewer keeps a reference to the cube (no copy) and sets it to ``None`` when closed.
    Image pixel ``(ix, iy)`` is ``(column, row)`` = ``(scan, slit)``.

Threading:
    All widgets live in the GUI thread.  Export runs in a ``QThread`` owned by
    ``ExportDialog`` (workers from ``modules.cube_io.export_workers``).

Example:
    >>> import numpy as np
    >>> from modules.cube_viewer import CubeViewerWindow           # doctest: +SKIP
    >>> cube = np.random.randint(0, 4096, (32, 48, 40), dtype=np.uint16)
    >>> win = CubeViewerWindow(cube, wavelengths=np.linspace(400, 1000, 40))  # doctest: +SKIP
    >>> win.show()                                                  # doctest: +SKIP
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtGui import QKeySequence, QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QSlider, QGroupBox, QComboBox, QPushButton,
    QFileDialog, QProgressBar, QMessageBox, QDialog,
    QSplitter, QShortcut, QScrollArea, QSizePolicy, QFrame,
    QStackedWidget,
)

from modules.live_view.mapping import DisplayMapper
from modules.cube_viewer.canvas import CubeImageCanvas
from modules.cube_preview.widget import CenteredPixmapLabel, TAB20_HEX
from modules.live_view.spectrum import SpectrumPlot
from modules.live_view.display_controls import DisplayControls
from modules.cube_viewer.controls import RGBCompositeControls
from modules.ui_kit.zoom_pan import ZoomPanMixin

from modules.cube_io.export_workers import (
    GeoTIFFWorker, HypercubeWorker, SpectrogramWorker,
)


# ══════════════════════════════════════════════════════════════
#  Export Dialog
# ══════════════════════════════════════════════════════════════

class ExportDialog(QDialog):
    """Non-modal "Export Datacube" dialog (ENVI BIP, GeoTIFF or raw binary).

    The cube ``(slit, scan, wavelength)`` is transposed to band-first
    ``(wavelength, slit, scan)`` (a copy) and written by a worker moved to its own
    ``QThread``; progress and completion are shown in the dialog.  Without
    ``wavelengths`` the band indices are used as wavelengths.

    Args:
        parent: Parent widget (normally the ``CubeViewerWindow``).
        cube: ``(slit, scan, wavelength)`` array.
        wavelengths: 1-D band wavelengths in nm, or ``None``.
        config: Calibration dict (``m`` nm/pixel at bin 1, ``c`` nm, optional
            ``pixel_size``); ``None`` is treated as ``{}``.
        camera_bin: Hardware bin factor of the camera (``None``/0 mean 1).

    Signals:
        None of its own; it reacts to the worker's ``progressed``, ``finished``
        (seconds) and ``error`` (traceback text) signals.
    """

    def __init__(self, parent, cube, wavelengths, config, camera_bin):
        super().__init__(parent)
        self.setWindowTitle("Export Datacube")
        self.setMinimumWidth(480)
        self.setModal(False)

        self.cube = cube
        self.wavelengths = wavelengths
        self.config = config or {}
        self.camera_bin = int(camera_bin or 1)
        self._thread = None
        self._worker = None
        self._save_path = ""

        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(16, 16, 16, 16)

        h, w, b = self.cube.shape
        mem = self.cube.nbytes / 1e6
        info = QLabel(
            f"Cube {h}\u00d7{w}\u00d7{b}  \u00b7  "
            f"{mem:.1f} MB  \u00b7  {self.cube.dtype}"
        )
        lay.addWidget(info)

        fmt_grp = QGroupBox("FORMAT")
        fmt_lay = QVBoxLayout()
        self.combo = QComboBox()
        self.combo.addItems([
            "ENVI BIP (.bip)",
            "GeoTIFF (.tif)",
            "Raw Spectrogram (.bin)",
        ])
        fmt_lay.addWidget(self.combo)
        fmt_grp.setLayout(fmt_lay)
        lay.addWidget(fmt_grp)

        path_grp = QGroupBox("OUTPUT PATH")
        path_lay = QHBoxLayout()
        self.path_label = QLabel("No file selected")
        self.btn_browse = QPushButton("Browse\u2026")
        self.btn_browse.clicked.connect(self._browse)
        path_lay.addWidget(self.path_label, stretch=1)
        path_lay.addWidget(self.btn_browse)
        path_grp.setLayout(path_lay)
        lay.addWidget(path_grp)

        self.progress = QProgressBar()
        self.progress.setMaximum(100)
        lay.addWidget(self.progress)

        self.lbl_status = QLabel("")
        lay.addWidget(self.lbl_status)

        btn_row = QHBoxLayout()
        self.btn_export = QPushButton("Start Export")
        self.btn_export.setMinimumHeight(38)
        self.btn_export.clicked.connect(self._export)

        btn_close = QPushButton("Close")
        btn_close.setMinimumHeight(38)
        btn_close.clicked.connect(self.close)

        btn_row.addWidget(self.btn_export)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

    def _browse(self):
        ext_map = {0: ".bip", 1: ".tif", 2: ".bin"}
        fmt_map = {
            0: "ENVI BIP (*.bip)",
            1: "GeoTIFF (*.tif)",
            2: "Binary (*.bin)",
        }
        idx = self.combo.currentIndex()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save As", "", fmt_map[idx],
        )
        if path:
            ext = ext_map[idx]
            if not path.lower().endswith(ext):
                path += ext
            self._save_path = path
            self.path_label.setText(path)

    def _export(self):
        if not self._save_path:
            QMessageBox.warning(self, "Error", "Select an output path first.")
            return

        cube_t = np.transpose(self.cube, (2, 0, 1)).copy()
        wl = (
            self.wavelengths
            if self.wavelengths is not None
            else np.arange(self.cube.shape[2])
        )
        idx = self.combo.currentIndex()

        self.btn_export.setEnabled(False)
        self.lbl_status.setText("Exporting\u2026")
        self.progress.setValue(0)

        if idx == 0:
            w = HypercubeWorker(
                save_path=self._save_path, config=self.config,
                hypercube=cube_t, linear_spectrum=wl,
                camera_bin=self.camera_bin,
            )
        elif idx == 1:
            w = GeoTIFFWorker(
                save_path=self._save_path, config=self.config,
                hypercube=cube_t, linear_spectrum=wl,
                camera_bin=self.camera_bin,
            )
        else:
            w = SpectrogramWorker(
                save_path=self._save_path, config=self.config,
                spectrogram=cube_t,
                pixel_size=float(self.config.get("pixel_size", 0.1)),
                camera_bin=self.camera_bin,
            )

        self._worker = w
        self._thread = QThread()
        w.moveToThread(self._thread)
        self._thread.started.connect(w.run)
        w.progressed.connect(self.progress.setValue)
        w.finished.connect(self._on_done)
        w.error.connect(self._on_error)
        self._thread.start()

    def _on_done(self, elapsed):
        self.lbl_status.setText(f"Completed in {elapsed:.2f}s")
        self.btn_export.setEnabled(True)
        QMessageBox.information(self, "Done", f"Saved:\n{self._save_path}")

    def _on_error(self, msg):
        self.lbl_status.setText("Export failed")
        self.btn_export.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)

    def closeEvent(self, event):
        """Stop and join the export thread, drop the cube reference (Qt override).

        Blocks until a running export finishes, because the thread is joined.
        """
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self.cube = None
        self._worker = None
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════
#  Cube Viewer Window
# ══════════════════════════════════════════════════════════════

class CubeViewerWindow(QWidget):
    """Interactive viewer for a ``(slit, scan, wavelength)`` hyperspectral cube.

    Shows one band at a time (slider, play/pause sweep, left/right keys) or an RGB
    composite; each left click on the image adds a spectrum curve (tab20 colour
    coded dot and legend entry) to the plot below.  Zoom with the wheel, pan with
    the middle button, ``+``/``-``/``0``/``1`` keys.  Includes display controls,
    cube info and an "Export Datacube" dialog (``E``).  Keyboard: Left/Right band,
    Space play/pause, R reset, E export.

    Unlike the widgets listed in the package ``__widgets__`` this one needs a cube,
    so it is not constructible without arguments.  The widget emits no custom
    signals; query it with ``status_info()`` and the ``current_band`` attribute.
    On ``closeEvent`` the cube, wavelengths, canvas and spectrum references are
    dropped and the export dialog is closed.

    Args:
        cube: 3-D array ``(slit, scan, wavelength)``; kept by reference.
        wavelengths: 1-D band wavelengths in nm (length ``bands``) or ``None``.
        resolution: Decimals shown for wavelengths in labels.
        display_mapper: Optional ``DisplayMapper`` whose ``bit_depth`` initialises
            the display controls (default 16 bit).
        config: Calibration dict forwarded to the export (``m``, ``c``, ``pixel_size``).
        camera_bin: Hardware bin factor forwarded to the export (default 1).

    Raises:
        ValueError: ``cube`` is not 3-D.

    Layout::

    ┌─── HSplitter ────────────────────────────────────────────┐
    │           │  ┌─ Overlay Bar (band nav) ────────────────┐  │
    │           │  ├─ VSplitter ────────────────────────────┐│  │
    │  SIDEBAR  │  │  QStackedWidget                       ││  │
    │           │  │    page 0: CubeImageCanvas     4/5    ││  │
    │  DISPLAY  │  │    page 1: CenteredPixmapLabel         ││  │
    │  RGB      │  ├─ SpectrumPlot                  1/5    ││  │
    │  CUBE INFO│  └───────────────────────────────────────┘│  │
    │  EXPORT   │  ┌─ Status Bar ──────────────────────────┐│  │
    │  KEYBOARD │  └───────────────────────────────────────┘│  │
    └───────────┴──────────────────────────────────────────┘
    """

    def __init__(
        self,
        cube: np.ndarray,
        wavelengths: np.ndarray | None = None,
        resolution: int = 2,
        display_mapper: DisplayMapper | None = None,
        config: dict | None = None,
        camera_bin: int | None = None,
    ):
        super().__init__()

        if cube.ndim != 3:
            raise ValueError(f"Cube must be 3D, got shape={cube.shape}")

        self.cube = cube
        self.h, self.w, self.bands = cube.shape
        self.wavelengths = wavelengths
        self.resolution = int(resolution)
        self.config = config or {}
        self.camera_bin = int(camera_bin or 1)

        self.current_band = 0
        self._export_dlg = None
        self._playing = False

        # ★ shared spectrum history — single source of truth
        self._spectrum_history: list[tuple[int, int, np.ndarray]] = []

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(80)
        self._anim_timer.timeout.connect(self._anim_step)

        # ── single-band canvas (1-click spectrum) ──
        self.canvas = CubeImageCanvas()
        self.canvas.pixel_clicked.connect(self._on_canvas_pixel_clicked)

        bd = 16
        if display_mapper is not None:
            bd = display_mapper.bit_depth
        self.display_controls = DisplayControls(bit_depth=bd)
        if display_mapper is not None:
            self.canvas.set_display_mapper(display_mapper)

        # ── spectrum plot ──
        self.spectrum = SpectrumPlot()

        # ── RGB controls (sidebar) ──
        self._rgb_controls = RGBCompositeControls()
        self._rgb_controls.set_cube(cube, wavelengths)
        self._rgb_controls.mode_changed.connect(self._on_display_mode_changed)
        self._rgb_controls.composite_updated.connect(self._on_composite_updated)

        # ── RGB image label (1-click spectrum) ──
        self._rgb_image = CenteredPixmapLabel()
        self._rgb_image.clicked.connect(self._rgb_on_clicked)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)
        self._rgb_image.zoom_changed.connect(self._on_zoom_changed)

        self._build_ui()
        self._setup_shortcuts()
        self.update_band_image()
        self.on_band_changed(0)

    # ──────────────────────────────────────────────────────
    #  BUILD UI
    # ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._hsplit = QSplitter(Qt.Horizontal)
        self._hsplit.setHandleWidth(3)

        sidebar = self._build_sidebar()
        self._hsplit.addWidget(sidebar)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        self._overlay = self._build_overlay_bar()
        right_lay.addWidget(self._overlay, stretch=0)

        self._display_stack = QStackedWidget()
        self._display_stack.addWidget(self.canvas)          # index 0
        self._display_stack.addWidget(self._rgb_image)      # index 1

        self._vsplit = QSplitter(Qt.Vertical)
        self._vsplit.setHandleWidth(3)
        self._vsplit.addWidget(self._display_stack)
        self._vsplit.addWidget(self.spectrum)
        self._vsplit.setStretchFactor(0, 4)
        self._vsplit.setStretchFactor(1, 1)
        right_lay.addWidget(self._vsplit, stretch=1)

        right_lay.addWidget(self._build_status_bar(), stretch=0)

        self._hsplit.addWidget(right)
        self._hsplit.setStretchFactor(0, 0)
        self._hsplit.setStretchFactor(1, 1)
        self._hsplit.setSizes([240, 800])

        root.addWidget(self._hsplit)

    def _build_sidebar(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(180)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(8, 10, 8, 10)
        lay.setSpacing(6)

        # mode toggle + RGB controls
        g_mode = QGroupBox("DISPLAY MODE")
        g_mode_lay = QVBoxLayout()
        g_mode_lay.addWidget(self._rgb_controls)
        g_mode.setLayout(g_mode_lay)
        lay.addWidget(g_mode)

        # display controls
        g1 = QGroupBox("DISPLAY")
        l1 = QVBoxLayout()
        l1.addWidget(self.display_controls)
        g1.setLayout(l1)
        lay.addWidget(g1)

        # ★ clear all points button
        self.btn_clear_points = QPushButton("Clear All Points")
        self.btn_clear_points.setMinimumHeight(32)
        self.btn_clear_points.clicked.connect(self._clear_all_points)
        lay.addWidget(self.btn_clear_points)

        # cube info
        g2 = QGroupBox("CUBE INFO")
        l2 = QFormLayout()
        l2.setSpacing(4)
        l2.addRow("Height :", self._val(str(self.h)))
        l2.addRow("Width  :", self._val(str(self.w)))
        l2.addRow("Bands  :", self._val(str(self.bands)))
        l2.addRow("Dtype  :", self._val(str(self.cube.dtype)))
        l2.addRow("Memory :", self._val(f"{self.cube.nbytes / 1e6:.1f} MB"))
        g2.setLayout(l2)
        lay.addWidget(g2)

        # export button
        self.btn_export = QPushButton("  Export Datacube \u2192")
        self.btn_export.setMinimumHeight(38)
        self.btn_export.clicked.connect(self._open_export)
        lay.addWidget(self.btn_export)

        # keyboard shortcuts
        g3 = QGroupBox("KEYBOARD")
        l3 = QVBoxLayout()
        l3.setSpacing(2)
        for key, desc in [
            ("\u2190 \u2192",  "Prev / Next band"),
            ("Space",         "Play / Pause sweep"),
            ("E",             "Export dialog"),
            ("R",             "Reset view"),
        ]:
            row = QHBoxLayout()
            kl = QLabel(key)
            kl.setFixedWidth(52)
            dl = QLabel(desc)
            row.addWidget(kl)
            row.addWidget(dl)
            l3.addLayout(row)
        g3.setLayout(l3)
        lay.addWidget(g3)

        lay.addStretch()
        scroll.setWidget(content)
        return scroll

    def _build_overlay_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("overlayBar")
        bar.setFixedHeight(48)

        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 6, 16, 6)
        lay.setSpacing(6)

        # ★ nav buttons
        self.btn_prev = self._nav_btn("\u25C0", lambda: self._step(-1))
        self.btn_play = self._nav_btn("\u25B6", self._toggle_play)
        self.btn_play.setCheckable(True)
        self.btn_next = self._nav_btn("\u25B6", lambda: self._step(1))

        # ★ band slider
        self.band_slider = QSlider(Qt.Horizontal)
        self.band_slider.setRange(0, max(0, self.bands - 1))
        self.band_slider.valueChanged.connect(self.on_band_changed)

        # ★ labels
        self.band_label = QLabel()
        self.band_label.setObjectName("valueLabel")
        self.band_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.band_label.setMinimumWidth(180)

        self.roi_label = QLabel("")
        self.roi_label.setObjectName("valueLabel")
        self.roi_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # ★ zoom controls
        self._zoom_label = QLabel("100%")
        self._zoom_label.setObjectName("valueLabel")
        self._zoom_label.setAlignment(Qt.AlignCenter)
        self._zoom_label.setFixedWidth(48)

        btn_zoom_out = self._nav_btn("\u2212", self._zoom_out)
        btn_zoom_in = self._nav_btn("+", self._zoom_in)
        btn_fit = self._nav_btn("Fit", self._zoom_fit)
        btn_fit.setFixedWidth(36)
        btn_1to1 = self._nav_btn("1:1", self._zoom_actual)
        btn_1to1.setFixedWidth(36)

        # ★ layout order: nav | slider | info | separator | zoom
        for w in (self.btn_prev, self.btn_play, self.btn_next):
            lay.addWidget(w)
        lay.addSpacing(6)
        lay.addWidget(self.band_slider, stretch=1)
        lay.addSpacing(12)
        lay.addWidget(self.band_label)
        lay.addWidget(self._sep())
        lay.addWidget(self.roi_label)
        lay.addWidget(self._sep())
        for w in (btn_zoom_out, self._zoom_label, btn_zoom_in, btn_fit, btn_1to1):
            lay.addWidget(w)

        return bar

    def _build_status_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("statusBar")
        bar.setFixedHeight(28)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 0, 12, 0)
        self.status_label = QLabel("Ready")
        lay.addWidget(self.status_label)
        return bar

    def _setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left),  self, lambda: self._step(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self._step(1))
        QShortcut(QKeySequence(Qt.Key_Space), self, self._toggle_play)
        QShortcut(QKeySequence(Qt.Key_E),     self, self._open_export)
        QShortcut(QKeySequence(Qt.Key_R),     self, self._reset)
        QShortcut(QKeySequence(Qt.Key_Plus),    self, self._zoom_in)
        QShortcut(QKeySequence(Qt.Key_Equal),   self, self._zoom_in)
        QShortcut(QKeySequence(Qt.Key_Minus),   self, self._zoom_out)
        QShortcut(QKeySequence(Qt.Key_0),       self, self._zoom_fit)
        QShortcut(QKeySequence(Qt.Key_1),       self, self._zoom_actual)
        
    # ──────────────────────────────────────────────────────
    #  DISPLAY MODE SWITCH — sync click points
    # ──────────────────────────────────────────────────────
    def _rgb_on_clicked(self, ix: int, iy: int):
        """Left-click on RGB image — extract spectrum."""
        spectrum = self.cube[iy, ix, :].astype(np.float32)
        self._add_spectrum_point(ix, iy, spectrum)
    # ★ _on_display_mode_changed — update zoom label when switching
    def _on_display_mode_changed(self, mode: str):
        is_rgb = (mode == "rgb")
        if is_rgb:
            self._display_stack.setCurrentIndex(1)
            self._overlay.setVisible(False)
        else:
            self._display_stack.setCurrentIndex(0)
            self._overlay.setVisible(True)
            self.update_band_image()

        self._sync_click_points_to_active_view()
        # ★ update zoom label for active canvas
        self._on_zoom_changed(self._active_canvas().zoom_percent)

    def _sync_click_points_to_active_view(self):
        """Re-draw all click points from history onto the active canvas."""
        is_rgb = (self._display_stack.currentIndex() == 1)

        # clear dots on BOTH canvases
        self.canvas.clear_click_points()
        self._rgb_image.clear_click_points()

        # redraw all dots on active canvas
        active = self._rgb_image if is_rgb else self.canvas
        for ix, iy, _ in self._spectrum_history:
            active.add_click_point(ix, iy)

    def _on_composite_updated(self, rgb8: np.ndarray):
        h, w, ch = rgb8.shape
        fmt = QImage.Format_RGB888
        qimg = QImage(rgb8.data, w, h, ch * w, fmt).copy()
        pixmap = QPixmap.fromImage(qimg)
        self._rgb_image.setPixmapScaled(pixmap)

        # ★ re-sync dots after pixmap update (paintEvent needs pixmap)
        self._sync_click_points_to_active_view()

    # ──────────────────────────────────────────────────────
    #  SINGLE-BAND MODE — 1-click spectrum
    # ──────────────────────────────────────────────────────

    def _on_canvas_pixel_clicked(self, ix: int, iy: int):
        """1 click on single-band canvas = 1 spectrum."""
        spectrum = self.cube[iy, ix, :].astype(np.float32)
        self._add_spectrum_point(ix, iy, spectrum)


    # ──────────────────────────────────────────────────────
    #  SHARED — add point to history + update everything
    # ──────────────────────────────────────────────────────

    def _add_spectrum_point(self, ix: int, iy: int, spectrum: np.ndarray):
        """Add a click point — shared by both modes."""
        self._spectrum_history.append((ix, iy, spectrum.copy()))

        # add dot on active canvas
        is_rgb = (self._display_stack.currentIndex() == 1)
        if is_rgb:
            self._rgb_image.add_click_point(ix, iy)
        else:
            self.canvas.add_click_point(ix, iy)

        # redraw spectra with legend
        self._redraw_spectra()

        # update info
        idx = len(self._spectrum_history)
        self.roi_label.setText(
            f"Point {idx}: x={ix}  y={iy}  ({idx} spectra)"
        )
        self._refresh_status()

    # ──────────────────────────────────────────────────────
    #  SHARED — redraw spectra with tab20 legend
    # ──────────────────────────────────────────────────────

    def _redraw_spectra(self):
        """Redraw all spectra (both modes) with tab20 legend."""
        import pyqtgraph as pg

        pw = self.spectrum.pw

        pw.clear()
        pw.addLegend()

        for i, (px, py, spectrum) in enumerate(self._spectrum_history):
            color = TAB20_HEX[i % len(TAB20_HEX)]
            pen = pg.mkPen(color=color, width=1.5)

            if (
                self.wavelengths is not None
                and len(self.wavelengths) == len(spectrum)
            ):
                x_data = self.wavelengths
            else:
                x_data = np.arange(len(spectrum), dtype=np.float32)

            pw.plot(
                x_data, spectrum, pen=pen,
                name=f"({px},{py})", connect="finite",
            )

        if self._spectrum_history:
            all_spectra = np.stack([s for _, _, s in self._spectrum_history])
            ymin = float(all_spectra.min())
            ymax = float(all_spectra.max())
            if ymax <= ymin:
                ymax = ymin + 1.0
            margin = (ymax - ymin) * 0.08 + 1e-6
            pw.setYRange(ymin - margin, ymax + margin, padding=0)

    # ──────────────────────────────────────────────────────
    #  CLEAR ALL POINTS
    # ──────────────────────────────────────────────────────

    def _clear_all_points(self):
        """Clear all spectrum points, dots, and plot."""
        self._spectrum_history.clear()
        self.canvas.clear_click_points()
        self._rgb_image.clear_click_points()
        self.roi_label.setText("")

        # ★ clear plot + legend ให้หมด (ไม่ใช่แค่ curve)
        import pyqtgraph as pg
        pw = self.spectrum.pw
        pw.clear()
        pw.addLegend()
        # re-add band marker (ถ้าต้องการ)
        if hasattr(self.spectrum, 'band_line'):
            self.spectrum.band_line.setVisible(False)
            pw.addItem(self.spectrum.band_line)

    # ──────────────────────────────────────────────────────
    #  BAND NAVIGATION
    # ──────────────────────────────────────────────────────

    def on_band_changed(self, band: int):
        """Select a band: update the label, the image and the status line.

        Args:
            band: Band index in ``[0, bands)`` (0-based, not a wavelength).
        """
        self.current_band = int(band)

        if self.wavelengths is not None and len(self.wavelengths) == self.bands:
            wl = self.wavelengths[self.current_band]
            self.band_label.setText(
                f"\u03bb {wl:.{self.resolution}f} nm  "
                f"({self.current_band}/{self.bands - 1})"
            )
        else:
            self.band_label.setText(
                f"Band {self.current_band} / {self.bands - 1}"
            )

        self.update_band_image()
        self._refresh_status()

    def _step(self, delta: int):
        v = max(0, min(self.current_band + delta, self.bands - 1))
        self.band_slider.setValue(v)

    def _toggle_play(self):
        self._playing = not self._playing
        self.btn_play.setChecked(self._playing)
        self.btn_play.setText("\u23F8" if self._playing else "\u25B6")
        (self._anim_timer.start if self._playing else self._anim_timer.stop)()

    def _anim_step(self):
        self.band_slider.setValue((self.current_band + 1) % self.bands)

    def _reset(self):
        self.band_slider.setValue(0)
        self._clear_all_points()

    # ──────────────────────────────────────────────────────
    #  DISPLAY
    # ──────────────────────────────────────────────────────

    def update_band_image(self):
        """Push the current band, a 2-D ``(slit, scan)`` view of the cube, to the canvas."""
        img = self.cube[:, :, self.current_band]
        self.canvas.set_array(img)

    def _on_display_mapper_changed(self, mapper: DisplayMapper):
        self.canvas.set_display_mapper(mapper)
        self.update_band_image()

    # ──────────────────────────────────────────────────────
    #  EXPORT
    # ──────────────────────────────────────────────────────

    def _open_export(self):
        if self._export_dlg is not None and self._export_dlg.isVisible():
            self._export_dlg.raise_()
            self._export_dlg.activateWindow()
            return

        self._export_dlg = ExportDialog(
            self, self.cube, self.wavelengths,
            self.config, self.camera_bin,
        )
        self._export_dlg.show()

    # ──────────────────────────────────────────────────────
    #  STATUS / INFO
    # ──────────────────────────────────────────────────────

    def _refresh_status(self):
        parts = [f"Cube {self.h}\u00d7{self.w}\u00d7{self.bands}"]
        if self.wavelengths is not None and len(self.wavelengths) == self.bands:
            parts.append(
                f"\u03bb={self.wavelengths[self.current_band]:.{self.resolution}f} nm"
            )
        if self._spectrum_history:
            parts.append(f"{len(self._spectrum_history)} spectra")
        self.status_label.setText("  \u00b7  ".join(parts))

    def status_info(self) -> str:
        """Return a one-line summary for a host status bar.

        Returns:
            str: Current band (with wavelength in nm when known), number of
            spectra picked and the cube shape ``H×W×bands``.
        """
        parts = []
        if self.wavelengths is not None and len(self.wavelengths) == self.bands:
            wl = self.wavelengths[self.current_band]
            parts.append(
                f"Band {self.current_band}/{self.bands - 1} "
                f"({wl:.{self.resolution}f} nm)"
            )
        else:
            parts.append(f"Band {self.current_band}/{self.bands - 1}")
        if self._spectrum_history:
            parts.append(f"{len(self._spectrum_history)} spectra")
        parts.append(f"Cube {self.h}\u00d7{self.w}\u00d7{self.bands}")
        return "  \u00b7  ".join(parts)

    # ──────────────────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────────────────
    # ★ zoom helpers
    def _active_canvas(self):
        """Return the active canvas (single-band or RGB)."""
        if self._display_stack.currentIndex() == 0:
            return self.canvas
        return self._rgb_image

    def _on_zoom_changed(self, percent: float):
        self._zoom_label.setText(f"{percent:.0f}%")

    def _zoom_in(self):
        self._active_canvas().zoom_in()

    def _zoom_out(self):
        self._active_canvas().zoom_out()

    def _zoom_fit(self):
        self._active_canvas().zoom_fit()

    def _zoom_actual(self):
        self._active_canvas().zoom_actual()
    @staticmethod
    def _nav_btn(text: str, slot) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("navBtn")
        btn.setFixedSize(32, 32)
        btn.clicked.connect(slot)
        return btn

    @staticmethod
    def _val(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("valueLabel")
        return lbl

    @staticmethod
    def _sep() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.VLine)
        f.setFrameShadow(QFrame.Sunken)
        return f

    def closeEvent(self, event):
        """Stop the sweep timer, close the export dialog and release data references (Qt override)."""
        self._anim_timer.stop()
        if self._export_dlg:
            self._export_dlg.close()
            self._export_dlg = None
        self.cube = None
        self.wavelengths = None
        self.canvas = None
        self.spectrum = None
        super().closeEvent(event)