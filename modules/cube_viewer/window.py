# windows/cube_viewer.py
"""Cube viewer window with single-band and RGB composite modes.
Both modes share one set of spectra: point clicks and rectangle / lasso areas, listed in the
sidebar's SPECTRA panel and drawn as tab20-coloured curves (areas with a +-std band).

Data conventions:
    The cube is ``(slit, scan, wavelength)`` = ``(H, W, bands)`` of any numeric dtype
    (typically uint16; a ``numpy.memmap`` from ``modules.cube_io.load_envi_cube_for_viewer``
    works).  ``wavelengths`` is a 1-D array in nm with one value per band, or ``None``.
    The viewer keeps a reference to the cube (no copy) and sets it to ``None`` when closed.
    Image pixel ``(ix, iy)`` is ``(column, row)`` = ``(scan, slit)``.
    Spectra are :class:`~modules.cube_viewer.spectra.SpectrumRecord` objects in
    ``CubeViewerWindow.spectrum_set``; their statistics come from
    :mod:`modules.cube_viewer.roi_stats` (pixel-centre rule, exact integer sums, std ddof=0).

Threading:
    All widgets live in the GUI thread.  A point spectrum is read synchronously.  An area
    is computed inline when the cube is an in-RAM ndarray and its bounding box holds at
    most ``inline_bytes`` (8 MiB) of cube data; otherwise - and always for memory maps - it
    goes to the ``JobRunner`` worker thread (``modules.cube_viewer.jobs``), which
    ``closeEvent`` shuts down before the cube is released.  Export runs in a ``QThread``
    owned by ``ExportDialog`` (workers from ``modules.cube_io.export_workers``).

Example:
    >>> import numpy as np
    >>> from modules.cube_viewer import CubeViewerWindow           # doctest: +SKIP
    >>> cube = np.random.randint(0, 4096, (32, 48, 40), dtype=np.uint16)
    >>> win = CubeViewerWindow(cube, wavelengths=np.linspace(400, 1000, 40))  # doctest: +SKIP
    >>> win.show()                                                  # doctest: +SKIP
"""

from __future__ import annotations

import html
import importlib
import math
import mmap
import os
from functools import partial
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import qtawesome as qta
from PyQt5.QtCore import QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QKeySequence, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QSlider, QGroupBox, QComboBox, QPushButton,
    QFileDialog, QProgressBar, QMessageBox, QDialog,
    QSplitter, QShortcut, QScrollArea, QSizePolicy, QFrame,
    QStackedWidget,
)

from modules.live_view.mapping import DisplayMapper
from modules.ui_kit.theme import PLOT_FOREGROUND
from modules.cube_viewer.canvas import COLORMAP_GROUPS, CubeImageCanvas, colormap_lut
from modules.cube_viewer.jobs import JobRunner
from modules.cube_viewer.roi_draw import TOOLS, _RoiPixmapLabel, roi_selection
from modules.cube_viewer.roi_stats import point_stats, polygon_mask, roi_median, roi_stats, snap_rect
from modules.cube_viewer.segment_session import SegmentSession

try:  # the Segment dialog (UI part 2); optional so the viewer works without it
    _segment_dialog = importlib.import_module("modules.cube_viewer.segment_dialog")
except ModuleNotFoundError as _exc:  # pragma: no cover - depends on the build state
    if _exc.name != "modules.cube_viewer.segment_dialog":
        raise
    _segment_dialog = None
from modules.cube_viewer.spectra import POLYGON_KINDS, SpectrumSet
from modules.cube_viewer.spectra_export import (
    HAVE_OPENPYXL, export_csv, export_xlsx, exportable, spectra_text,
)
from modules.cube_viewer.spectra_panel import SpectraPanel
from modules.cube_preview.widget import TAB20_HEX
from modules.live_view.spectrum import SpectrumPlot
from modules.live_view.display_controls import DisplayControls
from modules.cube_viewer.controls import RGBCompositeControls
from modules.ui_kit.zoom_pan import ZoomPanMixin

from modules.cube_io.export_workers import (
    GeoTIFFWorker, HypercubeWorker, SpectrogramWorker,
)

# Areas whose bounding box holds at most this much cube data (px x bands x itemsize) are
# computed in the GUI thread when the cube is in RAM: about 22k px of a 185-band uint16
# cube, ~5.5 ms (a float32 box at the limit: ~10 ms, the exact float merge costs more).
INLINE_BYTES = 8 << 20
# A lasso mask is rasterised in the GUI thread when its edges cross at most this many pixel
# rows in total (polygon_mask costs ~0.15 us per edge-row crossing: ~10 ms at the limit; a
# 300-vertex freehand lasso has a few thousand).  Larger ones are rasterised in the worker.
INLINE_MASK_CROSSINGS = 1 << 16
#: Starting sidebar width (px): the Band row (Display + Colormap) and tree columns fit.
SIDEBAR_WIDTH = 340
_CURVE_WIDTH = 1.5
_FILL_ALPHA = 50
_MESSAGE_MS = 6000
# While the Space sweep plays or the band slider is dragged, the band marker moves at most
# this often: each move repaints every curve under it (pyqtgraph strokes the whole 1.5 px
# curve for any exposed strip; about 0.5 ms per noisy point curve, 0.2-0.3 ms per area).
_MARKER_SWEEP_MS = 250
_AXIS_ORDER = "(slit, scan, wavelength) = (rows, columns, bands)"
_CONFIG_SCALARS = ("m", "c", "pixel_size", "resolution", "bit_depth", "sensor_bit_depth",
                   "full_scale")
_TOOL_HINTS = {
    "point": "Point tool: click a pixel to add its spectrum",
    "rect": "Rect tool: drag over the image (Esc cancels)",
    "lasso": "Lasso tool: hold the left button and draw around an area; release closes it",
    "polygon": "Polygon tool: click the corners; Enter / right-click / double-click closes, "
               "Backspace removes the last corner",
}


def _colormap_icon(name: str) -> QIcon:
    """A 48 x 10 gradient swatch of a matplotlib colormap, for the colormap combo."""
    strip = np.ascontiguousarray(np.repeat(colormap_lut(name)[None, ::4], 10, axis=0)[:, :48])
    image = QImage(strip.data, strip.shape[1], 10, 3 * strip.shape[1], QImage.Format_RGB888).copy()
    return QIcon(QPixmap.fromImage(image))


def _segment_dialog_factory():
    """``SegmentDialog`` class (``factory(viewer)``) when ``segment_dialog`` exists, else None.

    The dialog module is optional while it is being built; a missing module disables the
    'Segment...' button's dialog with a status message instead of failing.
    """
    return getattr(_segment_dialog, "SegmentDialog", None) if _segment_dialog is not None else None


def _cube_in_ram(cube) -> bool:
    """True for an ndarray whose data is not (a view of) a memory map."""
    if not isinstance(cube, np.ndarray):
        return False
    obj = cube
    while obj is not None:
        if isinstance(obj, (np.memmap, mmap.mmap)):
            return False
        obj = getattr(obj, "base", None)
    return True


def _lasso_crossings(vertices: np.ndarray) -> int:
    """Pixel-centre rows crossed by the closed polygon's edges (polygon_mask's work)."""
    rows = np.floor(vertices[:, 1] - 0.5)
    return int(np.abs(np.diff(np.append(rows, rows[:1]))).sum())


def _finite_range(*arrays) -> tuple:
    """``(min, max)`` over the finite values of the arrays, ``(inf, -inf)`` when none."""
    lo, hi = math.inf, -math.inf
    for arr in arrays:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            lo = min(lo, float(finite.min()))
            hi = max(hi, float(finite.max()))
    return lo, hi


def _fill_brush(color: str):
    """Brush of a record's +-std band: the record colour at alpha ~50."""
    qcolor = QColor(color)
    qcolor.setAlpha(_FILL_ALPHA)
    return pg.mkBrush(qcolor)


class _PlotEntry:
    """The plot items of one record: its curve and, for areas, the +-std fill."""

    __slots__ = ("curve", "fill", "lo", "hi", "y_mean", "y_std", "median")

    def __init__(self, curve, fill=None, lo=None, hi=None, y_mean=(math.inf, -math.inf),
                 y_std=(math.inf, -math.inf)):
        self.curve = curve
        self.fill = fill
        self.lo = lo                # off-scene PlotCurveItems the fill is built from
        self.hi = hi
        self.y_mean = y_mean        # finite (min, max) of the mean
        self.y_std = y_std          # finite (min, max) of mean -+ std
        self.median = None          # dashed median curve (areas, on request)


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

    One export runs at a time: Start is disabled while it runs, the worker's thread
    quits as soon as the worker reports, and it is joined before the next export and on
    close, so a running ``QThread`` is never destroyed (that aborts the application).

    Signals:
        exported(str): path of a file written successfully.  The dialog also reacts to
        the worker's ``progressed``, ``finished`` (seconds) and ``error`` (traceback text).
    """

    exported = pyqtSignal(str)

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
        self._busy = False                      # an export is writing
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
        if self._busy:
            return                              # one export at a time
        if not self._save_path:
            QMessageBox.warning(self, "Error", "Select an output path first.")
            return

        # A band-first view, not a copy: the writers go slice by slice (copying each line,
        # band or frame as they write), so the files are byte-identical without the
        # 2.75 s / +668 MB transpose copy that used to block the GUI thread.
        cube_t = np.transpose(self.cube, (2, 0, 1))
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

        self._join_thread()                     # the previous export's thread, already quitting
        self._worker = w
        self._busy = True
        self._thread = QThread()
        w.moveToThread(self._thread)
        self._thread.started.connect(w.run)
        w.progressed.connect(self.progress.setValue)
        w.finished.connect(self._on_done)
        w.error.connect(self._on_error)
        # The worker is done once it reports: end the thread's event loop then.
        w.finished.connect(self._thread.quit)
        w.error.connect(self._thread.quit)
        self._thread.start()

    def _join_thread(self):
        """Quit and wait for the export thread (blocks while an export is still writing)."""
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None

    def _on_done(self, elapsed):
        self._busy = False
        self.exported.emit(self._save_path)
        self.lbl_status.setText(f"Completed in {elapsed:.2f}s")
        self.btn_export.setEnabled(True)
        QMessageBox.information(self, "Done", f"Saved:\n{self._save_path}")

    def _on_error(self, msg):
        self._busy = False
        self.lbl_status.setText("Export failed")
        self.btn_export.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)

    def closeEvent(self, event):
        """Stop and join the export thread, drop the cube reference (Qt override).

        Blocks until a running export finishes, because the thread is joined.
        """
        self._join_thread()
        self.cube = None
        self._worker = None
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════
#  Cube Viewer Window
# ══════════════════════════════════════════════════════════════

class CubeViewerWindow(QWidget):
    """Interactive viewer for a ``(slit, scan, wavelength)`` hyperspectral cube.

    Shows one band at a time (slider, play/pause sweep, left/right keys) or an RGB
    composite.  The SPECTRA panel in the sidebar picks the tool:

    * Point (default, today's behaviour): each left click adds the pixel's spectrum - a
      tab20-coloured dot labelled ``(ix,iy)`` and a curve of that name.
    * Rect / Lasso: drag a rectangle or draw a freehand area; its per-band mean is
      plotted with a shaded +-std band (population std, ddof=0).  Pixels count when their
      centre is inside (``roi_stats``).  Esc cancels a drawing.

    The panel's tree lists every spectrum: the check box shows / hides its curve, std band
    and outline (and keeps it out of the export), F2 / double click renames it (legend,
    image tag and export header follow), Del deletes, and the context menu sets the colour,
    toggles the std band and copies the table.  'Export...' writes the checked spectra as a
    plain CSV or an XLSX workbook (:meth:`export_spectra`).  Zoom with the wheel, pan with
    the middle button, ``+``/``-``/``0``/``1`` keys.  Includes display controls, cube info
    and an "Export Datacube" dialog (``E``).  Keyboard: Left/Right band, Space play/pause,
    R reset (band 0 and fit-to-window zoom; the spectra are kept), E export.  While the
    spectra tree has the focus these keys act on the tree.  A dashed marker in the plot
    shows the current band (during the sweep and a slider drag it follows every 250 ms).

    Unlike the widgets listed in the package ``__widgets__`` this one needs a cube,
    so it is not constructible without arguments.  Its one signal, ``cube_exported(str)``,
    gives the path of each datacube the export dialog wrote; otherwise query it with
    ``status_info()``, ``spectrum_set`` and the ``current_band`` attribute.  On
    ``closeEvent`` the background jobs are stopped, the spectra cleared and
    the cube, wavelengths, canvas and plot references dropped; the export dialog is closed.

    Args:
        cube: 3-D array ``(slit, scan, wavelength)``; kept by reference.
        wavelengths: 1-D band wavelengths in nm (length ``bands``) or ``None``.
        resolution: Decimals shown for wavelengths in labels.
        display_mapper: Optional ``DisplayMapper`` whose ``bit_depth`` initialises
            the display controls (default 16 bit).
        config: Calibration dict forwarded to the export (``m``, ``c``, ``pixel_size``);
            a numeric ``full_scale`` is also the saturation level counted in the spectra
            statistics, and the camera scalars go into the XLSX ``Metadata`` sheet.
        camera_bin: Hardware bin factor forwarded to the export (default 1).
        source_path: Optional path of the cube file: its stem names the areas
            (``'<stem>-mean-<k>'``) and the default export file, and it is recorded in the
            XLSX metadata.  Without it the stem is ``'cube'``.

    Attributes:
        spectrum_set: The :class:`~modules.cube_viewer.spectra.SpectrumSet` (palette
            ``TAB20_HEX``).
        spectra_panel: The sidebar :class:`~modules.cube_viewer.spectra_panel.SpectraPanel`.
        inline_bytes: Inline limit for area statistics (class default ``INLINE_BYTES``).

    Raises:
        ValueError: ``cube`` is not 3-D.

    Layout::

    ┌─── HSplitter ────────────────────────────────────────────┐
    │           │  ┌─ Overlay Bar (band nav) ────────────────┐  │
    │  DISPLAY  │  ├─ VSplitter ────────────────────────────┐│  │
    │   MODE    │  │  QStackedWidget                ~3/5   ││  │
    │  DISPLAY  │  │    page 0: CubeImageCanvas             ││  │
    │  SPECTRA  │  │    page 1: _RoiPixmapLabel (RGB)       ││  │
    │  CUBE INFO│  ├─ SpectrumPlot                  ~2/5   ││  │
    │  EXPORT   │  └───────────────────────────────────────┘│  │
    │  KEYBOARD │  ┌─ Status Bar ──────────────────────────┐│  │
    │           │  └───────────────────────────────────────┘│  │
    └───────────┴──────────────────────────────────────────┘

    The image / plot splitter starts at 3:2 (``setSizes([600, 400])``, scaled to the
    available height: about 415 / 277 px in a 1000x800 MDI window); its 4:1 stretch
    factors only share out later resizes.  The plot keeps its 160 px minimum.
    """

    inline_bytes = INLINE_BYTES
    cube_exported = pyqtSignal(str)

    def __init__(
        self,
        cube: np.ndarray,
        wavelengths: np.ndarray | None = None,
        resolution: int = 2,
        display_mapper: DisplayMapper | None = None,
        config: dict | None = None,
        camera_bin: int | None = None,
        source_path=None,
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
        self.source_path = os.fspath(source_path) if source_path else None

        self.current_band = 0
        self._export_dlg = None
        self._playing = False
        self._closed = False

        # ★ spectra — single source of truth for points and areas
        stem = Path(self.source_path).stem if self.source_path else "cube"
        self.spectrum_set = SpectrumSet(TAB20_HEX, source_name=stem)
        self._plot_entries: dict[int, _PlotEntry] = {}
        self._selected_rid = None
        self._saturation = self._config_number("full_scale")
        # ENVI 'data ignore value' (config key data_ignore_value): no-data in every statistic
        self._ignore_value = self._config_number("data_ignore_value")
        self._in_ram = _cube_in_ram(cube)
        self._x = (np.asarray(wavelengths, dtype=np.float64) if self._has_wavelengths()
                   else np.arange(self.bands, dtype=np.float64))

        self._jobs = JobRunner(self)
        self._jobs.done.connect(self._on_job_done)
        self._jobs.failed.connect(self._on_job_failed)

        self._message = ""
        self._message_timer = QTimer(self)
        self._message_timer.setSingleShot(True)
        self._message_timer.setInterval(_MESSAGE_MS)
        self._message_timer.timeout.connect(self._clear_message)

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(80)
        self._anim_timer.timeout.connect(self._anim_step)

        self._marker_timer = QTimer(self)
        self._marker_timer.setSingleShot(True)
        self._marker_timer.setInterval(_MARKER_SWEEP_MS)
        self._marker_timer.timeout.connect(self._update_band_marker)

        # ── single-band canvas (1-click spectrum, ROI tools) ──
        self.canvas = CubeImageCanvas()
        self.canvas.pixel_clicked.connect(self._on_canvas_pixel_clicked)

        bd = 16
        if display_mapper is not None:
            bd = display_mapper.bit_depth
        self.display_controls = DisplayControls(bit_depth=bd)
        # RAW (true brightness) by default; the Display box switches RAW / Enhance.
        self.canvas.set_display_mapper(display_mapper if display_mapper is not None
                                       else self.display_controls.current_mapper())
        self.display_controls.mapper_changed.connect(self._on_display_mapper_changed)

        # ── spectrum plot ──
        self.spectrum = SpectrumPlot()
        pw = self.spectrum.pw
        # The viewer draws its own curves; SpectrumPlot's empty curve used to be
        # dropped by the first pw.clear(), so drop it here (nothing visible changes).
        pw.removeItem(self.spectrum.curve)
        # The band marker stays in the plot for good (moved by on_band_changed); it
        # must not pull the auto range of an empty plot towards its wavelength.
        pw.removeItem(self.spectrum.band_line)
        pw.addItem(self.spectrum.band_line, ignoreBounds=True)
        # R / G / B composite bands, shown in the RGB view instead of the band marker
        self._rgb_lines = []
        for colour in ("#ff5252", "#69f0ae", "#448aff"):
            line = pg.InfiniteLine(angle=90, movable=False,
                                   pen=pg.mkPen(colour, width=1.5, style=Qt.DashLine))
            line.setVisible(False)
            pw.addItem(line, ignoreBounds=True)
            self._rgb_lines.append(line)
        # hover read-out: a crosshair and the visible spectra's values at the nearest band
        self._hover_line = pg.InfiniteLine(angle=90, movable=False,
                                           pen=pg.mkPen("#9e9e9e", width=1, style=Qt.DotLine))
        self._hover_line.setVisible(False)
        pw.addItem(self._hover_line, ignoreBounds=True)
        self._hover_text = pg.TextItem(anchor=(1, 0), fill=pg.mkBrush(30, 32, 36, 220))   # top right: the legend is top left
        self._hover_text.setZValue(100)
        self._hover_text.setVisible(False)
        pw.addItem(self._hover_text, ignoreBounds=True)
        self._hover_proxy = pg.SignalProxy(pw.scene().sigMouseMoved, rateLimit=30,
                                           slot=self._on_plot_hover)
        self._medians: dict = {}            # rid -> (B,) float64 median, computed on request
        if not self._has_wavelengths():
            pw.getAxis("bottom").setLabel("Band")

        # ── RGB controls (sidebar) ──
        self._rgb_controls = RGBCompositeControls()
        self._rgb_controls.set_cube(cube, wavelengths)
        self._rgb_controls.mode_changed.connect(self._on_display_mode_changed)
        self._rgb_controls.composite_updated.connect(self._on_composite_updated)

        # ── RGB image label (1-click spectrum; ROI drawing and overlays) ──
        self._rgb_image = _RoiPixmapLabel()
        self._rgb_image.clicked.connect(self._rgb_on_clicked)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)
        self._rgb_image.zoom_changed.connect(self._on_zoom_changed)
        for view in (self.canvas, self._rgb_image):
            view.roi_drawn.connect(self._on_roi_drawn)
            view.roi_discarded.connect(self._show_message)

        # ── SPECTRA panel (sidebar) ──
        self.spectra_panel = SpectraPanel()
        self.spectra_panel.set_image_size(self.h, self.w)
        self.spectra_panel.set_cube_name(self.spectrum_set.source_name)
        self.spectra_panel.view_mode_requested.connect(self._rgb_controls.set_mode)
        self._rgb_controls.bands_changed.connect(lambda *_: self._update_rgb_markers())
        self._rgb_controls.mode_changed.connect(self.spectra_panel.set_view_mode)
        self.btn_clear_points = self.spectra_panel.btn_clear_points
        self.btn_clear_points.clicked.connect(self._clear_all_points)
        self.spectra_panel.tool_changed.connect(self.set_tool)
        self.spectra_panel.visibility_changed.connect(self.set_spectrum_visible)
        self.spectra_panel.rename_requested.connect(self.rename_spectrum)
        self.spectra_panel.color_requested.connect(self.set_spectrum_color)
        self.spectra_panel.show_std_changed.connect(self.set_spectrum_show_std)
        self.spectra_panel.show_median_changed.connect(self.set_spectrum_show_median)
        self.spectra_panel.copy_requested.connect(self.copy_spectra)
        self.spectra_panel.delete_requested.connect(self.delete_spectra)
        self.spectra_panel.selection_changed.connect(self._on_spectrum_selected)
        self.spectra_panel.export_requested.connect(self._export_spectra_dialog)
        self.spectra_panel.class_requested.connect(self.set_spectrum_class)

        # ── segmentation (Phase 2): own worker thread, overlay on both views ──
        self.segment_session = SegmentSession(
            cube, self.spectrum_set.records, views=(self.canvas, self._rgb_image), parent=self)
        self.segment_session.state_changed.connect(self._on_segmentation_changed)
        self.spectra_panel.segment_requested.connect(self._open_segment_dialog)
        self.spectra_panel.segmentation_overlay_requested.connect(
            self.segment_session.set_overlay_visible)
        self.spectra_panel.clear_segmentation_requested.connect(self.clear_segmentation)
        self._segment_dlg = None
        if not HAVE_OPENPYXL:
            self.spectra_panel.btn_export.setToolTip(
                "Save the checked spectra as CSV.  Excel (XLSX) export needs openpyxl: "
                "pip install openpyxl")

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
        # Initial 3:2 split.  Without it Qt weighs sizeHint x stretch and pins the
        # image at the canvas's 200 px minimum ([200, 492] in a 1000x800 MDI window).
        self._vsplit.setSizes([600, 400])
        right_lay.addWidget(self._vsplit, stretch=1)

        right_lay.addWidget(self._build_status_bar(), stretch=0)

        self._hsplit.addWidget(right)
        self._hsplit.setStretchFactor(0, 0)
        self._hsplit.setStretchFactor(1, 1)
        self._hsplit.setSizes([SIDEBAR_WIDTH, 800])

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
        # RAW / Enhance sits with the Band | RGB toggle; Band view only (RGB has its own stretch)
        band_box = QWidget()                # Band view only: RAW/Enhance + colormap
        band_lay = QVBoxLayout(band_box)
        band_lay.setContentsMargins(0, 0, 0, 0)
        band_lay.setSpacing(4)
        band_lay.addWidget(self.display_controls)
        band_lay.addLayout(self._build_colormap_row())
        g_mode_lay.addWidget(band_box)
        g_mode.setLayout(g_mode_lay)
        lay.addWidget(g_mode)
        self._display_group = band_box

        # ★ spectra: tools, tree, Export… and Clear All Points (the old button's slot)
        g_spectra = QGroupBox("SPECTRA")
        l_spectra = QVBoxLayout()
        l_spectra.addWidget(self.spectra_panel)
        g_spectra.setLayout(l_spectra)
        lay.addWidget(g_spectra)

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
        self.btn_export = QPushButton("  Export Datacube →")
        self.btn_export.setMinimumHeight(38)
        self.btn_export.clicked.connect(self._open_export)
        lay.addWidget(self.btn_export)

        # keyboard shortcuts
        g3 = QGroupBox("KEYBOARD")
        l3 = QVBoxLayout()
        l3.setSpacing(2)
        for key, desc in [
            ("← →",  "Prev / Next band"),
            ("Space",         "Play / Pause sweep"),
            ("E",             "Export dialog"),
            ("R",             "Reset band + zoom"),
            ("Esc",           "Cancel rect / lasso"),
            ("Del",           "Delete spectrum (tree)"),
            ("F2",            "Rename spectrum (tree)"),
        ]:
            row = QHBoxLayout()
            kl = QLabel(key)
            kl.setFixedWidth(52)
            dl = QLabel(desc)
            dl.setWordWrap(True)            # never sets the sidebar width
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
        self.btn_prev = self._nav_btn("◀", lambda: self._step(-1))
        self.btn_play = self._nav_btn("▶", self._toggle_play)
        self.btn_play.setCheckable(True)
        self.btn_next = self._nav_btn("▶", lambda: self._step(1))

        # ★ band slider
        self.band_slider = QSlider(Qt.Horizontal)
        self.band_slider.setRange(0, max(0, self.bands - 1))
        self.band_slider.valueChanged.connect(self.on_band_changed)
        self.band_slider.sliderReleased.connect(self._update_band_marker)

        # ★ labels
        self.band_label = QLabel()
        self.band_label.setObjectName("valueLabel")
        self.band_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.band_label.setMinimumWidth(180)

        self.roi_label = QLabel("")
        self.roi_label.setObjectName("valueLabel")
        self.roi_label.setTextFormat(Qt.PlainText)     # spectrum names are text, never markup
        self.roi_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # a long spectrum name is clipped instead of widening the whole window
        self.roi_label.setMinimumWidth(1)

        # ★ zoom controls
        self._zoom_label = QLabel("100%")
        self._zoom_label.setObjectName("valueLabel")
        self._zoom_label.setAlignment(Qt.AlignCenter)
        self._zoom_label.setFixedWidth(48)

        btn_zoom_out = self._nav_btn("−", self._zoom_out)
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
        self.status_label.setTextFormat(Qt.PlainText)  # messages quote spectrum names: never markup
        self.status_label.setMinimumWidth(1)      # long messages are clipped, never widen the window
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
        # Only while the focus is inside this viewer: a second viewer (or another module
        # window in the MDI area) never receives these keys.
        for shortcut in self.findChildren(QShortcut):
            if shortcut.parent() is self:
                shortcut.setContext(Qt.WidgetWithChildrenShortcut)

    # ──────────────────────────────────────────────────────
    #  DISPLAY MODE SWITCH
    # ──────────────────────────────────────────────────────
    def _rgb_on_clicked(self, ix: int, iy: int):
        """Left-click on RGB image (point tool) — add that pixel's spectrum."""
        self.add_point(ix, iy)

    # ★ _on_display_mode_changed — update zoom label when switching
    def _on_display_mode_changed(self, mode: str):
        is_rgb = (mode == "rgb")
        self._display_group.setVisible(not is_rgb)      # RAW/Enhance applies to the Band view only
        if is_rgb:
            self._display_stack.setCurrentIndex(1)
            self._overlay.setVisible(False)
            self.canvas.cancel_drawing()
        else:
            self._display_stack.setCurrentIndex(0)
            self._overlay.setVisible(True)
            self._rgb_image.cancel_drawing()
            self.update_band_image()

        # both views draw the same record overlays, so nothing needs re-syncing
        # ★ update zoom label for active canvas
        self._on_zoom_changed(self._active_canvas().zoom_percent)
        self._update_rgb_markers()

    def _on_composite_updated(self, rgb8: np.ndarray):
        h, w, ch = rgb8.shape
        fmt = QImage.Format_RGB888
        qimg = QImage(rgb8.data, w, h, ch * w, fmt).copy()
        pixmap = QPixmap.fromImage(qimg)
        self._rgb_image.setPixmapScaled(pixmap)

    # ──────────────────────────────────────────────────────
    #  SINGLE-BAND MODE — 1-click spectrum
    # ──────────────────────────────────────────────────────

    def _on_canvas_pixel_clicked(self, ix: int, iy: int):
        """1 click on single-band canvas = 1 spectrum."""
        self.add_point(ix, iy)

    # ──────────────────────────────────────────────────────
    #  SPECTRA — tools, points and areas
    # ──────────────────────────────────────────────────────

    def set_tool(self, name: str) -> None:
        """Select the image tool in both views and the panel.

        Args:
            name: ``'point'`` (click = pixel spectrum), ``'rect'`` or ``'lasso'``.

        Raises:
            ValueError: Unknown tool name.
        """
        if name not in TOOLS:
            raise ValueError(f"tool must be one of {TOOLS}, got {name!r}")
        changed = self.canvas.tool != name or self._rgb_image.tool != name
        self.canvas.set_tool(name)
        self._rgb_image.set_tool(name)
        if self.spectra_panel.tool() != name:
            self.spectra_panel.set_tool(name)
        if changed:
            self._show_message(_TOOL_HINTS[name])

    def add_point(self, ix: int, iy: int):
        """Add the spectrum of pixel ``(ix, iy)`` (column, row), as a left click does.

        The record is named ``'(ix,iy)'`` and coloured ``TAB20_HEX[n % 20]`` by creation
        count, exactly as the click dots and curves always were; its statistics are read
        synchronously (``point_stats``).

        Returns:
            SpectrumRecord: The new record (``'ready'``, or ``'failed'`` when no band of
            the pixel is finite).
        """
        if self._closed:
            return None
        record = self.spectrum_set.add("point", {"x": int(ix), "y": int(iy)})
        stats = point_stats(self.cube, int(ix), int(iy), saturation=self._saturation,
                            ignore_value=self._ignore_value)
        self.spectrum_set.set_stats(record.rid, stats)
        self._record_added(record)
        if record.state == "failed":
            self.spectra_panel.update_record(record, note="no finite value in any band")
        n = len(self.spectrum_set)
        self.roi_label.setText(f"Point {n}: x={ix}  y={iy}  ({n} spectra)")
        self._refresh_status()
        return record

    def _on_roi_drawn(self, kind: str, geometry: dict):
        """A view finished a rectangle or lasso."""
        self.add_area(kind, geometry)

    def add_area(self, kind: str, geometry: dict):
        """Add a rectangle or lasso spectrum (mean and std of the pixels inside).

        Small areas of an in-RAM cube (bounding box at most ``inline_bytes`` of cube data)
        are computed at once; the others are queued on the worker thread and the record
        stays ``'pending'`` (dashed outline, 'computing...') until the result arrives.

        Args:
            kind: ``'rect'``, ``'lasso'`` or ``'polygon'`` (vertex geometry like a lasso).
            geometry: Rect ``{'c0', 'r0', 'c1', 'r1'}`` (ints, half-open) or lasso
                ``{'vertices': [[x, y], ...]}`` (float image-corner coordinates), as
                emitted by ``roi_drawn``.

        Returns:
            SpectrumRecord | None: The record, or None when the area holds no pixel
            centre (discarded with a status message).

        Raises:
            ValueError: Unknown ``kind`` or invalid geometry values.
            KeyError: A geometry key is missing.
        """
        if self._closed:
            return None
        h, w = self.h, self.w
        mask = None
        if kind == "rect":
            bbox, _, n_px = roi_selection("rect", geometry, h, w)
            if n_px == 0:
                self._show_message("Rectangle discarded: it encloses no pixel centre")
                return None
            geometry = dict(geometry, c0=bbox[0], r0=bbox[1], c1=bbox[2], r1=bbox[3])
        elif kind in POLYGON_KINDS:
            label = kind.capitalize()
            vertices = np.asarray(geometry["vertices"], dtype=np.float64).reshape(-1, 2)
            if vertices.shape[0] < 3:
                self._show_message(f"{label} discarded: it needs at least 3 points")
                return None
            bbox = snap_rect(float(vertices[:, 0].min()), float(vertices[:, 1].min()),
                             float(vertices[:, 0].max()), float(vertices[:, 1].max()), h, w)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                self._show_message(f"{label} discarded: it encloses no pixel centre")
                return None
            if _lasso_crossings(vertices) <= INLINE_MASK_CROSSINGS:
                c0, r0, mask = polygon_mask(vertices, h, w)
                if not mask.any():
                    self._show_message(f"{label} discarded: it encloses no pixel centre")
                    return None
                bbox = (c0, r0, c0 + mask.shape[1], r0 + mask.shape[0])
        else:
            raise ValueError(f"kind must be 'rect', 'lasso' or 'polygon', got {kind!r}")

        record = self.spectrum_set.add(kind, geometry)
        self._record_added(record)
        c0, r0, c1, r1 = bbox
        cost = (c1 - c0) * (r1 - r0) * self.bands * np.dtype(self.cube.dtype).itemsize
        inline = self._in_ram and cost <= self.inline_bytes and (kind == "rect" or mask is not None)
        if inline:
            try:
                stats = roi_stats(self.cube, bbox, mask, saturation=self._saturation,
                                  ignore_value=self._ignore_value)
            except Exception as exc:  # noqa: BLE001 - shown on the record, never raised in a slot
                self._record_failed(record.rid, f"{type(exc).__name__}: {exc}")
            else:
                self._record_finished(record.rid, stats)
        else:
            self._jobs.submit(record.rid, self._area_job(record, bbox, mask))
            n = len(self.spectrum_set)
            self.roi_label.setText(f"{record.name}: computing…  ({n} spectra)")
            self._refresh_status()
        return record

    def _area_job(self, record, bbox, mask):
        """The worker callable for an area: (mask, if still needed, then) ``roi_stats``.

        It holds its own references to the cube and the geometry, never to widgets.
        """
        cube, saturation, ignore_value = self.cube, self._saturation, self._ignore_value
        kind, geometry, h, w = record.kind, record.geometry, self.h, self.w

        def job(cancel):
            box, local_mask = bbox, mask
            if kind in POLYGON_KINDS and local_mask is None:
                # a huge lasso: its mask is rasterised here, cancellable per ~10 ms batch
                box, local_mask, _ = roi_selection(kind, geometry, h, w, cancel=cancel)
            return roi_stats(cube, box, local_mask, saturation=saturation,
                             ignore_value=ignore_value, cancel=cancel)

        return job

    def _on_job_done(self, rid: int, stats):
        """Worker result (GUI thread); late results of deleted records are ignored."""
        if self._closed:
            return
        if rid < 0:                         # a median job (key -rid - 1)
            self._median_finished(-rid - 1, stats)
        else:
            self._record_finished(rid, stats)

    def _on_job_failed(self, rid: int, message: str):
        """Worker error (GUI thread)."""
        if self._closed:
            return
        if rid < 0:
            self._show_message(f"Median failed: {message}")
        else:
            self._record_failed(rid, message)

    # ── median spectra (Phase 4) ──────────────────────────

    @staticmethod
    def _median_key(rid: int) -> int:
        """JobRunner key of a record's median job (negative: never a record id)."""
        return -int(rid) - 1

    def set_spectrum_show_median(self, rid: int, show: bool) -> None:
        """Show / hide an area's dashed median curve; the first show computes it in the worker."""
        record = self.spectrum_set.get(rid)
        if record is None or not record.is_area:
            return
        record.show_median = bool(show)
        if show and rid not in self._medians:
            if not self._jobs.is_pending(self._median_key(rid)):
                self._jobs.submit(self._median_key(rid), self._median_job(record))
                self._show_message(f"{record.name}: computing the median…")
            return
        self._apply_median_curve(record)

    def _median_job(self, record):
        """Worker job: the median of a record's pixels (own references, no widgets)."""
        cube, ignore_value = self.cube, self._ignore_value
        kind, geometry, h, w = record.kind, record.geometry, self.h, self.w

        def job(cancel):
            box, mask, n_px = roi_selection(kind, geometry, h, w, cancel=cancel)
            if n_px == 0:
                raise ValueError("no pixel inside")
            return roi_median(cube, box, mask, ignore_value=ignore_value, cancel=cancel)
        return job

    def _median_finished(self, rid: int, median):
        """GUI thread: store a median and draw it if still wanted."""
        record = self.spectrum_set.get(rid)
        if record is None:
            return
        self._medians[rid] = np.asarray(median, dtype=np.float64)
        self._apply_median_curve(record)
        self._refresh_status()

    def _apply_median_curve(self, record):
        """Create / update / hide the dashed median curve of ``record``."""
        entry = self._plot_entries.get(record.rid)
        median = self._medians.get(record.rid)
        if entry is None or median is None:
            return
        pen = pg.mkPen(color=record.color, width=_CURVE_WIDTH, style=Qt.DashLine)
        if entry.median is None:
            entry.median = pg.PlotDataItem(self._x, median, pen=pen, connect="finite")
            self.spectrum.pw.addItem(entry.median)       # no name: no legend entry
        else:
            entry.median.setPen(pen)
        entry.median.setVisible(bool(record.visible and record.show_median))

    # ── R / G / B markers and hover read-out (Phase 4) ────

    def _band_x(self, band: int) -> float:
        """x position of a band on the plot (nm, or the band index)."""
        return float(self.wavelengths[band]) if self._has_wavelengths() else float(band)

    def _update_rgb_markers(self):
        """RGB view: dashed R / G / B lines at the composite bands; Band view: the band marker."""
        if self._closed:
            return
        rgb = self._rgb_controls.is_rgb_mode()
        for line, band in zip(self._rgb_lines, self._rgb_controls.rgb_bands()):
            line.setPos(self._band_x(band))
            line.setVisible(rgb)
        self.spectrum.band_line.setVisible(not rgb)

    def _on_plot_hover(self, event):
        """Crosshair + values of the visible spectra at the band nearest the pointer."""
        if self._closed:
            return
        pos = event[0]
        plot = self.spectrum.pw.getPlotItem()
        if not plot.sceneBoundingRect().contains(pos) or not self._plot_entries:
            self._hover_line.setVisible(False)
            self._hover_text.setVisible(False)
            return
        x = plot.vb.mapSceneToView(pos).x()
        band = int(np.argmin(np.abs(np.asarray(self._x, dtype=np.float64) - x)))
        bx = self._band_x(band)
        unit = f"{bx:.1f} nm" if self._has_wavelengths() else f"band {band}"
        lines = [f"<b>{unit}</b>  (band {band})"]
        for record in self.spectrum_set:
            if record.rid not in self._plot_entries or not record.visible:
                continue
            value = float(record.stats.mean[band])
            text = f"{value:,.1f}" if math.isfinite(value) else "-"
            if record.is_area and math.isfinite(float(record.stats.std[band])):
                text += f" &plusmn; {float(record.stats.std[band]):,.1f}"
            median = self._medians.get(record.rid)
            if median is not None and record.show_median and math.isfinite(float(median[band])):
                text += f" (med {float(median[band]):,.1f})"
            lines.append(f'<span style="color:{record.color}">&#9632;</span> '
                         f"{html.escape(record.name)}: {text}")
            if len(lines) > 12:
                lines.append("…")
                break
        self._hover_line.setPos(bx)
        self._hover_line.setVisible(True)
        (_x0, x1), (_y0, y1) = plot.vb.viewRange()
        self._hover_text.setHtml("<br>".join(lines))
        self._hover_text.setPos(x1, y1)
        self._hover_text.setVisible(True)

    def _record_finished(self, rid: int, stats):
        """Attach the statistics of an area and show it (or drop a 0-px lasso)."""
        record = self.spectrum_set.get(rid)
        if record is None:
            return
        if int(stats.n_px) == 0:
            self._discard_record(rid)
            return
        self.spectrum_set.set_stats(rid, stats)
        self.segment_session.mark_stale()
        n = len(self.spectrum_set)
        if record.is_ready:
            self.spectra_panel.update_record(record)
            self._add_plot_items(record)
            self.roi_label.setText(f"{record.name}: {int(stats.n_px):,} px  ({n} spectra)")
        else:
            self.spectra_panel.update_record(record, note="no valid pixel in any band")
            self.roi_label.setText(f"{record.name}: no valid pixel  ({n} spectra)")
            self._show_message(f"{record.name}: no valid pixel in any band")
        self._repaint_overlays()
        self._refresh_status()

    def _record_failed(self, rid: int, message: str):
        """Mark a record failed (job error or cancellation not asked for by the viewer)."""
        record = self.spectrum_set.set_stats(rid, None)
        if record is None:
            return
        self.spectra_panel.update_record(record, note=message)
        self._repaint_overlays()
        self.roi_label.setText(f"{record.name}: failed  ({len(self.spectrum_set)} spectra)")
        self._show_message(f"{record.name}: statistics failed ({message})")

    def _discard_record(self, rid: int):
        """Drop an area that turned out to hold no pixel centre (a thin lasso rasterised in
        the worker), as if it had never been added: when no record was added after it, the
        next one gets its colour and name (``SpectrumSet.discard``)."""
        self._jobs.cancel(rid)
        self._remove_plot_items(rid)
        self.spectra_panel.remove_record(rid)
        self.spectrum_set.discard(rid)
        self.segment_session.mark_stale()
        if self._selected_rid not in self.spectrum_set:
            self._selected_rid = None
        self._sync_overlays()
        n = len(self.spectrum_set)
        self.spectra_panel.btn_export.setEnabled(n > 0)
        self.roi_label.setText(f"Lasso discarded  ({n} spectra)" if n else "Lasso discarded")
        self._show_message("Lasso discarded: it encloses no pixel centre")

    def _record_added(self, record):
        """Show a new record in the panel, the views and (when ready) the plot."""
        self.segment_session.mark_stale()
        self.spectra_panel.add_record(record)
        if record.is_ready:
            self._add_plot_items(record)
        self._sync_overlays()
        self.spectra_panel.btn_export.setEnabled(True)

    # ──────────────────────────────────────────────────────
    #  SPECTRA — edit, hide, delete
    # ──────────────────────────────────────────────────────

    def rename_spectrum(self, rid: int, name: str) -> bool:
        """Rename a spectrum; legend, curve name, tree, image tag and export follow.

        A refused name (empty, used by another spectrum, control characters) leaves the
        old one everywhere and shows the reason in the status bar.

        Returns:
            bool: True when the name was accepted.
        """
        record = self.spectrum_set.get(rid)
        if record is None:
            return False
        old_class = record.class_name
        try:
            self.spectrum_set.rename(rid, name)
        except (TypeError, ValueError) as exc:
            self.spectra_panel.update_record(record)
            self._show_message(f"Name not changed: {exc}")
            return False
        if record.class_name != old_class:
            self.segment_session.mark_stale()
        entry = self._plot_entries.get(rid)
        if entry is not None:
            entry.curve.opts["name"] = record.name
            legend = self.spectrum.pw.getPlotItem().legend
            label = legend.getLabel(entry.curve) if legend is not None else None
            if label is not None:
                label.setText(html.escape(record.name, quote=False))
                legend.updateSize()
        self.spectra_panel.update_record(record)
        self._repaint_overlays()
        return True

    def set_spectrum_visible(self, rid: int, visible: bool) -> None:
        """Show or hide a spectrum: curve, std band, image outline and export together."""
        record = self.spectrum_set.get(rid)
        if record is None:
            return
        if bool(record.visible) != bool(visible):
            self.segment_session.mark_stale()
        record.visible = bool(visible)
        self._apply_visibility(record)
        self.spectra_panel.update_record(record)
        self._repaint_overlays()
        self._update_y_range()

    def set_spectrum_show_std(self, rid: int, show: bool) -> None:
        """Show or hide the shaded +-std band of an area spectrum (exports keep the std)."""
        record = self.spectrum_set.get(rid)
        if record is None:
            return
        record.show_std = bool(show)
        self._apply_visibility(record)
        self.spectra_panel.update_record(record)
        self._update_y_range()

    def set_spectrum_color(self, rid: int, color: str) -> None:
        """Give one spectrum a new ``'#rrggbb'`` colour (curve, band, legend, outline, tree)."""
        try:
            record = self.spectrum_set.set_color(rid, color)
        except KeyError:
            return
        except ValueError as exc:
            self._show_message(str(exc))
            return
        entry = self._plot_entries.get(rid)
        if entry is not None:
            entry.curve.setPen(pg.mkPen(color=record.color, width=_CURVE_WIDTH))
            if entry.fill is not None:
                entry.fill.setBrush(_fill_brush(record.color))
            if entry.median is not None:
                entry.median.setPen(pg.mkPen(color=record.color, width=_CURVE_WIDTH,
                                             style=Qt.DashLine))
            self._update_legend_sample(entry.curve)
        self.spectra_panel.update_record(record)
        self._repaint_overlays()
        self.segment_session.mark_stale()
        self.segment_session.refresh_colors()

    def set_spectrum_class(self, rid: int, class_name: str) -> bool:
        """Set the class of a spectrum (records with the same class are pooled by the
        segmentation).  A refused name (empty, control characters) leaves the old class.

        Returns:
            bool: True when the class was accepted.
        """
        record = self.spectrum_set.get(rid)
        if record is None:
            return False
        old = record.class_name
        try:
            self.spectrum_set.set_class(rid, class_name)
        except (TypeError, ValueError) as exc:
            self.spectra_panel.update_record(record)
            self._show_message(f"Class not changed: {exc}")
            return False
        self.spectra_panel.update_record(record)
        if record.class_name != old:
            self.segment_session.mark_stale()
        return True

    def delete_spectra(self, rids) -> int:
        """Delete spectra by rid; their pending jobs are cancelled.  Other colours stay.

        Returns:
            int: How many records were deleted.
        """
        removed = 0
        for rid in list(rids):
            self._jobs.cancel(rid)
            self._remove_plot_items(rid)
            self.spectra_panel.remove_record(rid)
            if self.spectrum_set.remove(rid) is not None:
                removed += 1
        if removed:
            self.segment_session.mark_stale()
        if self._selected_rid not in self.spectrum_set:
            self._selected_rid = None
        self._sync_overlays()
        self._update_y_range()
        n = len(self.spectrum_set)
        self.spectra_panel.btn_export.setEnabled(n > 0)
        if removed:
            self.roi_label.setText(f"{removed} deleted  ({n} spectra)" if n else "")
        self._refresh_status()
        return removed

    def copy_spectra(self, rids) -> str:
        """Copy the table of the given spectra (hidden ones too) to the clipboard as TSV.

        Returns:
            str: The text copied ('' when none of them is computed).
        """
        records = [r for r in (self.spectrum_set.get(rid) for rid in rids) if r is not None]
        try:
            text = spectra_text(records, self.wavelengths, visible_only=False)
        except ValueError as exc:
            self._show_message(f"Nothing copied: {exc}")
            return ""
        QApplication.clipboard().setText(text)
        self._show_message(f"Copied {len(exportable(records, visible_only=False))} spectra "
                           "as tab-separated text")
        return text

    def _on_spectrum_selected(self, rid):
        """Tree selection: draw that record with the thick outline in both views."""
        if self._closed:
            return
        self._selected_rid = rid
        self.canvas.set_selected_roi(rid)
        self._rgb_image.set_selected_roi(rid)

    # ──────────────────────────────────────────────────────
    #  SPECTRA — plot items (one curve per record, never pw.clear())
    # ──────────────────────────────────────────────────────

    def _add_plot_items(self, record):
        """Create the curve (and for areas the +-std fill) of a ready record."""
        if record.rid in self._plot_entries:
            return
        pw = self.spectrum.pw
        if pw.getPlotItem().legend is None:
            pw.addLegend()
        stats = record.stats
        mean = np.asarray(stats.mean, dtype=np.float64)
        curve = pw.plot(self._x, mean, pen=pg.mkPen(color=record.color, width=_CURVE_WIDTH),
                        name=record.name, connect="finite")
        escaped = html.escape(record.name, quote=False)
        if escaped != record.name:
            pw.getPlotItem().legend.getLabel(curve).setText(escaped)
        entry = _PlotEntry(curve, y_mean=_finite_range(mean))
        if record.is_area:
            std = np.asarray(stats.std, dtype=np.float64)
            lo, hi = mean - std, mean + std
            entry.lo = pg.PlotCurveItem(self._x, lo, connect="finite")    # off-scene
            entry.hi = pg.PlotCurveItem(self._x, hi, connect="finite")
            entry.fill = pg.FillBetweenItem(entry.lo, entry.hi, brush=_fill_brush(record.color))
            pw.addItem(entry.fill)
            entry.y_std = _finite_range(mean, lo, hi)
        self._plot_entries[record.rid] = entry
        self._apply_visibility(record)
        # a click on the legend sample toggles the curve: keep the record in step
        curve.visibleChanged.connect(partial(self._on_curve_visible_changed, record.rid))
        self._update_y_range()

    def _remove_plot_items(self, rid: int):
        """Remove a record's curve, legend entry and fill (the band marker stays)."""
        entry = self._plot_entries.pop(rid, None)
        if entry is None:
            return
        try:
            entry.curve.visibleChanged.disconnect()
        except TypeError:
            pass
        pw = self.spectrum.pw
        pw.removeItem(entry.curve)
        if entry.fill is not None:
            pw.removeItem(entry.fill)
        if entry.median is not None:
            pw.removeItem(entry.median)
        self._jobs.cancel(self._median_key(rid))
        self._medians.pop(rid, None)

    def _apply_visibility(self, record):
        """Show / hide the plot items of ``record`` from its ``visible`` / ``show_std``.

        The curve's legend sample is repainted too: it draws the hidden state, and
        pyqtgraph repaints it only after a click on the sample itself.
        """
        entry = self._plot_entries.get(record.rid)
        if entry is None:
            return
        entry.curve.setVisible(bool(record.visible))
        self._update_legend_sample(entry.curve)
        if entry.fill is not None:
            entry.fill.setVisible(bool(record.visible and record.show_std))
        if entry.median is not None:
            entry.median.setVisible(bool(record.visible and record.show_median))

    def _update_legend_sample(self, curve):
        """Repaint the legend sample of ``curve`` (after a pen or visibility change)."""
        legend = self.spectrum.pw.getPlotItem().legend
        for sample, _label in (legend.items if legend is not None else ()):
            if sample.item is curve:
                sample.update()
                return

    def _on_curve_visible_changed(self, rid: int):
        """The curve was shown / hidden from outside (legend click): follow it."""
        if self._closed:
            return
        entry = self._plot_entries.get(rid)
        record = self.spectrum_set.get(rid)
        if entry is None or record is None:
            return
        visible = entry.curve.isVisibleTo(entry.curve.parentItem())
        if visible != bool(record.visible):
            self.set_spectrum_visible(rid, visible)

    def _update_y_range(self):
        """Fit the y range to the visible curves (and shown std bands), 8 % margin."""
        lo, hi = math.inf, -math.inf
        for record in self.spectrum_set:
            entry = self._plot_entries.get(record.rid)
            if entry is None or not record.visible:
                continue
            a, b = entry.y_std if (entry.fill is not None and record.show_std) else entry.y_mean
            lo, hi = min(lo, a), max(hi, b)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return
        if hi <= lo:
            hi = lo + 1.0
        margin = (hi - lo) * 0.08 + 1e-6
        self.spectrum.pw.setYRange(lo - margin, hi + margin, padding=0)

    # ──────────────────────────────────────────────────────
    #  SPECTRA — overlays on both views
    # ──────────────────────────────────────────────────────

    def _sync_overlays(self):
        """Hand the current record list to both views (after add / delete / clear)."""
        records = self.spectrum_set.records()
        for view in (self.canvas, self._rgb_image):
            view.set_roi_overlays(records, self._selected_rid)

    def _repaint_overlays(self):
        """Repaint both views after an in-place record change (name, colour, state...)."""
        self.canvas.update()
        self._rgb_image.update()

    # ──────────────────────────────────────────────────────
    #  CLEAR ALL POINTS
    # ──────────────────────────────────────────────────────

    def _clear_all_points(self):
        """Clear all spectra (points and areas): records, jobs, overlays, plot and the
        segmentation."""
        self._jobs.cancel_all()
        self.segment_session.cancel()
        self.segment_session.clear()
        for rid in list(self._plot_entries):
            self._remove_plot_items(rid)
        self.spectrum_set.clear()
        self.spectra_panel.clear()
        self._selected_rid = None
        self._sync_overlays()
        self.spectra_panel.btn_export.setEnabled(False)
        self.roi_label.setText("")
        self._refresh_status()

    # ──────────────────────────────────────────────────────
    #  SEGMENTATION
    # ──────────────────────────────────────────────────────

    def clear_segmentation(self):
        """Remove the class overlay, the Segmentation node and the stored result
        (ROIs and spectra stay)."""
        self.segment_session.clear()

    def _on_segmentation_changed(self):
        """Session state changed: redraw the Segmentation node of the tree."""
        if not self._closed:
            self.spectra_panel.set_segmentation(self.segment_session.summary())

    def _open_segment_dialog(self):
        """'Segment...': show the non-modal Segment dialog (built by ``segment_dialog``)."""
        if self._closed:
            return
        if self._segment_dlg is None:
            factory = _segment_dialog_factory()
            if factory is None:
                self._show_message("The Segment dialog is not available")
                return
            self._segment_dlg = factory(self)
        self._segment_dlg.show()
        self._segment_dlg.raise_()
        self._segment_dlg.activateWindow()

    # ──────────────────────────────────────────────────────
    #  SPECTRA — export
    # ──────────────────────────────────────────────────────

    def export_spectra(self, path, ids=None) -> str:
        """Write spectra to ``path``: XLSX when it ends in ``.xlsx``, else plain CSV.

        Args:
            path: Target file.  ``.xlsx`` -> workbook with sheets Spectra, ROIs and
                Metadata (needs openpyxl); anything else -> the plain PVC-style CSV
                (``.csv`` appended when the name has no extension).
            ids: rids to export (computed ones, hidden included); None exports every
                visible computed spectrum.

        Returns:
            str: The path written.

        Raises:
            ValueError: Nothing to export (none ready / visible).
            ImportError: XLSX requested without openpyxl.
            OSError: The file cannot be written (e.g. open in Excel); it is unchanged.
        """
        if ids is None:
            records, visible_only = self.spectrum_set.records(), True
        else:
            records = [r for r in (self.spectrum_set.get(rid) for rid in ids) if r is not None]
            visible_only = False
        path = os.fspath(path)
        if path.lower().endswith(".xlsx"):
            return export_xlsx(path, records, self.wavelengths, self.export_metadata(),
                               visible_only=visible_only)
        return export_csv(path, records, self.wavelengths, visible_only=visible_only)

    def export_metadata(self) -> dict:
        """Metadata for the XLSX ``Metadata`` sheet: source, units, dtype, shape, axis order,
        saturation level and the camera scalars of ``config`` (format_version, std ddof,
        pixel rule and exported_at are added by the writer)."""
        dtype = np.dtype(self.cube.dtype) if self.cube is not None else None
        meta = {
            "source": self.source_path or "",
            "units": ("counts" if dtype is not None and dtype.kind in "iu"
                      else "unknown (float cube)"),
            "dtype": str(dtype),
            "shape": [self.h, self.w, self.bands],
            "axis_order": _AXIS_ORDER,
            "saturation_level": (self._saturation if self._saturation is not None
                                 else "none (not counted)"),
            "data_ignore_value": (self._ignore_value if self._ignore_value is not None
                                  else "none"),
            "camera_bin": self.camera_bin,
        }
        for key in _CONFIG_SCALARS:
            value = self.config.get(key)
            if isinstance(value, (bool, int, float, str, np.integer, np.floating)):
                meta[key] = value.item() if isinstance(value, np.generic) else value
        return meta

    def default_export_path(self) -> str:
        """``<stem>_spectra.csv`` next to the source file (or in the working folder)."""
        name = f"{self.spectrum_set.source_name}_spectra.csv"
        if self.source_path:
            return os.path.join(os.path.dirname(os.path.abspath(self.source_path)), name)
        return name

    def _export_spectra_dialog(self):
        """'Export...' button: ask for a CSV / XLSX file and write the checked spectra."""
        records = self.spectrum_set.records()
        count = len(exportable(records))
        if not count:
            QMessageBox.information(
                self, "Export spectra",
                "There is no checked spectrum with computed statistics to export.")
            return
        filters = "CSV (*.csv);;Excel workbook (*.xlsx)" if HAVE_OPENPYXL else "CSV (*.csv)"
        path, selected = QFileDialog.getSaveFileName(
            self, "Export spectra", self.default_export_path(), filters)
        if not path:
            return
        if not os.path.splitext(path)[1]:            # no extension typed: the chosen filter decides
            path += ".xlsx" if selected.startswith("Excel") else ".csv"
        try:
            written = self.export_spectra(path)
        except Exception as exc:  # noqa: BLE001 - every export error goes to the user
            QMessageBox.warning(self, "Export spectra", str(exc))
            return
        self._show_message(f"Exported {count} spectra: {written}")

    # ──────────────────────────────────────────────────────
    #  BAND NAVIGATION
    # ──────────────────────────────────────────────────────

    def on_band_changed(self, band: int):
        """Select a band: update the label, the image, the plot's band marker and the status line.

        The marker moves at once, except while the Space sweep plays or the slider is
        dragged: then it follows at most every 250 ms (each move repaints the curves under
        it) and jumps to the band when the sweep stops or the slider is released.

        Args:
            band: Band index in ``[0, bands)`` (0-based, not a wavelength).
        """
        self.current_band = int(band)
        if self._playing or self.band_slider.isSliderDown():
            if not self._marker_timer.isActive():
                self._marker_timer.start()
        else:
            self._update_band_marker()

        if self._has_wavelengths():
            wl = self.wavelengths[self.current_band]
            self.band_label.setText(
                f"λ {wl:.{self.resolution}f} nm  "
                f"({self.current_band}/{self.bands - 1})"
            )
            self.spectra_panel.set_band(
                self.current_band,
                f"{wl:.{self.resolution}f} nm (band {self.current_band})")
        else:
            self.band_label.setText(
                f"Band {self.current_band} / {self.bands - 1}"
            )
            self.spectra_panel.set_band(self.current_band)

        self.update_band_image()
        self._refresh_status()

    def _update_band_marker(self):
        """Put the dashed band marker at the current band, on the x axis the curves use.

        That is ``wavelengths[band]`` in nm when there is one wavelength per band,
        otherwise the band index.  A deferred move (sweep / slider drag) is done now.
        """
        self._marker_timer.stop()
        if self._closed:
            return
        if self._has_wavelengths():
            x = float(self.wavelengths[self.current_band])
        else:
            x = float(self.current_band)
        line = self.spectrum.band_line
        line.setPos(x)
        line.setVisible(not self._rgb_controls.is_rgb_mode())      # RGB view: R/G/B lines

    def _step(self, delta: int):
        v = max(0, min(self.current_band + delta, self.bands - 1))
        self.band_slider.setValue(v)

    def _toggle_play(self):
        self._playing = not self._playing
        self.btn_play.setChecked(self._playing)
        self.btn_play.setText("⏸" if self._playing else "▶")
        (self._anim_timer.start if self._playing else self._anim_timer.stop)()
        if not self._playing:
            self._update_band_marker()          # the marker lagged by up to 250 ms

    def _anim_step(self):
        self.band_slider.setValue((self.current_band + 1) % self.bands)

    def _reset(self):
        """R key: back to band 0 and fit-to-window zoom in both views; spectra are kept."""
        self.band_slider.setValue(0)
        self._rgb_image.zoom_fit()
        self.canvas.zoom_fit()

    # ──────────────────────────────────────────────────────
    #  DISPLAY
    # ──────────────────────────────────────────────────────

    def update_band_image(self):
        """Push the current band, a 2-D ``(slit, scan)`` view of the cube, to the canvas."""
        img = self.cube[:, :, self.current_band]
        self.canvas.set_array(img)

    def _build_colormap_row(self) -> QHBoxLayout:
        """'Colormap' combo (None + matplotlib's page, by section) and a Reverse check box."""
        self.colormap_combo = QComboBox()
        self.colormap_combo.setToolTip("Colour the band image (after RAW / Enhance); None = grey")
        self.colormap_combo.setIconSize(QSize(32, 10))
        # size to ~10 characters, not to the longest section title; the list opens wide
        self.colormap_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.colormap_combo.setMinimumContentsLength(6)
        self.colormap_combo.view().setMinimumWidth(260)
        self.colormap_combo.addItem("None", None)
        model = self.colormap_combo.model()
        for title, names in COLORMAP_GROUPS:
            self.colormap_combo.addItem(f"\u2500\u2500 {title} \u2500\u2500", None)
            model.item(self.colormap_combo.count() - 1).setEnabled(False)
            for name in names:
                self.colormap_combo.addItem(_colormap_icon(name), name, name)
        self.colormap_reverse = QPushButton()           # icon toggle: fits the sidebar
        self.colormap_reverse.setIcon(qta.icon("mdi6.swap-horizontal", color=PLOT_FOREGROUND,
                                               color_on="#ffffff"))
        self.colormap_reverse.setCheckable(True)
        self.colormap_reverse.setFixedSize(26, 24)
        self.colormap_reverse.setToolTip("Reverse the colormap (<name>_r)")
        self.colormap_reverse.setAccessibleName("Reverse colormap")
        self.colormap_reverse.setEnabled(False)
        self.colormap_combo.currentIndexChanged.connect(self._on_colormap_changed)
        self.colormap_reverse.toggled.connect(self._on_colormap_changed)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        icon = QLabel()                    # a palette icon, not 140 px of text
        icon.setPixmap(qta.icon("mdi6.palette-swatch", color=PLOT_FOREGROUND).pixmap(16, 16))
        icon.setToolTip("Colormap of the band image")
        row.addWidget(icon)
        row.addWidget(self.colormap_combo, stretch=1)
        row.addWidget(self.colormap_reverse)
        return row

    def set_colormap(self, name=None, reverse: bool = False) -> None:
        """Select a band-view colormap by name (``None`` = grey), as the combo does.

        Raises:
            KeyError: ``name`` is not in the list.
        """
        index = 0 if name is None else self.colormap_combo.findData(name)
        if index < 0:
            raise KeyError(f"unknown colormap {name!r}")
        self.colormap_reverse.blockSignals(True)
        self.colormap_reverse.setChecked(bool(reverse))
        self.colormap_reverse.blockSignals(False)
        self.colormap_combo.setCurrentIndex(index)
        self._on_colormap_changed()

    def _on_colormap_changed(self, *_):
        name = self.colormap_combo.currentData()
        self.colormap_reverse.setEnabled(name is not None)
        self.canvas.set_colormap(name, self.colormap_reverse.isChecked())

    def _on_display_mapper_changed(self, mapper: DisplayMapper):
        self.canvas.set_display_mapper(mapper)
        self.update_band_image()

    # ──────────────────────────────────────────────────────
    #  EXPORT
    # ──────────────────────────────────────────────────────

    def open_export_dialog(self):
        """Show the "Export Datacube" dialog (the one already open is raised)."""
        self._open_export()

    def _open_export(self):
        if self._export_dlg is not None and self._export_dlg.isVisible():
            self._export_dlg.raise_()
            self._export_dlg.activateWindow()
            return

        self._export_dlg = ExportDialog(
            self, self.cube, self.wavelengths,
            self.config, self.camera_bin,
        )
        self._export_dlg.exported.connect(self.cube_exported)
        self._export_dlg.show()

    # ──────────────────────────────────────────────────────
    #  STATUS / INFO
    # ──────────────────────────────────────────────────────

    def _refresh_status(self):
        parts = [f"Cube {self.h}×{self.w}×{self.bands}"]
        if self._has_wavelengths():
            parts.append(
                f"λ={self.wavelengths[self.current_band]:.{self.resolution}f} nm"
            )
        if len(self.spectrum_set):
            pending = sum(1 for key in self._jobs.pending() if key >= 0)   # not median jobs
            parts.append(f"{len(self.spectrum_set)} spectra"
                         + (f" ({pending} computing…)" if pending else ""))
        if self._message:
            parts.append(self._message)
        self.status_label.setText("  ·  ".join(parts))

    def _show_message(self, text: str):
        """Show ``text`` in the status bar for a few seconds."""
        self._message = str(text)
        self._message_timer.start()
        self._refresh_status()

    def _clear_message(self):
        """Drop the status-bar message (timer)."""
        self._message = ""
        if not self._closed:
            self._refresh_status()

    def status_info(self) -> str:
        """Return a one-line summary for a host status bar.

        Returns:
            str: Current band (with wavelength in nm when known), number of
            spectra and the cube shape ``H×W×bands``.
        """
        parts = []
        if self._has_wavelengths():
            wl = self.wavelengths[self.current_band]
            parts.append(
                f"Band {self.current_band}/{self.bands - 1} "
                f"({wl:.{self.resolution}f} nm)"
            )
        else:
            parts.append(f"Band {self.current_band}/{self.bands - 1}")
        if len(self.spectrum_set):
            parts.append(f"{len(self.spectrum_set)} spectra")
        parts.append(f"Cube {self.h}×{self.w}×{self.bands}")
        return "  ·  ".join(parts)

    # ──────────────────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────────────────
    def _has_wavelengths(self) -> bool:
        """True when there is one wavelength per band."""
        return self.wavelengths is not None and len(self.wavelengths) == self.bands

    def _config_number(self, key: str):
        """``config[key]`` as a float when it is a finite number, else None."""
        value = self.config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None

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
        """Stop the jobs and timers, clear the spectra, close the export dialog, then release
        the cube and the other data references (Qt override).

        The worker thread is joined first (a running ``roi_stats`` stops within one ~1 MB
        chunk, a lasso mask rasterised in the worker within one ~10 ms batch of
        ``polygon_mask``), so no job still reads the cube when it is dropped.  Safe to call
        twice.
        """
        if not self._closed:
            self._closed = True
            self._anim_timer.stop()
            self._message_timer.stop()
            self._marker_timer.stop()
            self._jobs.shutdown()
            if self._segment_dlg is not None:
                self._segment_dlg.close()
                self._segment_dlg = None
            self.segment_session.shutdown()
            self.spectrum_set.clear()
            self._plot_entries.clear()
            self.spectra_panel.clear()
            for view in (self.canvas, self._rgb_image):
                view.cancel_drawing()
                view.set_roi_overlays(())
            if self._export_dlg:
                self._export_dlg.close()
                self._export_dlg = None
            self.cube = None
            self.wavelengths = None
            self.canvas = None
            self.spectrum = None
        super().closeEvent(event)
