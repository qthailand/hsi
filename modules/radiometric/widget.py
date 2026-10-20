"""Calibration and post-processing window (PyQt5) for offline ENVI cubes.

``CalibrationWindow`` is the interactive front end of ``modules.radiometric.core``:
load a raw cube, compute white-reference (row-wise) or fluorescence factors from an
ROI, optionally resample/bin, preview and export a calibrated ENVI BIP cube.
``RoiDialog`` is the modal ROI picker used for the reference scans.

Data conventions:
    * Raw cubes are ENVI files on disk; a band is a float32 image
      ``(lines, samples)`` (slit rows x scan columns, counts). In the preview the
      ``y`` axis is ``lines`` and the ``x`` axis ``samples``; the zoom limits of the
      preview axes define the crop box ``(x0, x1, y0, y1)`` of every export.
    * ``load_from_memory`` accepts a 3-D array ``(bands, H, W)`` or ``(H, W, bands)``
      (axis 0 is treated as bands when it is shorter than the last axis) and writes
      it as a float32 BIP ENVI file in the system temp directory.
    * Profiles are plain ``.npy`` files: row-wise factors ``(lines, bands)``,
      fluorescence factors ``(bands,)`` (no version header).
    * Wavelengths are in nm. Exported cubes are ENVI BIP ``(lines, samples, bands)``.
    * The lazy preview helpers build temporary header-less memmaps of shape
      ``(lines * bands, samples)`` in the temp directory.

Threading:
    All widget methods must be called on the GUI thread. Numerical work
    (factor computation, cube export, temp memmap build) runs on a ``QThread`` owned
    by the window through ``Worker``; its progress drives the progress bar.

Example:
    >>> from PyQt5.QtWidgets import QApplication
    >>> from modules.radiometric import CalibrationWindow
    >>> app = QApplication([])                                  # doctest: +SKIP
    >>> win = CalibrationWindow(); win.show()                   # doctest: +SKIP
    >>> win.load_from_memory(cube, wavelengths)                 # doctest: +SKIP
"""
from .workers import Worker
from .core import _apply_spatial_binning, rowwise_calculation_heavy_logic, create_datacube_logic, create_datacube_resampled_logic, build_raw_cube_tmp, build_cal_cube_tmp, fluo_calculation_heavy_logic, REFLECTANCE_PROFILE_DIR, list_reflectance_profiles, check_factor_bands, save_factor_profile, load_factor_profile
from modules.cube_io.envi import parse_envi_header, _read_envi_band_float32, release_band_maps
# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import savgol_filter   # used by RoiDialog.update_spectrum_plot
import os
import sys
import traceback
import math
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QWidget, QFileDialog, QApplication, QLabel, QFormLayout, QLineEdit,
    QPushButton, QCheckBox, QProgressBar, QMessageBox, QDialog, QVBoxLayout,
    QHBoxLayout, QRadioButton, QGroupBox, QSlider, QComboBox
)
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import EllipseSelector, RectangleSelector
import matplotlib.pyplot as plt


# =============================================================================
# Helper: Background Worker
# =============================================================================

import spectral



# =============================================================================
# CalibrationWindow — เปิดได้ทั้งจาก Scan และ Browse (ไม่มี Dark)
# =============================================================================


class CalibrationWindow(QWidget):
    """Window that calibrates a raw ENVI cube and exports a processed BIP cube.

    Workflow (left column top to bottom, right column is a matplotlib preview):

    1. **Input**: ``Browse Raw ENVI`` (:meth:`event_browse`), or the host calls
       :meth:`load_from_scan` / :meth:`load_from_memory`.
    2. **Calibration**: row-wise reflectance calibration (pick a white-reference
       file, a reflectance Excel table and an ROI, then run
       ``rowwise_calculation_heavy_logic`` on the worker thread), or fluorescence
       calibration (ROI average + smoothing), or load a previously saved ``.npy``
       profile. Row-wise and fluorescence factors are mutually exclusive.
    3. **Resample** (optional): uniform wavelength grid ``lam_min + k * FWHM``.
    4. **Spatial binning** 1x1 / 2x2 / 4x4.
    5. **Create Datacube**: crops to the current preview zoom, applies the factors and
       writes an ENVI BIP file on a worker thread.

    In the preview, drag a rectangle (or double-click a pixel) to plot its spectrum in a
    separate matplotlib figure; right-click removes the last marker. Spectra use lazily
    built temporary memmaps (``Parameters`` check box).

    Signals:
        calibration_done(str, dict): emitted after a successful export with the output
            file path and the raw header dict.

    Host integration: call ``load_from_memory(cube, wavelengths)`` or
    ``load_from_scan(path)``; connect ``calibration_done``. The constructor sets the
    application-wide Fusion style and a dark palette. All methods are GUI-thread only.
    The window has no ``closeEvent`` and does not stop its worker thread on close.
    """

    raw_filepath: str = ''
    raw_meta: dict = None

    cubname: str = ''
    cubwidth: int = 0
    cubheight: int = 0
    cubband: int = 0
    isWavelength: bool = False

    rowwise_factors: np.ndarray = None
    _factor_wavelengths: np.ndarray = None      # nm of the active factors' bands, or None (unknown)
    _reference_wavelengths: np.ndarray = None   # nm of the reference a calibration is running on
    spectralon_rows_span: tuple = None
    thread: QThread = None
    worker: Worker = None

    bsq_raw_path: str = None
    bsq_raw_mm: np.memmap = None
    bsq_raw_shape: tuple = None
    bsq_raw_dtype: str = None

    bsq_cal_path: str = None
    bsq_cal_mm: np.memmap = None
    bsq_cal_shape: tuple = None
    bsq_cal_dtype: str = None

    wavelengths: np.ndarray = None

    calibration_done = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        """Build the two-column UI (parameters, calibration, resample, binning, preview) and set the dark palette.
        """
        super().__init__(parent)

        QApplication.setStyle("Fusion")
        base_font = self.font()
        base_font.setPointSize(11)
        self.setFont(base_font)

        from PyQt5.QtGui import QPalette, QColor
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(41, 45, 62))
        pal.setColor(QPalette.WindowText, QColor(230, 230, 230))
        pal.setColor(QPalette.Base, QColor(30, 34, 50))
        pal.setColor(QPalette.AlternateBase, QColor(41, 45, 62))
        pal.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
        pal.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
        pal.setColor(QPalette.Text, QColor(230, 230, 230))
        pal.setColor(QPalette.Button, QColor(58, 63, 85))
        pal.setColor(QPalette.ButtonText, QColor(230, 230, 230))
        pal.setColor(QPalette.BrightText, QColor(255, 0, 0))
        pal.setColor(QPalette.Highlight, QColor(87, 96, 130))
        pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        QApplication.setPalette(pal)

        self.setWindowTitle("Calibration & Post-Processing")
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinMaxButtonsHint)
        self.resize(1600, 900)

        self._raw_temp_building = False
        self._cal_temp_building = False
        self._post_build_actions = []
        self._source_mode = None

        main_layout = QHBoxLayout(self)

        # ================== Left Column ==================
        left_layout = QVBoxLayout()

        # --- Source Info ---
        source_box = QGroupBox("Source")
        source_lay = QVBoxLayout()
        self.label_source = QLabel("No data loaded")
        self.label_source.setStyleSheet("font-weight: bold; color: #90caf9;")
        source_lay.addWidget(self.label_source)
        source_box.setLayout(source_lay)
        left_layout.addWidget(source_box)

        # --- Parameters ---
        params_box = QGroupBox("Parameters")
        params_layout = QFormLayout()
        self.label = QLabel('...')
        self.name = QLabel('...')
        self.imWidth = QLineEdit()
        self.imWidth.setReadOnly(True)
        self.imHeight = QLineEdit()
        self.imHeight.setReadOnly(True)
        self.imBand = QLineEdit()
        self.imBand.setReadOnly(True)
        self.nanometer = QCheckBox()
        self.nanometer.setChecked(True)

        self.label_preview = QLabel('Preview Index:')
        self.preview_idx_edit = QLineEdit("113")
        self.preview_idx_edit.setReadOnly(True)
        self.preview_slider = QSlider(Qt.Horizontal)
        self.preview_slider.setMinimumWidth(100)
        self.preview_slider.setEnabled(False)
        self.avg_window_edit = QLineEdit("20")

        self.preview_slider.valueChanged.connect(self.on_slider_value_changed)

        params_layout.addRow(QLabel('Raw Data File:'), self.label)
        params_layout.addRow(QLabel('Datacube Name:'), self.name)

        dim_layout = QHBoxLayout()
        dim_layout.setContentsMargins(0, 0, 0, 0)
        dim_layout.addWidget(QLabel('Width:'))
        dim_layout.addWidget(self.imWidth)
        dim_layout.addWidget(QLabel('Height:'))
        dim_layout.addWidget(self.imHeight)
        dim_layout.addWidget(QLabel('Band:'))
        dim_layout.addWidget(self.imBand)
        params_layout.addRow(dim_layout)

        params_layout.addRow(QLabel('Has Wavelength:'), self.nanometer)

        preview_layout = QHBoxLayout()
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(self.label_preview)
        preview_layout.addWidget(self.preview_slider)
        preview_layout.addWidget(self.preview_idx_edit)
        params_layout.addRow(preview_layout)

        params_layout.addRow(QLabel('Avg. Window:'), self.avg_window_edit)

        self.chk_lazy_temp = QCheckBox("สร้าง temp เมื่อจำเป็น (แนะนำ)")
        self.chk_lazy_temp.setChecked(True)
        params_layout.addRow(self.chk_lazy_temp)
        params_box.setLayout(params_layout)

        # --- Input Data ---
        input_box = QGroupBox("1. Input Data")
        input_v = QVBoxLayout()

        input_h1 = QHBoxLayout()
        self.btn = QPushButton('Browse Raw ENVI...')
        self.btn.clicked.connect(self.event_browse)
        self.btn.setMinimumHeight(34)
        input_h1.addWidget(self.btn)

        self.btn_clear_inputs = QPushButton('Clear All Inputs')
        self.btn_clear_inputs.setMinimumHeight(34)
        self.btn_clear_inputs.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f; color: white;
                border-radius: 6px; padding: 6px 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: #e57373; }
            QPushButton:pressed { background-color: #b71c1c; }
            QPushButton:disabled { background-color: #5f1414; color: #888888; }
        """)
        self.btn_clear_inputs.clicked.connect(self.event_clear_inputs)

        input_v.addLayout(input_h1)
        input_v.addWidget(self.btn_clear_inputs)
        input_box.setLayout(input_v)

        # --- Calibration ---
        calib_box = QGroupBox("2. Calibration Options")
        calib_v = QVBoxLayout()

        # reference reflectance table: files of reflectance_profiles/, or Browse... for another one
        calib_h_reference = QHBoxLayout()
        self.ref_combo = QComboBox()
        self.ref_combo.setToolTip(f"Reflectance tables in {REFLECTANCE_PROFILE_DIR}")
        self.btn_browse_reference = QPushButton('Browse...')
        self.btn_browse_reference.clicked.connect(self.event_browse_reference)
        calib_h_reference.addWidget(QLabel('Reference:'))
        calib_h_reference.addWidget(self.ref_combo, 1)
        calib_h_reference.addWidget(self.btn_browse_reference)
        self.populate_reference_profiles()

        self.btn_rowwise = QPushButton('Calibrate with ROI (Row-wise)')
        self.btn_rowwise.clicked.connect(self.event_rowwise_calibration)
        self.btn_rowwise.setMinimumHeight(34)

        calib_h_bottom = QHBoxLayout()
        self.btn_load_profile = QPushButton('Load Reflectance Profile...')
        self.btn_load_profile.clicked.connect(self.event_load_profile)
        self.btn_save_profile = QPushButton('Save Reflectance Profile')
        self.btn_save_profile.clicked.connect(self.event_save_profile)
        for b in (self.btn_load_profile, self.btn_save_profile):
            b.setMinimumHeight(34)
        calib_h_bottom.addWidget(self.btn_load_profile)
        calib_h_bottom.addWidget(self.btn_save_profile)

        self.label_profile = QLabel("No reflectance profile loaded.")

        calib_h_fluo = QHBoxLayout()
        self.btn_fluo_cal = QPushButton('Calibrate Fluorescence (ROI Avg + Smooth)')
        self.btn_fluo_cal.clicked.connect(self.event_fluorescence_calibration)
        self.btn_fluo_cal.setMinimumHeight(34)
        calib_h_fluo.addWidget(self.btn_fluo_cal)

        calib_h_fluo_profile = QHBoxLayout()
        self.btn_load_fluo_profile = QPushButton('Load Fluorescence Profile...')
        self.btn_load_fluo_profile.clicked.connect(self.event_load_fluo_profile)
        self.btn_save_fluo_profile = QPushButton('Save Fluorescence Profile')
        self.btn_save_fluo_profile.clicked.connect(self.event_save_fluo_profile)
        for b in (self.btn_load_fluo_profile, self.btn_save_fluo_profile):
            b.setMinimumHeight(34)
        calib_h_fluo_profile.addWidget(self.btn_load_fluo_profile)
        calib_h_fluo_profile.addWidget(self.btn_save_fluo_profile)

        self.label_fluo_profile = QLabel("No fluorescence profile loaded.")

        from PyQt5.QtWidgets import QFrame
        calib_v.addLayout(calib_h_reference)
        calib_v.addWidget(self.btn_rowwise)
        calib_v.addLayout(calib_h_bottom)
        calib_v.addWidget(self.label_profile)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        calib_v.addWidget(line)

        calib_v.addLayout(calib_h_fluo)
        calib_v.addLayout(calib_h_fluo_profile)
        calib_v.addWidget(self.label_fluo_profile)

        calib_box.setLayout(calib_v)

        # --- Resample ---
        resamp_box = QGroupBox("3. Resample to Uniform Grid (λ step = FWHM)")
        resamp_main_layout = QVBoxLayout()

        self.chk_resample = QCheckBox("Resample lambda")
        self.chk_resample.setChecked(True)
        self.chk_resample.toggled.connect(self._update_bin_info)
        resamp_main_layout.addWidget(self.chk_resample)

        resamp_layout = QHBoxLayout()

        self.lam_min_edit = QLineEdit()
        self.fwhm_edit = QLineEdit("1.50")
        self.lam_max_edit = QLineEdit()
        self.lam_max_edit.setReadOnly(True)

        def _on_lambda_or_fwhm_changed():
            try:
                lmin = float(self.lam_min_edit.text())
                fwhm = float(self.fwhm_edit.text())
                if not hasattr(self, "_wl_min_cap") or not hasattr(self, "_wl_max_cap"):
                    return
                lmin = max(lmin, self._wl_min_cap)
                lmin = self._round2_ceil(lmin)
                lmax = self._calc_lmax_from(lmin, fwhm, self._wl_max_cap)
                self.lam_min_edit.setText(f"{lmin:.2f}")
                self.lam_max_edit.setText(f"{lmax:.2f}")
            except Exception:
                pass

        self.lam_min_edit.editingFinished.connect(_on_lambda_or_fwhm_changed)
        self.fwhm_edit.editingFinished.connect(_on_lambda_or_fwhm_changed)

        resamp_layout.addWidget(QLabel("λ_min (nm):"))
        resamp_layout.addWidget(self.lam_min_edit)
        resamp_layout.addWidget(QLabel("FWHM (nm):"))
        resamp_layout.addWidget(self.fwhm_edit)
        resamp_layout.addWidget(QLabel("λ_max (nm):"))
        resamp_layout.addWidget(self.lam_max_edit)

        resamp_main_layout.addLayout(resamp_layout)
        resamp_box.setLayout(resamp_main_layout)

        def toggle_resample_inputs(state):
            self.lam_min_edit.setEnabled(state)
            self.fwhm_edit.setEnabled(state)
        self.chk_resample.toggled.connect(toggle_resample_inputs)

        # --- Spatial Binning ---
        bin_box = QGroupBox("4. Spatial Binning")
        bin_v_layout = QVBoxLayout()
        self.bin_combo = QComboBox()
        self.bin_combo.addItems(["None (1x1) - Original Size",
                                 "Binning 2x2 (Reduce size 4x)",
                                 "Binning 4x4 (Reduce size 16x)"])
        self.bin_combo.currentIndexChanged.connect(self._update_bin_info)
        self.label_bin_info = QLabel("Output Size: - x - px")

        bin_v_layout.addWidget(self.bin_combo)
        bin_v_layout.addWidget(self.label_bin_info)
        bin_box.setLayout(bin_v_layout)

        # --- Create Button ---
        self.btn_create_datacube = QPushButton('Create Datacube')
        self.btn_create_datacube.clicked.connect(self.event_create_dispatch)
        self.btn_create_datacube.setMinimumHeight(36)

        create_h = QHBoxLayout()
        create_h.addWidget(self.btn_create_datacube)

        self.progess = QProgressBar()

        left_layout.addWidget(params_box)
        left_layout.addWidget(input_box)
        left_layout.addWidget(calib_box)
        left_layout.addWidget(resamp_box)
        left_layout.addWidget(bin_box)
        left_layout.addLayout(create_h)
        left_layout.addWidget(self.progess)
        left_layout.addStretch()

        # ================== Right Column ==================
        right_layout = QVBoxLayout()
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        self.navbar = NavigationToolbar2QT(self.canvas, self)
        self.axes = self.figure.add_subplot(1, 1, 1)
        right_layout.addWidget(self.navbar)
        right_layout.addWidget(self.canvas)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)

        self._colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        self._color_idx = 0
        self.spec_fig = None
        self.spec_ax = None
        self.markers = []

    # =====================================================================
    #  เปิดจากภายนอก
    # =====================================================================

    def load_from_scan(self, filepath: str, meta: dict = None):
        """Load an ENVI file produced by a scan (``meta`` optional header dict) and lock the Browse button.
        """
        self._source_mode = 'scan_file'
        self._load_file_internal(filepath, meta=meta, source_label="Loaded from Scan")

    def load_from_memory(self, cube_array, wavelengths=None, name="ScanData"):
        """Load an in-memory cube by writing it as a temporary float32 BIP ENVI file.

        Args:
            cube_array: 3-D array ``(bands, H, W)`` or ``(H, W, bands)`` of counts; axis 0
                is taken as bands if it is shorter than the last axis. Converted to float32
                (a copy).
            wavelengths: Optional sequence of band centres in nm, one per band.
            name: Label used for the temp file ``<tmpdir>/scan_<name>.bip`` and the title.

        Raises:
            ValueError: ``cube_array`` is not 3-D.
        """
        self._source_mode = 'scan_memory'
        import tempfile
        tmp_path = os.path.join(tempfile.gettempdir(), f"scan_{name}.bip")

        arr = np.asarray(cube_array, dtype=np.float32)
        if arr.ndim == 3:
            if arr.shape[0] < arr.shape[2]:
                bands, H, W = arr.shape
                arr = arr.transpose(1, 2, 0)  # CHW → HWC
            else:
                H, W, bands = arr.shape
        else:
            raise ValueError("cube_array must be 3D")

        # ★ BIP layout: (H, W, bands) → flatten
        bip = arr.reshape(H, W * bands)
        release_band_maps(tmp_path)   # the previous scan's file may still be mapped
        bip.tofile(tmp_path)

        hdr_path = tmp_path + '.hdr'
        with open(hdr_path, 'w') as h:
            h.write('ENVI\n')
            h.write(f'description = {{Scan Memory: {name}}}\n')
            h.write(f'samples = {W}\n')
            h.write(f'lines = {H}\n')
            h.write(f'bands = {bands}\n')
            h.write('data type = 4\n')
            h.write('interleave = bip\n')    # ★ BIP
            h.write('byte order = 0\n')
            if wavelengths is not None:
                wl_str = ','.join(f"{float(v):.2f}" for v in wavelengths)
                h.write(f'wavelength = {{{wl_str}}}\n')
                h.write('wavelength units = Nanometers\n')

        self._load_file_internal(tmp_path, meta=None,
                                source_label=f"From Memory: {name}")

    def _load_file_internal(self, filepath, meta=None, source_label="Browse"):
        """Read the header, fill the info fields, wavelength limits and preview slider, and show the first preview (warns on error).
        """
        try:
            self.raw_filepath = filepath
            self.raw_meta = meta if meta else self.parse_envi_header(filepath)

            parent_name = os.path.basename(os.path.dirname(filepath))
            self.cubname = parent_name if parent_name else os.path.splitext(
                os.path.basename(filepath))[0]
            self.label.setText(os.path.basename(filepath))
            self.name.setText(self.cubname)

            self.cubheight = self.raw_meta['lines']
            self.cubwidth = self.raw_meta['samples']
            self.cubband = self.raw_meta['bands']

            self.imHeight.setText(str(self.cubheight))
            self.imWidth.setText(str(self.cubwidth))
            self.imBand.setText(str(self.cubband))

            self.label_source.setText(f"{source_label}: {os.path.basename(filepath)}")

            wl = self.raw_meta.get('wavelength', None)
            if wl is not None:
                self.wavelengths = wl
                self._wl_min_cap = self._round2_ceil(float(np.min(wl)))
                self._wl_max_cap = self._round2_floor(float(np.max(wl)))
                default_fwhm = float(self.fwhm_edit.text() or "1.50")
                lmin = self._wl_min_cap
                lmax = self._calc_lmax_from(lmin, default_fwhm, self._wl_max_cap)
                self.lam_min_edit.setText(f"{lmin:.2f}")
                self.lam_max_edit.setText(f"{lmax:.2f}")

                self.preview_slider.setMinimum(0)
                self.preview_slider.setMaximum(len(wl) - 1)
                safe_idx = min(max(0, 112), len(wl) - 1)

                self.label_preview.setText('Preview wavelength (nm):')
                self.preview_slider.blockSignals(True)
                self.preview_slider.setValue(safe_idx)
                self.preview_slider.blockSignals(False)
                self.preview_idx_edit.setText(f"{wl[safe_idx]:.2f}")
                self.preview_slider.setEnabled(True)
            else:
                self.wavelengths = None
                self.label_preview.setText('Preview Index:')
                self.preview_idx_edit.setText("113")
                self.preview_slider.setEnabled(False)

            self._drop_factors_not_fitting_cube()
            self.update_preview_image()
            self.mark_button_done(self.btn)
            self._clear_memmap_temp_states()
            self._update_bin_info()

            if self._source_mode in ('scan_file', 'scan_memory'):
                self.btn.setEnabled(False)

        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Failed to load file: {str(e)}')

    # =====================================================================
    #  Utilities
    # =====================================================================

    _apply_spatial_binning = staticmethod(_apply_spatial_binning)

    def _update_bin_info(self):
        """Refresh the "Output Size" label from the preview zoom, binning factor, resample grid and dtype.
        """
        if not self.raw_meta:
            self.label_bin_info.setText("Output Size: - x - px")
            return
        idx = self.bin_combo.currentIndex()
        n = 4 if idx == 2 else (2 if idx == 1 else 1)
        try:
            x0, x1 = np.ceil(self.axes.get_xlim()).astype('uint').tolist()
            y1, y0 = np.ceil(self.axes.get_ylim()).astype('uint').tolist()
            raw_w, raw_h = (x1 - x0), (y1 - y0)
        except Exception:
            raw_w, raw_h = self.raw_meta['samples'], self.raw_meta['lines']
        new_w, new_h = raw_w // n, raw_h // n

        if self.chk_resample.isChecked():
            try:
                lmin = float(self.lam_min_edit.text())
                fwhm = float(self.fwhm_edit.text())
                lmax = float(self.lam_max_edit.text())
                num_bands = int(math.floor((lmax - lmin + 1e-9) / fwhm)) + 1
            except Exception:
                num_bands = self.raw_meta['bands']
        else:
            num_bands = self.raw_meta['bands']

        is_cal = (self.rowwise_factors is not None) or (getattr(self, 'factors', None) is not None)
        bytes_per_pixel = 4 if is_cal or int(self.raw_meta.get('data type', 12)) == 4 else 2
        total_bytes = new_w * new_h * num_bands * bytes_per_pixel
        size_mb = total_bytes / (1024 * 1024)
        size_text = f"{size_mb:.1f} MB" if size_mb < 1000 else f"{size_mb / 1024:.2f} GB"
        self.label_bin_info.setText(
            f"Output Size: {new_w}x{new_h} px | Est. Size: ~{size_text}")

    parse_envi_header = staticmethod(parse_envi_header)
    _read_envi_band_float32 = staticmethod(_read_envi_band_float32)

    # =====================================================================
    #  UI Thread helpers
    # =====================================================================

    def _run_in_thread(self, func, *args, **kwargs):
        """Disable the UI and start ``func`` on a new ``QThread`` through ``Worker``; return ``(worker, thread)`` so the caller can connect ``finished``.
        """
        self.set_ui_enabled(False)
        self.thread = QThread()
        self.worker = Worker(func, *args, **kwargs)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.error.connect(self.on_worker_error)
        self.worker.progress.connect(self.progess.setValue)
        self.thread.start()
        return self.worker, self.thread

    def set_ui_enabled(self, enabled):
        """Enable or disable all action buttons and the preview slider during background work.
        """
        btns = [self.btn, self.btn_clear_inputs,
                self.btn_rowwise, self.btn_save_profile, self.btn_load_profile,
                self.ref_combo, self.btn_browse_reference,
                self.btn_create_datacube,
                getattr(self, 'btn_fluo_cal', None),
                getattr(self, 'btn_load_fluo_profile', None),
                getattr(self, 'btn_save_fluo_profile', None)]
        for btn in btns:
            if btn is not None:
                btn.setEnabled(enabled)
        self.preview_slider.setEnabled(enabled and (self.wavelengths is not None))

    def on_worker_error(self, error_msg):
        """Show the worker traceback, re-enable the UI and stop the thread.
        """
        QMessageBox.warning(self, 'Error', error_msg)
        self.set_ui_enabled(True)
        self.progess.setValue(0)
        if self.thread:
            self.thread.quit()
            self.thread.wait()

    def mark_button_done(self, btn: QPushButton):
        """Colour a button green to show that its step is complete.
        """
        btn.setStyleSheet(
            "QPushButton{background-color:#2e7d32;color:white;border-radius:6px;"
            "padding:6px 10px;} QPushButton:disabled{background-color:#1b5e20;color:#bbbbbb;}")

    def on_slider_value_changed(self, index):
        """Slider slot: show the selected wavelength and redraw the preview band.
        """
        if self.wavelengths is None or index >= len(self.wavelengths):
            return
        self.preview_idx_edit.setText(f"{self.wavelengths[index]:.2f}")
        self.update_preview_image()

    # =====================================================================
    #  Browse Events
    # =====================================================================

    def event_browse(self):
        """Ask for a raw ENVI file and load it (browse mode).
        """
        filepath, _ = QFileDialog.getOpenFileName(
            None, 'Select Raw ENVI Data', '',
            'ENVI Files (*.bsq *.bil *.bip *.envi *.raw);;All Files (*)')
        if not filepath:
            return
        self._source_mode = 'browse'
        self.btn.setEnabled(True)
        self._load_file_internal(filepath, source_label="Browse ENVI")

    def event_clear_inputs(self):
        """Reset all loaded data, profiles, preview, markers and button states.
        """
        self.raw_filepath = ''
        self.raw_meta = None
        self.cubname = ''
        self.cubwidth = 0
        self.cubheight = 0
        self.cubband = 0
        self.wavelengths = None
        self.rowwise_factors = None
        self.factors = None
        self._factor_wavelengths = None
        self._source_mode = None

        self.label.setText('...')
        self.name.setText('...')
        self.label_source.setText("No data loaded")
        self.imWidth.setText('')
        self.imHeight.setText('')
        self.imBand.setText('')
        self.lam_min_edit.setText('')
        self.lam_max_edit.setText('')
        self.label_profile.setText("No reflectance profile loaded.")
        self.label_fluo_profile.setText("No fluorescence profile loaded.")
        self.label_preview.setText('Preview Index:')
        self.preview_idx_edit.setText("113")
        self.preview_slider.blockSignals(True)
        self.preview_slider.setValue(0)
        self.preview_slider.blockSignals(False)
        self.preview_slider.setEnabled(False)

        self.btn.setStyleSheet("")
        self.btn.setEnabled(True)
        self.btn_rowwise.setStyleSheet("")
        self.btn_load_profile.setStyleSheet("")
        self.btn_save_profile.setStyleSheet("")
        self.btn_create_datacube.setStyleSheet("")
        if hasattr(self, 'btn_fluo_cal'):
            self.btn_fluo_cal.setStyleSheet("")
        if hasattr(self, 'btn_load_fluo_profile'):
            self.btn_load_fluo_profile.setStyleSheet("")
        if hasattr(self, 'btn_save_fluo_profile'):
            self.btn_save_fluo_profile.setStyleSheet("")

        self.btn_rowwise.setEnabled(True)
        self.btn_save_profile.setEnabled(True)
        if hasattr(self, 'btn_fluo_cal'):
            self.btn_fluo_cal.setEnabled(True)
        if hasattr(self, 'btn_save_fluo_profile'):
            self.btn_save_fluo_profile.setEnabled(True)

        self._clear_memmap_temp_states()
        self.axes.clear()
        self.axes.set_title("")
        self.canvas.draw_idle()
        if self.rect_selector_main is not None:
            try:
                self.rect_selector_main.set_active(False)
                self.rect_selector_main.disconnect_events()
            except Exception:
                pass
            self.rect_selector_main = None
        try:
            self.canvas.mpl_disconnect(self._click_cid)
        except Exception:
            pass
        self.markers.clear()
        if self.spec_ax is not None:
            self.spec_ax.clear()
            if self.spec_fig is not None:
                self.spec_fig.canvas.draw_idle()

    def _clear_memmap_temp_states(self):
        """Forget the temporary raw/calibrated memmaps and any pending build callbacks.
        """
        self.bsq_raw_path = self.bsq_raw_mm = self.bsq_raw_shape = self.bsq_raw_dtype = None
        self.bsq_cal_path = self.bsq_cal_mm = self.bsq_cal_shape = self.bsq_cal_dtype = None
        self._raw_temp_building = False
        self._cal_temp_building = False
        self._post_build_actions.clear()

    def _reference_refused(self, ref_meta) -> bool:
        """Warn and return True if factors from ``ref_meta``'s bands would not fit the loaded cube."""
        if not self.raw_meta:
            return False                        # checked when a cube is loaded
        try:
            check_factor_bands(ref_meta['bands'], ref_meta.get('wavelength'), self.raw_meta)
        except ValueError as e:
            QMessageBox.warning(self, 'Reference not used', str(e))
            return True
        return False

    def _drop_factors_not_fitting_cube(self):
        """Clear the active factors, with a warning, if they do not fit the cube just loaded."""
        factors = self.rowwise_factors if self.rowwise_factors is not None else getattr(self, 'factors', None)
        if factors is None:
            return
        try:
            check_factor_bands(np.shape(factors)[-1], self._factor_wavelengths, self.raw_meta)
            return
        except ValueError as e:
            message = str(e)
        self.rowwise_factors = None
        self.factors = None
        self._factor_wavelengths = None
        self.label_profile.setText("No reflectance profile loaded.")
        self.label_fluo_profile.setText("No fluorescence profile loaded.")
        for btn in (self.btn_rowwise, self.btn_save_profile, self.btn_fluo_cal, self.btn_save_fluo_profile):
            btn.setEnabled(True)
        for btn in (self.btn_rowwise, self.btn_load_profile, self.btn_save_profile,
                    self.btn_fluo_cal, self.btn_load_fluo_profile, self.btn_save_fluo_profile):
            btn.setStyleSheet("")
        QMessageBox.warning(self, 'Calibration cleared', f"{message}\n\nThe calibration factors were cleared.")

    # =====================================================================
    #  Calibration UI
    # =====================================================================

    def event_save_profile(self):
        """Save the row-wise factors ``(lines, bands)`` as a ``.npy`` file.
        """
        if self.rowwise_factors is None:
            QMessageBox.warning(self, 'Error', 'No row-wise factors to save.')
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Calibration Profile", "", "NumPy Array (*.npy)")
        if save_path:
            try:
                save_factor_profile(save_path, self.rowwise_factors, self._factor_wavelengths)
                QMessageBox.information(self, 'Success', f'Profile saved:\n{save_path}')
                self.mark_button_done(self.btn_save_profile)
            except Exception as e:
                QMessageBox.warning(self, 'Error', str(e))

    def event_load_profile(self):
        """Load row-wise factors from a ``.npy`` file (replaces fluorescence factors) and refresh the preview.
        """
        load_path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration Profile", "", "NumPy Array (*.npy)")
        if load_path:
            try:
                factors, wavelengths = load_factor_profile(load_path)
                if self.raw_meta:
                    check_factor_bands(np.shape(factors)[-1], wavelengths, self.raw_meta)
                self.rowwise_factors = factors
                self.factors = None
                self._factor_wavelengths = wavelengths
                self.label_profile.setText(f"Loaded: {os.path.basename(load_path)}")
                QMessageBox.information(self, 'Success', f'Profile loaded:\n{load_path}')
                self.mark_button_done(self.btn_load_profile)
                self.update_preview_image()
                self._clear_memmap_temp_states()
                self.btn_rowwise.setEnabled(False)
                self.btn_save_profile.setEnabled(False)
            except Exception as e:
                QMessageBox.warning(self, 'Error', str(e))

    def populate_reference_profiles(self, select=None):
        """Fill the Reference list with the tables of ``reflectance_profiles/``.

        Args:
            select: Path to select afterwards (added to the list if it is not in the folder);
                by default the previous selection is kept, else the first table is selected.
        """
        previous = select or self.ref_combo.currentData()
        self.ref_combo.clear()
        for path in list_reflectance_profiles():
            self.ref_combo.addItem(path.stem, str(path))
        if previous and self.ref_combo.findData(str(previous)) < 0:
            self.ref_combo.addItem(os.path.basename(str(previous)), str(previous))
        index = self.ref_combo.findData(str(previous)) if previous else -1
        self.ref_combo.setCurrentIndex(index if index >= 0 else 0)

    def event_browse_reference(self):
        """Pick a reflectance table outside ``reflectance_profiles/`` and select it."""
        ref_excel, _ = QFileDialog.getOpenFileName(
            self, 'Select Reference Reflectance Table', str(REFLECTANCE_PROFILE_DIR),
            'Excel Files (*.xlsx *.xls)')
        if ref_excel:
            self.populate_reference_profiles(select=ref_excel)

    def event_rowwise_calibration(self):
        # ★ ถ้ามาจาก scan (memory/file) — ใช้ cube ที่โหลดไว้เลย
        """Choose reference file (unless scan mode), Excel table and ROI, then compute row-wise factors on the worker thread.
        """
        if self._source_mode in ('scan_file', 'scan_memory'):
            ref_filepath = self.raw_filepath
            ref_meta = self.raw_meta
        else:
            # Browse mode — ต้องเลือก reference file
            ref_filepath, _ = QFileDialog.getOpenFileName(
                None, 'Select Reference Data (.envi)', '',
                'ENVI Files (*.bsq *.bil *.bip *.envi *.raw);;All Files (*)')
            if not ref_filepath:
                return
            try:
                ref_meta = self.parse_envi_header(ref_filepath)
            except Exception as e:
                QMessageBox.warning(self, 'Error', str(e))
                return

        # ★ ทุก path ผ่านตรงนี้ — reflectance table ที่เลือกใน Reference
        ref_excel = self.ref_combo.currentData()
        if not ref_excel or not os.path.isfile(ref_excel):
            QMessageBox.warning(self, 'Error', 'Select a reference reflectance table (Reference / Browse...).')
            return
        try:
            avg_window_size = int(self.avg_window_edit.text())
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Invalid averaging window size.')
            return

        if self._reference_refused(ref_meta):
            return
        mask = self.select_roi_on_main(ref_filepath, ref_meta, ref_excel)
        if mask is None:
            return

        self._reference_wavelengths = ref_meta.get('wavelength')
        self.progess.setMaximum(ref_meta['bands'])
        self.progess.setValue(0)
        worker, thread = self._run_in_thread(
            self.rowwise_calculation_heavy_logic,
            ref_filepath=ref_filepath, ref_meta=ref_meta,
            ref_excel=ref_excel,
            avg_window_size=avg_window_size, mask=mask
        )
        worker.finished.connect(self.on_rowwise_calibration_finished)

    def on_rowwise_calibration_finished(self, result):
        """Store the row-wise factors and ROI row span, refresh the preview and re-enable the UI.
        """
        if result:
            self.rowwise_factors = result['factors']
            self.spectralon_rows_span = result['span']
            self.factors = None
            self._factor_wavelengths = self._reference_wavelengths
            self.label_profile.setText("Row-wise factors computed.")
            QMessageBox.information(self, 'Row-wise Calibration',
                                    'Row-wise factors computed successfully.')
            self.mark_button_done(self.btn_rowwise)
            self.update_preview_image()

        self.set_ui_enabled(True)
        self.progess.setValue(0)
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self._clear_memmap_temp_states()

    # =====================================================================
    #  Create Datacube
    # =====================================================================

    def event_create_dispatch(self):
        """Create Datacube button: export resampled or original-band cube depending on the Resample check box.
        """
        if self.chk_resample.isChecked():
            self.event_create_resampled()
        else:
            self.event_create()

    def event_create(self):
        """Ask for a ``.bip`` path and export the cropped, calibrated, binned cube with the original bands.
        """
        if not self.raw_filepath:
            QMessageBox.warning(self, "Error", "Please select a raw ENVI file first.")
            return
        x0, x1 = np.ceil(self.axes.get_xlim()).astype('uint').tolist()
        y1, y0 = np.ceil(self.axes.get_ylim()).astype('uint').tolist()
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Datacube As",
            os.path.join(os.path.dirname(self.raw_filepath),
                         f"{self.cubname}_calibrated.bip"),
            "ENVI BIP Files (*.bip)")
        if not save_path:
            return
        if not save_path.lower().endswith('.bip'):
            save_path += '.bip'

        self.progess.setMaximum(self.raw_meta['bands'])
        self.progess.setValue(0)

        idx = self.bin_combo.currentIndex()
        n_bin = 4 if idx == 2 else (2 if idx == 1 else 1)

        params = {
            "raw_filepath": self.raw_filepath, "raw_meta": self.raw_meta,
            "save_path": save_path,
            "crop_coords": (x0, x1, y0, y1),
            "is_wavelength": self.nanometer.isChecked(),
            "band_count": self.raw_meta['bands'],
            "factors": getattr(self, 'factors', None),
            "rowwise_factors": self.rowwise_factors,
            "factor_wavelengths": self._factor_wavelengths,
            "binning_factor": n_bin,
        }
        worker, thread = self._run_in_thread(self.create_datacube_logic, params)
        worker.finished.connect(self.on_create_finished)

    def on_create_finished(self, result_path):
        """Report the saved file, emit ``calibration_done(path, raw_meta)`` and re-enable the UI.
        """
        if result_path:
            QMessageBox.information(self, 'Success', f'Datacube saved:\n{result_path}')
            self.mark_button_done(self.btn_create_datacube)
            self.calibration_done.emit(result_path, self.raw_meta)
        self.set_ui_enabled(True)
        self.progess.setValue(0)
        if self.thread:
            self.thread.quit()
            self.thread.wait()

    def select_roi_on_main(self, ref_filepath, ref_meta, ref_excel):
        """Open :class:`RoiDialog` for a reflectance reference; return the boolean mask or ``None`` if cancelled.
        """
        wl = ref_meta.get('wavelength', None)
        dlg = RoiDialog(parent=self, ref_filepath=ref_filepath,
                        ref_meta=ref_meta, wl=wl)
        if dlg.exec_() != QDialog.Accepted:
            return None
        return dlg.get_mask()

    # =====================================================================
    #  Heavy Logic (static — no dark anywhere)
    # =====================================================================

    rowwise_calculation_heavy_logic = staticmethod(rowwise_calculation_heavy_logic)

    create_datacube_logic = staticmethod(create_datacube_logic)

    # =====================================================================
    #  Resampled Datacube
    # =====================================================================

    def event_create_resampled(self):
        """Ask for a ``.bip`` path, build the target wavelength grid from lambda min/FWHM/max and export the resampled cube.
        """
        if not self.raw_filepath:
            QMessageBox.warning(self, "Error", "Please select a raw ENVI file first.")
            return
        x0, x1 = np.ceil(self.axes.get_xlim()).astype('uint').tolist()
        y1, y0 = np.ceil(self.axes.get_ylim()).astype('uint').tolist()

        default_path = os.path.join(os.path.dirname(self.raw_filepath),
                                    f"{self.cubname}_calibrated.bip")
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Resampled Datacube (BIP)", default_path,
            "ENVI BIP Files (*.bip)")
        if not save_path:
            return
        if not save_path.lower().endswith('.bip'):
            save_path += '.bip'

        try:
            lmin = float(self.lam_min_edit.text())
            fwhm = float(self.fwhm_edit.text())
            lmax = float(self.lam_max_edit.text())
            lmin = max(lmin, self._wl_min_cap)
            lmax = min(lmax, self._wl_max_cap)
            n = int(math.floor((lmax - lmin + 1e-9) / fwhm)) + 1
            lam_target = np.round(lmin + np.arange(n, dtype=np.float32) * fwhm, 2)
            if lam_target.size <= 0:
                QMessageBox.warning(self, "Error", "Invalid λ grid.")
                return
            self.progess.setMaximum(int(lam_target.size))
            self.progess.setValue(0)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Cannot prepare wavelength grid: {e}")
            return

        idx = self.bin_combo.currentIndex()
        n_bin = 4 if idx == 2 else (2 if idx == 1 else 1)

        params = {
            "raw_filepath": self.raw_filepath, "raw_meta": self.raw_meta,
            "save_path": save_path,
            "crop_coords": (x0, x1, y0, y1),
            "is_wavelength": self.nanometer.isChecked(),
            "factors": getattr(self, 'factors', None),
            "rowwise_factors": getattr(self, 'rowwise_factors', None),
            "factor_wavelengths": self._factor_wavelengths,
            "lam_target": lam_target.tolist(),
            "binning_factor": n_bin
        }
        worker, thread = self._run_in_thread(
            self.create_datacube_resampled_logic, params)
        worker.finished.connect(self.on_create_finished)

    create_datacube_resampled_logic = staticmethod(create_datacube_resampled_logic)

    # =====================================================================
    #  Helper Utils
    # =====================================================================

    def _round2_floor(self, x: float) -> float:
        """Round down to 2 decimals (nm).
        """
        return math.floor(x * 100.0 + 1e-9) / 100.0

    def _round2_ceil(self, x: float) -> float:
        """Round up to 2 decimals (nm).
        """
        return math.ceil(x * 100.0 - 1e-12) / 100.0

    def _calc_lmax_from(self, lmin: float, fwhm: float, max_cap: float) -> float:
        """Largest ``lmin + n * fwhm`` not above ``max_cap`` (rounded to 2 decimals).
        """
        if fwhm <= 0:
            return self._round2_floor(max_cap)
        n = math.floor((max_cap - lmin + 1e-9) / fwhm)
        return self._round2_floor(lmin + n * fwhm)

    def _load_wavelengths_current(self):
        """Return the wavelength axis in nm, or ``0..bands-1`` if the cube has none.
        """
        if self.wavelengths is not None:
            return self.wavelengths
        return np.arange(int(self.imBand.text() or "1"))

    # =====================================================================
    #  Preview
    # =====================================================================

    current_preview_band: int = None
    click_circle_artist = None
    rect_selector_main = None

    def update_preview_image(self):
        """Redraw the preview of the selected band, with the active factors applied, scaled 1st-99th percentile.
        """
        if not self.raw_filepath or not self.raw_meta:
            return
        try:
            idx = max(0, min(self.preview_slider.value(), self.raw_meta['bands'] - 1))
            self.current_preview_band = idx

            im = self._read_envi_band_float32(self.raw_filepath, self.raw_meta, idx)
            if getattr(self, 'rowwise_factors', None) is not None:
                fac = self.rowwise_factors[:im.shape[0], idx].reshape(-1, 1)
                im = im * fac
            elif getattr(self, 'factors', None) is not None:
                im = im * float(self.factors[idx])

            m, M = np.nanpercentile(im, 1), np.nanpercentile(im, 99)
            show = np.zeros_like(im) if M <= m else np.clip((im - m) / (M - m), 0, 1)

            self.axes.clear()
            self.axes.imshow(show, interpolation='none', cmap='gray')
            self.axes.set_title(f'Preview (Band #{idx + 1})')
            self.canvas.draw_idle()
            self.attach_interaction_handlers()
        except Exception as e:
            print("update_preview_image error:", e)

    def attach_interaction_handlers(self):
        """Connect canvas click events and (re)create the rectangle selector on the preview axes.
        """
        try:
            self.canvas.mpl_disconnect(self._click_cid)
        except Exception:
            pass
        self._click_cid = self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        if self.rect_selector_main is not None:
            try:
                self.rect_selector_main.set_active(False)
                self.rect_selector_main.disconnect_events()
            except Exception:
                pass
            self.rect_selector_main = None

        try:
            self.rect_selector_main = RectangleSelector(
                self.axes, self.on_rect_release_spectrum,
                useblit=True, interactive=True, drag_from_anywhere=True,
                props=dict(edgecolor='cyan', linewidth=1.5, fill=False))
        except TypeError:
            self.rect_selector_main = RectangleSelector(
                self.axes, self.on_rect_release_spectrum,
                useblit=True, interactive=True, drag_from_anywhere=True)

    def on_canvas_click(self, event):
        """Canvas slot: double-click plots a pixel spectrum, right-click removes the last marker.
        """
        if event.inaxes != self.axes:
            return
        if event.button == 3:
            self.remove_last_marker()
            return
        if getattr(event, "dblclick", False) and event.button == 1:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            color = self._next_color()
            circ = Circle((x, y), radius=6, fill=False, edgecolor=color, lw=2)
            self.axes.add_patch(circ)
            self.canvas.draw_idle()
            self._plot_pixel_spectrum(x, y, color, f'Pt ({x},{y})', None, circ)

    def on_rect_release_spectrum(self, eclick, erelease):
        """Rectangle-selector slot: mark the box and plot its mean +/- std spectrum.
        """
        if any(v is None for v in [eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata]):
            return
        x1, y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
        x2, y2 = int(round(erelease.xdata)), int(round(erelease.ydata))
        x1, x2 = sorted((max(0, x1), max(0, x2)))
        y1, y2 = sorted((max(0, y1), max(0, y2)))
        if x2 - x1 < 1 or y2 - y1 < 1:
            return
        color = self._next_color()
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, lw=2)
        self.axes.add_patch(rect)
        self.canvas.draw_idle()
        self._plot_roi_spectrum(x1, y1, x2, y2, color,
                                f'ROI x[{x1},{x2}) y[{y1},{y2})', None, rect)

    def _read_band_calibrated(self, band_idx):
        """Read one band as float32 with the active factors applied.
        """
        im = self._read_envi_band_float32(self.raw_filepath, self.raw_meta, band_idx)
        if getattr(self, 'rowwise_factors', None) is not None:
            fac = self.rowwise_factors[:im.shape[0], band_idx].reshape(-1, 1)
            im = im * fac
        elif getattr(self, 'factors', None) is not None:
            im = im * float(self.factors[band_idx])
        return im

    # =====================================================================
    #  Spectrum Plotting
    # =====================================================================

    def _next_color(self):
        """Return the next colour of the matplotlib cycle for a new marker.
        """
        c = self._colors[self._color_idx % len(self._colors)]
        self._color_idx += 1
        return c

    def _ensure_spec_ax(self, ylabel='Intensity'):
        """Create (or reuse) the separate spectrum figure and set its y label.
        """
        if (self.spec_fig is None) or (not plt.fignum_exists(self.spec_fig.number)):
            self.spec_fig = plt.figure()
            self.spec_ax = self.spec_fig.add_subplot(111)
            self.spec_ax.set_xlabel('Wavelength (nm)')
            self.spec_ax.grid(True, alpha=0.35)
            self.spec_fig.canvas.mpl_connect('close_event', self._on_spec_fig_closed)
        elif self.spec_ax is None:
            self.spec_ax = self.spec_fig.add_subplot(111)
        self.spec_ax.set_ylabel(ylabel)
        return self.spec_ax

    def _on_spec_fig_closed(self, event):
        """Forget the spectrum figure when the user closes it.
        """
        self.spec_fig = None
        self.spec_ax = None

    def _add_marker(self, patch, line, fill):
        """Record a preview patch and its spectrum artists so they can be removed together.
        """
        self.markers.append({'patch': patch, 'line': line, 'fill': fill})

    def remove_last_marker(self):
        """Remove the newest marker from the preview and the spectrum plot.
        """
        if not self.markers:
            return
        last = self.markers.pop()
        if self._color_idx > 0:
            self._color_idx -= 1
        try:
            if last['patch'] is not None:
                last['patch'].remove()
        except Exception:
            pass
        if self.spec_ax is not None:
            try:
                if last['line'] is not None:
                    last['line'].remove()
            except Exception:
                pass
            try:
                if last['fill'] is not None:
                    last['fill'].remove()
            except Exception:
                pass
            try:
                self.spec_ax.legend().remove()
            except Exception:
                pass
            if self.spec_ax is not None:
                handles, labels = self.spec_ax.get_legend_handles_labels()
                if handles:
                    self.spec_ax.legend()
                if self.spec_fig is not None:
                    self.spec_fig.canvas.draw_idle()
        self.canvas.draw_idle()

    def _plot_pixel_spectrum(self, x, y, color, label, ylabel=None, patch=None):
        """Plot the spectrum of pixel ``(x, y)``, building the lazy temp memmap first if needed (then retries).
        """
        def _retry():
            self._plot_pixel_spectrum(x, y, color, label, ylabel, patch)

        if self.chk_lazy_temp.isChecked():
            try:
                if (self.rowwise_factors is not None) or (getattr(self, 'factors', None) is not None):
                    if not self.ensure_cal_memmap(on_ready=_retry):
                        return
                else:
                    if not self.ensure_raw_memmap(on_ready=_retry):
                        return
            except Exception as e:
                print("lazy memmap (pixel) error:", e)

        wl = self._load_wavelengths_current()

        if self.bsq_cal_mm is not None:
            L, S, B = self.bsq_cal_shape
            spec = np.array([self.bsq_cal_mm[y + i * L, x]
                             if (0 <= x < S and 0 <= y < L) else np.nan
                             for i in range(B)], dtype=np.float32)
            used_label = 'Reflectance (a.u.)'
        elif self.bsq_raw_mm is not None:
            L, S, B = self.bsq_raw_shape
            spec = np.array([self.bsq_raw_mm[y + i * L, x]
                             if (0 <= x < S and 0 <= y < L) else np.nan
                             for i in range(B)], dtype=np.float32)
            used_label = 'Intensity'
        else:
            B = self.raw_meta['bands']
            spec = np.zeros(B, dtype=np.float32)
            for i in range(B):
                im = self._read_band_calibrated(i)
                spec[i] = im[y, x] if (0 <= y < im.shape[0] and 0 <= x < im.shape[1]) else np.nan
            used_label = ('Reflectance (a.u.)'
                          if (self.rowwise_factors is not None or getattr(self, 'factors', None) is not None)
                          else 'Intensity')

        ax = self._ensure_spec_ax(ylabel or used_label)
        line, = ax.plot(wl, spec, label=label, color=color)
        ax.legend()
        ax.relim()
        ax.autoscale_view()
        self.spec_fig.canvas.draw_idle()
        self._add_marker(patch=patch, line=line, fill=None)

    def _plot_roi_spectrum(self, x1, y1, x2, y2, color, label, ylabel=None, patch=None):
        """Plot mean +/- std over the box ``[x1, x2) x [y1, y2)`` for all bands, building the lazy temp memmap first if needed.
        """
        def _retry():
            self._plot_roi_spectrum(x1, y1, x2, y2, color, label, ylabel, patch)

        if self.chk_lazy_temp.isChecked():
            try:
                if (self.rowwise_factors is not None) or (getattr(self, 'factors', None) is not None):
                    if not self.ensure_cal_memmap(on_ready=_retry):
                        return
                else:
                    if not self.ensure_raw_memmap(on_ready=_retry):
                        return
            except Exception as e:
                print("lazy memmap (roi) error:", e)

        wl = self._load_wavelengths_current()

        def _mean_std_from_mm(mm, shape):
            L, S, B = shape
            xi1, xi2 = max(0, x1), min(S, x2)
            yi1, yi2 = max(0, y1), min(L, y2)
            if xi2 - xi1 < 1 or yi2 - yi1 < 1:
                return None, None
            mm3 = np.asarray(mm).reshape((B, L, S))
            roi = mm3[:, yi1:yi2, xi1:xi2]
            return roi.mean(axis=(1, 2), dtype=np.float64), roi.std(axis=(1, 2), dtype=np.float64)

        used_label = 'Intensity'
        m = s = None

        if self.bsq_cal_mm is not None:
            m, s = _mean_std_from_mm(self.bsq_cal_mm, self.bsq_cal_shape)
            used_label = 'Reflectance (a.u.)'
        elif self.bsq_raw_mm is not None:
            m, s = _mean_std_from_mm(self.bsq_raw_mm, self.bsq_raw_shape)
            used_label = 'Intensity'

        if m is None or s is None:
            B = self.raw_meta['bands']
            m = np.empty(B, dtype=np.float32)
            s = np.empty(B, dtype=np.float32)
            for i in range(B):
                im = self._read_band_calibrated(i)
                roi = im[y1:min(y2, im.shape[0]), x1:min(x2, im.shape[1])]
                m[i] = np.nanmean(roi) if roi.size else np.nan
                s[i] = np.nanstd(roi) if roi.size else np.nan
            if self.rowwise_factors is not None or getattr(self, 'factors', None) is not None:
                used_label = 'Reflectance (a.u.)'

        ax = self._ensure_spec_ax(ylabel or used_label)
        line, = ax.plot(wl, m, label=label, color=color)
        fill = ax.fill_between(wl, m - s, m + s, alpha=0.25, label=None, color=color)
        ax.legend()
        ax.relim()
        ax.autoscale_view()
        self.spec_fig.canvas.draw_idle()
        self._add_marker(patch=patch, line=line, fill=fill)

    # =====================================================================
    #  Lazy Temp cubes
    # =====================================================================

    build_raw_cube_tmp = staticmethod(build_raw_cube_tmp)

    build_cal_cube_tmp = staticmethod(build_cal_cube_tmp)

    def ensure_raw_memmap(self, on_ready=None):
        """Make sure the cropped uint16 temp memmap exists.

        Returns:
            bool: True if it is ready now; False if a build was started or is running or
            there is no data (``on_ready`` is called when the build finishes).
        """
        if self.bsq_raw_mm is not None:
            return True
        if self._raw_temp_building:
            if callable(on_ready):
                self._post_build_actions.append(on_ready)
            return False
        if not self.raw_filepath:
            return False

        x0, x1 = np.ceil(self.axes.get_xlim()).astype('uint').tolist()
        y1, y0 = np.ceil(self.axes.get_ylim()).astype('uint').tolist()
        import tempfile
        tmp_raw = os.path.join(tempfile.gettempdir(),
                               f"rawcube_{os.path.basename(self.raw_filepath)}.bsq")
        self._raw_temp_building = True
        self.set_ui_enabled(False)
        self.progess.setMaximum(self.raw_meta['bands'])
        self.progess.setValue(0)

        worker, thread = self._run_in_thread(
            self.build_raw_cube_tmp,
            params={'folder': self.raw_filepath, 'raw_meta': self.raw_meta,
                    'save_path': tmp_raw, 'crop': (x0, x1, y0, y1)})

        def _raw_ready(meta):
            try:
                self.bsq_raw_path = meta['path']
                self.bsq_raw_mm = np.memmap(
                    self.bsq_raw_path, mode='r', dtype=np.uint16,
                    shape=(meta['lines'] * meta['bands'], meta['samples']))
                self.bsq_raw_shape = (meta['lines'], meta['samples'], meta['bands'])
                self.bsq_raw_dtype = meta['dtype']
            finally:
                self._raw_temp_building = False
                self.set_ui_enabled(True)
                self.progess.setValue(0)
                if self.thread:
                    self.thread.quit()
                    self.thread.wait()
                if callable(on_ready):
                    try:
                        on_ready()
                    except Exception as _e:
                        print("on_ready raw error:", _e)
                while self._post_build_actions:
                    try:
                        self._post_build_actions.pop(0)()
                    except Exception as _e:
                        print("post action raw error:", _e)

        worker.finished.connect(_raw_ready)
        return False

    def ensure_cal_memmap(self, on_ready=None):
        """Make sure the calibrated float32 temp memmap exists (needs factors).

        Returns:
            bool: True if ready now; False if a build was started/running or no factors or
            data exist (``on_ready`` is called after the build).
        """
        if self.bsq_cal_mm is not None:
            return True
        if (self.rowwise_factors is None) and (getattr(self, 'factors', None) is None):
            return False
        if self._cal_temp_building:
            if callable(on_ready):
                self._post_build_actions.append(on_ready)
            return False
        if not self.raw_filepath:
            return False

        x0, x1 = np.ceil(self.axes.get_xlim()).astype('uint').tolist()
        y1, y0 = np.ceil(self.axes.get_ylim()).astype('uint').tolist()
        import tempfile
        tmp_cal = os.path.join(tempfile.gettempdir(),
                               f"calcube_{os.path.basename(self.raw_filepath)}.bsq")
        self._cal_temp_building = True
        self.set_ui_enabled(False)
        self.progess.setMaximum(self.raw_meta['bands'])
        self.progess.setValue(0)

        worker, thread = self._run_in_thread(
            self.build_cal_cube_tmp,
            params={'raw_filepath': self.raw_filepath, 'raw_meta': self.raw_meta,
                    'save_path': tmp_cal, 'crop': (x0, x1, y0, y1),
                    'factors': getattr(self, 'factors', None),
                    'rowwise_factors': getattr(self, 'rowwise_factors', None),
                    'factor_wavelengths': self._factor_wavelengths})

        def _cal_ready(meta):
            try:
                self.bsq_cal_path = meta['path']
                self.bsq_cal_mm = np.memmap(
                    self.bsq_cal_path, mode='r', dtype=np.float32,
                    shape=(meta['lines'] * meta['bands'], meta['samples']))
                self.bsq_cal_shape = (meta['lines'], meta['samples'], meta['bands'])
                self.bsq_cal_dtype = meta['dtype']
            finally:
                self._cal_temp_building = False
                self.set_ui_enabled(True)
                self.progess.setValue(0)
                if self.thread:
                    self.thread.quit()
                    self.thread.wait()
                if callable(on_ready):
                    try:
                        on_ready()
                    except Exception as _e:
                        print("on_ready cal error:", _e)
                while self._post_build_actions:
                    try:
                        self._post_build_actions.pop(0)()
                    except Exception as _e:
                        print("post action cal error:", _e)

        worker.finished.connect(_cal_ready)
        return False

    # =====================================================================
    #  Fluo Calibration
    # =====================================================================

    def event_fluorescence_calibration(self):
        """Choose a reference file and ROI, then compute fluorescence factors on the worker thread.
        """
        ref_filepath, _ = QFileDialog.getOpenFileName(
            None, 'Select Fluorescence Reference (.envi)', '',
            'ENVI Files (*.bsq *.bil *.bip *.envi *.raw);;All Files (*)')
        if not ref_filepath:
            return
        try:
            ref_meta = self.parse_envi_header(ref_filepath)
        except Exception as e:
            QMessageBox.warning(self, 'Error', str(e))
            return

        if self._reference_refused(ref_meta):
            return
        mask = self.select_roi_fluo_main(ref_filepath, ref_meta)
        if mask is None:
            return

        self._reference_wavelengths = ref_meta.get('wavelength')
        self.progess.setMaximum(ref_meta['bands'])
        self.progess.setValue(0)
        worker, thread = self._run_in_thread(
            self.fluo_calculation_heavy_logic,
            ref_filepath=ref_filepath, ref_meta=ref_meta, mask=mask)
        worker.finished.connect(self.on_fluo_calibration_finished)

    def select_roi_fluo_main(self, ref_filepath, ref_meta):
        """Open :class:`RoiDialog` with a live spectrum plot; return the mask or ``None`` if cancelled.
        """
        wl = ref_meta.get('wavelength', None)
        dlg = RoiDialog(parent=self, ref_filepath=ref_filepath,
                        ref_meta=ref_meta, wl=wl, show_spectrum=True)
        if dlg.exec_() != QDialog.Accepted:
            return None
        return dlg.get_mask()

    fluo_calculation_heavy_logic = staticmethod(fluo_calculation_heavy_logic)

    def on_fluo_calibration_finished(self, result):
        """Store the fluorescence factors (clearing row-wise ones), refresh the preview and re-enable the UI.
        """
        if result:
            self.factors = result['factors']
            self.rowwise_factors = None
            self._factor_wavelengths = self._reference_wavelengths
            self.label_fluo_profile.setText("Fluorescence factors (Smoothed) computed.")
            self.label_profile.setText("No reflectance profile loaded.")
            QMessageBox.information(self, 'Fluorescence Calibration',
                                    'Fluorescence factors computed successfully.')
            if hasattr(self, 'btn_fluo_cal'):
                self.mark_button_done(self.btn_fluo_cal)
            self.update_preview_image()

        self.set_ui_enabled(True)
        self.progess.setValue(0)
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self._clear_memmap_temp_states()

    def event_save_fluo_profile(self):
        """Save the fluorescence factors ``(bands,)`` as a ``.npy`` file.
        """
        if self.factors is None:
            QMessageBox.warning(self, 'Error', 'No Fluorescence factors to save.')
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Fluorescence Profile", "", "NumPy Array (*.npy)")
        if save_path:
            try:
                save_factor_profile(save_path, self.factors, self._factor_wavelengths)
                QMessageBox.information(self, 'Success', f'Profile saved:\n{save_path}')
                self.mark_button_done(self.btn_save_fluo_profile)
            except Exception as e:
                QMessageBox.warning(self, 'Error', str(e))

    def event_load_fluo_profile(self):
        """Load fluorescence factors from a ``.npy`` file (replaces row-wise factors) and refresh the preview.
        """
        load_path, _ = QFileDialog.getOpenFileName(
            self, "Load Fluorescence Profile", "", "NumPy Array (*.npy)")
        if load_path:
            try:
                factors, wavelengths = load_factor_profile(load_path)
                if self.raw_meta:
                    check_factor_bands(np.shape(factors)[-1], wavelengths, self.raw_meta)
                self.factors = factors
                self.rowwise_factors = None
                self._factor_wavelengths = wavelengths
                self.label_fluo_profile.setText(f"Loaded: {os.path.basename(load_path)}")
                self.label_profile.setText("No reflectance profile loaded.")
                QMessageBox.information(self, 'Success', f'Profile loaded:\n{load_path}')
                self.mark_button_done(self.btn_load_fluo_profile)
                self.update_preview_image()
                self._clear_memmap_temp_states()
                self.btn_fluo_cal.setEnabled(False)
                self.btn_save_fluo_profile.setEnabled(False)
            except Exception as e:
                QMessageBox.warning(self, 'Error', str(e))


# =============================================================================
# RoiDialog (ไม่มี Dark ใดๆ)
# =============================================================================


class RoiDialog(QDialog):
    """Modal dialog to draw a circular or rectangular ROI on a reference scan.

    Shows an RGB composite (nearest bands to 650/550/450 nm, when the cube spans
    480-630 nm) or a single band chosen by a slider. The mask is read with
    :meth:`get_mask` after ``exec_()`` returns ``Accepted``. GUI thread only.

    Args:
        parent: Parent widget.
        ref_filepath: ENVI data file of the reference scan.
        ref_meta: Its header dict (``lines``, ``samples``, ``bands``).
        wl: Wavelengths in nm (default ``0..bands-1``).
        show_spectrum: Also plot the live raw/smoothed ROI-mean spectrum.
    """
    def __init__(self, parent=None, ref_filepath=None, ref_meta=None,
                 wl=None, show_spectrum=False):
        """Build the dialog and show the first image (arguments as in the class docstring).
        """
        super().__init__(parent)
        self.setWindowTitle('Select Region of Interest (ROI)')

        self.ref_filepath = ref_filepath
        self.ref_meta = ref_meta
        self.wl = wl if wl is not None else np.arange(ref_meta['bands'])
        self.show_spectrum = show_spectrum

        self.H, self.W = ref_meta['lines'], ref_meta['samples']
        self.mask = np.zeros((self.H, self.W), dtype=bool)
        self.selector = None
        self._cached_rgb = None

        self.img_artist = None
        self.roi_patch = None

        main_layout = QHBoxLayout(self)
        left_widget = QWidget()
        lay = QVBoxLayout(left_widget)

        mode_layout = QHBoxLayout()
        self.rb_rgb = QRadioButton("RGB Color")
        self.rb_single = QRadioButton("Single Band")
        mode_layout.addWidget(self.rb_rgb)
        mode_layout.addWidget(self.rb_single)
        lay.addLayout(mode_layout)

        self.slider_group = QWidget()
        slider_lay = QHBoxLayout(self.slider_group)
        self.band_slider = QSlider(Qt.Horizontal)
        self.band_slider.setRange(0, ref_meta['bands'] - 1)
        self.band_label = QLabel("Band: 0")
        slider_lay.addWidget(self.band_label)
        slider_lay.addWidget(self.band_slider)
        lay.addWidget(self.slider_group)

        self.fig = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        tool = NavigationToolbar2QT(self.canvas, self)
        lay.addWidget(tool)
        lay.addWidget(self.canvas)

        roi_group = QGroupBox("ROI Settings")
        roi_lay = QHBoxLayout()
        self.rb_circle = QRadioButton("Circle")
        self.rb_rect = QRadioButton("Rectangle")
        self.rb_circle.setChecked(True)
        roi_lay.addWidget(self.rb_circle)
        roi_lay.addWidget(self.rb_rect)
        roi_group.setLayout(roi_lay)
        lay.addWidget(roi_group)

        btns = QHBoxLayout()
        ok = QPushButton('Confirm ROI')
        cancel = QPushButton('Cancel')
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        lay.addLayout(btns)
        main_layout.addWidget(left_widget)

        if self.show_spectrum:
            self.fig_spec = Figure(figsize=(5, 5))
            self.canvas_spec = FigureCanvas(self.fig_spec)
            self.ax_spec = self.fig_spec.add_subplot(111)
            self.ax_spec.set_title("Live ROI Spectrum")
            main_layout.addWidget(self.canvas_spec)
            self.resize(1200, 750)
        else:
            self.resize(750, 750)

        has_rgb = (np.min(self.wl) <= 480) and (np.max(self.wl) >= 630)
        if has_rgb:
            self.rb_rgb.setChecked(True)
            self.slider_group.hide()
        else:
            self.rb_rgb.setEnabled(False)
            self.rb_single.setChecked(True)
            self.slider_group.show()

        self.rb_rgb.toggled.connect(self.update_display)
        self.rb_single.toggled.connect(self.update_display)
        self.band_slider.valueChanged.connect(self.update_display)
        self.rb_circle.toggled.connect(self.init_selector)
        self.rb_rect.toggled.connect(self.init_selector)

        self.update_display()

    def _get_band_data(self, idx):
        """Read one band of the reference file as float32 ``(lines, samples)``.
        """
        return CalibrationWindow._read_envi_band_float32(
            self.ref_filepath, self.ref_meta, idx)

    def update_display(self):
        """Redraw the RGB composite or the selected band, percentile-scaled.
        """
        if self.rb_rgb.isChecked():
            self.slider_group.hide()
            if self._cached_rgb is None:
                def nearest(target):
                    return int(np.argmin(np.abs(self.wl - target)))

                r = self._get_band_data(nearest(650))
                g = self._get_band_data(nearest(550))
                b = self._get_band_data(nearest(450))

                def norm(x):
                    m, M = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
                    return np.clip((x - m) / (M - m + 1e-6), 0, 1)

                self._cached_rgb = np.dstack([norm(r), norm(g), norm(b)])
            display_data = self._cached_rgb
            title = "View Mode: RGB Color"
        else:
            self.slider_group.show()
            idx = self.band_slider.value()
            self.band_label.setText(f"Band: {idx} ({self.wl[idx]:.2f} nm)")
            img = self._get_band_data(idx)
            m, M = np.nanpercentile(img, 1), np.nanpercentile(img, 99)
            display_data = np.clip((img - m) / (M - m + 1e-6), 0, 1)
            title = f"View Mode: Single Band ({self.wl[idx]:.2f} nm)"

        if self.img_artist is None:
            self.img_artist = self.ax.imshow(display_data, interpolation='none')
            self.init_selector()
        else:
            self.img_artist.set_data(display_data)

        self.ax.set_title(title)
        self.canvas.draw_idle()

    def init_selector(self):
        """Create a circle or rectangle selector according to the radio buttons.
        """
        if self.selector is not None:
            self.selector.set_visible(False)
            self.selector.set_active(False)
            self.selector = None
        if self.roi_patch is not None:
            self.roi_patch.remove()
            self.roi_patch = None

        kwargs = dict(useblit=True, interactive=True, drag_from_anywhere=True)
        style = dict(edgecolor='cyan', linewidth=1.5, fill=False)

        if self.rb_circle.isChecked():
            self.selector = EllipseSelector(self.ax, self.on_roi_move, **kwargs, props=style)
        else:
            self.selector = RectangleSelector(self.ax, self.on_roi_move, **kwargs, props=style)
        self.canvas.draw_idle()

    def on_roi_move(self, eclick, erelease):
        """Selector slot: rebuild the boolean mask ``(lines, samples)`` and ROI outline.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if x1 is None or x2 is None or y1 is None or y2 is None:
            return

        dist = np.hypot(x1 - x2, y1 - y2)
        if dist < 1:
            self.selector.set_visible(False)
            self.canvas.draw_idle()
            return

        self.selector.set_visible(True)
        if self.roi_patch is not None:
            self.roi_patch.remove()

        if self.rb_circle.isChecked():
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            radius = min(abs(x1 - x2) / 2.0, abs(y1 - y2) / 2.0)
            yy, xx = np.indices((self.H, self.W))
            self.mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (radius ** 2)
            self.roi_patch = Circle((cx, cy), radius=radius, fill=False, color='blue', lw=2)
        else:
            xs = sorted([int(round(x1)), int(round(x2))])
            ys = sorted([int(round(y1)), int(round(y2))])
            self.mask = np.zeros((self.H, self.W), dtype=bool)
            self.mask[max(0, ys[0]):min(self.H, ys[1]),
                      max(0, xs[0]):min(self.W, xs[1])] = True
            self.roi_patch = Rectangle((xs[0], ys[0]), xs[1] - xs[0],
                                       ys[1] - ys[0], fill=False, edgecolor='blue', lw=2)

        self.ax.add_patch(self.roi_patch)
        self.canvas.draw_idle()
        if self.show_spectrum:
            self.update_spectrum_plot()

    def update_spectrum_plot(self):
        """Plot the raw and smoothed ROI-mean spectrum (reads every band).
        """
        if not self.mask.any() or not self.ref_filepath:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            num_bands = self.ref_meta['bands']
            avg_spec = np.zeros(num_bands, dtype=np.float32)
            for i in range(num_bands):
                im = self._get_band_data(i)
                avg_spec[i] = np.nanmean(im[self.mask]) if self.mask.any() else 1.0

            smoothed = savgol_filter(avg_spec, 11, 3) if len(avg_spec) > 11 else avg_spec
            self.ax_spec.clear()
            self.ax_spec.plot(self.wl, avg_spec, color='lightgray', label='Raw')
            self.ax_spec.plot(self.wl, smoothed, color='red', label='Smoothed')
            self.ax_spec.legend()
            self.canvas_spec.draw_idle()
        finally:
            QApplication.restoreOverrideCursor()

    def get_mask(self):
        """Return the boolean ROI mask ``(lines, samples)`` (all False if nothing was drawn).
        """
        return self.mask


# =============================================================================
# Standalone entry point
# =============================================================================


if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)

    try:
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    except ImportError:
        pass

    win = CalibrationWindow()
    win.show()
    sys.exit(app.exec_())
