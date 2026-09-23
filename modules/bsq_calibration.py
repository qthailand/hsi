# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import traceback
import math
import re
import pandas as pd
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
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


# =============================================================================
# Helper: Background Worker
# =============================================================================

import spectral

class Worker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.kwargs['progress_callback'] = self.progress
            result = self.fn(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


# =============================================================================
# CalibrationWindow — เปิดได้ทั้งจาก Scan และ Browse (ไม่มี Dark)
# =============================================================================


class CalibrationWindow(QWidget):
    """
    หน้าต่าง Calibrate + Post-processing
    เปิดได้ 3 ทาง:
      1. load_from_scan(filepath)
      2. load_from_memory(array, wl)
      3. browse ด้วยปุ่มปกติ
    Dark subtraction ทั้งหมดถูกตัดออก — ระบบ scan จัดการเอง
    """

    raw_filepath: str = ''
    raw_meta: dict = None

    cubname: str = ''
    cubwidth: int = 0
    cubheight: int = 0
    cubband: int = 0
    isWavelength: bool = False

    rowwise_factors: np.ndarray = None
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
        self.navbar = NavigationToolbar2QT(self.canvas)
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
        self._source_mode = 'scan_file'
        self._load_file_internal(filepath, meta=meta, source_label="Loaded from Scan")

    def load_from_memory(self, cube_array, wavelengths=None, name="ScanData"):
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

    @staticmethod
    def _apply_spatial_binning(img, n, to_uint16=False):
        if n <= 1:
            return img
        h, w = img.shape
        h_new, w_new = h // n, w // n
        img_cropped = img[:h_new * n, :w_new * n]
        reshaped = img_cropped.reshape(h_new, n, w_new, n)
        binned = reshaped.mean(axis=(1, 3))
        if to_uint16:
            return np.round(binned).astype(np.uint16)
        return binned.astype(np.float32)

    def _update_bin_info(self):
        if not self.raw_meta:
            self.label_bin_info.setText("Output Size: - x - px")
            return
        idx = self.bin_combo.currentIndex()
        n = 4 if idx == 2 else (2 if idx == 1 else 1)
        try:
            x0, x1 = np.ceil(self.axes.get_xlim()).astype('uint')
            y1, y0 = np.ceil(self.axes.get_ylim()).astype('uint')
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

    @staticmethod
    def parse_envi_header(filepath):
        hdr_path = filepath + '.hdr'
        if not os.path.exists(hdr_path):
            hdr_path = os.path.splitext(filepath)[0] + '.hdr'
        if not os.path.exists(hdr_path):
            raise ValueError(f"Header file not found for {filepath}")

        meta = {'data_type': 4, 'byte_order': 0,
                'interleave': 'bsq', 'description': 'Processed Data'}
        with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
            txt = f.read()

        m_desc = re.search(r'description\s*=\s*\{([^}]*)\}', txt, re.IGNORECASE | re.DOTALL)
        if m_desc:
            meta['description'] = m_desc.group(1).strip().replace('\n', ' ')
        else:
            m_desc_simple = re.search(
                r'^description\s*=\s*(.*)$', txt, re.IGNORECASE | re.MULTILINE)
            if m_desc_simple:
                meta['description'] = m_desc_simple.group(1).strip()

        for line in txt.split('\n'):
            if '=' in line and 'description' not in line.lower() and 'wavelength' not in line.lower():
                k, v = line.split('=', 1)
                meta[k.strip().lower()] = v.strip()

        m_wl = re.search(r'wavelength\s*=\s*\{([^}]*)\}', txt, re.IGNORECASE | re.DOTALL)
        if m_wl:
            vals = m_wl.group(1).replace('\n', ' ').split(',')
            meta['wavelength'] = np.array(
                [float(v.strip()) for v in vals if v.strip() != ''], dtype=np.float32)
        else:
            meta['wavelength'] = None

        meta['lines'] = int(meta['lines'])
        meta['samples'] = int(meta['samples'])
        meta['bands'] = int(meta['bands'])
        return meta
    @staticmethod
    def _read_envi_band_float32(filepath, meta, band_idx):
        L, S, B = meta['lines'], meta['samples'], meta['bands']
        dt_code = int(meta.get('data type', meta.get('data_type', '4')))
        dt_map = {12: (np.uint16, 2), 4: (np.float32, 4), 2: (np.int16, 2),
                1: (np.uint8, 1), 5: (np.float64, 8)}
        dt, bpp = dt_map.get(dt_code, (np.float32, 4))

        interleave = str(meta.get('interleave', 'bsq')).lower().strip()

        with open(filepath, 'rb') as f:
            if interleave == 'bsq':
                f.seek(band_idx * L * S * bpp)
                raw = f.read(L * S * bpp)
                return np.frombuffer(raw, dtype=dt).reshape(L, S).astype(np.float32)

            elif interleave == 'bip':
                data = np.frombuffer(f.read(), dtype=dt).reshape(L, S, B)
                return data[:, :, band_idx].astype(np.float32)

            elif interleave == 'bil':
                data = np.frombuffer(f.read(), dtype=dt).reshape(L, B, S)
                return data[:, band_idx, :].astype(np.float32)

            else:
                raise ValueError(f"Unsupported interleave: {interleave}")

    # =====================================================================
    #  UI Thread helpers
    # =====================================================================

    def _run_in_thread(self, func, *args, **kwargs):
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
        btns = [self.btn, self.btn_clear_inputs,
                self.btn_rowwise, self.btn_save_profile, self.btn_load_profile,
                self.btn_create_datacube,
                getattr(self, 'btn_fluo_cal', None),
                getattr(self, 'btn_load_fluo_profile', None),
                getattr(self, 'btn_save_fluo_profile', None)]
        for btn in btns:
            if btn is not None:
                btn.setEnabled(enabled)
        self.preview_slider.setEnabled(enabled and (self.wavelengths is not None))

    def on_worker_error(self, error_msg):
        QMessageBox.warning(self, 'Error', error_msg)
        self.set_ui_enabled(True)
        self.progess.setValue(0)
        if self.thread:
            self.thread.quit()
            self.thread.wait()

    def mark_button_done(self, btn: QPushButton):
        btn.setStyleSheet(
            "QPushButton{background-color:#2e7d32;color:white;border-radius:6px;"
            "padding:6px 10px;} QPushButton:disabled{background-color:#1b5e20;color:#bbbbbb;}")

    def on_slider_value_changed(self, index):
        if self.wavelengths is None or index >= len(self.wavelengths):
            return
        self.preview_idx_edit.setText(f"{self.wavelengths[index]:.2f}")
        self.update_preview_image()

    # =====================================================================
    #  Browse Events
    # =====================================================================

    def event_browse(self):
        filepath, _ = QFileDialog.getOpenFileName(
            None, 'Select Raw ENVI Data', '',
            'ENVI Files (*.bsq *.envi *.raw);;All Files (*)')
        if not filepath:
            return
        self._source_mode = 'browse'
        self.btn.setEnabled(True)
        self._load_file_internal(filepath, source_label="Browse ENVI")

    def event_clear_inputs(self):
        self.raw_filepath = ''
        self.raw_meta = None
        self.cubname = ''
        self.cubwidth = 0
        self.cubheight = 0
        self.cubband = 0
        self.wavelengths = None
        self.rowwise_factors = None
        self.factors = None
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
        self.bsq_raw_path = self.bsq_raw_mm = self.bsq_raw_shape = self.bsq_raw_dtype = None
        self.bsq_cal_path = self.bsq_cal_mm = self.bsq_cal_shape = self.bsq_cal_dtype = None
        self._raw_temp_building = False
        self._cal_temp_building = False
        self._post_build_actions.clear()

    # =====================================================================
    #  Calibration UI
    # =====================================================================

    def event_save_profile(self):
        if self.rowwise_factors is None:
            QMessageBox.warning(self, 'Error', 'No row-wise factors to save.')
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Calibration Profile", "", "NumPy Array (*.npy)")
        if save_path:
            try:
                np.save(save_path, self.rowwise_factors)
                QMessageBox.information(self, 'Success', f'Profile saved:\n{save_path}')
                self.mark_button_done(self.btn_save_profile)
            except Exception as e:
                QMessageBox.warning(self, 'Error', str(e))

    def event_load_profile(self):
        load_path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration Profile", "", "NumPy Array (*.npy)")
        if load_path:
            try:
                self.rowwise_factors = np.load(load_path)
                self.factors = None
                self.label_profile.setText(f"Loaded: {os.path.basename(load_path)}")
                QMessageBox.information(self, 'Success', f'Profile loaded:\n{load_path}')
                self.mark_button_done(self.btn_load_profile)
                self.update_preview_image()
                self._clear_memmap_temp_states()
                self.btn_rowwise.setEnabled(False)
                self.btn_save_profile.setEnabled(False)
            except Exception as e:
                QMessageBox.warning(self, 'Error', str(e))

    def event_rowwise_calibration(self):
        # ★ ถ้ามาจาก scan (memory/file) — ใช้ cube ที่โหลดไว้เลย
        if self._source_mode in ('scan_file', 'scan_memory'):
            ref_filepath = self.raw_filepath
            ref_meta = self.raw_meta
        else:
            # Browse mode — ต้องเลือก reference file
            ref_filepath, _ = QFileDialog.getOpenFileName(
                None, 'Select Reference Data (.envi)', '',
                'ENVI Files (*.bsq *.envi *.raw);;All Files (*)')
            if not ref_filepath:
                return
            try:
                ref_meta = self.parse_envi_header(ref_filepath)
            except Exception as e:
                QMessageBox.warning(self, 'Error', str(e))
                return

        # ★ ทุก path ผ่านตรงนี้ — เลือก Excel
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ref_excel, _ = QFileDialog.getOpenFileName(
            self, 'Select Spectralon Excel', script_dir, 'Excel Files (*.xlsx *.xls)')
        if not ref_excel:
            return
        try:
            avg_window_size = int(self.avg_window_edit.text())
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Invalid averaging window size.')
            return

        mask = self.select_roi_on_main(ref_filepath, ref_meta, ref_excel)
        if mask is None:
            return

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
        if result:
            self.rowwise_factors = result['factors']
            self.spectralon_rows_span = result['span']
            self.factors = None
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
        if self.chk_resample.isChecked():
            self.event_create_resampled()
        else:
            self.event_create()

    def event_create(self):
        if not self.raw_filepath:
            QMessageBox.warning(self, "Error", "Please select a raw ENVI file first.")
            return
        x0, x1 = np.ceil(self.axes.get_xlim()).astype('uint')
        y1, y0 = np.ceil(self.axes.get_ylim()).astype('uint')
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
            "binning_factor": n_bin,
        }
        worker, thread = self._run_in_thread(self.create_datacube_logic, params)
        worker.finished.connect(self.on_create_finished)

    def on_create_finished(self, result_path):
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
        wl = ref_meta.get('wavelength', None)
        dlg = RoiDialog(parent=self, ref_filepath=ref_filepath,
                        ref_meta=ref_meta, wl=wl)
        if dlg.exec_() != QDialog.Accepted:
            return None
        return dlg.get_mask()

    # =====================================================================
    #  Heavy Logic (static — no dark anywhere)
    # =====================================================================

    @staticmethod
    def rowwise_calculation_heavy_logic(ref_filepath, ref_meta, ref_excel,
                                        avg_window_size, mask,
                                        progress_callback):
        data_ref = pd.read_excel(ref_excel, header=None).to_numpy()
        f_ref = CubicSpline(data_ref[:, 0], data_ref[:, 1])
        wl = ref_meta['wavelength']

        H, W = ref_meta['lines'], ref_meta['samples']
        if mask.shape != (H, W):
            raise ValueError('ROI mask shape does not match reference image.')

        rows_has_roi = np.where(mask.any(axis=1))[0]
        if len(rows_has_roi) == 0:
            raise ValueError('Selected ROI does not cover any pixels.')
        top_roi, bot_roi = int(rows_has_roi.min()), int(rows_has_roi.max())

        def pick_rows_in_roi(y):
            nearest = rows_has_roi[np.argmin(np.abs(rows_has_roi - y))]
            idx = np.where(rows_has_roi == nearest)[0][0]
            left = max(0, idx - (avg_window_size // 2))
            right = min(len(rows_has_roi), left + avg_window_size)
            if right - left < avg_window_size:
                left = max(0, right - avg_window_size)
            return rows_has_roi[left:right]

        rowwise_factors = np.zeros((H, len(wl)), dtype=np.float32)
        R_ref_per_band = f_ref(wl).astype(np.float32)

        progress_callback.emit(0)
        for i in range(ref_meta['bands']):
            ref_img = CalibrationWindow._read_envi_band_float32(ref_filepath, ref_meta, i)
            ref_roi = np.where(mask[:, :ref_img.shape[1]], ref_img[:, :ref_img.shape[1]], np.nan)
            fallback = np.nanmedian(ref_roi)
            if not np.isfinite(fallback) or fallback <= 0:
                fallback = 1.0

            for y in range(ref_roi.shape[0]):
                y_ref = int(np.clip(y, top_roi, bot_roi))
                rows_window = pick_rows_in_roi(y_ref)
                mean_val = np.nanmean(ref_roi[rows_window, :]) if len(rows_window) > 0 else fallback
                if not np.isfinite(mean_val) or mean_val <= 0:
                    mean_val = fallback
                rowwise_factors[y, i] = R_ref_per_band[i] / mean_val

            progress_callback.emit(i + 1)
        return {'factors': rowwise_factors, 'span': (top_roi, bot_roi)}

    @staticmethod
    def create_datacube_logic(params, progress_callback):
        raw_filepath = params['raw_filepath']
        raw_meta = params['raw_meta']
        save_path = params['save_path']
        x0, x1, y0, y1 = params['crop_coords']
        is_wavelength = params['is_wavelength']
        band_count = params['band_count']
        factors = params['factors']
        rowwise_factors = params['rowwise_factors']
        n_bin = params.get('binning_factor', 1)

        cubheight, cubwidth = (y1 - y0) // n_bin, (x1 - x0) // n_bin
        is_cal = (rowwise_factors is not None) or (factors is not None)
        in_dt = int(raw_meta.get('data type', 12))
        if is_cal:
            out_dt, out_dtype = 4, np.float32
        else:
            out_dt, out_dtype = in_dt, (np.float32 if in_dt == 4 else np.uint16)

        hdr_path = save_path + '.hdr'
        with open(hdr_path, 'w') as hopen:
            hopen.write('ENVI\n')
            old_desc = raw_meta.get('description', 'Processed Data')
            bin_text = f"Spatial Binning = {n_bin}x{n_bin}" if n_bin > 1 else "Original Spatial Res"
            hopen.write(f"description = {{\n  {old_desc},\n  {bin_text}\n}}\n")
            hopen.write(f'samples = {cubwidth}\n')
            hopen.write(f'lines = {cubheight}\n')
            hopen.write(f'bands = {band_count}\n')
            hopen.write(f'data type = {out_dt}\n')
            hopen.write('interleave = bip\n')
            hopen.write('byte order = 0\n')
            hopen.write('wavelength units = Nanometers\n')
            if is_wavelength and raw_meta.get('wavelength') is not None:
                wl_str = ','.join([str(item) for item in raw_meta['wavelength']])
                hopen.write(f'wavelength = {{{wl_str}}}\n')

        c = np.zeros((cubheight * band_count, cubwidth), dtype=out_dtype)
        progress_callback.emit(0)

        for i in range(band_count):
            im = CalibrationWindow._read_envi_band_float32(raw_filepath, raw_meta, i)
            roi_img = im[y0:y1, x0:x1]
            if rowwise_factors is not None:
                roi_img *= rowwise_factors[y0:y1, i].reshape(-1, 1)
            elif factors is not None:
                roi_img *= float(factors[i])
            out = CalibrationWindow._apply_spatial_binning(
                roi_img, n_bin, to_uint16=(out_dtype == np.uint16))
            c[cubheight * i: cubheight * (i + 1)] = out
            progress_callback.emit(i + 1)

        bip = c.reshape((band_count, cubheight, cubwidth)).transpose(1, 2, 0)  # HWC
        bip.tofile(save_path)
        return save_path

    # =====================================================================
    #  Resampled Datacube
    # =====================================================================

    def event_create_resampled(self):
        if not self.raw_filepath:
            QMessageBox.warning(self, "Error", "Please select a raw ENVI file first.")
            return
        x0, x1 = np.ceil(self.axes.get_xlim()).astype('uint')
        y1, y0 = np.ceil(self.axes.get_ylim()).astype('uint')

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
            "lam_target": lam_target.tolist(),
            "binning_factor": n_bin
        }
        worker, thread = self._run_in_thread(
            self.create_datacube_resampled_logic, params)
        worker.finished.connect(self.on_create_finished)

    @staticmethod
    def create_datacube_resampled_logic(params, progress_callback):
        raw_filepath = params['raw_filepath']
        raw_meta = params['raw_meta']
        save_path = params['save_path']
        x0, x1, y0, y1 = params['crop_coords']
        is_wavelength = params['is_wavelength']
        factors = params['factors']
        rowwise_factors = params['rowwise_factors']
        lam_target = np.array(params['lam_target'], dtype=np.float32)
        n_bin = params.get('binning_factor', 1)

        wl_all = raw_meta['wavelength']
        idx_right = np.searchsorted(wl_all, lam_target, side='left')
        idx_left = np.clip(idx_right - 1, 0, len(wl_all) - 1)
        idx_right = np.clip(idx_right, 0, len(wl_all) - 1)
        denom = np.maximum(wl_all[idx_right] - wl_all[idx_left], 1e-6)
        alpha = (lam_target - wl_all[idx_left]) / denom

        cubheight, cubwidth = (y1 - y0) // n_bin, (x1 - x0) // n_bin
        is_cal = (rowwise_factors is not None) or (factors is not None)
        in_dt = int(raw_meta.get('data type', 12))
        if is_cal:
            out_dt, out_dtype_str, out_dtype_np = 4, 'float32', np.float32
        else:
            out_dt = in_dt
            out_dtype_str = 'float32' if in_dt == 4 else 'uint16'
            out_dtype_np = np.float32 if in_dt == 4 else np.uint16

        hdr_path = save_path + '.hdr'
        with open(hdr_path, 'w') as hopen:
            hopen.write('ENVI\n')
            old_desc = raw_meta.get('description', 'Processed Data')
            bin_text = f"Spatial Binning = {n_bin}x{n_bin}" if n_bin > 1 else "Original Spatial Res"
            hopen.write(f"description = {{\n  {old_desc},\n  Resampled = True,\n  {bin_text}\n}}\n")
            hopen.write(f'samples = {cubwidth}\n')
            hopen.write(f'lines = {cubheight}\n')
            hopen.write(f'bands = {len(lam_target)}\n')
            hopen.write(f'data type = {out_dt}\n')
            hopen.write('interleave = bip\n')
            hopen.write('byte order = 0\n')
            hopen.write('wavelength units = Nanometers\n')
            if is_wavelength:
                wl_str = ','.join(f"{float(v):.2f}" for v in lam_target)
                hopen.write(f'wavelength = {{{wl_str}}}\n')

        c = np.zeros((cubheight * len(lam_target), cubwidth), dtype=out_dtype_str)
        progress_callback.emit(0)

        for j in range(len(lam_target)):
            iL, iR, a = int(idx_left[j]), int(idx_right[j]), float(alpha[j])
            imL = CalibrationWindow._read_envi_band_float32(raw_filepath, raw_meta, iL)
            imR = CalibrationWindow._read_envi_band_float32(raw_filepath, raw_meta, iR)

            Wmin = min(imL.shape[1], imR.shape[1])
            roiL = imL[y0:y1, x0:min(x1, x0 + Wmin)]
            roiR = imR[y0:y1, x0:min(x1, x0 + Wmin)]

            if rowwise_factors is not None:
                roiL *= rowwise_factors[y0:y1, iL].reshape(-1, 1)
                roiR *= rowwise_factors[y0:y1, iR].reshape(-1, 1)
            elif factors is not None:
                roiL *= float(factors[iL])
                roiR *= float(factors[iR])

            roiJ = (1.0 - a) * roiL + a * roiR
            out = CalibrationWindow._apply_spatial_binning(
                roiJ, n_bin, to_uint16=(out_dtype_np == np.uint16))
            c[cubheight * j: cubheight * (j + 1)] = out
            progress_callback.emit(j + 1)

        bip = c.reshape((len(lam_target), cubheight, cubwidth)).transpose(1, 2, 0)  # HWC
        bip.tofile(save_path)
        return save_path

    # =====================================================================
    #  Helper Utils
    # =====================================================================

    def _round2_floor(self, x: float) -> float:
        return math.floor(x * 100.0 + 1e-9) / 100.0

    def _round2_ceil(self, x: float) -> float:
        return math.ceil(x * 100.0 - 1e-12) / 100.0

    def _calc_lmax_from(self, lmin: float, fwhm: float, max_cap: float) -> float:
        if fwhm <= 0:
            return self._round2_floor(max_cap)
        n = math.floor((max_cap - lmin + 1e-9) / fwhm)
        return self._round2_floor(lmin + n * fwhm)

    def _load_wavelengths_current(self):
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
        c = self._colors[self._color_idx % len(self._colors)]
        self._color_idx += 1
        return c

    def _ensure_spec_ax(self, ylabel='Intensity'):
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
        self.spec_fig = None
        self.spec_ax = None

    def _add_marker(self, patch, line, fill):
        self.markers.append({'patch': patch, 'line': line, 'fill': fill})

    def remove_last_marker(self):
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

    @staticmethod
    def build_raw_cube_tmp(params, progress_callback):
        raw_filepath = params['folder']
        raw_meta = params['raw_meta']
        tmp_path = params['save_path']
        x0, x1, y0, y1 = params['crop']

        lines, samples = (y1 - y0), (x1 - x0)
        bands = raw_meta['bands']
        mm = np.memmap(tmp_path, mode='w+', dtype=np.uint16, shape=(lines * bands, samples))

        progress_callback.emit(0)
        for i in range(bands):
            im = CalibrationWindow._read_envi_band_float32(raw_filepath, raw_meta, i)
            roi = im[y0:y1, x0:x1]
            mm[lines * i: lines * (i + 1)] = np.clip(roi, 0, 65535).astype(np.uint16)
            progress_callback.emit(i + 1)
        mm.flush()
        return {'path': tmp_path, 'lines': lines, 'samples': samples,
                'bands': bands, 'dtype': 'uint16'}

    @staticmethod
    def build_cal_cube_tmp(params, progress_callback):
        raw_filepath = params['raw_filepath']
        raw_meta = params['raw_meta']
        tmp_path = params['save_path']
        x0, x1, y0, y1 = params['crop']
        factors = params.get('factors', None)
        rowwise = params.get('rowwise_factors', None)

        lines, samples = (y1 - y0), (x1 - x0)
        bands = raw_meta['bands']
        mm = np.memmap(tmp_path, mode='w+', dtype=np.float32, shape=(lines * bands, samples))

        progress_callback.emit(0)
        for i in range(bands):
            im = CalibrationWindow._read_envi_band_float32(raw_filepath, raw_meta, i)
            roi = im[y0:y1, x0:x1]
            if rowwise is not None:
                fac = rowwise[y0:y1, i].reshape(-1, 1)
                roi = roi * fac
            elif factors is not None:
                roi = roi * float(factors[i])
            mm[lines * i: lines * (i + 1)] = roi.astype(np.float32)
            progress_callback.emit(i + 1)
        mm.flush()
        return {'path': tmp_path, 'lines': lines, 'samples': samples,
                'bands': bands, 'dtype': 'float32'}

    def ensure_raw_memmap(self, on_ready=None):
        if self.bsq_raw_mm is not None:
            return True
        if self._raw_temp_building:
            if callable(on_ready):
                self._post_build_actions.append(on_ready)
            return False
        if not self.raw_filepath:
            return False

        x0, x1 = np.ceil(self.axes.get_xlim()).astype('uint')
        y1, y0 = np.ceil(self.axes.get_ylim()).astype('uint')
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

        x0, x1 = np.ceil(self.axes.get_xlim()).astype('uint')
        y1, y0 = np.ceil(self.axes.get_ylim()).astype('uint')
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
                    'rowwise_factors': getattr(self, 'rowwise_factors', None)})

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
        ref_filepath, _ = QFileDialog.getOpenFileName(
            None, 'Select Fluorescence Reference (.envi)', '',
            'ENVI Files (*.bsq *.envi *.raw);;All Files (*)')
        if not ref_filepath:
            return
        try:
            ref_meta = self.parse_envi_header(ref_filepath)
        except Exception as e:
            QMessageBox.warning(self, 'Error', str(e))
            return

        mask = self.select_roi_fluo_main(ref_filepath, ref_meta)
        if mask is None:
            return

        self.progess.setMaximum(ref_meta['bands'])
        self.progess.setValue(0)
        worker, thread = self._run_in_thread(
            self.fluo_calculation_heavy_logic,
            ref_filepath=ref_filepath, ref_meta=ref_meta, mask=mask)
        worker.finished.connect(self.on_fluo_calibration_finished)

    def select_roi_fluo_main(self, ref_filepath, ref_meta):
        wl = ref_meta.get('wavelength', None)
        dlg = RoiDialog(parent=self, ref_filepath=ref_filepath,
                        ref_meta=ref_meta, wl=wl, show_spectrum=True)
        if dlg.exec_() != QDialog.Accepted:
            return None
        return dlg.get_mask()

    @staticmethod
    def fluo_calculation_heavy_logic(ref_filepath, ref_meta, mask, progress_callback):
        num_bands = ref_meta['bands']
        avg_ref_spectrum = np.zeros(num_bands, dtype=np.float32)

        progress_callback.emit(0)
        for i in range(num_bands):
            img = CalibrationWindow._read_envi_band_float32(ref_filepath, ref_meta, i)
            roi_data = img[mask[:, :img.shape[1]]]
            avg_ref_spectrum[i] = np.nanmean(roi_data) if roi_data.size > 0 else 1.0
            progress_callback.emit(i + 1)

        try:
            smoothed = savgol_filter(avg_ref_spectrum, window_length=11, polyorder=3)
            smoothed = np.where(smoothed <= 0, 1.0, smoothed)
        except Exception:
            smoothed = np.where(avg_ref_spectrum <= 0, 1.0, avg_ref_spectrum)

        return {'factors': 1.0 / smoothed}

    def on_fluo_calibration_finished(self, result):
        if result:
            self.factors = result['factors']
            self.rowwise_factors = None
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
        if self.factors is None:
            QMessageBox.warning(self, 'Error', 'No Fluorescence factors to save.')
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Fluorescence Profile", "", "NumPy Array (*.npy)")
        if save_path:
            try:
                np.save(save_path, self.factors)
                QMessageBox.information(self, 'Success', f'Profile saved:\n{save_path}')
                self.mark_button_done(self.btn_save_fluo_profile)
            except Exception as e:
                QMessageBox.warning(self, 'Error', str(e))

    def event_load_fluo_profile(self):
        load_path, _ = QFileDialog.getOpenFileName(
            self, "Load Fluorescence Profile", "", "NumPy Array (*.npy)")
        if load_path:
            try:
                self.factors = np.load(load_path)
                self.rowwise_factors = None
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
    def __init__(self, parent=None, ref_filepath=None, ref_meta=None,
                 wl=None, show_spectrum=False):
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
        tool = NavigationToolbar2QT(self.canvas)
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
        return CalibrationWindow._read_envi_band_float32(
            self.ref_filepath, self.ref_meta, idx)

    def update_display(self):
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
