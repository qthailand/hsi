import sys
import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from smile.backends.joe.calibration_logic import (
    detect_peaks_across_rows,
    cluster_peak_lines,
    build_candidate_lines,
    remove_outlier_lines,
    refine_parabolic_peaks,
    build_trend_models,
    generate_smile_maps,
    remap_image,
    compute_diagnostic_results,
    detect_keystone_edges,
    build_keystone_model,
)

# --- 1. Calibration logic imported from smile.calibration_logic ---

class PrecisionOpticsGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spector: HSI Calibration & Diagnostic Suite (Pro Version)")
        self.setGeometry(50, 50, 1600, 950)
        
        # Data storage
        self.raw_img = None
        self.corrected_img = None
        self.keystone_ref_img = None
        self.detected_peaks_coords = [] 
        self.keystone_model = None
        self.ui_lines = []
        
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- Create Tabs ---
        self.tabs = QTabWidget()
        self.tab_main = QWidget()
        self.tab_diag = QWidget()
        self.tab_spec_comp = QWidget() 
        self.tab_raw_diag = QWidget()  
        
        self.tabs.addTab(self.tab_main, "📂 Calibration Control")
        self.tabs.addTab(self.tab_diag, "📊 Diagnostic Dashboard")
        self.tabs.addTab(self.tab_spec_comp, "📈 Spectral Alignment")
        self.tabs.addTab(self.tab_raw_diag, "🔍 Raw Analysis (Unfiltered)")
        layout.addWidget(self.tabs)

        # ==========================================
        # TAB 1: MAIN CALIBRATION
        # ==========================================
        main_tab_layout = QHBoxLayout(self.tab_main)
        controls = QVBoxLayout()
        
        # Group 1: Input
        file_group = QGroupBox("1. Input Image")
        file_lay = QVBoxLayout()
        
        # --- ส่วนที่เพิ่มใหม่: Dropdown สำหรับเลือก Bin ---
        bin_lay = QHBoxLayout()
        bin_lay.addWidget(QLabel("Target Binning:"))
        self.combo_bin = QComboBox()
        self.combo_bin.addItems(["Bin1", "Bin2", "Bin3", "Bin4", "Bin8"])
        self.combo_bin.setCurrentIndex(3) # ตั้งค่าเริ่มต้นที่ Bin4
        bin_lay.addWidget(self.combo_bin)
        file_lay.addLayout(bin_lay)
        # ----------------------------------------------
        
        self.btn_load = QPushButton("📂 Load Raw HSI Frame")
        self.btn_load.clicked.connect(self.load_image)
        file_lay.addWidget(self.btn_load)
        file_group.setLayout(file_lay)
        controls.addWidget(file_group)

        # Group 2: Detection Parameters (DBSCAN & Scanning)
        detect_group = QGroupBox("2. Detection & Clustering")
        detect_lay = QGridLayout()
        self.edit_sample_rows = QLineEdit("40")
        self.edit_dbscan_eps = QLineEdit("10.0")
        self.edit_dbscan_min = QLineEdit("5")
        
        detect_lay.addWidget(QLabel("Sample Rows (Scanning):"), 0, 0)
        detect_lay.addWidget(self.edit_sample_rows, 0, 1)
        detect_lay.addWidget(QLabel("DBSCAN Eps:"), 1, 0)
        detect_lay.addWidget(self.edit_dbscan_eps, 1, 1)
        detect_lay.addWidget(QLabel("DBSCAN Min Samples:"), 2, 0)
        detect_lay.addWidget(self.edit_dbscan_min, 2, 1)
        detect_group.setLayout(detect_lay)
        controls.addWidget(detect_group)

        # Group 3: Smile Calibration & Physical Constraints
        smile_group = QGroupBox("3. Smile & Physics Constraints")
        smile_lay = QGridLayout()
        self.edit_x_start = QLineEdit("0")
        self.edit_x_end = QLineEdit("Max")
        self.edit_thresh = QLineEdit("25")
        self.edit_sigma = QLineEdit("5.0")
        self.edit_min_pts_line = QLineEdit("15") # Min Pts/Line
        
        smile_lay.addWidget(QLabel("ROI Start X:"), 0, 0); smile_lay.addWidget(self.edit_x_start, 0, 1)
        smile_lay.addWidget(QLabel("ROI End X:"), 1, 0); smile_lay.addWidget(self.edit_x_end, 1, 1)
        smile_lay.addWidget(QLabel("Peak Threshold:"), 2, 0); smile_lay.addWidget(self.edit_thresh, 2, 1)
        smile_lay.addWidget(QLabel("Smoothing Sigma:"), 3, 0); smile_lay.addWidget(self.edit_sigma, 3, 1)
        smile_lay.addWidget(QLabel("Min Points/Line:"), 4, 0); smile_lay.addWidget(self.edit_min_pts_line, 4, 1)
        
        self.btn_preview_smile = QPushButton("👁️ Detect & Preview Smile")
        self.btn_preview_smile.setFixedHeight(35)
        self.btn_preview_smile.clicked.connect(self.preview_smile)
        smile_lay.addWidget(self.btn_preview_smile, 5, 0, 1, 2)
        smile_group.setLayout(smile_lay)
        controls.addWidget(smile_group)

        # Group 4: Keystone Option
        key_group = QGroupBox("4. Keystone Correction")
        key_lay = QVBoxLayout()
        self.check_keystone = QCheckBox("Enable Keystone Correction")
        self.btn_load_key = QPushButton("📂 Load Keystone Reference")
        self.btn_load_key.setEnabled(False)
        self.btn_load_key.clicked.connect(self.load_keystone_reference)
        self.check_keystone.toggled.connect(lambda b: self.btn_load_key.setEnabled(b))
        key_lay.addWidget(self.check_keystone)
        key_lay.addWidget(self.btn_load_key)
        key_group.setLayout(key_lay)
        controls.addWidget(key_group)

        # Group 5: Process
        out_group = QGroupBox("5. Execute")
        out_lay = QVBoxLayout()
        self.btn_process = QPushButton("🚀 Apply Unified Correction")
        self.btn_process.setFixedHeight(50)
        self.btn_process.setStyleSheet("background-color: #2c3e50; color: white; font-weight: bold; font-size: 14px;")
        self.btn_process.clicked.connect(self.apply_unified_correction)
        out_lay.addWidget(self.btn_process)
        out_group.setLayout(out_lay)
        controls.addWidget(out_group)
        
        controls.addStretch()
        main_tab_layout.addLayout(controls, 1)

        # Viewers (Tab 1)
        display_lay = QVBoxLayout()
        self.win1 = pg.GraphicsLayoutWidget()
        self.view1 = self.win1.addPlot(title="Smile Detection Preview (Filtered Lines)")
        self.img1 = pg.ImageItem()
        self.view1.addItem(self.img1)
        self.view1.invertY(True)
        self.view1.showGrid(x=True, y=True, alpha=0.5)

        self.win2 = pg.GraphicsLayoutWidget()
        self.view2 = self.win2.addPlot(title="Unified Correction Result")
        self.img2 = pg.ImageItem()
        self.view2.addItem(self.img2)
        self.view2.invertY(True)
        self.view2.showGrid(x=True, y=True, alpha=0.5)
        
        self.img1.mouseDoubleClickEvent = self.on_image_double_clicked
        self.img2.mouseDoubleClickEvent = self.on_image_double_clicked
        
        display_lay.addWidget(self.win1)
        display_lay.addWidget(self.win2)
        main_tab_layout.addLayout(display_lay, 4)

        # ==========================================
        # TAB 2: DIAGNOSTIC DASHBOARD
        # ==========================================
        diag_layout = QHBoxLayout(self.tab_diag)
        self.win_diag = pg.GraphicsLayoutWidget()
        self.plot_morph = self.win_diag.addPlot(title="Spectral Line Morphology (Pre vs Post)")
        self.plot_morph.showGrid(x=True, y=True, alpha=0.3)
        self.win_diag.nextRow()
        self.plot_resid = self.win_diag.addPlot(title="Residual Deviation (Sub-pixel Error)")
        self.plot_resid.showGrid(x=True, y=True, alpha=0.3)
        diag_layout.addWidget(self.win_diag, 4)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        diag_layout.addWidget(self.stats_text, 1)

        # ==========================================
        # TAB 3: SPECTRAL ALIGNMENT
        # ==========================================
        spec_layout = QHBoxLayout(self.tab_spec_comp)
        spec_sidebar = QVBoxLayout()
        spec_roi_group = QGroupBox("View Settings")
        spec_roi_lay = QGridLayout()
        self.edit_spec_x_start = QLineEdit("0")
        self.edit_spec_x_end = QLineEdit("Max")
        spec_roi_lay.addWidget(QLabel("Zoom X Start:"), 0, 0); spec_roi_lay.addWidget(self.edit_spec_x_start, 0, 1)
        spec_roi_lay.addWidget(QLabel("Zoom X End:"), 1, 0); spec_roi_lay.addWidget(self.edit_spec_x_end, 1, 1)
        self.btn_refresh_spec = QPushButton("🔄 Refresh View")
        self.btn_refresh_spec.clicked.connect(self.update_spectral_alignment)
        spec_roi_lay.addWidget(self.btn_refresh_spec, 2, 0, 1, 2)
        spec_roi_group.setLayout(spec_roi_lay)
        spec_sidebar.addWidget(spec_roi_group)
        spec_sidebar.addStretch()
        spec_layout.addLayout(spec_sidebar, 1)

        spec_plot_container = QVBoxLayout()
        self.plot_spec_pre = pg.PlotWidget(title="RAW: Row Overlap (Smile Visualized)")
        self.plot_spec_post = pg.PlotWidget(title="CORRECTED: Row Overlap (Alignment Verified)")
        for p in [self.plot_spec_pre, self.plot_spec_post]:
            p.addLegend(); p.setLabel('left', 'Intensity'); p.setLabel('bottom', 'Spectral Pixel (X)')
            p.showGrid(x=True, y=True, alpha=0.3)
        spec_plot_container.addWidget(self.plot_spec_pre)
        spec_plot_container.addWidget(self.plot_spec_post)
        spec_layout.addLayout(spec_plot_container, 4)

        # ==========================================
        # TAB 4: RAW ANALYSIS
        # ==========================================
        raw_diag_layout = QVBoxLayout(self.tab_raw_diag)
        self.win_raw = pg.GraphicsLayoutWidget()
        self.view_raw = self.win_raw.addPlot(title="DBSCAN Clusters & Raw Parabolic Models")
        self.img_raw = pg.ImageItem()
        self.view_raw.addItem(self.img_raw)
        self.view_raw.invertY(True)
        self.view_raw.showGrid(x=True, y=True, alpha=0.3)
        raw_diag_layout.addWidget(self.win_raw)

    # --- METHODS ---

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', "Images (*.jpg *.png *.tif *.bmp)")
        if fname:
            self.raw_img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            self.img1.setImage(self.raw_img.T)
            self.img_raw.setImage(self.raw_img.T)
            h, w = self.raw_img.shape
            self.edit_x_end.setText(str(w))
            self.edit_spec_x_end.setText(str(w))
            self.view1.autoRange()
            self.view_raw.autoRange()
            self.update_spectral_alignment()

    def preview_smile(self):
        if self.raw_img is None: return
        h, w = self.raw_img.shape
        
        # Read GUI inputs
        x_start = int(self.edit_x_start.text())
        x_end = int(self.edit_x_end.text() if self.edit_x_end.text() != "Max" else w)
        thresh = int(self.edit_thresh.text())
        sigma = float(self.edit_sigma.text())
        num_samples = int(self.edit_sample_rows.text())
        dbscan_eps = float(self.edit_dbscan_eps.text())
        dbscan_min = int(self.edit_dbscan_min.text())
        min_pts_line = int(self.edit_min_pts_line.text())

        sample_rows = np.linspace(10, h-10, num_samples, dtype=int)
        raw_peaks_arr = detect_peaks_across_rows(self.raw_img, sample_rows, sigma, thresh, x_start, x_end)

        if raw_peaks_arr.size == 0:
            return

        self.view_raw.clear()
        self.view_raw.addItem(self.img_raw)
        self.view_raw.addItem(pg.ScatterPlotItem(x=raw_peaks_arr[:, 0], y=raw_peaks_arr[:, 1], symbol='o', size=4, brush='r'))

        labels = cluster_peak_lines(raw_peaks_arr, dbscan_eps, dbscan_min)
        candidate_lines = build_candidate_lines(raw_peaks_arr, labels, min_pts_line)
        if not candidate_lines:
            return

        valid_candidates = remove_outlier_lines(candidate_lines, sigma_threshold=2.5)
        y_plot = np.linspace(0, h-1, 100)

        self.view1.clear()
        self.view1.addItem(self.img1)
        self.detected_peaks_coords = refine_parabolic_peaks(self.raw_img, valid_candidates, w)

        for line in self.detected_peaks_coords:
            color = pg.intColor(line['label'], values=20)
            self.view1.addItem(pg.ScatterPlotItem(x=line['pts'][:, 0], y=line['pts'][:, 1], symbol='x', size=8, pen=pg.mkPen(color)))
            self.view1.addItem(pg.PlotCurveItem(np.polyval(line['coeffs'], y_plot), y_plot, pen=pg.mkPen(color, width=2)))

    def apply_unified_correction(self):
        """
        ฟังก์ชันคำนวณและประยุกต์ใช้การแก้ไข Smile และ Keystone 
        พร้อมบันทึกผลลัพธ์เป็นไฟล์ .npy สำหรับใช้งานภายนอก โดยอ้างอิงชื่อ Bin จาก Dropdown
        """
        if self.raw_img is None or not self.detected_peaks_coords: 
            return
            
        h, w = self.raw_img.shape
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            # 1. Global Trend Modeling (สร้างโมเดลความบิดเบี้ยวจากเส้นสเปกตรัมที่ตรวจจับได้)
            trend_a, trend_b = build_trend_models(self.detected_peaks_coords, h)
            mx, my = generate_smile_maps(w, h, trend_a, trend_b, self.keystone_model if self.check_keystone.isChecked() else None)
            self.corrected_img = remap_image(self.raw_img, mx, my, interpolation=cv2.INTER_CUBIC)
            self.img2.setImage(self.corrected_img.T)
            
            # --- ส่วนที่แก้ไขใหม่: ดึงค่า Bin จาก Dropdown ใน GUI ---
            # ต้องแน่ใจว่าได้เพิ่ม self.combo_bin ไว้ใน initUI() แล้วตามที่แนะนำไปก่อนหน้านี้
            selected_bin_text = self.combo_bin.currentText() # ได้ข้อความ เช่น "Bin4"
            bin_val = selected_bin_text.replace("Bin", "")   # ตัดคำว่า Bin ออก เหลือแค่ตัวเลข "4"
                
            # สร้างชื่อไฟล์แบบ Dynamic ตามที่ผู้ใช้เลือก
            map_x_name = f'smile_map_x_bin{bin_val}.npy'
            map_y_name = f'smile_map_y_bin{bin_val}.npy'
            
            import os
            np.save(map_x_name, mx)
            np.save(map_y_name, my)
            
            # แจ้งเตือนผู้ใช้ผ่าน Console
            print(f">>> [SYSTEM] User manually selected: {selected_bin_text}")
            print(f">>> [SYSTEM] Correction Maps saved to '{map_x_name}' and '{map_y_name}'")
            
            # 4. Validation (รันระบบตรวจสอบความถูกต้องหลังแก้ไข)
            self.run_diagnostic()
            self.update_spectral_alignment()
            
        except Exception as e:
            print(f">>> [ERROR] Correction failed: {e}")
            
        finally:
            QApplication.restoreOverrideCursor()
            
    def run_diagnostic(self):
        if self.corrected_img is None or not self.detected_peaks_coords:
            return
        sigma = float(self.edit_sigma.text())
        thresh = int(self.edit_thresh.text())
        num_samples = int(self.edit_sample_rows.text())
        min_pts_line = int(self.edit_min_pts_line.text())

        self.plot_morph.clear(); self.plot_resid.clear()

        diag_results = compute_diagnostic_results(
            self.corrected_img,
            self.detected_peaks_coords,
            sigma,
            thresh,
            num_samples,
            min_pts_line,
            dbscan_eps=10,
            dbscan_min=5,
        )
        if not diag_results:
            return

        y_range = np.linspace(0, self.corrected_img.shape[0] - 1, 100)
        for r in diag_results:
            color = pg.intColor(r['label'], values=20)
            self.plot_morph.addItem(pg.PlotCurveItem(np.polyval(r['orig_coeffs'], y_range), y_range, pen=pg.mkPen(color, style=Qt.DashLine)))
            self.plot_morph.addItem(pg.PlotCurveItem(np.polyval(r['post_coeffs'], y_range), y_range, pen=pg.mkPen(color, width=2)))
            self.plot_resid.addItem(pg.PlotCurveItem(r['post_res'], r['post_y'], pen=pg.mkPen(color, width=1.5)))

        diag_results.sort(key=lambda x: x['x'])
        stats = "<h3>Diagnostic Results</h3><hr>"
        for r in diag_results:
            color = pg.intColor(r['label'], values=20)
            pre_pp = np.ptp(r['orig_res'])
            post_pp = np.ptp(r['post_res'])
            imp = ((pre_pp - post_pp) / pre_pp * 100) if pre_pp > 0 else 0
            stats += f"<div style='margin-bottom:10px;'><b style='color:{color.name()}'>X: {r['x']:.1f}</b><br>" \
                     f"Pre a: {r['orig_coeffs'][0]:.2e} | Post a: {r['post_coeffs'][0]:.2e}<br>" \
                     f"Post P-P: {post_pp:.3f} px | <b style='color:#27ae60'>Imp: {imp:.1f}%</b></div><hr>"
        self.stats_text.setHtml(stats)

    def update_spectral_alignment(self):
        if self.raw_img is None: return
        h, w = self.raw_img.shape
        rows = {'Top': int(h*0.1), 'Mid': int(h*0.5), 'Btm': int(h*0.9)}
        clrs = {'Top': 'y', 'Mid': 'c', 'Btm': 'm'}
        
        try: 
            x_min = int(self.edit_spec_x_start.text())
            x_max = int(self.edit_spec_x_end.text() if self.edit_spec_x_end.text() != "Max" else w)
        except: x_min, x_max = 0, w

        max_int = 0
        for p in [self.plot_spec_pre, self.plot_spec_post]: p.clear()

        for label, y_idx in rows.items():
            raw_data = self.raw_img[y_idx, :].astype(float)
            self.plot_spec_pre.plot(raw_data, pen=pg.mkPen(clrs[label]), name=label)
            roi_raw = raw_data[max(0, x_min):min(w, x_max)]
            if len(roi_raw) > 0 and np.max(roi_raw) > max_int: max_int = np.max(roi_raw)

            if self.corrected_img is not None:
                cor_data = self.corrected_img[y_idx, :].astype(float)
                self.plot_spec_post.plot(cor_data, pen=pg.mkPen(clrs[label]), name=label)

        for p in [self.plot_spec_pre, self.plot_spec_post]:
            p.setXRange(x_min, x_max, padding=0)
            p.setYRange(0, max_int * 1.2)

    # --- KEYSTONE METHODS ---
    def load_keystone_reference(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Keystone Ref', '', "Images (*.jpg *.png *.tif *.bmp)")
        if fname:
            self.keystone_ref_img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if self.keystone_ref_img is not None: self.show_keystone_inspector()

    def show_keystone_inspector(self):
        h, w = self.keystone_ref_img.shape
        valid_edges = detect_keystone_edges(self.keystone_ref_img)
        if not valid_edges:
            return

        self.key_win = QDialog(self); self.key_win.setWindowTitle("Keystone Inspector")
        layout = QVBoxLayout(self.key_win); plot_win = pg.GraphicsLayoutWidget(); layout.addWidget(plot_win)
        self.key_plot = plot_win.addPlot(); self.key_plot.addItem(pg.ImageItem(self.keystone_ref_img.T)); self.key_plot.invertY(True)
        self.ui_lines = []
        for e in valid_edges:
            curve = pg.PlotCurveItem(np.array([0, w]), e['poly'](np.array([0, w])), 
                                     pen=pg.mkPen((0, 255, 0) if e['type'] == 1 else (255, 50, 50), width=3), clickable=True)
            curve.edge_data = e; curve.sigClicked.connect(self.remove_line_from_inspector)
            self.key_plot.addItem(curve); self.ui_lines.append(curve)

        btn = QPushButton("Confirm Keystone Model"); btn.clicked.connect(self.finalize_keystone_model); layout.addWidget(btn)
        self.key_win.resize(1000, 600); self.key_win.exec_()

    def remove_line_from_inspector(self, curve):
        self.key_plot.removeItem(curve)
        if curve in self.ui_lines: self.ui_lines.remove(curve)

    def finalize_keystone_model(self):
        if len(self.ui_lines) < 2:
            return
        edge_info = sorted([c.edge_data for c in self.ui_lines], key=lambda x: x['avg_y'])
        w = self.keystone_ref_img.shape[1]
        self.keystone_model = build_keystone_model(edge_info, w)
        self.key_win.accept()

    def on_image_double_clicked(self, event):
        pos = event.pos(); y_idx = int(pos.y())
        if self.raw_img is not None and 0 <= y_idx < self.raw_img.shape[0]: 
            self.show_spectrum_comparison(y_idx)

    def show_spectrum_comparison(self, y):
        diag = QDialog(self); diag.setWindowTitle(f"Spectrum Row: {y}"); layout = QVBoxLayout(diag)
        plot = pg.PlotWidget(); plot.addLegend(); plot.showGrid(x=True, y=True, alpha=0.3)
        plot.plot(self.raw_img[y,:].astype(float), pen=pg.mkPen('y', style=Qt.DashLine, width=1), name="Original")
        if self.corrected_img is not None:
            plot.plot(self.corrected_img[y,:].astype(float), pen=pg.mkPen('w', width=1.5), name="Corrected")
        layout.addWidget(plot); diag.resize(900, 400); diag.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv); window = PrecisionOpticsGUI(); window.show(); sys.exit(app.exec_())