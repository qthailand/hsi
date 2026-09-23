# windows/main_window.py
"""HSI System main window with macOS-style title bar."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QMdiArea,
    QMdiSubWindow,
    QFileDialog,
    QMessageBox,
    QAction,
    QMenuBar,
    QToolBar,
    QVBoxLayout
)
from PyQt5.QtGui import QPixmap, QColor, QIcon


from config import load_app_config
from utils.cube import load_envi_cube_for_viewer
from widgets.title_bar import IOSTitleBar
from windows.scan_window import ScanWindow
from controllers.scan_controller import ScanController
from widgets.resizable_subwindow import ResizableMdiSubWindow
from modules.bsq_calibration import CalibrationWindow

logger = logging.getLogger(__name__)

_STATUS_POLL_MS = 250


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setWindowTitle("HSI System")
        self.config = load_app_config("backend.json")

        # ── 1. actions ───────────────────────────────────────
        self.scan_action = QAction("Scan", self)
        self.scan_action.triggered.connect(self.create_scan_window)

        self.open_envi_action = QAction("Open ENVI", self)
        self.open_envi_action.triggered.connect(self.open_envi_cube)

        self.exit_action = QAction("Exit", self)
        self.exit_action.triggered.connect(self.close)

        # ── 2. menubar (สร้างก่อน) ────────────────────────────
        menubar = QMenuBar(self)
        menubar.setNativeMenuBar(False)
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #2b2b2b;
                color: #e0e0e0;
                padding: 2px 6px;
            }
            QMenuBar::item:selected {
                background-color: #4a4a4a;
                border-radius: 4px;
            }
            QMenu {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #555;
            }
            QMenu::item:selected {
                background-color: #505050;
            }
        """)

        file_menu = menubar.addMenu("File")
        file_menu.addAction(self.scan_action)
        file_menu.addAction(self.open_envi_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        modules_menu = menubar.addMenu("Modules")
        act_open_a = QAction("Open BSQ Generator", self)
        act_open_a.triggered.connect(self.open_module_a)
        modules_menu.addAction(act_open_a)

        # # ── 3. toolbar ───────────────────────────────────────
        # toolbar = QToolBar("Quick Actions", self)
        # toolbar.setMovable(False)
        # toolbar.addAction(self.scan_action)
        # toolbar.addAction(self.open_envi_action)
        # self.addToolBar(toolbar)

        # ── 4. ★ ห่อ title bar + menubar ลงใน widget เดียว ★─
        self._top_container = QWidget()
        _top_vbox = QVBoxLayout(self._top_container)
        _top_vbox.setContentsMargins(0, 0, 0, 0)
        _top_vbox.setSpacing(0)

        self.title_bar = IOSTitleBar(self, title="HSI System")
        _top_vbox.addWidget(self.title_bar)
        _top_vbox.addWidget(menubar)          # ← menubar อยู่ใต้ title bar

        self.setMenuWidget(self._top_container)  # ← เซ็ตทั้งก้อน
        # ไม่ต้องเรียก inject_menubar อีก

        # ── 5. MDI area ──────────────────────────────────────
        self.mdi = QMdiArea()

        self.mdi.setViewMode(QMdiArea.SubWindowView)  # ✅ สำคัญมาก
        self.mdi.setOption(QMdiArea.DontMaximizeSubWindowOnActivation, True)

        self.setCentralWidget(self.mdi)

        # ── 6. state + timers ────────────────────────────────
        self.scan_window = None
        self.scan_subwindow = None

        self._create_status_timer()
        self.create_scan_window()

    # ── status bar polling ────────────────────────────────────────

    def _create_status_timer(self):
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(_STATUS_POLL_MS)
        self._status_timer.timeout.connect(self._update_status_bar)
        self._status_timer.start()
        self.mdi.subWindowActivated.connect(
            lambda _: self._update_status_bar()
        )

    def _update_status_bar(self):
        sub = self.mdi.activeSubWindow()
        if sub is None:
            self.statusBar().showMessage("")
            return
        widget = sub.widget()
        if hasattr(widget, "status_info"):
            self.statusBar().showMessage(widget.status_info())
        else:
            self.statusBar().showMessage("")

    # ── scan window ───────────────────────────────────────────────

    def create_scan_window(self):
        if self.scan_subwindow is not None:
            try:
                self.mdi.setActiveSubWindow(self.scan_subwindow)
                self.scan_subwindow.showNormal()
                self.scan_subwindow.raise_()
                self.scan_subwindow.activateWindow()
                return
            except RuntimeError as r:
                self.scan_subwindow = None
                self.scan_window = None
                logger.error(r)

        scan = ScanWindow(config=self.config)
        scan.cube_ready.connect(self.create_cube_viewer)
        scan.controller = ScanController(scan, config=self.config)

        sub = ResizableMdiSubWindow()
        sub.setWidget(scan)
        sub.setWindowTitle("Scan / Live Preview")
        sub.setAttribute(Qt.WA_DeleteOnClose, True)
        sub.destroyed.connect(
            lambda: setattr(self, "scan_window", None)
            or setattr(self, "scan_subwindow", None)
        )

        self.mdi.addSubWindow(sub)
        sub.setWindowIcon(self.make_icon("#ef5350"))   # แดง
        sub.showMaximized()

        self.scan_window = scan
        self.scan_subwindow = sub

    # ── ENVI loader ───────────────────────────────────────────────

    def open_envi_cube(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ENVI Header", "",
            "ENVI Header (*.hdr);;All Files (*)",
        )
        if not path:
            return
        try:
            cube, wavelengths = load_envi_cube_for_viewer(path)
            self.create_cube_viewer(
                cube=cube, wavelengths=wavelengths,
                title=f"ENVI Cube {Path(path).name}",
            )
        except Exception as e:
            logger.exception("Failed to open ENVI cube: %s", e)
            QMessageBox.critical(self, "Open ENVI Failed", str(e))

    def create_cube_viewer(self, cube, wavelengths=None, title=None):
        from windows.cube_viewer import CubeViewerWindow

        mapper = None
        if self.scan_window is not None:
            try:
                mapper = self.scan_window.display_controls.current_mapper()
            except Exception:
                pass

        # ★ แปลง AppConfig → dict
        cfg = self.config
        if not isinstance(cfg, dict):
            cfg = {}
            for key in ('m', 'c', 'pixel_size', 'resolution',
                        'raw_inverted', 'default_bin',
                        'dark_frame_folder', 'dark_frame_n_frames',
                        'camera_backend', 'default_gain',
                        'default_exposure_ms'):
                if hasattr(self.config, key):
                    cfg[key] = getattr(self.config, key)

        try:
            viewer = CubeViewerWindow(
                cube,
                wavelengths=wavelengths,
                resolution=self.config.resolution,
                display_mapper=mapper,
                config=cfg,
                camera_bin=getattr(self.scan_window, '_camera_bin', 1),
            )
        except Exception as e:
            QMessageBox.critical(self, "Viewer Error", str(e))
            return

        sub = QMdiSubWindow()
        sub.setWidget(viewer)
        sub.setWindowTitle(title or f"Cube Viewer {cube.shape}")
        sub.setAttribute(Qt.WA_DeleteOnClose)

        self.mdi.addSubWindow(sub)
        sub.resize(1000, 800)
        sub.show()
        sub.raise_()
        sub.activateWindow()

    def make_icon(self, color: str) -> QIcon:
        px = QPixmap(16, 16)
        px.fill(QColor(color))
        return QIcon(px)

    def open_module_a(self):
        """สร้าง CalibrationWindow เป็น SubWindow ข้างใน MdiArea"""

        # สร้าง Widget จาก Module
        widget = CalibrationWindow()

        # ห่อเป็น SubWindow
        sub = QMdiSubWindow()
        sub.setWidget(widget)
        sub.setWindowTitle("BSQ Generation (ENVI Input Mode)")
        sub.setAttribute(Qt.WA_DeleteOnClose)

        self.mdi.addSubWindow(sub)
        sub.show()

        # ปรับขนาดเริ่มต้นให้เต็ม MdiArea
        sub.resize(self.mdi.size().width() - 20, self.mdi.size().height() - 20)

    def closeEvent(self, event):
        self.mdi.closeAllSubWindows()
        super().closeEvent(event)
