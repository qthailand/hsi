# windows/main_window.py
"""HSI System main window with macOS-style title bar.

The window is a thin application shell: it owns the MDI area, the scan window and the
menus, and it hosts the feature modules through :mod:`windows.integrations` (listed in
:mod:`windows.registry`). It implements the *host* interface those integrations use
(``config``, ``frame_exchange``, ``scan_controller``, ``open_widget``, ``grab_for``,
``show_status``, ``warn``, ``error``, ``ensure_scan_window``, ``parent_widget``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QMdiArea,
    QMdiSubWindow,
    QFileDialog,
    QMessageBox,
    QAction,
    QMenuBar,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
)
from PyQt5.QtGui import QPixmap, QColor, QIcon, QBrush


import qtawesome as qta

from config import load_app_config
from modules.cube_io import as_analysis_cube, envi_scan_metadata, envi_value_fields, load_envi_cube_for_viewer
from modules.ui_kit.title_bar import IOSTitleBar
from modules.ui_kit import theme
from windows.scan_window import ScanWindow
from windows.registry import default_integrations
from controllers.scan_controller import ScanController
from modules.ui_kit.resizable_subwindow import ResizableMdiSubWindow

logger = logging.getLogger(__name__)

_STATUS_POLL_MS = 250


class MainWindow(QMainWindow):
    """The application shell (see the module docstring)."""

    def __init__(self, persist_settings: bool = False):
        """Build the shell and open the scan window.

        Args:
            persist_settings: Write camera/stage settings the user changes back to
                ``config/*.json`` for the next start. Only the application entry point
                (``main.py``) sets it, so tests and tools never overwrite the user's settings.
        """
        super().__init__()
        self.persist_settings = bool(persist_settings)

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setWindowTitle("HSI System")
        from modules.compute_runtime import initialize_compute
        self.compute_capabilities = initialize_compute()
        self.config = load_app_config()
        from modules.frame_exchange import FrameExchange
        self.frame_exchange = FrameExchange(self)
        self.frame_exchange.capture_requested.connect(self._request_snapshot)

        # Feature modules (Smile, wavelength calibration, ...): wiring lives in windows.integrations
        self.integrations = default_integrations(self)

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
        menubar.setStyleSheet(f"""
            QMenuBar {{
                background-color: {theme.SIDEBAR};
                color: #e0e0e0;
                padding: 2px 6px;
            }}
            QMenuBar::item:selected {{
                background-color: {theme.FIELD_HOVER};
                border-radius: 4px;
            }}
            QMenu {{
                background-color: {theme.SIDEBAR};
                color: #e0e0e0;
                border: 1px solid {theme.BORDER};
            }}
            QMenu::item:selected {{
                background-color: {theme.ACCENT};
                color: white;
            }}
        """)

        file_menu = menubar.addMenu("File")
        file_menu.addAction(self.scan_action)
        file_menu.addAction(self.open_envi_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        modules_menu = menubar.addMenu("Modules")
        for integration in self.integrations:
            for label, callback in integration.menu_entries():
                action = QAction(label, self)
                action.triggered.connect(callback)
                modules_menu.addAction(action)

        # ── 4. ★ ห่อ title bar + menubar ลงใน widget เดียว ★─
        self._top_container = QWidget()
        _top_vbox = QVBoxLayout(self._top_container)
        _top_vbox.setContentsMargins(0, 0, 0, 0)
        _top_vbox.setSpacing(0)

        self.title_bar = IOSTitleBar(self, title="HSI System")
        _top_vbox.addWidget(self.title_bar)
        _top_vbox.addWidget(menubar)          # ← menubar อยู่ใต้ title bar

        self.setMenuWidget(self._top_container)  # ← เซ็ตทั้งก้อน

        # ── 5. MDI area ──────────────────────────────────────
        self.mdi = QMdiArea()

        self.mdi.setViewMode(QMdiArea.SubWindowView)  # ✅ สำคัญมาก
        self.mdi.setOption(QMdiArea.DontMaximizeSubWindowOnActivation, True)
        self.mdi.setBackground(QBrush(QColor(theme.PAGE)))

        self.setCentralWidget(self.mdi)

        # ── 6. state + timers ────────────────────────────────
        self.scan_window = None
        self.scan_subwindow = None

        self._create_status_timer()
        self.create_scan_window()

    # ── host interface for the integrations ───────────────────────

    @property
    def scan_controller(self):
        """The current ``ScanController``, or ``None`` while no scan window is open."""
        return self.scan_window.controller if self.scan_window is not None else None

    @property
    def parent_widget(self) -> QWidget:
        """Widget to use as the parent of dialogs."""
        return self

    def ensure_scan_window(self):
        """Create the scan window if needed, or bring it forward."""
        self.create_scan_window()

    def open_widget(self, widget: QWidget, title: str, size=None) -> QMdiSubWindow:
        """Show ``widget`` in an MDI sub-window.

        Args:
            widget: The widget to embed; the sub-window deletes it when closed.
            title: Sub-window title.
            size: ``(width, height)`` in pixels, or ``None`` to fill the MDI work area.
        """
        sub = QMdiSubWindow()
        sub.setWidget(widget)
        sub.setWindowTitle(title)
        sub.setAttribute(Qt.WA_DeleteOnClose)
        self.mdi.addSubWindow(sub)
        sub.show()
        if size is None:
            sub.resize(self.mdi.size().width() - 20, self.mdi.size().height() - 20)
        else:
            sub.resize(*size)
        return sub

    def grab_for(self, widget) -> None:
        """Ask the camera for one oriented frame on behalf of ``widget`` (``begin_grab``/``snapshot_failed``)."""
        try:
            widget.begin_grab(self.frame_exchange.request())
        except Exception as exc:
            widget.snapshot_failed(None, str(exc))

    def show_status(self, text: str) -> None:
        """Show a message in the status bar."""
        self.statusBar().showMessage(text)

    def warn(self, title: str, text: str) -> None:
        """Show a warning dialog."""
        QMessageBox.warning(self, title, text)

    def error(self, title: str, text: str) -> None:
        """Show an error dialog."""
        QMessageBox.critical(self, title, text)

    # ── status bar polling ────────────────────────────────────────

    def _create_status_timer(self):
        # Sensor temperature: a permanent widget (qtawesome icon + text) at the right of the
        # status bar, shown while the active window reports one (``status_temperature()``)
        self.temp_status = QWidget()
        temp_lay = QHBoxLayout(self.temp_status)
        temp_lay.setContentsMargins(0, 0, 6, 0)
        temp_lay.setSpacing(3)
        self.temp_status_icon = QLabel()
        self.temp_status_icon.setPixmap(
            qta.icon("fa5s.thermometer-half", color="#ef5350").pixmap(14, 14))
        self.temp_status_text = QLabel()
        temp_lay.addWidget(self.temp_status_icon)
        temp_lay.addWidget(self.temp_status_text)
        self.statusBar().addPermanentWidget(self.temp_status)
        self.temp_status.hide()

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(_STATUS_POLL_MS)
        self._status_timer.timeout.connect(self._update_status_bar)
        self._status_timer.start()
        self.mdi.subWindowActivated.connect(
            lambda _: self._update_status_bar()
        )

    def _update_status_bar(self):
        sub = self.mdi.activeSubWindow()
        widget = sub.widget() if sub is not None else None
        temp = widget.status_temperature() if hasattr(widget, "status_temperature") else None
        if temp is not None:
            self.temp_status_text.setText(f"{temp:.1f}°C")
        self.temp_status.setVisible(temp is not None)
        if sub is None:
            self.statusBar().showMessage("")
            return
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

        # Re-read config/*.json so a camera changed in camera.json is used when Scan is reopened.
        if self.scan_window is None and getattr(self, "_scan_opened_once", False):
            try:
                self.config = load_app_config()
            except Exception as exc:
                logger.exception("Cannot reload configuration")
                QMessageBox.critical(self, "Configuration", f"Cannot reload config, keeping the previous one:\n{exc}")
        self._scan_opened_once = True
        logger.info("Opening scan window with camera_backend=%s", self.config.camera_backend)

        scan = ScanWindow(config=self.config)
        scan.cube_ready.connect(self._create_scan_cube_viewer)
        from windows.integrations import RadiometricIntegration
        for integration in self.integrations:
            if isinstance(integration, RadiometricIntegration):
                scan.calibration_requested.connect(
                    lambda cube, wl, integration=integration: integration.open_cube(
                        cube, wl, metadata=self._scan_export_metadata()))
        try:
            scan.controller = ScanController(scan, config=self.config,
                                             persist_settings=self.persist_settings)
        except Exception as exc:
            # No camera (or it failed to open): keep the main window usable, e.g. for Open ENVI.
            logger.exception("Cannot open the camera")
            scan.close()
            scan.deleteLater()
            QMessageBox.critical(self, "Camera Error", f"Cannot open the camera:\n{exc}")
            return
        scan.controller.preview_thread.snapshot_ready.connect(scan.snapshot_ready)
        scan.snapshot_ready.connect(self.frame_exchange.publish)
        scan.snapshot_failed.connect(self.frame_exchange.reject)
        for integration in self.integrations:
            integration.attach_scan(scan.controller)

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
                source_path=path,
                header_values=envi_value_fields(path),
                camera_bits=envi_scan_metadata(path),     # kept by a re-export
            )
        except Exception as e:
            logger.exception("Failed to open ENVI cube: %s", e)
            QMessageBox.critical(self, "Open ENVI Failed", str(e))

    def _scan_export_metadata(self):
        """The camera's bit depth and the last scan's metadata (``ScanController.last_scan_header``:
        start time, exposure, gain, temperatures, corrections, ...) for export headers, or None."""
        if self.scan_controller is None:
            return None
        camera_bits = None
        try:
            from modules.acquisition import camera_bit_info
            camera_bits = camera_bit_info(self.scan_controller.backend)
        except Exception:
            logger.debug("Camera bit depth unavailable for the export header", exc_info=True)
        return {**(camera_bits or {}), **self.scan_controller.last_scan_header}

    def _create_scan_cube_viewer(self, cube, wavelengths=None):
        """Open a viewer for the cube just scanned, carrying the scan's metadata to its export header."""
        self.create_cube_viewer(cube, wavelengths, camera_bits=self._scan_export_metadata())

    def create_cube_viewer(self, cube, wavelengths=None, title=None, camera_bits=None, source_path=None,
                           header_values=None):
        """Open a Cube Viewer; ``camera_bits`` (``camera_bit_info``) goes into the ENVI export header.

        ``source_path`` (the opened ENVI header) names the viewer's area spectra and its
        default spectra export file; None for a cube that has no file yet.
        """
        from modules.cube_viewer.window import CubeViewerWindow

        # Any source (BIP/BIL/BSQ file, or a scan) -> contiguous BIP for the analysis;
        # BSQ/BIL are transposed once here (about 1 s per 640 MiB), BIP passes through.
        if not cube.flags.c_contiguous:
            self.statusBar().showMessage("Preparing cube for analysis (transposing to BIP)\u2026")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                cube = as_analysis_cube(cube)
            finally:
                QApplication.restoreOverrideCursor()
                self.statusBar().clearMessage()

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
        if header_values:               # the cube's own 'data ignore value' / 'data full scale'
            cfg = {**cfg, **header_values}
        if camera_bits:
            cfg = {**cfg, **camera_bits}

        try:
            viewer = CubeViewerWindow(
                cube,
                wavelengths=wavelengths,
                resolution=self.config.resolution,
                display_mapper=mapper,
                config=cfg,
                camera_bin=self.scan_controller.camera_bin if self.scan_controller is not None else 1,
                source_path=source_path,
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

    def _request_snapshot(self, request_id):
        if self.scan_window is None:
            self.frame_exchange.reject(request_id, "Open Scan before grabbing a frame")
        else:
            self.scan_window.request_snapshot(request_id)

    def closeEvent(self, event):
        self.mdi.closeAllSubWindows()
        for sub in self.mdi.subWindowList():
            controller = getattr(sub.widget(), 'controller', None)
            stage = getattr(controller, 'stage', None)
            if stage is not None and stage.isRunning():
                event.ignore()
                return
        super().closeEvent(event)
