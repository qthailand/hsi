"""How each feature module is plugged into the HSI main window.

Every ``modules.*`` widget is self-contained: it only emits signals and receives data
through methods. The classes here are the *only* place that knows how such a widget is
wired to the running application (camera frames, scan controller, configuration, status
messages). The main window itself stays a thin shell: it builds its *Modules* menu from
:func:`windows.registry.default_integrations` and calls :meth:`Integration.attach_scan`
whenever a scan controller is (re)created.

The host (normally :class:`windows.main_window.MainWindow`) must provide:

``config``
    The ``AppConfig`` in use.
``frame_exchange``
    A :class:`modules.frame_exchange.FrameExchange` (one-shot "grab a frame" requests).
``scan_controller``
    The current ``ScanController`` or ``None`` while no scan window is open.
``ensure_scan_window()``
    Create or raise the scan window.
``open_widget(widget, title, size=None)``
    Show ``widget`` in a sub-window (``size=None`` fills the work area).
``grab_for(widget)``
    Start a one-shot frame grab for a widget that has ``begin_grab`` / ``snapshot_failed``.
``show_status(text)``, ``warn(title, text)``, ``error(title, text)``
    User feedback.
``parent_widget``
    A QWidget usable as a dialog parent.

Calibrations per camera setting: Smile, Wavelength and FWHM share one
:class:`modules.contracts.CalibrationStore` (:func:`calibration_store`) and read the camera's
current :class:`~modules.contracts.CameraSetting` from the scan controller
(:func:`camera_setting`), so each module finds the files of the active camera/bin itself.
A saved FWHM is re-read by the live session (bin width when ``resolution_source`` is ``"fwhm"``).
"""
from __future__ import annotations

import logging
from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)


def calibration_store(host):
    """The calibration store of the configured camera backend.

    Wavelength profiles: ``<config>/calibration/wavelength/<backend>/wavelength_bin<N>.json``
    (FWHM: ``resolution_bin<N>.json`` in the same folder);
    smile models: ``config.smile_map_folder`` (already per backend), ``smile_model_bin<N>.npz``.
    """
    from pathlib import Path
    from modules.contracts import CalibrationStore
    config = host.config
    store = getattr(config, "_config_store", None)
    folder = (store.wavelength_folder(config.camera_backend) if store else
              Path("config") / "calibration" / "wavelength" / str(config.camera_backend).lower())
    return CalibrationStore(folder, config.smile_map_folder)


def camera_setting(host):
    """The connected camera's current ``CameraSetting``, or ``None`` while no scan window is open."""
    controller = host.scan_controller
    return controller.camera_setting() if controller is not None else None


class Integration:
    """Base class: one feature module wired into the application.

    Args:
        host: The application shell (see the module docstring for what it provides).
    """

    def __init__(self, host):
        self.host = host

    def menu_entries(self) -> List[Tuple[str, Callable]]:
        """Return ``[(menu label, callable)]`` for the *Modules* menu (callables accept ``*args``)."""
        return []

    def attach_scan(self, controller) -> None:
        """Hook called after a ``ScanController`` is created (connect to its worker signals)."""


class RadiometricIntegration(Integration):
    """Offline BSQ/radiometric calibration window (``modules.radiometric``)."""

    def menu_entries(self):
        return [("Open BSQ Generator", self.open)]

    def open(self, *_):
        """Open the calibration window, filling the work area."""
        from modules.radiometric import CalibrationWindow
        self.host.open_widget(CalibrationWindow(), "BSQ Generation (ENVI Input Mode)", None)

    def open_cube(self, cube, wavelengths=None):
        """Open the calibration window on a scanned cube (Scan window's *Open Calibration*)."""
        try:
            from modules.radiometric import CalibrationWindow
            widget = CalibrationWindow()
            widget.load_from_memory(cube_array=cube, wavelengths=wavelengths, name="ScanResult")
            self.host.open_widget(widget, "BSQ Generation (Scan Result)", None)
        except Exception as exc:
            logger.exception("Cannot open calibration on the scanned cube")
            self.host.error("Open Calibration", str(exc))


class SmileIntegration(Integration):
    """Smile analysis window and the *Load Smile Model* action (``modules.smile``)."""

    def menu_entries(self):
        return [("Open Smile Analysis", self.open), ("Load Smile Model", self.load_model)]

    def open(self, *_):
        """Open Smile Analysis; *Grab* asks for one oriented frame, *Apply* stores and loads the model."""
        from modules.smile import SmileCorrectionWidget
        widget = SmileCorrectionWidget()
        widget.set_grab_available(True)
        widget.set_calibration_source(calibration_store(self.host), lambda: camera_setting(self.host))
        widget.applyRequested.connect(self._apply_model)
        widget.grabRequested.connect(lambda: self.host.grab_for(widget))
        self.host.frame_exchange.frame_ready.connect(widget.accept_snapshot)
        self.host.frame_exchange.failed.connect(widget.snapshot_failed)
        self.host.open_widget(widget, "Smile Analysis / Calibration", (1100, 800))

    def _apply_model(self, path):
        """Load a model the widget just stored for the current setting into the live session."""
        try:
            controller = self.host.scan_controller
            if controller is None:
                raise RuntimeError("Model stored, but no scan window is open to load it into")
            controller.load_smile_model(path)
            self.host.show_status(f"Smile model applied: {path}")
        except Exception as exc:
            self.host.warn("Smile Model", str(exc))

    def load_model(self, *_):
        """Ask for a Smile model file and queue it in the acquisition worker."""
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self.host.parent_widget, "Load Smile Model", "", "Smile calibration (*.npz)")
        if not path:
            return
        try:
            controller = self.host.scan_controller
            if controller is None:
                raise RuntimeError("Open the scan window first")
            controller.load_smile_model(path)
        except Exception as exc:
            self.host.error("Smile Model", str(exc))


class WavelengthIntegration(Integration):
    """Wavelength calibration panel: grab frames, apply to the live session, save as startup default."""

    def __init__(self, host):
        super().__init__(host)
        self._pending_default = None
        self._pending_store = {}        # id(profile) -> (profile, widget, setting) awaiting the worker

    def menu_entries(self):
        return [("Wavelength Calibration", self.open)]

    def attach_scan(self, controller):
        worker = controller.preview_thread
        worker.wavelength_failed.connect(self._on_failed)
        worker.wavelength_applied.connect(self._on_applied)

    def open(self, *_):
        """Open the calibration widget wired to the camera (bin, grabs) and the scan controller."""
        from modules.wavelength_calibration import WavelengthCalibrationWidget
        widget = WavelengthCalibrationWidget()
        widget.set_host_available(True)
        widget.set_bin_provider(lambda: self.host.scan_controller.camera_bin if self.host.scan_controller else None)
        widget.set_calibration_source(calibration_store(self.host), lambda: camera_setting(self.host))
        widget.grabRequested.connect(lambda: self.host.grab_for(widget))
        widget.applyRequested.connect(lambda profile: self._apply(profile, widget))
        widget.saveDefaultRequested.connect(lambda profile: self._save_default(profile, widget))
        controller = self.host.scan_controller
        if controller is not None:
            controller.preview_thread.wavelength_applied.connect(widget.calibration_applied)
        self.host.frame_exchange.frame_ready.connect(widget.accept_snapshot)
        self.host.frame_exchange.failed.connect(widget.snapshot_failed)
        self.host.open_widget(widget, "Wavelength Calibration", (1200, 900))

    def _apply(self, profile, widget=None):
        """Apply to the live session; once the worker accepts it, store it for the camera setting."""
        try:
            controller = self.host.scan_controller
            if controller is None:
                raise RuntimeError("Open Scan first")
            setting = widget.setting if widget is not None and widget.setting is not None else controller.camera_setting()
            problems = controller.camera_setting().mismatch(setting)
            if problems:
                raise RuntimeError("The camera is now at a different setting (" + "; ".join(problems) + "); grab a new frame")
            controller.apply_wavelength_calibration(profile)
            self._pending_store[id(profile)] = (profile, widget, setting)
            self.host.show_status("Wavelength calibration queued for the next frame")
            return True
        except Exception as exc:
            self.host.warn("Wavelength Calibration", str(exc))
            return False

    def _save_default(self, profile, widget=None):
        try:
            if self.host.scan_controller is None:
                raise RuntimeError("Open Scan first")
            if self._pending_default is not None:
                raise RuntimeError("A default calibration save is already pending")
            if self._apply(profile, widget):
                self._pending_default = profile
        except Exception as exc:
            self.host.warn("Save Default Calibration", str(exc))

    def _on_failed(self, profile, message):
        self._pending_store.pop(id(profile), None)
        if profile is self._pending_default:
            self._pending_default = None
        self.host.warn("Wavelength Calibration", message)

    def _store_applied(self, profile):
        """Write an applied profile to the calibration store for the setting it was measured at."""
        entry = self._pending_store.pop(id(profile), None)
        if entry is None or entry[0] is not profile:
            return
        _, widget, setting = entry
        try:
            path = calibration_store(self.host).save_wavelength(profile, setting)
            logger.info("Wavelength calibration stored for %s: %s", setting.label(), path)
            self.host.show_status(f"Wavelength calibration applied and saved for {setting.label()}")
            if widget is not None:
                try:
                    widget.calibration_stored(profile, str(path), setting.label())
                except RuntimeError:            # widget already closed
                    pass
        except Exception as exc:
            self.host.warn("Wavelength Calibration", f"Applied for this session, but not saved for the setting: {exc}")

    def _on_applied(self, profile):
        self._store_applied(profile)
        if profile is not self._pending_default:
            return
        self._pending_default = None
        config = self.host.config
        try:
            config._config_store.save_wavelength_default(profile, config)
            config.default_bin = profile.camera_bin
            config._wavelength_profile = profile
            self.host.show_status("Wavelength calibration saved as startup default")
        except Exception as exc:
            self.host.warn("Save Default Calibration", f"Applied for this session, but default was not saved: {exc}")


class SpectralResolutionIntegration(Integration):
    """FWHM / spectral resolution window (``modules.spectral_resolution``)."""

    def menu_entries(self):
        return [("FWHM / Spectral Resolution", self.open)]

    def open(self, *_):
        """Open the panel: *Grab* asks for one frame; calibrations of the same setting load by themselves."""
        from modules.spectral_resolution import SpectralResolutionWidget
        widget = SpectralResolutionWidget()
        widget.set_grab_available(True)
        widget.set_calibration_source(calibration_store(self.host), lambda: camera_setting(self.host))
        widget.grabRequested.connect(lambda: self.host.grab_for(widget))
        widget.resolutionStored.connect(self._on_stored)
        self.host.frame_exchange.frame_ready.connect(widget.accept_snapshot)
        self.host.frame_exchange.failed.connect(widget.snapshot_failed)
        self.host.open_widget(widget, "FWHM / Spectral Resolution", (1200, 850))

    def _on_stored(self, path):
        """Let the live session re-read the FWHM it bins with (applied when no scan is running)."""
        controller = self.host.scan_controller
        if controller is not None:
            controller.reload_spectral_resolution()
        self.host.show_status(f"FWHM saved: {path}")


class ReferenceROIIntegration(Integration):
    """Visual reference-ROI selection: a preview scan feeds the widget, the saved profile drives the live overlay."""

    def __init__(self, host):
        super().__init__(host)
        from modules.reference_roi import ReferenceROIStore
        self.store = ReferenceROIStore(host.config._config_store.root)
        self._pending = None          # (request_id, widget) of the running preview scan
        self._request_id = 0
        try:
            self.profile = self.store.load_active()
        except Exception:
            logger.exception('Cannot load active reference ROI profile')
            self.profile = None

    def menu_entries(self):
        return [('Reference ROI Selection', self.open)]

    def attach_scan(self, controller):
        controller.set_reference_roi_profile(self.profile)
        controller.reference_preview_ready = self._ready
        controller.reference_preview_failed = self._failed

    def open(self, *_):
        """Open the selection widget and connect its preview/save/cancel requests."""
        from modules.reference_roi import ReferenceROIWidget
        widget = ReferenceROIWidget()
        profile = self.profile
        if profile:
            widget.status.setText(f'Active: {profile.name}, rows [{profile.y_start}, {profile.y_end}). Run Preview Scan to select a new band.')
        widget.previewRequested.connect(lambda: self._request_preview(widget))
        widget.cancelRequested.connect(lambda: self._cancel_preview(widget))
        widget.saveRequested.connect(lambda profile: self._save_profile(widget, profile))
        widget.deactivateRequested.connect(lambda: self._save_profile(widget, None))
        widget.destroyed.connect(lambda: self._cancel_preview(widget))
        self.host.open_widget(widget, 'Reference ROI / Visual Selection', (900, 650))

    def _request_preview(self, widget):
        try:
            if self._pending is not None:
                raise RuntimeError('Another reference preview is already running')
            self.host.ensure_scan_window()
            self._request_id += 1
            self._pending = (self._request_id, widget)
            widget.status.setText('Scanning with the travel/speed configured in Scan Window; waiting for stage completion…')
            self.host.scan_controller.start_reference_preview(self._request_id)
        except Exception as exc:
            if self._pending and self._pending[1] is widget:
                self._pending = None
            widget.fail_preview(str(exc))

    def _cancel_preview(self, widget):
        if self._pending and self._pending[1] is widget:
            self._pending = None
            if self.host.scan_controller is not None:
                self.host.scan_controller.stop_scan_manually()
            try:
                widget.fail_preview('Preview cancelled. Wait for the stage to stop before scanning again.')
            except RuntimeError:
                pass

    def _ready(self, request_id, image, context):
        if self._pending and self._pending[0] == request_id:
            _, widget = self._pending
            self._pending = None
            widget.accept_preview(image, context)

    def _failed(self, request_id, message):
        if self._pending and self._pending[0] == request_id:
            _, widget = self._pending
            self._pending = None
            widget.fail_preview(message)

    def _save_profile(self, widget, profile):
        try:
            controller = self.host.scan_controller
            if profile is None:
                self.store.clear_active()
                saved = None
            else:
                if controller is None or not profile.matches(controller.reference_roi_context(getattr(controller, '_last_spatial_height', 0))):
                    raise RuntimeError('Camera configuration changed; run Preview Scan again')
                saved = self.store.save_active(profile)
            self.profile = saved
            if controller is not None:
                controller.set_reference_roi_profile(saved)
            widget.status.setText(f'Saved active profile: {saved.name}, rows [{saved.y_start}, {saved.y_end})' if saved else 'Reference ROI profile deactivated')
        except Exception as exc:
            widget.status.setText(f'Could not save profile: {exc}')
