"""Wavelength calibration: linear pixel-to-wavelength fit, peak finding, Hg-Ar auto matching, widget.

Qt-free API (re-exported from :mod:`modules.wavelength_calibration.core`):
:class:`WavelengthCalibration` (profile, JSON ``save``/``load``), :func:`fit_wavelength`,
:func:`row_spectrum`, :func:`smooth_spectrum` (Savitzky-Golay), :func:`detect_peaks`,
:func:`refine_peaks`, :func:`auto_calibrate`, :class:`AutoCalibrationResult`, and the
line lists :data:`LEGACY_HGAR` and :data:`HGAR_LINES`. FWHM / spectral resolution lives in
``modules.spectral_resolution``.
The Qt panel :class:`WavelengthCalibrationWidget` is loaded lazily on first access,
so ``import modules.wavelength_calibration`` never imports PyQt5.

Data conventions:
    Spectral images are ``(slit rows, spectral columns)``; pixels are spectral-column
    indices at the image's hardware bin (the bin is read from the camera, never
    edited by the user); wavelengths are nm.

Threading:
    The Qt-free API is pure and thread-safe; the widget must be used from the GUI thread.

Example:
    >>> from modules.wavelength_calibration import fit_wavelength
    >>> cal = fit_wavelength([100, 400, 700], [500.0, 600.0, 700.0])
    >>> round(cal.rms_nm, 6)
    0.0
"""
from .core import WavelengthCalibration, fit_wavelength, row_spectrum, smooth_spectrum, detect_peaks, LEGACY_HGAR, HGAR_LINES, refine_peaks, auto_calibrate, AutoCalibrationResult

from .report import build_report, write_report
from .line_search import load_line_list, search_calibration

__all__ = ["WavelengthCalibration", "fit_wavelength", "row_spectrum", "smooth_spectrum",
           "detect_peaks", "LEGACY_HGAR", "HGAR_LINES", "refine_peaks", "auto_calibrate",
           "AutoCalibrationResult", "build_report", "write_report", "load_line_list", "search_calibration",
           "WavelengthCalibrationWidget"]
__requires__ = ("contracts", "image")
__widgets__ = ("WavelengthCalibrationWidget",)

def __getattr__(name):
    """Lazily import ``WavelengthCalibrationWidget`` so that importing the package never loads Qt.

    Raises:
        AttributeError: For any other unknown name.
    """
    if name=='WavelengthCalibrationWidget':
        from .widget import WavelengthCalibrationWidget
        return WavelengthCalibrationWidget
    raise AttributeError(name)
