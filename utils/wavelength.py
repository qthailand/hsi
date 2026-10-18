"""Compatibility imports; implementations live in reusable packages."""

from modules.spectral_processing.wavelength import wavelength_axis_from_frame_width, spectrum_from_corrected_row, parse_envi_wavelengths

__all__ = ['wavelength_axis_from_frame_width', 'spectrum_from_corrected_row', 'parse_envi_wavelengths']
