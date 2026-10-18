"""Qt-independent spectral processing: wavelength binning and wavelength helpers.

Public API:

* ``LambdaBinner`` - bins raw spectrograms onto a uniform wavelength grid
  (Numba, uint16 in / uint16 out).
* ``wavelength_axis_from_frame_width`` - linear pixel -> nm axis.
* ``spectrum_from_corrected_row`` - one spectrum from a corrected frame.
* ``parse_envi_wavelengths`` - wavelength list (nm) from ENVI metadata.

Data conventions:
    A raw spectrogram is ``(frames, spatial, spectral)`` (the last axis is the
    sensor spectral pixel axis).  A single sensor frame / spectral image is
    ``(slit, wavelength)``.  A cube is ``(slit, scan, wavelength)``.
    Wavelengths are in nm (float32); pixel -> wavelength is ``m * pixel + c``.

Threading:
    Everything is Qt-free and may be called from any thread.  A
    ``LambdaBinner`` instance must not be re-fitted while another thread
    transforms with it, and ``transform`` sets the process-wide Numba thread
    count.

Example:
    >>> import numpy as np
    >>> from modules.spectral_processing import LambdaBinner
    >>> raw = np.ones((2, 3, 100), dtype=np.uint16)        # (frames, spatial, spectral)
    >>> binner = LambdaBinner(m=0.29, c=400.0, step=2.0).fit_from(raw)
    >>> binner.transform(raw).shape == (2, 3, binner.n_bins)
    True
"""
from .binning import LambdaBinner
from .wavelength import wavelength_axis_from_frame_width, spectrum_from_corrected_row, parse_envi_wavelengths

__all__ = (
    "LambdaBinner",
    "wavelength_axis_from_frame_width",
    "spectrum_from_corrected_row",
    "parse_envi_wavelengths",
)
__requires__ = ("image",)
