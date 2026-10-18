"""Shared data layouts and conventions that several modules must agree on.

Modules depend on ``modules.contracts`` instead of on each other for anything they
exchange: the cube container (:class:`CubeData`), the reference-row rule
(:func:`default_reference_row`), the wavelength-calibration file layouts
(:func:`normalize_wavelength_config`) and where the calibration of a camera setting is
stored (:class:`CameraSetting`, :class:`CalibrationStore`).

Data conventions:
    Cubes are ``(slit, scan, wavelength)``; spectral images are ``(slit, wavelength)``.
    Wavelengths are in nm, pixel positions in px (zero-based).

Threading:
    Everything here is pure and thread-safe.

Example:
    >>> import numpy as np
    >>> from modules.contracts import CubeData, default_reference_row, normalize_wavelength_config
    >>> cube = CubeData(np.zeros((4, 5, 6)), wavelengths=np.linspace(400, 900, 6))
    >>> cube.validate()
    >>> default_reference_row(552)
    276
    >>> cfg = normalize_wavelength_config({"m": 0.1, "c": 400.0, "pixels": [10, 20],
    ...                                    "wavelengths": [401.0, 402.0], "selected_row": 3})
    >>> cfg["m_nm_per_px"], cfg["standards"][0]["x_px"], cfg["reference_row_y"]
    (0.1, 10.0, 3)
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass(frozen=True)
class CubeData:
    """Cube and explicit axes. Array ownership remains with the caller.

    Construction performs no conversion, allocation or validation. Call validate()
    at an import/export boundary, not in the acquisition loop.

    Attributes:
        array: 3-D data array (any dtype); stored by reference, never copied.
        axes: Names of the three array axes in order; default ``("slit", "scan", "wavelength")``.
            Must be unique and contain ``"wavelength"``.
        wavelengths: Optional 1-D wavelength vector in nm, one value per sample of the
            wavelength axis.
        units: Unit of the array values (default ``"counts"``).

    Thread-safety: the container is frozen, but the array it references is not; do not
    modify the array while another thread reads it.
    """
    array: np.ndarray
    axes: Tuple[str, str, str] = ("slit", "scan", "wavelength")
    wavelengths: Optional[np.ndarray] = None
    units: str = "counts"

    def validate(self) -> None:
        """Check the axes and (optional) wavelength vector.

        Raises:
            ValueError: ``array`` is not 3-D, ``axes`` is not three unique names including
                ``"wavelength"``, or ``wavelengths`` is not 1-D with one value per spectral-axis sample.
        """
        if self.array.ndim != 3 or len(self.axes) != 3:
            raise ValueError("Cube requires three explicit axes")
        if len(set(self.axes)) != 3 or "wavelength" not in self.axes:
            raise ValueError("Cube axes must be unique and include wavelength")
        if self.wavelengths is not None:
            if self.wavelengths.ndim != 1 or len(self.wavelengths) != self.array.shape[self.axes.index("wavelength")]:
                raise ValueError("Wavelength count must match the spectral axis")

from .geometry import default_reference_row
from .wavelength import WavelengthCalibration, normalize_wavelength_config
from .calibration_set import CalibrationStore, CameraSetting, StoredResolution, StoredWavelength

__all__ = ["CubeData", "default_reference_row", "normalize_wavelength_config", "WavelengthCalibration",
           "CameraSetting", "CalibrationStore", "StoredWavelength", "StoredResolution"]
__requires__ = ()
