"""Wavelength-calibration file layouts and the converter between them.

Two JSON layouts exist and are both accepted wherever "a wavelength calibration" is
loaded:

* **Profile** (written by ``modules.wavelength_calibration``): ``format_version`` 1,
  ``units`` ``"nm"``, ``m``, ``c``, ``pixels``, ``wavelengths``, ``camera_bin``,
  ``frame_shape``, ``selected_row``. ``m``/``c`` refer to the pixels of the image
  the profile was measured on (its own hardware bin).
* **Standards** (used by Smile quality analysis): ``m_nm_per_px``, ``c_nm``,
  ``standards`` = ``[{wavelength_nm, x_px, half_window_px?, blended?}]`` and an
  optional ``reference_row_y``.

Use :func:`normalize_wavelength_config` to obtain the *Standards* layout from either.
:class:`WavelengthCalibration` is the in-memory form of the *Profile* layout and owns
its ``save``/``load``; ``modules.wavelength_calibration`` re-exports it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def normalize_wavelength_config(config: dict) -> dict:
    """Return the *Standards* layout for a profile or an already-Standards dict.

    Args:
        config: Parsed JSON of either layout (see module docstring).

    Returns:
        ``config`` unchanged if it already has ``standards``; otherwise a new dict
        with ``m_nm_per_px``, ``c_nm``, ``standards``, ``reference_row_y`` (from
        ``selected_row``) and ``source``.

    Raises:
        ValueError: if the object is neither layout, lacks coefficients, is not in
            nm, or has no matched points.
    """
    if not isinstance(config, dict):
        raise ValueError("Wavelength calibration file must contain a JSON object.")
    if "standards" in config:
        missing = [k for k in ("m_nm_per_px", "c_nm") if k not in config]
        if missing:
            raise ValueError("Wavelength calibration is missing: " + ", ".join(missing))
        return config
    if "pixels" in config and "wavelengths" in config and "m" in config and "c" in config:
        if config.get("units", "nm") != "nm":
            raise ValueError("Wavelength calibration profile must be in nm.")
        pixels, wavelengths = list(config["pixels"]), list(config["wavelengths"])
        if not pixels or len(pixels) != len(wavelengths):
            raise ValueError("Wavelength calibration profile has no matched points.")
        result = {
            "m_nm_per_px": float(config["m"]),
            "c_nm": float(config["c"]),
            "standards": [{"wavelength_nm": float(w), "x_px": float(x)}
                          for x, w in zip(pixels, wavelengths)],
            "source": "wavelength_calibration_profile",
        }
        if config.get("selected_row") is not None:
            result["reference_row_y"] = int(config["selected_row"])
        return result
    raise ValueError("Unrecognized wavelength calibration file: expected 'standards' with "
                     "'m_nm_per_px'/'c_nm', or a profile saved by Wavelength Calibration.")


@dataclass(frozen=True)
class WavelengthCalibration:
    """Linear pixel-to-wavelength calibration (``wavelength = m * pixel + c``) and its provenance.

    Immutable. ``pixels``/``wavelengths`` are the matched points the fit used.

    Attributes:
        m: Slope in nm per pixel, for pixels of the image the profile was measured on
            (i.e. at its own hardware bin ``camera_bin``).
        c: Intercept in nm.
        r2: Coefficient of determination of the fit (dimensionless).
        rms_nm: Root-mean-square residual of the fit in nm.
        pixels: Matched pixel positions (spectral-column index of the image, px).
        wavelengths: Matching wavelengths in nm (same length as ``pixels``).
        camera_bin: Hardware bin of the image the pixels were measured on (>= 1).
        frame_shape: ``(slit rows, spectral columns)`` of the image, or ``()`` if unknown.
        selected_row: Slit row (0-based) whose spectrum was used.

    Thread-safety: instances are immutable and may be shared between threads.
    """
    m: float
    c: float
    r2: float
    rms_nm: float
    pixels: tuple
    wavelengths: tuple
    camera_bin: int = 1
    frame_shape: tuple = ()
    selected_row: int = 0

    @property
    def base_m(self):
        """HSI multiplies the unbinned slope by the current hardware bin."""
        return self.m / self.camera_bin

    def validate(self):
        """Check internal consistency.

        Raises:
            ValueError: ``camera_bin`` < 1, non-finite or non-positive ``m`` (or non-finite ``c``,
                ``r2``, ``rms_nm``), fewer than two matched points (or mismatched lengths),
                non-finite points, or duplicate pixel positions.
        """
        if self.camera_bin < 1 or not np.isfinite([self.m,self.c,self.r2,self.rms_nm]).all() or self.m <= 0:
            raise ValueError('Calibration requires a positive slope and hardware bin')
        if len(self.pixels)<2 or len(self.pixels)!=len(self.wavelengths):
            raise ValueError('Calibration requires at least two matched points')
        if not np.isfinite(self.pixels).all() or not np.isfinite(self.wavelengths).all():
            raise ValueError('Calibration points must be finite')
        if len(set(self.pixels))!=len(self.pixels):
            raise ValueError('Pixel coordinates must be distinct')

    def save(self,path):
        """Validate and write the profile as JSON (``format_version`` 1, ``units`` ``"nm"``).

        The file is written to ``<path>.tmp`` and then moved over ``path``, so an existing
        profile is never left half-written. See the package README for the layout.

        Args:
            path: Destination file path (str or ``os.PathLike``).

        Raises:
            ValueError: The profile is invalid (see :meth:`validate`) or contains NaN/inf.
            OSError: The file cannot be written.
        """
        self.validate()
        data=asdict(self);data['format_version']=1;data['units']='nm';data['stage']='post_dark_hot_pixel_orientation'
        dest=Path(path);tmp=dest.with_name(dest.name+'.tmp')
        tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(dest)

    @classmethod
    def load(cls,path):
        """Read and validate a profile written by :meth:`save`.

        Args:
            path: Profile JSON file path.

        Returns:
            WavelengthCalibration: The loaded, validated profile (lists become tuples).

        Raises:
            ValueError: ``format_version`` is not 1, ``units`` is not ``"nm"`` or the content
                fails :meth:`validate`.
            KeyError: A required field is missing.
            OSError: The file cannot be read.
        """
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('format_version')!=1 or data.get('units')!='nm':raise ValueError('Unsupported calibration profile')
        fields={k:data[k] for k in cls.__dataclass_fields__}
        for k in ['pixels','wavelengths','frame_shape']:fields[k]=tuple(fields[k])
        result=cls(**fields);result.validate();return result
