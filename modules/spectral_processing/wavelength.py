"""Wavelength-axis helpers for corrected frames and ENVI metadata.

Data conventions:
    A corrected frame is ``(slit, wavelength)`` (a trailing channel axis is
    tolerated and dropped).  Wavelengths are nm, float32.  Pixel ``p`` maps to
    ``m * p + c`` nm.

Threading:
    Pure functions, no shared state except the ``"HSI"`` logger; any thread.

Example:
    >>> import numpy as np
    >>> from modules.spectral_processing.wavelength import (
    ...     wavelength_axis_from_frame_width, spectrum_from_corrected_row)
    >>> wavelength_axis_from_frame_width(4, 0.5, 400.0).tolist()
    [400.0, 400.5, 401.0, 401.5]
    >>> spectrum_from_corrected_row(np.arange(12).reshape(3, 4), 1).tolist()
    [4, 5, 6, 7]
"""
from __future__ import annotations

import logging

import numpy as np

from modules.image import raw_2d_frame

logger = logging.getLogger("HSI")


def wavelength_axis_from_frame_width(
    width: int,
    m: float,
    c: float,
) -> np.ndarray:
    """Linear wavelength axis ``m * pixel + c``.

    Args:
        width: Number of spectral pixels.
        m: Slope, nm per pixel.
        c: Intercept, nm (wavelength of pixel 0).

    Returns:
        numpy.ndarray: New float32 array of shape ``(width,)`` in nm.
    """
    pixel = np.arange(width)
    wavelength = m * pixel + c
    return wavelength.astype(np.float32)


def spectrum_from_corrected_row(
    corrected_frame: np.ndarray,
    selected_y: int,
) -> np.ndarray:
    """Extract one slit row (a spectrum) from a corrected frame.

    Args:
        corrected_frame: ``(slit, wavelength)`` frame (or ``(slit, wavelength, C)``,
            channel 0 is used).
        selected_y: Slit (row) index; clamped into ``[0, slit - 1]``.

    Returns:
        numpy.ndarray: C-contiguous 1-D array ``(wavelength,)`` in the frame's
        dtype.  It may be a view of ``corrected_frame``; copy before mutating.

    Raises:
        ValueError: if the frame is not 2-D or 3-D.
    """
    frame = raw_2d_frame(corrected_frame)

    h, _ = frame.shape
    selected_y = max(0, min(h - 1, int(selected_y)))

    return np.ascontiguousarray(frame[selected_y, :])


def parse_envi_wavelengths(envi_image, cube: np.ndarray) -> np.ndarray | None:
    """Read the band wavelengths of an ENVI image as nm.

    Args:
        envi_image: Object with a ``metadata`` dict (e.g. ``spectral.io.envi``
            image).  The keys ``wavelength``/``Wavelength``/``wavelengths``/
            ``Wavelengths`` are tried; the value may be a list or a
            ``{a, b, ...}`` string, entries may carry ``nm``/``um`` suffixes
            (the suffix is stripped, not converted).
        cube: Cube ``(lines, samples, bands)``; its last axis length must equal
            the number of wavelengths.

    Returns:
        numpy.ndarray | None: New float32 array ``(bands,)`` in nm, multiplied by
        1000 when ``wavelength units`` says micrometres; ``None`` (with a log
        warning) if the field is missing, unparsable, ``cube`` is not 3-D or the
        counts differ.  Never raises for bad metadata.
    """
    metadata = getattr(envi_image, "metadata", {}) or {}

    wavelengths = (
        metadata.get("wavelength")
        or metadata.get("Wavelength")
        or metadata.get("wavelengths")
        or metadata.get("Wavelengths")
    )

    if wavelengths is None:
        logger.warning("ENVI metadata has no wavelength field")
        return None

    if isinstance(wavelengths, str):
        text = wavelengths.strip().strip("{}")
        wavelengths = [v.strip() for v in text.split(",") if v.strip()]

    try:
        values = []
        for value in wavelengths:
            s = str(value).strip()
            s = s.replace("{", "").replace("}", "")
            s = s.replace("nm", "").replace("NM", "")
            s = s.replace("µm", "").replace("um", "")
            values.append(float(s))

        arr = np.asarray(values, dtype=np.float32)

    except Exception as e:
        logger.warning("Failed to parse ENVI wavelengths: %s", e)
        return None

    if cube.ndim != 3:
        return None

    bands = cube.shape[2]

    if arr.shape[0] != bands:
        logger.warning(
            "Wavelength count mismatch: wavelengths=%d cube_bands=%d",
            arr.shape[0],
            bands,
        )
        return None

    unit = (
        metadata.get("wavelength units")
        or metadata.get("Wavelength Units")
        or metadata.get("wavelength_units")
        or ""
    )

    unit = str(unit).lower().strip()

    if unit in ("um", "µm") or "microm" in unit:
        arr = arr * 1000.0

    return arr