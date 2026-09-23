from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("HSI")


def wavelength_axis_from_frame_width(
    width: int,
    m: float,
    c: float,
) -> np.ndarray:
    pixel = np.arange(width)
    wavelength = m * pixel + c
    return wavelength.astype(np.float32)


def spectrum_from_corrected_row(
    corrected_frame: np.ndarray,
    selected_y: int,
) -> np.ndarray:
    from .image import raw_2d_frame

    frame = raw_2d_frame(corrected_frame)

    h, _ = frame.shape
    selected_y = max(0, min(h - 1, int(selected_y)))

    return np.ascontiguousarray(frame[selected_y, :])


def parse_envi_wavelengths(envi_image, cube: np.ndarray) -> np.ndarray | None:
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