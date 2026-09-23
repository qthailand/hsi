from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import spectral as sp

from .wavelength import parse_envi_wavelengths

logger = logging.getLogger("HSI")


def create_pushbroom_cube_from_corrected(
    corrected_frames: list[np.ndarray],
) -> np.ndarray:
    stack = np.asarray(corrected_frames)

    if stack.ndim == 4:
        stack = stack[:, :, :, 0]

    if stack.ndim != 3:
        raise ValueError(f"Unsupported corrected frame stack shape: {stack.shape}")

    cube = stack.transpose(1, 0, 2)
    return np.ascontiguousarray(cube)


def load_envi_cube_for_viewer(
    hdr_path: str | Path,
) -> tuple[np.ndarray, np.ndarray | None]:
    hdr_path = Path(hdr_path)

    logger.info("Opening ENVI cube: %s", hdr_path)

    envi_image = sp.open_image(str(hdr_path))
    cube = envi_image.open_memmap(interleave="bip")

    if cube.ndim != 3:
        raise ValueError(f"ENVI cube must be 3D, got shape={cube.shape}")

    wavelengths = parse_envi_wavelengths(envi_image, cube)

    logger.info(
        "Loaded ENVI cube for viewer: shape=%s dtype=%s wavelengths=%s",
        cube.shape,
        cube.dtype,
        None if wavelengths is None else wavelengths.shape,
    )

    return cube, wavelengths