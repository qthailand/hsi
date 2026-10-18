"""Assemble pushbroom cubes from corrected frames and open ENVI cubes lazily.

Data conventions:
    Corrected frame ``(slit, wavelength)``; cube ``(slit, scan, wavelength)``;
    wavelengths in nm (float32).

Threading:
    Plain functions without shared state; safe from any thread.  The cube returned
    by ``load_envi_cube_for_viewer`` is a read-only-style memory map: keep the
    file in place while it is in use.

Example:
    >>> import numpy as np
    >>> from modules.cube_io.core import create_pushbroom_cube_from_corrected
    >>> cube = create_pushbroom_cube_from_corrected([np.ones((2, 5), np.uint16)] * 3)
    >>> cube.shape, cube.dtype
    ((2, 3, 5), dtype('uint16'))
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from modules.spectral_processing.wavelength import parse_envi_wavelengths

logger = logging.getLogger("HSI")


def create_pushbroom_cube_from_corrected(
    corrected_frames: list[np.ndarray],
) -> np.ndarray:
    """Stack (slit, wavelength) frames into a (slit, scan, wavelength) cube.

    Args:
        corrected_frames: Non-empty list/tuple of ``(slit, wavelength)`` frames
            (``(slit, wavelength, C)`` frames use channel 0), one per scan
            position, all of the same shape; or an array stack
            ``(scan, slit, wavelength[, C])``.

    Returns:
        numpy.ndarray: New C-contiguous cube ``(slit, scan, wavelength)`` owned
        by the caller.  dtype is the common ``numpy.result_type`` of the frames.

    Raises:
        ValueError: the stack is not 3-D (after dropping a 4th channel axis),
            e.g. empty input.

    For a list of frames the cube is allocated once and filled frame by frame,
    so no intermediate (scan, slit, wavelength) stack is built (peak memory is
    the frames plus one cube instead of the frames plus two copies).
    """
    if isinstance(corrected_frames, (list, tuple)) and len(corrected_frames) > 0:
        frames = [np.asarray(f) for f in corrected_frames]
        if frames[0].ndim == 3:
            frames = [f[:, :, 0] for f in frames]
        if frames[0].ndim == 2 and all(f.shape == frames[0].shape for f in frames):
            dtype = np.result_type(*{f.dtype for f in frames})
            slit, wavelength = frames[0].shape
            cube = np.empty((slit, len(frames), wavelength), dtype=dtype)
            for i, frame in enumerate(frames):
                cube[:, i, :] = frame
            return cube

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
    """Open an ENVI cube as a memory map together with its wavelengths.

    Args:
        hdr_path: Path of the ENVI header (``.hdr``) as understood by
            ``spectral.open_image``.

    Returns:
        tuple: ``(cube, wavelengths)``.  ``cube`` is a memory-mapped array
        ``(lines, samples, bands)`` (BIP interleave; for files written by this
        package this is ``(slit, scan, wavelength)``) backed by the data file -
        not loaded into RAM and not a copy.  ``wavelengths`` is a new float32
        array ``(bands,)`` in nm, or ``None`` if absent/unusable.

    Raises:
        ValueError: the opened cube is not 3-D.
        ImportError: the ``spectral`` package is not installed.
        Exception: errors from ``spectral.open_image`` (missing or invalid file) propagate.
    """
    import spectral as sp

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