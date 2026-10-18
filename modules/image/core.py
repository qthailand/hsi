"""Qt-free frame helpers: 2-D extraction, orientation, dark subtraction, hot pixels.

Data conventions:
    Frames are ``(slit, wavelength)`` arrays, usually uint16 (rows = slit axis).
    Shape mismatches are logged on the ``"HSI"`` logger and the input is returned
    unchanged rather than raising.

Threading:
    No shared state; safe to call from any thread.

Example:
    >>> import numpy as np
    >>> from modules.image.core import correct_hot_pixels
    >>> frame = np.full((5, 5), 10, dtype=np.uint16); frame[2, 2] = 999
    >>> hot = np.zeros((5, 5), dtype=bool); hot[2, 2] = True
    >>> int(correct_hot_pixels(frame, hot)[2, 2])
    10
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

logger = logging.getLogger("HSI")


def raw_2d_frame(frame: np.ndarray) -> np.ndarray:
    """Return a C-contiguous 2-D ``(slit, wavelength)`` frame.

    Args:
        frame: 2-D frame, or 3-D ``(slit, wavelength, C)`` (channel 0 is taken),
            or ``None``.

    Returns:
        numpy.ndarray: The frame as a C-contiguous 2-D array.  This is the
        input itself (a view, no copy) when it is already 2-D and contiguous,
        otherwise a new array.  ``None`` is passed through.

    Raises:
        ValueError: if ``frame.ndim`` is neither 2 nor 3.
    """
    if frame is None:
        return frame

    if frame.ndim == 2:
        return np.ascontiguousarray(frame)

    if frame.ndim == 3:
        return np.ascontiguousarray(frame[:, :, 0])

    raise ValueError(f"Unexpected frame ndim: {frame.ndim}")


def apply_raw_orientation(
    raw_frame: np.ndarray,
    raw_inverted: bool = True,
) -> np.ndarray:
    """Optionally flip the slit axis of a raw frame.

    Args:
        raw_frame: ``(slit, wavelength)`` frame (see :func:`raw_2d_frame`).
        raw_inverted: If true, reverse the slit (row) axis.

    Returns:
        numpy.ndarray: C-contiguous 2-D frame.  A new array when flipped; may be
        a view of ``raw_frame`` when not flipped and already contiguous.
    """
    raw = raw_2d_frame(raw_frame)

    if raw_inverted:
        raw = raw[::-1, :]

    return np.ascontiguousarray(raw)


def subtract_dark_frame(
    frame: np.ndarray,
    dark_frame: np.ndarray,
) -> np.ndarray:
    """Subtract a dark frame, clamping at zero.

    Args:
        frame: Image ``(slit, wavelength)``.
        dark_frame: Dark reference of the same shape (any dtype).

    Returns:
        numpy.ndarray: New C-contiguous array with ``frame``'s dtype; negative
        results are clamped to 0 (saturating, unsigned-safe) for integer
        frames.  ``frame`` itself (not a copy) is returned if either input is
        ``None`` or the shapes differ (a warning is logged).  Note: when both
        inputs share a float dtype, ``cv2.subtract`` is used and negative
        values are not clamped.
    """
    if frame is None or dark_frame is None:
        return frame

    if frame.shape != dark_frame.shape:
        logger.warning(
            "Dark frame shape mismatch: frame=%s dark_frame=%s",
            frame.shape, dark_frame.shape,
        )
        return frame

    if frame.dtype == dark_frame.dtype:
        return np.ascontiguousarray(cv2.subtract(frame, dark_frame))

    corrected = np.subtract(
        frame.astype(np.int32),
        dark_frame.astype(np.int32),
        dtype=np.int32,
    )

    if np.issubdtype(frame.dtype, np.integer):
        np.clip(corrected, 0, np.iinfo(frame.dtype).max, out=corrected)
        return np.ascontiguousarray(corrected.astype(frame.dtype))

    corrected = np.clip(corrected, 0, None)
    return np.ascontiguousarray(corrected.astype(frame.dtype))


def correct_hot_pixels(
    frame: np.ndarray,
    hot_pixel_map: np.ndarray | None,
) -> np.ndarray:
    """Replace flagged pixels by the mean of their 3x3 neighbours.

    Args:
        frame: Image ``(slit, wavelength)``.
        hot_pixel_map: Array of the same shape; non-zero entries mark hot
            pixels.  ``None`` disables the correction.

    Returns:
        numpy.ndarray: New C-contiguous array with ``frame``'s dtype in which
        every flagged pixel is the rounded mean of its in-bounds neighbours
        (the pixel itself excluded; neighbours are taken from the uncorrected
        frame, so adjacent hot pixels contribute their raw values).  ``frame``
        itself (no copy) is returned when the map is ``None``, empty, or the
        shapes differ (a warning is logged).
    """
    if frame is None or hot_pixel_map is None:
        return frame

    if frame.shape != hot_pixel_map.shape:
        logger.warning(
            "Hot pixel map shape mismatch: frame=%s hot_map=%s",
            frame.shape, hot_pixel_map.shape,
        )
        return frame

    coords = np.column_stack(np.nonzero(hot_pixel_map))
    if coords.size == 0:
        return frame

    kernel = np.array(
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
        dtype=np.uint8,
    )

    base = frame.astype(np.float32)

    neighbor_sum = cv2.filter2D(
        base, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT,
    )

    neighbor_count = cv2.filter2D(
        np.ones_like(frame, dtype=np.uint8),
        ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT,
    ).astype(np.float32)

    neighbor_count = np.maximum(neighbor_count, 1.0)
    neighbor_avg = neighbor_sum / neighbor_count

    coords = coords.astype(np.intp)

    corrected = frame.copy()
    corrected[coords[:, 0], coords[:, 1]] = np.round(
        neighbor_avg[coords[:, 0], coords[:, 1]]
    ).astype(frame.dtype)

    return np.ascontiguousarray(corrected)

def save_png(path, frame) -> None:
    """Write a 2-D uint8/uint16 frame as an 8/16-bit grayscale PNG, values unscaled.

    Args:
        path: Destination file (any characters; written through Python, not ``cv2.imwrite``).
        frame: 2-D ``(slit, wavelength)`` array of dtype uint8 or uint16.

    Raises:
        ValueError: ``frame`` is not 2-D or not uint8/uint16 (a PNG cannot hold other types).
        OSError: encoding or writing failed.
    """
    a = np.asarray(frame)
    if a.ndim != 2:
        raise ValueError(f"PNG export needs a 2-D image, got shape {a.shape}")
    if a.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"PNG export needs uint8 or uint16 data, got {a.dtype}")
    ok, data = cv2.imencode(".png", np.ascontiguousarray(a))
    if not ok:
        raise OSError("PNG encoding failed")
    with open(path, "wb") as f:
        f.write(data.tobytes())


def png_name(prefix: str, camera_bin=None, tags=()) -> str:
    """Default export file name, e.g. ``smile_bin2_20261014-153012.png``.

    Args:
        prefix: Leading word (module or source).
        camera_bin: Hardware bin, or ``None`` to omit.
        tags: Extra words appended in order (e.g. ``("dark", "smile")``).
    """
    parts = [prefix]
    if camera_bin:
        parts.append(f"bin{int(camera_bin)}")
    parts.append(time.strftime("%Y%m%d-%H%M%S"))
    parts.extend(tags)
    return "_".join(parts) + ".png"
