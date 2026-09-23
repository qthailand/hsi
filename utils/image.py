from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger("HSI")


def raw_2d_frame(frame: np.ndarray) -> np.ndarray:
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
    raw = raw_2d_frame(raw_frame)

    if raw_inverted:
        raw = raw[::-1, :]

    return np.ascontiguousarray(raw)


def subtract_dark_frame(
    frame: np.ndarray,
    dark_frame: np.ndarray,
) -> np.ndarray:
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