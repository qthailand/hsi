"""Saturation test against a per-pixel ceiling (Qt-free).

Data conventions:
    ``ceiling`` holds, for every display pixel, the value a clipped sensor pixel shows
    there (full scale minus the dark that was subtracted, after Smile and binning; see
    ``AcquisitionSink.saturation_ceiling``). It has the same shape as ``values``.

Example:
    >>> import numpy as np
    >>> saturated_mask(np.array([50000, 51000, 30000]), np.array([51000, 51000, 51000])).tolist()
    [False, True, False]
"""
from __future__ import annotations

import numpy as np

SATURATION_FRACTION = 0.99


def saturated_mask(values, ceiling, fraction: float = SATURATION_FRACTION) -> np.ndarray:
    """Boolean mask of values at or above ``fraction`` of their ceiling.

    Args:
        values: Display values (any numeric dtype).
        ceiling: Per-pixel ceiling, same shape as ``values``.
        fraction: Share of the ceiling that counts as saturated (0.99, as the histogram).

    Returns:
        numpy.ndarray: ``bool`` array of ``values``' shape. Pixels whose ceiling is 0
        (outside the Smile-corrected area) are never saturated.
    """
    ceiling = np.asarray(ceiling, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    return (ceiling > 0) & (values >= ceiling * fraction)


#: Frames above this many pixels are subsampled by 4 for the exposure histogram (speed).
HISTOGRAM_SUBSAMPLE_ABOVE = 500_000
#: Bins of the exposure histogram (values scaled to 8 bit).
HISTOGRAM_BINS = 256


def exposure_histogram(frame, bit_depth: int, ceiling=None):
    """Exposure histogram of a raw frame and its saturated share (Qt-free).

    Values are scaled to 8 bit with the sensor's full scale ``2**bit_depth - 1`` into
    :data:`HISTOGRAM_BINS` bins. A pixel is saturated at >= 99 % of its ``ceiling``
    (:func:`saturated_mask`) when one of ``frame``'s shape is given, else at >= 99 % of full
    scale. Frames above :data:`HISTOGRAM_SUBSAMPLE_ABOVE` pixels use every 4th pixel.

    Args:
        frame: Raw frame of any shape (flattened), sensor counts. Not modified.
        bit_depth: Sensor bit depth.
        ceiling: Optional per-pixel saturation ceiling of ``frame``'s shape.

    Returns:
        tuple: ``(counts, saturated_counts, saturated_percent)``: float64 ``(256,)`` pixels per
        bin; float64 ``(256,)`` saturated pixels per bin (``None`` without a ceiling: then the
        saturated pixels are the top bins); the saturated share in %.

    Example:
        >>> counts, sat, pct = exposure_histogram(np.array([0, 4095, 4095, 100]), 12)
        >>> int(counts[255]), sat, pct
        (2, None, 50.0)
    """
    flat = np.asarray(frame).ravel()
    flat_ceiling = None
    if ceiling is not None and np.shape(ceiling) == np.shape(frame):
        flat_ceiling = np.asarray(ceiling).ravel()
    if flat.size > HISTOGRAM_SUBSAMPLE_ABOVE:
        flat = flat[::4]
        if flat_ceiling is not None:
            flat_ceiling = flat_ceiling[::4]
    raw_max = float((1 << int(bit_depth)) - 1)
    flat_8bit = np.clip(flat.astype(np.float32) * (255.0 / raw_max), 0, 255).astype(np.uint8)
    counts = np.bincount(flat_8bit, minlength=HISTOGRAM_BINS).astype(np.float64)
    if flat_ceiling is not None:
        sat = saturated_mask(flat, flat_ceiling)
        n_sat = int(np.count_nonzero(sat))
        sat_counts = np.bincount(flat_8bit[sat], minlength=HISTOGRAM_BINS).astype(np.float64)
    else:
        n_sat = int(np.count_nonzero(flat >= raw_max * SATURATION_FRACTION))
        sat_counts = None
    percent = (n_sat / flat.size) * 100.0 if flat.size else 0.0
    return counts, sat_counts, percent
