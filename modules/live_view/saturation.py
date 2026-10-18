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
