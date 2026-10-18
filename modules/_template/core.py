"""Qt-free logic of the module.

Keep everything that is not drawing in here: it can then be tested without a display
and reused from scripts, servers and other GUI toolkits.
"""
from __future__ import annotations


def scale(values, factor: float) -> list:
    """Multiply every value by ``factor``.

    Args:
        values: Iterable of numbers.
        factor: Multiplier (dimensionless).

    Returns:
        A new list of floats; ``values`` is not modified.

    Example:
        >>> scale([1, 2], 0.5)
        [0.5, 1.0]
    """
    return [float(v) * factor for v in values]
