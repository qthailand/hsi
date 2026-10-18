"""Row/axis conventions shared by modules that must agree on the same pixel.

Data conventions:
    Spectral images are ``(rows, columns) = (slit position, spectral pixel)``;
    rows and columns are zero-based.
"""


def default_reference_row(height: int) -> int:
    """Return the zero-based *reference row* used for smile and wavelength work.

    The rule is the centre row, rounded down to ``height / 2 - 1`` for odd
    heights (the agreed convention of Smile Analysis). Wavelength calibration,
    Smile correction and quality diagnostics must all use the same row, because
    spectral lines shift with the row when the optics have smile.

    Args:
        height: Number of image rows (> 0).

    Returns:
        The reference row index. ``height == 1`` yields ``-1``; callers that can
        see such an image should clamp to ``0``.

    Raises:
        ValueError: if ``height`` is not positive.

    Example:
        >>> default_reference_row(552)
        276
        >>> default_reference_row(553)
        275
    """
    if height <= 0:
        raise ValueError("height must be positive")
    half_height = height / 2.0
    center = int(half_height // 1)
    if not half_height.is_integer():
        center -= 1
    return center
