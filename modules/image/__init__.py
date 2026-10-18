"""Frame orientation, dark subtraction, hot-pixel and PNG export helpers, independent of Qt.

Data conventions:
    A sensor frame / spectral image is ``(slit, wavelength)`` (rows = slit/spatial
    axis, columns = spectral axis), normally uint16.  Functions that accept a
    ``(slit, wavelength, C)`` array use channel 0.

Threading:
    Pure functions with no shared state; callable from any thread.

Example:
    >>> import numpy as np
    >>> from modules.image import apply_raw_orientation, subtract_dark_frame
    >>> frame = np.full((4, 6), 100, dtype=np.uint16)
    >>> dark = np.full((4, 6), 130, dtype=np.uint16)
    >>> int(subtract_dark_frame(frame, dark).max())        # saturates at 0
    0
    >>> apply_raw_orientation(np.arange(8).reshape(2, 4)).tolist()
    [[4, 5, 6, 7], [0, 1, 2, 3]]
"""
from .core import raw_2d_frame, apply_raw_orientation, subtract_dark_frame, correct_hot_pixels, save_png, png_name

__all__ = ("raw_2d_frame", "apply_raw_orientation", "subtract_dark_frame", "correct_hot_pixels", "save_png", "png_name")
__requires__ = ()
