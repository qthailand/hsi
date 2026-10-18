"""Compatibility imports; implementations live in reusable packages."""

from modules.image import raw_2d_frame, apply_raw_orientation, subtract_dark_frame, correct_hot_pixels

__all__ = ['raw_2d_frame', 'apply_raw_orientation', 'subtract_dark_frame', 'correct_hot_pixels']
