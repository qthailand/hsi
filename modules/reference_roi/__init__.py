"""Standalone reference ROI profiles, visual selection and HSI scan bridge.

Qt-free: ``ReferenceROIProfile``, ``ReferenceROIStore``, ``select_rgb``,
``compose_preview``. Qt (loaded lazily): ``ReferenceROIWidget``.

Example:
    >>> import numpy as np
    >>> from modules.reference_roi import select_rgb, compose_preview
    >>> frame = np.random.rand(8, 5)
    >>> compose_preview([select_rgb(frame, None)] * 4).shape
    (8, 4, 3)
"""
from .core import ReferenceROIProfile, ReferenceROIStore, select_rgb, compose_preview

__all__ = ['ReferenceROIProfile', 'ReferenceROIStore', 'ReferenceROIWidget', 'select_rgb', 'compose_preview']
__requires__ = ()
__widgets__ = ('ReferenceROIWidget',)

def __getattr__(name):
    if name == 'ReferenceROIWidget':
        from .widget import ReferenceROIWidget
        return ReferenceROIWidget
    raise AttributeError(name)
