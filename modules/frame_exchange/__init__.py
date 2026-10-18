"""Application-owned snapshot exchange, independent of any consumer module.

``FrameSnapshot`` (plain dataclass) is importable without Qt. ``FrameExchange``
(a ``QObject`` hub) is loaded lazily on first access, so
``import modules.frame_exchange`` does not import PyQt5.

Example:
    >>> import numpy as np
    >>> from modules.frame_exchange import FrameSnapshot
    >>> FrameSnapshot(1, np.zeros((2, 2)), 0.0, 1, False, False, False).camera_bin
    1
"""
from .snapshot import FrameSnapshot

__all__ = ['FrameExchange', 'FrameSnapshot']
__requires__ = ()


def __getattr__(name):
    if name == 'FrameExchange':
        from .hub import FrameExchange
        return FrameExchange
    raise AttributeError(name)
