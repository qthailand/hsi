"""``FrameSnapshot``: the immutable one-frame packet exchanged by ``FrameExchange`` (Qt-free).

Data conventions:
    ``image`` is a 2-D spectral image ``(slit, wavelength)`` (typically ``uint16``),
    taken after dark/hot-pixel correction and orientation, hence
    ``stage = 'post_dark_hot_pixel_orientation'``. ``timestamp`` is the capture time
    in seconds (the producer's clock). ``camera_bin`` is the hardware bin factor.

    The dataclass is frozen, but ``image`` is a numpy array and is *not* copied:
    consumers must treat it as read-only and copy before modifying it.

Threading:
    Plain data; safe to hand between threads as long as nobody mutates ``image``.

Example:
    >>> import numpy as np
    >>> from modules.frame_exchange import FrameSnapshot
    >>> snap = FrameSnapshot(1, np.zeros((4, 5), np.uint16), 0.0, 1, False, True, True)
    >>> snap.stage
    'post_dark_hot_pixel_orientation'
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class FrameSnapshot:
    """One captured camera frame plus the settings it was captured with.

    Attributes:
        request_id: Id returned by ``FrameExchange.request`` that this frame answers.
        image: 2-D ``(slit, wavelength)`` array; shared, read-only by convention.
        timestamp: Capture time in seconds.
        camera_bin: Hardware bin factor of ``image``.
        raw_inverted: Whether the raw frame orientation was inverted.
        dark_enabled: Whether dark correction was enabled when captured.
        dark_applied: Whether dark correction was actually applied to ``image``.
        stage: Processing stage of ``image``.
    """
    request_id: int
    image: object
    timestamp: float
    camera_bin: int
    raw_inverted: bool
    dark_enabled: bool
    dark_applied: bool
    stage: str = 'post_dark_hot_pixel_orientation'
