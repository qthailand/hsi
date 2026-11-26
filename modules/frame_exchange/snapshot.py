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
        clipped: Optional boolean mask, same shape and orientation as ``image``: pixels at the
            sensor's full scale in the RAW frame (before dark subtraction, which moves a
            saturated pixel below full scale and hides it). ``None`` when the producer does not
            know.
        saturation_level: Optional raw full-scale level (DN) the mask was taken against.
    """
    request_id: int
    image: object
    timestamp: float
    camera_bin: int
    raw_inverted: bool
    dark_enabled: bool
    dark_applied: bool
    stage: str = 'post_dark_hot_pixel_orientation'
    clipped: object = None
    saturation_level: object = None


#: Processing stages a :class:`FrameStages` packet can hold, in pipeline order:
#: ``raw`` (camera counts, slit axis in display orientation), ``dark`` (after dark subtraction
#: and hot-pixel repair), ``geometry`` (after the smile / keystone remap, full sensor),
#: ``valid`` (``geometry`` cropped to the rows and columns the cube keeps) and ``binned``
#: (the line the cube records: wavelength bands, keystone rows cropped).
FRAME_STAGES = ('raw', 'dark', 'geometry', 'valid', 'binned')


@dataclass(frozen=True)
class FrameStages:
    """Several processing stages of ONE camera frame, for saving or comparing them.

    Attributes:
        images: ``{stage: 2-D array}`` for the requested stages (keys from
            :data:`FRAME_STAGES`, in pipeline order); each array is a read-only copy.
            Every stage except ``binned`` has ``(slit, spectral pixel)`` axes in the same
            orientation, so pixel ``(y, x)`` of ``raw``, ``dark`` and ``geometry`` is the same
            position (``valid`` is offset by ``(valid_rows[0], valid_columns[0])``).
        timestamp: Capture time in seconds (the producer's monotonic clock).
        camera_bin: Hardware bin factor.
        raw_inverted: The slit axis of the sensor frame was flipped (also in ``raw``).
        dark_applied: Dark subtraction / hot-pixel repair was applied (``dark`` differs from ``raw``).
        smile_applied: A smile model was applied (``geometry`` differs from ``dark``).
        keystone_applied: That model also corrected the keystone.
        valid_rows: ``(top, bottom)`` rows left out of ``valid`` and ``binned``.
        valid_columns: ``(left, right)`` spectral columns left out of ``valid``.
        wavelengths_nm: Centre wavelength of each ``binned`` column, or ``None``.
    """
    images: dict
    timestamp: float
    camera_bin: int
    raw_inverted: bool
    dark_applied: bool
    smile_applied: bool
    keystone_applied: bool
    valid_rows: tuple = (0, 0)
    valid_columns: tuple = (0, 0)
    wavelengths_nm: object = None
