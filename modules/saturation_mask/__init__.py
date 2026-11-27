"""Saturation mask: raw clipped pixels followed through the correction pipeline to the recorded bands.

A raw pixel that clipped is found before dark subtraction and carried through hot-pixel repair,
orientation, the smile (+ keystone) remap, the keystone row crop and lambda binning with a
logical OR (any source tap with a non-zero weight), so a band that averaged in one clipped pixel
is flagged even though its value is below full scale. CPU reference; the intensities are never
touched.

* :func:`detect_raw` - clipped / near-saturated raw pixels (:class:`RawSaturation`);
* :func:`hot_pixel_mask`, :class:`GeometryMask`, :func:`bin_masks`, :func:`crop_rows` - the stages;
* :class:`SaturationPropagator` - a raw frame to :class:`FrameSaturation` (the binned layout);
* :func:`tap_support` / :func:`tap_weights` - OpenCV's 1/32 fixed-point tap tables (Lanczos4: 8 taps);
* :func:`pack_mask` / :func:`unpack_mask` - bit packing for recordings;
* :func:`level_evidence` - the camera's full scale, or why there is none to trust;
* :class:`SaturationRecord` - one recorded frame, compact; :func:`scan_status` /
  :func:`scan_summary` - ``checked`` / ``partial`` / ``not_checked`` of a recording.

Data conventions:
    Raw frames ``(slit rows, spectral columns)`` in sensor orientation; corrected masks in the
    corrected frame's layout; binned masks ``(slit rows, bands)`` after the keystone row crop,
    like the recorded lines. ``clipped`` = ``raw >= level``; ``near`` = ``raw >= near_ratio *
    level`` (clipped included). Fractions in 0..1.

Threading:
    Pure functions; one :class:`SaturationPropagator` per processing thread (it caches maps).

Example:
    >>> import numpy as np
    >>> from modules.saturation_mask import detect_raw
    >>> detect_raw(np.array([[4095, 10]]), 4095).raw_clipped_fraction
    0.5
"""
from .core import (NEAR_RATIO, POLICY, TAPS, FrameSaturation, GeometryMask, RawSaturation,
                   SaturationPropagator, bin_masks, crop_rows, detect_raw, hot_pixel_mask, pack_mask,
                   tap_support, tap_weights, unpack_mask)
from .record import (STATUS_CHECKED, STATUS_NOT_CHECKED, STATUS_PARTIAL, LevelEvidence, SaturationRecord,
                     level_evidence, scan_status, scan_summary)

__all__ = ["NEAR_RATIO", "POLICY", "TAPS", "RawSaturation", "FrameSaturation", "GeometryMask",
           "SaturationPropagator", "detect_raw", "hot_pixel_mask", "bin_masks", "crop_rows",
           "tap_weights", "tap_support", "pack_mask", "unpack_mask",
           "STATUS_CHECKED", "STATUS_PARTIAL", "STATUS_NOT_CHECKED", "LevelEvidence", "level_evidence",
           "SaturationRecord", "scan_status", "scan_summary"]
__requires__ = ("preprocessing",)      # its prepare_maps: the same fixed-point maps as the intensities
