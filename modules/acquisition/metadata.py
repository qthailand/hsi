"""Provenance of a recorded scan: everything needed to interpret (or re-simulate) its raw frames.

The raw spectrogram kept by *Keep Spectrogram* is a stack of untouched sensor frames. On its own
it does not say which hardware bin, exposure or gain produced it, whether HSI flips the slit axis
before processing, or which wavelength calibration applied. :func:`build_raw_metadata` collects
those facts when a scan starts so they can be stored next to the data (HDF5 attributes written by
:func:`modules.cube_io.hdf5.save_hdf5`).

Data conventions:
    The result is a flat ``dict`` of ``str``/``int``/``float``/``bool`` values (HDF5-attribute
    friendly). Exposure is milliseconds, ``m_*`` slopes are nm per pixel, ``c_nm``,
    ``resolution_nm`` (bin width actually used) and ``fwhm_nm`` are nm, ``degree*`` are degrees.
    ``resolution_source`` says whether the bin width came from the config or the stored FWHM. ``bit_depth`` is the frames' bit depth,
    ``sensor_bit_depth`` the real data bits before scaling and ``full_scale`` the largest value
    the frames can hold (see :func:`camera_bit_info`).

Threading:
    Pure; call from any thread.

Example:
    >>> from types import SimpleNamespace as NS
    >>> cam = NS(camera_id=0, bit_depth=16, pixel_size=3.75, width=804, height=552, gain=1, exposure_ms=1.0)
    >>> backend = NS(info=cam, current_bin=2, gain=1, exposure_ms=1.0)
    >>> config = NS(camera_backend='poa', camera_index=0, raw_inverted=True, m=0.2903, c=399.791, resolution=2)
    >>> meta = build_raw_metadata(backend=backend, config=config, scan={'stage_type': 'rotation', 'degree': 10.0})
    >>> meta['camera_bin'], meta['m_nm_per_px'], meta['raw_inverted']
    (2, 0.5806, True)
"""
from __future__ import annotations

import datetime
from typing import Optional

STAGE_DESCRIPTION = "raw sensor frames before dark/hot-pixel/orientation/smile and spectral binning"


def _number(value, kind):
    """``kind(value)`` or ``None`` when the camera does not report the value."""
    return None if value is None else kind(value)


def camera_bit_info(backend) -> dict:
    """Bit depth of the frames, real bits before scaling, and the frames' full-scale value.

    The real bits sit at the top of the frame's bits (POA 12 -> 16, GenICam 10 -> 16), so the
    full scale is ``(2**sensor_bit_depth - 1) << (bit_depth - sensor_bit_depth)``: 65520 for
    12 -> 16, 65472 for 10 -> 16, 1023 for an unscaled 10-bit XIMEA frame.

    Args:
        backend: Camera backend with ``info.bit_depth`` and optionally ``sensor_bit_depth()``.

    Returns:
        ``{'bit_depth', 'sensor_bit_depth', 'full_scale'}`` (ints), or ``{}`` if the backend
        reports no bit depth.
    """
    bits = _number(getattr(getattr(backend, "info", None), "bit_depth", None), int)
    if bits is None:
        return {}
    try:
        sensor_bits = int(backend.sensor_bit_depth())
    except Exception:
        sensor_bits = bits
    sensor_bits = max(1, min(sensor_bits, bits))
    return {"bit_depth": bits, "sensor_bit_depth": sensor_bits,
            "full_scale": ((1 << sensor_bits) - 1) << (bits - sensor_bits)}


def build_raw_metadata(*, backend, config, scan: Optional[dict] = None,
                       started_at: Optional[datetime.datetime] = None,
                       binning: Optional[dict] = None) -> dict:
    """Describe the camera, processing settings and scan motion of the scan that is starting.

    Args:
        backend: The camera backend (``info``, ``current_bin``, and optionally ``gain`` /
            ``exposure_ms`` attributes holding the values actually applied).
        config: Object with ``camera_backend``, ``camera_index``, ``raw_inverted``, ``m`` (nm/pixel
            at bin 1), ``c`` (nm) and ``resolution`` (nm per output bin).
        scan: Stage settings of this scan, e.g. ``{'stage_type': 'rotation', 'degree': 10,
            'degree_per_second': 1}``; keys with ``None`` values are left out.
        started_at: Scan start time (defaults to now, local time with UTC offset).
        binning: The engine's ``binning_info()`` (bin width actually used and its source);
            without it ``resolution_nm`` is ``config.resolution``.

    Returns:
        A flat dict (see the module docstring for units). ``m_nm_per_px`` is the slope for the
        recorded frames' own pixels (``m * camera_bin``), ``m_nm_per_px_bin1`` the unbinned one.
    """
    info = backend.info
    bin_value = int(getattr(backend, "current_bin", 1) or 1)
    started = started_at or datetime.datetime.now().astimezone()
    base_m = float(config.m)
    meta = {
        "stage": STAGE_DESCRIPTION,
        "software": "HSI",
        "started_at": started.isoformat(timespec="seconds"),
        "camera_backend": str(config.camera_backend),
        "camera_index": int(getattr(config, "camera_index", 0) or 0),
        "camera_bin": bin_value,
        "frame_height": int(info.height),
        "frame_width": int(info.width),
        "bit_depth": _number(getattr(info, "bit_depth", None), int),
        **{key: value for key, value in camera_bit_info(backend).items() if key != "bit_depth"},
        "pixel_size": _number(getattr(info, "pixel_size", None), float),
        "gain": getattr(backend, "gain", None),
        "exposure_ms": getattr(backend, "exposure_ms", None),
        "raw_inverted": bool(config.raw_inverted),
        "m_nm_per_px_bin1": base_m,
        "m_nm_per_px": base_m * bin_value,
        "c_nm": float(config.c),
        "resolution_nm": float(config.resolution),
    }
    meta.update(binning or {})
    for key, value in (scan or {}).items():
        meta[key] = value
    return {key: value for key, value in meta.items() if value is not None}
