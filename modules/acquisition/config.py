"""Processing settings of the acquisition pipeline, owned by the acquisition module."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def parse_wavelength_ranges(value) -> dict:
    """Check the ``wavelength_range_nm`` setting and return it as ``{backend: (lo, hi)}``.

    Args:
        value: ``{camera_backend: [lo_nm, hi_nm] or null}`` as read from JSON, or ``None``.
            A missing setting or a ``null`` entry means no range (the full calibrated range).
            Values are rounded to 0.1 nm, the resolution of the Scan Window control.

    Returns:
        dict: Lower-case backend name -> ``(lo, hi)`` floats (entries with ``null`` omitted).

    Raises:
        ValueError: if the setting is not an object, or an entry is not two finite numbers
            with lo < hi.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"wavelength_range_nm must be an object {{camera_backend: [start_nm, end_nm]}}, not {value!r}")
    ranges = {}
    for backend, entry in value.items():
        if entry is None:
            continue
        try:
            lo, hi = (round(float(v), 1) for v in entry)
        except (TypeError, ValueError):
            raise ValueError(f"wavelength_range_nm[{backend!r}] must be [start_nm, end_nm], not {entry!r}") from None
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"wavelength_range_nm[{backend!r}] must be finite with start < end, not {entry!r}")
        ranges[str(backend).lower()] = (lo, hi)
    return ranges


@dataclass
class ProcessingConfig:
    """Spectral binning, orientation and preprocessing choices.

    Attributes:
        m: Dispersion in nm per pixel of the *unbinned* sensor (the engine multiplies by the hardware bin).
        c: Wavelength (nm) of spectral pixel 0.
        resolution: Width of one output wavelength bin in nm (always used when
            ``resolution_source`` is ``"config"``; the fallback for ``"fwhm"``).
        resolution_source: ``"config"`` (bin width = ``resolution``) or ``"fwhm"`` (bin width =
            ``fwhm_step_factor`` × the median FWHM stored for the camera setting by
            ``modules.spectral_resolution``, at least 2 spectral pixels; ``resolution`` when no
            usable FWHM is stored).
        fwhm_step_factor: Bin width per FWHM when ``resolution_source`` is ``"fwhm"``.
        raw_inverted: Flip the slit axis of every frame (sensor mounted upside down).
        preprocessing_backend: ``"auto"``, ``"cpu"``, ``"opencv_cuda"`` or ``"cupy_cuda"``.
        rgb_target_wavelengths: Wavelengths (nm) used for the default R, G, B preview bands.
        profile_timing: Log per-stage timings of the acquisition loop (``[profile]`` lines);
            GPU per-kernel stages are sampled on one frame in ten.
        wavelength_range_nm: Recorded wavelength range per camera backend,
            ``{backend: (lo_nm, hi_nm)}`` (see :func:`parse_wavelength_ranges`); a backend
            without an entry records the full calibrated range.
    """

    m: float = 0.2903
    c: float = 399.7910
    resolution: float = 2
    resolution_source: str = "config"
    fwhm_step_factor: float = 1.0
    raw_inverted: bool = True
    preprocessing_backend: str = "auto"
    rgb_target_wavelengths: tuple = (645.0, 555.0, 465.0)
    profile_timing: bool = True
    wavelength_range_nm: dict = field(default_factory=dict, repr=False)   # the 'Loaded config' log stays as before
