"""Processing settings of the acquisition pipeline, owned by the acquisition module."""
from __future__ import annotations

from dataclasses import dataclass


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
