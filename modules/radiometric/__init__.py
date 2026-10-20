"""Offline radiometric calibration of ENVI cubes, with an optional GUI.

Qt-free (``core``): ``compute_rowwise_factors`` (white-reference factors per slit
row and band), ``compute_fluorescence_factors`` (ROI-average factors, Savitzky-Golay
smoothed), ``process_cube`` (crop, calibrate, spatially bin, write ENVI BIP),
``resample_cube`` (same, plus linear resampling to a wavelength grid),
``build_raw_cube_tmp`` / ``build_cal_cube_tmp`` (temporary band-stack memmaps),
``check_factor_bands`` (refuses factors made for other bands) and
``save_factor_profile`` / ``load_factor_profile`` (``.npy`` plus wavelength sidecar).
Qt (loaded lazily, so ``import modules.radiometric`` does not import PyQt5):
``CalibrationWindow``, ``RoiDialog`` and the thread adapter ``Worker``.

Data conventions:
    * Inputs are ENVI files on disk, read band by band as float32 ``(lines, samples)``
      images in sensor counts; wavelengths are in nm.
    * Row-wise factors are ``(lines, bands)``, fluorescence factors ``(bands,)``;
      both multiply counts. Crop boxes are ``(x0, x1, y0, y1)`` half-open pixels.
    * Exported cubes are ENVI BIP ``(lines, samples, bands)`` plus a ``.hdr``.

Threading:
    Core functions are synchronous and thread-agnostic (heavy: run them off the GUI
    thread); ``progress_callback(int)`` receives band counts. GUI classes: GUI thread
    only; ``CalibrationWindow`` runs its own work on a ``QThread`` through ``Worker``.

Example:
    >>> from modules.cube_io.envi import parse_envi_header
    >>> from modules.radiometric import compute_fluorescence_factors
    >>> meta = parse_envi_header('white_reference.bsq')          # doctest: +SKIP
    >>> mask = ...  # boolean (lines, samples) ROI                # doctest: +SKIP
    >>> factors = compute_fluorescence_factors('white_reference.bsq', meta, mask)['factors']  # doctest: +SKIP
"""
from .core import rowwise_calculation_heavy_logic, fluo_calculation_heavy_logic, create_datacube_logic, create_datacube_resampled_logic, build_raw_cube_tmp, build_cal_cube_tmp, REFLECTANCE_PROFILE_DIR, list_reflectance_profiles, check_factor_bands, save_factor_profile, load_factor_profile

__all__ = ['rowwise_calculation_heavy_logic', 'fluo_calculation_heavy_logic', 'create_datacube_logic', 'create_datacube_resampled_logic', 'build_raw_cube_tmp', 'build_cal_cube_tmp', 'REFLECTANCE_PROFILE_DIR', 'list_reflectance_profiles', 'check_factor_bands', 'save_factor_profile', 'load_factor_profile', 'compute_rowwise_factors', 'compute_fluorescence_factors', 'process_cube', 'resample_cube', 'CalibrationWindow', 'RoiDialog', 'Worker']
__requires__ = ('cube_io',)
__widgets__ = ('CalibrationWindow',)

def __getattr__(name):
    if name in {'CalibrationWindow', 'RoiDialog'}:
        from . import widget
        return getattr(widget,name)
    if name=='Worker':
        from .workers import Worker
        return Worker
    raise AttributeError(name)

# Short public names for new integrations; GUI method names remain available.
compute_rowwise_factors = rowwise_calculation_heavy_logic
compute_fluorescence_factors = fluo_calculation_heavy_logic
process_cube = create_datacube_logic
resample_cube = create_datacube_resampled_logic
