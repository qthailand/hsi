"""Offline reference calibration and cube processing, independent of Qt.

Formulas, interpolation, clipping and output layouts preserve the source GUI.
Callbacks receive integer progress values (band counts), not Qt signals.
This processing is never inserted into live camera acquisition.

Data conventions:
    * Inputs are ENVI files on disk (``.hdr`` + binary, any interleave that
      ``modules.cube_io.envi`` reads) described by the dict returned by
      ``parse_envi_header``: ``lines`` (first spatial axis, the slit direction),
      ``samples`` (second spatial axis, the scan direction), ``bands``,
      ``wavelength`` (float array in nm or ``None``) and ``data type``.
      Bands are read one at a time as float32 ``(lines, samples)`` images in raw
      sensor counts, so memory use stays at one or two bands.
    * Calibration factors are multipliers applied to counts: ``rowwise`` factors
      are ``(lines, bands)`` float32 (one factor per slit row and band), global
      "fluorescence" factors are ``(bands,)`` (one factor per band).
    * Crop boxes are ``(x0, x1, y0, y1)`` half-open pixel ranges, ``x`` along
      ``samples`` and ``y`` along ``lines``.
    * Cubes written by ``create_datacube_logic`` / ``create_datacube_resampled_logic``
      are ENVI BIP files, ``(lines, samples, bands)`` little-endian, float32 when a
      calibration is applied, otherwise the input dtype (uint16 unless the input is
      float32), with a ``<save_path>.hdr`` text header (wavelengths in nm).
    * The temporary cubes of ``build_raw_cube_tmp`` / ``build_cal_cube_tmp`` are
      header-less band-stacked memmaps of shape ``(lines * bands, samples)``:
      element ``[y + b * lines, x]`` is pixel ``(y, x)`` of band ``b``.

Threading:
    Every function is plain synchronous NumPy/SciPy code with no shared state and
    no Qt objects, so it may run on any thread. The GUI runs them on a worker
    thread through ``modules.radiometric.workers.Worker``. ``progress_callback``
    is called from the calling thread with an ``int``: ``0`` at start and then the
    number of bands processed so far (``1 .. n_bands``), not a percentage.

Example:
    >>> import numpy as np
    >>> from modules.cube_io.envi import parse_envi_header
    >>> from modules.radiometric import compute_fluorescence_factors
    >>> meta = parse_envi_header('reference.bsq')                # doctest: +SKIP
    >>> mask = np.zeros((meta['lines'], meta['samples']), bool)  # doctest: +SKIP
    >>> mask[10:50, 20:80] = True                                # doctest: +SKIP
    >>> compute_fluorescence_factors('reference.bsq', meta, mask)['factors'].shape  # doctest: +SKIP
    (bands,)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from modules.cube_io.envi import parse_envi_header, _read_envi_band_float32

#: Reflectance tables of the reference targets (Spectralon, PVC, ...): Excel sheets without
#: header, column 0 wavelength (nm), column 1 reflectance. Drop a new file here to offer it.
REFLECTANCE_PROFILE_DIR = Path(__file__).resolve().parent / 'reflectance_profiles'


def list_reflectance_profiles(folder=None):
    """Reflectance tables (``.xlsx``/``.xls``) in ``folder``, sorted by file name.

    Args:
        folder: Directory to list; defaults to :data:`REFLECTANCE_PROFILE_DIR`.

    Returns:
        list[pathlib.Path]: Absolute paths; empty if the folder does not exist. Excel
        lock files (``~$...``) are skipped.
    """
    folder = Path(folder) if folder is not None else REFLECTANCE_PROFILE_DIR
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in ('.xlsx', '.xls') and not p.name.startswith('~$')),
                  key=lambda p: p.name.lower())

def _apply_spatial_binning(img, n, to_uint16=False):
    """Average spatial blocks, dropping incomplete edge blocks as before."""
    if n <= 1:
        return img
    h, w = img.shape
    h_new, w_new = h // n, w // n
    img_cropped = img[:h_new * n, :w_new * n]
    reshaped = img_cropped.reshape(h_new, n, w_new, n)
    binned = reshaped.mean(axis=(1, 3))
    if to_uint16:
        return np.round(binned).astype(np.uint16)
    return binned.astype(np.float32)


def rowwise_calculation_heavy_logic(ref_filepath, ref_meta, ref_excel,
                                    avg_window_size, mask,
                                    progress_callback=None):
    """Fit reference-spectrum factors per image row using the selected ROI."""
    progress_callback = progress_callback or (lambda value: None)
    data_ref = pd.read_excel(ref_excel, header=None).to_numpy()
    f_ref = CubicSpline(data_ref[:, 0], data_ref[:, 1])
    wl = ref_meta['wavelength']

    H, W = ref_meta['lines'], ref_meta['samples']
    if mask.shape != (H, W):
        raise ValueError('ROI mask shape does not match reference image.')

    rows_has_roi = np.where(mask.any(axis=1))[0]
    if len(rows_has_roi) == 0:
        raise ValueError('Selected ROI does not cover any pixels.')
    top_roi, bot_roi = int(rows_has_roi.min()), int(rows_has_roi.max())

    def pick_rows_in_roi(y):
        nearest = rows_has_roi[np.argmin(np.abs(rows_has_roi - y))]
        idx = np.where(rows_has_roi == nearest)[0][0]
        left = max(0, idx - (avg_window_size // 2))
        right = min(len(rows_has_roi), left + avg_window_size)
        if right - left < avg_window_size:
            left = max(0, right - avg_window_size)
        return rows_has_roi[left:right]

    rowwise_factors = np.zeros((H, len(wl)), dtype=np.float32)
    R_ref_per_band = f_ref(wl).astype(np.float32)

    progress_callback(0)
    for i in range(ref_meta['bands']):
        ref_img = _read_envi_band_float32(ref_filepath, ref_meta, i)
        ref_roi = np.where(mask[:, :ref_img.shape[1]], ref_img[:, :ref_img.shape[1]], np.nan)
        fallback = np.nanmedian(ref_roi)
        if not np.isfinite(fallback) or fallback <= 0:
            fallback = 1.0

        for y in range(ref_roi.shape[0]):
            y_ref = int(np.clip(y, top_roi, bot_roi))
            rows_window = pick_rows_in_roi(y_ref)
            mean_val = np.nanmean(ref_roi[rows_window, :]) if len(rows_window) > 0 else fallback
            if not np.isfinite(mean_val) or mean_val <= 0:
                mean_val = fallback
            rowwise_factors[y, i] = R_ref_per_band[i] / mean_val

        progress_callback(i + 1)
    return {'factors': rowwise_factors, 'span': (top_roi, bot_roi)}


def create_datacube_logic(params, progress_callback=None):
    """Crop, apply factors and spatially bin an ENVI cube; write BIP output."""
    progress_callback = progress_callback or (lambda value: None)
    raw_filepath = params['raw_filepath']
    raw_meta = params['raw_meta']
    save_path = params['save_path']
    x0, x1, y0, y1 = params['crop_coords']
    is_wavelength = params['is_wavelength']
    band_count = params['band_count']
    factors = params['factors']
    rowwise_factors = params['rowwise_factors']
    n_bin = params.get('binning_factor', 1)

    cubheight, cubwidth = (y1 - y0) // n_bin, (x1 - x0) // n_bin
    is_cal = (rowwise_factors is not None) or (factors is not None)
    in_dt = int(raw_meta.get('data type', 12))
    if is_cal:
        out_dt, out_dtype = 4, np.float32
    else:
        out_dt, out_dtype = in_dt, (np.float32 if in_dt == 4 else np.uint16)

    hdr_path = save_path + '.hdr'
    with open(hdr_path, 'w') as hopen:
        hopen.write('ENVI\n')
        old_desc = raw_meta.get('description', 'Processed Data')
        bin_text = f"Spatial Binning = {n_bin}x{n_bin}" if n_bin > 1 else "Original Spatial Res"
        hopen.write(f"description = {{\n  {old_desc},\n  {bin_text}\n}}\n")
        hopen.write(f'samples = {cubwidth}\n')
        hopen.write(f'lines = {cubheight}\n')
        hopen.write(f'bands = {band_count}\n')
        hopen.write(f'data type = {out_dt}\n')
        hopen.write('interleave = bip\n')
        hopen.write('byte order = 0\n')
        hopen.write('wavelength units = Nanometers\n')
        if is_wavelength and raw_meta.get('wavelength') is not None:
            wl_str = ','.join([str(item) for item in raw_meta['wavelength']])
            hopen.write(f'wavelength = {{{wl_str}}}\n')

    c = np.zeros((cubheight * band_count, cubwidth), dtype=out_dtype)
    progress_callback(0)

    for i in range(band_count):
        im = _read_envi_band_float32(raw_filepath, raw_meta, i)
        roi_img = im[y0:y1, x0:x1]
        if rowwise_factors is not None:
            roi_img *= rowwise_factors[y0:y1, i].reshape(-1, 1)
        elif factors is not None:
            roi_img *= float(factors[i])
        out = _apply_spatial_binning(
            roi_img, n_bin, to_uint16=(out_dtype == np.uint16))
        c[cubheight * i: cubheight * (i + 1)] = out
        progress_callback(i + 1)

    bip = c.reshape((band_count, cubheight, cubwidth)).transpose(1, 2, 0)  # HWC
    bip.tofile(save_path)
    return save_path


def create_datacube_resampled_logic(params, progress_callback=None):
    """Apply factors and linearly resample bands before spatial binning."""
    progress_callback = progress_callback or (lambda value: None)
    raw_filepath = params['raw_filepath']
    raw_meta = params['raw_meta']
    save_path = params['save_path']
    x0, x1, y0, y1 = params['crop_coords']
    is_wavelength = params['is_wavelength']
    factors = params['factors']
    rowwise_factors = params['rowwise_factors']
    lam_target = np.array(params['lam_target'], dtype=np.float32)
    n_bin = params.get('binning_factor', 1)

    wl_all = raw_meta['wavelength']
    idx_right = np.searchsorted(wl_all, lam_target, side='left')
    idx_left = np.clip(idx_right - 1, 0, len(wl_all) - 1)
    idx_right = np.clip(idx_right, 0, len(wl_all) - 1)
    denom = np.maximum(wl_all[idx_right] - wl_all[idx_left], 1e-6)
    alpha = (lam_target - wl_all[idx_left]) / denom

    cubheight, cubwidth = (y1 - y0) // n_bin, (x1 - x0) // n_bin
    is_cal = (rowwise_factors is not None) or (factors is not None)
    in_dt = int(raw_meta.get('data type', 12))
    if is_cal:
        out_dt, out_dtype_str, out_dtype_np = 4, 'float32', np.float32
    else:
        out_dt = in_dt
        out_dtype_str = 'float32' if in_dt == 4 else 'uint16'
        out_dtype_np = np.float32 if in_dt == 4 else np.uint16

    hdr_path = save_path + '.hdr'
    with open(hdr_path, 'w') as hopen:
        hopen.write('ENVI\n')
        old_desc = raw_meta.get('description', 'Processed Data')
        bin_text = f"Spatial Binning = {n_bin}x{n_bin}" if n_bin > 1 else "Original Spatial Res"
        hopen.write(f"description = {{\n  {old_desc},\n  Resampled = True,\n  {bin_text}\n}}\n")
        hopen.write(f'samples = {cubwidth}\n')
        hopen.write(f'lines = {cubheight}\n')
        hopen.write(f'bands = {len(lam_target)}\n')
        hopen.write(f'data type = {out_dt}\n')
        hopen.write('interleave = bip\n')
        hopen.write('byte order = 0\n')
        hopen.write('wavelength units = Nanometers\n')
        if is_wavelength:
            wl_str = ','.join(f"{float(v):.2f}" for v in lam_target)
            hopen.write(f'wavelength = {{{wl_str}}}\n')

    c = np.zeros((cubheight * len(lam_target), cubwidth), dtype=out_dtype_str)
    progress_callback(0)

    for j in range(len(lam_target)):
        iL, iR, a = int(idx_left[j]), int(idx_right[j]), float(alpha[j])
        imL = _read_envi_band_float32(raw_filepath, raw_meta, iL)
        imR = _read_envi_band_float32(raw_filepath, raw_meta, iR)

        Wmin = min(imL.shape[1], imR.shape[1])
        roiL = imL[y0:y1, x0:min(x1, x0 + Wmin)]
        roiR = imR[y0:y1, x0:min(x1, x0 + Wmin)]

        if rowwise_factors is not None:
            roiL *= rowwise_factors[y0:y1, iL].reshape(-1, 1)
            roiR *= rowwise_factors[y0:y1, iR].reshape(-1, 1)
        elif factors is not None:
            roiL *= float(factors[iL])
            roiR *= float(factors[iR])

        roiJ = (1.0 - a) * roiL + a * roiR
        out = _apply_spatial_binning(
            roiJ, n_bin, to_uint16=(out_dtype_np == np.uint16))
        c[cubheight * j: cubheight * (j + 1)] = out
        progress_callback(j + 1)

    bip = c.reshape((len(lam_target), cubheight, cubwidth)).transpose(1, 2, 0)  # HWC
    bip.tofile(save_path)
    return save_path


def build_raw_cube_tmp(params, progress_callback=None):
    """Build the original uint16 cropped temporary band-stack memmap."""
    progress_callback = progress_callback or (lambda value: None)
    raw_filepath = params['folder']
    raw_meta = params['raw_meta']
    tmp_path = params['save_path']
    x0, x1, y0, y1 = params['crop']

    lines, samples = (y1 - y0), (x1 - x0)
    bands = raw_meta['bands']
    mm = np.memmap(tmp_path, mode='w+', dtype=np.uint16, shape=(lines * bands, samples))

    progress_callback(0)
    for i in range(bands):
        im = _read_envi_band_float32(raw_filepath, raw_meta, i)
        roi = im[y0:y1, x0:x1]
        mm[lines * i: lines * (i + 1)] = np.clip(roi, 0, 65535).astype(np.uint16)
        progress_callback(i + 1)
    mm.flush()
    return {'path': tmp_path, 'lines': lines, 'samples': samples,
            'bands': bands, 'dtype': 'uint16'}


def build_cal_cube_tmp(params, progress_callback=None):
    """Build a float32 calibrated temporary band-stack memmap."""
    progress_callback = progress_callback or (lambda value: None)
    raw_filepath = params['raw_filepath']
    raw_meta = params['raw_meta']
    tmp_path = params['save_path']
    x0, x1, y0, y1 = params['crop']
    factors = params.get('factors', None)
    rowwise = params.get('rowwise_factors', None)

    lines, samples = (y1 - y0), (x1 - x0)
    bands = raw_meta['bands']
    mm = np.memmap(tmp_path, mode='w+', dtype=np.float32, shape=(lines * bands, samples))

    progress_callback(0)
    for i in range(bands):
        im = _read_envi_band_float32(raw_filepath, raw_meta, i)
        roi = im[y0:y1, x0:x1]
        if rowwise is not None:
            fac = rowwise[y0:y1, i].reshape(-1, 1)
            roi = roi * fac
        elif factors is not None:
            roi = roi * float(factors[i])
        mm[lines * i: lines * (i + 1)] = roi.astype(np.float32)
        progress_callback(i + 1)
    mm.flush()
    return {'path': tmp_path, 'lines': lines, 'samples': samples,
            'bands': bands, 'dtype': 'float32'}


def fluo_calculation_heavy_logic(ref_filepath, ref_meta, mask, progress_callback=None):
    """Invert the smoothed ROI-average spectrum to form fluorescence factors."""
    progress_callback = progress_callback or (lambda value: None)
    num_bands = ref_meta['bands']
    avg_ref_spectrum = np.zeros(num_bands, dtype=np.float32)

    progress_callback(0)
    for i in range(num_bands):
        img = _read_envi_band_float32(ref_filepath, ref_meta, i)
        roi_data = img[mask[:, :img.shape[1]]]
        avg_ref_spectrum[i] = np.nanmean(roi_data) if roi_data.size > 0 else 1.0
        progress_callback(i + 1)

    try:
        smoothed = savgol_filter(avg_ref_spectrum, window_length=11, polyorder=3)
        smoothed = np.where(smoothed <= 0, 1.0, smoothed)
    except Exception:
        smoothed = np.where(avg_ref_spectrum <= 0, 1.0, avg_ref_spectrum)

    return {'factors': 1.0 / smoothed}


