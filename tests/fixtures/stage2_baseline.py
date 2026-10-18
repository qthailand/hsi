"""Frozen numerical baseline from supplied HSI for extraction regression."""
import os,re
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
class CalibrationWindow: pass
def _apply_spatial_binning(img, n, to_uint16=False):
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

def parse_envi_header(filepath):
    hdr_path = filepath + '.hdr'
    if not os.path.exists(hdr_path):
        hdr_path = os.path.splitext(filepath)[0] + '.hdr'
    if not os.path.exists(hdr_path):
        raise ValueError(f"Header file not found for {filepath}")

    meta = {'data_type': 4, 'byte_order': 0,
            'interleave': 'bsq', 'description': 'Processed Data'}
    with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
        txt = f.read()

    m_desc = re.search(r'description\s*=\s*\{([^}]*)\}', txt, re.IGNORECASE | re.DOTALL)
    if m_desc:
        meta['description'] = m_desc.group(1).strip().replace('\n', ' ')
    else:
        m_desc_simple = re.search(
            r'^description\s*=\s*(.*)$', txt, re.IGNORECASE | re.MULTILINE)
        if m_desc_simple:
            meta['description'] = m_desc_simple.group(1).strip()

    for line in txt.split('\n'):
        if '=' in line and 'description' not in line.lower() and 'wavelength' not in line.lower():
            k, v = line.split('=', 1)
            meta[k.strip().lower()] = v.strip()

    m_wl = re.search(r'wavelength\s*=\s*\{([^}]*)\}', txt, re.IGNORECASE | re.DOTALL)
    if m_wl:
        vals = m_wl.group(1).replace('\n', ' ').split(',')
        meta['wavelength'] = np.array(
            [float(v.strip()) for v in vals if v.strip() != ''], dtype=np.float32)
    else:
        meta['wavelength'] = None

    meta['lines'] = int(meta['lines'])
    meta['samples'] = int(meta['samples'])
    meta['bands'] = int(meta['bands'])
    return meta

def _read_envi_band_float32(filepath, meta, band_idx):
    L, S, B = meta['lines'], meta['samples'], meta['bands']
    dt_code = int(meta.get('data type', meta.get('data_type', '4')))
    dt_map = {12: (np.uint16, 2), 4: (np.float32, 4), 2: (np.int16, 2),
            1: (np.uint8, 1), 5: (np.float64, 8)}
    dt, bpp = dt_map.get(dt_code, (np.float32, 4))

    interleave = str(meta.get('interleave', 'bsq')).lower().strip()

    with open(filepath, 'rb') as f:
        if interleave == 'bsq':
            f.seek(band_idx * L * S * bpp)
            raw = f.read(L * S * bpp)
            return np.frombuffer(raw, dtype=dt).reshape(L, S).astype(np.float32)

        elif interleave == 'bip':
            data = np.frombuffer(f.read(), dtype=dt).reshape(L, S, B)
            return data[:, :, band_idx].astype(np.float32)

        elif interleave == 'bil':
            data = np.frombuffer(f.read(), dtype=dt).reshape(L, B, S)
            return data[:, band_idx, :].astype(np.float32)

        else:
            raise ValueError(f"Unsupported interleave: {interleave}")

def rowwise_calculation_heavy_logic(ref_filepath, ref_meta, ref_excel,
                                    avg_window_size, mask,
                                    progress_callback):
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

    progress_callback.emit(0)
    for i in range(ref_meta['bands']):
        ref_img = CalibrationWindow._read_envi_band_float32(ref_filepath, ref_meta, i)
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

        progress_callback.emit(i + 1)
    return {'factors': rowwise_factors, 'span': (top_roi, bot_roi)}

def create_datacube_logic(params, progress_callback):
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
    progress_callback.emit(0)

    for i in range(band_count):
        im = CalibrationWindow._read_envi_band_float32(raw_filepath, raw_meta, i)
        roi_img = im[y0:y1, x0:x1]
        if rowwise_factors is not None:
            roi_img *= rowwise_factors[y0:y1, i].reshape(-1, 1)
        elif factors is not None:
            roi_img *= float(factors[i])
        out = CalibrationWindow._apply_spatial_binning(
            roi_img, n_bin, to_uint16=(out_dtype == np.uint16))
        c[cubheight * i: cubheight * (i + 1)] = out
        progress_callback.emit(i + 1)

    bip = c.reshape((band_count, cubheight, cubwidth)).transpose(1, 2, 0)  # HWC
    bip.tofile(save_path)
    return save_path

def create_datacube_resampled_logic(params, progress_callback):
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
    progress_callback.emit(0)

    for j in range(len(lam_target)):
        iL, iR, a = int(idx_left[j]), int(idx_right[j]), float(alpha[j])
        imL = CalibrationWindow._read_envi_band_float32(raw_filepath, raw_meta, iL)
        imR = CalibrationWindow._read_envi_band_float32(raw_filepath, raw_meta, iR)

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
        out = CalibrationWindow._apply_spatial_binning(
            roiJ, n_bin, to_uint16=(out_dtype_np == np.uint16))
        c[cubheight * j: cubheight * (j + 1)] = out
        progress_callback.emit(j + 1)

    bip = c.reshape((len(lam_target), cubheight, cubwidth)).transpose(1, 2, 0)  # HWC
    bip.tofile(save_path)
    return save_path

def build_raw_cube_tmp(params, progress_callback):
    raw_filepath = params['folder']
    raw_meta = params['raw_meta']
    tmp_path = params['save_path']
    x0, x1, y0, y1 = params['crop']

    lines, samples = (y1 - y0), (x1 - x0)
    bands = raw_meta['bands']
    mm = np.memmap(tmp_path, mode='w+', dtype=np.uint16, shape=(lines * bands, samples))

    progress_callback.emit(0)
    for i in range(bands):
        im = CalibrationWindow._read_envi_band_float32(raw_filepath, raw_meta, i)
        roi = im[y0:y1, x0:x1]
        mm[lines * i: lines * (i + 1)] = np.clip(roi, 0, 65535).astype(np.uint16)
        progress_callback.emit(i + 1)
    mm.flush()
    return {'path': tmp_path, 'lines': lines, 'samples': samples,
            'bands': bands, 'dtype': 'uint16'}

def build_cal_cube_tmp(params, progress_callback):
    raw_filepath = params['raw_filepath']
    raw_meta = params['raw_meta']
    tmp_path = params['save_path']
    x0, x1, y0, y1 = params['crop']
    factors = params.get('factors', None)
    rowwise = params.get('rowwise_factors', None)

    lines, samples = (y1 - y0), (x1 - x0)
    bands = raw_meta['bands']
    mm = np.memmap(tmp_path, mode='w+', dtype=np.float32, shape=(lines * bands, samples))

    progress_callback.emit(0)
    for i in range(bands):
        im = CalibrationWindow._read_envi_band_float32(raw_filepath, raw_meta, i)
        roi = im[y0:y1, x0:x1]
        if rowwise is not None:
            fac = rowwise[y0:y1, i].reshape(-1, 1)
            roi = roi * fac
        elif factors is not None:
            roi = roi * float(factors[i])
        mm[lines * i: lines * (i + 1)] = roi.astype(np.float32)
        progress_callback.emit(i + 1)
    mm.flush()
    return {'path': tmp_path, 'lines': lines, 'samples': samples,
            'bands': bands, 'dtype': 'float32'}

def fluo_calculation_heavy_logic(ref_filepath, ref_meta, mask, progress_callback):
    num_bands = ref_meta['bands']
    avg_ref_spectrum = np.zeros(num_bands, dtype=np.float32)

    progress_callback.emit(0)
    for i in range(num_bands):
        img = CalibrationWindow._read_envi_band_float32(ref_filepath, ref_meta, i)
        roi_data = img[mask[:, :img.shape[1]]]
        avg_ref_spectrum[i] = np.nanmean(roi_data) if roi_data.size > 0 else 1.0
        progress_callback.emit(i + 1)

    try:
        smoothed = savgol_filter(avg_ref_spectrum, window_length=11, polyorder=3)
        smoothed = np.where(smoothed <= 0, 1.0, smoothed)
    except Exception:
        smoothed = np.where(avg_ref_spectrum <= 0, 1.0, avg_ref_spectrum)

    return {'factors': 1.0 / smoothed}

CalibrationWindow._apply_spatial_binning=staticmethod(_apply_spatial_binning)
CalibrationWindow.parse_envi_header=staticmethod(parse_envi_header)
CalibrationWindow._read_envi_band_float32=staticmethod(_read_envi_band_float32)
CalibrationWindow.rowwise_calculation_heavy_logic=staticmethod(rowwise_calculation_heavy_logic)
CalibrationWindow.create_datacube_logic=staticmethod(create_datacube_logic)
CalibrationWindow.create_datacube_resampled_logic=staticmethod(create_datacube_resampled_logic)
CalibrationWindow.build_raw_cube_tmp=staticmethod(build_raw_cube_tmp)
CalibrationWindow.build_cal_cube_tmp=staticmethod(build_cal_cube_tmp)
CalibrationWindow.fluo_calculation_heavy_logic=staticmethod(fluo_calculation_heavy_logic)
