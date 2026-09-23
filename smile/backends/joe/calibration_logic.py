import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN


def gaussian_func(x, a, mu, sigma, c):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c


def get_peak_slope_method(img, row_idx, sigma, thresh, x_start, x_end):
    row_data = img[row_idx, :].astype(float)
    smoothed = gaussian_filter1d(row_data, sigma=sigma)
    roi_segment = smoothed[x_start:x_end]
    diff = np.diff(roi_segment)
    indices = np.where((diff[:-1] > 0) & (diff[1:] < 0))[0] + 1
    return [idx + x_start for idx in indices if roi_segment[idx] > thresh]


def detect_peaks_across_rows(img, sample_rows, sigma, thresh, x_start, x_end):
    raw_peaks = []
    for y in sample_rows:
        peaks = get_peak_slope_method(img, y, sigma, thresh, x_start, x_end)
        for p in peaks:
            raw_peaks.append([p, y])
    return np.array(raw_peaks) if raw_peaks else np.empty((0, 2), dtype=float)


def cluster_peak_lines(raw_peaks_arr, eps, min_samples):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(raw_peaks_arr[:, 0:1])
    labels = clustering.labels_
    return labels


def build_candidate_lines(raw_peaks_arr, labels, min_pts_line):
    unique_labels = sorted([l for l in set(labels) if l != -1])
    candidate_lines = []
    for l in unique_labels:
        pts = raw_peaks_arr[labels == l]
        if len(pts) < min_pts_line:
            continue
        coeffs_raw = np.polyfit(pts[:, 1], pts[:, 0], 2)
        candidate_lines.append({'label': l, 'pts': pts, 'coeffs': coeffs_raw, 'a': coeffs_raw[0]})
    return candidate_lines


def remove_outlier_lines(candidate_lines, sigma_threshold=2.5):
    if not candidate_lines:
        return []
    all_a = np.array([line['a'] for line in candidate_lines])
    mu_a, std_a = np.mean(all_a), np.std(all_a)
    return [line for line in candidate_lines if abs(line['a'] - mu_a) <= sigma_threshold * std_a]


def refine_parabolic_peaks(raw_img, valid_candidates, w):
    detected_peaks_coords = []
    for line in valid_candidates:
        refined = []
        for px, py in line['pts']:
            ix = int(px)
            if 5 < ix < w - 5:
                xf = np.arange(ix - 5, ix + 6)
                yf = raw_img[py, ix - 5:ix + 6].astype(float)
                try:
                    p0 = [np.max(yf), px, 1.5, np.min(yf)]
                    popt, _ = curve_fit(gaussian_func, xf, yf, p0=p0, maxfev=200)
                    refined.append([popt[1], py])
                except Exception:
                    refined.append([px, py])
        if not refined:
            continue
        refined = np.array(refined)
        coeffs_final = np.polyfit(refined[:, 1], refined[:, 0], 2)
        detected_peaks_coords.append({'label': line['label'], 'pts': refined, 'coeffs': coeffs_final})
    return detected_peaks_coords


def build_trend_models(detected_peaks_coords, h):
    line_models = np.array([[np.polyval(d['coeffs'], h / 2), d['coeffs'][0], d['coeffs'][1]] for d in detected_peaks_coords])
    line_models = line_models[line_models[:, 0].argsort()]
    trend_a = np.poly1d(np.polyfit(line_models[:, 0], line_models[:, 1], 1))
    trend_b = np.poly1d(np.polyfit(line_models[:, 0], line_models[:, 2], 1))
    return trend_a, trend_b


def generate_smile_maps(w, h, trend_a, trend_b, keystone_model=None):
    gy, gx = np.mgrid[0:h, 0:w]
    sA, sB = trend_a(gx), trend_b(gx)
    sC = gx - (sA * (h / 2) ** 2 + sB * (h / 2))
    mx = (sA * gy ** 2 + sB * gy + sC).astype(np.float32)
    my = gy.astype(np.float32)
    if keystone_model is not None:
        my = (gy / keystone_model(gx)).astype(np.float32)
    return mx, my


def remap_image(raw_img, mx, my, interpolation=cv2.INTER_CUBIC):
    return cv2.remap(raw_img, mx, my, interpolation)


def compute_diagnostic_results(corrected_img, detected_peaks_coords, sigma, thresh, num_samples, min_pts_line, dbscan_eps=10, dbscan_min=5):
    h, w = corrected_img.shape
    sample_rows = np.linspace(10, h - 10, num_samples, dtype=int)
    new_raw_peaks = detect_peaks_across_rows(corrected_img, sample_rows, sigma, thresh, 0, w)
    if new_raw_peaks.size == 0:
        return []
    labels = cluster_peak_lines(new_raw_peaks, dbscan_eps, dbscan_min)
    diag_results = []
    for nl in sorted([l for l in set(labels) if l != -1]):
        pts = new_raw_peaks[labels == nl]
        refined_post = []
        for px, py in pts:
            ix = int(px)
            if 5 < ix < w - 5:
                xf = np.arange(ix - 5, ix + 6)
                yf = corrected_img[py, ix - 5:ix + 6].astype(float)
                try:
                    popt, _ = curve_fit(gaussian_func, xf, yf, p0=[np.max(yf), px, 1.5, np.min(yf)], maxfev=200)
                    refined_post.append([popt[1], py])
                except Exception:
                    refined_post.append([px, py])
        if len(refined_post) < min_pts_line:
            continue
        refined_post = np.array(refined_post)
        post_coeffs = np.polyfit(refined_post[:, 1], refined_post[:, 0], 2)
        cur_x = np.polyval(post_coeffs, h / 2)
        best_match = None
        min_dist = 50
        for orig in detected_peaks_coords:
            dist = abs(cur_x - np.polyval(orig['coeffs'], h / 2))
            if dist < min_dist:
                min_dist = dist
                best_match = orig
        if best_match is None:
            continue
        orig_coeffs = best_match['coeffs']
        orig_res = best_match['pts'][:, 0] - np.polyval(orig_coeffs, best_match['pts'][:, 1])
        post_res = refined_post[:, 0] - np.polyval(post_coeffs, refined_post[:, 1])
        diag_results.append({
            'x': cur_x,
            'label': best_match['label'],
            'orig_coeffs': orig_coeffs,
            'post_coeffs': post_coeffs,
            'orig_res': orig_res,
            'post_res': post_res,
            'post_y': refined_post[:, 1],
        })
    return diag_results


def detect_keystone_edges(ref_img, sigma=1.5, height_ratio=0.3, distance=30, sample_count=25):
    h, w = ref_img.shape
    sample_x = np.linspace(100, w - 100, sample_count, dtype=int)
    all_pts = []
    for x in sample_x:
        col_data = ref_img[:, x].astype(float)
        grad = np.diff(gaussian_filter1d(col_data, sigma=sigma))
        t_peaks, _ = find_peaks(grad, height=np.max(grad) * height_ratio, distance=distance)
        for p in t_peaks:
            all_pts.append([x, p, 1])
        b_peaks, _ = find_peaks(-grad, height=np.max(-grad) * height_ratio, distance=distance)
        for p in b_peaks:
            all_pts.append([x, p, -1])
    if not all_pts:
        return []
    pts_arr = np.array(all_pts)
    labels = DBSCAN(eps=15, min_samples=5).fit(pts_arr[:, 1:2]).labels_
    raw_edges = []
    for l in sorted([l for l in set(labels) if l != -1]):
        pts = pts_arr[labels == l]
        raw_edges.append({
            'avg_y': np.mean(pts[:, 1]),
            'type': 1 if np.mean(pts[:, 2]) > 0 else -1,
            'poly': np.poly1d(np.polyfit(pts[:, 0], pts[:, 1], 1))
        })
    raw_edges.sort(key=lambda x: x['avg_y'])
    valid_edges = []
    i = 0
    while i < len(raw_edges) - 1:
        if raw_edges[i]['type'] == 1 and raw_edges[i + 1]['type'] == -1:
            valid_edges.extend([raw_edges[i], raw_edges[i + 1]])
            i += 2
            continue
        i += 1
    return valid_edges


def build_keystone_model(valid_edges, width, sample_count=50):
    if len(valid_edges) < 2:
        return None
    pairs = [(valid_edges[i], valid_edges[i + 1]) for i in range(len(valid_edges) - 1)
             if valid_edges[i]['type'] == 1 and valid_edges[i + 1]['type'] == -1]
    if not pairs:
        return None
    sample_x = np.linspace(0, width, sample_count)
    spans = [[x, np.mean([pair[1]['poly'](x) - pair[0]['poly'](x) for pair in pairs])] for x in sample_x]
    dist_data = np.array(spans)
    if np.mean(dist_data[:, 1]) == 0:
        return None
    return np.poly1d(np.polyfit(dist_data[:, 0], dist_data[:, 1] / np.mean(dist_data[:, 1]), 1))
