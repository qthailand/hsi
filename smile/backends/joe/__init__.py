import numpy as np
from typing import Optional
from ..base import SmileBackend, SmileParams
from .calibration_logic import (
    detect_peaks_across_rows,
    cluster_peak_lines,
    build_candidate_lines,
    remove_outlier_lines,
    refine_parabolic_peaks,
    build_trend_models,
    generate_smile_maps,
    remap_image,
    compute_diagnostic_results,
    detect_keystone_edges,
    build_keystone_model,
)


class JoeSmileBackend(SmileBackend):
    def detect_peaks(self, raw_img: np.ndarray, params: SmileParams):
        h, w = raw_img.shape
        x_end = params.x_end if params.x_end is not None else w
        sample_rows = np.linspace(10, h - 10, params.sample_rows, dtype=int)
        raw_peaks_arr = detect_peaks_across_rows(raw_img, sample_rows, params.sigma, params.thresh, params.x_start, x_end)
        if raw_peaks_arr.size == 0:
            return []

        labels = cluster_peak_lines(raw_peaks_arr, params.dbscan_eps, params.dbscan_min)
        candidate_lines = build_candidate_lines(raw_peaks_arr, labels, params.min_pts_line)
        valid_candidates = remove_outlier_lines(candidate_lines)
        return refine_parabolic_peaks(raw_img, valid_candidates, w)

    def build_correction(self, raw_img: np.ndarray, detected_peaks, keystone_model=None):
        if not detected_peaks:
            return None, None
        trend_a, trend_b = build_trend_models(detected_peaks, raw_img.shape[0])
        return generate_smile_maps(raw_img.shape[1], raw_img.shape[0], trend_a, trend_b, keystone_model)

    def remap(self, raw_img: np.ndarray, mx, my):
        return remap_image(raw_img, mx, my)

    def diagnose(self, corrected_img: np.ndarray, detected_peaks, params: SmileParams):
        return compute_diagnostic_results(
            corrected_img,
            detected_peaks,
            params.sigma,
            params.thresh,
            params.sample_rows,
            params.min_pts_line,
            dbscan_eps=params.dbscan_eps,
            dbscan_min=params.dbscan_min,
        )

    def create_keystone_model(self, keystone_ref_img: np.ndarray):
        valid_edges = detect_keystone_edges(keystone_ref_img)
        return build_keystone_model(valid_edges, keystone_ref_img.shape[1])
