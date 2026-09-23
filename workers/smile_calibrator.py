# workers/smile_calibrator.py
"""Generate smile correction maps from a raw calibration frame.

Uses calibration_logic from smile.backends.joe to detect spectral lines,
build trend models, and generate cv2.remap maps.

Usage:
    calibrator = SmileCalibrator(output_folder="smile_maps/")
    calibrator.calibrate(raw_frame, bin_value=4)
    # → smile_maps/smile_map_x_bin4.npy
    # → smile_maps/smile_map_y_bin4.npy
"""

from __future__ import annotations

import os
import logging

import cv2
import numpy as np

from smile.backends.joe.calibration_logic import (
    detect_peaks_across_rows,
    cluster_peak_lines,
    build_candidate_lines,
    remove_outlier_lines,
    refine_parabolic_peaks,
    build_trend_models,
    generate_smile_maps,
)

logger = logging.getLogger("HSI")


class SmileCalibrator:
    """Generate smile correction maps from a raw calibration frame.

    Default parameters tuned for HSI spectral line detection.
    Override via constructor or method arguments.
    """

    def __init__(self, output_folder: str = "smile_maps/"):
        self._output_folder = output_folder

        # ★ default detection parameters
        self.sample_rows_count = 40
        self.dbscan_eps = 10.0
        self.dbscan_min_samples = 5
        self.min_pts_per_line = 15
        self.peak_threshold = 25
        self.smoothing_sigma = 5.0
        self.roi_x_start = 0
        self.roi_x_end = -1  # -1 = auto (full width)

    @property
    def output_folder(self) -> str:
        return self._output_folder

    def set_output_folder(self, folder: str) -> None:
        self._output_folder = folder

    def calibrate(
        self,
        frame: np.ndarray,
        bin_value: int,
        # ★ override parameters ได้
        sample_rows_count: int | None = None,
        dbscan_eps: float | None = None,
        dbscan_min_samples: int | None = None,
        min_pts_per_line: int | None = None,
        peak_threshold: int | None = None,
        smoothing_sigma: float | None = None,
        roi_x_start: int | None = None,
        roi_x_end: int | None = None,
    ) -> bool:
        """Run full smile calibration on a single frame.

        Args:
            frame: 2D raw calibration frame (h, w)
            bin_value: binning factor (used in filename)

        Returns:
            True if maps generated successfully
        """
        if frame is None or frame.ndim != 2:
            logger.error("Calibration requires a 2D frame")
            return False

        h, w = frame.shape

        # ★ use defaults or overrides
        n_samples = sample_rows_count or self.sample_rows_count
        eps = dbscan_eps if dbscan_eps is not None else self.dbscan_eps
        min_samp = dbscan_min_samples if dbscan_min_samples is not None else self.dbscan_min_samples
        min_pts = min_pts_per_line or self.min_pts_per_line
        thresh = peak_threshold or self.peak_threshold
        sigma = smoothing_sigma if smoothing_sigma is not None else self.smoothing_sigma
        x_start = roi_x_start if roi_x_start is not None else self.roi_x_start
        x_end = roi_x_end if roi_x_end is not None and roi_x_end > 0 else w

        logger.info(
            "Smile calibration: frame=%dx%d bin=%d samples=%d eps=%.1f "
            "min_samp=%d min_pts=%d thresh=%d sigma=%.1f roi=[%d:%d]",
            h, w, bin_value, n_samples, eps, min_samp, min_pts, thresh, sigma, x_start, x_end,
        )

        try:
            # ── Step 1: detect peaks across sample rows ──
            sample_rows = np.linspace(10, h - 10, n_samples, dtype=int)
            raw_peaks = detect_peaks_across_rows(
                frame, sample_rows, sigma, thresh, x_start, x_end
            )

            if raw_peaks.size == 0:
                logger.error("No peaks detected — check threshold/sigma")
                return False

            logger.info("Detected %d raw peak points", len(raw_peaks))

            # ── Step 2: cluster into spectral lines ──
            labels = cluster_peak_lines(raw_peaks, eps, min_samp)
            n_lines = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info("Found %d spectral line clusters", n_lines)

            if n_lines == 0:
                logger.error("No spectral line clusters found — check DBSCAN params")
                return False

            # ── Step 3: build candidate lines ──
            candidates = build_candidate_lines(raw_peaks, labels, min_pts)
            logger.info("Candidate lines: %d (min_pts=%d)", len(candidates), min_pts)

            if not candidates:
                logger.error("No candidate lines passed min_pts filter")
                return False

            # ── Step 4: remove outlier lines ──
            valid = remove_outlier_lines(candidates, sigma_threshold=2.5)
            logger.info("Valid lines after outlier removal: %d", len(valid))

            if not valid:
                logger.error("All lines removed as outliers")
                return False

            # ── Step 5: refine with sub-pixel Gaussian fitting ──
            refined = refine_parabolic_peaks(frame, valid, w)
            logger.info("Refined spectral lines: %d", len(refined))

            if not refined:
                logger.error("Sub-pixel refinement failed")
                return False

            # ── Step 6: build global trend models ──
            trend_a, trend_b = build_trend_models(refined, h)

            # ── Step 7: generate remap maps ──
            map_x, map_y = generate_smile_maps(w, h, trend_a, trend_b)

            # ── Step 8: save ──
            os.makedirs(self._output_folder, exist_ok=True)

            path_x = os.path.join(
                self._output_folder, f"smile_map_x_bin{bin_value}.npy"
            )
            path_y = os.path.join(
                self._output_folder, f"smile_map_y_bin{bin_value}.npy"
            )

            np.save(path_x, map_x)
            np.save(path_y, map_y)

            logger.info("Smile maps saved: %s, %s", path_x, path_y)
            logger.info("Map shape: x=%s y=%s", map_x.shape, map_y.shape)

            return True

        except Exception as e:
            logger.error("Smile calibration failed: %s", e)
            return False