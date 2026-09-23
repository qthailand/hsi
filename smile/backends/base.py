from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SmileParams:
    x_start: int = 0
    x_end: Optional[int] = None
    thresh: int = 25
    sigma: float = 5.0
    sample_rows: int = 40
    dbscan_eps: float = 10.0
    dbscan_min: int = 5
    min_pts_line: int = 15


class SmileBackend(ABC):
    @abstractmethod
    def detect_peaks(self, raw_img: np.ndarray, params: SmileParams):
        """Detect spectral peak lines and return the line models."""
        raise NotImplementedError

    @abstractmethod
    def build_correction(self, raw_img: np.ndarray, detected_peaks, keystone_model=None):
        """Build remapping correction maps from detected peak data."""
        raise NotImplementedError

    @abstractmethod
    def remap(self, raw_img: np.ndarray, mx, my):
        """Remap the input image using the computed maps."""
        raise NotImplementedError

    @abstractmethod
    def diagnose(self, corrected_img: np.ndarray, detected_peaks, params: SmileParams):
        """Run diagnostics on the corrected image."""
        raise NotImplementedError

    @abstractmethod
    def create_keystone_model(self, keystone_ref_img: np.ndarray):
        """Create a keystone correction model from reference imagery."""
        raise NotImplementedError
