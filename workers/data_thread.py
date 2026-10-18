from modules.spectral_processing import LambdaBinner
import logging
import time

import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal as Signal
from PyQt5.QtWidgets import QPushButton

logger = logging.getLogger(__name__)

RGB_TARGET_WAVELENGTHS_VIS = (645.0, 555.0, 465.0)
RGB_TARGET_WAVELENGTHS_SWIR = (1500.0, 1300.0, 1100.0)


def _rgb_band_indices(lambda_raw):
    targets = RGB_TARGET_WAVELENGTHS_SWIR if float(
        lambda_raw.max()) > 1000.0 else RGB_TARGET_WAVELENGTHS_VIS
    return [int(np.argmin(np.abs(lambda_raw - t))) for t in targets]


class DataThread(QThread):
    completed = Signal(object, object)
    rgb_ready = Signal(object)
    config = None
    spectrograms = None
    pixelSize = 0
    sx = 0
    button: QPushButton = None
    imageWidget = None
    inverted = True
    bin = 1

    def __init__(self, parent: QObject = None):
        super().__init__(parent)
        self.finished.connect(self.deleteLater)
        # Slot lives on this QThread object (GUI-thread affinity), so the
        # emit from run() is delivered as a queued call on the GUI thread.
        self.rgb_ready.connect(self._apply_rgb)

    def _apply_rgb(self, rgb):
        widget = self.imageWidget
        if widget is not None:
            widget.setImage(rgb, autoLevels=False, levels=[0.0, 1.0])

    def run(self):
        try:
            start_time = time.time()
            bin_factor = int(getattr(self, 'bin', 1) or 1)
            mm = float(self.config['m']) * bin_factor
            cc = float(self.config['c'])
            step = float(self.config['resolution'])
            self.inverted = self.config.get('raw_inverted', True)
            binner = LambdaBinner(mm, cc, step=step,
                                  threads=8).fit_from(self.spectrograms)
            binner.warmup()  # อุ่น JIT ครั้งเดียว (จะช่วยให้ครั้งจริงเร็วขึ้น)
            hypercube_binned, lambda_out = binner.transform_to_cube(
                self.spectrograms)
            if self.inverted:
                hypercube_binned = hypercube_binned[:, ::-1, :]
            print(f"Processing Time: {time.time() - start_time:.2f} seconds")
            n_bins = hypercube_binned.shape[0]
            if n_bins > 2:
                i_r, i_g, i_b = _rgb_band_indices(lambda_out)
            else:
                i_b = max(0, n_bins // 4)
                i_g = max(0, n_bins // 2)
                i_r = max(0, min(n_bins - 1, n_bins * 3 // 4))

            def _norm(band2d):
                b = band2d.astype(np.float32)
                p2, p98 = np.percentile(b, (2, 98))
                if p98 > p2:
                    return np.clip((b - p2) / (p98 - p2), 0.0, 1.0)
                return np.zeros_like(b)

            # cube[i, :, :] = (slit_height, scan_lines) → .T → (scan_lines, slit_height)
            rgb = np.stack([
                _norm(hypercube_binned[i_r, :, :].T),
                _norm(hypercube_binned[i_g, :, :].T),
                _norm(hypercube_binned[i_b, :, :].T),
            ], axis=-1)  # (scan_lines, slit_height, 3)
            self.rgb_ready.emit(rgb)
            self.completed.emit(hypercube_binned, lambda_out.round(2))

        except Exception:
            logger.exception("DataThread failed")


class PreviewCubeThread(QThread):
    """Builds a partial RGB cube preview from a snapshot of in-progress frames."""

    previewDone = Signal(object)   # emits (rgb, cube, lambda_out) tuple

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frames_snapshot = None  # list of 2D uint16 frames
        self.config = None
        self.bin = 1
        self.inverted = True

    def run(self):
        try:
            if not self.frames_snapshot or len(self.frames_snapshot) < 2:
                return
            spectrograms = np.stack(self.frames_snapshot, axis=0).astype(np.uint16)
            # spectrograms: (scan_lines, slit_height, wavelength_cols)
            # Downsample spatial axes proportionally so slit_height ≤ 400.
            # wavelength axis (axis-2) is never touched.
            _MAX_Y = 400
            sc_n, sh, wc = spectrograms.shape
            logger.debug("PreviewCubeThread: input spectrograms shape=(%d, %d, %d)", sc_n, sh, wc)
            if sh > _MAX_Y:
                _t_resize = time.time()
                step_y = max(1, sh // _MAX_Y)
                step_x = max(1, int(round(sc_n * step_y / sh)))
                spectrograms = spectrograms[::step_x, ::step_y, :]
                logger.debug(
                    "PreviewCubeThread: spatial subsample (%d,%d)->(%d,%d) in %.3fs",
                    sc_n, sh, spectrograms.shape[0], spectrograms.shape[1],
                    time.time() - _t_resize,
                )

            # Generate RGB directly from raw spectrogram columns — no binner needed for preview.
            # spectrograms: (scan_lines, slit_height, wavelength_cols)
            _, _, wc = spectrograms.shape
            i_b = wc // 4
            i_g = wc // 2
            i_r = wc * 3 // 4

            if self.inverted:
                spectrograms = spectrograms[:, ::-1, :]

            def norm_band(band2d):
                b = band2d.astype(np.float32)
                p2, p98 = np.percentile(b, (2, 98))
                if p98 > p2:
                    return np.clip((b - p2) / (p98 - p2), 0.0, 1.0)
                return np.zeros_like(b)

            if self.config is not None:
                bin_factor = int(getattr(self, 'bin', 1) or 1)
                mm = float(self.config['m']) * bin_factor
                cc = float(self.config['c'])
                lambda_raw = (np.arange(wc) * mm + cc).round(2)
                i_r, i_g, i_b = _rgb_band_indices(lambda_raw)
            else:
                lambda_raw = np.arange(wc).round(2)
                i_b = wc // 4
                i_g = wc // 2
                i_r = wc * 3 // 4

            r = norm_band(spectrograms[:, :, i_r])
            g = norm_band(spectrograms[:, :, i_g])
            b_ch = norm_band(spectrograms[:, :, i_b])
            rgb = np.stack([r, g, b_ch], axis=-1)   # (scan_lines, slit_height, 3)

            # Build pseudo-cube and wavelength axis from calibration formula.
            # pseudo_cube: (wc, slit_height, scan_lines) — same layout as full cube
            bin_factor = int(getattr(self, 'bin', 1) or 1)
            mm = float(self.config['m']) * bin_factor if self.config is not None else 1.0
            cc = float(self.config['c']) if self.config is not None else 0.0
            lambda_raw = (np.arange(wc) * mm + cc).round(2)
            pseudo_cube = spectrograms.transpose(2, 1, 0)  # (wc, slit_height, scan_lines)
            self.previewDone.emit((rgb, pseudo_cube, lambda_raw))
        except Exception:
            logger.exception("PreviewCubeThread failed")
