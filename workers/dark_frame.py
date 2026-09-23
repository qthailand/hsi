"""
DarkFrameLibrary — captures, computes, stores, and loads dark frame references.

Storage layout (one .npz per (bin, exposure, gain) combination):
    dark_frames/
        dark_b1_e100.0_g5.0.npz
        dark_b2_e50.0_g1.0.npz
        ...

Each .npz contains:
    mean          — pre-computed mean dark frame (original dtype, ready to subtract)
    hot_pixel_map — uint8 mask, 255 = hot pixel
    bin           — scalar int
    exposure      — scalar float (ms)
    gain          — scalar float
    n_frames      — scalar int (number of frames averaged)
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DarkFrameLibrary:
    """
    Manages dark frame capture, storage, and retrieval.

    Usage — capture:
        lib = DarkFrameLibrary(folder='dark_frames', n_frames=50)
        lib.start_capture(bin=1, exposure=100.0, gain=5.0)
        for frame in camera_frames:
            done = lib.add_frame(frame)
            if done:
                break
        result = lib.load(bin=1, exposure=100.0, gain=5.0)

    Usage — load existing:
        result = lib.load(bin=1, exposure=100.0, gain=5.0)
        if result:
            dark_mean = result['mean']
            hot_map   = result['hot_pixel_map']
    """

    def __init__(self, folder: str, n_frames: int = 50):
        self.folder = folder
        self.n_frames = n_frames
        os.makedirs(folder, exist_ok=True)

        # Capture state (reset by start_capture)
        self._collecting: bool = False
        self._stack: List[np.ndarray] = []
        self._bin: int = 1
        self._exposure: float = 0.0
        self._gain: float = 0.0

    # ------------------------------------------------------------------
    #  Capture
    # ------------------------------------------------------------------

    def start_capture(self, bin: int, exposure: float, gain: float) -> None:
        """Begin a new accumulation session for the given camera parameters."""
        self._collecting = True
        self._stack = []
        self._bin = int(bin)
        self._exposure = float(exposure)
        self._gain = float(gain)
        logger.info(
            "DarkFrameLibrary: starting capture — bin=%d exposure=%.1f gain=%.1f target=%d",
            self._bin, self._exposure, self._gain, self.n_frames,
        )

    def add_frame(self, frame: np.ndarray) -> bool:
        """
        Add one frame to the accumulation buffer.

        Returns True when n_frames have been collected, the mean has been
        computed, and the result has been saved to disk.
        Raw frames are discarded immediately after the mean is computed.
        """
        if not self._collecting:
            return False

        if self._stack and frame.shape != self._stack[0].shape:
            # ★ stale frame จาก queue — reset ด้วย frame ใหม่
            logger.warning(
                "DarkFrameLibrary: shape changed %s → %s, resetting",
                self._stack[0].shape, frame.shape,
            )
            self._stack.clear()

        self._stack.append(frame.copy())

        if len(self._stack) >= self.n_frames:
            self._compute_and_save()
            self._collecting = False
            return True

        return False

    def cancel(self) -> None:
        """Abort current capture without saving anything."""
        self._collecting = False
        self._stack.clear()
        logger.info("DarkFrameLibrary: capture cancelled")

    @property
    def progress(self) -> tuple:
        """(collected_frames, target_frames)"""
        return len(self._stack), self.n_frames

    @property
    def is_collecting(self) -> bool:
        return self._collecting

    # ------------------------------------------------------------------
    #  Load / Query
    # ------------------------------------------------------------------

    def load(self, bin: int, exposure: float, gain: float) -> Optional[Dict]:
        """
        Load the stored dark frame for the given parameters.

        Returns a dict:
            {
                'mean':          np.ndarray,   # dark mean frame, original dtype
                'hot_pixel_map': np.ndarray,   # uint8 mask (255 = hot)
                'bin':           int,
                'exposure':      float,
                'gain':          float,
                'n_frames':      int,
            }
        Returns None if no matching dark frame exists.
        """
        path = self._filepath(bin, exposure, gain)
        if not os.path.isfile(path):
            return None
        try:
            data = np.load(path)
            return {
                'mean':          data['mean'],
                'hot_pixel_map': data['hot_pixel_map'],
                'bin':           int(data['bin']),
                'exposure':      float(data['exposure']),
                'gain':          float(data['gain']),
                'n_frames':      int(data['n_frames']),
            }
        except Exception:
            logger.exception("DarkFrameLibrary: failed to load %s", path)
            return None

    def exists(self, bin: int, exposure: float, gain: float) -> bool:
        """Return True if a dark frame is stored for this combination."""
        return os.path.isfile(self._filepath(bin, exposure, gain))

    def delete(self, bin: int, exposure: float, gain: float) -> None:
        """Delete a stored dark frame combination."""
        path = self._filepath(bin, exposure, gain)
        if os.path.isfile(path):
            os.remove(path)
            logger.info("DarkFrameLibrary: deleted %s", path)

    def list_keys(self) -> List[Dict]:
        """
        Return metadata for all stored dark frame combinations.

        Each entry: {'bin', 'exposure', 'gain', 'n_frames', 'filename'}
        """
        keys = []
        for name in sorted(os.listdir(self.folder)):
            if not name.endswith('.npz'):
                continue
            path = os.path.join(self.folder, name)
            try:
                data = np.load(path)
                keys.append({
                    'bin':      int(data['bin']),
                    'exposure': float(data['exposure']),
                    'gain':     float(data['gain']),
                    'n_frames': int(data['n_frames']),
                    'filename': name,
                })
            except Exception:
                logger.warning("DarkFrameLibrary: cannot read metadata from %s", name)
        return keys

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _key(self, bin: int, exposure: float, gain: float) -> str:
        return f"dark_b{int(bin)}_e{float(exposure):.1f}_g{float(gain):.1f}"

    def _filepath(self, bin: int, exposure: float, gain: float) -> str:
        return os.path.join(self.folder, self._key(bin, exposure, gain) + '.npz')

    def _compute_and_save(self) -> None:
        if not self._stack:
            return

        stack = np.stack(self._stack, axis=0).astype(np.float32)
        original_dtype = self._stack[0].dtype
        mean = np.round(np.mean(stack, axis=0)).astype(original_dtype)

        hot_pixel_map = self._detect_hot_pixels(mean)

        path = self._filepath(self._bin, self._exposure, self._gain)
        np.savez_compressed(
            path,
            mean=mean,
            hot_pixel_map=hot_pixel_map,
            bin=np.array(self._bin,     dtype=np.int32),
            exposure=np.array(self._exposure, dtype=np.float32),
            gain=np.array(self._gain,    dtype=np.float32),
            n_frames=np.array(len(self._stack), dtype=np.int32),
        )
        logger.info(
            "DarkFrameLibrary: saved %s (%d frames, %d hot pixels)",
            path, len(self._stack), int(hot_pixel_map.sum() > 0 and np.count_nonzero(hot_pixel_map)),
        )
        self._stack.clear()

    def _detect_hot_pixels(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect hot pixels using a normalised threshold of mean + 4σ
        (or 99.9th percentile, whichever is higher).
        Returns a uint8 mask where 255 = hot pixel.
        """
        if np.issubdtype(frame.dtype, np.integer):
            max_val = float(np.iinfo(frame.dtype).max)
        else:
            max_val = float(np.max(frame)) or 1.0

        img_norm = frame.astype(np.float64) / max_val
        mean_val = np.mean(img_norm)
        std_val  = np.std(img_norm)
        perc_val = np.percentile(img_norm, 99.9)
        threshold = max(mean_val + 4.0 * std_val, perc_val)

        mask = img_norm > threshold
        if not mask.any() and std_val > 0:
            fallback = max(mean_val + 3.0 * std_val, np.percentile(img_norm, 99.5))
            mask = img_norm > fallback

        hot_map = np.zeros_like(frame, dtype=np.uint8)
        hot_map[mask] = 255
        return hot_map
