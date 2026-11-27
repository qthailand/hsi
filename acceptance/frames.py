"""Raw frames for the acceptance tests: replay files and deterministic synthetic frames (numpy only).

Synthetic frames are *test input*, not calibration data: a bright gradient of the camera's real
bit depth (sensor bits shifted to the delivered bits), with 0.1 % of the pixels at the given
clip level and 0.1 % just below it, the same every run (seeded).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def load_replay(folder) -> List[np.ndarray]:
    """Every raw frame in ``folder``'s ``*.npy`` files (2-D frames or stacks), as uint16."""
    frames = []
    for path in sorted(Path(folder).glob("*.npy")):
        data = np.load(path)
        frames.extend([data] if data.ndim == 2 else list(data))
    if not frames:
        raise RuntimeError(f"no .npy raw frames in {folder}")
    return [np.ascontiguousarray(f, dtype=np.uint16) for f in frames]


def synthetic(shape, delivered_bits: int, sensor_bits: int, level: int, count: int = 8) -> List[np.ndarray]:
    """``count`` deterministic frames of ``shape`` (see the module docstring)."""
    rng = np.random.default_rng(7)
    h, w = (int(v) for v in shape)
    shift = int(delivered_bits) - int(sensor_bits)
    top = (1 << int(sensor_bits)) - 2
    out = []
    for _ in range(int(count)):
        base = np.outer(np.linspace(0.05, 0.6, h), np.linspace(0.3, 1.0, w)) * top
        f = (np.clip(base + rng.normal(0, 8, (h, w)), 0, top).astype(np.int64) << max(0, shift)).astype(np.uint16)
        n = max(1, h * w // 1000)
        f[rng.integers(0, h, n), rng.integers(0, w, n)] = int(level)
        f[rng.integers(0, h, n), rng.integers(0, w, n)] = int(0.99 * int(level))
        out.append(f)
    return out


def save_frames(frames, folder) -> Path:
    """Write ``frames`` as one ``raw_frames.npy`` stack in ``folder``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "raw_frames.npy"
    np.save(path, np.stack(frames))
    return path


__all__ = ["load_replay", "synthetic", "save_frames"]
