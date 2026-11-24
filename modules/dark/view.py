"""``DarkViewDialog``: look at the dark reference in use.

Shown when the scan window's *Dark* status card is clicked. A dark should have no spectral
structure, only an offset and fixed pattern. Light that reached the sensor while it was captured
(room fluorescent tubes through an uncapped lens) shows as narrow lines, and subtracting such a
dark removes those counts from every pixel: the reflectance then has false peaks on targets
brighter than the reference panel and dips on darker ones.

Top: the dark's column profile (median over the slit) against wavelength, with the darks the
current file replaced (``history/``) dashed, and the lines :func:`~modules.dark.library.light_leak`
finds marked. Bottom: each slit row minus its running median along the spectrum, smoothed over a
few rows, as an image ``(slit row, wavelength)``: leaked lines are bright curves that follow the
smile; a clean dark is uniform noise.

Data conventions:
    The dark is ``(slit, wavelength)`` in sensor counts, raw orientation (not flipped).

Threading:
    GUI thread only; files are read when the dialog is created.

Example:
    >>> from modules.dark.view import DarkViewDialog                         # doctest: +SKIP
    >>> DarkViewDialog(library.load(2, 8.0, 1.0, 21.0), wavelength=(0.726, 837.6)).show()   # doctest: +SKIP
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout

from .library import HISTORY_FOLDER, LEAK_WINDOW, light_leak

__all__ = ["DarkViewDialog", "spectral_structure", "replaced_darks"]

#: Darks from ``history/`` drawn with the current one (newest first).
MAX_HISTORY = 3
#: Slit rows averaged per line of the structure image.
STRUCTURE_ROWS = 8


def spectral_structure(mean: np.ndarray, window: int = LEAK_WINDOW, rows: int = STRUCTURE_ROWS) -> np.ndarray:
    """Each slit row minus its running median over ``window`` columns, averaged over blocks of
    ``rows`` rows: ``float32 (ceil(slit / rows), wavelength)``. Offsets and slow fixed pattern go,
    narrow spectral lines stay."""
    from scipy.ndimage import median_filter
    frame = np.asarray(mean, dtype=np.float32)
    h = frame.shape[0] - frame.shape[0] % rows if frame.shape[0] >= rows else frame.shape[0]
    rows = rows if frame.shape[0] >= rows else 1
    blocks = frame[:h].reshape(h // rows, rows, frame.shape[1]).mean(axis=1)
    return (blocks - median_filter(blocks, size=(1, int(window)), mode="nearest")).astype(np.float32)


def replaced_darks(path: Optional[str], limit: int = MAX_HISTORY) -> List[Tuple[str, str]]:
    """``[(label, path), ...]`` of the darks ``path`` replaced (``history/<stem>.<time>.npz``),
    newest first, at most ``limit``."""
    if not path:
        return []
    stem = os.path.splitext(os.path.basename(path))[0]
    pattern = os.path.join(os.path.dirname(path), HISTORY_FOLDER, glob.escape(stem) + ".*.npz")
    found = sorted(glob.glob(pattern), reverse=True)[:int(limit)]
    return [(os.path.basename(p)[len(stem) + 1:-4], p) for p in found]


class DarkViewDialog(QDialog):
    """Plots of one loaded dark (see the module docstring).

    Args:
        dark: Dict of :meth:`modules.dark.library.DarkFrameLibrary.load` (``mean``,
            ``hot_pixel_map``, ``bin``, ``exposure``, ``gain``, ``n_frames``, ``temperature``,
            ``captured_at``, ``filename``, ``path``).
        wavelength: ``(nm per column, nm of column 0)`` at the dark's bin, or ``None`` to plot
            sensor columns.
        parent: Parent widget.
    """

    def __init__(self, dark: dict, wavelength: Optional[Tuple[float, float]] = None, parent=None):
        super().__init__(parent)
        self.dark = dark
        mean = np.asarray(dark["mean"])
        self.setWindowTitle(f"Dark reference {dark.get('filename', '')}")
        cols = np.arange(mean.shape[1], dtype=np.float64)
        if wavelength is not None:
            m, c = (float(v) for v in wavelength)
            x, x_label = c + m * cols, "wavelength (nm)"
        else:
            m = c = None
            x, x_label = cols, "sensor column"
        self.leak = light_leak(mean)

        layout = QVBoxLayout(self)
        hot = dark.get("hot_pixel_map")
        parts = [f"bin {dark.get('bin', '?')}", f"{float(dark.get('exposure', 0)):g} ms",
                 f"gain {float(dark.get('gain', 0)):g}", f"{dark.get('n_frames', '?')} frames",
                 f"{int(np.count_nonzero(hot))} hot pixels" if hot is not None else ""]
        if dark.get("temperature") is not None:
            parts.append(f"{float(dark['temperature']):.1f} °C")
        parts.append(f"captured {dark['captured_at']}" if dark.get("captured_at") else "capture time not recorded")
        self.summary_label = QLabel(f"<b>{dark.get('filename', '')}</b> · " + " · ".join(p for p in parts if p))
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        if self.leak is None:
            verdict = "✓ No spectral lines: no light reached the sensor."
            style = "color: #2e7d32;"
        else:
            where = ", ".join(f"{x[col]:.0f}" for col in self.leak["columns"][:6])
            unit = " nm" if m is not None else " (columns)"
            verdict = (f"⚠ Light reached the sensor while this dark was captured: lines at {where}{unit}, "
                       f"up to {self.leak['peak_counts']:.0f} counts ({self.leak['ratio']:.0f} × noise). "
                       "Subtracting it puts false peaks / dips into reflectance there. Capture it again with "
                       "the lens capped and the room lights off, then make new flat-field and radiometric references.")
            style = "color: #c62828; font-weight: bold;"
        self.verdict_label = QLabel(verdict)
        self.verdict_label.setWordWrap(True)
        self.verdict_label.setStyleSheet(style)
        layout.addWidget(self.verdict_label)

        self.figure = Figure(figsize=(9.5, 6.8), layout="tight")
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)
        top, bottom = self.figure.subplots(2, 1, gridspec_kw=dict(height_ratios=(1, 1.1)))

        top.plot(x, np.median(mean, axis=0), lw=1.0, color="C0", label=f"in use ({dark.get('filename', '')})")
        self.history = replaced_darks(dark.get("path"))
        for i, (label, path) in enumerate(self.history):
            try:
                with np.load(path) as old:
                    old_mean = old["mean"]
                if old_mean.shape == mean.shape:
                    top.plot(x, np.median(old_mean, axis=0), lw=0.8, ls="--", color=f"C{i + 1}",
                             label=f"replaced, captured {label}")
            except Exception:                              # an unreadable history file is skipped
                continue
        if self.leak is not None:
            for col in self.leak["columns"]:
                top.axvline(x[col], color="#c62828", lw=0.7, ls=":")
        top.set_xlabel(x_label)
        top.set_ylabel("dark counts (median over slit)")
        top.set_title("Column profile: a dark should be flat apart from fixed pattern", fontsize=9, loc="left")
        top.legend(fontsize=8)
        top.grid(alpha=0.3)

        structure = spectral_structure(mean)
        spread = float(np.percentile(np.abs(structure), 99.5)) or 1.0
        if self.leak is not None:
            spread = max(spread, 0.5 * self.leak["peak_counts"])
        image = bottom.imshow(structure, aspect="auto", cmap="RdBu_r", vmin=-spread, vmax=spread,
                              interpolation="nearest", extent=(x[0], x[-1], mean.shape[0] - 1, 0))
        bottom.set_xlabel(x_label)
        bottom.set_ylabel("sensor slit row (raw)")
        bottom.set_title(f"Spectral structure: each row minus its running median ({STRUCTURE_ROWS}-row blocks) - "
                         "leaked lines are curves that follow the smile", fontsize=9, loc="left")
        self.figure.colorbar(image, ax=bottom, label="counts")
        self.resize(1000, 800)
