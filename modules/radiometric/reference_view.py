"""``ReferenceViewDialog``: look at a stored radiometric reference.

Shown when the scan window's *Radiometric* status card is clicked. Top: the factor of each band
(reflectance per count) at five slit rows, on a logarithmic axis. Bottom: every factor divided by
its band's median over the slit, as an image ``(slit row, band)``: the inverse of how the light
and the optics fall off along the slit. A flat image means every row has the same factor (an ROI
that covered few rows, without a flat-field profile).

Data conventions:
    Factors are ``(lines, bands)`` float32, NaN for bands outside the reflectance table.

Threading:
    GUI thread only; the factors are read from disk when the dialog is created.

Example:
    >>> from modules.radiometric.reference_view import ReferenceViewDialog      # doctest: +SKIP
    >>> ReferenceViewDialog(store.active()[0]).show()                           # doctest: +SKIP
"""
from __future__ import annotations

import warnings

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout

__all__ = ["ReferenceViewDialog"]


class ReferenceViewDialog(QDialog):
    """Plots of one ``RadiometricReference`` (see the module docstring).

    Args:
        reference: A stored ``RadiometricReference``.
        parent: Parent widget.
    """

    def __init__(self, reference, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Radiometric reference {reference.id}")
        self.reference = reference
        factors = np.asarray(reference.factors(), dtype=np.float64)
        lines, bands = factors.shape
        wl = np.asarray(reference.key.wavelengths, dtype=np.float64)
        flat = reference.info.get("flatfield")
        layout = QVBoxLayout(self)
        parts = [reference.key.label(), f"table {reference.reflectance_profile}",
                 f"ROI rows {reference.span[0]}-{reference.span[1]} of {lines}",
                 f"flat-field {flat}" if flat else "no flat-field profile",
                 f"scanned {reference.acquired_at}", f"{reference.age_hours():.1f} h old"]
        if reference.sensor_temperature_c is not None:
            parts.insert(-2, f"sensor {reference.sensor_temperature_c:.1f} °C")
        self.summary_label = QLabel(" · ".join(parts))
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        self.figure = Figure(figsize=(9.5, 6.5), layout="tight")
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)
        top, bottom = self.figure.subplots(2, 1)
        for row in np.unique(np.linspace(0, lines - 1, 5).round().astype(int)):
            top.plot(wl, factors[row], lw=0.9, label=f"row {row}")
        if np.any(factors[np.isfinite(factors)] > 0):
            top.set_yscale("log")
        top.set_xlabel("wavelength (nm)")
        top.set_ylabel("factor (reflectance / count)")
        top.set_title("Factors per band at five slit rows", fontsize=9, loc="left")
        top.legend(fontsize=8, ncol=5)
        top.grid(alpha=0.3, which="both")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # bands that are all NaN
            relative = factors / np.nanmedian(factors, axis=0)
        image = bottom.imshow(relative, aspect="auto", cmap="viridis", interpolation="nearest",
                              extent=(wl[0], wl[-1], lines - 1, 0) if bands > 1 else None)
        bottom.axhline(reference.span[0], color="white", lw=0.6, ls=":")
        bottom.axhline(reference.span[1], color="white", lw=0.6, ls=":")
        bottom.set_xlabel("wavelength (nm)")
        bottom.set_ylabel("slit row")
        bottom.set_title("Factor relative to its band's median over the slit (dotted: the ROI's rows)",
                         fontsize=9, loc="left")
        self.figure.colorbar(image, ax=bottom, label="relative factor")
        self.resize(980, 760)
