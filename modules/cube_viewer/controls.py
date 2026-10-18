# widgets/cube_viewer_controls.py
"""RGB composite controls for CubeViewerWindow.

Data conventions:
    The cube is ``(slit, scan, wavelength)`` (the "H, W, Bands" of the code);
    wavelengths are a 1-D array in nm with one value per band, or ``None``.
    Default bands: 645 / 555 / 465 nm (nearest bands) for VIS/NIR cubes, or
    90 % / 50 % / 10 % of the band index when wavelengths are missing,
    mismatched, or start above 900 nm (SWIR).  The composite is an ``(H, W, 3)``
    uint8 array (percentile stretched per channel, then re-stretched with the
    sidebar range slider).

Threading:
    GUI thread only; the composite is computed synchronously in the slot.

Example:
    >>> import numpy as np
    >>> from modules.cube_viewer.controls import RGBCompositeControls   # doctest: +SKIP
    >>> ctl = RGBCompositeControls()                                     # doctest: +SKIP
    >>> ctl.composite_updated.connect(lambda rgb8: print(rgb8.shape))    # doctest: +SKIP
    >>> ctl.set_cube(np.zeros((8, 8, 30), np.uint16), None)              # doctest: +SKIP
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QPushButton,
)

from modules.ui_kit.range_slider import RangeSlider


def _auto_rgb_bands(
    n_bands: int, wavelengths: np.ndarray | None,
) -> tuple[int, int, int]:
    """Pick sensible default R/G/B band indices based on wavelength range."""
    if wavelengths is None or len(wavelengths) != n_bands:
        b_idx = max(0, int(n_bands * 0.10))
        g_idx = max(0, int(n_bands * 0.50))
        r_idx = max(0, min(n_bands - 1, int(n_bands * 0.90)))
        return (r_idx, g_idx, b_idx)

    wl = wavelengths
    wl_min = float(wl[0])
    wl_max = float(wl[-1])

    # SWIR range (>900nm) -> percentage
    if wl_min > 900:
        b_idx = max(0, int(n_bands * 0.10))
        g_idx = max(0, int(n_bands * 0.50))
        r_idx = max(0, min(n_bands - 1, int(n_bands * 0.90)))
        return (r_idx, g_idx, b_idx)

    # VIS/NIR -> closest band by target wavelength
    r_idx = int(np.argmin(np.abs(wl - 645.0)))
    g_idx = int(np.argmin(np.abs(wl - 555.0)))
    b_idx = int(np.argmin(np.abs(wl - 465.0)))
    return (r_idx, g_idx, b_idx)


class RGBCompositeControls(QWidget):
    """Sidebar controls: mode toggle + band selectors + stretch.

    Offers a "Single Band" / "RGB Composite" toggle, three band spin boxes
    (R, G, B), a wavelength read-out and a low/high stretch range slider (the
    last three are visible only in RGB mode).  Construct with no arguments, then
    call ``set_cube``.  GUI thread only.

    Signals:
        mode_changed(str): ``"single"`` or ``"rgb"``, emitted when a mode button
            is clicked (after the controls are updated).
        composite_updated(numpy.ndarray): ``(H, W, 3)`` uint8 RGB image, emitted
            whenever a band spin box or the stretch slider changes and a cube is
            set (also while in single-band mode, so hosts should ignore it
            unless ``is_rgb_mode()``), and when RGB mode is switched on.
    """

    mode_changed = pyqtSignal(str)
    composite_updated = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cube: np.ndarray | None = None
        self._wavelengths: np.ndarray | None = None
        self._raw_rgb8: np.ndarray | None = None
        self._n_bands: int = 0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # mode toggle
        self._btn_single = QPushButton("Single Band")
        self._btn_single.setCheckable(True)
        self._btn_single.setChecked(True)
        self._btn_rgb = QPushButton("RGB Composite")
        self._btn_rgb.setCheckable(True)

        self._btn_single.clicked.connect(lambda: self._set_mode("single"))
        self._btn_rgb.clicked.connect(lambda: self._set_mode("rgb"))

        mode_row = QHBoxLayout()
        mode_row.setSpacing(0)
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.addWidget(self._btn_single)
        mode_row.addWidget(self._btn_rgb)
        lay.addLayout(mode_row)

        # RGB band selectors
        self._spin_r = QSpinBox()
        self._spin_r.setPrefix("R: ")
        self._spin_r.setRange(0, 0)
        self._spin_r.valueChanged.connect(self._on_band_changed)

        self._spin_g = QSpinBox()
        self._spin_g.setPrefix("G: ")
        self._spin_g.setRange(0, 0)
        self._spin_g.valueChanged.connect(self._on_band_changed)

        self._spin_b = QSpinBox()
        self._spin_b.setPrefix("B: ")
        self._spin_b.setRange(0, 0)
        self._spin_b.valueChanged.connect(self._on_band_changed)

        band_row = QHBoxLayout()
        band_row.setSpacing(4)
        band_row.addWidget(self._spin_r)
        band_row.addWidget(self._spin_g)
        band_row.addWidget(self._spin_b)
        lay.addLayout(band_row)

        self._rgb_bands_label = QLabel("\u03bb: \u2014 / \u2014 / \u2014 nm")
        self._rgb_bands_label.setStyleSheet("font-size: 10px; color: #9e9e9e;")
        lay.addWidget(self._rgb_bands_label)

        # stretch slider
        self._stretch_label = QLabel("2.0% \u2013 98.0%")
        self._stretch_label.setStyleSheet("font-size: 10px; color: #9e9e9e;")
        self._stretch_label.setFixedWidth(90)

        self._stretch_slider = RangeSlider(low=0.02, high=0.98)
        self._stretch_slider.range_changed.connect(self._on_stretch_changed)

        stretch_row = QHBoxLayout()
        stretch_row.setContentsMargins(0, 0, 0, 0)
        stretch_row.setSpacing(6)
        stretch_row.addWidget(self._stretch_slider, stretch=1)
        stretch_row.addWidget(self._stretch_label)
        lay.addLayout(stretch_row)

        # hide RGB controls by default
        for w in (
            self._spin_r, self._spin_g, self._spin_b,
            self._rgb_bands_label,
            self._stretch_slider, self._stretch_label,
        ):
            w.setVisible(False)

    # ── public API ────────────────────────────────────────

    def set_cube(self, cube: np.ndarray, wavelengths: np.ndarray | None):
        """Set cube data and pick default R/G/B bands.

        Args:
            cube: ``(slit, scan, wavelength)`` array.  Held by reference.
            wavelengths: 1-D band wavelengths in nm (one per band) or ``None``.
        """
        self._cube = cube
        self._wavelengths = wavelengths
        self._n_bands = cube.shape[2]

        for spin in (self._spin_r, self._spin_g, self._spin_b):
            spin.blockSignals(True)
            spin.setRange(0, self._n_bands - 1)
            spin.blockSignals(False)

        ri, gi, bi = _auto_rgb_bands(self._n_bands, wavelengths)
        self._spin_r.setValue(ri)
        self._spin_g.setValue(gi)
        self._spin_b.setValue(bi)
        self._update_wavelength_label()

    def is_rgb_mode(self) -> bool:
        """Return ``True`` when the "RGB Composite" button is active."""
        return self._btn_rgb.isChecked()

    # ── mode toggle ───────────────────────────────────────

    def _set_mode(self, mode: str):
        is_rgb = (mode == "rgb")
        self._btn_single.setChecked(not is_rgb)
        self._btn_rgb.setChecked(is_rgb)

        for w in (
            self._spin_r, self._spin_g, self._spin_b,
            self._rgb_bands_label,
            self._stretch_slider, self._stretch_label,
        ):
            w.setVisible(is_rgb)

        if is_rgb:
            self._update_composite()

        self.mode_changed.emit(mode)

    # ── band change ───────────────────────────────────────

    def _on_band_changed(self):
        self._update_wavelength_label()
        self._update_composite()

    def _update_wavelength_label(self):
        if self._wavelengths is None:
            self._rgb_bands_label.setText(
                f"Band: {self._spin_r.value()} / "
                f"{self._spin_g.value()} / "
                f"{self._spin_b.value()}"
            )
        else:
            wl = self._wavelengths
            r = wl[self._spin_r.value()]
            g = wl[self._spin_g.value()]
            b = wl[self._spin_b.value()]
            self._rgb_bands_label.setText(
                f"\u03bb: {r:.1f} / {g:.1f} / {b:.1f} nm"
            )

    # ── composite ─────────────────────────────────────────

    def _update_composite(self):
        if self._cube is None:
            return

        ri = self._spin_r.value()
        gi = self._spin_g.value()
        bi = self._spin_b.value()

        # cube shape = (H, W, Bands) -> index axis 2
        r = self._cube[:, :, ri].astype(np.float32)
        g = self._cube[:, :, gi].astype(np.float32)
        b = self._cube[:, :, bi].astype(np.float32)

        rgb = np.stack([r, g, b], axis=-1)

        rgb8 = np.empty_like(rgb, dtype=np.uint8)
        for ch in range(3):
            band = rgb[:, :, ch]
            lo = np.percentile(band, 2.0)
            hi = np.percentile(band, 98.0)
            if hi <= lo:
                hi = lo + 1.0
            out = (band - lo) * (255.0 / (hi - lo))
            rgb8[:, :, ch] = np.clip(out, 0, 255).astype(np.uint8)

        self._raw_rgb8 = rgb8.copy()

        self._on_stretch_changed(
            self._stretch_slider.low, self._stretch_slider.high,
        )

    # ── stretch ───────────────────────────────────────────

    def _on_stretch_changed(self, low: float, high: float):
        self._stretch_label.setText(
            f"{low*100:.1f}% \u2013 {high*100:.1f}%"
        )

        if self._raw_rgb8 is None:
            return

        rgb8 = self._raw_rgb8
        stretched = np.empty_like(rgb8)
        for ch in range(3):
            band = rgb8[:, :, ch].astype(np.float32)
            lo = np.percentile(band, low * 100)
            hi = np.percentile(band, high * 100)
            if hi <= lo:
                hi = lo + 1.0
            out = (band - lo) * (255.0 / (hi - lo))
            stretched[:, :, ch] = np.clip(out, 0, 255).astype(np.uint8)

        self.composite_updated.emit(stretched)