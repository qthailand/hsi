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
import qtawesome as qta
from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QPushButton, QButtonGroup,
)

from modules.ui_kit.range_slider import RangeSlider
from modules.ui_kit.theme import ACCENT, ACCENT_DARK, BORDER, CARD, FIELD_HOVER, PLOT_FOREGROUND

#: Joined (segmented) look of the Band / RGB toggle: one strip, outer corners rounded only.
_MODE_TOGGLE_STYLE = f"""
QPushButton {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 0px;
    padding: 2px 10px; margin: 0px;
}}
QPushButton:hover {{ background: {FIELD_HOVER}; }}
QPushButton:checked {{ background: {ACCENT_DARK}; border-color: {ACCENT}; color: #ffffff; }}
QPushButton[segment="first"] {{ border-top-left-radius: 4px; border-bottom-left-radius: 4px; }}
QPushButton[segment="last"] {{ border-top-right-radius: 4px; border-bottom-right-radius: 4px;
                                border-left: none; }}
"""


def _auto_rgb_bands(
    n_bands: int, wavelengths: np.ndarray | None,
) -> tuple[int, int, int]:
    """Pick default R/G/B band indices.

    True colour (bands nearest 645 / 555 / 465 nm) only when the cube covers 465-645 nm;
    otherwise (SWIR, NIR-only, e.g. 854-1700 nm) a false-colour spread over the cube's own
    wavelength span: R at 90 %, G at 50 %, B at 10 % (by wavelength, so descending band
    order works too).  Without wavelengths the same spread is taken by band index.
    """
    if wavelengths is None or len(wavelengths) != n_bands:
        b_idx = max(0, int(n_bands * 0.10))
        g_idx = max(0, int(n_bands * 0.50))
        r_idx = max(0, min(n_bands - 1, int(n_bands * 0.90)))
        return (r_idx, g_idx, b_idx)

    wl = np.asarray(wavelengths, dtype=np.float64)
    wl_min = float(np.nanmin(wl))
    wl_max = float(np.nanmax(wl))

    if wl_min <= 465.0 and wl_max >= 645.0:
        targets = (645.0, 555.0, 465.0)                # visible range covered: true colour
    else:
        span = wl_max - wl_min
        targets = (wl_min + 0.9 * span, wl_min + 0.5 * span, wl_min + 0.1 * span)
    r_idx, g_idx, b_idx = (int(np.nanargmin(np.abs(wl - t))) for t in targets)
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
    #: (r, g, b) band indices after any R / G / B spin box change (both views).
    bands_changed = pyqtSignal(int, int, int)
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

        # mode toggle: one segmented Band | RGB strip (exclusive button group)
        mode_strip = QWidget()
        mode_strip.setStyleSheet(_MODE_TOGGLE_STYLE)
        mode_row = QHBoxLayout(mode_strip)
        mode_row.setSpacing(0)
        mode_row.setContentsMargins(0, 0, 0, 0)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._btn_single = QPushButton("Band")
        self._btn_rgb = QPushButton("RGB")
        for button, icon, tip, segment in (
                (self._btn_single, "mdi6.image-filter-black-white", "Single Band: one wavelength band", "first"),
                (self._btn_rgb, "mdi6.palette", "RGB Composite: three bands as red, green and blue", "last")):
            button.setIcon(qta.icon(icon, color=PLOT_FOREGROUND, color_on="#ffffff"))
            button.setIconSize(QSize(16, 16))
            button.setCheckable(True)
            button.setToolTip(tip)
            button.setProperty("segment", segment)
            self._mode_group.addButton(button)
            mode_row.addWidget(button)
        mode_row.addStretch()
        self._btn_single.setChecked(True)

        self._btn_single.clicked.connect(lambda: self._set_mode("single"))
        self._btn_rgb.clicked.connect(lambda: self._set_mode("rgb"))
        lay.addWidget(mode_strip)

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

        for spin in (self._spin_r, self._spin_g, self._spin_b):
            spin.setMinimumWidth(90)        # below the ~115 px hint: the row fits the sidebar
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
        for spin, value in ((self._spin_r, ri), (self._spin_g, gi), (self._spin_b, bi)):
            spin.blockSignals(True)         # one composite later, not one per spin box
            spin.setValue(value)
            spin.blockSignals(False)
        self._update_wavelength_label()
        self._update_composite()            # only builds it when the RGB view is showing

    def set_mode(self, mode: str) -> None:
        """Show the ``'single'`` (Band) or ``'rgb'`` view, as the toggle buttons do.

        Emits ``mode_changed`` (also when the mode does not change).

        Raises:
            ValueError: ``mode`` is not ``'single'`` or ``'rgb'``.
        """
        if mode not in ("single", "rgb"):
            raise ValueError(f"mode must be 'single' or 'rgb', got {mode!r}")
        self._set_mode(mode)

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

    def rgb_bands(self) -> tuple:
        """``(r, g, b)`` band indices of the composite."""
        return (self._spin_r.value(), self._spin_g.value(), self._spin_b.value())

    def _on_band_changed(self):
        self._update_wavelength_label()
        self._update_composite()
        self.bands_changed.emit(*self.rgb_bands())

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
        if self._cube is None or not self.is_rgb_mode():
            return                          # Band view: built when switching to RGB

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