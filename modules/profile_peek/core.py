"""Qt-free content of a profile peek: one curve and a few lines that sum up a calibration.

A *peek* is the small card a status card opens before the full viewer: the curve that matters
most, two or three facts, and a warning when there is one. The builders here take the objects
the full viewers already show (a dark dict, a Smile model, a flat-field profile, a radiometric
reference) by duck typing, so this module imports no other ``modules.*`` package.

Data conventions:
    Dark ``mean`` is ``(slit, wavelength)`` counts. A Smile model's ``map_x`` is ``(rows, cols)``
    source columns of each output pixel. A flat-field profile is ``(slit row, band)`` relative
    response. Reference factors are ``(slit row, band)`` reflectance per count. Wavelengths in nm.
    Curves are float64 copies; nothing keeps a reference to the caller's arrays.

Threading:
    Pure functions; call them from any thread.

Example:
    >>> import numpy as np
    >>> peek = dark_peek({"mean": np.ones((4, 6)), "exposure": 2.0, "gain": 1})
    >>> peek.title, peek.curve.x_label
    ('Dark', 'sensor column')
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

#: Peek colours, matching the status card palette: name -> ``'#RRGGBB'``.
COLORS = {"green": "#00C98D", "red": "#FF637C", "orange": "#FF9868", "cyan": "#00BFC7", "grey": "#8A929E"}


@dataclass(frozen=True)
class PeekCurve:
    """One curve of a peek.

    Attributes:
        x, y: 1-D float64 arrays of equal length (NaN points are drawn as gaps).
        x_label, y_label: Axis captions with units, e.g. ``'wavelength (nm)'``.
        log_y: Draw ``y`` on a log scale (radiometric factors span decades).
        reference: Optional horizontal guide value, e.g. ``1.0`` for a relative response.
    """

    x: np.ndarray
    y: np.ndarray
    x_label: str = ""
    y_label: str = ""
    log_y: bool = False
    reference: Optional[float] = None


@dataclass(frozen=True)
class PeekData:
    """What a peek shows.

    Attributes:
        title: Card title, e.g. ``'Smile + keystone'``.
        color: Key of :data:`COLORS` (the status of the thing shown).
        lines: Short facts, one per line (two or three).
        curve: The curve to plot, or ``None`` for a text-only peek.
        warning: One sentence shown in orange, or ``''``.
        open_label: Text of the button that opens the full viewer; ``''`` for none (nothing to
            open yet).
        actions: Further buttons, ``((key, label), ...)`` in order, e.g. ``(("capture",
            "Capture"),)``; a press closes the peek and the widget emits the key.
    """

    title: str
    color: str = "cyan"
    lines: Tuple[str, ...] = field(default_factory=tuple)
    curve: Optional[PeekCurve] = None
    warning: str = ""
    open_label: str = "Open ▸"
    actions: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)


def _curve(x, y, **kwargs) -> PeekCurve:
    x = np.asarray(x, dtype=np.float64).ravel().copy()
    y = np.asarray(y, dtype=np.float64).ravel().copy()
    if x.shape != y.shape:
        raise ValueError(f"curve x {x.shape} and y {y.shape} differ")
    return PeekCurve(x, y, **kwargs)


def _join(*parts) -> str:
    return " · ".join(str(p) for p in parts if p not in (None, ""))


def dark_peek(dark: dict, wavelength: Optional[Tuple[float, float]] = None, leak: Optional[dict] = None) -> PeekData:
    """Peek of the dark in use: its column profile (median over the slit).

    Args:
        dark: The loaded dark dict (``mean`` ``(slit, wavelength)``; optional ``exposure``,
            ``gain``, ``bin``, ``n_frames``, ``temperature``, ``captured_at``, ``hot_pixel_map``).
        wavelength: ``(m, c)`` with ``nm = c + m * column`` at the dark's bin, or ``None``
            (the x axis is then the sensor column).
        leak: :func:`modules.dark.library.light_leak` of the dark (``None`` = clean or not checked).

    Returns:
        PeekData: green, or orange with a warning when light leaked in.

    Raises:
        ValueError: ``dark`` has no 2-D ``mean``.
    """
    mean = np.asarray(dark.get("mean"), dtype=np.float64)
    if mean.ndim != 2:
        raise ValueError("dark mean must be (slit, wavelength)")
    cols = np.arange(mean.shape[1], dtype=np.float64)
    if wavelength is not None:
        m, c = (float(v) for v in wavelength)
        x, x_label, unit = c + m * cols, "wavelength (nm)", " nm"
    else:
        x, x_label, unit = cols, "sensor column", ""
    hot = dark.get("hot_pixel_map")
    lines = [_join(f"{float(dark.get('exposure', 0)):g} ms", f"gain {float(dark.get('gain', 0)):g}",
                   f"bin {dark['bin']}" if dark.get("bin") is not None else "",
                   f"{float(dark['temperature']):.1f} °C" if dark.get("temperature") is not None else ""),
             _join(f"{dark['n_frames']} frames" if dark.get("n_frames") is not None else "",
                   f"{int(np.count_nonzero(hot))} hot pixels" if hot is not None else "")]
    if dark.get("captured_at"):
        lines.append(f"captured {dark['captured_at']}")
    warning = ""
    if leak:
        where = ", ".join(f"{x[int(col)]:.0f}" for col in list(leak.get("columns") or [])[:4]
                          if 0 <= int(col) < x.size)
        warning = f"Light leak: lines at {where}{unit}" if where else "Light leak in this dark"
    return PeekData("Dark", "orange" if warning else "green", tuple(l for l in lines if l),
                    _curve(x, np.median(mean, axis=0), x_label=x_label, y_label="counts"), warning, "View ▸")


def smile_peek(calibration, path: str = "") -> PeekData:
    """Peek of a Smile (or Smile + keystone) model: the smile along the slit.

    The curve is the horizontal shift ``map_x - column`` at the centre column, row by row: the
    bow of a spectral line that the model straightens.

    Args:
        calibration: The loaded model (``map_x`` ``(rows, cols)``; optional ``has_keystone``,
            ``created``, ``rms_before``, ``rms_after``).
        path: Model file, for the title line.

    Returns:
        PeekData: green.
    """
    import os
    map_x = np.asarray(calibration.map_x, dtype=np.float64)
    rows, cols = map_x.shape
    centre = cols // 2
    keystone = bool(getattr(calibration, "has_keystone", False))
    before, after = (float(getattr(calibration, n, np.nan)) for n in ("rms_before", "rms_after"))
    lines = [_join(os.path.basename(path) if path else "", getattr(calibration, "created", "") or ""),
             f"straightness {before:.3f} → {after:.3f} px" if np.isfinite(before) and np.isfinite(after) else ""]
    return PeekData("Smile + keystone" if keystone else "Smile", "green", tuple(l for l in lines if l),
                    _curve(np.arange(rows), map_x[:, centre] - centre, x_label="slit row", y_label="shift (px)",
                           reference=0.0), "", "View ▸")


def flatfield_peek(profile) -> PeekData:
    """Peek of a flat-field profile: the response along the slit (median over the bands).

    Args:
        profile: A ``FlatFieldProfile`` (``profile()`` ``(slit row, band)``, ``key.label()``,
            ``acquired_at``, ``coverage``; optional ``info['steps']``).

    Returns:
        PeekData: cyan, orange with a warning when the profile has brightness steps.
    """
    values = np.asarray(profile.profile(), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        response = np.nanmedian(values, axis=1)
    steps = list((getattr(profile, "info", None) or {}).get("steps") or [])
    lines = [profile.key.label(),
             _join(f"scanned {profile.acquired_at}",
                   f"{float(profile.coverage):.0%} of the slit" if getattr(profile, "coverage", None) is not None else "")]
    warning = f"Brightness steps at {len(steps)} row{'s' if len(steps) != 1 else ''}" if steps else ""
    return PeekData("Flat-field", "orange" if warning else "cyan", tuple(lines),
                    _curve(np.arange(values.shape[0]), response, x_label="slit row",
                           y_label="relative response", reference=1.0), warning, "View ▸")


def reference_peek(reference) -> PeekData:
    """Peek of a radiometric reference: its factors per band (median over the slit), log scale.

    Args:
        reference: A ``RadiometricReference`` (``factors()`` ``(slit row, band)``,
            ``key.wavelengths``, ``key.label()``, ``acquired_at``, ``age_hours()``, ``info``).

    Returns:
        PeekData: green.
    """
    factors = np.asarray(reference.factors(), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(factors, axis=0)
    flat = (getattr(reference, "info", None) or {}).get("flatfield")
    lines = [reference.key.label(),
             _join(f"scanned {reference.acquired_at}", f"{float(reference.age_hours()):.1f} h old"),
             f"flat-field {flat}" if flat else "no flat-field profile"]
    positive = median[np.isfinite(median)]
    return PeekData("Radiometric", "green", tuple(lines),
                    _curve(reference.key.wavelengths, median, x_label="wavelength (nm)",
                           y_label="factor", log_y=bool(positive.size and np.all(positive > 0))), "", "View ▸")


def text_peek(title: str, value: str = "", details: Sequence[str] = (), color: str = "cyan",
              open_label: str = "Open ▸", actions: Sequence[Tuple[str, str]] = ()) -> PeekData:
    """A peek with no curve: the card's value and its tooltip lines (e.g. the wavelength calibration).

    Args:
        title: Card title.
        value: The card's value, shown as the first line.
        details: Further lines (empty ones are dropped; at most five are kept).
        color: Key of :data:`COLORS`.
        open_label: Button text (``''``: no Open button).
        actions: Further buttons ``((key, label), ...)``, see :attr:`PeekData.actions`.

    Returns:
        PeekData: without a curve.
    """
    lines = [str(value)] if value else []
    lines += [str(d).strip() for d in details if str(d).strip()]
    return PeekData(title, color if color in COLORS else "cyan", tuple(lines[:6]), None, "", open_label,
                    tuple((str(k), str(l)) for k, l in actions))
