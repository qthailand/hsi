"""Tab-separated text report of a wavelength calibration (fit, points, resolution, spectrum).

The file opens directly in Excel/LibreOffice or ``numpy.loadtxt``/``pandas.read_csv``
(``sep='\\t', comment='#'``). It is made of sections; each starts with a comment line
``# [name]`` followed by a header row and data rows, and sections are separated by a blank
line::

    # HSI wavelength calibration report
    # created<TAB>2026-10-07T10:00:00+07:00
    # [fit]                  key / value rows (m, c, R², RMS, bin, row, smoothing ...)
    # [points]               pixel, wavelength_nm, fit_nm, residual_nm, residual_px
    # [resolution]           one row per measured line (FWHM px/nm, R = λ/Δλ, method)
    # [resolution_summary]   key / value rows (median/min/max)
    # [spectrum]             pixel, wavelength_nm, intensity_raw, intensity_smoothed

Sections without data (no fit yet, FWHM not analysed) are left out.

Data conventions:
    Pixels are spectral-column indices at the image's hardware bin; wavelengths and widths
    are nm; ``m`` is nm/pixel at that bin (``m_base`` = unbinned). The file is UTF-8 with a
    BOM (so Excel shows it correctly); headers are ASCII.

Threading:
    Pure file writing; call from any thread.
"""
from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _cell(value) -> str:
    """Format one value: integers as is, floats with up to 10 significant digits, NaN/None empty."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "" if not math.isfinite(float(value)) else f"{float(value):.10g}"
    return str(value).replace("\t", " ").replace("\n", " ")


def _section(lines, name, header, rows):
    lines.append(f"# [{name}]")
    lines.append("\t".join(header))
    lines.extend("\t".join(_cell(v) for v in row) for row in rows)
    lines.append("")


def build_report(*, profile=None, pixels: Sequence[float] = (), wavelengths: Sequence[Optional[float]] = (),
                 line_widths: Sequence = (), summary=None, spectrum_raw=None, spectrum_smoothed=None,
                 settings: Optional[dict] = None, created: Optional[datetime.datetime] = None) -> str:
    """Return the report text (see the module docstring for the layout).

    Args:
        profile: The ``WavelengthCalibration`` (or ``None`` if not calculated yet).
        pixels: Point-table pixels.
        wavelengths: Matching known wavelengths (nm); ``None`` where the table cell is empty.
        line_widths: ``LineWidth`` results of the FWHM analysis.
        summary: ``ResolutionSummary`` for ``line_widths``.
        spectrum_raw: Unsmoothed spectrum of the selected row (1-D).
        spectrum_smoothed: The spectrum as displayed (smoothed if smoothing is on).
        settings: Extra key/value pairs for the ``[fit]`` section (row, smoothing, peak settings ...).
        created: Timestamp (default: now, local time with UTC offset).
    """
    created = created or datetime.datetime.now().astimezone()
    lines = ["# HSI wavelength calibration report", f"# created\t{created.isoformat(timespec='seconds')}", ""]
    info = []
    if profile is not None:
        info += [("m_nm_per_px", profile.m), ("c_nm", profile.c), ("m_base_nm_per_px", profile.base_m),
                 ("r2", profile.r2), ("rms_nm", profile.rms_nm), ("camera_bin", profile.camera_bin),
                 ("frame_shape", "x".join(str(v) for v in profile.frame_shape) if profile.frame_shape else ""),
                 ("selected_row", profile.selected_row), ("n_points", len(profile.pixels))]
    info += list((settings or {}).items())
    if info:
        _section(lines, "fit", ["key", "value"], info)

    if len(pixels):
        rows = []
        for x, y in zip(pixels, wavelengths):
            fitted = profile.m * x + profile.c if profile is not None else None
            residual = (y - fitted) if (fitted is not None and y is not None) else None
            residual_px = residual / profile.m if residual is not None and profile.m else None
            rows.append((x, y, fitted, residual, residual_px))
        _section(lines, "points", ["pixel", "wavelength_nm", "fit_nm", "residual_nm", "residual_px"], rows)

    if len(line_widths):
        _section(lines, "resolution",
                 ["wavelength_nm", "center_px", "fwhm_px", "fwhm_halfmax_px", "fwhm_nm", "resolving_power",
                  "amplitude", "background", "method"],
                 [(w.wavelength_nm, w.center_px, w.fwhm_px, w.fwhm_px_halfmax, w.fwhm_nm, w.resolving_power,
                   w.amplitude, w.background, w.method) for w in line_widths])
        if summary is not None:
            _section(lines, "resolution_summary", ["key", "value"],
                     [("n_lines", summary.n_lines), ("fwhm_px_median", summary.fwhm_px_median),
                      ("fwhm_px_min", summary.fwhm_px_min), ("fwhm_px_max", summary.fwhm_px_max),
                      ("fwhm_nm_median", summary.fwhm_nm_median), ("fwhm_nm_min", summary.fwhm_nm_min),
                      ("fwhm_nm_max", summary.fwhm_nm_max),
                      ("resolving_power_median", summary.resolving_power_median)])

    if spectrum_raw is not None:
        raw = np.asarray(spectrum_raw, dtype=float)
        smoothed = np.asarray(spectrum_smoothed, dtype=float) if spectrum_smoothed is not None else raw
        x = np.arange(len(raw))
        wl = profile.m * x + profile.c if profile is not None else [None] * len(raw)
        _section(lines, "spectrum", ["pixel", "wavelength_nm", "intensity_raw", "intensity_smoothed"],
                 zip(x, wl, raw, smoothed))
    return "\n".join(lines).rstrip("\n") + "\n"


def write_report(path, **parts) -> Path:
    """Write :func:`build_report` text to ``path`` (UTF-8 with BOM); returns the path."""
    path = Path(path)
    path.write_text(build_report(**parts), encoding="utf-8-sig", newline="\n")
    return path
