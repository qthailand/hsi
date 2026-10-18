"""Excel (``.xlsx``) export of a :class:`~modules.spectral_resolution.core.ResolutionResult`.

Sheets:

* ``Summary``  - line profile and baseline, overall median/min/max FWHM (px, nm),
  resolving power, line counts.
* ``Lines``    - one row per line (``P1``, ``P2``, ... in input order): wavelength, pixel,
  median FWHM px/nm with the 16-84 % spread, resolving power, valid blocks, most frequent
  flag, line profile.
* ``Blocks``   - one row per (row block, line): rows, centre, FWHM px/nm, half-max width,
  amplitude, SNR, components fitted together, flag, line profile.
* ``Settings`` - measurement parameters (profile, fit window, baseline method and every
  effective baseline parameter), camera setting, calibration files used and the selected
  peaks. A dict-valued setting becomes one row per key (``baseline_params.lam``); a list of
  dicts one row per item and key (``selected_peaks.P1.pixel``).

Data conventions:
    Pixels are spectral columns at the frame's hardware bin; nm for wavelengths and widths;
    rejected widths (NaN) are empty cells.

Requires ``pandas`` and ``openpyxl``.

Example:
    >>> from modules.spectral_resolution.excel_report import write_resolution_excel  # doctest: +SKIP
    >>> write_resolution_excel("fwhm.xlsx", result, {"camera": "genicam #0 · bin 2"})  # doctest: +SKIP
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

__all__ = ["resolution_tables", "write_resolution_excel"]


def _cell(value):
    """Excel-friendly value: finite numbers, ``None`` for NaN, ``str`` for everything else."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _setting_rows(settings) -> list:
    """``(name, value)`` rows of a settings dict, nested dicts / lists of dicts flattened."""
    rows = []
    for key, value in settings.items():
        if isinstance(value, dict):
            if not value:
                rows.append((key, "(none)"))
            rows.extend((f"{key}.{k}", _cell(v)) for k, v in value.items())
        elif isinstance(value, (list, tuple)) and value and all(isinstance(v, dict) for v in value):
            for i, item in enumerate(value):
                label = item.get("label", str(i + 1))
                rows.extend((f"{key}.{label}.{k}", _cell(v)) for k, v in item.items() if k != "label")
        else:
            rows.append((key, _cell(value)))
    return rows


def resolution_tables(result, info=None) -> dict:
    """``{sheet name: pandas.DataFrame}`` for ``result`` plus ``info`` (``{name: value}``) in Settings."""
    import pandas as pd

    tables = {}
    s = result.summary()
    profile = result.settings.get("profile", "gaussian")
    head = [("profile", profile), ("baseline", result.settings.get("baseline", "none"))]
    tables["Summary"] = pd.DataFrame(head + [(k, _cell(v)) for k, v in s.items()], columns=["parameter", "value"])
    tables["Lines"] = pd.DataFrame([{
        "peak": f"P{j + 1}", "line": j,
        "wavelength_nm": _cell(l.wavelength_nm), "pixel": _cell(l.pixel),
        "fwhm_px_median": _cell(l.fwhm_px_median), "fwhm_px_p16": _cell(l.fwhm_px_p16),
        "fwhm_px_p84": _cell(l.fwhm_px_p84), "fwhm_nm_median": _cell(l.fwhm_nm_median),
        "fwhm_nm_p16": _cell(l.fwhm_nm_p16), "fwhm_nm_p84": _cell(l.fwhm_nm_p84),
        "resolving_power_median": _cell(l.resolving_power_median), "valid_blocks": l.n_valid,
        "blocks": l.n_blocks, "main_flag": l.main_flag, "profile": profile} for j, l in enumerate(result.lines)])
    rows = []
    for (r0, r1), block in zip(result.block_rows, result.measurements):
        for j, w in enumerate(block):
            rows.append({"row_first": int(r0), "row_last": int(r1), "line": j,
                         "wavelength_nm": _cell(w.wavelength_nm), "center_px": _cell(w.center_px),
                         "fwhm_px": _cell(w.fwhm_px), "fwhm_nm": _cell(w.fwhm_nm),
                         "fwhm_px_halfmax": _cell(w.fwhm_px_halfmax), "amplitude": _cell(w.amplitude),
                         "snr": _cell(w.snr), "components": w.n_components, "flag": w.flag,
                         "resolving_power": _cell(w.resolving_power), "profile": w.profile})
    tables["Blocks"] = pd.DataFrame(rows)
    settings = {"exported_at": _dt.datetime.now().isoformat(timespec="seconds")}
    settings.update(result.settings)
    settings.update(info or {})
    tables["Settings"] = pd.DataFrame(_setting_rows(settings), columns=["name", "value"])
    return tables


def write_resolution_excel(path, result, info=None) -> str:
    """Write :func:`resolution_tables` to ``path`` (``.xlsx`` appended if missing); returns the path.

    Raises:
        ImportError: ``openpyxl`` is not installed.
    """
    import pandas as pd
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError("Excel export needs openpyxl: pip install openpyxl") from exc
    path = str(path)
    if not path.lower().endswith(".xlsx"):
        path += ".xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in resolution_tables(result, info).items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            for column in sheet.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in column[:200])
                sheet.column_dimensions[column[0].column_letter].width = min(max(10, width + 2), 60)
    return path
