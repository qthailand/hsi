"""Excel (``.xlsx``) export of a smile correction result and its spectral-quality analysis.

One workbook holds everything needed to review or re-plot a smile calibration outside
the application. Sheets (the quality sheets appear only when a quality report is given):

* ``Summary``      - fit results (lines used, RMS before/after, CV RMS, margin, ...).
* ``Lines``        - every detected emission line: column, FWHM, height, used, per-line RMS.
* ``Traces``       - per slit row, the traced column of each line before and after correction.
* ``Measurements`` - the points the surface was fitted to, with the fitted value and residual.
* ``Surface``      - polynomial coefficients ``a_ij`` (power of y, power of x) and their stderr.
* ``Quality summary`` - paired before/after statistics (count, bias, RMSE, p95, max).
* ``Quality rows``    - per row and reference line: centre, wavelength error, FWHM, height,
  area, fit RMS and correlation shift, before and after.
* ``Uncertainty``     - bootstrap shift standard deviation and 95 % interval per row/line.
* ``Metadata``        - wavelength coefficients, reference row and any caller metadata.

Data conventions:
    Rows/columns are zero-based detector pixels; shifts and positions are in pixels,
    wavelengths in nm. Missing or rejected values (NaN) are written as empty cells.

Requires ``pandas`` and ``openpyxl``.

Example:
    >>> from modules.smile.excel_report import write_smile_excel   # doctest: +SKIP
    >>> write_smile_excel("smile_report.xlsx", analysis, quality=report)  # doctest: +SKIP
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

__all__ = ["build_smile_tables", "write_smile_excel"]

QUALITY_KEYS = ("center_px", "wavelength_error_nm", "fwhm_nm", "peak_height",
                "area_counts_px", "fit_rms_counts", "correlation_shift_px", "correlation_score")


def _scalar(value):
    """Excel-friendly cell value: numbers stay numbers, NaN/inf become ``None``, the rest ``str``."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return str(value)


def _trace_rms(trace) -> float:
    """RMS deviation of one trace from its median (the module's straightness measure)."""
    t = np.asarray(trace, dtype=float)
    t = t[np.isfinite(t)]
    return float(np.sqrt(np.mean((t - np.median(t)) ** 2))) if t.size > 10 else float("nan")


def build_smile_tables(analysis, quality=None, uncertainty=None, metadata=None) -> dict:
    """Build the workbook content as ``{sheet name: pandas.DataFrame}``.

    Args:
        analysis: A :class:`~modules.smile.analysis.SmileAnalysis`.
        quality: Optional :class:`~modules.smile.quality.QualityReport`.
        uncertainty: Optional dict from :func:`~modules.smile.quality.estimate_shift_uncertainty`
            evaluated at ``quality.rows`` x reference columns.
        metadata: Optional ``{name: value}`` added to the ``Metadata`` sheet (e.g. source file).

    Returns:
        Ordered dict of DataFrames, one per sheet.
    """
    import pandas as pd

    r = analysis
    used_columns = {float(ln.column) for ln in r.used_lines}
    tables = {}

    fit = r.fit_info or {}
    summary = [
        ("Lines used", len(r.used_lines), "", "lines in the surface fit"),
        ("Lines detected", len(r.all_lines), "", ""),
        ("RMS before", r.rms_before, "px", "straightness, deviation from each line's median"),
        ("RMS after", r.rms_after, "px", "in-sample"),
        ("CV RMS", r.cv_rms, "px", "leave-one-line-out cross-validation"),
        ("Surface fit RMS", fit.get("rms"), "px", ""),
        ("Fit points", fit.get("n"), "", ""),
        ("Fit points used", fit.get("n_used"), "", ""),
        ("Degrees of freedom", fit.get("dof"), "", ""),
        ("deg_y", r.surface.deg_y, "", "polynomial degree along the slit"),
        ("deg_x", r.surface.deg_x, "", "polynomial degree along the spectrum"),
        ("Margin left", r.margin[0], "px", "invalid border columns after correction"),
        ("Margin right", r.margin[1], "px", ""),
        ("Reference row (y_ref)", r.y_ref, "px", "row whose positions are kept"),
        ("Interpolation", r.interp, "", ""),
        ("Saturation level", r.sat_level, "counts", ""),
        ("Noise", r.noise, "counts", "estimated per-pixel noise"),
        ("Frame height", r.frame_orig.shape[0], "px", ""),
        ("Frame width", r.frame_orig.shape[1], "px", ""),
        ("Surface y range", "%g .. %g" % r.surface.y_rng, "px", ""),
        ("Surface x range", "%g .. %g" % r.surface.x_rng, "px", ""),
    ]
    tables["Summary"] = pd.DataFrame(
        [(k, _scalar(v), u, n) for k, v, u, n in summary],
        columns=["parameter", "value", "unit", "note"])

    line_rows = []
    for i, ln in enumerate(r.all_lines):
        col = ln.column
        line_rows.append({
            "line": i,
            "column_px": float(col),
            "reference_px": _scalar(r.refs.get(col)),
            "fwhm_px": _scalar(ln.fwhm),
            "height_counts": _scalar(ln.height),
            "saturated": bool(ln.saturated),
            "used_in_fit": float(col) in used_columns,
            "rms_before_px": _scalar(_trace_rms(r.traces_orig[col])) if col in r.traces_orig else None,
            "rms_after_px": _scalar(_trace_rms(r.traces_corr[col])) if col in r.traces_corr else None,
        })
    tables["Lines"] = pd.DataFrame(line_rows, columns=[
        "line", "column_px", "reference_px", "fwhm_px", "height_counts", "saturated",
        "used_in_fit", "rms_before_px", "rms_after_px"])

    traces = {"row": np.arange(r.frame_orig.shape[0])}
    for ln in r.all_lines:
        col = ln.column
        if col in r.traces_orig:
            traces["before_%g" % col] = np.asarray(r.traces_orig[col], dtype=float)
        if col in r.traces_corr:
            traces["after_%g" % col] = np.asarray(r.traces_corr[col], dtype=float)
    tables["Traces"] = pd.DataFrame(traces)

    meas_id = np.asarray(r.meas_id, dtype=int)
    used = list(r.used_lines)
    tables["Measurements"] = pd.DataFrame({
        "row_y": np.asarray(r.meas_y, dtype=float),
        "reference_x_px": np.asarray(r.meas_x, dtype=float),
        "line": meas_id,
        "line_column_px": [float(used[k].column) if 0 <= k < len(used) else np.nan for k in meas_id],
        "measured_shift_px": np.asarray(r.meas_v, dtype=float),
        "fitted_shift_px": np.asarray(r.meas_v, dtype=float) - np.asarray(r.residuals, dtype=float),
        "residual_px": np.asarray(r.residuals, dtype=float),
    })

    coef = np.asarray(r.surface.coef, dtype=float)
    stderr = np.asarray(fit.get("stderr", np.full(coef.size, np.nan)), dtype=float)
    powers = [(i, j) for i in range(r.surface.deg_y + 1) for j in range(r.surface.deg_x + 1)]
    tables["Surface"] = pd.DataFrame({
        "power_y": [p[0] for p in powers],
        "power_x": [p[1] for p in powers],
        "coefficient": coef[:len(powers)],
        "stderr": stderr[:len(powers)] if stderr.size >= len(powers) else np.nan,
    })

    meta = {"exported_at": _dt.datetime.now().isoformat(timespec="seconds")}
    if quality is not None:
        q = quality
        stat_rows = []
        for name, stats in q.summary.items():
            if isinstance(stats, dict):
                stat_rows.append({"metric": name, **{k: _scalar(v) for k, v in stats.items()}})
        tables["Quality summary"] = pd.DataFrame(
            stat_rows, columns=["metric", "count", "bias", "rmse", "p95_abs", "max_abs"])
        valid = pd.DataFrame({
            "wavelength_nm": np.asarray(q.wavelengths_nm, dtype=float),
            "excluded_blend": list(q.metadata.get("excluded_blends", [False] * len(q.wavelengths_nm))),
            "valid_fraction_before": q.summary.get("valid_fraction_before_by_line"),
            "valid_fraction_after": q.summary.get("valid_fraction_after_by_line"),
        })
        tables["Quality by line"] = valid

        rows = np.asarray(q.rows)
        waves = np.asarray(q.wavelengths_nm, dtype=float)
        long = {"row_y": np.repeat(rows, waves.size), "wavelength_nm": np.tile(waves, rows.size)}
        for key in QUALITY_KEYS:
            for stage, data in (("before", q.before), ("after", q.after)):
                if key in data:
                    long["%s_%s" % (key, stage)] = np.asarray(data[key], dtype=float).reshape(-1)
        tables["Quality rows"] = pd.DataFrame(long)

        if uncertainty is not None:
            unc = {"row_y": np.repeat(rows, waves.size), "wavelength_nm": np.tile(waves, rows.size)}
            for key in ("std_px", "lower95_px", "upper95_px", "supported"):
                if key in uncertainty:
                    unc[key] = np.broadcast_to(np.asarray(uncertainty[key]),
                                               (rows.size, waves.size)).reshape(-1)
            tables["Uncertainty"] = pd.DataFrame(unc)
            meta["bootstrap_repeats"] = uncertainty.get("successful_repeats")
            meta["bootstrap_scope"] = uncertainty.get("scope")
        meta.update(q.metadata)
    meta.update(metadata or {})
    tables["Metadata"] = pd.DataFrame(
        [(k, _scalar(v)) for k, v in meta.items()], columns=["name", "value"])
    return tables


def write_smile_excel(path, analysis, quality=None, uncertainty=None, metadata=None) -> str:
    """Write :func:`build_smile_tables` to ``path`` (``.xlsx`` appended if missing).

    Args:
        path: Output file.
        analysis, quality, uncertainty, metadata: See :func:`build_smile_tables`.

    Returns:
        The path written.

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
    tables = build_smile_tables(analysis, quality, uncertainty, metadata)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            sheet = writer.sheets[name[:31]]
            sheet.freeze_panes = "A2"
            for column in sheet.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in column[:200])
                sheet.column_dimensions[column[0].column_letter].width = min(max(10, width + 2), 60)
    return path
