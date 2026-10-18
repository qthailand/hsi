"""Backward-compatible facade for the modular :mod:`smile` package.

Implementation lives in focused modules: ``core``, ``surface``,
``analysis``, ``calibration``, ``corrector`` and the optional ``widget``.
This file preserves legacy imports without maintaining a second copy of the
processing and GUI implementation.

Run the standalone application with ``python -m smile`` or
``python -m smile.smile``. Run benchmarks with ``python -m smile.test_cmd``.
"""
from __future__ import annotations

from .analysis import SmileAnalysis, analyze_smile, verify_on_frame
from .calibration import (
    SmileCalibration,
    build_maps,
    calibration_from_analysis,
    correct_smile,
    crop_margin,
    load_calibration,
    load_calibration_fast,
    margin_from_map,
    save_calibration,
    save_calibration_fast,
)
from .core import (
    Line,
    auto_sat_level,
    detect_lines,
    estimate_noise,
    load_frame,
    robust_sigma,
    save_frame,
    straightness_rms,
    trace_line,
)
from .corrector import SmileCorrector
from .surface import Surface2D, cross_validate_lines, fit_surface_2d

__version__ = "2.4.0"

__all__ = [
    "load_frame", "save_frame", "robust_sigma", "auto_sat_level",
    "estimate_noise", "Line", "detect_lines", "trace_line",
    "straightness_rms", "Surface2D", "fit_surface_2d",
    "cross_validate_lines", "build_maps", "margin_from_map",
    "correct_smile", "crop_margin", "SmileAnalysis", "analyze_smile",
    "verify_on_frame", "SmileCalibration", "calibration_from_analysis",
    "save_calibration", "load_calibration", "save_calibration_fast",
    "load_calibration_fast", "SmileCorrector", "PlotCanvas",
    "SmileCorrectionWidget", "SmileMainWindow", "launch_gui", "HAVE_QT",
]

_GUI_EXPORTS = {
    "HAVE_QT", "PlotCanvas", "SmileCorrectionWidget",
    "SmileMainWindow", "launch_gui",
}


def __getattr__(name: str):
    """Resolve the optional Qt exports only when explicitly requested."""
    if name in _GUI_EXPORTS:
        from . import widget

        value = getattr(widget, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    """Run the standalone GUI entry point."""
    from .widget import launch_gui

    try:
        raise SystemExit(launch_gui())
    except ImportError as exc:
        raise SystemExit(
            f"{exc}\nInstall GUI dependencies with: "
            "pip install PyQt5 matplotlib"
        ) from exc


if __name__ == "__main__":
    main()
