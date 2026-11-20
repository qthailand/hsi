"""Light / dark colours for the smile and keystone widgets, following the application theme.

The HSI app runs a dark theme (qdarktheme + ``modules.ui_kit.theme`` dashboard colours); the
widgets also run stand-alone with the default light Qt style. :func:`is_dark` tells which one is
active, :func:`colors` gives the matching palette for labels / chips / cards, and
:func:`theme_figure` recolours a matplotlib figure (background, axes, text, legends, colour bars,
and near-black "ink" lines / labels) so every plot matches the window it is drawn in.

Data conventions: colours are ``'#RRGGBB'`` strings or ``rgba(...)`` strings for Qt style sheets.

Threading: GUI thread only (reads the QApplication style sheet / palette).
"""
from __future__ import annotations

from typing import Optional

__all__ = ["is_dark", "colors", "theme_figure"]

try:                                       # the app's dashboard colours (dark theme)
    from modules.ui_kit import theme as _ui
    _DARK_PAGE, _DARK_CARD, _DARK_BORDER = _ui.PAGE, _ui.CARD, _ui.BORDER
    _DARK_FG, _ACCENT = _ui.PLOT_FOREGROUND, _ui.ACCENT
except Exception:                          # noqa: BLE001 - stand-alone use without ui_kit
    _DARK_PAGE, _DARK_CARD, _DARK_BORDER, _DARK_FG, _ACCENT = "#1e2024", "#2b2e33", "#3d4148", "#c9ccd1", "#22c06b"

_LIGHT = dict(
    dark=False, page="#ffffff", card="#fbfcfd", border="#d5dbe1", fg="#1b2631", text="#2c3e50", muted="#5d6d7e",
    hint="#7b8a97", good="#1e8449", warn="#b9770e", bad="#c0392b", busy="#2471a3", todo="#95a5a6",
    chip_good=("#e9f7ef", "#1e8449", "#a9dfbf"), chip_warn=("#fef5e7", "#9c640c", "#f8c471"),
    chip_bad=("#fdedec", "#a93226", "#f5b7b1"), chip_info=("#eef3f8", "#34495e", "#c9d6e3"),
    plot_bg="#ffffff", plot_axes="#ffffff", plot_fg="#1b2631", plot_grid="#d5dbe1",
)
_DARK = dict(
    dark=True, page=_DARK_PAGE, card=_DARK_CARD, border=_DARK_BORDER, fg=_DARK_FG, text=_DARK_FG, muted="#9aa1aa",
    hint="#8b929b", good="#4cd88a", warn="#f2b84b", bad="#ff7b6e", busy="#5dade2", todo="#7f8790",
    chip_good=("rgba(34, 192, 107, 0.16)", "#6fe3a3", "rgba(34, 192, 107, 0.45)"),
    chip_warn=("rgba(242, 184, 75, 0.15)", "#f5c46b", "rgba(242, 184, 75, 0.45)"),
    chip_bad=("rgba(255, 123, 110, 0.15)", "#ff8f84", "rgba(255, 123, 110, 0.45)"),
    chip_info=("rgba(201, 204, 209, 0.07)", _DARK_FG, _DARK_BORDER),
    plot_bg=_DARK_CARD, plot_axes=_DARK_CARD, plot_fg=_DARK_FG, plot_grid="#41464e",
)


def is_dark(widget=None) -> bool:
    """True when the application (or *widget*) shows a dark theme.

    The app's dashboard style sheet sets the page colour (``modules.ui_kit.theme.PAGE``); a dark
    window palette (light text on a dark window) counts too.
    """
    try:
        from PyQt5.QtGui import QPalette
        from PyQt5.QtWidgets import QApplication
    except Exception:                       # noqa: BLE001 - no Qt: plots stay light
        return False
    app = QApplication.instance()
    if app is None:
        return False
    sheet = (app.styleSheet() or "").lower()
    if _DARK_PAGE.lower() in sheet or "qdarktheme" in sheet:
        return True
    pal = widget.palette() if widget is not None else app.palette()
    return pal.color(QPalette.Window).lightness() < 110 and pal.color(QPalette.WindowText).lightness() > 150


def colors(widget=None, dark: Optional[bool] = None) -> dict:
    """The colour set for the active theme (see the module docstring)."""
    return dict(_DARK if (is_dark(widget) if dark is None else dark) else _LIGHT)


def _luminance(c) -> float:
    from matplotlib.colors import to_rgba
    r, g, b, _a = to_rgba(c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def theme_figure(fig, dark: bool) -> None:
    """Recolour *fig* for the theme: background, axes frames, ticks, labels, titles, legends,
    colour bars, and near-black ("ink") lines and texts drawn for a white page. Images, colour
    maps and coloured lines are kept. Idempotent and cheap (walks the artists once)."""
    from matplotlib.lines import Line2D
    c = _DARK if dark else _LIGHT
    if getattr(fig, "_smile_theme", None) == dark and not dark:
        return
    fig._smile_theme = dark
    fig.set_facecolor(c["plot_bg"])
    fg = c["plot_fg"]

    def ink(col, limit=0.3) -> bool:
        try:
            return _luminance(col) < limit
        except (ValueError, TypeError):
            return False

    def white(col) -> bool:
        try:
            return _luminance(col) > 0.92
        except (ValueError, TypeError):
            return False
    for t in fig.texts:
        if dark and ink(t.get_color(), 0.45):
            t.set_color(fg)
    for ax in fig.axes:
        if ax.get_facecolor() is not None and (white(ax.get_facecolor()) or not dark):
            ax.set_facecolor(c["plot_axes"])
        for spine in ax.spines.values():
            spine.set_edgecolor(c["plot_grid"] if dark else spine.get_edgecolor())
        ax.tick_params(colors=fg, which="both")
        for lab in (ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title):
            if dark and ink(lab.get_color(), 0.45):
                lab.set_color(fg)
        if not dark:
            continue
        for t in ax.texts:
            if ink(t.get_color(), 0.45):
                t.set_color(fg)
            patch = t.get_bbox_patch()
            if patch is not None and white(patch.get_facecolor()):
                patch.set_facecolor(c["card"])
                patch.set_edgecolor(c["border"])
        for line in ax.get_lines():
            if isinstance(line, Line2D) and ink(line.get_color()):
                line.set_color(fg)
        for coll in ax.collections:
            try:
                ec = coll.get_edgecolor()
                if len(ec) and ink(ec[0]) and not len(coll.get_array() if coll.get_array() is not None else []):
                    coll.set_edgecolor(fg)
            except Exception:              # noqa: BLE001 - leave odd collections as drawn
                pass
        for g in ax.get_xgridlines() + ax.get_ygridlines():
            g.set_color(c["plot_grid"])
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_facecolor(c["card"])
            leg.get_frame().set_edgecolor(c["border"])
            for t in leg.get_texts():
                t.set_color(fg)
            if leg.get_title() is not None:
                leg.get_title().set_color(fg)
