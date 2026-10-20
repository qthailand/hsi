"""Dashboard colours for the dark theme: green accent, dark grey page.

Only colours change; sizes, padding and corner radii stay as the base theme has them::

    app.setStyleSheet(dashboard_stylesheet(qdarktheme.load_stylesheet("dark")))

``dashboard_stylesheet`` moves every blue tint in the base stylesheet (accent,
selection, pressed and focus colours) to the green accent hue. Pastel blues are
made a little more saturated and less bright so the green reads like the accent
instead of mint; dark tints keep their brightness, so hover/pressed shades stay as
distinct as before.

Data conventions:
    Colours are ``'#RRGGBB'`` hex strings (module constants below). The base
    stylesheet writes colours as ``rgba(r, g, b, a)`` with float components.

Threading:
    Pure string building; no Qt import.
"""

from __future__ import annotations

import colorsys
import re

PAGE = "#1e2024"
SIDEBAR = "#25282c"
CARD = "#2b2e33"
FIELD_HOVER = "#3c4047"
BORDER = "#3d4148"
ACCENT = "#22c06b"
ACCENT_DARK = "#1a9a55"
#: Splitter handle bars: the base grey (#3f4042) hardly shows on the page; hover stays the accent.
HANDLE = "#5c6370"

#: Plot colours for pyqtgraph.
PLOT_BACKGROUND = CARD
PLOT_FOREGROUND = "#c9ccd1"

_ACCENT_HUE = colorsys.rgb_to_hsv(0x22 / 255, 0xC0 / 255, 0x6B / 255)[0]
_BLUE_HUES = (190 / 360, 235 / 360)
_MIN_SATURATION = 0.2
_ACCENT_MIN_SATURATION = 0.6
_ACCENT_MAX_VALUE = 0.8
_RGBA = re.compile(r"rgba\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")


def _to_accent_hue(match: re.Match) -> str:
    """Return the ``rgba(...)`` text with a blue hue moved to the accent hue (others unchanged)."""
    r, g, b, a = (float(v) for v in match.groups())
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    if s < _MIN_SATURATION or not _BLUE_HUES[0] <= h <= _BLUE_HUES[1]:
        return match.group(0)
    s, v = max(s, _ACCENT_MIN_SATURATION), min(v, _ACCENT_MAX_VALUE)
    r, g, b = (round(c * 255) for c in colorsys.hsv_to_rgb(_ACCENT_HUE, s, v))
    return f"rgba({r:.3f}, {g:.3f}, {b:.3f}, {a:.3f})"


def dashboard_stylesheet(base: str) -> str:
    """Return ``base`` (the qdarktheme dark QSS) recoloured to the green accent, plus the page
    and splitter handle colours."""
    return _RGBA.sub(_to_accent_hue, base) + f"""
    QMainWindow, QMdiArea {{
        background: {PAGE};
    }}
    QSplitter::handle {{
        background-color: {HANDLE};
    }}
    """
