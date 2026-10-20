# ui_kit

Small, application-independent PyQt5 building blocks: a dual-handle range slider, a
zoom/pan mixin, an edge-resizable MDI sub-window and a macOS-style title bar.

## Overview

| Class | File | Purpose |
|-------|------|---------|
| `RangeSlider` | `range_slider.py` | two-handle slider, values are fractions `0.0..1.0` |
| `ZoomPanMixin` | `zoom_pan.py` | wheel zoom + middle-button pan for any `QWidget` |
| `ResizableMdiSubWindow` | `resizable_subwindow.py` | `QMdiSubWindow` resizable from every edge/corner |
| `IOSTitleBar`, `TrafficLightButton` | `title_bar.py` | dark custom title bar with traffic-light buttons |
| `dashboard_stylesheet` | `theme.py` | recolours the qdarktheme QSS to a green accent (colours only; splitter handles a lighter grey so the drag bars show) |

## Requirements

* Python 3.9+, PyQt5
* `qtawesome` (only `title_bar.py`, for the hover icons)
* No other `modules.*` package (`__requires__ = ()`).

## Stand-alone use

```
python -m modules.ui_kit
```

opens a window with a `RangeSlider` and a read-out of its range.

## Embedding in an application

```python
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from modules.ui_kit import RangeSlider, ZoomPanMixin

app = QApplication([])
panel = QWidget()
layout = QVBoxLayout(panel)
slider = RangeSlider(low=0.02, high=0.98)
label = QLabel()
slider.range_changed.connect(lambda lo, hi: label.setText(f"{lo:.2f} .. {hi:.2f}"))
layout.addWidget(slider)
layout.addWidget(label)
panel.show()


# Zoom/pan for your own image widget:
class Canvas(QWidget, ZoomPanMixin):
    def __init__(self):
        super().__init__()
        self._init_zoom_pan()

    def _zp_image_size(self):            # (width, height) of the image in pixels
        return 640, 480

    def wheelEvent(self, e):
        self._zp_wheel_event(e)
    # also forward mouse press/move/release/double-click to _zp_mouse_*()
```

`IOSTitleBar(window, title)` needs the window it controls:

```python
bar = IOSTitleBar(main_window, "My app")
main_window.setMenuWidget(bar)
bar.inject_menubar(main_window.menuBar())
```

## Data formats / conventions

* `RangeSlider` positions are floats in `0.0 .. 1.0`; the minimum gap while dragging is `0.01`.
* Zoom 1.0 (100 %) = image fits the widget; limits 0.05 .. 100. Image coordinates are
  `(x, y) = (column, row)`; widget coordinates are pixels.
* `ResizableMdiSubWindow`: `MARGIN = 8` px edge zone, `MIN_W = 200`, `MIN_H = 100`.

## API summary

* `RangeSlider(low=0.02, high=0.98)`: `low`, `high`, `set_range(low, high)` (silent),
  `reset(low, high)` (emits); signal `range_changed(float, float)`.
* `ZoomPanMixin`: `zoom_percent`, `zoom_fit()`, `zoom_actual()`, `zoom_in()`, `zoom_out()`;
  host implements `_zp_image_size()` and optionally `_zp_on_zoom_changed()`.
* `ResizableMdiSubWindow`: use like `QMdiSubWindow`; call `setWidget(...)` after the content is built.
* `IOSTitleBar(window, title)`: `inject_menubar(menubar)`, `title_label`, `btn_close/minimize/maximize`.

`__widgets__ = ("RangeSlider", "ResizableMdiSubWindow")` (constructible with no arguments).

## Threading

Everything is GUI-thread only. `import modules.ui_kit` does not import PyQt5; classes are
loaded lazily on first access.

## Tests

`tests/test_module_contract.py` (`-k ui_kit`) checks the contract and builds every `__widgets__`
class inside a `QMainWindow`; `python tools/module_audit.py` prints the audit table.
