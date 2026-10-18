# modules/sharpness

Live focus aid for a pushbroom spectrograph: a Qt-free sharpness score and a narrow
sidebar widget that shows it.

## Overview

* `sharpness_score(image, axis=None)`: larger = sharper. `axis=None` uses the variance of
  the Laplacian; `axis=0` measures focus along the slit (use this for pushbroom images).
* `ImageSharpness`: small wrapper (`LoadImage(data).SharpnessScore(axis)`).
* `SharpnessPanel` (`widget.py`): 48 px wide `QWidget` with a large number and a vertical
  bar showing `score / running peak`. The number turns amber while `set_scanning(True)`.

## Requirements

numpy and OpenCV (`cv2`) for the score; PyQt5 for the widget only
(`import modules.sharpness` does not import it). No other `modules.*` package.

## Stand-alone use

```
python -m modules.sharpness image.npy --axis 0      # print the score of a saved (slit, wavelength) image
python -m modules.sharpness --gui                   # demo window with the sidebar
```

## Embedding

```python
from modules.sharpness import SharpnessPanel, sharpness_score

panel = SharpnessPanel()                  # no arguments, GUI thread
layout.addWidget(panel)                   # fixed width, expands vertically
panel.set_value(sharpness_score(frame, axis=0))     # e.g. ~5 times per second
panel.set_scanning(True)                  # during a scan
panel.reset()                             # back to the idle em dash
# with modules.acquisition:  thread.sharpness_ready.connect(panel.set_value)
```

The panel has no signals and paints with fixed dark colours.

## Data conventions

Images are 2-D `(slit, wavelength)` of any numeric dtype (converted to float32). The score
is unit-less; the bar uses a running peak that decays by a factor 0.9995 per `set_value`
call (floor 1.0), so after refocusing the bar re-scales within minutes.

## API summary

`sharpness_score`, `ImageSharpness`, `SharpnessPanel.set_value`, `.set_scanning`, `.reset`.

## Threading

The score functions are pure and thread-safe (the acquisition worker computes them);
`SharpnessPanel` must be used from the GUI thread only.

## Tests

`python -m pytest -q tests -k sharpness` and `tests/test_module_contract.py`.
