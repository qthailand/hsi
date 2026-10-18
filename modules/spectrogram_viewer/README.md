# spectrogram_viewer

## Overview

`SpectrogramViewer` is a `QWidget` for browsing a raw spectrogram frame by frame (slider, keyboard) and saving the whole array to HDF5 in the background.

## Requirements

* PyQt5, NumPy; `h5py` for saving (imported lazily by `cube_io`).
* Other modules: `cube_io` (`HDF5SaveWorker`, `save_hdf5`).
* `import modules.spectrogram_viewer` does not import PyQt5.

## Stand-alone use

```
python -m modules.spectrogram_viewer        # demo with a synthetic (40, 128, 160) spectrogram
```

Keys: Left/Right = +-1 frame, Shift+Left/Right = +-10, Home/End = first/last, Ctrl+S = save.

## Embedding in an application

```python
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from modules.spectrogram_viewer import SpectrogramViewer

app = QApplication.instance() or QApplication([])
viewer = SpectrogramViewer()                       # no arguments
viewer.load(np.random.randint(0, 4096, (20, 64, 96), dtype=np.uint16))   # (n, h, v)
host = QMainWindow()
host.setCentralWidget(viewer)
host.show()
```

## Data formats / conventions

* Spectrogram `(n, h, v)` = `(frames, spatial, spectral)`, any numeric dtype (usually uint16); held by reference, never modified. `numpy.memmap` is fine.
* Display: each frame is min-max normalised to 8 bit.
* Saved file: HDF5, dataset `"spectrogram"` (gzip 4, one-frame chunks), attributes `shape`, `dtype` (see `modules.cube_io.save_hdf5`).

## API summary

| Name | Purpose |
|------|---------|
| `SpectrogramViewer(parent=None)` | the widget; no custom signals |
| `SpectrogramViewer.load(spectrogram)` | set the `(n, h, v)` array, show frame 0, enable saving |

## Threading

GUI thread only. Saving runs in a `QThread` (`HDF5SaveWorker`) owned by the viewer, with a progress dialog; do not modify the array while a save runs.

## Tests

Contract checks in `tests/test_module_contract.py` (headless import, embedding in a foreign window). Run `python -m pytest -q tests`.
