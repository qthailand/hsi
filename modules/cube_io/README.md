# cube_io

## Overview

Cube assembly and file export/import. The core is Qt-free; thin Qt adapters (below) are optional:

* `create_pushbroom_cube_from_corrected` - stack corrected frames into a `(slit, scan, wavelength)` cube.
* `load_envi_cube_for_viewer` - open an ENVI cube as a memory map plus its wavelengths.
* `parse_envi_header`, `read_envi_band` - minimal ENVI header parser and single-band float32 reader.
* `HypercubeWriter` (ENVI BIP), `GeoTIFFWriter`, `SpectrogramWriter` (raw binary), `save_hdf5` - exporters.
* Qt adapters, loaded lazily: `HypercubeWorker`, `GeoTIFFWorker`, `SpectrogramWorker` (`export_workers.py`), `HDF5SaveWorker` (`hdf5_worker.py`), `CubeWorker` (`cube_worker.py`).

## Requirements

* NumPy. Optional, imported only when used: `spectral` (loading ENVI), `h5py` (HDF5), `rasterio` (GeoTIFF).
* Other modules: `spectral_processing` (`__requires__ = ("spectral_processing",)`, for `parse_envi_wavelengths`).
* No PyQt5 for the core; the Qt adapters need PyQt5. `import modules.cube_io` never imports it.

## Stand-alone use

```
python -m modules.cube_io CUBE.hdr                       # inspect: JSON with shape, dtype, wavelength_count
python -m modules.cube_io SPEC.npy --format hdf5 --output out.h5
python -m modules.cube_io SPEC.npy --format spectrogram --output out.spec [--m 0.2903 --c 399.791 --bin 1 --pixel-size 0.1]
python -m modules.cube_io CUBE.npy --format envi --output out.img --wavelengths WL.npy [--m .. --c .. --bin ..]
python -m modules.cube_io CUBE.npy --format geotiff --output out.tif --wavelengths WL.npy
```

For `hdf5` / `spectrogram` the NPY must be 3-D `(frames, spatial, spectral)`; for `envi` / `geotiff` it must be band-first `(wavelength, slit, scan)` and `WL.npy` a 1-D array with one nm value per band. The CLI defaults are `--m .2903 --c 399.791 --bin 1 --pixel-size .1`.

## Embedding in an application

```python
import numpy as np
from modules.cube_io import (create_pushbroom_cube_from_corrected, load_envi_cube_for_viewer,
                             HypercubeWriter, SpectrogramWriter, save_hdf5)

cube = create_pushbroom_cube_from_corrected(frames)       # (slit, scan, wavelength)
band_first = cube.transpose(2, 0, 1)                      # (wavelength, slit, scan) for the writers

writer = HypercubeWriter(save_path="scan.img", config={"m": 0.2903, "c": 399.791},
                         hypercube=band_first, linear_spectrum=wavelengths_nm,
                         camera_bin=1, progress=print)
seconds = writer.run()                                    # blocking; run it in a worker thread

cube, wl = load_envi_cube_for_viewer("scan.img.hdr")      # memmap (lines, samples, bands), wl in nm or None
```

Qt adapters (PyQt5 applications):

```python
from PyQt5.QtCore import QThread
from modules.cube_io import HypercubeWorker, HDF5SaveWorker

worker = HypercubeWorker(save_path="scan.img", config={"m": 0.2903, "c": 399.791},
                         hypercube=band_first, linear_spectrum=wavelengths_nm, camera_bin=1)
thread = QThread()
worker.moveToThread(thread)
thread.started.connect(worker.run)
worker.progressed.connect(progress_bar.setValue)     # int percent
worker.finished.connect(lambda seconds: thread.quit())
worker.error.connect(print)                          # traceback text
thread.start()

saver = HDF5SaveWorker("spec.h5", spectrogram)       # a QThread itself
saver.finished.connect(print); saver.failed.connect(print)
saver.start()
```

## Data formats / conventions

* In memory: corrected frame `(slit, wavelength)`; cube `(slit, scan, wavelength)`. ENVI BIP on disk: `lines = slit`, `samples = scan`, `bands = wavelength`.
* `HypercubeWriter` / `GeoTIFFWriter` take a band-first `(wavelength, slit, scan)` array (uint8, int16, uint16, int32, uint32, int64, uint64, float32 or float64) and `linear_spectrum` in nm. `HypercubeWriter` writes `save_path` (raw BIP in the array's dtype) and `save_path + '.hdr'` (LF line endings; `data type` and `byte order` from the dtype, `interleave = BIP`, `acquisition time` in UTC - the scan's `config['started_at']` when given, else the export time - `wavelength units = Nanometers`, wavelength list). Other dtypes raise `ValueError` before anything is written.
* `GeoTIFFWriter`: LZW, tiled, one band per wavelength with `WAVELENGTH` / `WAVELENGTH_UNIT` tags, in the cube's dtype; no georeference (a pushbroom scan has none).
* `SpectrogramWriter`: 16-byte little-endian header `<fffHH` = `(m * camera_bin, c, pixel_size, height, width)` followed by raw frames (`(frames, height, width)`, array dtype, not recorded). Written to `<path>.tmp` and renamed atomically; cancellable. This file has no `format_version` field.
* `save_hdf5`: dataset `"spectrogram"` (gzip 4, one-frame chunks), attributes `shape`, `dtype`.
* `config` dict: `m` (nm per pixel at hardware bin 1) and `c` (nm), required by `SpectrogramWriter` only. `camera_bin` is the hardware bin factor.
* Scan metadata in the ENVI header (written only when present in `config`; a missing field means unknown): `data bit depth`, `sensor bit depth`, `data full scale`; `sensor temperature start/end/min/max`, `dark temperature`, `target temperature` (°C); `exposure time (ms)`, `sensor gain`, `frame rate (Hz)`, `camera bin`, `dark corrected`, `hot pixel corrected`, `hot pixel count`, `smile corrected` (1/0, what the frames actually went through), `smile model`, `stage type`, `linear speed (um/s)`, `linear speed achieved (um/s)`, `rotation speed (deg/s)`, `rotation speed achieved (deg/s)`. `metadata_from_header` / `envi_scan_metadata` read them back as `config` keys (an opened cube's re-export keeps them), and the radiometric crop/resample outputs copy them (`carried_header_lines`; the raw-count fields only when uncalibrated).
* `parse_envi_header` returns lower-cased string fields, int `lines` / `samples` / `bands` and a float32 `wavelength` array in nm (`wavelength units` of micrometres are converted) or `None`; `read_envi_band` decodes ENVI data types 1, 2, 3, 4, 5, 12, 13, 14 and 15 (others raise `ValueError`), reads BSQ, BIL and BIP through a memory map (only the requested band), honours `byte order` and `header offset`, and keeps the maps of the last two files open: call `release_band_maps(path)` before overwriting or deleting such a file (Windows cannot replace a mapped file).

## API summary

| Name | Purpose |
|------|---------|
| `create_pushbroom_cube_from_corrected(frames)` | list of `(slit, wavelength)` frames -> `(slit, scan, wavelength)` cube (new array) |
| `load_envi_cube_for_viewer(hdr_path)` | `(memmap cube, wavelengths_nm or None)` |
| `parse_envi_header(filepath)` / `read_envi_band(filepath, meta, band_idx)` | header dict / one band as float32 `(lines, samples)` |
| `HypercubeWriter(...).run()` | ENVI BIP export, returns seconds |
| `GeoTIFFWriter(...).run()` | GeoTIFF export, returns seconds |
| `SpectrogramWriter(...).run()` / `.cancel()` | raw spectrogram export, returns seconds |
| `save_hdf5(path, data, progress=None)` | HDF5 export, returns `path` |
| `HypercubeWorker` / `GeoTIFFWorker` / `SpectrogramWorker(**writer_kwargs)` | Qt adapters (lazy); `run()`, signals `finished(float)`, `progressed(int)`, `error(str)` |
| `HDF5SaveWorker(path, data, parent=None)` | `QThread` saving with `save_hdf5`; signals `progress(int)`, `finished(str)`, `failed(str)` |
| `CubeWorker(array, parent=None)` | `QObject`; `run()` emits `ascontiguos(object)`, `finished()` |

## Threading

The core has no Qt. Writer `run()` methods block and are meant for a worker thread; the optional `progress(int percent)` callback is invoked on that thread. `SpectrogramWriter.cancel()` may be called from another thread. Loaders and parsers are stateless. The adapters: `HypercubeWorker` / `GeoTIFFWorker` / `SpectrogramWorker` are `QObject`s built in the GUI thread, moved to a `QThread`, and started by connecting `thread.started` to `run` (signals `finished(float seconds)`, `progressed(int)`, `error(str traceback)`; `SpectrogramWorker.cancel()` is thread-safe). `HDF5SaveWorker` is a `QThread` (`progress(int)`, `finished(str path)`, `failed(str)`); do not modify the array while it runs. `CubeWorker.run` makes an array C-contiguous and emits `ascontiguos(object)` then `finished()`. Keep the data file in place while a memory-mapped cube is in use.

## Tests

Covered by `tests/test_package_regression.py` and `tests/test_stage2_regression.py`; contract checks in `tests/test_module_contract.py`. Run `python -m pytest -q tests`.
