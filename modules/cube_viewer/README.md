# cube_viewer

## Overview

Interactive viewer for a hyperspectral cube `(slit, scan, wavelength)`:

* `CubeViewerWindow` - the full viewer: band slider / play-pause sweep, single-band and RGB-composite display, zoom and pan, a SPECTRA panel with point / rectangle / lasso tools (a left click adds a pixel spectrum as before; a rectangle or lasso adds the mean of the area with a shaded +-std band), a spectra tree (show / hide, rename, colour, delete, copy), spectra export to plain CSV or XLSX, cube info and an "Export Datacube" dialog (ENVI BIP, GeoTIFF, raw binary).
* `CubeImageCanvas` - single-band image canvas with zoom/pan, `pixel_clicked(ix, iy)`, dot overlay and the rectangle / lasso tools of `roi_draw.py`.
* `RGBCompositeControls` - single-band / RGB toggle, R/G/B band spin boxes (default 645/555/465 nm) and stretch slider.

## Requirements

* PyQt5, NumPy, `pyqtgraph` (spectrum plot); `matplotlib` only indirectly (cube_preview builds the shared tab20 palette with it); `openpyxl` optionally, for the XLSX spectra export (without it only CSV is offered).
* Other modules: `cube_io` (export workers, ENVI loading), `cube_preview`, `live_view` (spectrum plot, display controls), `ui_kit` (zoom/pan, range slider) - see `__requires__`.
* `import modules.cube_viewer` does not import PyQt5; classes load on first attribute access.

## Stand-alone use

```
python -m modules.cube_viewer               # synthetic 96 x 128 x 60 cube
python -m modules.cube_viewer scan.hdr      # ENVI cube (opened as a memory map)
```

Keys: Left/Right = band, Space = play/pause, R = back to band 0 and fit-to-window zoom (the spectra are kept), E = export, `+` `-` `0` `1` = zoom in/out/fit/100 %, Esc = cancel a rectangle / lasso being drawn. In the spectra tree: F2 = rename, Del = delete the selected spectra, Space = show / hide; while the tree has the focus the single-key viewer shortcuts act on the tree instead (see below).

Layout: the image / plot splitter starts at 3:2 (`setSizes([600, 400])`: about 415 / 277 px in the 1000 x 800 MDI window, where it used to pin the image at 200 px); the plot keeps its 160 px minimum. A dashed marker in the plot shows the current band (wavelength, or band index without wavelengths). A single step moves it at once; during the Space sweep and while the slider is dragged it follows at most every 250 ms and jumps to the band when the sweep stops or the slider is released, because every move repaints all the curves under it (see the measurements below).

## Embedding in an application

`CubeViewerWindow` takes a cube, so it is not constructible without arguments (it is not in `__widgets__`; `CubeImageCanvas` and `RGBCompositeControls` are).

```python
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from modules.cube_viewer import CubeViewerWindow

app = QApplication.instance() or QApplication([])
cube = np.random.randint(0, 4096, (32, 48, 40), dtype=np.uint16)   # (slit, scan, wavelength)
wavelengths = np.linspace(400, 1000, 40)                            # nm, or None

viewer = CubeViewerWindow(cube, wavelengths, resolution=1,
                          config={"m": 0.2903, "c": 399.791}, camera_bin=1)
host = QMainWindow()
host.setCentralWidget(viewer)       # or viewer.show() as a top-level window
host.show()
print(viewer.status_info())
```

Closing the widget releases its references to the cube; the host keeps ownership of the array it passed in.

## Data formats / conventions

* Cube `(slit, scan, wavelength)` = `(H, W, bands)`, any numeric dtype, held by reference.
* Wavelengths: 1-D, nm, one per band, or `None` (band indices are then shown and exported).
* Pixel `(ix, iy)` = `(column, row)`; the spectrum is `cube[iy, ix, :]`.
* Export transposes (copy) to band-first `(wavelength, slit, scan)` for the `cube_io` writers; `config` carries `m`, `c` and optional `pixel_size`.

## API summary

| Name | Purpose |
|------|---------|
| `CubeViewerWindow(cube, wavelengths=None, resolution=2, display_mapper=None, config=None, camera_bin=None, source_path=None)` | full viewer; `on_band_changed(band)`, `update_band_image()`, `status_info()`; spectra: `spectrum_set`, `set_tool(name)`, `add_point(ix, iy)`, `add_area(kind, geometry)`, `rename_spectrum(rid, name)`, `set_spectrum_visible(rid, visible)`, `set_spectrum_show_std(rid, show)`, `set_spectrum_color(rid, color)`, `delete_spectra(rids)`, `copy_spectra(rids)`, `export_spectra(path, ids=None)`, `export_metadata()`, `default_export_path()` |
| `spectra_panel.SpectraPanel()` | the sidebar SPECTRA group: signals `tool_changed(str)`, `visibility_changed(int, bool)`, `rename_requested(int, str)`, `color_requested(int, str)`, `show_std_changed(int, bool)`, `copy_requested(list)`, `delete_requested(list)`, `selection_changed(object)`, `export_requested()`; `tool()`, `set_tool`, `add_record`, `update_record(record, note=None)`, `remove_record`, `clear`, `record_ids`, `selected_rids`, `current_rid`, `select`, `edit_name`, `set_band`, `tooltip_text(rid)`, `context_menu(rid)`; attributes `tree`, `btn_export`, `btn_clear_points` |
| `CubeImageCanvas()` | signals `pixel_clicked(int, int)`, `zoom_changed(float)`, `roi_drawn(str, object)`, `roi_discarded(str)`; `set_array`, `set_display_mapper`, `add_click_point`, `clear_click_points`, plus the `RoiDrawMixin` API below |
| `roi_draw.RoiDrawMixin` (canvas and the RGB view `_RoiPixmapLabel`) | `tool`, `set_tool('point' / 'rect' / 'lasso')`, `is_drawing`, `cancel_drawing()`, `drawing_preview()`, `set_roi_overlays(records, selected_rid)`, `set_selected_roi(rid)`, `roi_overlay_ids()`, `set_class_overlay(pixmap, opacity)`, `class_overlay()` |
| `roi_draw.roi_selection(kind, geometry, H, W)` | `(bbox, local_mask, n_px)` of a record geometry, ready for `roi_stats` |
| `jobs.JobRunner(parent=None)` | signals `done(int, object)`, `failed(int, str)`; `submit(rid, fn)`, `cancel(rid)`, `cancel_all()`, `is_pending(rid)`, `pending()`, `shutdown(timeout_ms=5000)` |
| `RGBCompositeControls()` | signals `mode_changed(str)`, `composite_updated(ndarray)`; `set_cube(cube, wavelengths)`, `is_rgb_mode()` |

## Spectra in the viewer (SPECTRA panel)

The SPECTRA group sits in the sidebar slot of the old 'Clear All Points' button. The button moved inside the group and is still `btn_clear_points` with the same slot.

```
[Point] [Rect] [Lasso]                tool buttons (exclusive)
+- Name ----------------- N px -+
| v Spectra (3)                 |
|   [x] # (120,48)            1 |     check box = shown and exported, # = colour
|   [x] # leaf-mean-0     9,801 |
|   [ ] # leaf-mean-1 computing |
+-------------------------------+
[Export...]  [Clear All Points]
```

* **Point** (default) works exactly as before. A left click adds `cube[iy, ix, :]` as a curve named `(ix,iy)`, with a dot in `TAB20_HEX[creation_count % 20]`. The status line still shows 'N spectra'.
* **Rect / Lasso** draw an area (see the drawing tools below). The record is named `<file stem>-mean-<k>` from `source_path`, or `cube-mean-<k>` without one. It is plotted as the per-band mean with a shaded mean +- std band (population std, ddof = 0). A rectangle partly outside the image is clipped to it. An area with no pixel centre inside is discarded with a status message.
* **Tree**:
  * The check box shows or hides the curve, the std band and the image outline together. A hidden record is left out of the export.
  * F2 or a double click renames. The name is stripped and must be non-empty and unique (case-sensitive). A refused name reverts, with the reason in the status bar. After a rename the legend label, `curve.name()`, the image tag and the export header follow.
  * Del deletes the selected records.
  * The context menu has Rename, Set colour... (`QColorDialog`), Show +-sigma (areas only), Copy as text (the export table as TSV, hidden records included) and Delete.
  * Selecting a record draws its outline thick.
  * The tooltip shows kind, N px, bbox, and mean +- std at the current band, or 'computing...'. Names are always shown as text, never as markup (the tooltip escapes them; the labels above and below the image are plain text).
  * A record whose area has no valid pixel in any band (all NaN) is shown as 'failed'. It is not plotted or exported.
* **Keys**: while the tree has the focus it accepts `ShortcutOverride` for Left, Right, Up, Down, Space, Home, End, F2, Delete, E, R, 0, 1, +, = and -. Those keys then move in, check or search the tree instead of stepping bands, playing the sweep or opening the export dialog.
* **Plot**:
  * Each record gets one `PlotDataItem` (record colour, 1.5 px, `name=record.name`).
  * An area also gets a `FillBetweenItem` (alpha 50) built from two off-scene `PlotCurveItem`s. It has no legend entry, and pyqtgraph's own CSV export ignores it.
  * The plot is never cleared: add, rename, hide and delete touch only that record's items. Deleting a record never recolours the others.
  * A click on a legend sample hides the curve; the record, tree and outline follow.
  * The y range fits the visible curves (and shown std bands) with an 8 % margin. Without wavelengths the x axis reads 'Band'.
* **Export...** opens a save dialog with 'CSV (*.csv);;Excel workbook (*.xlsx)'. The default name is `<stem>_spectra.csv` next to the source file. Without openpyxl only CSV is offered, and the button tooltip says `pip install openpyxl`. Only checked, computed records are written; errors go to a message box.
  * `export_spectra(path, ids=None)` does the same without the dialog: XLSX for a `.xlsx` path, CSV otherwise. Explicit `ids` include hidden records.
  * The XLSX metadata (`export_metadata()`) holds source, units (`counts` for integer cubes), dtype, shape `[H, W, B]`, axis order, the saturation level, `camera_bin` and the config scalars `m`, `c`, `pixel_size`, `resolution`, `bit_depth`, `sensor_bit_depth` and `full_scale`. The writer adds `format_version = 1`, `std_ddof = 0`, the pixel rule and `exported_at`.
* **Computation**:
  * A point is read synchronously.
  * An area is computed in the GUI thread when the cube is an in-RAM ndarray and its bounding box holds at most `inline_bytes` = 8 MiB of cube data (px x bands x itemsize).
  * Everything else goes to the `JobRunner` worker, and so does every memory-mapped cube (BSQ / BIL / BIP files) whatever the area's size. The record stays 'pending' until the result arrives: dashed outline, 'computing...' in the tree, 'N spectra (k computing...)' in the status bar.
  * A lasso mask is rasterised in the GUI thread when its edges cross at most 65,536 pixel rows in total, otherwise in the worker.
  * Delete and Clear cancel pending jobs, and a late result of a deleted record is ignored.
  * A lasso whose mask, rasterised in the worker, holds no pixel centre is dropped as if it had never been added: the label says 'Lasso discarded', and if nothing was added after it the next area gets its colour and name (`SpectrumSet.discard`). A failed area shows '<name>: failed' in the label above the image as well as in the tree and the status bar.
  * A numeric `config['full_scale']` is the saturation level counted per band (`n_saturated`). Saturated values are counted, not excluded.
* **Measured** on an i9-12900K, offscreen, in-RAM 1024 x 1762 x 185 uint16 cube with 49 other records (medians):
  * add a point: 2.2 ms; add a 100 x 100 rectangle inline: 7.5 ms;
  * an area at the 8 MiB limit, `roi_stats` alone: 5.4-5.7 ms (150 x 150 uint16), 10-11 ms (106 x 106 float32; 8 ms before the float merge gained its two correction terms);
  * rename 0.3-1.2 ms, hide / show 0.05 ms, delete one record 1.3 ms, canvas repaint with the overlays 1.1 ms;
  * band change 50-60 ms for the band display, plus the band-marker move. pyqtgraph strokes every whole 1.5 px curve again for any strip the marker exposes (a full plot repaint costs the same), so a single step adds about 0.5 ms per noisy point spectrum (50 points: 23-27 ms) and about 0.4 ms per area (30 areas: 9-11 ms). During the Space sweep the marker moves at most every 250 ms: with 50 point spectra the GUI thread spends 60-62 ms of CPU per band step, against 78-80 ms with a marker move on every step;
  * `polygon_mask`: about 0.12-0.16 us per edge-row crossing, so 0.9 ms for a 300-vertex lasso (1,600 crossings) and 16-20 ms for 128,000;
  * a whole-image area: submitted in 0.3 ms, ready after about 0.67 s in the worker; closing the viewer during it returns in 2.2 ms;
  * a pathological 5000-vertex full-height zigzag lasso, whose mask is rasterised in the worker (5.1 M edge-row crossings, 0.67-0.97 s of `polygon_mask`): closing the viewer 50-250 ms after drawing it returns in 8-12 ms (0.8-0.95 s before `polygon_mask` polled `cancel`), and after deleting it a rectangle queued behind it is ready 15-16 ms later.

## ROI statistics (`roi_stats.py`)

numpy + stdlib only, and safe in a worker thread. Coordinates are image corners: pixel `(c, r)` covers `[c, c + 1) x [r, r + 1)`, with x = column = scan and y = row = slit.

* **Pixel rule**: a pixel belongs to an ROI when its centre does.
  * Rectangles use the closed rule `c0 = ceil(min(x) - 0.5)`, `c1 = floor(max(x) - 0.5) + 1` (rows alike), clipped and half-open. So `xmin = 3.5` includes column 3.
  * A lasso uses the even-odd rule at pixel centres, bit-identical to matplotlib's `Path.contains_points`.
* **Statistics** per band: valid-pixel count, mean, population std (ddof = 0), min, max and the number of values `>= saturation`.
  * Integer cubes of at most 16 bits use exact integer sums (`method = 'exact-int'`). Mean and std are correctly rounded, and `sum1` / `sum2` are kept.
  * Float cubes and wider integers use a chunked two-pass float64 (`'float64-two-pass'`) that skips NaN, +-inf and the ignore value per band. Rows are merged about the rounded ROI mean `mu` with the full decomposition `M2 = sum(Q_r) + sum(n_r (m_r - mu)^2) + 2 sum((m_r - mu) e_r) - E^2 / n` (`e_r` and `E` are the residuals of the rounded row and ROI means, all sums `math.fsum`): the std is within 1 ulp of exact rational references, also for data with a large offset, where the merge without the last two terms was up to 3e-9 relative off (`2e9 + {0..3}`) and `numpy.std` is about 4e-9 off (`1e12 + N(0, 1)`). The two terms cost 10-30 % on float cubes.
  * A point is `'point'`: the pixel's values, std 0.
  * A band without valid pixels gives NaN; a record with no valid band is 'failed'.
* **Memory**: only the ROI's bounding box is read, in row chunks of about 1 MiB. The cancel callback is polled once per chunk, and `polygon_mask(..., cancel=)` polls it once per batch of 65,536 edge-row crossings (about 10 ms).
* **Measured** on the same cube: 100 x 100 rectangle 3.2-3.4 ms, 400 x 400 32-33 ms, a 20 % lasso 80 ms plus 1 ms for the mask, the whole image 350-360 ms, each with a few MiB of scratch memory. The float32 path takes about 2.5-3x longer per pixel (400 x 400: 85-99 ms, 100 x 100: 9.3-10.2 ms).

## Spectra records and export

`spectra.py` (numpy only): `SpectrumSet(palette, source_name)` holds one `SpectrumRecord` per point / rect / lasso in creation order (`add`, `remove`, `rename`, `set_color`, `set_stats`, `get`, `records(visible_only)`, `clear`). Names are unique (case-sensitive, stripped, non-empty, no control or line-separator characters; Thai, commas and quotes are fine); defaults are `(ix,iy)` (then `(ix,iy)-2`, `-3`, ...) for points and `<source>-mean-<k>` (k counts areas from 0, skipping a name already taken) for areas. Colour = `palette[creation_count % len]`, assigned once, so deleting never recolours; `clear()` restarts colours and default names (today's look after "Clear All Points") but never reuses a `rid`, so a late worker result for a deleted record is recognised and ignored. `class_name` defaults to the name and follows renames until it is set separately. `set_stats(rid, stats)` makes a record `ready`, or `failed` when `stats` is None or no band has a valid pixel.

`spectra_export.py`: `spectra_table` builds one table that `export_csv`, `export_xlsx` and `spectra_text` (clipboard TSV) all write:

```
wavelength,"(120,48)_mean","(120,48)_std",leaf-mean-0_mean,leaf-mean-0_std
854.17,812.0,0.0,1503.2215338,41.0634577
```

* First column `wavelength` (nm, the cube's own band order) or `band` (0-based) when there are no wavelengths or their count does not match. Then `<name>_mean`, `<name>_std` per record; a point's std is 0.0. Std is the population std (ddof = 0).
* Numbers are written with `repr`, so they read back bit-exactly with `csv` plus `float()` or `pandas.read_csv(path, float_precision="round_trip")`. Plain `pandas.read_csv(path)` uses pandas' faster default float parser and is only accurate to about 1e-12 relative (pandas 2.2.3: about 40 % of `repr` strings come back a few ulp off). float32 wavelengths are written as their shortest round-trip text (`854.17`). An empty cell is a band with no valid pixel (NaN).
* Exported records (one rule for CSV, XLSX and clipboard): state `ready` with stats, and visible (unchecked) unless `visible_only=False` is passed (e.g. "Copy as text" on a hidden selected record); pending and failed records are always skipped; nothing exportable raises `ValueError`. `show_std` only affects the plot.
* **CSV**: a plain table in exactly the style of the user's `PVC_reflectance_with_std.csv`: header row, then numbers, CRLF, no comment or metadata rows, UTF-8 without BOM (`utf-8-sig` only when a header has non-ASCII text, so Excel shows Thai names). Point names contain a comma, so the csv module quotes those header cells. **The CSV deliberately omits `format_version`** (user decision 2026-10-19, matching their PVC files; an exception to MODULE_CONTRACT "File formats"). The XLSX `Metadata` sheet carries `format_version = 1`.
* **XLSX** (needs `openpyxl`; `HAVE_OPENPYXL`; no pandas): sheets `Spectra` (the same table), `ROIs` (rid, name, class, kind, n_px, min n_valid, max n_saturated, pixel bbox, lasso vertices as JSON, colour, visible, created_at, method) and `Metadata` (format_version, exported_at, std_ddof, pixel rule, wavelength order, then the caller's source, units, dtype, shape, axis_order and camera scalars). openpyxl formats floats with `%.16g`, which is lossy, so floats go in as numeric cells holding `repr` text. `.xlsx` is appended when it is missing.
* Files are written to a temporary file next to the target and moved into place. A failed export, for example when the file is open in Excel, leaves the old file untouched and no temporary file behind. `export_csv` appends `.csv` only when the name has no extension.
* Measured on an i9-12900K for 50 spectra x 185 bands (median of 15 / 7 runs): table 0.4 ms, CSV about 10 ms (278 kB), TSV about 10.5 ms, XLSX about 190-210 ms (lossy plain openpyxl: about 100 ms). Importing `spectra_export` cold takes about 240 ms once per process, about 225 ms of it openpyxl; `spectra` (with `roi_stats`) about 11 ms.

## Drawing tools and overlays (`roi_draw.py`)

`RoiDrawMixin` gives both image views (the single-band `CubeImageCanvas` and the RGB view `_RoiPixmapLabel`, a subclass of cube_preview's `CenteredPixmapLabel`, which is not changed) three tools:

* `point` (default): today's click - `pixel_clicked` / `clicked(ix, iy)`.
* `rect`: drag a rubber band; the box snapped by `roi_stats.snap_rect` (closed pixel-centre rule) is drawn on the pixel edges with its size while dragging, so what you see is what is averaged. Release emits `roi_drawn('rect', {'c0', 'r0', 'c1', 'r1'})`.
* `lasso`: freehand; a vertex is added after at least 2 widget px and 1 image px of movement (at most 5000); release closes it and emits `roi_drawn('lasso', {'vertices': [[x, y], ...]})` in float image-corner coordinates.

A rectangle without a pixel centre, a lasso with fewer than 3 vertices or whose bounding box holds no centre, and Esc all end in `roi_discarded(reason)` instead. Points do not go through `roi_drawn`. Wheel zoom and middle-button pan work while drawing (the drawing is kept in image coordinates); a double click still zooms to fit, and a lasso closes on release, never on a double click. `roi_selection(kind, geometry, H, W)` turns a geometry into `(bbox, local_mask, n_px)` for `roi_stats` - a thin lasso can still contain 0 px, which the caller must handle.

`set_roi_overlays(records, selected_rid)` draws `SpectrumRecord`s: areas as a 1 px outline in the record colour with a 1 px dark drop shadow (dashed while pending, dotted when failed, 2.5 px on a dark halo when selected) plus a name tag; points as today's dot with the name as tag. Each view keeps its old tag placement: the RGB view flips a tag to the left of its dot at the right edge and clamps it to the top edge, as `CenteredPixmapLabel` always did, while the single-band canvas does neither, as its old click dots did (`_roi_point_tag_inside`). Hidden records are not drawn. Records are kept by reference, so after an in-place edit a repaint is enough. `set_class_overlay(pixmap, opacity)` is the phase-2 hook for a class map (drawn over the image, under the outlines, nearest neighbour); nothing produces one yet.

Measured paint cost (offscreen, i9-12900K, 1000 x 640 view of a 1762 x 1024 image; 50 records = 30 rects, 14 lassos of 300 vertices, one self-overlapping 5000-vertex lasso, 5 points; medians, which vary by up to 2x run to run on this CPU):

* The rendered record layer is cached as a pixmap until the size, zoom / pan, selection or a shown record attribute changes: a repaint then adds 0.5 ms at fit (base paint about 1 ms), and band changes and drawing reuse it.
* A re-render (every pan / zoom frame) adds about 3-9 ms; the first one, which also builds the cached outline paths, 9-12 ms at fit.
* A 5000-vertex lasso being drawn: 3-5 ms per paint, 5-11 us per mouse move. Outlines are 1 px lines because Qt draws those on a fast path: the self-overlapping 5000-vertex lasso takes 0.4 ms as a 1 px line and 60 ms as a 3.5 px halo plus 1.5 px line.
* The base image itself still costs 45-50 ms per paint at 1000 % zoom (existing per-paint `pixmap.scaled`, unchanged).

## Threading

All widgets live in the GUI thread. The export dialog moves a `cube_io` export worker to its own `QThread`; closing the dialog or the viewer joins that thread (blocks until a running export ends).

The viewer computes points and small in-RAM areas in the GUI thread, and everything else on its `JobRunner` (see "Spectra in the viewer"). `closeEvent` runs in this order: it stops the timers, shuts the job runner down (cancel, quit, wait; a running `roi_stats` stops within one chunk, a lasso mask rasterised in the worker within one `polygon_mask` batch of about 10 ms), clears the spectra and the overlays, closes the export dialog, and only then drops the cube. It is safe to call twice.

`jobs.JobRunner` runs ROI statistics in the background: one persistent `QThread` with one `_JobWorker` (the export dialog's pattern), started on the first `submit(rid, fn)`. Jobs run FIFO, one at a time; `fn(cancel)` gets a `threading.Event.is_set` to poll (`roi_stats(..., cancel=cancel)` polls it once per ~1 MB chunk, `roi_selection(..., cancel=cancel)` / `polygon_mask` once per batch of crossings). At most one job per record id: submitting again cancels the earlier one. After `cancel(rid)` returns nothing is emitted for that job, even when its result is already queued, so a late result of a deleted record never arrives. `done(rid, result)` and `failed(rid, 'Type: message')` arrive in the GUI thread. `shutdown()` (call it in `closeEvent` before dropping the cube) cancels everything, quits the thread and waits; if its timeout (default 5 s) expires, the thread is parked until it finishes instead of being destroyed while running.

Measured on the in-RAM 1024 x 1762 x 185 uint16 cube: a whole-image `roi_stats` job takes 370-395 ms in the worker while a 2 ms GUI timer keeps firing (largest gap 3.3 ms, the same as with no job running); `cancel` stops it within 0.2-0.3 ms; `shutdown()` during it returns in 0.8-1.3 ms; a trivial job's submit-to-`done` round trip is about 1.1 ms (bounded by the polling loop of the measurement).

## Tests

Contract checks in `tests/test_module_contract.py`; export adapters are exercised by `tests/test_stage2_regression.py`. Cube viewer test files:

* `tests/test_cube_viewer_roi_draw.py`: ROI drawing and overlays (QTest drags offscreen, both views).
* `tests/test_cube_viewer_jobs.py`: the job runner.
* `tests/test_cube_viewer_phase0.py`: the splitter, band marker, R key and import rule.
* `tests/test_cube_viewer_roi_stats.py`: ROI statistics.
* `tests/test_cube_viewer_spectra_export.py`: records and the CSV / XLSX / TSV export.
* `tests/test_cube_viewer_window.py`: the window with the SPECTRA panel. It covers point mode (unchanged), rect / lasso through the widgets, pending -> ready in the worker, memory maps, rename, delete / hide, the band marker (also during the sweep), tree focus keys, close during a job (also during a worker-side lasso mask), failed areas, the legend samples, names shown as text and never as markup, the stand-alone launcher's `source_path`, the context menu and the export dialog.

Run `python -m pytest -q tests`.
