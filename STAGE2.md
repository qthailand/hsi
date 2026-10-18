# Stage 2 — BSQ/reference calibration and export packages

Install this overlay on the latest HSI with split config, wavelength calibration,
Smile and Grab. No configuration files or calibration data are replaced.
The existing Modules > Open BSQ Generator menu and scan-to-BSQ workflow remain.

## Architecture
modules/radiometric/core.py: numerical rowwise/fluorescence factors, spatial
binning, crop, resampling and temporary cube construction. No Qt dependencies.
modules/radiometric/widget.py: original CalibrationWindow/RoiDialog interaction.
modules/radiometric/workers.py: Qt signals translated into callable callbacks.
modules/radiometric/__main__.py: standalone GUI and offline processing CLI.
modules/cube_io/envi.py: original ENVI header parsing and band reads.
modules/cube_io/writers.py: ENVI, GeoTIFF and custom spectrogram writers.
modules/cube_io/hdf5.py: standalone HDF5 writer with original gzip/chunking.
workers/export_workers.py and workers/hdf5_worker.py: thin Qt adapters.
modules/bsq_calibration.py: import bridge only for external callers; HSI now
imports the radiometric package directly.

Formulas, ROI handling, interpolation, clipping, file layouts, factor profile
NPY files, progress units and cancellation behavior remain unchanged. Calibration
here means the reference-factor operations already present in the BSQ GUI; no new
absolute radiometric model has been introduced. All processing is offline.
Camera acquisition, buffers, Smile, Grab, preview limit and per-frame kernels
are untouched. This stage does not claim a new production FPS measurement.

## Python API
from modules.radiometric import compute_rowwise_factors, compute_fluorescence_factors, process_cube, resample_cube
from modules.cube_io import parse_envi_header, read_envi_band, HypercubeWriter, GeoTIFFWriter, SpectrogramWriter, save_hdf5
Processing functions accept an optional progress_callback(int). Unlike the old
GUI static methods, this is a callable, not a Qt signal object. Qt adapters pass
signal.emit. Cube processing retains the original params dictionary keys.

Export writers use the original keyword arguments plus progress=callable.
writer.run() returns elapsed seconds and raises exceptions; Qt adapters emit
finished/error as before. Hypercube/GeoTIFF writer input is band-first
(bands, lines, samples), while spectrogram/HDF5 input is (frames, height, width).
The HSI viewer still performs its original conversion before export.
Rasterio/pyproj load only when GeoTIFF is selected; h5py only for HDF5.

## Standalone commands (from hsi)
python -m modules.radiometric
python -m modules.radiometric --input scene.bin --output crop.bin --crop 0 400 0 200 --spatial-bin 2
python -m modules.radiometric --input scene.bin --output calibrated.bin --rowwise factors.npy
python -m modules.radiometric --input scene.bin --output resampled.bin --target-grid 420 700 2
python -m modules.radiometric --input reference.bin --output factors.npy --compute rowwise --roi-mask roi.npy --reference-spectrum reference.xlsx --avg-window 20
python -m modules.radiometric --input reference.bin --output fluorescence.npy --compute fluorescence --roi-mask roi.npy
python -m modules.cube_io scene.bin.hdr
python -m modules.cube_io cube_band_first.npy --format envi --output cube.bin --wavelengths wavelengths.npy
python -m modules.cube_io frames.npy --format hdf5 --output frames.h5

Core requirements: numpy, scipy, pandas, openpyxl; spectral for ENVI viewer loading.
GUI: PyQt5, matplotlib (the existing HSI GUI dependencies).
Optional exports: rasterio, pyproj, h5py. No new vendor SDK is required.

## Verification and retained limits
83 tests passed, 1 skipped. Baseline numerical definitions are frozen in test
fixtures without importing the old acquisition code. Comparisons cover ENVI
BSQ/BIL/BIP reads, rowwise/fluorescence factors, calibrated/raw spatial binning,
resampled data and headers, temporary memmaps, byte-identical ENVI output,
GeoTIFF data/metadata, spectrogram layout/cancel, HDF5 and Qt signal delivery.
The existing widget was also constructed and visually inspected offscreen.
Physical-camera and production-PC timing tests were not performed.

Existing format assumptions were intentionally retained for this structural
stage: little-endian, zero-offset input for the extracted ENVI band reader;
uint16/float32 cube-processing paths; and the old fixed GeoTIFF location/scale.
GeoTIFF export does not acquire new geographic metadata. These require separate
behavioral changes and dedicated fixtures, rather than silently changing outputs.
BIP/BIL readers retain the old per-band whole-file read behavior. This stage
removes GUI coupling; it does not yet optimize offline I/O or memory consumption.
