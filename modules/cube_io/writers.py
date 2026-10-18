"""Qt-independent export services preserving existing file formats and metadata.

Data conventions:
    * ``HypercubeWriter`` / ``GeoTIFFWriter``: ``hypercube`` is band-first
      ``(wavelength, slit, scan)`` uint16; ``linear_spectrum`` is the 1-D band
      wavelengths in nm.  The ENVI output is BIP (``lines = slit``,
      ``samples = scan``, ``bands = wavelength``), little-endian, data type 12.
    * ``SpectrogramWriter``: ``spectrogram`` is ``(frames, height, width)``.  The
      binary file is a 16-byte little-endian header ``<fffHH`` =
      ``(m * camera_bin, c, pixel_size, height, width)`` followed by the raw
      frames (C order, array dtype, not recorded in the header).  There is no
      ``format_version`` field.
    * ``config`` is a dict with calibration ``m`` (nm/pixel at bin 1) and ``c`` (nm);
      ``camera_bin`` is the hardware binning factor.

Threading:
    Each writer's ``run()`` is blocking and meant for a worker thread; ``progress``
    is called on that thread with an int percentage.  ``SpectrogramWriter.cancel()``
    is safe to call from another thread (sets a flag checked once per frame).

Example:
    >>> import numpy as np, os, tempfile
    >>> from modules.cube_io.writers import HypercubeWriter
    >>> from modules.cube_io.envi import parse_envi_header
    >>> path = os.path.join(tempfile.mkdtemp(), "cube.img")
    >>> cube = np.zeros((4, 2, 3), np.uint16)                 # (wavelength, slit, scan)
    >>> seconds = HypercubeWriter(save_path=path, config={"m": 0.29, "c": 400.0},
    ...                           hypercube=cube,
    ...                           linear_spectrum=np.linspace(400, 410, 4)).run()
    >>> meta = parse_envi_header(path)
    >>> meta["lines"], meta["samples"], meta["bands"]
    (2, 3, 4)
"""
import os
import struct
import time
import traceback
from datetime import datetime
import numpy as np

#: Scan temperatures (°C) written to the ENVI header when present in ``config``
#: (``ScanController.last_scan_temperatures``): ``(config key, header field)``.
TEMPERATURE_HEADER_FIELDS = (
    ('sensor_temperature_c_start', 'sensor temperature start'),
    ('sensor_temperature_c_end', 'sensor temperature end'),
    ('sensor_temperature_c_min', 'sensor temperature min'),
    ('sensor_temperature_c_max', 'sensor temperature max'),
    ('dark_temperature_c', 'dark temperature'),
    ('target_temperature_c', 'target temperature'),
)

class GeoTIFFWriter:
    """Write a band-first cube as a tiled, LZW-compressed multi-band GeoTIFF.

    The georeferencing is a fixed placeholder (origin lat/lon 7.007027, 100.499458,
    EPSG:32647, 0.1 m pixels); it does not use the ``config``/pixel size.
    Requires ``pyproj`` and ``rasterio`` (imported when constructing).
    """

    def __init__(self, *, save_path: str, config: dict, hypercube: np.ndarray, linear_spectrum: np.ndarray, camera_bin: int = 1, progress=None):
        """Prepare the GeoTIFF profile (no file is written yet).

        Args:
            save_path: Output ``.tif`` path.
            config: Calibration dict with ``m`` and ``c`` (nm); validated in ``run``.
            hypercube: Band-first ``(wavelength, slit, scan)`` uint16 array; kept
                by reference, not copied.
            linear_spectrum: 1-D band wavelengths (nm), one per band.
            camera_bin: Camera hardware bin (``0``/``None`` mean 1).
            progress: Optional ``callable(int percent)``.

        Raises:
            ValueError: ``hypercube`` is not 3-D (tuple unpacking of its shape).
            ImportError: ``pyproj``/``rasterio`` missing.
        """
        self.progress = progress or (lambda value: None)
        from pyproj import Transformer
        from rasterio.control import GroundControlPoint
        from rasterio.crs import CRS
        from rasterio.transform import Affine, from_gcps
        self.save_path = save_path
        self.config = config
        self.camera_bin = int(camera_bin or 1)
        self.hypercube = hypercube
        self.linear_spectrum = linear_spectrum
        self.count, self.hight, self.width = self.hypercube.shape

        lat_tl = 7.007027108462115
        lon_tl = 100.49945822505273
        px = 0.1
        py = 0.1

        t = Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)
        X_left, Y_top = t.transform(lon_tl, lat_tl)
        transform = Affine(px, 0.0, X_left, 0.0, -py, Y_top)

        right = X_left + self.width * px
        bottom = Y_top - self.hight * py

        gcps = [
            GroundControlPoint(row=0, col=0, x=X_left, y=Y_top),
            GroundControlPoint(row=0, col=self.width - 1, x=X_left + (self.width - 1) * px, y=Y_top),
            GroundControlPoint(row=self.hight - 1, col=self.width - 1, x=X_left + (self.width - 1) * px, y=Y_top - (self.hight - 1) * py),
            GroundControlPoint(row=self.hight - 1, col=0, x=X_left, y=Y_top - (self.hight - 1) * py),
        ]

        transform = from_gcps(gcps)
        self.geo_meta = {
            'tiled': True,
            'driver': 'GTiff',
            'transform': transform,
            'crs': CRS.from_epsg(32647),
            'width': self.width,
            'height': self.hight,
            'count': self.count,
            'dtype': np.uint16,
            'tiled': True,
            'compress': 'LZW',
            'predictor': 2,
        }


    def run(self):
        """Write the GeoTIFF band by band, with per-band wavelength tags.

        Returns:
            float: Elapsed seconds (rounded to 0.01).

        Raises:
            ValueError: ``hypercube`` is ``None`` or has no shape.
            TypeError: ``config['m']`` is missing/non-numeric.
            Exception: rasterio/IO errors propagate.  Not cancellable; the file
                is written in place (no temporary file).
        """
        import rasterio
        t0 = time.time()
        try:
            if self.hypercube is None or not hasattr(self.hypercube, "shape"):
                raise ValueError("spectrogram is None or not a numpy array")
            bin_factor = int(getattr(self, 'camera_bin', 1) or 1)
            m = float(self.config.get('m')) * bin_factor
            c = float(self.config.get('c'))
            percents = 0

            with rasterio.open(self.save_path, "w", **self.geo_meta) as dst:
                for i in range(self.count):
                    data = self.hypercube[i]
                    wl = self.linear_spectrum[i]
                    dst.update_tags(bidx=i + 1, WAVELENGTH=f"{wl:.3f}", WAVELENGTH_UNIT="nm")
                    dst.set_band_description(i + 1, f"Band {i + 1} (λ={wl:.3f} nm)")
                    dst.write(data, indexes=i + 1)
                    percents = int(((i + 1) / self.count) * 100)
                    self.progress(percents)
            return float(np.round(time.time() - t0, 2))
        except Exception:
            raise


class HypercubeWriter:
    """Write a band-first cube as ENVI BIP ``save_path`` plus ``save_path + '.hdr'``."""
    chunk_size = 100 * 1024 * 1024
    def __init__(self, *, save_path: str, config: dict, fsync: bool = True, hypercube: np.ndarray, linear_spectrum: np.ndarray, camera_bin: int = 1, progress=None):
        """Store the export parameters (no file is written yet).

        Args:
            save_path: Data file path; the header goes to ``save_path + '.hdr'``.
            config: Calibration dict with ``m`` and ``c`` (nm); validated in ``run``.
                Optional ``bit_depth``, ``sensor_bit_depth`` and ``full_scale`` (from
                :func:`modules.acquisition.camera_bit_info`) are written to the header as
                ``data bit depth``, ``sensor bit depth`` and ``data full scale``; optional scan
                temperatures (°C, see :data:`TEMPERATURE_HEADER_FIELDS`) as ``sensor temperature
                start/end/min/max``, ``dark temperature`` and ``target temperature``.
            fsync: Accepted for API symmetry; currently unused.
            hypercube: Band-first ``(wavelength, slit, scan)`` array, kept by
                reference.  The header always declares data type 12 (uint16).
            linear_spectrum: 1-D band wavelengths in nm (written to the header).
            camera_bin: Camera hardware bin (``0``/``None`` mean 1).
            progress: Optional ``callable(int percent)``.
        """
        self.progress = progress or (lambda value: None)
        self.save_path = save_path
        self.config = config
        self.camera_bin = int(camera_bin or 1)
        self.hypercube = hypercube
        self.linear_spectrum = linear_spectrum
        self.fsync = fsync


    def run(self):
        """Write the BIP data (one slit line at a time) and the ENVI header.

        Returns:
            float: Elapsed seconds (rounded to 0.01).

        Raises:
            ValueError: ``hypercube`` is ``None``/not an array, or not 3-D.
            TypeError: ``config['m']`` is missing/non-numeric.
            OSError: file errors propagate.  Not cancellable, not atomic.
        """
        t0 = time.time()
        try:
            if self.hypercube is None or not hasattr(self.hypercube, "shape"):
                raise ValueError("spectrogram is None or not a numpy array")

            bands, lines, samples = self.hypercube.shape

            bin_factor = int(getattr(self, 'camera_bin', 1) or 1)
            m = float(self.config.get('m')) * bin_factor
            c = float(self.config.get('c'))
            percents = 0

            bip = self.hypercube.transpose(1,2,0)
            with open(f'{self.save_path}', 'wb') as fb:
                for i in range(bip.shape[0]):
                    fb.write(bip[i, :, :].copy())
                    percents = int(((i + 1) / bip.shape[0]) * 100)
                    self.progress(percents)

            acquisition = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

            with open(f'{self.save_path}.hdr', "w") as file:
                file.write('ENVI' + os.linesep)
                file.write('description = {Truth image generated by NARIT Feat. PSU}' + os.linesep)
                file.write('data type = 12' + os.linesep)
                file.write('byte order = 0' + os.linesep)
                file.write('interleave = BIP' + os.linesep)
                file.write(f'acquisition time={acquisition}' + os.linesep)
                file.write(f'lines = {lines}' + os.linesep)
                file.write(f'samples = {samples}' + os.linesep)
                file.write(f'bands = {bands}' + os.linesep)
                for key, name in (('bit_depth', 'data bit depth'), ('sensor_bit_depth', 'sensor bit depth'),
                                  ('full_scale', 'data full scale')):
                    if self.config.get(key) is not None:
                        file.write(f'{name} = {int(self.config[key])}' + os.linesep)
                for key, name in TEMPERATURE_HEADER_FIELDS:
                    if self.config.get(key) is not None:
                        file.write(f'{name} = {float(self.config[key]):.2f}' + os.linesep)
                file.write('wavelength units = Nanometers' + os.linesep)
                file.write('wavelength = {%s}' % ', '.join(self.linear_spectrum.astype('str').tolist()) + os.linesep)
            return float(np.round(time.time() - t0, 2))

        except Exception:
            raise


class SpectrogramWriter:
    """Write a raw spectrogram (header + frames) atomically and cancellably.

    Data goes to ``save_path + '.tmp'`` and is renamed to ``save_path`` on success;
    the temporary file is removed on error or cancellation.
    """
    def __init__(self, *, save_path: str, config: dict, spectrogram: np.ndarray, pixel_size: float, camera_bin: int = 1, fsync: bool = True, progress=None):
        """Store the export parameters (no file is written yet).

        Args:
            save_path: Output file path (parent directory is created).
            config: Calibration dict with ``m`` (nm/pixel at bin 1) and ``c`` (nm).
            spectrogram: ``(frames, height, width)`` array or memmap; iterated
                per frame, kept by reference.
            pixel_size: Spatial pixel size written to the header.
            camera_bin: Camera hardware bin; the stored slope is ``m * camera_bin``.
            fsync: ``os.fsync`` before the final rename.
            progress: Optional ``callable(int percent)``.
        """
        self.progress = progress or (lambda value: None)
        self.save_path = save_path
        self.config = config
        self.spectrogram = spectrogram
        self.pixel_size = float(pixel_size)
        self.camera_bin = int(camera_bin or 1)
        self.fsync = fsync
        self._cancelled = False


    def cancel(self):
        """Request cancellation; ``run`` stops before the next frame (thread-safe flag)."""
        self._cancelled = True


    def run(self):
        """Write the file.

        Returns:
            float: Elapsed seconds (rounded to 0.01); ``0.0`` for an empty
            spectrogram.  On cancellation the partial file is deleted,
            ``progress(0)`` is called and the elapsed time is returned.

        Raises:
            ValueError: ``spectrogram`` is ``None``/not an array or has fewer
                than 3 dimensions.
            TypeError: ``config['m']`` is missing/non-numeric.
            OSError: file errors propagate (the temporary file is removed).
        """
        t0 = time.time()
        try:
            if self.spectrogram is None or not hasattr(self.spectrogram, "shape"):
                raise ValueError("spectrogram is None or not a numpy array")

            shp = self.spectrogram.shape
            if len(shp) < 3:
                raise ValueError(f"Invalid spectrogram shape: {shp} (expected at least 3 dims: N,H,W)")
            number, height, width = shp[0], shp[1], shp[2]
            if number == 0:
                self.progress(100)
                return float(0.0)
                return

            bin_factor = int(getattr(self, 'camera_bin', 1) or 1)
            m = float(self.config.get('m')) * bin_factor
            c = float(self.config.get('c'))
            header_bin = struct.pack('<fffHH', m, c, self.pixel_size, height, width)

            tmp_path = self.save_path + ".tmp"
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)

            last_pct = -1
            with open(tmp_path, mode='wb', buffering=128 * 1024 * 1024) as f:
                f.write(header_bin)
                for i, frame in enumerate(self.spectrogram):
                    if self._cancelled:
                        try:
                            f.flush()
                            if self.fsync:
                                os.fsync(f.fileno())
                        except Exception:
                            pass
                        f.close()
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                        self.progress(0)
                        return float(np.round(time.time() - t0, 2))
                        return
                    f.write(frame.tobytes())
                    pct = int((i + 1) * 100 / number)
                    if pct != last_pct:
                        self.progress(pct)
                        last_pct = pct

                f.flush()
                if self.fsync:
                    os.fsync(f.fileno())

            os.replace(tmp_path, self.save_path)
            return float(np.round(time.time() - t0, 2))

        except Exception:
            try:
                if os.path.exists(self.save_path + ".tmp"):
                    os.remove(self.save_path + ".tmp")
            except Exception:
                pass
            raise


