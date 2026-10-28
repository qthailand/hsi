"""ENVI export -> read round trips: data type, byte order, wavelengths and header format."""
import sys
import types
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from modules.cube_io.envi import (ENVI_DATA_TYPES, _read_envi_band_float32, envi_data_type,
                                  parse_envi_header, release_band_maps)
from modules.cube_io.writers import GeoTIFFWriter, HypercubeWriter

L, S, B = 5, 7, 4                       # lines (slit), samples (scan), bands
WL = np.array([400.5, 410.25, 420.0, 430.75])
DTYPES = [np.uint8, np.int16, np.uint16, np.int32, np.float32, np.float64]


@pytest.fixture(autouse=True)
def _release_maps():
    yield
    release_band_maps()          # let pytest delete tmp_path files (Windows)


def cube(dtype):
    """(lines, samples, bands) values that fit every dtype, negative ones for signed types."""
    a = np.arange(L * S * B, dtype=np.float64).reshape(L, S, B)
    if np.dtype(dtype).kind == 'f':
        a = a * 0.37 - 20.0
    elif np.dtype(dtype).kind == 'i':
        a = a - 40
    return a.astype(dtype)


def export(path, data, config=None):
    """Write ``data`` (lines, samples, bands) as the cube viewer does (band-first view)."""
    HypercubeWriter(save_path=str(path), config=config if config is not None else {},
                    hypercube=np.transpose(data, (2, 0, 1)), linear_spectrum=WL).run()
    return parse_envi_header(str(path))


def write_by_hand(path, data, interleave, header_extra='', encoding='utf-8'):
    """A third-party style ENVI file (LF header) of ``data`` (lines, samples, bands)."""
    code, order = envi_data_type(data.dtype)
    disk = {'bip': data, 'bil': data.transpose(0, 2, 1), 'bsq': data.transpose(2, 0, 1)}[interleave]
    np.ascontiguousarray(disk).tofile(path)
    with open(str(path) + '.hdr', 'w', newline='\n', encoding=encoding) as h:
        h.write(f'ENVI\nsamples = {S}\nlines = {L}\nbands = {B}\ndata type = {code}\n'
                f'interleave = {interleave}\nbyte order = {order}\n{header_extra}')
    return parse_envi_header(str(path))


def read_all(path, meta):
    return np.stack([_read_envi_band_float32(str(path), meta, b) for b in range(meta['bands'])], axis=-1)


@pytest.mark.parametrize('dtype', DTYPES)
def test_writer_header_matches_the_data(tmp_path, dtype):
    data = cube(dtype)
    path = tmp_path / 'cube.img'
    meta = export(path, data)
    assert int(meta['data type']) == envi_data_type(dtype)[0]
    assert int(meta['byte order']) == 0
    assert (meta['lines'], meta['samples'], meta['bands']) == (L, S, B)
    assert path.stat().st_size == data.nbytes
    np.testing.assert_array_equal(np.fromfile(path, dtype=dtype).reshape(L, S, B), data)
    np.testing.assert_array_equal(read_all(path, meta), data.astype(np.float32))
    np.testing.assert_allclose(meta['wavelength'], WL)
    assert meta['wavelength units'] == 'Nanometers'


@pytest.mark.parametrize('interleave', ['bsq', 'bil', 'bip'])
@pytest.mark.parametrize('dtype', DTYPES + [np.uint32, np.int64, np.uint64])
def test_reader_decodes_every_type_and_interleave(tmp_path, interleave, dtype):
    data = cube(dtype)
    path = tmp_path / f'cube.{interleave}'
    meta = write_by_hand(path, data, interleave)
    np.testing.assert_array_equal(read_all(path, meta), data.astype(np.float32))


@pytest.mark.parametrize('dtype', ['>u2', '>f4', '>i4'])
def test_big_endian_cube_declares_byte_order_1(tmp_path, dtype):
    data = cube(np.dtype(dtype).newbyteorder('=')).astype(dtype)
    path = tmp_path / 'cube.img'
    meta = export(path, data)
    assert int(meta['byte order']) == 1
    np.testing.assert_array_equal(read_all(path, meta), data.astype(np.float32))


def test_float32_export_opens_in_the_viewer_loader(tmp_path):
    """The Cube Viewer's Export Datacube of an opened reflectance cube (spectral reads it back)."""
    from modules.cube_io import load_envi_cube_for_viewer
    data = cube(np.float32)
    path = tmp_path / 'reflectance.img'
    export(path, data)
    loaded, wavelengths = load_envi_cube_for_viewer(str(path) + '.hdr')
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, data)
    np.testing.assert_allclose(wavelengths, WL)
    del loaded


@pytest.mark.parametrize('units,encoding', [('Micrometers', 'utf-8'), ('micrometers', 'utf-8'), ('um', 'utf-8'),
                                            ('µm', 'utf-8'), ('µm', 'cp1252'), ('Microns', 'utf-8')])
def test_micrometre_header_is_converted_to_nm(tmp_path, units, encoding):
    wl_um = ', '.join(str(w / 1000.0) for w in WL)
    meta = write_by_hand(tmp_path / 'cube.bsq', cube(np.uint16), 'bsq',
                         f'wavelength units = {units}\nwavelength = {{{wl_um}}}\n', encoding)
    np.testing.assert_allclose(meta['wavelength'], WL, rtol=1e-6)
    assert meta['wavelength units'] == 'Nanometers'


def test_nanometre_and_other_units_are_kept(tmp_path):
    wl = ', '.join(str(w) for w in WL)
    meta = write_by_hand(tmp_path / 'a.bsq', cube(np.uint16), 'bsq',
                         f'wavelength units = Nanometers\nwavelength = {{{wl}}}\n')
    np.testing.assert_allclose(meta['wavelength'], WL)
    meta = write_by_hand(tmp_path / 'b.bsq', cube(np.uint16), 'bsq',
                         'wavelength units = Index\nwavelength = {1, 2, 3, 4}\n')
    assert meta['wavelength units'] == 'Index'
    np.testing.assert_array_equal(meta['wavelength'], [1, 2, 3, 4])


@pytest.mark.parametrize('code', [6, 9, 99])
def test_unsupported_data_type_is_an_error(tmp_path, code):
    path = tmp_path / 'cube.bsq'
    meta = write_by_hand(path, cube(np.float32), 'bsq')
    meta['data type'] = str(code)
    with pytest.raises(ValueError, match=f'Unsupported ENVI data type .{code}'):
        _read_envi_band_float32(str(path), meta, 0)


@pytest.mark.parametrize('dtype', [np.bool_, np.int8, np.float16, np.complex64])
def test_writer_rejects_types_envi_cannot_store(tmp_path, dtype):
    path = tmp_path / 'cube.img'
    with pytest.raises(ValueError, match='cannot be written to ENVI'):
        export(path, np.zeros((L, S, B), dtype=dtype))
    assert not path.exists()


def test_header_uses_lf_line_endings(tmp_path):
    path = tmp_path / 'cube.img'
    export(path, cube(np.uint16), {'bit_depth': 12, 'sensor_temperature_c_start': 21.5})
    raw = (tmp_path / 'cube.img.hdr').read_bytes()
    assert b'\r' not in raw
    assert raw.startswith(b'ENVI\n')
    assert b'sensor temperature start = 21.50\n' in raw


def test_writer_needs_no_wavelength_calibration(tmp_path):
    """No config['m'] / config['c']: ENVI export still works (they were never used)."""
    meta = export(tmp_path / 'cube.img', cube(np.uint16), {})
    assert meta['bands'] == B


def test_acquisition_time_is_the_scan_start_in_utc(tmp_path):
    meta = export(tmp_path / 'cube.img', cube(np.uint16), {'started_at': '2026-10-27T10:15:30+07:00'})
    assert meta['acquisition time'] == '2026-10-27T03:15:30.000000Z'


def test_acquisition_time_defaults_to_now_in_utc(tmp_path):
    before = datetime.now(timezone.utc)
    meta = export(tmp_path / 'cube.img', cube(np.uint16), {})
    written = datetime.strptime(meta['acquisition time'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
    assert before - timedelta(seconds=1) <= written <= datetime.now(timezone.utc) + timedelta(seconds=1)


@pytest.mark.parametrize('resample', [False, True])
@pytest.mark.parametrize('dtype', [np.int16, np.int32, np.float64, np.uint8])
def test_uncalibrated_crop_keeps_the_input_type(tmp_path, dtype, resample):
    """Radiometric crop / resample without a calibration: header type == data type."""
    from modules.radiometric import core
    data = cube(dtype)
    src = tmp_path / 'in.bip'
    meta = write_by_hand(src, data, 'bip', 'wavelength = {%s}\n' % ', '.join(str(w) for w in WL))
    out = tmp_path / 'out.bip'
    params = dict(raw_filepath=str(src), raw_meta=meta, save_path=str(out), crop_coords=(1, 6, 0, 4),
                  is_wavelength=True, band_count=B, factors=None, rowwise_factors=None,
                  binning_factor=1, lam_target=WL.astype(np.float32))
    (core.create_datacube_resampled_logic if resample else core.create_datacube_logic)(params)
    out_meta = parse_envi_header(str(out))
    assert int(out_meta['data type']) == envi_data_type(dtype)[0]
    assert out.stat().st_size == 4 * 5 * B * np.dtype(dtype).itemsize
    np.testing.assert_allclose(read_all(out, out_meta), data[0:4, 1:6, :].astype(np.float32), rtol=1e-6)


class _FakeDataset:
    def __init__(self, profile):
        self.profile, self.bands, self.tags, self.descriptions = profile, {}, {}, {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update_tags(self, bidx, **tags):
        self.tags[bidx] = tags

    def set_band_description(self, bidx, text):
        self.descriptions[bidx] = text

    def write(self, data, indexes):
        self.bands[indexes] = np.array(data)


@pytest.fixture
def fake_rasterio(monkeypatch):
    """rasterio is optional (not installed in CI): record what the writer asks it to do."""
    opened = []
    rasterio = types.ModuleType('rasterio')
    errors = types.ModuleType('rasterio.errors')
    errors.NotGeoreferencedWarning = type('NotGeoreferencedWarning', (UserWarning,), {})

    def open_(path, mode, **profile):
        opened.append(_FakeDataset(profile))
        return opened[-1]
    rasterio.open, rasterio.errors = open_, errors
    monkeypatch.setitem(sys.modules, 'rasterio', rasterio)
    monkeypatch.setitem(sys.modules, 'rasterio.errors', errors)
    return opened


@pytest.mark.parametrize('dtype', [np.uint16, np.float32, np.float64, np.int16])
def test_geotiff_keeps_dtype_and_has_no_made_up_georeference(tmp_path, fake_rasterio, dtype):
    band_first = np.transpose(cube(dtype), (2, 0, 1))
    GeoTIFFWriter(save_path=str(tmp_path / 'c.tif'), config={}, hypercube=band_first, linear_spectrum=WL).run()
    (dst,) = fake_rasterio
    assert dst.profile['dtype'] == np.dtype(dtype).name
    assert 'crs' not in dst.profile and 'transform' not in dst.profile
    assert dst.profile['predictor'] == (3 if np.dtype(dtype).kind == 'f' else 2)
    for b in range(B):
        assert dst.bands[b + 1].dtype == dtype
        np.testing.assert_array_equal(dst.bands[b + 1], band_first[b])
        assert dst.tags[b + 1]['WAVELENGTH'] == f'{WL[b]:.3f}'


def test_every_envi_code_round_trips_through_envi_data_type():
    for code, dtype in ENVI_DATA_TYPES.items():
        assert envi_data_type(dtype) == (code, 0)
