"""Band reads of ENVI files in every interleave (used by the radiometric BSQ generator)."""
import numpy as np
import pytest

from modules.cube_io import envi
from modules.cube_io.envi import _read_envi_band_float32, parse_envi_header, release_band_maps

L, S, B = 5, 7, 4


@pytest.fixture(autouse=True)
def _release_maps():
    yield
    release_band_maps()          # let pytest delete tmp_path files (Windows)


def cube():
    return np.arange(L * S * B, dtype=np.float32).reshape(L, S, B)       # (lines, samples, bands)


def write(path, interleave, dtype, dt_code, big_endian=False, offset=0):
    data = cube().astype(dtype)
    data = {'bip': data, 'bil': data.transpose(0, 2, 1), 'bsq': data.transpose(2, 0, 1)}[interleave]
    data = data.astype(np.dtype(dtype).newbyteorder('>' if big_endian else '<'))
    with open(path, 'wb') as f:
        f.write(b'\0' * offset)
        f.write(np.ascontiguousarray(data).tobytes())
    with open(str(path) + '.hdr', 'w') as h:
        h.write(f'ENVI\nsamples = {S}\nlines = {L}\nbands = {B}\ndata type = {dt_code}\n'
                f'interleave = {interleave}\nbyte order = {int(big_endian)}\nheader offset = {offset}\n')
    return parse_envi_header(str(path))


@pytest.mark.parametrize('interleave', ['bsq', 'bil', 'bip'])
@pytest.mark.parametrize('dtype,dt_code', [(np.uint16, 12), (np.float32, 4), (np.int16, 2)])
@pytest.mark.parametrize('big_endian,offset', [(False, 0), (True, 0), (False, 128)])
def test_every_band_matches_the_cube(tmp_path, interleave, dtype, dt_code, big_endian, offset):
    path = tmp_path / f'cube.{interleave}'
    meta = write(path, interleave, dtype, dt_code, big_endian, offset)
    for b in range(B):
        band = _read_envi_band_float32(str(path), meta, b)
        assert band.dtype == np.float32 and band.shape == (L, S)
        np.testing.assert_array_equal(band, cube()[:, :, b])


def test_a_truncated_file_is_reported(tmp_path):
    path = tmp_path / 'cube.bip'
    meta = write(path, 'bip', np.uint16, 12)
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match='header describes'):
        _read_envi_band_float32(str(path), meta, 0)


def test_released_file_can_be_replaced_and_is_read_again(tmp_path):
    """Maps are cached; after release_band_maps the file can be deleted/rewritten (Windows)."""
    path = tmp_path / 'cube.bip'
    meta = write(path, 'bip', np.float32, 4)
    _read_envi_band_float32(str(path), meta, 1)
    release_band_maps(str(path))
    path.unlink()
    assert not path.exists()
    with open(path, 'wb') as f:
        f.write((cube() + 1000).astype(np.float32).tobytes())
    np.testing.assert_array_equal(_read_envi_band_float32(str(path), meta, 1), cube()[:, :, 1] + 1000)
    release_band_maps()


def test_at_most_two_files_stay_mapped(tmp_path):
    for i in range(4):
        path = tmp_path / f'cube{i}.bsq'
        _read_envi_band_float32(str(path), write(path, 'bsq', np.uint16, 12), 0)
    assert len(envi._maps) == 2
    release_band_maps()
    assert not envi._maps


def test_open_calibration_twice_overwrites_the_mapped_scan_file():
    """Each Open Calibration rewrites <tmp>/scan_ScanResult.bip; the previous map must not block it."""
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from modules.radiometric import CalibrationWindow
    first, second = cube(), cube() + 500
    window = CalibrationWindow()
    try:
        for data in (first, second):
            window.load_from_memory(cube_array=data, wavelengths=np.arange(B) + 400.0, name='ScanResult')
            band = _read_envi_band_float32(window.raw_filepath, window.raw_meta, 2)
            np.testing.assert_array_equal(band, data[:, :, 2])
    finally:
        window.close()
        release_band_maps()
        app.processEvents()
