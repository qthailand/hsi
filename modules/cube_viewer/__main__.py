"""Stand-alone cube viewer: ``python -m modules.cube_viewer [file.hdr]``.

Opens the given ENVI header (cube loaded as a memory map) or, without an argument, a
synthetic ``(slit, scan, wavelength)`` cube so the viewer can be tried immediately.
"""
from __future__ import annotations

import argparse


def synthetic_cube(slit: int = 96, scan: int = 128, bands: int = 60):
    """Build a demo cube.

    Returns:
        tuple: ``(cube, wavelengths_nm)`` with ``cube`` uint16 ``(slit, scan, bands)``
        and ``wavelengths_nm`` a float array of ``bands`` values from 400 to 1000 nm.
    """
    import numpy as np

    wl = np.linspace(400.0, 1000.0, bands)
    y, x = np.mgrid[0:slit, 0:scan]
    spatial = 0.5 + 0.5 * np.sin(x / 9.0) * np.cos(y / 7.0)
    spectral = np.exp(-((wl - 650.0) / 150.0) ** 2)
    cube = 200.0 + 3000.0 * spatial[:, :, None] * spectral[None, None, :]
    return cube.astype(np.uint16), wl


def build_window(path: str | None = None):
    """Create the viewer window.

    Args:
        path: ENVI ``.hdr`` path, or ``None`` for a synthetic cube.

    Returns:
        CubeViewerWindow: Not yet shown.  Requires a ``QApplication``.  For a file the
        window gets it as ``source_path``: areas are named ``'<file stem>-mean-<k>'``,
        the spectra export defaults to ``<stem>_spectra.csv`` next to it and the XLSX
        metadata records it (as in the main application).
    """
    from modules.cube_viewer import CubeViewerWindow

    if path:
        from modules.cube_io import load_envi_cube_for_viewer

        cube, wavelengths = load_envi_cube_for_viewer(path)
        return CubeViewerWindow(cube, wavelengths, source_path=path)
    cube, wavelengths = synthetic_cube()
    return CubeViewerWindow(cube, wavelengths)


def main(argv=None) -> int:
    """Parse the command line, show the viewer and run the Qt event loop.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", help="ENVI header (.hdr); omit for a synthetic cube")
    args = parser.parse_args(argv)
    from modules._support import launch

    return launch(lambda: build_window(args.file), "Cube Viewer", (1200, 800))


if __name__ == "__main__":
    raise SystemExit(main())
