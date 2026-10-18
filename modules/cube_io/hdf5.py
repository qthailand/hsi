"""Standalone HDF5 spectrogram writer with the original compression/chunking.

Data conventions:
    ``data`` is ``(frames, spatial, spectral)`` of any numeric dtype.  The file
    holds one dataset ``"spectrogram"`` of the same shape/dtype (gzip level 4,
    chunks of one frame) and root attributes ``shape``, ``dtype``, ``format_version`` (1),
    ``axes`` (``"frame,slit,spectral"``) and, when given, the scan metadata (camera bin,
    exposure, gain, ``raw_inverted``, m/c ... see ``modules.acquisition.metadata``).

Threading:
    Blocking call; run it in a worker thread.  ``progress`` is invoked on the
    calling thread.  Requires ``h5py`` (imported lazily).

Example:
    >>> import numpy as np, os, tempfile
    >>> from modules.cube_io.hdf5 import save_hdf5     # doctest: +SKIP
    >>> path = save_hdf5(os.path.join(tempfile.mkdtemp(), "s.h5"),
    ...                  np.zeros((4, 3, 5), np.uint16))   # doctest: +SKIP
"""

def save_hdf5(path, data, progress=None, metadata=None):
    """Write a spectrogram to an HDF5 file.

    Args:
        path: Output file path (overwritten).
        data: ``(frames, spatial, spectral)`` array (a ``numpy.memmap`` is fine;
            it is read frame by frame).
        progress: Optional callable receiving an int percentage (0-100); called
            roughly every ``max(1, frames // 20)`` frames and finally with 100.
        metadata: Optional flat dict stored as root attributes (``str``, numbers, ``bool``,
            sequences of numbers). ``None`` values are skipped; ``shape``, ``dtype``,
            ``format_version`` and ``axes`` are reserved and cannot be overridden.

    Returns:
        The ``path`` that was written.

    Raises:
        ImportError: ``h5py`` is not installed.
        OSError: the file cannot be created or written.
    """
    import h5py
    progress = progress or (lambda value: None)
    n_frames = data.shape[0]
    chunk_size = max(1, n_frames // 20)  # 20 progress updates

    with h5py.File(path, "w") as f:
        ds = f.create_dataset(
            "spectrogram",
            shape=data.shape,
            dtype=data.dtype,
            compression="gzip",
            compression_opts=4,
            chunks=(1, *data.shape[1:]),
        )

        for i in range(n_frames):
            ds[i] = data[i]
            if (i + 1) % chunk_size == 0:
                pct = int((i + 1) / n_frames * 100)
                progress(pct)

        for key, value in (metadata or {}).items():
            if value is None or key in ("shape", "dtype", "format_version", "axes"):
                continue
            f.attrs[key] = value
        f.attrs["shape"] = data.shape
        f.attrs["dtype"] = str(data.dtype)
        f.attrs["format_version"] = 1
        f.attrs["axes"] = "frame,slit,spectral"

    progress(100)
    return path
