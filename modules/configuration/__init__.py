"""Persistent application settings: a split ``config/`` directory plus the active wavelength profile.

:class:`ConfigStore` reads and writes the ``config/`` directory (one JSON file per
section, an ``app.json`` manifest published last) and the wavelength files under
``calibration/wavelength/``. It depends only on :mod:`modules.contracts` (the shared
wavelength-profile format) to validate and convert the active profile, and has no Qt dependency.

Data conventions:
    Wavelength slope ``m`` is nm per pixel at the *unbinned* sensor (``base_m``), ``c`` is nm,
    exposure is ms. Paths stored in the manifest are relative to the project root
    (the parent of the ``config/`` directory).

Threading:
    No locking. Call from one thread at a time (writes are atomic per file, but the
    manifest read-modify-write in ``save_wavelength_default`` is not safe against
    concurrent writers). Do not call in the frame loop.

Example:
    >>> import tempfile, pathlib
    >>> from modules.configuration import ConfigStore
    >>> root = pathlib.Path(tempfile.mkdtemp())
    >>> store = ConfigStore(root / "config")
    >>> store.manifest.name
    'app.json'
"""
from .store import ConfigStore

__all__ = ["ConfigStore"]
__requires__ = ("contracts",)
