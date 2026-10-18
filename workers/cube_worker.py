"""Compatibility shim: moved to :mod:`modules.cube_io.cube_worker`."""
import importlib as _importlib

_module = _importlib.import_module("modules.cube_io.cube_worker")
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
