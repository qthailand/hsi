"""Compatibility shim: moved to :mod:`modules.cube_viewer.canvas`."""
import importlib as _importlib

_module = _importlib.import_module("modules.cube_viewer.canvas")
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
