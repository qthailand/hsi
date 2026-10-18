"""Compatibility shim: moved to :mod:`modules.cube_viewer.window`."""
import importlib as _importlib

_module = _importlib.import_module("modules.cube_viewer.window")
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
