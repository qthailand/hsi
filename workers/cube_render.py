"""Compatibility shim: moved to :mod:`modules.cube_preview.render`."""
import importlib as _importlib

_module = _importlib.import_module("modules.cube_preview.render")
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
