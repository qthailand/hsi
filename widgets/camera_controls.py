"""Compatibility shim: moved to :mod:`modules.camera.widget`."""
import importlib as _importlib

_module = _importlib.import_module("modules.camera.widget")
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
