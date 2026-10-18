"""Compatibility shim: moved to :mod:`modules.live_view.mapping`."""
import importlib as _importlib

_module = _importlib.import_module("modules.live_view.mapping")
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
