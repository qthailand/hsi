"""Compatibility shim: moved to :mod:`modules.live_view.display_controls`."""
import importlib as _importlib

_module = _importlib.import_module("modules.live_view.display_controls")
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
