"""Compatibility shim: moved to :mod:`modules.live_view.image_canvas`."""
import importlib as _importlib

_module = _importlib.import_module("modules.live_view.image_canvas")
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
