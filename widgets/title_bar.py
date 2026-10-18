"""Compatibility shim: moved to :mod:`modules.ui_kit.title_bar`."""
import importlib as _importlib

_module = _importlib.import_module("modules.ui_kit.title_bar")
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
