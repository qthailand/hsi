"""Infrastructure shared by every ``modules.*`` package (not a feature module).

Currently provides :func:`launch`, the one way a module's ``__main__`` opens its
widget as a stand-alone window. Packages may import ``modules._support`` without
declaring it in ``__requires__``.
"""
from .launch import launch

__all__ = ["launch"]
