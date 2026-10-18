"""Motorised stage for pushbroom scanning (XIMC 8SMC1 rotation and linear stages).

* :class:`StageSequence` - Qt-free scan sequence (open, home, scan, return home, close)
  with callbacks; usable from scripts and other GUI toolkits.
* :class:`StageThread` - the same sequence on a ``QThread`` with signals.
* :class:`Stage8SMC1` / :class:`Stage8SMC1Linear` - drivers for the vendor library
  ``libximc`` (imported only when used).

Data conventions:
    Rotation: degrees and degrees per second. Linear: millimetres and
    millimetres/second (``speed_um_s`` in the sequence is micrometres/second).

Threading:
    The sequence runs on one worker thread; ``request_stop`` is callable from any
    thread.

Example:
    >>> from modules.stage import StageSequence
    >>> events = []
    >>> class Dummy:
    ...     _opened = False
    ...     def open(self): self._opened = True
    ...     def close(self): self._opened = False
    ...     def set_speed_dps(self, v): pass
    ...     def set_position(self, *a): pass
    ...     def set_degree(self, v): events.append(('degree', v))
    >>> seq = StageSequence(stage_factory=Dummy, on_before=lambda: events.append('before'),
    ...                     on_after=lambda: events.append('after'))
    >>> seq.degree = 20; seq.run(); seq.completed, events
    (True, ['before', ('degree', 20), 'after'])
"""
from .config import StageConfig
from .sequence import StageSequence

__all__ = ["StageConfig", "StageSequence", "StageThread", "Stage8SMC1", "Stage8SMC1Linear"]
__requires__ = ()
__widgets__ = ()

_LAZY = {
    "StageThread": (".thread", "StageThread"),
    "Stage8SMC1": (".smc1", "Stage8SMC1"),
    "Stage8SMC1Linear": (".smc1", "Stage8SMC1Linear"),
}


def __getattr__(name):
    """Load the Qt wrapper or the vendor-SDK drivers on first access."""
    if name in _LAZY:
        import importlib
        module, attribute = _LAZY[name]
        return getattr(importlib.import_module(module, __name__), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
