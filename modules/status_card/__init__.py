"""Reusable status card; requires only PyQt5, with no HSI dependencies.

``StatusCard`` and ``PALETTE`` are loaded lazily, so ``import modules.status_card``
does not import PyQt5 until one of those names is accessed.

Example:
    >>> from PyQt5.QtWidgets import QApplication
    >>> app = QApplication.instance() or QApplication([])
    >>> from modules.status_card import StatusCard
    >>> StatusCard("Storage").set_content("42 GB", "Free", color="cyan")
"""

__all__ = ['StatusCard', 'PALETTE']
__requires__ = ()
# StatusCard needs a ``title`` argument, so it is not no-argument constructible.
__widgets__ = ()


def __getattr__(name):
    if name in ('StatusCard', 'PALETTE'):
        from . import widget
        return getattr(widget, name)
    raise AttributeError(name)
