"""Compose generic cards into the scan correction status display.

``CorrectionStatusDisplay`` owns two :class:`modules.status_card.StatusCard` widgets
("Dark" and "Smile") and updates them from the ``correction_status`` dict that
:class:`modules.acquisition.AcquisitionEngine` reports (``CameraPreviewThread`` emits it
as the ``correction_status`` signal).

Data conventions:
    The status dict is ``{'dark': (state, detail), 'smile': (state, detail)}`` where
    ``state`` is one of the keys of ``STATE_COLORS`` and ``detail`` is a short text.

Threading:
    GUI thread only; connect a ``CameraPreviewThread`` signal (queued) to
    :meth:`CorrectionStatusDisplay.update_status`.

Example:
    >>> # display = CorrectionStatusDisplay(parent_widget)
    >>> # layout.addWidget(display.dark); layout.addWidget(display.smile)
    >>> # thread.correction_status.connect(display.update_status)
"""
from PyQt5.QtCore import QObject
from modules.status_card import StatusCard

STATE_COLORS = {'Applied': 'green', 'Missing': 'red', 'Off': 'orange', 'Waiting': 'cyan', 'Capturing': 'cyan'}

class CorrectionStatusDisplay(QObject):
    """Pair of independently reusable Dark and Smile cards.

    A ``QObject`` (not a widget): it creates the cards but the host places them in its own
    layout. Because it is not a ``QWidget`` it is not listed in the package ``__widgets__``.

    Attributes:
        dark (StatusCard): Card for the dark-frame correction.
        smile (StatusCard): Card for the smile-correction model.

    The host must provide: a layout to hold ``dark`` and ``smile`` and a source of
    status dicts (see the module docstring). Use from the GUI thread only.
    """
    def __init__(self, parent=None):
        """Create both cards showing "Waiting / Waiting for camera".

        Args:
            parent: Optional parent; it is also passed as the parent widget of both cards.
        """
        super().__init__(parent)
        self.dark = StatusCard('Dark', parent)
        self.dark.set_content('Waiting', 'Waiting for camera')
        self.smile = StatusCard('Smile', parent)
        self.smile.set_content('Waiting', 'Waiting for camera')

    def update_status(self, status):
        """Show a new correction status on the two cards.

        Args:
            status: ``{'dark': (state, detail), 'smile': (state, detail)}``. ``state``
                must be a key of ``STATE_COLORS`` ('Applied', 'Missing', 'Off',
                'Waiting', 'Capturing'); the card colour follows from it.

        Raises:
            KeyError: a key or state is missing from the dict.
        """
        for key in ('dark', 'smile'):
            state, detail = status[key]
            getattr(self, key).set_content(state, detail, STATE_COLORS[state])
