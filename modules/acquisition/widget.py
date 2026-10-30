"""Compose generic cards into the scan correction status display.

``CorrectionStatusDisplay`` owns three :class:`modules.status_card.StatusCard` widgets
("Dark", "Smile" and "Wavelength") and updates them from the ``correction_status`` dict that
:class:`modules.acquisition.AcquisitionEngine` reports (``CameraPreviewThread`` emits it
as the ``correction_status`` signal).

Data conventions:
    The status dict is ``{'dark': (state, detail), 'smile': (state, detail)}`` plus,
    optionally, ``'wavelength': (state, detail)`` and ``'warnings'`` (a tuple of texts,
    shown as the cards' tooltip), where ``state`` is one of the keys of ``STATE_COLORS``
    and ``detail`` is a short text.

Threading:
    GUI thread only; connect a ``CameraPreviewThread`` signal (queued) to
    :meth:`CorrectionStatusDisplay.update_status`.

Example:
    >>> # display = CorrectionStatusDisplay(parent_widget)
    >>> # layout.addWidget(display.dark); layout.addWidget(display.smile); layout.addWidget(display.wavelength)
    >>> # thread.correction_status.connect(display.update_status)
"""
from PyQt5.QtCore import QObject
from modules.status_card import StatusCard

STATE_COLORS = {'Applied': 'green', 'Missing': 'red', 'Off': 'orange', 'Waiting': 'cyan', 'Capturing': 'cyan'}

class CorrectionStatusDisplay(QObject):
    """Independently reusable Dark, Smile and Wavelength cards.

    A ``QObject`` (not a widget): it creates the cards but the host places them in its own
    layout. Because it is not a ``QWidget`` it is not listed in the package ``__widgets__``.

    Attributes:
        dark (StatusCard): Card for the dark-frame correction.
        smile (StatusCard): Card for the smile-correction model.
        wavelength (StatusCard): Card for the wavelength calibration (stored for this
            camera setting, applied from the calibration window, or an old m/c kept).

    The host must provide: a layout to hold the cards and a source of status dicts (see the
    module docstring). Use from the GUI thread only.
    """
    def __init__(self, parent=None):
        """Create the cards showing "Waiting / Waiting for camera".

        Args:
            parent: Optional parent; it is also passed as the parent widget of the cards.
        """
        super().__init__(parent)
        self.dark = StatusCard('Dark', parent)
        self.dark.set_content('Waiting', 'Waiting for camera')
        self.smile = StatusCard('Smile', parent)
        self.smile.set_content('Waiting', 'Waiting for camera')
        self.wavelength = StatusCard('Wavelength', parent)
        self.wavelength.set_content('Waiting', 'Waiting for camera')
        self._base_tips = {}          # card key -> its own tooltip text (before any warnings)

    def update_status(self, status):
        """Show a new correction status on the cards.

        Args:
            status: ``{'dark': (state, detail), 'smile': (state, detail)}``, optionally with
                ``'wavelength': (state, detail)`` (the card keeps its text without it) and
                ``'warnings'`` (texts set as every card's tooltip). ``state`` must be a key
                of ``STATE_COLORS`` ('Applied', 'Missing', 'Off', 'Waiting', 'Capturing');
                the card colour follows from it.

        Raises:
            KeyError: a key or state is missing from the dict.
        """
        for key in ('dark', 'smile', 'wavelength'):
            if key == 'wavelength' and key not in status:
                continue
            state, detail = status[key]
            card = getattr(self, key)
            card.set_content(state, detail, STATE_COLORS[state])
            self._base_tips[key] = f'{card.title.text()}: {state}\n{detail}'
        # rebuilt from each card's own text every time: warnings never pile up or outlive their cause
        warnings = status.get('warnings') or ()
        text = ('\n\nCalibration warnings:\n' + '\n'.join(f'- {w}' for w in warnings)) if warnings else ''
        for key in ('dark', 'smile', 'wavelength'):
            getattr(self, key).setToolTip(self._base_tips.get(key, '') + text)
