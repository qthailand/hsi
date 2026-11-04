"""Enter a camera's sensor pixel pitch, and a status-bar chip that asks for it until it is set.

Most camera SDKs do not report the pixel pitch (GenICam and XIMEA backends report ``1.0``), yet
real-size measurements and scan-speed planning need it. The user enters it once per camera; it
is stored in the camera profile (:meth:`modules.contracts.CameraProfile.set_pixel_pitch`).

* :class:`PixelPitchDialog` - shows which camera the value is for (backend, model, serial),
  pre-fills the stored value or the SDK's, and saves on OK. A value the SDK reports is only a
  suggestion: the user still confirms it.
* :class:`PixelPitchStatus` - a small clickable chip for a status bar. It is visible while the
  open camera's profile has no pitch (orange), while the stored pitch needs a second look (orange:
  entered for another camera, or more than 1 % from the SDK's value) and while the profile cannot
  be told apart from other cameras' (red, the camera reports neither model nor serial); clicking
  it opens the dialog. It never blocks anything.

Data conventions:
    Pixel pitch in micrometres at bin 1 (the sensor's own pixel).

Threading:
    GUI thread only. Saving writes ``profile.json`` (small, synchronous).

Example:
    >>> # chip = PixelPitchStatus()
    >>> # window.statusBar().addPermanentWidget(chip)
    >>> # chip.set_profile(controller.camera_profile, sdk_pitch_um=sdk_pixel_pitch_um(backend.info.pixel_size))
"""
from __future__ import annotations

import logging
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit,
                             QToolButton, QVBoxLayout)

from modules.contracts import PIXEL_PITCH_RANGE_UM, CameraProfile

logger = logging.getLogger(__name__)

#: Chip colours: pitch missing or to re-check (orange), profile shared by unidentifiable cameras (red).
_CHIP_COLORS = {"missing": "#ffa726", "check": "#ffa726", "shared": "#ef5350"}
#: Note pre-filled with the SDK's pitch; cleared once the user enters another value.
_SDK_NOTE = "reported by the camera SDK"


def camera_label(profile: CameraProfile) -> str:
    """``"<model> #<serial> (<backend>)"`` with ``?`` for what the SDK did not report."""
    return f"{profile.model or '?'} #{profile.serial or '?'} ({profile.backend})"


class PixelPitchDialog(QDialog):
    """Ask for the sensor pixel pitch of one camera and save it in its profile.

    Constructible with no arguments (nothing can be saved then); the host normally passes the
    open camera's profile. ``OK`` is enabled once a value is entered.

    Args:
        profile: Camera profile to save into, or ``None``.
        sdk_pitch_um: Pitch the SDK reported (already filtered with
            :func:`modules.contracts.sdk_pixel_pitch_um`), used as the suggestion when the
            profile has none.
        parent: Parent widget.

    Signals:
        pitch_saved(float): Emitted with the saved pitch (µm, bin 1) after a successful save.
    """

    pitch_saved = pyqtSignal(float)

    def __init__(self, profile: Optional[CameraProfile] = None, sdk_pitch_um: Optional[float] = None,
                 parent=None):
        super().__init__(parent)
        self.profile = profile
        self.sdk_pitch_um = sdk_pitch_um
        self.setWindowTitle("Sensor pixel pitch")
        layout = QVBoxLayout(self)

        intro = QLabel("Enter the sensor's pixel pitch from the camera datasheet. It is the size of one "
                       "<b>unbinned</b> pixel; binned pixels are scaled automatically.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.camera_text = QLabel(camera_label(profile) if profile is not None else "no camera")
        self.camera_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Camera:", self.camera_text)
        self.sdk_text = QLabel(f"{sdk_pitch_um:g} µm" if sdk_pitch_um else "not reported")
        form.addRow("SDK reports:", self.sdk_text)
        low, high = PIXEL_PITCH_RANGE_UM
        self.pitch = QDoubleSpinBox()
        self.pitch.setDecimals(3)
        self.pitch.setRange(0.0, high)        # 0 shows "not set"; values below `low` are refused
        self.pitch.setSingleStep(0.01)
        self.pitch.setSuffix(" µm")
        self.pitch.setSpecialValueText("not set")
        form.addRow("Pixel pitch (bin 1):", self.pitch)
        self.note = QLineEdit()
        self.note.setPlaceholderText("source, e.g. datasheet / sensor type")
        form.addRow("Note:", self.note)
        layout.addLayout(form)

        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #ef6c00;")
        layout.addWidget(self.warning)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Save")
        self.buttons.accepted.connect(self.save)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        sensor = profile.sensor if profile is not None else None
        if sensor is not None:
            self.pitch.setValue(sensor.pixel_pitch_um)
            self.note.setText(sensor.note)
        elif sdk_pitch_um:
            self.pitch.setValue(sdk_pitch_um)
            self.note.setText(_SDK_NOTE)
        self.pitch.valueChanged.connect(self._refresh)
        self._refresh()

    def value(self) -> Optional[float]:
        """The entered pitch in µm, or ``None`` when it is not set or out of range."""
        low, high = PIXEL_PITCH_RANGE_UM
        value = self.pitch.value()
        return value if low <= value <= high else None

    def _refresh(self, *_):
        """Enable Save for a valid value and show what deserves a second look."""
        value = self.value()
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(self.profile is not None and value is not None)
        differs_from_sdk = (value is not None and bool(self.sdk_pitch_um)
                            and abs(value - self.sdk_pitch_um) > 0.01 * self.sdk_pitch_um)
        if differs_from_sdk and self.note.text() == _SDK_NOTE:
            self.note.clear()                    # the value no longer comes from the SDK
        notes = []
        if self.pitch.value() and value is None:
            notes.append(f"The pitch must be {PIXEL_PITCH_RANGE_UM[0]:g}-{PIXEL_PITCH_RANGE_UM[1]:g} µm.")
        if self.profile is not None:
            if not self.profile.known:
                notes.append("This camera reports neither model nor serial: the value is shared by every "
                             "such camera of this backend.")
            elif self.profile.serial is None:
                notes.append(f"This camera reports no serial: every {self.profile.model} shares this value.")
            sensor = self.profile.sensor
            if sensor is not None and (sensor.model, sensor.serial) != (self.profile.model, self.profile.serial):
                notes.append(f"The stored value was entered for {sensor.model or '?'} #{sensor.serial or '?'}, "
                             f"not for this camera: check it and save.")
        if differs_from_sdk:
            notes.append(f"Differs from the {self.sdk_pitch_um:g} µm the camera SDK reports.")
        self.warning.setText("\n".join(notes))
        self.warning.setVisible(bool(notes))

    def save(self):
        """Save the value in the profile and close; a failure is shown in the dialog."""
        value = self.value()
        if self.profile is None or value is None:
            return
        try:
            sensor = self.profile.set_pixel_pitch(value, self.note.text())
        except (OSError, ValueError) as exc:
            logger.warning("Could not save the pixel pitch of %s: %s", self.profile.root, exc)
            self.warning.setText(f"Could not save: {exc}")
            self.warning.setVisible(True)
            return
        logger.info("Pixel pitch of %s set to %g µm", camera_label(self.profile), sensor.pixel_pitch_um)
        self.pitch_saved.emit(sensor.pixel_pitch_um)
        self.accept()


class PixelPitchStatus(QToolButton):
    """Status-bar chip: "pixel pitch not set" until the open camera's profile has one.

    Hidden when there is no profile, or when the pitch is set, belongs to this identified camera
    and agrees with the SDK (:meth:`CameraProfile.pitch_warnings`). Clicking it opens
    :class:`PixelPitchDialog`. Constructible with no
    arguments; the host calls :meth:`set_profile` once the camera is open.

    Signals:
        pitch_saved(float): Forwarded from the dialog after the user saved a pitch.
    """

    pitch_saved = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.profile: Optional[CameraProfile] = None
        self.sdk_pitch_um: Optional[float] = None
        self.state = "none"
        self.setAutoRaise(True)
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(self.open_dialog)
        self.hide()

    def set_profile(self, profile: Optional[CameraProfile], sdk_pitch_um: Optional[float] = None):
        """Show the state of ``profile`` (``None``: no camera, the chip hides)."""
        self.profile, self.sdk_pitch_um = profile, sdk_pitch_um
        self.refresh()

    def refresh(self):
        """Re-read the profile and update text, colour and visibility.

        ``state`` becomes ``"none"`` (no profile), ``"missing"`` (no pitch), ``"shared"`` (the
        camera reports neither model nor serial), ``"check"`` (a pitch that needs a second look)
        or ``"ok"``.
        """
        profile = self.profile
        warnings = []
        if profile is None:
            self.state = "none"
        elif not profile.known:
            self.state = "shared"
        elif profile.pixel_pitch_um is None:
            self.state = "missing"
        else:
            warnings = profile.pitch_warnings(self.sdk_pitch_um, sharing=False)
            self.state = "check" if warnings else "ok"
        if self.state in ("none", "ok"):
            self.hide()
            return
        pitch = profile.pixel_pitch_um
        if self.state == "shared":
            text = (f"⚠ Pixel pitch {pitch:g} µm: camera not identified, profile shared"
                    if pitch is not None else "⚠ Pixel pitch not set (camera not identified)")
            tip = ("The camera reports neither model nor serial, so every such camera uses this "
                   "profile. Check the value whenever the camera changes. Click to edit.")
        elif self.state == "check":
            text = f"⚠ Check pixel pitch {pitch:g} µm: {profile.model or profile.serial} (click to review)"
            tip = "\n".join(["The stored pixel pitch needs a second look:"] + [f"• {w}" for w in warnings])
        else:
            text = f"⚠ Pixel pitch not set: {profile.model or profile.serial} (click to enter)"
            tip = ("The sensor pixel pitch of this camera is unknown. Real sizes and scan-speed "
                   "planning need it. Click to enter it from the datasheet.")
        self.setText(text)
        self.setToolTip(tip)
        self.setStyleSheet(f"QToolButton {{ color: {_CHIP_COLORS[self.state]}; font-weight: bold; }}")
        self.show()

    def open_dialog(self):
        """Open :class:`PixelPitchDialog` for the current profile (nothing without one)."""
        if self.profile is None:
            return
        dialog = PixelPitchDialog(self.profile, self.sdk_pitch_um, parent=self.window())
        dialog.pitch_saved.connect(self.pitch_saved)
        dialog.exec_()
        self.refresh()
