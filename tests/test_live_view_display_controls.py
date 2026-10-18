"""live_view: the display bar shows the camera's bit depth read-only."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_bit_depth_is_shown_with_its_source_and_cannot_be_edited():
    _app()
    from PyQt5.QtWidgets import QAbstractSpinBox, QPushButton
    from modules.live_view import DisplayControls
    controls = DisplayControls(bit_depth=16)
    mappers = []
    controls.mapper_changed.connect(mappers.append)
    assert controls._bit_label.text() == "16 bit"

    controls.set_bit_depth(16, sensor_bits=10)
    assert controls._bit_label.text() == "10→16 bit" and mappers[-1].bit_depth == 16
    controls.set_bit_depth(12)
    assert controls._bit_label.text() == "12 bit" and controls.current_mapper().bit_depth == 12
    assert not controls.findChildren(QAbstractSpinBox) and not controls.findChildren(QPushButton)
