"""Profile peeks: the builders, the popup, and the status-card click that opens them."""
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest

from modules.profile_peek import (PeekData, dark_peek, flatfield_peek, reference_peek, smile_peek,
                                  text_peek)


@pytest.fixture(scope="module")
def app():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# ---------------------------------------------------------------- builders (Qt-free)
def test_dark_peek_plots_the_column_profile_against_wavelength():
    mean = np.tile(np.arange(6, dtype=np.float64), (4, 1)) + 100
    dark = {"mean": mean, "exposure": 0.1, "gain": 10, "bin": 1, "temperature": 20.04, "n_frames": 16,
            "captured_at": "2026-11-25T09:00:00", "hot_pixel_map": np.eye(4, 6, dtype=bool)}
    peek = dark_peek(dark, wavelength=(2.0, 400.0))
    assert peek.title == "Dark" and peek.color == "green" and not peek.warning
    assert np.allclose(peek.curve.x, 400 + 2 * np.arange(6)) and peek.curve.x_label == "wavelength (nm)"
    assert np.allclose(peek.curve.y, mean[0])
    assert peek.lines[0] == "0.1 ms · gain 10 · bin 1 · 20.0 °C"
    assert peek.lines[1] == "16 frames · 4 hot pixels" and "2026-11-25" in peek.lines[2]
    assert peek.open_label == "View ▸"


def test_dark_peek_warns_about_a_light_leak_in_the_units_of_the_axis():
    dark = {"mean": np.zeros((3, 10))}
    leak = {"columns": [4, 7], "peak_counts": 30.0, "ratio": 12.0}
    assert dark_peek(dark, wavelength=(10.0, 500.0), leak=leak).warning == "Light leak: lines at 540, 570 nm"
    peek = dark_peek(dark, leak=leak)
    assert peek.color == "orange" and peek.warning == "Light leak: lines at 4, 7"
    assert peek.curve.x_label == "sensor column"
    with pytest.raises(ValueError):
        dark_peek({"mean": np.zeros(5)})


def test_dark_peek_copies_the_array():
    mean = np.ones((2, 3))
    peek = dark_peek({"mean": mean})
    mean[:] = 9
    assert np.allclose(peek.curve.y, 1.0)


def test_smile_peek_is_the_shift_at_the_centre_column():
    rows, cols = 50, 21
    r = np.arange(rows)[:, None]
    bow = 0.002 * (r - 24.5) ** 2                     # a line bowed along the slit
    map_x = np.arange(cols)[None, :] + bow + np.zeros((rows, cols))
    cal = NS(map_x=map_x, has_keystone=True, created="2026-11-18", rms_before=0.412, rms_after=0.066)
    peek = smile_peek(cal, "C:/models/smile_bin1.npz")
    assert peek.title == "Smile + keystone" and peek.color == "green"
    assert np.allclose(peek.curve.y, bow[:, 0]) and peek.curve.x.size == rows
    assert peek.curve.reference == 0.0 and peek.curve.x_label == "slit row"
    assert peek.lines == ("smile_bin1.npz · 2026-11-18", "straightness 0.412 → 0.066 px")
    assert smile_peek(NS(map_x=map_x)).title == "Smile"
    assert smile_peek(NS(map_x=map_x)).lines == ()


def _flat_profile(steps=()):
    table = np.outer(np.linspace(0.8, 1.2, 30), np.ones(8))
    table[5, 3] = np.nan
    return NS(profile=lambda: table, key=NS(label=lambda: "POA · bin 2"), acquired_at="2026-11-20T10:00",
              coverage=0.97, info={"steps": list(steps)})


def test_flatfield_peek_is_the_response_along_the_slit():
    peek = flatfield_peek(_flat_profile())
    assert peek.title == "Flat-field" and peek.color == "cyan" and not peek.warning
    assert np.allclose(peek.curve.y, np.linspace(0.8, 1.2, 30)) and peek.curve.reference == 1.0
    assert peek.lines == ("POA · bin 2", "scanned 2026-11-20T10:00 · 97% of the slit")
    stepped = flatfield_peek(_flat_profile(steps=[(12, 0.04)]))
    assert stepped.color == "orange" and stepped.warning == "Brightness steps at 1 row"


def test_reference_peek_is_the_factor_per_band_on_a_log_scale():
    wl = np.array([400.0, 500.0, 600.0])
    factors = np.array([[1e-4, 2e-4, 4e-4], [1e-4, 2e-4, np.nan], [3e-4, 2e-4, 4e-4]])
    ref = NS(factors=lambda: factors, key=NS(wavelengths=wl, label=lambda: "bin 1 · 3 bands"),
             acquired_at="2026-11-24T08:00", age_hours=lambda: 26.04, info={"flatfield": "ff-0007"})
    peek = reference_peek(ref)
    assert peek.title == "Radiometric" and peek.curve.log_y
    assert np.allclose(peek.curve.y, [1e-4, 2e-4, 4e-4]) and np.allclose(peek.curve.x, wl)
    assert peek.lines == ("bin 1 · 3 bands", "scanned 2026-11-24T08:00 · 26.0 h old", "flat-field ff-0007")
    ref.info = {}
    assert reference_peek(ref).lines[-1] == "no flat-field profile"


def test_text_peek_keeps_the_value_and_the_non_empty_lines():
    peek = text_peek("Wavelength", "Stored", ["bin 1", "", "  residual 0.09 nm "], color="nope")
    assert peek.curve is None and peek.color == "cyan"
    assert peek.lines == ("Stored", "bin 1", "residual 0.09 nm")


# ---------------------------------------------------------------- the popup
def test_peek_widget_shows_the_curve_or_hides_the_plot(app):
    from modules.profile_peek import ProfilePeek
    peek = ProfilePeek()
    try:
        peek.set_data(flatfield_peek(_flat_profile(steps=[(3, 0.1)])))
        assert peek.title_label.text() == "Flat-field" and not peek.plot.isHidden()
        assert peek.warning_label.text().endswith("Brightness steps at 1 row") and not peek.warning_label.isHidden()
        assert peek.open_btn.text() == "View ▸"
        x, y = peek._curve.getData()
        assert len(x) == 30
        peek.set_data(text_peek("Wavelength", "Stored", ["bin 1"]))
        assert peek.plot.isHidden() and peek.warning_label.isHidden()
        assert peek.lines_label.text() == "Stored\nbin 1" and peek.open_btn.text() == "Open ▸"
    finally:
        peek.deleteLater()


def test_peek_opens_below_its_anchor_and_open_closes_it(app):
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QPushButton
    from modules.profile_peek import ProfilePeek
    anchor = QPushButton("card")
    anchor.setGeometry(200, 200, 92, 92)
    anchor.show()
    app.processEvents()
    peek = ProfilePeek()
    try:
        opened = []
        peek.openRequested.connect(lambda: opened.append(True))
        peek.set_data(dark_peek({"mean": np.zeros((2, 4))}))
        peek.popup(anchor)
        app.processEvents()
        assert peek.isVisible() and peek.windowFlags() & Qt.Popup
        bottom = anchor.mapToGlobal(anchor.rect().bottomLeft()).y()
        assert peek.y() > bottom or peek.y() + peek.height() < anchor.mapToGlobal(anchor.rect().topLeft()).y()
        peek.open_btn.click()
        assert opened == [True] and not peek.isVisible()
        peek.popup(anchor)
        app.processEvents()
        QTest.keyClick(peek, Qt.Key_Return)
        assert opened == [True, True] and not peek.isVisible()
    finally:
        peek.deleteLater()
        anchor.deleteLater()


# ---------------------------------------------------------------- wiring
def test_card_click_shows_the_peek_and_open_opens_the_full_view(app):
    from PyQt5.QtWidgets import QWidget
    from windows.main_window import MainWindow
    card = QWidget()
    card.setGeometry(100, 100, 92, 92)
    card.show()
    host = QWidget()
    opened = []
    data = {"dark": dark_peek({"mean": np.zeros((2, 4))}), "smile": None}
    host.scan_controller = NS(win=NS(status_card=lambda key: card))
    host._profile_peek_data = lambda key: data[key]
    host._open_profile = opened.append
    try:
        MainWindow._on_status_card_clicked(host, "dark")
        app.processEvents()
        peek = host._profile_peek
        assert peek.isVisible() and peek.title_label.text() == "Dark" and opened == []
        peek.open_btn.click()
        assert opened == ["dark"]
        MainWindow._on_status_card_clicked(host, "smile")     # nothing to peek at: open directly
        assert opened == ["dark", "smile"]
        host._profile_peek_data = lambda key: 1 / 0             # a broken peek never blocks the view
        MainWindow._on_status_card_clicked(host, "dark")
        assert opened == ["dark", "smile", "dark"]
    finally:
        host.deleteLater()
        card.deleteLater()


def test_scan_controller_peeks_at_the_dark_and_the_smile_model():
    from controllers.scan_controller import ScanController
    dark = {"mean": np.zeros((3, 5)), "bin": 2, "path": "d.npz"}
    fake = NS(dark_reference=dark, win=NS(selected_bin=lambda: 1), config=NS(m=1.5, c=390.0),
              _stored_dark_leak=lambda d: {"columns": [2]})
    fake._dark_wavelength = types.MethodType(ScanController._dark_wavelength, fake)
    peek = ScanController.dark_peek_data(fake)
    assert np.allclose(peek.curve.x, 390 + 3.0 * np.arange(5))             # m at the dark's bin 2
    assert peek.warning == "Light leak: lines at 396 nm"
    fake.dark_reference = None
    assert ScanController.dark_peek_data(fake) is None
    model = NS(map_x=np.tile(np.arange(4.0), (6, 1)))
    fake.preview_thread = NS(_smile=NS(calibration=model, path="m.npz"))
    assert ScanController.smile_peek_data(fake).lines == ("m.npz",)
    fake.preview_thread = NS(_smile=None)
    assert ScanController.smile_peek_data(fake) is None


def test_radiometric_integration_peeks_at_the_profile_and_the_newest_reference():
    from windows.integrations import RadiometricIntegration
    integration = RadiometricIntegration(NS(scan_controller=NS()))
    integration.refresh_flatfield_status = lambda: _flat_profile()
    assert integration.flatfield_peek_data().title == "Flat-field"
    integration.refresh_flatfield_status = lambda: None
    assert integration.flatfield_peek_data() is None

    def ref(rid, acquired):
        return NS(id=rid, acquired=acquired, factors=lambda: np.ones((2, 2)),
                  key=NS(wavelengths=np.array([500.0, 600.0]), label=lambda: rid),
                  acquired_at=acquired, age_hours=lambda: 1.0, info={})
    store = NS(active=lambda: [ref("old", "2026-11-01"), ref("new", "2026-11-24")])
    integration.reference_store = lambda controller: store
    assert integration.reference_peek_data().lines[0] == "new"
    store.active = lambda: []
    assert integration.reference_peek_data() is None


def test_scan_window_finds_each_status_card(app):
    import config as app_config
    from windows.scan_window import ScanWindow
    win = ScanWindow(app_config.AppConfig())
    try:
        assert win.status_card("dark") is win.correction_status.dark
        assert win.status_card("flatfield") is win.flatfield_card
        assert win.status_card("nope") is None
    finally:
        win.close()
        win.deleteLater()
        app.processEvents()


def test_peek_data_is_frozen():
    peek = PeekData("x")
    with pytest.raises(Exception):
        peek.title = "y"


def test_peek_action_buttons_close_it_and_emit_their_key(app):
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QPushButton
    from modules.profile_peek import ProfilePeek
    anchor = QPushButton("card")
    anchor.show()
    peek = ProfilePeek()
    try:
        got, opened = [], []
        peek.actionRequested.connect(got.append)
        peek.openRequested.connect(lambda: opened.append(True))
        peek.set_data(text_peek("Dark", "No dark frame", ["for 6 ms"], open_label="",
                                actions=(("capture", "Capture"), ("reload", "Reload"))))
        assert peek.open_btn.isHidden() and list(peek.action_buttons) == ["capture", "reload"]
        peek.popup(anchor)
        app.processEvents()
        peek.action_buttons["reload"].click()
        assert got == ["reload"] and not peek.isVisible()
        peek.popup(anchor)
        app.processEvents()
        QTest.keyClick(peek, Qt.Key_Return)                     # no Open: Enter = the first action
        assert got == ["reload", "capture"] and opened == []
        peek.set_data(dark_peek({"mean": np.zeros((2, 4))}))    # no actions given: none shown
        assert peek.action_buttons == {} and not peek.open_btn.isHidden()
    finally:
        peek.deleteLater()
        anchor.deleteLater()


def test_enter_presses_the_focused_button_and_the_popup_sits_by_its_real_width(app):
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QPushButton
    from modules.profile_peek import ProfilePeek
    from modules.profile_peek.widget import PEEK_WIDTH
    anchor = QPushButton("card")
    anchor.show()
    peek = ProfilePeek()
    try:
        got = []
        peek.actionRequested.connect(got.append)
        peek.set_data(text_peek("Dark", "No dark frame", [], open_label="",
                                actions=(("capture", "Capture"), ("reload", "Reload"))))
        peek.popup(anchor)
        app.processEvents()
        peek.action_buttons["reload"].setFocus()
        QTest.keyClick(peek, Qt.Key_Return)          # Enter on Reload is Reload, not Capture
        assert got == ["reload"]
        # placed by its real (fixed) width, not by the plot's 600 px size hint
        peek.set_data(dark_peek({"mean": np.ones((4, 8))}))
        screen = anchor.screen().availableGeometry()
        anchor.move(screen.right() - anchor.width() - 2, screen.top() + 10)
        peek.popup(anchor)
        app.processEvents()
        assert peek.width() == PEEK_WIDTH
        assert peek.x() == min(anchor.mapToGlobal(anchor.rect().topLeft()).x(), screen.right() - PEEK_WIDTH)
    finally:
        peek.deleteLater()
        anchor.deleteLater()


def test_a_narrow_log_axis_still_gets_two_labels():
    from modules.profile_peek.widget import _decade_ticks

    def inside(y):
        ticks = _decade_ticks(y)
        lo, hi = np.log10(min(y)), np.log10(max(y))
        return [label for value, label in ticks[0] if lo <= value <= hi]

    assert inside([1e-4, 1e-1]) == ["0.0001", "0.001", "0.01", "0.1"]     # decades only
    assert inside([2e-3, 5e-3]) == ["0.002", "0.005"]                     # 1-2-5
    assert len(inside([3e-3, 4.5e-3])) >= 2                               # every 1..9
    assert _decade_ticks([3e-3, 3e-3]) is None and _decade_ticks([0, -1]) is None


def test_text_peek_says_how_many_lines_it_left_out():
    peek = text_peek("Wavelength", "Applied", [f"warning {i}" for i in range(8)])
    assert len(peek.lines) == 6 and peek.lines[-1] == "… 4 more"


# ---------------------------------------------------------------- saturation level
def _tree_hash(root) -> dict:
    import hashlib
    from pathlib import Path
    root = Path(root)
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _sat_record(bin_=2, status="verified", level=65504, steps=True):
    from modules.contracts import PlateauValue, SaturationIdentity, SaturationProfileRecord
    ident = SaturationIdentity("poa", "Apollo-M MAX", "CAMP7161", "RAW16", 16, 12, bin_)
    plateaus = (PlateauValue(65504, "clipping", "clipping plateau: absent at 125 ms, appears at 250 ms"),
                PlateauValue(65520, "unconfirmed", "seen at one exposure only"))
    stored = ({"gain": 1.0, "exposure_ms": 250.0, "top_values": [[65504, 900.0], [65488, 30.0]],
               "plateaus": [[65504, 900.0]]},
              {"gain": 1.0, "exposure_ms": 1000.0, "top_values": [[65504, 50000.0], [65488, 80.0], [65472, 70.0]],
               "plateaus": [[65504, 50000.0]]}) if steps else ()
    verified = status == "verified"
    return SaturationProfileRecord(ident, status, level if verified else None, 65520, plateaus=plateaus,
                                   gain_range=(1.0, 200.0) if verified else None, gains_tested=(1.0, 200.0),
                                   steps=stored, reasons=() if verified else ("levels differ between gains",),
                                   measured_at="2026-11-27T13:21:13")


SAT_CAMERA = {"model": "Apollo-M MAX", "serial": "CAMP7161", "pixel_format": "RAW16", "delivered_bits": 16,
              "sensor_bits": 12, "bin": 2, "gain": 1.0, "nominal_full_scale": 65520}
SAT_BINS = [{"bin": 1, "status": "verified", "level": 65504}, {"bin": 2, "status": "verified", "level": 65504},
            {"bin": 3, "status": "inconclusive", "level": None}]


def test_saturation_peek_shows_the_active_profile_and_every_bin():
    from modules.profile_peek import saturation_peek
    peek = saturation_peek(SAT_CAMERA, _sat_record(), SAT_BINS, "verified")
    assert peek.title == "Saturation level" and peek.color == "green" and peek.warning == ""
    assert peek.lines[0] == "Apollo-M MAX · CAMP7161"                              # camera identity
    assert peek.lines[1] == "RAW16 · 12→16 bit · nominal 65520"                     # format, bits, nominal
    assert peek.lines[2] == "bin 2: 65504 · near ≥ 64194 · gain 1–200"              # level, near, gain range
    assert peek.lines[3] == "measured 2026-11-27 13:21"
    assert peek.lines[4] == "bins 1 ✓65504  ▶2 ✓65504  3 ?"                        # every bin, active marked
    assert peek.lines[5].startswith("plateau 65504: absent at 125 ms, appears at 250 ms; also 65520")
    assert peek.lines[6] == "histogram 1000 ms, gain 1 (top 3 values kept)"         # the step with most plateau
    assert peek.curve.log_y and list(peek.curve.x) == [65472, 65488, 65504]        # stored upper histogram
    assert list(peek.curve.y) == [70.0, 80.0, 50000.0] and peek.open_label == "Open ▸" and not peek.actions
    assert saturation_peek(SAT_CAMERA, _sat_record(steps=False), SAT_BINS, "verified").curve is None


def test_saturation_peek_says_why_the_level_is_not_used():
    from modules.profile_peek import saturation_peek
    missing = saturation_peek(dict(SAT_CAMERA, bin=4), None, SAT_BINS, "missing", "no saturation profile for bin 4")
    assert missing.color == "red" and missing.curve is None and "bin 4: missing" in missing.lines
    assert missing.warning == "Bin 4 not measured: raw clipping is not checked"
    inconclusive = saturation_peek(SAT_CAMERA, _sat_record(status="inconclusive"), SAT_BINS, "inconclusive")
    assert inconclusive.color == "orange" and "levels differ between gains" in inconclusive.warning
    assert inconclusive.lines[2] == "bin 2: inconclusive · tested gain 1–200"
    incompatible = saturation_peek(SAT_CAMERA, _sat_record(), SAT_BINS, "incompatible",
                                   "serial number 'OTHER' != 'CAMP7161'")
    assert incompatible.color == "orange" and "serial number" in incompatible.warning
    outside = saturation_peek(dict(SAT_CAMERA, gain=300.0), _sat_record(), SAT_BINS, "gain_outside",
                              "gain 300.0 outside the verified range 1-200")
    assert outside.warning == "Gain 300 outside the verified gain 1–200: not checked" and outside.color == "orange"


class _NoCamera:
    """A backend at bin 2 that fails the test if anything commands it (the peek is read-only)."""

    current_bin, gain, exposure_ms = 2, 1.0, 500.0
    info = NS(bit_depth=16, image_format="POAImgFormat.POA_RAW16", height=552, width=804)

    def sensor_bit_depth(self):
        return 12

    def model_name(self):
        return "Apollo-M MAX"

    def serial_number(self):
        return "CAMP7161"

    def __getattr__(self, name):                       # set_params, set_bin, start_stream, read_frame_raw, ...
        raise AssertionError(f"the peek called the camera: {name}")


def test_scan_controller_saturation_peek_reads_the_profiles_only(tmp_path):
    from controllers.scan_controller import ScanController
    from modules.contracts import CameraProfile
    from modules.saturation_profile import SaturationProfileStore
    profile = CameraProfile.for_camera(tmp_path, "poa", "Apollo-M MAX", "CAMP7161")
    store = SaturationProfileStore(profile.saturation_folder)
    for record in (_sat_record(1), _sat_record(2), _sat_record(3, status="inconclusive")):
        store.save(record)
    (profile.saturation_folder / "saturation_Mono12_d16s12_bin1.json").write_text("{}")    # another format: ignored
    before = _tree_hash(tmp_path)
    fake = ScanController.__new__(ScanController)
    fake.camera_profile, fake.backend, fake.config = profile, _NoCamera(), NS(camera_backend="poa")
    peek = ScanController.saturation_peek_data(fake)
    assert peek.color == "green" and peek.lines[4] == "bins 1 ✓65504  ▶2 ✓65504  3 ?"
    assert _tree_hash(tmp_path) == before                                   # nothing written
    fake.backend.current_bin = 4
    assert ScanController.saturation_peek_data(fake).color == "red"         # bin 4 not measured
    fake.camera_profile = None
    assert ScanController.saturation_peek_data(fake) is None                # no profile: open the window


def test_a_long_peek_text_is_not_cut_off(app):
    """Regression: the popup is a top-level widget of fixed width, whose layout did not ask the
    word-wrapped lines for their height: a long text (the saturation peek) lost its last lines."""
    from modules.profile_peek import ProfilePeek, saturation_peek
    peek = ProfilePeek()
    try:
        peek.set_data(saturation_peek(SAT_CAMERA, _sat_record(), SAT_BINS, "gain_outside",
                                      "gain 300 outside the verified range"))
        peek.popup()
        app.processEvents()
        width = peek.lines_label.width()
        assert peek.lines_label.height() >= peek.lines_label.heightForWidth(width)
        assert peek.warning_label.height() >= peek.warning_label.heightForWidth(peek.warning_label.width())
    finally:
        peek.close()
        peek.deleteLater()


def test_the_saturation_row_peeks_and_open_opens_the_measurement_window(app):
    from PyQt5.QtWidgets import QWidget
    from modules.profile_peek import saturation_peek
    from windows.main_window import MainWindow
    card = QWidget()
    card.setGeometry(100, 100, 92, 92)
    card.show()
    host = QWidget()
    opened = []
    peek_data = saturation_peek(SAT_CAMERA, _sat_record(), SAT_BINS, "verified")
    host.scan_controller = NS(win=NS(status_card=lambda key: card), saturation_peek_data=lambda: peek_data,
                              open_saturation_measurement=lambda: opened.append("window"))
    host._open_profile = types.MethodType(MainWindow._open_profile, host)
    host.integrations = []
    try:
        assert MainWindow._profile_peek_data(host, "saturation") is peek_data
        host._profile_peek_data = types.MethodType(MainWindow._profile_peek_data, host)
        MainWindow._on_status_card_clicked(host, "saturation")
        app.processEvents()
        peek = host._profile_peek
        assert peek.isVisible() and peek.title_label.text() == "Saturation level" and opened == []
        assert peek.plot.isVisible() and "▶2" in peek.lines_label.text()
        peek.open_btn.click()
        assert opened == ["window"]                                         # opens it; measures nothing
    finally:
        host.deleteLater()
        card.deleteLater()
