"""*Save as Flat-field Profile* and its use by *Save as Radiometric Reference* (Qt part).

The bench scene of ``test_flatfield``: a strip spanning the slit, a Spectralon 40 disc on a few
slit rows, targets near the slit ends. Message boxes are replaced as in the reference dialog tests.
"""
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtWidgets import QApplication  # noqa: E402

from modules.flatfield import FlatFieldStore  # noqa: E402
from modules.radiometric import RadiometricReferenceStore, apply_rowwise_factors  # noqa: E402
from test_flatfield import L, S, SEAM, TABLE, WL, scene  # noqa: E402
from test_radiometric_reference import META as KEY_META  # noqa: E402

META = {**KEY_META, "full_scale": 65520, "started_at": "2026-11-04T10:12:00+07:00",
        "sensor_temperature_c_start": 25.0, "sensor_temperature_c_end": 25.4}


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def prepare(dlg, app):
    dlg.errors, dlg.questions = [], []
    dlg._report_error = lambda title, text: dlg.errors.append(text)
    dlg._confirm = lambda title, text: dlg.questions.append(text) or True
    dlg.inset_spin.setValue(0)
    dlg.show()
    app.processEvents()
    return dlg


def close(dlg, app):
    dlg.close()
    dlg.deleteLater()
    app.processEvents()


def make_flatfield(app, folder, cube, combine=False):
    from modules.flatfield.dialog import FlatFieldDialog
    dlg = prepare(FlatFieldDialog(cube, WL, META, FlatFieldStore(folder)), app)
    dlg._run_combine = (lambda d: d.combine()) if combine else (lambda d: d.reject())   # never modal
    try:
        dlg.seam_spin.setValue(5.0)
        dlg.set_rect(2.6, 6.4, -0.4, L - 0.6)                    # the strip, every slit row
        saved = dlg.create_profile()
        return saved, list(dlg.errors), list(dlg.questions)
    finally:
        close(dlg, app)


def test_flatfield_dialog_stores_a_profile_and_reports_the_joint(app, tmp_path):
    cube = scene()[0]
    saved, errors, questions = make_flatfield(app, tmp_path / "flatfield", cube)
    assert not errors and saved is not None and saved.id.startswith("flat_")
    assert saved.info["seam_rows"] == list(range(SEAM[0], SEAM[1] + 1))
    assert len(questions) == 1 and "29-30" in questions[0]       # asked before interpolating the joint
    assert saved.profile().shape == (L, len(WL)) and saved.coverage == pytest.approx((L - 2) / L)
    assert FlatFieldStore(tmp_path / "flatfield").find(saved.key).profile.id == saved.id
    assert saved.sensor_temperature_c == pytest.approx(25.2)        # mean of start / end


def test_flatfield_dialog_refuses_a_target_that_does_not_span_the_slit(app, tmp_path):
    from modules.flatfield.dialog import FlatFieldDialog
    cube = scene()[0]
    dlg = prepare(FlatFieldDialog(cube, WL, META, FlatFieldStore(tmp_path / "flatfield")), app)
    try:
        assert not dlg.profile_combo.isVisible() and not dlg.activate_box.isVisible()   # no reflectance table here
        assert dlg.problems() == ["No ROI drawn yet"]
        dlg.set_rect(17.6, 25.4, 25.6, 33.4)                     # the disc only
        assert not dlg.create_btn.isEnabled() and "must span the slit" in dlg.problems()[0]
        assert dlg.create_profile() is None and dlg.errors
        assert FlatFieldStore(tmp_path / "flatfield").profiles() == []
    finally:
        close(dlg, app)
    no_store = prepare(FlatFieldDialog(cube, WL, {**META, "dark_corrected": False}, None), app)
    try:
        text = " ".join(no_store.problems())
        assert "dark subtraction was off" in text and "No camera profile is open" in text
    finally:
        close(no_store, app)


def test_reference_dialog_uses_the_flatfield_profile(app, tmp_path):
    from modules.radiometric import RadiometricReferenceDialog
    cube, _strip, _disc, targets = scene()
    flat_store = FlatFieldStore(tmp_path / "flatfield")
    flat, _errors, _questions = make_flatfield(app, flat_store.folder, cube)
    dlg = prepare(RadiometricReferenceDialog(cube, WL, META, RadiometricReferenceStore(tmp_path / "radiometric"),
                                             flatfield_store=flat_store), app)
    try:
        assert dlg.flatfield.id == flat.id and dlg.flatfield_box.isChecked() and dlg.uses_flatfield()
        dlg.set_profile(TABLE)
        dlg.set_rect(17.6, 25.4, 25.6, 33.4)                     # the Spectralon disc: 8 of 60 slit rows
        rows = [i for i in dlg._quality_items() if i.text.startswith("ROI covers")]
        assert rows[0].level == "ok" and rows[0].text.endswith(f"from flat-field {flat.id}")
        reference = dlg.create_profile()
        assert reference is not None and not dlg.errors and not dlg.questions
    finally:
        close(dlg, app)
    assert reference.info["flatfield"] == flat.id and reference.info["window_fit"] == "flatfield"
    assert reference.header_fields()["reflectance_reference_cube"].endswith(f", flat-field {flat.id})")
    np.testing.assert_array_equal(reference.flatfield(), flat.profile())   # kept with the reference
    reflectance = apply_rowwise_factors(cube, reference.factors())
    for mask, value in targets.values():                         # targets near the slit ends
        assert np.allclose(reflectance[mask].mean(axis=0), value, rtol=2e-3)


def test_reference_dialog_warns_when_a_small_roi_has_no_flatfield(app, tmp_path):
    from modules.radiometric import RadiometricReferenceDialog
    cube, _strip, _disc, targets = scene()
    for flat_store, why in ((None, ""), (FlatFieldStore(tmp_path / "empty"), "no flat-field profile")):
        dlg = prepare(RadiometricReferenceDialog(cube, WL, META, RadiometricReferenceStore(tmp_path / "radiometric"),
                                                 flatfield_store=flat_store), app)
        try:
            assert dlg.flatfield is None and not dlg.flatfield_box.isEnabled() and not dlg.uses_flatfield()
            dlg.set_profile(TABLE)
            dlg.set_rect(17.6, 25.4, 25.6, 33.4)
            items = [i for i in dlg._quality_items() if i.text.startswith("ROI covers")]
            assert len(items) == 1 and items[0].level == "warn" and "rows 26-33" in items[0].text
            assert "A flat-field profile" in items[0].text and why in items[0].text
            dlg.set_rect(2.6, 6.4, -0.4, L - 0.6)                # an ROI spanning the slit needs none
            assert all(i.level == "ok" for i in dlg._quality_items() if i.text.startswith("ROI covers"))
        finally:
            close(dlg, app)


def test_a_profile_of_another_bin_is_not_offered(app, tmp_path):
    from modules.radiometric import RadiometricReferenceDialog
    cube = scene()[0]
    flat_store = FlatFieldStore(tmp_path / "flatfield")
    make_flatfield(app, flat_store.folder, cube)
    dlg = prepare(RadiometricReferenceDialog(cube, WL, {**META, "camera_bin": 4},
                                             RadiometricReferenceStore(tmp_path / "radiometric"),
                                             flatfield_store=flat_store), app)
    try:
        assert dlg.flatfield is None and "camera bin" in dlg.flatfield_reason
        assert "camera bin" in dlg.flatfield_box.toolTip()
    finally:
        close(dlg, app)


def test_scan_window_button_and_integration(app, monkeypatch, tmp_path):
    from modules.contracts import CameraProfile
    from windows.integrations import RadiometricIntegration
    import modules.flatfield.dialog as dialog_module
    profile = CameraProfile.for_camera(tmp_path, "poa", "Apollo-M MAX", "CAMP7161")
    controller = NS(camera_profile=profile, last_scan_saturation_ceiling=None, win=None)
    statuses, opened = [], []
    host = NS(scan_controller=controller, parent_widget=None, show_status=statuses.append,
              error=lambda title, text: statuses.append(f"ERROR {text}"))

    class FakeDialog:
        def __init__(self, cube, wavelengths, metadata=None, store=None, saturation_ceiling=None, parent=None):
            opened.append(store)
            self.reference = NS(id="flat_20261104_101200")

        def exec_(self):
            return 1

    monkeypatch.setattr(dialog_module, "FlatFieldDialog", FakeDialog)
    integration = RadiometricIntegration(host)
    saved = integration.save_flatfield(scene()[0], WL, metadata=META)
    assert saved.id == "flat_20261104_101200" and opened[0].folder == profile.flatfield_folder
    assert "flat_20261104_101200 stored" in statuses[0]
    assert RadiometricIntegration.flatfield_store(NS(camera_profile=None)) is None


def test_store_status_reports_the_profile_of_the_selected_bin(app, tmp_path):
    import datetime
    store = FlatFieldStore(tmp_path / "flatfield")
    state, text, profile = store.status(2)
    assert state == "off" and profile is None and "no profile" in text
    saved, _errors, _questions = make_flatfield(app, store.folder, scene()[0])
    now = datetime.datetime.fromisoformat(META["started_at"]) + datetime.timedelta(hours=2)
    state, text, profile = store.status(2, now=now)
    assert state == "on" and profile.id == saved.id
    # On this 60-row bench strip the step test runs with the seam window capped at 15 rows and
    # reads the falloff beside the joint as small steps; the text reports what was stored.
    steps = len(saved.info["steps"])
    assert text == (f"Flat-field: bin 2 - 58/60 rows - response "
                    f"{saved.info['response_min']:.2f}-{saved.info['response_max']:.2f} - 2 joint rows - "
                    + (f"{steps} brightness step{'s' if steps != 1 else ''} (!) - " if steps else "")
                    + "2.0 h old")
    assert "5 days old" in store.status(None, now=now + datetime.timedelta(days=5))[1]
    state, text, profile = store.status(4)
    assert state == "off" and profile is None and text == "Flat-field: no profile for bin 4 (have bin 2)"


def test_scan_window_flatfield_badge_and_view(app, tmp_path):
    import config as app_config
    from modules.contracts import CameraProfile
    from windows.integrations import RadiometricIntegration
    from windows.scan_window import ScanWindow
    win = ScanWindow(app_config.AppConfig())
    dialog = None
    try:
        assert not win.flatfield_card.is_clickable() and not win.save_flatfield_btn.isEnabled()
        cube = np.zeros((4, 5, 3), np.uint16)
        win.set_cube_ready(cube, np.array([400.0, 500.0, 600.0]))
        got = []
        win.flatfield_profile_requested.connect(lambda c, wl: got.append(c))
        win.save_flatfield_btn.click()
        assert got and got[0] is cube

        profile = CameraProfile.for_camera(tmp_path, "poa", "Apollo-M MAX", "CAMP7161")
        controller = NS(camera_profile=profile, win=win, last_scan_saturation_ceiling=None)
        host = NS(scan_controller=controller, parent_widget=None, show_status=lambda text: None,
                  error=lambda title, text: None)
        integration = RadiometricIntegration(host)
        win.selected_bin = lambda: 2
        assert integration.refresh_flatfield_status() is None
        card = win.flatfield_card
        assert card.value.text() == "Off" and "no profile" in card.detail.text()
        assert not card.is_clickable() and not card.hint.isVisible()
        saved, _errors, _questions = make_flatfield(app, profile.flatfield_folder, scene()[0])
        integration.refresh_status(controller)                    # the badge follows the radiometric one
        assert card.value.text() == "Ready" and "58/60 rows" in card.toolTip()
        assert card.is_clickable() and saved.id in card.toolTip() and "Click to view" in card.toolTip()
        assert "No active radiometric reference uses it yet" in card.toolTip()
        assert "Apollo-M MAX" in card.toolTip()
        # a radiometric reference made with the profile and active: the card says Applied
        from modules.radiometric import RadiometricReferenceStore, ReferenceKey
        ref_store = RadiometricReferenceStore(profile.radiometric_folder)
        mask = np.zeros((L, S), bool)
        mask[26:34, 18:26] = True
        reference = ref_store.save(np.ones((L, len(WL)), np.float32), ReferenceKey.from_scan(META, WL, L),
                                   reflectance_profile=TABLE.name, avg_window_size=1, mask=mask,
                                   info={"flatfield": saved.id}, activate=True)
        integration.refresh_flatfield_status()
        assert card.value.text() == "Applied" and reference.id in card.toolTip()
        win.selected_bin = lambda: 4
        integration.refresh_flatfield_status()
        assert card.value.text() == "Off" and "no profile for bin 4" in card.toolTip()
        assert not card.is_clickable()
        win.selected_bin = lambda: 2
        requested = []
        from PyQt5.QtCore import QPoint, Qt
        from PyQt5.QtTest import QTest
        win.status_card_clicked.connect(requested.append)
        integration.refresh_flatfield_status()
        win.show()
        app.processEvents()
        QTest.mouseClick(card, Qt.LeftButton, pos=QPoint(10, 10))          # the card is the button
        QTest.mouseClick(win.correction_status.dark, Qt.LeftButton, pos=QPoint(10, 10))   # its peek: capture / reload
        QTest.mouseClick(win.correction_status.smile, Qt.LeftButton, pos=QPoint(10, 10))
        assert requested == ["flatfield", "dark", "smile"]
        integration.refresh_status(controller)
        assert win.radiometric_card.value.text() == "Applied" and win.radiometric_card.is_clickable()
        view = integration.view_reference()
        assert reference.id in view.windowTitle() and f"flat-field {saved.id}" in view.summary_label.text()
        view.close()
        view.deleteLater()
        dialog = integration.view_flatfield()
        assert dialog is not None and saved.id in dialog.windowTitle() and "2 joint rows" in dialog.summary_label.text()
    finally:
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()
        # also when an assertion failed above: an unsaved scan makes close() ask in a message box,
        # which crashes the offscreen platform (segfault instead of a test failure)
        win.mark_result_saved(cube)
        win.close()
        win.deleteLater()
        app.processEvents()


def test_five_square_status_cards_in_a_grid(app):
    import config as app_config
    from windows.scan_window import ScanWindow
    win = ScanWindow(app_config.AppConfig())
    try:
        win.show()
        app.processEvents()
        # Wavelength first: it is always on (the band grid needs it), then the ones that switch
        assert [c.title.text() for c in win.status_cards] == ["WAVELENGTH", "DARK", "SMILE", "FLAT-FIELD", "RADIOMETRIC",
                                                               "SATURATION"]
        # the settings column draws each card as a Corrections row, in card order
        rows = [win.correction_rows[key] for key in ("wavelength", "dark", "smile", "flatfield", "radiometric",
                                                     "saturation")]
        assert rows == list(win.status_cards) and all(card.is_row() and card.isVisible() for card in rows)
        assert not any(card.detail.isVisible() or card.title.isVisible() for card in rows)   # detail: tooltip
        tops = [card.mapTo(win, card.rect().topLeft()).y() for card in rows]
        assert tops == sorted(tops)
        win.set_radiometric_status("on", "Auto radiometric: on - Apollo-M MAX, bin 2 - scanned 1.2 h ago")
        assert win.radiometric_card.value.text() == "Applied" and "scanned 1.2 h ago" in win.radiometric_card.toolTip()
        assert win.radiometric_card.row_state.text() == "Applied"                       # the row shows it
        win.set_radiometric_status("expired", "Auto radiometric: reference expired - x")
        assert win.radiometric_card.value.text() == "Expired"
        win.set_radiometric_status("off", "Auto radiometric: off (no active reference)")
        assert win.radiometric_card.value.text() == "Off"
        win.correction_status.update_status({"dark": ("Capturing", "x"), "smile": ("Applied", "y"),
                                             "wavelength": ("Missing", "z")})
        app.processEvents()
        assert win.correction_status.dark.row_state.text() == "Capturing"                    # the row shows the state
    finally:
        win.close()
        win.deleteLater()
        app.processEvents()


def _moved_joint(cube):
    """The bench strip with its joint moved from rows 29-30 to rows 44-45."""
    moved = cube.copy()
    moved[SEAM[0]:SEAM[1] + 1] = cube[SEAM[0] - 1][None]
    moved[44:46, 2:8] *= 0.2
    return moved


def test_flatfield_dialog_combines_with_the_current_profile(app, tmp_path):
    from modules.flatfield.dialog import FlatFieldCombineDialog, FlatFieldDialog
    folder = tmp_path / "flatfield"
    first, errors, _ = make_flatfield(app, folder, scene()[0])
    assert first is not None and not errors

    dlg = prepare(FlatFieldDialog(_moved_joint(scene()[0]), WL, META, FlatFieldStore(folder)), app)
    shown = []
    try:
        assert dlg.combine_box.isVisible() and dlg.combine_box.isChecked()
        assert first.id in dlg.combine_box.text()

        def run(combine_dialog):
            assert isinstance(combine_dialog, FlatFieldCombineDialog)
            assert [s.id for s in combine_dialog.scans][0] == first.id and len(combine_dialog.scans) == 2
            shown.append([i.text for i in combine_dialog.qc_items])
            shown.append(combine_dialog.plan)
            assert combine_dialog.next_ax is not None
            combine_dialog.combine()
        dlg._run_combine = run
        dlg.seam_spin.setValue(5.0)
        dlg.set_rect(2.6, 6.4, -0.4, L - 0.6)
        combined = dlg.create_profile()
    finally:
        close(dlg, app)
    store = FlatFieldStore(folder)
    assert combined is not None and len(combined.combined_from) == 2 and combined.combined_from[0] == first.id
    assert store.find(combined.key).profile.id == combined.id
    assert shown and any("radiometric reference again" in t for t in shown[0])
    assert shown[1] is not None and shown[1].get("joints_now") == [44]           # advice drawn for the next scan
    single = store.load(combined.combined_from[1])
    assert single.info["seam_rows"] == [44, 45] and np.isnan(single.measured()[44:46]).all()

    # the next scan combines with every scan behind the combination
    dlg = prepare(FlatFieldDialog(scene()[0], WL, META, store), app)
    try:
        assert dlg.combine_base.id == combined.id and "(2 scans)" in dlg.combine_box.text()
        dlg._run_combine = lambda d: d.reject()                       # "Use ... alone"
        dlg.seam_spin.setValue(5.0)
        dlg.set_rect(2.6, 6.4, -0.4, L - 0.6)
        alone = dlg.create_profile()
    finally:
        close(dlg, app)
    assert alone is not None and not alone.combined_from
    assert store.find(alone.key).profile.id == alone.id               # the newest, on its own


def test_flatfield_view_shows_a_combined_profile(app, tmp_path):
    from modules.flatfield.dialog import FlatFieldViewDialog
    folder = tmp_path / "flatfield"
    first = make_flatfield(app, folder, scene()[0])[0]
    combined = make_flatfield(app, folder, _moved_joint(scene()[0]), combine=True)[0]
    assert combined.combined_from[0] == first.id
    view = FlatFieldViewDialog(combined)
    try:
        assert "Combined of 2 scans" in view.summary_label.text()
    finally:
        view.deleteLater()
