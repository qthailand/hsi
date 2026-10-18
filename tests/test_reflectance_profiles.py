"""Reference reflectance tables shipped in modules/radiometric/reflectance_profiles."""
from modules.radiometric import REFLECTANCE_PROFILE_DIR, list_reflectance_profiles


def test_the_shipped_tables_are_listed():
    names = [p.name for p in list_reflectance_profiles()]
    assert REFLECTANCE_PROFILE_DIR.name == 'reflectance_profiles'
    assert {'Spectralon_ref-005.xlsx', 'Spectralon_ref-020.xlsx', 'Spectralon_ref-040.xlsx',
            'PVC_reflectance.xlsx'} <= set(names)
    assert names == sorted(names, key=str.lower)


def test_only_excel_tables_and_no_lock_files(tmp_path):
    for name in ('b.xlsx', 'A.xls', '~$b.xlsx', 'notes.txt'):
        (tmp_path / name).write_bytes(b'')
    (tmp_path / 'sub.xlsx').mkdir()
    assert [p.name for p in list_reflectance_profiles(tmp_path)] == ['A.xls', 'b.xlsx']
    assert list_reflectance_profiles(tmp_path / 'missing') == []


def test_window_offers_the_tables_and_a_browsed_one(tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import modules.radiometric.widget as widget
    window = widget.CalibrationWindow()
    combo = window.ref_combo
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert 'Spectralon_ref-020' in labels and combo.currentData()
    other = tmp_path / 'my_target.xlsx'
    other.write_bytes(b'')
    monkeypatch.setattr(widget.QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(other), '')))
    window.event_browse_reference()
    assert combo.currentData() == str(other) and combo.currentText() == 'my_target.xlsx'
    window.close()
    app.processEvents()
