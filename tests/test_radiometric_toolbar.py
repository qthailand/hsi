"""The calibration window must build with older matplotlib, whose NavigationToolbar2QT requires ``parent``."""
import ast
from pathlib import Path


def test_every_navigation_toolbar_gets_a_parent():
    source = (Path(__file__).resolve().parents[1] / 'modules' / 'radiometric' / 'widget.py').read_text(encoding='utf-8')
    calls = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'NavigationToolbar2QT']
    assert calls
    for call in calls:
        assert len(call.args) + len(call.keywords) >= 2, f'line {call.lineno}: pass the parent widget'


def test_calibration_window_builds_with_a_parent_requiring_toolbar(monkeypatch):
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import modules.radiometric.widget as widget
    real = widget.NavigationToolbar2QT

    def old_signature(canvas, parent, coordinates=True):     # matplotlib before parent became optional
        return real(canvas, parent, coordinates)

    monkeypatch.setattr(widget, 'NavigationToolbar2QT', old_signature)
    window = widget.CalibrationWindow()
    assert window.navbar is not None
    window.close()
    app.processEvents()
