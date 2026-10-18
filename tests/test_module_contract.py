"""Enforce docs/MODULE_CONTRACT.md on every package under modules/."""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import module_audit as audit  # noqa: E402

NAMES = audit.module_names()
REPORTS = {name: audit.audit_module(name) for name in NAMES}


@pytest.mark.parametrize("name", NAMES)
def test_static_rules(name):
    """R1, R2, R4, R6: layout, declared dependencies, docstrings, launcher."""
    assert not REPORTS[name].issues, "\n".join(REPORTS[name].issues)


def test_no_dependency_cycles():
    assert audit.find_cycle(REPORTS) is None


@pytest.mark.parametrize("name", NAMES)
def test_import_does_not_load_qt(name):
    """R3: headless use must not pull in PyQt5."""
    assert not audit.imports_qt(name)


@pytest.mark.parametrize("name", [n for n in NAMES if REPORTS[n].widgets])
def test_widgets_embed_in_a_foreign_window(name):
    """R5: each declared widget builds with no arguments inside any QMainWindow and closes."""
    from PyQt5.QtCore import QEvent
    from PyQt5.QtWidgets import QApplication, QMainWindow
    app = QApplication.instance() or QApplication([])
    package = importlib.import_module(f"modules.{name}")
    for widget_name in REPORTS[name].widgets:
        window = QMainWindow()
        widget = getattr(package, widget_name)()
        window.setCentralWidget(widget)
        window.show()
        app.processEvents()
        window.close()
        widget.close()                      # a widget owning threads must stop them here ...
        widget.deleteLater()                # ... and survive being destroyed by its host
        window.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()


def test_the_template_itself_follows_the_contract():
    """New modules start from modules/_template, so it must be a valid module."""
    report = audit.audit_module("_template")
    assert not report.issues, "\n".join(report.issues)
    assert not audit.imports_qt("_template")


def test_module_index_is_current():
    """docs/MODULES.md is generated; run `python tools/module_index.py` after changing a module."""
    import module_index
    assert module_index.main(["--check"]) == 0, "docs/MODULES.md is stale: run python tools/module_index.py"
