"""Tests for package import boundaries and the compatibility facade."""
import subprocess
import sys
from pathlib import Path


PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)


def _run_isolated(code):
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=PACKAGE_PARENT,
    )


def test_processing_import_does_not_load_gui_stack():
    code = (
        "import sys, smile; "
        "assert 'smile.widget' not in sys.modules; "
        "assert 'matplotlib' not in sys.modules; "
        "assert not any(name.startswith('PyQt5') for name in sys.modules)"
    )
    _run_isolated(code)


def test_legacy_module_is_a_facade():
    code = (
        "import sys, smile; from smile.smile import SmileCorrector; "
        "assert SmileCorrector is smile.SmileCorrector; "
        "assert 'smile.widget' not in sys.modules"
    )
    _run_isolated(code)


def test_cli_entry_point_does_not_require_gui():
    code = (
        "import sys; import smile.test_cmd; "
        "assert 'smile.widget' not in sys.modules"
    )
    _run_isolated(code)


def test_gui_api_is_available_as_lazy_exports():
    code = (
        "import smile; "
        "assert {'SmileCorrectionWidget', 'SmileMainWindow', 'launch_gui'} "
        "<= set(smile.__all__)"
    )
    _run_isolated(code)
