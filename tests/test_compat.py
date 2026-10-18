"""Guard the deployment targets: Jetson Orin runs Python 3.10 with NumPy 1.x."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compat_check  # noqa: E402


def test_no_python_3_11_or_numpy_2_only_constructs():
    """``python tools/compat_check.py --python 3.10`` must report nothing."""
    assert compat_check.main(["--python", "3.10"]) == 0
