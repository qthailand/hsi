"""Shared helpers of the acceptance suite: paths, statuses, hashing, git, JSON / subprocess I/O.

No path is hard-coded: the project root is the parent of this folder; the configuration comes
from the application's own ``config.load_app_config`` (optionally another config folder).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_DIR = Path(__file__).resolve().parent

PASS, FAIL, SKIPPED = "PASS", "FAIL", "SKIPPED"
#: Test categories: synthetic unit tests (no production data) vs production acceptance tests.
SYNTHETIC, PRODUCTION, INTEGRITY = "synthetic", "production", "integrity"


@dataclass
class TestResult:
    """One line of the acceptance report.

    Attributes:
        test_id: Stable id (``production.detection_on_off``).
        category: ``synthetic`` / ``production`` / ``integrity``.
        status: ``PASS`` / ``FAIL`` / ``SKIPPED``.
        reason: Why (always set; for PASS the criterion met).
        metrics: Numbers for the CSV / HTML.
        details: Anything else (JSON report only).
        duration_s: Wall time of the test.
    """

    test_id: str
    category: str
    status: str
    reason: str
    metrics: Dict = field(default_factory=dict)
    details: Dict = field(default_factory=dict)
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        """JSON form."""
        return asdict(self)


def now_iso() -> str:
    """Local time, ISO 8601 with offset."""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(root: Optional[Path]) -> Dict[str, str]:
    """``{relative path: sha256}`` of every file under ``root`` (``{}`` when it does not exist)."""
    if root is None or not Path(root).exists():
        return {}
    root = Path(root)
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def diff_hashes(before: Dict[str, str], after: Dict[str, str]) -> Dict[str, List[str]]:
    """Files ``changed`` / ``added`` / ``removed`` between two :func:`hash_tree` results."""
    return {"changed": sorted(k for k in set(before) & set(after) if before[k] != after[k]),
            "added": sorted(set(after) - set(before)), "removed": sorted(set(before) - set(after))}


def git(*args, cwd: Path = PROJECT_ROOT) -> Optional[str]:
    """Output of ``git <args>`` in the project, or ``None`` when git is missing or fails."""
    try:
        out = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def software_identity() -> dict:
    """Git revision of the code under test, and whether the working tree has local changes."""
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_revision": head or "unknown", "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "local_changes": None if status is None else bool(status),
            "changed_files": [] if not status else [line[3:] for line in status.splitlines()][:50]}


def platform_identity() -> dict:
    """OS, CPU, Python and library versions."""
    info = {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
            "node": platform.node(), "python": sys.version.split()[0], "executable": sys.executable,
            "cpu_count": os.cpu_count()}
    for name in ("numpy", "numba", "cv2", "PyQt5.QtCore", "psutil", "cupy"):
        try:
            module = __import__(name, fromlist=["x"])
            info[name.split(".")[0]] = getattr(module, "__version__", getattr(module, "QT_VERSION_STR", "present"))
        except Exception:                                      # noqa: BLE001 - optional
            info[name.split(".")[0]] = None
    model = Path("/proc/device-tree/model")
    if model.exists():                                         # Jetson / ARM boards
        try:
            info["board"] = model.read_text(errors="ignore").strip("\x00\n ")
        except OSError:
            pass
    return info


def write_json(path: Path, data) -> None:
    """Write ``data`` as indented JSON (creates the folder)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")


def read_json(path: Path):
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def subprocess_env() -> dict:
    """Environment for workers: headless Qt, the project importable."""
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    if platform.system() == "Windows":
        env.setdefault("QT_QPA_FONTDIR", str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env
