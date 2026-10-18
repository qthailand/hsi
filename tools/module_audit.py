"""Audit ``modules/*`` against docs/MODULE_CONTRACT.md.

Run ``python tools/module_audit.py`` for a report (exit code 1 if any rule fails);
``tests/test_module_contract.py`` runs the same checks under pytest.

The checks are static (AST) except R3, which imports the package in a fresh
interpreter to prove PyQt5 is not loaded.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "modules"
APP_LAYERS = {"windows", "widgets", "workers", "controllers", "utils", "config",
              "QTNarit", "console", "main", "ImageSharpness", "Stage8SMC1", "BiningTools"}
DOC_THRESHOLD = 0.80
SKIP_FILES = {"conftest.py", "__main__.py"}
VENDOR_DIRS = {"poa", "ximea"}   # vendor SDK wrappers inside modules/camera; not our API


@dataclass
class ModuleReport:
    """Result of auditing one module package."""

    name: str
    requires: tuple = ()
    widgets: tuple = ()
    actual_requires: set = field(default_factory=set)
    doc_total: int = 0
    doc_done: int = 0
    issues: list = field(default_factory=list)

    @property
    def doc_ratio(self) -> float:
        """Fraction of public classes/functions/methods that have a docstring."""
        return self.doc_done / self.doc_total if self.doc_total else 1.0


def module_names() -> list:
    """Names of all feature packages under ``modules/`` (``_``-prefixed ones excluded)."""
    return sorted(p.name for p in MODULES.iterdir()
                  if p.is_dir() and (p / "__init__.py").exists() and not p.name.startswith("_"))


def _literal_tuple(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                return None
            return tuple(value)
    return None


def _sources(package: Path):
    for path in sorted(package.rglob("*.py")):
        if ("__pycache__" in path.parts or path.name in SKIP_FILES or path.name.startswith("test_")
                or (VENDOR_DIRS & set(path.relative_to(package).parts[:-1]))):
            continue
        yield path


def _imports(tree: ast.Module, package: str, path: Path, depth_from_modules: int):
    """Yield dotted names imported by ``tree``; relative imports are resolved to ``modules.*``."""
    directory = ["modules"] + list(path.parent.relative_to(MODULES).parts)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    yield node.module
                continue
            base = directory[: len(directory) - (node.level - 1)]
            if node.module:
                yield ".".join(base + [node.module])
            else:
                for alias in node.names:
                    yield ".".join(base + [alias.name])


def audit_module(name: str) -> ModuleReport:
    """Check R1, R2, R4 and R6 for module ``name`` (R3 and R5 are runtime checks)."""
    package = MODULES / name
    report = ModuleReport(name)
    init_tree = ast.parse((package / "__init__.py").read_text(encoding="utf-8"))

    if not ast.get_docstring(init_tree):
        report.issues.append("R1: __init__.py has no docstring")
    if _literal_tuple(init_tree, "__all__") is None and not any(
            isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in n.targets)
            for n in init_tree.body):
        report.issues.append("R1: __init__.py defines no __all__")
    requires = _literal_tuple(init_tree, "__requires__")
    if requires is None:
        report.issues.append("R1: __requires__ must be a literal tuple of module names")
        requires = ()
    report.requires = requires
    report.widgets = _literal_tuple(init_tree, "__widgets__") or ()
    if not (package / "README.md").exists():
        report.issues.append("R4: README.md missing")
    if report.widgets and not (package / "__main__.py").exists():
        report.issues.append("R6: has __widgets__ but no __main__.py")

    for path in _sources(package):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(ROOT).as_posix()
        if not ast.get_docstring(tree):
            report.issues.append(f"R4: {rel} has no module docstring")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                report.doc_total += 1
                report.doc_done += bool(ast.get_docstring(node))
        for imported in _imports(tree, name, path, 0):
            parts = imported.split(".")
            if parts[0] in APP_LAYERS:
                report.issues.append(f"R2: {rel} imports application layer '{imported}'")
            if parts[0] == "modules" and len(parts) > 1 and not parts[1].startswith("_") and parts[1] != name:
                report.actual_requires.add(parts[1])

    undeclared = report.actual_requires - set(requires)
    if undeclared:
        report.issues.append(f"R2: imports {sorted(undeclared)} but __requires__ does not list them")
    unknown = set(requires) - set(module_names())
    if unknown:
        report.issues.append(f"R2: __requires__ names unknown modules {sorted(unknown)}")
    if report.doc_ratio < DOC_THRESHOLD:
        report.issues.append(
            f"R4: docstring coverage {report.doc_ratio:.0%} < {DOC_THRESHOLD:.0%} "
            f"({report.doc_done}/{report.doc_total})")
    return report


def find_cycle(reports: dict):
    """Return one dependency cycle (list of names) among declared requirements, or None."""
    state = {}

    def visit(node, stack):
        state[node] = 1
        for dep in reports[node].requires if node in reports else ():
            if dep not in reports:
                continue
            if state.get(dep) == 1:
                return stack + [node, dep]
            if dep not in state:
                found = visit(dep, stack + [node])
                if found:
                    return found
        state[node] = 2
        return None

    for name in reports:
        if name not in state:
            found = visit(name, [])
            if found:
                return found
    return None


def imports_qt(name: str) -> bool:
    """R3: True if ``import modules.<name>`` loads PyQt5 (checked in a fresh interpreter)."""
    code = f"import sys; import modules.{name}; sys.exit(1 if any(m == 'PyQt5' or m.startswith('PyQt5.') for m in sys.modules) else 0)"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"import modules.{name} failed:\n{result.stderr[-800:]}")
    return result.returncode == 1


def main() -> int:
    """Print the audit table; return 1 if any module breaks a rule."""
    reports = {name: audit_module(name) for name in module_names()}
    failed = 0
    cycle = find_cycle(reports)
    print(f"{'module':26}{'docs':>6}  {'requires':40} issues")
    for name, rep in reports.items():
        if imports_qt(name):
            rep.issues.append("R3: importing the package loads PyQt5 (make widget imports lazy)")
        failed += bool(rep.issues)
        print(f"{name:26}{rep.doc_ratio:>6.0%}  {', '.join(rep.requires) or '-':40} {len(rep.issues) or 'ok'}")
        for issue in rep.issues:
            print(f"      - {issue}")
    if cycle:
        failed += 1
        print("Dependency cycle:", " -> ".join(cycle))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
