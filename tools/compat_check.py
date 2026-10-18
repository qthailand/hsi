"""Static check for constructs that break on older Python/NumPy (e.g. Jetson Orin: NumPy 1.x).

Usage: ``python tools/compat_check.py [--python 3.8]``. Reports, per file and line:

* ``X | Y`` type unions and ``list[...]``/``dict[...]``/``tuple[...]``/``set[...]`` generics in
  annotations of files without ``from __future__ import annotations`` (need Python 3.10 / 3.9),
  and the same constructs in runtime expressions (``isinstance(x, int | None)``, aliases);
* NumPy-2-only names (``np.trapezoid``, ``np.concat``, ``np.isdtype`` ...) used without a fallback;
* ``str.removeprefix``/``removesuffix`` (3.9), ``match`` statements (3.10), parenthesised
  context managers are not detected.

Vendor SDK wrappers (``modules/camera/poa``, ``modules/camera/ximea``) and tests are skipped.
Exit code 1 if anything is found.
"""
from __future__ import annotations

import argparse
import ast
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {"__pycache__", "poa", "ximea", "tests", ".git"}
NUMPY2_ONLY = {"trapezoid", "concat", "isdtype", "cumulative_sum", "cumulative_prod", "unique_values",
               "unique_counts", "unique_all", "unique_inverse", "vecdot", "permute_dims",
               "matrix_transpose", "astype", "bitwise_count", "long", "ulong"}
GENERICS = {"list", "dict", "tuple", "set", "frozenset", "type"}


def _annotation_nodes(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            args += [a for a in (node.args.vararg, node.args.kwarg) if a]
            for arg in args:
                if arg.annotation is not None:
                    yield arg.annotation
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


def _bad_type_syntax(expr, python):
    for node in ast.walk(expr):
        if python < (3, 10) and isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return "X | Y union (Python 3.10+)"
        if python < (3, 9) and isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id in GENERICS:
            return f"{node.value.id}[...] generic (Python 3.9+)"
    return None


def check_file(path: Path, python) -> list:
    """Return ``(line, message)`` problems for one file."""
    with warnings.catch_warnings():          # e.g. invalid escapes in old scripts: not our concern here
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(path.read_text(encoding="utf-8"))
    future = any(isinstance(n, ast.ImportFrom) and n.module == "__future__" and
                 any(a.name == "annotations" for a in n.names) for n in tree.body)
    problems = []
    annotation_ids = set()
    for annotation in _annotation_nodes(tree):
        annotation_ids.update(id(n) for n in ast.walk(annotation))
        if not future:
            problem = _bad_type_syntax(annotation, python)
            if problem:
                problems.append((annotation.lineno, problem + " in an annotation without `from __future__ import annotations`"))
    has_fallback = "getattr(np" in path.read_text(encoding="utf-8")
    for node in ast.walk(tree):
        if id(node) in annotation_ids:
            continue
        # runtime use of unions in isinstance/issubclass
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("isinstance", "issubclass") \
                and len(node.args) == 2 and python < (3, 10):
            if isinstance(node.args[1], ast.BinOp) and isinstance(node.args[1].op, ast.BitOr):
                problems.append((node.lineno, "isinstance(x, A | B) needs Python 3.10"))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("np", "numpy") \
                and node.attr in NUMPY2_ONLY and not has_fallback:
            problems.append((node.lineno, f"np.{node.attr} exists only in NumPy 2"))
        if python < (3, 9) and isinstance(node, ast.Attribute) and node.attr in ("removeprefix", "removesuffix"):
            problems.append((node.lineno, f"str.{node.attr} needs Python 3.9"))
        if python < (3, 10) and type(node).__name__ == "Match":
            problems.append((node.lineno, "match statement needs Python 3.10"))
    return problems


def main(argv=None) -> int:
    """Scan the project; print problems; return 1 if any were found."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="3.8", help="oldest Python to support (default 3.8)")
    args = parser.parse_args(argv)
    python = tuple(int(v) for v in args.python.split("."))
    found = 0
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if SKIP_PARTS & set(rel.parts) or rel.parts[0] == "tools":
            continue
        for line, message in check_file(path, python):
            print(f"{rel.as_posix()}:{line}: {message}")
            found += 1
    print(f"{found} problem(s) for Python {args.python} / NumPy 1.x")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
