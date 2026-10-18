"""Report .py files whose code changed, ignoring docstrings and module-contract markers.

Usage: ``python tools/ast_equiv.py <snapshot_dir> <modules_dir>``. Used to prove a
documentation pass did not change behaviour: any file listed has non-doc changes
and must be reviewed.
"""
import ast
import sys
from pathlib import Path

MARKERS = {"__all__", "__requires__", "__widgets__"}


def normalized(source: str) -> str:
    """Dump the AST of ``source`` without docstrings and ``__all__``-style markers."""
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in MARKERS for t in n.targets))]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def main(snapshot: str, current: str) -> int:
    """Print changed/new/missing files; return the number of changed files."""
    snap, cur = Path(snapshot), Path(current)
    changed = 0
    for path in sorted(snap.rglob("*.py")):
        other = cur / path.relative_to(snap)
        if not other.exists():
            print("MISSING ", path.relative_to(snap)); changed += 1; continue
        if normalized(path.read_text(encoding="utf-8")) != normalized(other.read_text(encoding="utf-8")):
            print("CHANGED ", path.relative_to(snap)); changed += 1
    for path in sorted(cur.rglob("*.py")):
        if "__pycache__" not in path.parts and not (snap / path.relative_to(cur)).exists():
            print("NEW     ", path.relative_to(cur))
    return changed


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1], sys.argv[2]) else 0)
