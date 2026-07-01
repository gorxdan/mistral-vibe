from __future__ import annotations

import ast
from pathlib import Path
import re

from tests import TESTS_ROOT

VIBE_PKG = TESTS_ROOT.parent / "vibe"
CONFIG_PKG = VIBE_PKG / "core" / "config"
MODEL_ROOTS = {"BaseModel", "BaseSettings"}
TYPE_IGNORE_BUDGET = 1
NOQA_BUDGET = 3


def _prod_files() -> list[Path]:
    return [p for p in VIBE_PKG.rglob("*.py") if "workflows/bundled" not in str(p)]


def _base_names(node: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for base in node.bases:
        match base:
            case ast.Name(id=name) | ast.Subscript(value=ast.Name(id=name)):
                names.add(name)
    return names


def _sets_extra(node: ast.ClassDef) -> bool:
    for stmt in node.body:
        match stmt:
            case ast.Assign(
                targets=[ast.Name(id="model_config")], value=ast.Call(keywords=kwargs)
            ):
                return any(kw.arg == "extra" for kw in kwargs)
    return False


def test_config_models_declare_extra() -> None:
    classes: dict[str, tuple[set[str], bool, str]] = {}
    for path in CONFIG_PKG.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = _base_names(node)
                classes[node.name] = (bases, _sets_extra(node), f"{path}:{node.lineno}")

    def is_model(name: str) -> bool:
        if name in MODEL_ROOTS:
            return True
        entry = classes.get(name)
        return entry is not None and any(is_model(b) for b in entry[0])

    def covered(name: str) -> bool:
        entry = classes.get(name)
        if entry is None:
            return False
        return entry[1] or any(covered(b) for b in entry[0])

    violations = [
        f"{loc} {name}"
        for name, (bases, _, loc) in classes.items()
        if any(is_model(b) for b in bases) and not covered(name)
    ]
    assert not violations, (
        "config models missing explicit model_config extra= (AGENTS.md: "
        "`ConfigDict(extra=...)` always set):\n" + "\n".join(sorted(violations))
    )


def _count_marker(pattern: str) -> list[str]:
    rx = re.compile(pattern)
    return [
        f"{path}:{i}"
        for path in _prod_files()
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if rx.search(line)
    ]


def test_prod_type_ignore_ratchet() -> None:
    hits = _count_marker(r"#\s*type:\s*ignore")
    assert len(hits) <= TYPE_IGNORE_BUDGET, (
        f"prod `type: ignore` count grew past the ratchet "
        f"({len(hits)} > {TYPE_IGNORE_BUDGET}); fix at source:\n" + "\n".join(hits)
    )


def test_prod_noqa_ratchet() -> None:
    hits = _count_marker(r"#\s*noqa")
    assert len(hits) <= NOQA_BUDGET, (
        f"prod `noqa` count grew past the ratchet ({len(hits)} > {NOQA_BUDGET}); "
        f"fix at source:\n" + "\n".join(hits)
    )
