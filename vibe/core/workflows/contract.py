from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.utils.io import read_safe

_MAX_GREP_FILE_BYTES = 1_000_000
_MAX_INVARIANT_SNIPPETS = 3
_SUMMARY_MAX = 3
_GREP_SKIP_DIRS = frozenset({".git"})


class ContractOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    must_exist: bool = True
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    must_match: list[str] = Field(default_factory=list)
    must_not_match: list[str] = Field(default_factory=list)
    min_size: int | None = None
    max_size: int | None = None


class ContractInvariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grep: str
    must_match: bool = True
    description: str = ""


class ContractTest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str
    expect: str | None = None
    timeout: int = 60


class ContractSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outputs: list[ContractOutput] = Field(default_factory=list)
    invariants: list[ContractInvariant] = Field(default_factory=list)
    tests: list[ContractTest] = Field(default_factory=list)


class ContractViolation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    message: str
    path: str = ""


class ContractReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    delivered: bool = False
    violations: list[ContractViolation] = Field(default_factory=list)

    def summary(self) -> str:
        if self.passed and self.delivered:
            return "contract passed (delivered)"
        if self.passed:
            return "contract passed but delivery skipped (branch kept)"
        detail = "; ".join(v.message for v in self.violations[:_SUMMARY_MAX])
        more = (
            ""
            if len(self.violations) <= _SUMMARY_MAX
            else f" (+{len(self.violations) - _SUMMARY_MAX} more)"
        )
        return f"contract failed ({len(self.violations)} violation(s)): {detail}{more}"


class ContractFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report: ContractReport
    error: str

    def __bool__(self) -> bool:
        return False

    def get(self, key: str, default: Any = None) -> Any:
        return default


def _confine(root: Path, rel: str) -> Path | None:
    base = root.resolve()
    try:
        target = (base / rel).resolve()
    except (OSError, RuntimeError):
        return None
    try:
        target.relative_to(base)
    except ValueError:
        return None
    return target


def _iter_tree(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in _GREP_SKIP_DIRS for part in path.parts):
            continue
        yield path


def _check_output(root: Path, out: ContractOutput) -> list[ContractViolation]:
    where = _confine(root, out.path)
    if where is None:
        return [
            ContractViolation(
                category="output",
                path=out.path,
                message=f"{out.path}: path escapes the worktree root",
            )
        ]
    if not where.exists():
        if out.must_exist:
            return [
                ContractViolation(
                    category="output",
                    path=out.path,
                    message=f"{out.path}: required output missing",
                )
            ]
        return []
    if where.is_dir():
        return [
            ContractViolation(
                category="output",
                path=out.path,
                message=f"{out.path}: expected a file but found a directory",
            )
        ]
    violations: list[ContractViolation] = []
    size = where.stat().st_size
    if out.min_size is not None and size < out.min_size:
        violations.append(
            ContractViolation(
                category="output",
                path=out.path,
                message=f"{out.path}: size {size} < min_size {out.min_size}",
            )
        )
    if out.max_size is not None and size > out.max_size:
        violations.append(
            ContractViolation(
                category="output",
                path=out.path,
                message=f"{out.path}: size {size} > max_size {out.max_size}",
            )
        )
    text = read_safe(where).text
    for needle in out.must_contain:
        if needle not in text:
            violations.append(
                ContractViolation(
                    category="output",
                    path=out.path,
                    message=f"{out.path}: missing required substring {needle!r}",
                )
            )
    for needle in out.must_not_contain:
        if needle in text:
            violations.append(
                ContractViolation(
                    category="output",
                    path=out.path,
                    message=f"{out.path}: forbidden substring {needle!r} present",
                )
            )
    for pattern in out.must_match:
        if not re.search(pattern, text):
            violations.append(
                ContractViolation(
                    category="output",
                    path=out.path,
                    message=f"{out.path}: required pattern {pattern!r} not found",
                )
            )
    for pattern in out.must_not_match:
        if re.search(pattern, text):
            violations.append(
                ContractViolation(
                    category="output",
                    path=out.path,
                    message=f"{out.path}: forbidden pattern {pattern!r} present",
                )
            )
    return violations


def _check_invariant(root: Path, inv: ContractInvariant) -> list[ContractViolation]:
    try:
        regex = re.compile(inv.grep)
    except re.error as e:
        return [
            ContractViolation(
                category="invariant", message=f"invalid grep pattern {inv.grep!r}: {e}"
            )
        ]
    found = False
    snippets: list[str] = []
    for path in _iter_tree(root):
        try:
            if path.stat().st_size > _MAX_GREP_FILE_BYTES:
                continue
            text = read_safe(path).text
        except OSError:
            continue
        match = regex.search(text)
        if match:
            found = True
            if len(snippets) < _MAX_INVARIANT_SNIPPETS:
                rel = path.relative_to(root).as_posix()
                snippets.append(f"{rel}: ...{match.group(0)}...")
    label = inv.description or inv.grep
    if inv.must_match and not found:
        return [
            ContractViolation(
                category="invariant",
                message=f"invariant {label!r}: pattern {inv.grep!r} not found anywhere",
            )
        ]
    if not inv.must_match and found:
        return [
            ContractViolation(
                category="invariant",
                message=f"invariant {label!r}: forbidden pattern {inv.grep!r} present ({'; '.join(snippets)})",
            )
        ]
    return []


def _check_test(root: Path, test: ContractTest) -> list[ContractViolation]:
    try:
        proc = subprocess.run(
            shlex.split(test.command),
            cwd=str(root),
            shell=False,
            capture_output=True,
            timeout=test.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [
            ContractViolation(
                category="test",
                message=f"test {test.command!r} timed out after {test.timeout}s",
            )
        ]
    except (OSError, ValueError) as e:
        return [
            ContractViolation(
                category="test", message=f"test {test.command!r} could not run: {e}"
            )
        ]
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace")[-200:]
        return [
            ContractViolation(
                category="test",
                message=f"test {test.command!r} exited {proc.returncode}: {tail.strip()}",
            )
        ]
    if test.expect is not None and test.expect not in proc.stdout.decode(
        "utf-8", "replace"
    ):
        return [
            ContractViolation(
                category="test",
                message=f"test {test.command!r} stdout missing expected substring {test.expect!r}",
            )
        ]
    return []


def verify_contract(root: Path, spec: ContractSpec) -> ContractReport:
    violations: list[ContractViolation] = []
    for out in spec.outputs:
        violations.extend(_check_output(root, out))
    for inv in spec.invariants:
        violations.extend(_check_invariant(root, inv))
    for test in spec.tests:
        violations.extend(_check_test(root, test))
    return ContractReport(passed=not violations, violations=violations)
