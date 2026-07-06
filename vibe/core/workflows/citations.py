from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from vibe.core.utils.io import read_safe
from vibe.core.workflows.contract import _confine

_MAX_PATH_BYTES = 1_000_000
_SUMMARY_MAX = 3
_LINE_RANGE_PARTS = 2


class CitationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items_path: str
    path_field: str
    line_field: str | None = None
    snippet_field: str | None = None
    require_all: bool = True
    strict: bool = False


class CitationViolation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    message: str
    index: int = 0


class CitationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    items_checked: int = 0
    items_verified: int = 0
    violations: list[CitationViolation] = []
    dropped_indices: list[int] = []

    def summary(self) -> str:
        if self.passed:
            return f"citations verified ({self.items_verified}/{self.items_checked})"
        detail = "; ".join(v.message for v in self.violations[:_SUMMARY_MAX])
        more = (
            ""
            if len(self.violations) <= _SUMMARY_MAX
            else f" (+{len(self.violations) - _SUMMARY_MAX} more)"
        )
        return (
            f"citations failed ({len(self.violations)} violation(s), "
            f"{len(self.dropped_indices)} dropped): {detail}{more}"
        )


class CitationFailure(dict):
    """Falsy dict returned when strict citation verification fails. Mirrors
    ContractFailure / SchemaValidationFailure: filter with
    ``[r for r in results if r]`` (truthiness), NOT ``isinstance(r, dict)``.
    """

    def __init__(self, *, report: Any, error: str = "") -> None:
        report_data = (
            report.model_dump(mode="json") if hasattr(report, "model_dump") else report
        )
        super().__init__(report=report_data, error=error)

    def __bool__(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _navigate(output: Any, dotted_path: str) -> tuple[list[Any], bool]:
    """Resolve a dotted path (``"results.findings"``) through nested dicts to a
    list. Returns ``(items, found)``; ``found`` is False if any segment is
    missing or the terminal value is not a list.
    """
    current: Any = output
    for segment in dotted_path.split("."):
        if isinstance(current, dict) and segment in current:
            current = current[segment]
        else:
            return [], False
    if isinstance(current, list):
        return current, True
    return [], False


def _parse_line_spec(value: Any) -> tuple[int, int] | None:
    """Parse a line reference into a (lo, hi) inclusive range. Accepts an int
    or a ``"lo-hi"`` string. Returns None if the value is not parseable.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value, value
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            n = int(text)
            return n, n
        if "-" in text:
            parts = text.split("-")
            if len(parts) == _LINE_RANGE_PARTS:
                lo_s, hi_s = parts[0].strip(), parts[1].strip()
                if lo_s.isdigit() and hi_s.isdigit():
                    return int(lo_s), int(hi_s)
    return None


_LoadFn = Callable[[str], tuple[bool, str, int]]


def _make_loader(root: Path) -> _LoadFn:
    cache: dict[str, tuple[bool, str, int]] = {}

    def _load(rel: str) -> tuple[bool, str, int]:
        if rel in cache:
            return cache[rel]
        where = _confine(root, rel)
        if where is None or not where.exists() or where.is_dir():
            cache[rel] = (False, "", 0)
            return cache[rel]
        try:
            if where.stat().st_size > _MAX_PATH_BYTES:
                cache[rel] = (False, "", 0)
                return cache[rel]
            text = read_safe(where).text
        except OSError:
            cache[rel] = (False, "", 0)
            return cache[rel]
        line_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        cache[rel] = (True, text, line_count)
        return cache[rel]

    return _load


def _check_item(
    item: Any, index: int, spec: CitationSpec, load: _LoadFn
) -> tuple[list[CitationViolation], bool]:
    """Verify one finding's citations. Returns ``(violations, verified)`` —
    ``verified`` is True when the item passed every check and should be kept.
    """
    if not isinstance(item, dict):
        return (
            [
                CitationViolation(
                    category="path",
                    index=index,
                    message=f"item {index}: not an object, cannot extract citation",
                )
            ],
            False,
        )

    rel = item.get(spec.path_field)
    if not rel or not isinstance(rel, str):
        if spec.require_all:
            return (
                [
                    CitationViolation(
                        category="path",
                        index=index,
                        message=f"item {index}: missing path field {spec.path_field!r}",
                    )
                ],
                False,
            )
        return [], False

    ok, text, line_count = load(rel)
    if not ok:
        return (
            [
                CitationViolation(
                    category="path",
                    index=index,
                    message=f"item {index}: cited path {rel!r} not found under root",
                )
            ],
            False,
        )

    violations: list[CitationViolation] = []

    if spec.line_field is not None and spec.line_field in item:
        parsed = _parse_line_spec(item[spec.line_field])
        if parsed is None:
            violations.append(
                CitationViolation(
                    category="line",
                    index=index,
                    message=f"item {index}: unparseable line ref {item[spec.line_field]!r}",
                )
            )
        else:
            lo, hi = parsed
            if lo < 1 or hi < lo or hi > max(line_count, 1):
                violations.append(
                    CitationViolation(
                        category="line",
                        index=index,
                        message=(
                            f"item {index}: line {item[spec.line_field]!r} out of range "
                            f"for {rel} ({line_count} line(s))"
                        ),
                    )
                )

    if spec.snippet_field is not None and spec.snippet_field in item:
        snippet = item[spec.snippet_field]
        if not isinstance(snippet, str) or snippet not in text:
            violations.append(
                CitationViolation(
                    category="snippet",
                    index=index,
                    message=(
                        f"item {index}: snippet not found in {rel}"
                        if isinstance(snippet, str)
                        else f"item {index}: snippet field is not a string"
                    ),
                )
            )

    return violations, not violations


def verify_citations(root: Path, output: Any, spec: CitationSpec) -> CitationReport:
    """Reconcile citation claims in ``output`` against the filesystem at
    ``root``. For each item at ``spec.items_path``, checks that the cited path
    exists (confined to ``root``), that the optional line reference is in range,
    and that the optional snippet appears in the file. No model in the loop —
    the gate is deterministic, like ``verify_contract``.
    """
    items, found = _navigate(output, spec.items_path)
    if not found:
        return CitationReport(
            passed=True,
            items_checked=0,
            items_verified=0,
            violations=[],
            dropped_indices=[],
        )

    load = _make_loader(root)
    violations: list[CitationViolation] = []
    dropped: list[int] = []
    verified = 0
    for i, item in enumerate(items):
        item_violations, item_ok = _check_item(item, i, spec, load)
        if item_violations:
            violations.extend(item_violations)
            dropped.append(i)
        if item_ok:
            verified += 1

    return CitationReport(
        passed=not violations,
        items_checked=len(items),
        items_verified=verified,
        violations=violations,
        dropped_indices=dropped,
    )


def apply_citation_report(
    output: dict[str, Any], report: CitationReport, spec: CitationSpec
) -> dict[str, Any] | CitationFailure:
    """Reconcile parsed agent output against a citation report. Non-strict
    (default): drops items at ``dropped_indices`` from the array at
    ``items_path`` and attaches the report under ``citation_report``. Strict:
    returns a falsy ``CitationFailure`` if any violation occurred.
    """
    if report.passed:
        result = {**output, "citation_report": report.model_dump(mode="json")}
        return result

    if spec.strict:
        return CitationFailure(report=report, error=report.summary())

    items, found = _navigate(output, spec.items_path)
    if not found:
        return {**output, "citation_report": report.model_dump(mode="json")}

    drop_set = set(report.dropped_indices)
    kept = [item for i, item in enumerate(items) if i not in drop_set]

    result = dict(output)
    _set_dotted(result, spec.items_path, kept)
    result["citation_report"] = report.model_dump(mode="json")
    return result


def _set_dotted(target: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set a value at a dotted path in ``target``, creating intermediate dicts."""
    segments = dotted_path.split(".")
    current: dict[str, Any] = target
    for segment in segments[:-1]:
        if segment not in current or not isinstance(current[segment], dict):
            current[segment] = {}
        current = current[segment]
    current[segments[-1]] = value
