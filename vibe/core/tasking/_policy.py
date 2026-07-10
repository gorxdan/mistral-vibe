from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum, auto
import hashlib
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import orjson
from pydantic import BaseModel

from vibe.core.tasking._path_scope import path_matches_scope
from vibe.core.tasking.models import TaskBrief
from vibe.core.tools._task_manifest import (
    TaskManifestError,
    TaskToolManifest,
    resolve_task_manifest,
)
from vibe.core.tools.utils import is_team_metadata_path

if TYPE_CHECKING:
    from vibe.core._verification_runner import TrustedCheck
    from vibe.core.usage import SpendEnvelopeLimits
    from vibe.core.verification_state import VerificationState

__all__ = [
    "BoundTaskContract",
    "TaskContractAuthority",
    "TaskContractError",
    "TaskContractViolation",
]


class TaskContractAuthority(StrEnum):
    USER = auto()
    LEAD = auto()
    TRUSTED_RECIPE = auto()


class TaskContractError(ValueError):
    pass


class TaskContractViolation(TaskContractError):
    pass


def _has_glob(pattern: str) -> bool:
    return any(character in pattern for character in "*?[")


def _pattern_is_within(candidate: str, trusted: str) -> bool:
    if trusted in {candidate, "**"}:
        return True
    if not _has_glob(candidate):
        return path_matches_scope(candidate, trusted)
    if not _has_glob(trusted):
        prefix = trusted.rstrip("/")
        return candidate.startswith(f"{prefix}/")
    if trusted.endswith("/**"):
        return candidate.startswith(trusted.removesuffix("**"))
    return False


def _relative_scope_path(path: Path, workspace_root: Path) -> str:
    try:
        return path.relative_to(workspace_root).as_posix()
    except ValueError as e:
        raise TaskContractViolation(
            f"path escapes the bound task workspace: {path}"
        ) from e


def _is_control_plane_path(path: str) -> bool:
    parts = tuple(part.casefold() for part in PurePosixPath(path).parts)
    return any(part in {".agents", ".git", ".vibe"} for part in parts) or any(
        part == "agents.md" for part in parts
    )


def _matches_denied_path(path: str, patterns: Sequence[str]) -> bool:
    folded_path = path.casefold()
    return any(
        path_matches_scope(path, pattern)
        or path_matches_scope(folded_path, pattern.casefold())
        for pattern in patterns
    )


@dataclass(frozen=True, slots=True)
class BoundTaskContract:
    authority: TaskContractAuthority
    workspace_root: Path
    objective: str
    allowed_paths: tuple[str, ...]
    denied_paths: tuple[str, ...]
    acceptance_check_ids: tuple[str, ...]
    trusted_checks: tuple[TrustedCheck, ...]
    manifest: TaskToolManifest
    brief_hash: str
    max_tokens: int | None
    max_cost_usd: float | None
    max_calls: int | None
    deadline: datetime | None

    @classmethod
    def bind(
        cls,
        brief: TaskBrief,
        *,
        authority: TaskContractAuthority,
        workspace_root: Path,
        verification_state: VerificationState | None,
    ) -> BoundTaskContract:
        try:
            manifest = resolve_task_manifest(brief.manifest)
        except TaskManifestError as e:
            raise TaskContractError(str(e)) from e

        recipe = verification_state.trusted_recipe if verification_state else None
        if recipe is None:
            raise TaskContractError(
                "structured tasks require a trusted verification recipe"
            )

        checks_by_name = {check.name: check for check in recipe.checks}
        unknown_checks = [
            check_id
            for check_id in brief.acceptance_checks
            if check_id not in checks_by_name
        ]
        if unknown_checks:
            names = ", ".join(unknown_checks)
            raise TaskContractError(f"untrusted acceptance check IDs: {names}")

        outside_recipe = [
            candidate
            for candidate in brief.allowed_paths
            if not any(
                _pattern_is_within(candidate, trusted)
                for trusted in recipe.allowed_paths
            )
        ]
        if outside_recipe:
            paths = ", ".join(outside_recipe)
            raise TaskContractError(
                f"task paths exceed the trusted verification recipe: {paths}"
            )

        budget = brief.budget
        return cls(
            authority=authority,
            workspace_root=workspace_root.resolve(),
            objective=str(brief.objective),
            allowed_paths=tuple(brief.allowed_paths),
            denied_paths=tuple(brief.denied_paths),
            acceptance_check_ids=tuple(brief.acceptance_checks),
            trusted_checks=tuple(
                checks_by_name[name] for name in brief.acceptance_checks
            ),
            manifest=manifest,
            brief_hash=hashlib.sha256(
                orjson.dumps(brief.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS)
            ).hexdigest(),
            max_tokens=budget.max_tokens if budget else None,
            max_cost_usd=budget.max_cost_usd if budget else None,
            max_calls=budget.max_calls if budget else None,
            deadline=brief.deadline,
        )

    @property
    def allowed_tools(self) -> frozenset[str]:
        return frozenset(self.manifest.tools)

    @property
    def search_exclude_patterns(self) -> tuple[str, ...]:
        return (".agents/", ".git/", ".vibe/", "AGENTS.md")

    def spend_limits(self) -> SpendEnvelopeLimits:
        from vibe.core.usage import SpendEnvelopeLimits

        return SpendEnvelopeLimits(
            max_total_tokens=self.max_tokens,
            max_cost_usd=self.max_cost_usd,
            max_calls=self.max_calls,
            deadline_at=self.deadline.timestamp() if self.deadline else None,
        )

    def allows_search_result(self, path: str | Path) -> bool:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        try:
            resolved = candidate.resolve()
            relative = _relative_scope_path(resolved, self.workspace_root)
        except (OSError, TaskContractViolation):
            return False
        if is_team_metadata_path(resolved):
            return False
        if _is_control_plane_path(relative):
            return False
        if _matches_denied_path(relative, self.denied_paths):
            return False
        return any(
            path_matches_scope(relative, pattern) for pattern in self.allowed_paths
        )

    def enforce_tool_call(self, tool_name: str, arguments: BaseModel | object) -> None:
        if tool_name not in self.allowed_tools:
            raise TaskContractViolation(
                f"tool {tool_name!r} is outside manifest {self.manifest.name!r}"
            )

        if tool_name == "glob":
            pattern = getattr(arguments, "pattern", None)
            if isinstance(pattern, str) and Path(pattern).expanduser().is_absolute():
                raise TaskContractViolation(
                    "absolute glob patterns are outside the bound task path policy"
                )
            if isinstance(pattern, str) and _is_control_plane_path(pattern):
                raise TaskContractViolation(
                    "glob pattern targets host-owned harness control-plane files"
                )
        if tool_name == "lsp":
            operation = getattr(arguments, "operation", None)
            operation_name = getattr(operation, "value", operation)
            if operation_name == "workspace_symbol":
                raise TaskContractViolation(
                    "workspace-wide LSP queries are outside the bound task path policy"
                )

        path_value: object | None = None
        if tool_name in {"glob", "grep", "write_file"}:
            path_value = getattr(arguments, "path", None)
        elif tool_name in {"edit", "lsp", "read"}:
            path_value = getattr(arguments, "file_path", None)
        if not isinstance(path_value, str) or not path_value.strip():
            if tool_name in {"edit", "glob", "grep", "lsp", "read", "write_file"}:
                raise TaskContractViolation(
                    f"tool {tool_name!r} did not provide a valid path"
                )
            return

        requested = Path(path_value).expanduser()
        if not requested.is_absolute():
            requested = self.workspace_root / requested
        resolved = requested.resolve()
        if is_team_metadata_path(resolved):
            raise TaskContractViolation(
                f"path is host-owned team coordination metadata: {resolved}"
            )
        relative = _relative_scope_path(resolved, self.workspace_root)
        if _is_control_plane_path(relative):
            raise TaskContractViolation(
                f"path is host-owned harness control-plane metadata: {relative}"
            )
        if _matches_denied_path(relative, self.denied_paths):
            raise TaskContractViolation(
                f"path is denied by the task contract: {relative}"
            )
        if not any(
            path_matches_scope(relative, pattern) for pattern in self.allowed_paths
        ):
            raise TaskContractViolation(
                f"path is outside the task contract allowlist: {relative}"
            )

    def validate_changed_paths(self, changed_paths: Sequence[str]) -> None:
        violations: list[str] = []
        control_plane_violations: list[str] = []
        for path in changed_paths:
            candidate = Path(path)
            if candidate.is_absolute() or ".." in candidate.parts:
                violations.append(path)
                continue
            normalized = candidate.as_posix()
            if _is_control_plane_path(normalized):
                control_plane_violations.append(normalized)
                continue
            if _matches_denied_path(normalized, self.denied_paths):
                violations.append(normalized)
                continue
            if not any(
                path_matches_scope(normalized, pattern)
                for pattern in self.allowed_paths
            ):
                violations.append(normalized)
        if control_plane_violations:
            rendered = ", ".join(sorted(set(control_plane_violations)))
            raise TaskContractViolation(
                f"candidate changed host-owned control-plane paths: {rendered}"
            )
        if violations:
            rendered = ", ".join(sorted(set(violations)))
            raise TaskContractViolation(
                f"candidate changed paths outside the task contract: {rendered}"
            )
