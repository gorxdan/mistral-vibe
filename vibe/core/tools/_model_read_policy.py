from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from vibe.core.config.harness_files import get_harness_files_manager
from vibe.core.paths import LOG_DIR, VIBE_HOME
from vibe.core.prompts import PROMPTS_DIR, SystemPrompt, UtilityPrompt

if TYPE_CHECKING:
    from vibe.core.config import TrustedExecutionTopologyConfig
    from vibe.core.tools.base import InvokeContext


class ManagedReadPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ManagedReadScope:
    roots: tuple[Path, ...]
    denied_roots: tuple[Path, ...]
    active_files: frozenset[Path]

    def resolve(self, path: str | Path) -> Path:
        resolved = _resolve_path(path)
        if any(_is_within(resolved, root) for root in self.denied_roots):
            raise ManagedReadPolicyError(
                f"managed model reads cannot access host logs, receipts, or "
                f"runtime state: {resolved}"
            )
        if any(_is_within(resolved, root) for root in self.roots):
            return resolved
        if resolved in self.active_files:
            return resolved
        raise ManagedReadPolicyError(
            "managed model reads must stay within the assigned candidate, control, "
            f"evidence, scratchpad, or active host resource: {resolved}"
        )


def managed_read_policy_active(ctx: InvokeContext | None) -> bool:
    return _managed_topology(ctx) is not None


def managed_read_scope(ctx: InvokeContext | None) -> ManagedReadScope | None:
    topology = _managed_topology(ctx)
    if topology is None:
        return None
    roots = [
        _resolve_path(topology.candidate_worktree),
        _resolve_path(topology.control_worktree),
        _resolve_path(topology.evidence_workspace),
    ]
    if ctx is not None and ctx.scratchpad_dir is not None:
        roots.append(_resolve_path(ctx.scratchpad_dir))
    roots.extend(_host_skill_roots(ctx))
    return ManagedReadScope(
        roots=tuple(dict.fromkeys(roots)),
        denied_roots=_denied_roots(),
        active_files=_active_prompt_files(ctx),
    )


def resolve_managed_read_path(
    path: str | Path, ctx: InvokeContext | None
) -> Path | None:
    scope = managed_read_scope(ctx)
    if scope is None:
        return None
    return scope.resolve(path)


def _managed_topology(
    ctx: InvokeContext | None,
) -> TrustedExecutionTopologyConfig | None:
    if ctx is None or ctx.verification_state is None:
        return None
    recipe = ctx.verification_state.trusted_recipe
    if recipe is None:
        return None
    return recipe.config.execution_topology


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ManagedReadPolicyError(
            "managed read target could not be resolved safely"
        ) from exc


def _denied_roots() -> tuple[Path, ...]:
    return tuple(
        dict.fromkeys(
            _resolve_path(root)
            for root in (
                LOG_DIR.path,
                VIBE_HOME.path / "verification",
                Path("/run"),
                Path("/var/run"),
            )
        )
    )


def _host_skill_roots(ctx: InvokeContext | None) -> tuple[Path, ...]:
    if ctx is None or ctx.skill_manager is None:
        return ()
    try:
        available = ctx.skill_manager.available_skills.values()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return ()

    roots: list[Path] = []
    for skill in available:
        skill_dir = getattr(skill, "skill_dir", None)
        if skill_dir is None:
            continue
        try:
            resolved = _resolve_path(skill_dir)
        except ManagedReadPolicyError:
            continue
        roots.append(resolved)
    return tuple(dict.fromkeys(roots))


def _active_prompt_files(ctx: InvokeContext | None) -> frozenset[Path]:
    if ctx is None or ctx.agent_manager is None:
        return frozenset()
    try:
        config = ctx.agent_manager.config
        manager = get_harness_files_manager()
        custom_dirs = [
            *(Path(path) for path in config.prompt_paths),
            *manager.project_prompts_dirs,
            *manager.user_prompts_dirs,
        ]
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return frozenset()

    selected: list[Path] = []
    system_id = str(config.system_prompt_id)
    if system_id == SystemPrompt.VERIFIER:
        selected.append(SystemPrompt.VERIFIER.path)
    elif path := _first_prompt_path(
        system_id,
        custom_dirs,
        {prompt.value: prompt.path for prompt in SystemPrompt},
        fallback_dir=PROMPTS_DIR,
    ):
        selected.append(path)

    if path := _first_prompt_path(
        str(config.compaction_prompt_id),
        custom_dirs,
        {UtilityPrompt.COMPACT.value: UtilityPrompt.COMPACT.path},
    ):
        selected.append(path)
    return frozenset(_resolve_path(path) for path in selected)


def _first_prompt_path(
    prompt_id: str,
    custom_dirs: Iterable[Path],
    builtins: dict[str, Path],
    *,
    fallback_dir: Path | None = None,
) -> Path | None:
    if (
        not prompt_id
        or prompt_id in {".", ".."}
        or "/" in prompt_id
        or "\\" in prompt_id
    ):
        return None
    for directory in custom_dirs:
        candidate = (Path(directory) / prompt_id).with_suffix(".md")
        if candidate.is_file():
            return candidate
    builtin = builtins.get(prompt_id.casefold())
    if builtin is not None and builtin.is_file():
        return builtin
    if fallback_dir is None:
        return None
    fallback = (fallback_dir / prompt_id).with_suffix(".md")
    return fallback if fallback.is_file() else None


def _is_within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


__all__ = [
    "ManagedReadPolicyError",
    "ManagedReadScope",
    "managed_read_policy_active",
    "managed_read_scope",
    "resolve_managed_read_path",
]
