from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from vibe.core.paths import LOG_DIR, VIBE_HOME
from vibe.core.tasking._path_scope import path_matches_scope
from vibe.core.tools._command_tokens import split_bash_tokens
from vibe.core.tools.command_safety import unwrapped_command

if TYPE_CHECKING:
    from vibe.core.verification_state import VerificationState

_PROTECTED_GIT_PARTS = frozenset({".git"})
_MANAGED_CONTROL_DIRECTORIES = frozenset({".agents", ".claude", ".codex", ".vibe"})
_MANAGED_CONTROL_FILES = frozenset({"agents.md", "claude.md"})
_PATH_MUTATORS = frozenset({
    "chmod",
    "chown",
    "cp",
    "find",
    "install",
    "ln",
    "mkdir",
    "mv",
    "rm",
    "rmdir",
    "sed",
    "tee",
    "touch",
    "truncate",
})


class ManagedWritePolicyError(ValueError):
    pass


@dataclass(frozen=True)
class ManagedWriteScope:
    root: Path
    relative: PurePosixPath
    root_identity: tuple[int, int]


def model_protected_roots(extra_roots: Iterable[Path] = ()) -> tuple[Path, ...]:
    roots = [LOG_DIR.path.resolve(), (VIBE_HOME.path / "verification").resolve()]
    roots.extend(Path(root).expanduser().resolve() for root in extra_roots)
    return tuple(dict.fromkeys(roots))


def verification_protected_roots(state: VerificationState | None) -> tuple[Path, ...]:
    if state is None or state.trusted_recipe is None:
        return ()
    topology = state.trusted_recipe.config.execution_topology
    if topology is None:
        return ()
    roots = [
        Path(topology.control_worktree).expanduser().resolve(),
        Path(topology.evidence_workspace).expanduser().resolve(),
    ]
    if topology.state == "verification":
        roots.append(Path(topology.candidate_worktree).expanduser().resolve())
    return tuple(roots)


def protected_model_write_reason(
    path: str | Path, *, extra_roots: Iterable[Path] = ()
) -> str | None:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError):
        return "model write target could not be resolved safely"
    folded_parts = tuple(part.casefold() for part in resolved.parts)
    if any(part in _PROTECTED_GIT_PARTS for part in folded_parts):
        return f"Git control metadata is host-owned and read-only: {resolved}"
    if any(
        folded_parts[index : index + 2]
        in {(".vibe", "logs"), (".vibe", "verification")}
        for index in range(len(folded_parts) - 1)
    ):
        return f"Vibe host state is read-only to model tools: {resolved}"
    for root in model_protected_roots(extra_roots):
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return f"Vibe host state is read-only to model tools: {resolved}"
    return None


def managed_candidate_write_reason(
    path: str | Path,
    state: VerificationState | None,
    *,
    scratchpad_dir: Path | None = None,
) -> str | None:
    try:
        managed_candidate_write_scope(path, state, scratchpad_dir=scratchpad_dir)
    except ManagedWritePolicyError as exc:
        return str(exc)
    return None


def managed_candidate_write_scope(
    path: str | Path,
    state: VerificationState | None,
    *,
    scratchpad_dir: Path | None = None,
) -> ManagedWriteScope | None:
    recipe = state.trusted_recipe if state is not None else None
    topology = recipe.config.execution_topology if recipe is not None else None
    allowed_paths = recipe.allowed_paths if recipe is not None else ()
    if topology is None:
        return None

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        resolved = candidate.resolve()
        candidate_root = Path(topology.candidate_worktree).expanduser().resolve()
        if scratchpad_dir is not None:
            scratchpad_root = scratchpad_dir.expanduser().resolve()
            if resolved == scratchpad_root or resolved.is_relative_to(scratchpad_root):
                stat_result = os.stat(scratchpad_root, follow_symlinks=False)
                return ManagedWriteScope(
                    root=scratchpad_root,
                    relative=PurePosixPath(resolved.relative_to(scratchpad_root)),
                    root_identity=(stat_result.st_dev, stat_result.st_ino),
                )
    except (OSError, RuntimeError):
        raise ManagedWritePolicyError(
            "managed write target could not be resolved safely"
        ) from None

    if topology.state == "verification":
        raise ManagedWritePolicyError(
            f"the frozen verification candidate is read-only: {resolved}"
        )
    try:
        relative = resolved.relative_to(candidate_root).as_posix()
    except ValueError:
        raise ManagedWritePolicyError(
            f"managed model writes must stay inside the candidate worktree: {resolved}"
        ) from None
    relative_parts = tuple(part.casefold() for part in PurePosixPath(relative).parts)
    if relative_parts and (
        relative_parts[0] in _MANAGED_CONTROL_DIRECTORIES
        or any(part in _MANAGED_CONTROL_FILES for part in relative_parts)
    ):
        raise ManagedWritePolicyError(
            f"managed harness control files are host-owned: {relative}"
        )
    allowed = any(path_matches_scope(relative, pattern) for pattern in allowed_paths)
    if not allowed:
        raise ManagedWritePolicyError(
            f"path is outside the managed candidate allowlist: {relative}"
        )
    try:
        stat_result = os.stat(candidate_root, follow_symlinks=False)
    except OSError:
        raise ManagedWritePolicyError(
            "managed write target could not be resolved safely"
        ) from None
    return ManagedWriteScope(
        root=candidate_root,
        relative=PurePosixPath(relative),
        root_identity=(stat_result.st_dev, stat_result.st_ino),
    )


def hard_control_plane_command_reason(
    command_parts: list[str], raw_command: str, *, extra_roots: Iterable[Path] = ()
) -> str | None:
    for part in command_parts:
        unwrapped = unwrapped_command(part) or part
        try:
            tokens = split_bash_tokens(unwrapped)
        except ValueError:
            continue
        if not tokens:
            continue
        command = tokens[0].rsplit("/", 1)[-1]
        args = tokens[1:]
        if command == "git" and _git_control_mutation(args):
            return (
                f"Git control-plane mutation is host-owned and cannot run from "
                f"model bash: {part!r}"
            )
        if command not in _PATH_MUTATORS:
            continue
        for token in args:
            if token.startswith("-"):
                continue
            if reason := protected_model_write_reason(token, extra_roots=extra_roots):
                return reason

    folded = raw_command.casefold()
    if any(
        marker in folded
        for marker in (".git/worktrees", ".git/refs", ".git/logs", ".git/packed-refs")
    ) and any(mutator in folded for mutator in _PATH_MUTATORS):
        return "Git control metadata is host-owned and cannot be mutated from bash"
    return None


def _git_control_mutation(args: list[str]) -> bool:
    if not args:
        return False
    index = 0
    while index < len(args) and args[index].startswith("-"):
        option = args[index]
        index += 2 if option in {"-C", "--git-dir", "--work-tree"} else 1
    if index >= len(args):
        return False
    subcommand = args[index]
    subcommand_args = args[index + 1 :]
    match subcommand:
        case "update-ref":
            mutating = True
        case "worktree":
            mutating = not subcommand_args or subcommand_args[0] != "list"
        case "reset":
            mutating = "--hard" in subcommand_args
        case "clean":
            mutating = any(
                arg.startswith("-") and "f" in arg for arg in subcommand_args
            )
        case "branch" | "tag":
            mutating = any(arg in {"-d", "-D", "--delete"} for arg in subcommand_args)
        case "reflog":
            mutating = bool(
                subcommand_args and subcommand_args[0] in {"delete", "expire"}
            )
        case _:
            mutating = False
    return mutating


__all__ = [
    "ManagedWritePolicyError",
    "ManagedWriteScope",
    "hard_control_plane_command_reason",
    "managed_candidate_write_reason",
    "managed_candidate_write_scope",
    "model_protected_roots",
    "protected_model_write_reason",
    "verification_protected_roots",
]
