from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def dedup_paths(paths: Iterable[Path]) -> list[Path]:
    """Resolve and dedup paths, preserving first-occurrence order."""
    resolved = [p.resolve() for p in paths]
    return [p for i, p in enumerate(resolved) if p not in resolved[:i]]


def resolved_within(path: Path, root: Path) -> bool:
    """True if ``path`` stays within ``root`` after resolving symlinks.

    Guards project-sourced artifact discovery (AGENTS.md targets, skills/tools
    dirs, hooks) against symlinks that escape the trusted root — a planted
    ``AGENTS.md`` or ``.vibe/skills`` symlink pointing outside would otherwise
    pull external content into the system prompt or harness config. Legitimate
    in-tree symlinks (target still under root) pass.
    """
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (ValueError, OSError):
        return False


_VIBE_DIR = Path(".vibe")
_TOOLS_SUBDIR = _VIBE_DIR / "tools"
_VIBE_SKILLS_SUBDIR = _VIBE_DIR / "skills"
_AGENTS_SUBDIR = _VIBE_DIR / "agents"
_WORKFLOWS_SUBDIR = _VIBE_DIR / "workflows"
_AGENTS_DIR = Path(".agents")
_AGENTS_SKILLS_SUBDIR = _AGENTS_DIR / "skills"


@dataclass(frozen=True)
class LocalConfigDirs:
    """Local config directories discovered at a project root."""

    config_dirs: tuple[Path, ...] = ()
    tools: tuple[Path, ...] = ()
    skills: tuple[Path, ...] = ()
    agents: tuple[Path, ...] = ()
    workflows: tuple[Path, ...] = ()

    def __or__(self, other: LocalConfigDirs) -> LocalConfigDirs:
        return LocalConfigDirs(
            config_dirs=tuple(dedup_paths([*self.config_dirs, *other.config_dirs])),
            tools=tuple(dedup_paths([*self.tools, *other.tools])),
            skills=tuple(dedup_paths([*self.skills, *other.skills])),
            agents=tuple(dedup_paths([*self.agents, *other.agents])),
            workflows=tuple(dedup_paths([*self.workflows, *other.workflows])),
        )


def find_local_config_dirs(root: Path) -> LocalConfigDirs:
    """Inspect *root* for ``.vibe/`` and ``.agents/`` config directories.

    Only the root itself is examined — no recursion into subdirectories.
    """
    resolved = root.resolve()
    config_dirs: list[Path] = []
    tools: list[Path] = []
    skills: list[Path] = []
    agents: list[Path] = []
    workflows: list[Path] = []

    vibe_dir = resolved / _VIBE_DIR
    if _safe_is_dir(vibe_dir):
        has_content = False
        if _safe_is_dir(candidate := resolved / _TOOLS_SUBDIR) and resolved_within(
            candidate, resolved
        ):
            tools.append(candidate)
            has_content = True
        if _safe_is_dir(
            candidate := resolved / _VIBE_SKILLS_SUBDIR
        ) and resolved_within(candidate, resolved):
            skills.append(candidate)
            has_content = True
        if _safe_is_dir(candidate := resolved / _AGENTS_SUBDIR) and resolved_within(
            candidate, resolved
        ):
            agents.append(candidate)
            has_content = True
        if _safe_is_dir(candidate := resolved / _WORKFLOWS_SUBDIR) and resolved_within(
            candidate, resolved
        ):
            workflows.append(candidate)
            has_content = True
        prompts_dir = vibe_dir / "prompts"
        config_candidate = vibe_dir / "config.toml"
        if (
            has_content
            or (_safe_is_dir(prompts_dir) and resolved_within(prompts_dir, resolved))
            or (
                _safe_is_file(config_candidate)
                and resolved_within(config_candidate, resolved)
            )
        ):
            config_dirs.append(vibe_dir)

    agents_dir = resolved / _AGENTS_DIR
    if (
        _safe_is_dir(agents_dir)
        and _safe_is_dir(candidate := resolved / _AGENTS_SKILLS_SUBDIR)
        and resolved_within(candidate, resolved)
    ):
        skills.append(candidate)
        config_dirs.append(agents_dir)

    return LocalConfigDirs(
        config_dirs=tuple(config_dirs),
        tools=tuple(tools),
        skills=tuple(skills),
        agents=tuple(agents),
        workflows=tuple(workflows),
    )
