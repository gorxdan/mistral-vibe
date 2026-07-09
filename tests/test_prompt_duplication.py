from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import build_test_vibe_config
from vibe.core.agents import AgentManager
from vibe.core.baseline_scaling import BaselineTier
from vibe.core.config import VibeConfig
from vibe.core.skills.manager import SkillManager
from vibe.core.system_prompt import get_universal_system_prompt
from vibe.core.tools.manager import ToolManager
from vibe.core.worktree.manager import WorktreeHandle, worktree_manager


def _config_with_project_context() -> VibeConfig:
    return build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=True,
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
    )


def _render_with_worktree(tier: BaselineTier | None = None) -> str:
    config = _config_with_project_context()
    handle = WorktreeHandle(
        original_repo_root=Path.cwd(),
        worktree_path=Path.cwd(),
        branch="vibe/test-dup-guard",
        create_head_sha="0" * 40,
    )
    worktree_manager._active = handle
    try:
        if tier is None:
            return get_universal_system_prompt(
                ToolManager(lambda: config),
                config,
                SkillManager(lambda: config),
                AgentManager(lambda: config),
            )
        return get_universal_system_prompt(
            ToolManager(lambda: config),
            config,
            SkillManager(lambda: config),
            AgentManager(lambda: config),
            tier=tier,
        )
    finally:
        worktree_manager._active = None


def test_worktree_isolation_heading_not_duplicated() -> None:
    prompt = _render_with_worktree()
    headings = [
        line for line in prompt.splitlines() if line.startswith("## Worktree isolation")
    ]
    assert len(headings) == 1, (
        f"Expected exactly one '## Worktree isolation' heading, got {len(headings)}:\n"
        + "\n".join(headings)
    )


def test_sandbox_pid_warning_present_in_large_tier() -> None:
    from vibe.core.baseline_scaling import BaselineTier

    prompt = _render_with_worktree(tier=BaselineTier.LARGE)
    assert "process-liveness" in prompt
    assert "sandboxed process scan" in prompt


def test_sandbox_pid_warning_present_in_small_tier() -> None:
    from vibe.core.baseline_scaling import BaselineTier

    prompt = _render_with_worktree(tier=BaselineTier.SMALL)
    assert "process-liveness" in prompt
    assert "sandboxed process scan" in prompt


_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROMPTS_DIR = _REPO_ROOT / "vibe" / "core" / "prompts"
_TOOL_PROMPTS_DIR = _REPO_ROOT / "vibe" / "core" / "tools" / "builtins" / "prompts"
_SYSTEM_PROMPT_PY = _REPO_ROOT / "vibe" / "core" / "system_prompt.py"
_SKILLS_BUILTINS_DIR = _REPO_ROOT / "vibe" / "core" / "skills" / "builtins"


def _corpus_files() -> list[Path]:
    files = list(_PROMPTS_DIR.glob("*.md"))
    files.extend(_TOOL_PROMPTS_DIR.glob("*.md"))
    files.append(_SYSTEM_PROMPT_PY)
    files.extend(_SKILLS_BUILTINS_DIR.glob("*.py"))
    return [f for f in sorted(files) if f.exists()]


def _files_containing(phrase: str) -> set[Path]:
    return {f for f in _corpus_files() if phrase in f.read_text(encoding="utf-8")}


# Caps = current carrier count; bump only with a documented reason.
_RATCHET_CASES = [
    # recon: orchestration only (task.md slimmed to "Local tools first").
    pytest.param(
        "establish a baseline", 1, "recon-before-delegating", id="recon-triplet"
    ),
    # don't-poll: le-chaton + launch_workflow + workflow_results.
    pytest.param("Do not poll", 3, "workflow-dont-poll", id="dont-poll"),
]


@pytest.mark.parametrize("phrase,max_files,label", _RATCHET_CASES)
def test_known_duplicated_concepts_do_not_spread(
    phrase: str, max_files: int, label: str
) -> None:
    carriers = _files_containing(phrase)
    assert len(carriers) <= max_files, (
        f"'{phrase}' ({label}) spread to {len(carriers)} files (cap {max_files}). "
        f"New carrier(s): {carriers}. "
        "Point the new site at the canonical home, or bump the cap here with a reason."
    )
