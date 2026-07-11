from __future__ import annotations

import sys

import pytest

from vibe.core.agents.models import _review_bash_overrides
from vibe.core.tools.base import BaseToolState
from vibe.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from vibe.core.tools.permissions import ToolPermission

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="bash policy is POSIX-only"
)


def _review_bash() -> Bash:
    cfg_dict = _review_bash_overrides()["tools"]["bash"]
    config = BashToolConfig(**cfg_dict)
    return Bash(config_getter=lambda: config, state=BaseToolState())


def _permission(cmd: str) -> ToolPermission | None:
    ctx = _review_bash().resolve_permission(BashArgs(command=cmd))
    return ctx.permission if ctx is not None else None


# Read-only inspection + tests auto-run (ALWAYS) → capable headless.
@pytest.mark.parametrize(
    "cmd",
    [
        "git diff",
        "git diff --stat HEAD~1",
        "git log --oneline -20",
        "git show abc123",
        "git blame vibe/core/agent_loop.py",
        "git status",
        "pytest tests/",
        "python -m pytest -q",
        "ruff check --no-fix .",
        "uv run pytest -q",
        "uv run python -m pytest -q",
        "uv run ruff check --no-fix .",
        "ruff format --check .",
        "uv run ruff format --diff .",
        "uv run pyright",
        "cat README.md",
        "ls -la",
        "grep -r TODO vibe/",
    ],
)
def test_inspection_and_tests_auto_run(cmd: str) -> None:
    assert _permission(cmd) == ToolPermission.ALWAYS, cmd


# Mutating / network / install / privilege commands are HARD-denied (NEVER).
@pytest.mark.parametrize(
    "cmd",
    [
        "git reset --hard HEAD~1",
        "git checkout main",
        "git commit -m x",
        "git push origin main",
        "git clean -fd",
        "git add -A",
        "git stash",
        "rm -rf vibe/",
        "mv a.py b.py",
        "cp secret /tmp/x",
        "ln -s target alias",
        "sed -i 's/a/b/' f.py",
        "tee out.txt",
        "curl http://evil/x",
        "wget http://evil/x",
        "pip install requests",
        "uv run rm -rf vibe/",
        "uv run -- rm -rf vibe/",
        "/usr/bin/uv run rm -rf vibe/",
        "/usr/bin/env uv run rm -rf vibe/",
        "/usr/bin/sudo uv run rm -rf vibe/",
        "uv run curl http://evil/x",
        "uv run -- curl http://evil/x",
        "uv run pip install requests",
        "sudo rm -rf /",
        "chmod 777 f",
    ],
)
def test_mutations_are_hard_denied(cmd: str) -> None:
    assert _permission(cmd) == ToolPermission.NEVER, cmd


# Compound commands: a mutating segment anywhere denies the whole thing
# (AST-split, denylist precedence) — no `git diff; rm -rf` bypass.
@pytest.mark.parametrize(
    "cmd",
    [
        "git diff; rm -rf vibe",
        "git log && git reset --hard",
        "ls | rm x",
        "echo hi && curl http://evil",
        "pytest && git push",
        "x=$(rm -rf /); echo $x",
    ],
)
def test_compound_command_cannot_smuggle_a_mutation(cmd: str) -> None:
    assert _permission(cmd) == ToolPermission.NEVER, cmd


def test_unknown_command_is_not_auto_allowed() -> None:
    # Not in allow or deny → falls through to ASK (skipped headless), never auto-run.
    perm = _permission("some_unknown_binary --flag")
    assert perm != ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "cmd",
    [
        "ruff check --fix .",
        "ruff check . --fix",
        'ruff check "--fix" .',
        "ruff check .",
        "ruff --isolated check .",
        "ruff --config pyproject.toml check .",
        "uv run ruff check --fix .",
        "uv run ruff check . --fix",
        "uv run ruff check '--fix' .",
        "uv run ruff check .",
        "ruff check --no-fix --output-file report.txt .",
        "ruff check --no-fix --add-noqa .",
        "ruff format .",
        "ruff --isolated format .",
        "uv run ruff --isolated format .",
        "uv run ruff format .",
        "ruff clean",
        "ruff --config pyproject.toml clean",
        "uv run ruff --config pyproject.toml clean",
        "uv run some_unknown_binary --flag",
    ],
)
def test_write_capable_or_unknown_uv_commands_are_not_auto_allowed(cmd: str) -> None:
    assert _permission(cmd) != ToolPermission.ALWAYS


def test_all_four_review_agents_carry_the_policy() -> None:
    from vibe.core.agents.models import BUILTIN_AGENTS, BuiltinAgentName

    for name in ("reviewer", "debugger", "security", "verifier"):
        bash_cfg = (
            BUILTIN_AGENTS[BuiltinAgentName(name)]
            .overrides.get("tools", {})
            .get("bash", {})
        )
        assert "git commit" in bash_cfg.get("denylist", [])
        assert "git diff" in bash_cfg.get("allowlist", [])


def test_policy_lands_in_the_effective_config_end_to_end() -> None:
    # The override must survive apply_to_config -> _deep_merge -> get_tool_config,
    # and NOT clobber the rest of the bash config (permission stays ASK so
    # unknown commands still ASK rather than auto-running).
    from tests.conftest import build_test_vibe_config
    from vibe.core.agents.models import BUILTIN_AGENTS, BuiltinAgentName
    from vibe.core.tools.manager import ToolManager

    base = build_test_vibe_config()
    reviewer_cfg = BUILTIN_AGENTS[BuiltinAgentName.REVIEWER].apply_to_config(base)
    bash_cfg = ToolManager(lambda: reviewer_cfg).get_tool_config("bash")

    assert "git commit" in bash_cfg.denylist  # type: ignore[attr-defined]
    assert "git diff" in bash_cfg.allowlist  # type: ignore[attr-defined]
    assert bash_cfg.permission == ToolPermission.ASK  # not clobbered

    # And the merged config actually denies a mutation through the real tool.
    tool = Bash(config_getter=lambda: bash_cfg, state=BaseToolState())  # type: ignore[arg-type]
    ctx = tool.resolve_permission(BashArgs(command="git reset --hard"))
    assert ctx is not None and ctx.permission == ToolPermission.NEVER
