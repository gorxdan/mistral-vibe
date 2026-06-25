from __future__ import annotations

from tests.conftest import build_test_vibe_config
from vibe.core.agents.models import BUILTIN_AGENTS, AgentType, BuiltinAgentName
from vibe.core.prompts import load_system_prompt
from vibe.core.tools.manager import ToolManager

_NEW = ("debugger", "planner", "security", "editor")


def _registry_tool_names() -> set[str]:
    cfg = build_test_vibe_config()
    tm = ToolManager(lambda: cfg)
    return set(tm.available_tools.keys())


def test_new_agents_enabled_tools_resolve() -> None:
    names = _registry_tool_names() | {"lsp"}  # lsp is opt-in but a valid builtin
    assert {"read", "grep", "bash", "write_file", "edit"} <= names, names
    for n in _NEW:
        et = BUILTIN_AGENTS[BuiltinAgentName(n)].overrides.get("enabled_tools", [])
        missing = [t for t in et if t not in names]
        assert not missing, f"{n} references unknown tools: {missing}"


def test_new_agents_are_spawnable_subagents() -> None:
    # The task tool only spawns AgentType.SUBAGENT profiles.
    for n in _NEW:
        prof = BUILTIN_AGENTS[BuiltinAgentName(n)]
        assert prof.agent_type == AgentType.SUBAGENT


def test_new_agent_prompts_load_and_are_nonempty() -> None:
    for n in _NEW:
        sp = load_system_prompt(n)
        assert sp.strip()
        # Every methodology prompt forbids the same anti-patterns (no fluff).
        assert "Never:" in sp or "Never " in sp


def test_editor_is_write_capable_others_read_only() -> None:
    editor = BUILTIN_AGENTS[BuiltinAgentName.EDITOR].overrides["enabled_tools"]
    assert "write_file" in editor and "edit" in editor and "bash" not in editor
    for n in ("debugger", "planner", "security"):
        et = BUILTIN_AGENTS[BuiltinAgentName(n)].overrides["enabled_tools"]
        assert "write_file" not in et and "edit" not in et
