from __future__ import annotations

from datetime import date
import sys

import pytest

from tests.conftest import build_test_vibe_config
from vibe.core.agents import AgentManager
from vibe.core.scratchpad import init_scratchpad
from vibe.core.skills.manager import SkillManager
from vibe.core.system_prompt import get_universal_system_prompt
from vibe.core.tools.manager import ToolManager


def test_get_universal_system_prompt_includes_windows_prompt_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("COMSPEC", "C:\\Windows\\System32\\cmd.exe")

    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
    )
    tool_manager = ToolManager(lambda: config)
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(lambda: config)

    prompt = get_universal_system_prompt(
        tool_manager, config, skill_manager, agent_manager
    )

    assert "You are Chaton, a super useful programming assistant." in prompt
    assert (
        "The operating system is Windows with shell `C:\\Windows\\System32\\cmd.exe`"
        in prompt
    )
    assert "DO NOT use Unix commands like `ls`, `grep`, `cat`" in prompt
    assert "Use: `dir` (Windows) for directory listings" in prompt
    assert "Use: backslashes (\\\\) for paths" in prompt
    assert "Check command availability with: `where command` (Windows)" in prompt
    assert "Script shebang: Not applicable on Windows" in prompt


def test_orchestration_section_present_in_normal_mode() -> None:
    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
    )
    assert config.effort_mode == "normal"
    tool_manager = ToolManager(lambda: config)
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(lambda: config)

    prompt = get_universal_system_prompt(
        tool_manager, config, skill_manager, agent_manager
    )

    # Normal mode: the host is directed to orchestrate subagents (no workflows).
    assert "# Available Subagents" in prompt
    assert "## Orchestrating Subagents" in prompt
    assert "Orchestration is a default skill" in prompt
    assert "`task`" in prompt
    # The profile→use map is present.
    for profile in ("`explore`", "`research`", "`reviewer`", "`debugger`"):
        assert profile in prompt
    # The le-chaton workflow-orchestration SECTION stays le-chaton-only — it
    # must NOT leak into the normal prompt. (The launch_workflow tool's own
    # description is always present and legitimately mentions "workflow"; we
    # only assert the le-chaton section itself is absent.)
    assert "Le Chaton Mode" not in prompt
    assert "le chaton effort mode" not in prompt


def test_debugger_subagent_registered_with_systematic_prompt() -> None:
    from vibe.core.agents.models import BUILTIN_AGENTS, AgentType, BuiltinAgentName
    from vibe.core.prompts import load_system_prompt

    debugger = BUILTIN_AGENTS[BuiltinAgentName.DEBUGGER]
    assert debugger.agent_type == AgentType.SUBAGENT
    assert debugger.overrides["enabled_tools"] == ["read", "grep", "bash"]
    assert debugger.overrides["system_prompt_id"] == "debugger"

    # The dedicated prompt embeds the canonical systematic-debugging skill
    # (no skill-tool dependency — the subagent can't load skills).
    sp = load_system_prompt("debugger")
    assert "systematic-debugging" in sp.lower()
    # The Iron Law + all four phases (Pattern analysis was the gap in v1).
    assert "NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST" in sp
    for phase in ("Phase 1", "Phase 2", "Phase 3", "Phase 4"):
        assert phase in sp
    assert "Pattern analysis" in sp
    # The 3-fixes → question-architecture escalation, and the return format.
    assert "architecture" in sp.lower()
    assert "ROOT CAUSE:" in sp


def test_planner_security_editor_registered() -> None:
    from vibe.core.agents.models import BUILTIN_AGENTS, AgentType, BuiltinAgentName
    from vibe.core.prompts import load_system_prompt

    expected = {
        "planner": (["read", "grep"], "Clarify the goal"),
        "security": (["read", "grep", "bash"], "threat-model"),
        "editor": (["read", "grep", "write_file", "edit"], "Read before editing"),
    }
    for name, (tools, marker) in expected.items():
        prof = BUILTIN_AGENTS[BuiltinAgentName(name)]
        assert prof.agent_type == AgentType.SUBAGENT
        assert prof.overrides["enabled_tools"] == tools
        assert prof.overrides["system_prompt_id"] == name
        assert marker in load_system_prompt(name)

    # Security prompt is defensive — must forbid weaponization.
    sec = load_system_prompt("security")
    assert "DEFENSIVE" in sec and "do NOT write exploits" in sec
    # Editor prompt is honest about its write reality: worktree = git isolation,
    # NOT a security sandbox; plain-task writes are approval-gated/skipped.
    ed = load_system_prompt("editor")
    assert "isolation='worktree'" in ed
    assert "not a security sandbox" in ed
    assert "approval-gated" in ed


def test_orchestration_map_includes_planner_security_not_editor() -> None:
    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
    )
    prompt = get_universal_system_prompt(
        ToolManager(lambda: config),
        config,
        SkillManager(lambda: config),
        AgentManager(lambda: config),
    )
    # planner + security are delegable read-only investigators → in the map.
    assert "- `planner` —" in prompt
    assert "- `security` —" in prompt
    # editor is workflow-only (writes skipped in a plain task) → NOT in the map,
    # but still listed in the available-subagents inventory.
    assert "- `editor` —" not in prompt
    assert "**editor**" in prompt


def test_debugger_listed_in_available_subagents() -> None:
    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
    )
    tool_manager = ToolManager(lambda: config)
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(lambda: config)
    prompt = get_universal_system_prompt(
        tool_manager, config, skill_manager, agent_manager
    )
    assert "**debugger**" in prompt  # appears in the # Available Subagents list


def test_scratchpad_section_included_when_passed() -> None:
    sp = init_scratchpad("test-session")
    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
    )
    tool_manager = ToolManager(lambda: config)
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(lambda: config)

    prompt = get_universal_system_prompt(
        tool_manager, config, skill_manager, agent_manager, scratchpad_dir=sp
    )

    assert "# Scratchpad Directory" in prompt
    assert sp is not None
    assert str(sp) in prompt


def test_scratchpad_section_absent_when_not_passed() -> None:
    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
    )
    tool_manager = ToolManager(lambda: config)
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(lambda: config)

    prompt = get_universal_system_prompt(
        tool_manager, config, skill_manager, agent_manager
    )

    assert "Scratchpad Directory" not in prompt


def test_headless_section_included_when_enabled() -> None:
    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
    )
    tool_manager = ToolManager(lambda: config)
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(lambda: config)

    prompt = get_universal_system_prompt(
        tool_manager, config, skill_manager, agent_manager, headless=True
    )

    assert "# Headless Mode" in prompt
    assert "no human is available to respond" in prompt


def test_headless_section_absent_by_default() -> None:
    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
    )
    tool_manager = ToolManager(lambda: config)
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(lambda: config)

    prompt = get_universal_system_prompt(
        tool_manager, config, skill_manager, agent_manager
    )

    assert "Headless Mode" not in prompt


def test_current_date_placeholder_substituted_in_prompt() -> None:
    config = build_test_vibe_config(
        system_prompt_id="cli",
        include_project_context=False,
        include_prompt_detail=False,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
    )
    tool_manager = ToolManager(lambda: config)
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(lambda: config)

    prompt = get_universal_system_prompt(
        tool_manager, config, skill_manager, agent_manager
    )

    today = date.today()
    expected = f"Today's date is {today.isoformat()} ({today.strftime('%A')})."
    assert expected in prompt
    assert "$current_date" not in prompt
