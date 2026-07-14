from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from tests.conftest import build_test_vibe_config
from vibe.core.agents import AgentManager
from vibe.core.baseline_scaling import BaselineTier
from vibe.core.config import ModelConfig
from vibe.core.scratchpad import init_scratchpad
from vibe.core.skills.manager import SkillManager
from vibe.core.system_prompt import get_universal_system_prompt
from vibe.core.tools.manager import ToolManager


def test_get_universal_system_prompt_includes_windows_prompt_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mutating process-global sys.platform can poison platform-specific imports
    # in another test collected by the same xdist worker.
    monkeypatch.setattr("vibe.core.system_prompt.is_windows", lambda: True)
    monkeypatch.setattr(
        "vibe.core.system_prompt.get_platform_display_name", lambda: "Windows"
    )
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

    assert "You are Mistral Vibe, a super useful programming assistant." in prompt
    assert (
        "The operating system is Windows with shell `C:\\Windows\\System32\\cmd.exe`"
        in prompt
    )
    assert "DO NOT use Unix commands like `ls`, `grep`, `cat`" in prompt
    assert "Use: `dir` (Windows) for directory listings" in prompt
    assert "Use: backslashes (\\\\) for paths" in prompt
    assert "Check command availability with: `where command` (Windows)" in prompt
    assert "Script shebang: Not applicable on Windows" in prompt


def test_small_tier_drops_gated_sections_large_keeps_them() -> None:
    from vibe.core.baseline_scaling import BaselineTier

    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
        include_model_info=True,
        include_commit_signature=False,
        include_humanizer_guidance=True,
    )
    tool_manager = ToolManager(lambda: config)
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(lambda: config)

    def build(tier: BaselineTier) -> str:
        return get_universal_system_prompt(
            tool_manager, config, skill_manager, agent_manager, tier=tier
        )

    large = build(BaselineTier.LARGE)
    small = build(BaselineTier.SMALL)

    # LARGE (the default) keeps the orchestration prose; SMALL drops it.
    assert "## Orchestrating Subagents" in large
    assert "## Orchestrating Subagents" not in small
    # The subagents list itself stays (tool subset is tier-invariant).
    assert "# Available Subagents" in small
    assert "## Verification invariant" in small
    assert "freeze all intended edits and commits" in small
    assert "## Investigation invariant" in small
    assert "## Verification contract" not in small
    assert "## Investigation contract" not in small
    # SMALL is strictly smaller.
    assert len(small) < len(large)


def test_default_tier_is_large() -> None:
    from vibe.core.baseline_scaling import BaselineTier

    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
    )
    tm, sm, am = (
        ToolManager(lambda: config),
        SkillManager(lambda: config),
        AgentManager(lambda: config),
    )
    default = get_universal_system_prompt(tm, config, sm, am)
    large = get_universal_system_prompt(tm, config, sm, am, tier=BaselineTier.LARGE)
    assert default == large


def test_orchestration_section_present_in_normal_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "vibe.core.system_prompt.lsp_running_extensions", lambda: (".py",)
    )
    config = build_test_vibe_config(
        installed_components=["lsp"],
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
    assert "Local tools first, delegation second" in prompt
    assert "map files with `glob`" in prompt
    assert "symbols and callers with `lsp`" in prompt
    assert "Do not delegate trivia" in prompt
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


def test_orchestration_section_uses_grep_when_lsp_is_unavailable() -> None:
    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
    )
    tool_manager = ToolManager(lambda: config)
    prompt = get_universal_system_prompt(
        tool_manager, config, SkillManager(lambda: config), AgentManager(lambda: config)
    )

    assert "locate central symbols and callers with `grep`" in prompt
    assert "symbols and callers with `lsp`" not in prompt
    assert "`read`/`grep`/`glob` directly" in prompt


def test_le_chaton_requires_local_reconnaissance_before_workflows(monkeypatch) -> None:
    monkeypatch.setattr(
        "vibe.core.system_prompt.lsp_running_extensions", lambda: (".py",)
    )
    config = build_test_vibe_config(
        installed_components=["lsp"],
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
        effort_mode="le-chaton",
    )
    prompt = get_universal_system_prompt(
        ToolManager(lambda: config),
        config,
        SkillManager(lambda: config),
        AgentManager(lambda: config),
    )

    assert "## Le Chaton Mode" in prompt
    assert "Do not launch a workflow as the first repository-discovery step" in prompt
    assert "First use local `glob` and `lsp`" in prompt
    assert "A broad label such as 'analyze this repo' does not by itself" in prompt
    assert "File count alone is not a reason to delegate" in prompt
    # workflow path must carry the same narrow-brief steer as the task-tool prose
    assert "give each agent one question" in prompt
    assert "breadth comes from more agents, not bigger prompts" in prompt


def test_le_chaton_reconnaissance_uses_grep_without_lsp() -> None:
    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
        effort_mode="le-chaton",
    )
    prompt = get_universal_system_prompt(
        ToolManager(lambda: config),
        config,
        SkillManager(lambda: config),
        AgentManager(lambda: config),
    )

    assert "First use local `glob` and `grep`" in prompt


@pytest.mark.parametrize(
    "tier", [BaselineTier.SMALL, BaselineTier.MEDIUM, BaselineTier.LARGE]
)
def test_subagent_prompt_omits_host_only_orchestration_sections(
    tier: BaselineTier,
) -> None:
    config = build_test_vibe_config(
        effort_mode="le-chaton",
        include_model_info=True,
        models=[
            ModelConfig(name="one", alias="one", provider="mistral"),
            ModelConfig(name="two", alias="two", provider="mistral"),
        ],
    )
    prompt = get_universal_system_prompt(
        ToolManager(lambda: config, host=False),
        config,
        SkillManager(lambda: config),
        AgentManager(lambda: config),
        host_orchestration=False,
        tier=tier,
    )

    for marker in (
        "# Available Subagents",
        "## Orchestrating Subagents",
        "## Verification contract",
        "## Verification invariant",
        "## Investigation contract",
        "## Investigation invariant",
        "## Le Chaton Mode",
        "## Le Chaton orchestration invariant",
        "Models available for subagents",
    ):
        assert marker not in prompt
    assert "First use local `glob` and `lsp`" not in prompt


@pytest.mark.parametrize("tier_name", ["small", "medium"])
def test_le_chaton_orchestration_invariant_survives_compact_tiers(
    tier_name: str,
) -> None:
    from vibe.core.baseline_scaling import BaselineTier

    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
        effort_mode="le-chaton",
    )
    tool_manager = ToolManager(lambda: config)
    prompt = get_universal_system_prompt(
        tool_manager,
        config,
        SkillManager(lambda: config),
        AgentManager(lambda: config),
        tier=BaselineTier(tier_name),
    )

    assert "## Le Chaton orchestration invariant" in prompt
    assert "`work_strategy`" in prompt
    assert "before the first mutating tool" in prompt
    assert "reassess" in prompt.lower()
    assert "work_strategy" in tool_manager.available_tools


def test_large_le_chaton_prompt_has_adaptive_work_strategy_guidance() -> None:
    from vibe.core.baseline_scaling import BaselineTier

    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
        effort_mode="le-chaton",
    )
    tool_manager = ToolManager(lambda: config)
    prompt = get_universal_system_prompt(
        tool_manager,
        config,
        SkillManager(lambda: config),
        AgentManager(lambda: config),
        tier=BaselineTier.LARGE,
    )

    assert "## Le Chaton Mode" in prompt
    assert "`work_strategy`" in prompt
    assert "before the first mutating tool" in prompt
    assert "adaptive" in prompt.lower()
    assert "host remains hands-on" in prompt.lower()
    for route in ("`direct`", "`task`", "`workflow`", "`team`"):
        assert route in prompt
    for signal in (
        "localized",
        "sequentially coupled",
        "independent lanes",
        "adversarial review",
        "long-running",
        "capability unavailable",
    ):
        assert signal in prompt.lower()
    assert "work_strategy" in tool_manager.available_tools


def test_le_chaton_recovery_and_monitor_prose_moved_to_workflow_skill() -> None:
    from vibe.core.skills.builtins import BUILTIN_SKILLS

    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
        effort_mode="le-chaton",
    )
    prompt = get_universal_system_prompt(
        ToolManager(lambda: config),
        config,
        SkillManager(lambda: config),
        AgentManager(lambda: config),
    )

    # The recovery ladder and the /workflows key map moved to the skill.
    assert "Re-run that phase with `max_concurrency=1`" not in prompt
    assert "`Retry-After` (honored up to 60s)" not in prompt
    assert "x (stop), p (pause/resume)" not in prompt

    guide = BUILTIN_SKILLS["workflow-authoring"].prompt
    assert "Retries exhausted" in guide
    assert "Re-run that phase with `max_concurrency=1`" in guide
    assert "`Retry-After` (honored up to 60s)" in guide
    assert "x (stop), p (pause/resume)" in guide

    # The decision-time routing lines stay inline.
    assert "**Deferral (pick by intent):**" in prompt
    assert "**Don't poll.**" in prompt


def _routing_common() -> dict[str, object]:
    from vibe.core.config import ModelConfig

    # The autouse test config ships a single model; the routing note needs 2+.
    return {
        "system_prompt_id": "tests",
        "include_project_context": False,
        "include_prompt_detail": True,
        "include_commit_signature": False,
        "include_humanizer_guidance": False,
        "models": [
            ModelConfig(name="model-a", provider="mistral", alias="alpha"),
            ModelConfig(name="model-b", provider="mistral", alias="beta"),
        ],
    }


def _prompt_for(config, **kwargs) -> str:
    return get_universal_system_prompt(
        ToolManager(lambda: config),
        config,
        SkillManager(lambda: config),
        AgentManager(lambda: config),
        **kwargs,
    )


def test_model_routing_list_present_with_task_tool_and_multiple_models() -> None:
    config = build_test_vibe_config(**_routing_common())
    assert len(config.models) > 1

    prompt = _prompt_for(config)

    assert "Models available for subagents" in prompt
    assert "`alpha` (mistral)" in prompt
    assert "`beta` (mistral)" in prompt


def test_model_routing_list_absent_when_profile_excludes_task() -> None:
    config = build_test_vibe_config(enabled_tools=["read", "grep"], **_routing_common())

    assert "Models available for subagents" not in _prompt_for(config)


def test_model_routing_list_absent_with_single_model() -> None:
    from vibe.core.config import ModelConfig

    common = {
        **_routing_common(),
        "models": [ModelConfig(name="only", provider="mistral", alias="only")],
    }
    config = build_test_vibe_config(**common)

    assert "Models available for subagents" not in _prompt_for(config)


def test_model_routing_list_tier_gated_at_new_emission_site() -> None:
    from vibe.core.baseline_scaling import BaselineTier

    config = build_test_vibe_config(**_routing_common())

    assert "Models available for subagents" in _prompt_for(
        config, tier=BaselineTier.MEDIUM
    )
    assert "Models available for subagents" not in _prompt_for(
        config, tier=BaselineTier.SMALL
    )


def test_model_routing_list_survives_prompt_detail_off() -> None:
    # Legacy emission site: include_model_info users with prompt detail off
    # kept the routing note before the task-prose relocation; still must.
    common = {**_routing_common(), "include_prompt_detail": False}
    config = build_test_vibe_config(**common)

    assert "Models available for subagents" in _prompt_for(config)


def test_skill_pointer_lines_stripped_without_skill_tool() -> None:
    config = build_test_vibe_config(
        enabled_tools=["read", "grep", "todo", "ask_user_question"],
        **{k: v for k, v in _routing_common().items() if k != "models"},
    )

    prompt = _prompt_for(config)

    assert "skill" not in ToolManager(lambda: config).manifest_tools
    assert "`tool-guides` skill." not in prompt


def test_skill_pointer_lines_present_with_skill_tool() -> None:
    config = build_test_vibe_config(
        enabled_tools=["read", "grep", "todo", "skill"],
        **{k: v for k, v in _routing_common().items() if k != "models"},
    )

    assert "`tool-guides` skill." in _prompt_for(config)


def test_debugger_subagent_registered_with_systematic_prompt() -> None:
    from vibe.core.agents.models import BUILTIN_AGENTS, AgentType, BuiltinAgentName
    from vibe.core.prompts import load_system_prompt

    debugger = BUILTIN_AGENTS[BuiltinAgentName.DEBUGGER]
    assert debugger.agent_type == AgentType.SUBAGENT
    assert debugger.overrides["enabled_tools"] == [
        "read",
        "grep",
        "glob",
        "lsp",
        "bash",
    ]
    assert debugger.overrides["system_prompt_id"] == "debugger"

    # The dedicated prompt embeds the systematic debugging methodology
    # (no skill-tool dependency — the subagent can't load skills).
    sp = load_system_prompt("debugger")
    assert "root cause" in sp.lower()
    # The Iron Law + all four phases (Pattern analysis was the gap in v1).
    assert "NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST" in sp
    for phase in ("Phase 1", "Phase 2", "Phase 3", "Phase 4"):
        assert phase in sp
    assert "Pattern" in sp
    # The 3-fixes → question-architecture escalation, and the return format.
    assert "architecture" in sp.lower()
    assert "ROOT CAUSE:" in sp


def test_planner_security_editor_registered() -> None:
    from vibe.core.agents.models import BUILTIN_AGENTS, AgentType, BuiltinAgentName
    from vibe.core.prompts import load_system_prompt

    expected = {
        "planner": (["read", "grep", "glob", "lsp"], "Clarify the goal"),
        "security": (["read", "grep", "glob", "lsp", "bash"], "threat-model"),
        "editor": (
            ["read", "grep", "glob", "lsp", "write_file", "edit"],
            "Read before editing",
        ),
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
    # NOT a security sandbox; under the task default it auto-isolates, so a
    # plain-task call auto-approves its writes (not skipped).
    ed = load_system_prompt("editor")
    assert "isolation='worktree'" in ed
    assert "not a security sandbox" in ed
    assert "auto-isolate" in ed.lower()

    # Grunt shares worker's write surface and auto-isolation behavior.
    gr = load_system_prompt("grunt")
    assert "auto-isolate" in gr.lower()
    assert "approval-gated and skipped" not in gr


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
    # planner + security are delegable read-only investigators → in the picker.
    assert "- `planner` —" in prompt
    assert "- `security` —" in prompt
    # editor is write-capable (auto-isolates), not an investigator → not in the
    # picker, but still listed in the available-subagents inventory.
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


def test_lsp_priority_section_requires_exposure_and_running_coverage(
    monkeypatch,
) -> None:
    common = {
        "system_prompt_id": "tests",
        "include_project_context": False,
        "include_prompt_detail": True,
        "include_model_info": False,
        "include_commit_signature": False,
        "include_humanizer_guidance": False,
    }
    heading = "## LSP is available — use it for symbol-level work"

    off = build_test_vibe_config(**common)
    prompt_off = get_universal_system_prompt(
        ToolManager(lambda: off),
        off,
        SkillManager(lambda: off),
        AgentManager(lambda: off),
    )
    assert heading not in prompt_off

    on = build_test_vibe_config(installed_components=["lsp"], **common)
    prompt_cold = get_universal_system_prompt(
        ToolManager(lambda: on), on, SkillManager(lambda: on), AgentManager(lambda: on)
    )
    assert heading not in prompt_cold

    monkeypatch.setattr(
        "vibe.core.system_prompt.lsp_running_extensions", lambda: (".py", ".pyi")
    )
    prompt_on = get_universal_system_prompt(
        ToolManager(lambda: on), on, SkillManager(lambda: on), AgentManager(lambda: on)
    )
    assert heading in prompt_on
    assert "running language-server route for `.py`, `.pyi`" in prompt_on
    assert "use only an advertised operation" in prompt_on
    # Trigger→action pairs: the question names the lsp operation, not emphasis.
    assert "who calls X / what does X call" in prompt_on
    assert "where is X defined / what type is X" in prompt_on
    assert "find_references" in prompt_on
    # The emphasis layer that did not move usage is gone.
    assert "hard requirement, not a preference" not in prompt_on
    assert "MUST resolve symbols through `lsp`" not in prompt_on


def _config_reference_common() -> dict[str, object]:
    return {
        "system_prompt_id": "tests",
        "include_project_context": False,
        "include_prompt_detail": False,
        "include_model_info": False,
        "include_commit_signature": False,
        "include_humanizer_guidance": False,
    }


def test_config_reference_section_absent_by_default() -> None:
    config = build_test_vibe_config(**_config_reference_common())
    prompt = get_universal_system_prompt(
        ToolManager(lambda: config),
        config,
        SkillManager(lambda: config),
        AgentManager(lambda: config),
    )

    assert "## Configuring Vibe (quick reference)" not in prompt


def test_config_reference_section_present_when_explicitly_enabled() -> None:
    config = build_test_vibe_config(
        include_config_reference=True, **_config_reference_common()
    )
    prompt = get_universal_system_prompt(
        ToolManager(lambda: config),
        config,
        SkillManager(lambda: config),
        AgentManager(lambda: config),
    )

    assert "## Configuring Vibe (quick reference)" in prompt
    # The handful of high-value facts an agent needs without loading a skill.
    assert "[[mcp_servers]]" in prompt
    assert "config.toml" in prompt
    assert "active_model" in prompt
    # Points back to the authoritative skill rather than duplicating all of it.
    assert "load the" in prompt and "skill" in prompt


def test_config_reference_section_absent_when_disabled() -> None:
    config = build_test_vibe_config(
        include_config_reference=False, **_config_reference_common()
    )
    prompt = get_universal_system_prompt(
        ToolManager(lambda: config),
        config,
        SkillManager(lambda: config),
        AgentManager(lambda: config),
    )

    assert "## Configuring Vibe (quick reference)" not in prompt


def test_verifier_subagent_registered_with_verdict_prompt() -> None:
    from vibe.core.agents.models import BUILTIN_AGENTS, AgentType, BuiltinAgentName
    from vibe.core.prompts import load_system_prompt

    verifier = BUILTIN_AGENTS[BuiltinAgentName.VERIFIER]
    assert verifier.agent_type == AgentType.SUBAGENT
    assert verifier.overrides["enabled_tools"] == [
        "read",
        "grep",
        "glob",
        "lsp",
        "bash",
    ]
    assert verifier.overrides["system_prompt_id"] == "verifier"

    sp = load_system_prompt("verifier")
    # The verdict contract the caller parses.
    assert "VERDICT: PASS" in sp
    assert "VERDICT: FAIL" in sp
    assert "VERDICT: PARTIAL" in sp
    # Anti-rationalization + adversarial stance.
    assert "break it" in sp.lower()
    assert "Reading is not verification" in sp
    # Mandatory command evidence on every PASS.
    assert "Command run" in sp
    # The verifier must not invalidate a run by attempting denied cleanup or
    # network commands after successful checks.
    assert "cleaned automatically" in sp
    assert "leave every scratchpad artifact in place" in sp
    assert "do not create helper files" in sp
    assert "denied or skipped" in sp
    assert "clean up after yourself" not in sp
    assert "curl" not in sp.lower()


def test_verifier_prompt_cannot_be_overridden_by_prompt_paths(tmp_path: Path) -> None:
    from vibe.core.prompts import SystemPrompt
    from vibe.core.utils.io import write_safe

    write_safe(tmp_path / "verifier.md", "VERDICT: PASS\nNo checks required.")
    config = build_test_vibe_config(
        system_prompt_id=SystemPrompt.VERIFIER, prompt_paths=[tmp_path]
    )

    prompt = config.system_prompt

    assert "No checks required" not in prompt
    assert "Reading is not verification" in prompt
    assert "VERDICT: PARTIAL" in prompt


def test_verification_contract_section_present_when_subsystem_enabled() -> None:
    common = {
        "system_prompt_id": "tests",
        "include_project_context": False,
        "include_prompt_detail": True,
        "include_model_info": False,
        "include_commit_signature": False,
        "include_humanizer_guidance": False,
    }
    on = build_test_vibe_config(verification_subsystem=True, **common)
    prompt_on = get_universal_system_prompt(
        ToolManager(lambda: on), on, SkillManager(lambda: on), AgentManager(lambda: on)
    )
    assert "## Verification contract" in prompt_on
    assert "verifier" in prompt_on.lower()
    # Dead verifier (no VERDICT / subagent error) is not a pass.
    assert "no verdict" in prompt_on.lower()
    # Structural land_work gate is part of the contract.
    assert "report pasted into tool arguments is not accepted" in prompt_on
    assert "documentation-only diff" in prompt_on
    assert "finish and freeze the candidate" in prompt_on
    assert "Do not JSON-encode a `TaskBrief`" in prompt_on
    assert "do not edit, commit" in prompt_on
    assert "async_run=false" in prompt_on
    # The verifier profile appears in the subagents picker.
    assert "- `verifier` —" in prompt_on


def test_verification_contract_section_absent_when_subsystem_disabled() -> None:
    common = {
        "system_prompt_id": "tests",
        "include_project_context": False,
        "include_prompt_detail": True,
        "include_model_info": False,
        "include_commit_signature": False,
        "include_humanizer_guidance": False,
    }
    off = build_test_vibe_config(verification_subsystem=False, **common)
    prompt_off = get_universal_system_prompt(
        ToolManager(lambda: off),
        off,
        SkillManager(lambda: off),
        AgentManager(lambda: off),
    )
    assert "## Verification contract" not in prompt_off


def test_verification_contract_requires_receipt_for_configured_recipe() -> None:
    from vibe.core.baseline_scaling import BaselineTier
    from vibe.core.config import (
        TrustedExecutionTopologyConfig,
        TrustedVerificationCheckConfig,
        TrustedVerificationRecipeConfig,
    )

    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
        trusted_verification_recipe=TrustedVerificationRecipeConfig(
            recipe_version="test-v1",
            task_brief="Implement the requested change",
            acceptance_contract="The focused checks pass",
            allowed_paths=("vibe/**", "tests/**"),
            checks=(
                TrustedVerificationCheckConfig(
                    name="focused",
                    argv=("/usr/bin/true",),
                    executable_sha256="0" * 64,
                    environment_attestation_path="/usr/bin/true",
                    environment_attestation_sha256="1" * 64,
                ),
            ),
            execution_topology=TrustedExecutionTopologyConfig(
                packet_id="I00-P01",
                packet_path="docs/design/packet.md",
                state="active",
                control_worktree="/maintenance/control",
                control_sha="1" * 40,
                candidate_worktree="/maintenance/candidate",
                candidate_branch="maintenance/i00-p01",
                baseline_sha="2" * 40,
                upstream_sha="3" * 40,
                evidence_workspace="/maintenance/evidence",
                run_id="i00-p01-run",
                runner_id="linux-runner",
            ),
        ),
    )
    tool_manager = ToolManager(lambda: config)
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(lambda: config)

    large = get_universal_system_prompt(
        tool_manager, config, skill_manager, agent_manager
    )
    small = get_universal_system_prompt(
        tool_manager, config, skill_manager, agent_manager, tier=BaselineTier.SMALL
    )

    for prompt in (large, small):
        assert "prebound" in prompt
        assert "verify_work" in prompt
        assert "READY_FOR_HOST_FREEZE:" in prompt
        assert "Bash sees the candidate read-only" in prompt
        assert "explicit `/**` pattern" in prompt
        assert "current durable receipt" in prompt
        assert "trivial" in prompt
        assert "waivers cannot replace" in prompt
        assert "Host-managed execution topology" in prompt
        assert "Packet `I00-P01` is host-bound in `active` state" in prompt
        assert "Never substitute a ref" in prompt


def test_investigation_contract_section_present_when_subsystem_enabled() -> None:
    common = {
        "system_prompt_id": "tests",
        "include_project_context": False,
        "include_prompt_detail": True,
        "include_model_info": False,
        "include_commit_signature": False,
        "include_humanizer_guidance": False,
    }
    on = build_test_vibe_config(investigation_subsystem=True, **common)
    prompt_on = get_universal_system_prompt(
        ToolManager(lambda: on), on, SkillManager(lambda: on), AgentManager(lambda: on)
    )
    assert "## Investigation contract" in prompt_on
    # The contract states both the rule and the exempt set (guidance, not gate).
    assert "reproduce" in prompt_on.lower()
    assert "Exempt" in prompt_on


def test_investigation_contract_section_absent_when_subsystem_disabled() -> None:
    common = {
        "system_prompt_id": "tests",
        "include_project_context": False,
        "include_prompt_detail": True,
        "include_model_info": False,
        "include_commit_signature": False,
        "include_humanizer_guidance": False,
    }
    off = build_test_vibe_config(investigation_subsystem=False, **common)
    prompt_off = get_universal_system_prompt(
        ToolManager(lambda: off),
        off,
        SkillManager(lambda: off),
        AgentManager(lambda: off),
    )
    assert "## Investigation contract" not in prompt_off


def test_raw_workflow_authoring_guide_remains_lazy() -> None:
    # The ~3.2k launch_workflow authoring guide must NOT be injected into every
    # system prompt; it loads on demand via the `workflow-authoring` skill. The
    # tool stays discoverable (concise stub + schema) and the full guide lives
    # in the skill body (single source = launch_workflow.md).
    from vibe.core.skills.builtins import BUILTIN_SKILLS

    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
        include_humanizer_guidance=False,
        effort_mode="le-chaton",
    )
    tool_manager = ToolManager(lambda: config)
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(lambda: config)

    prompt = get_universal_system_prompt(
        tool_manager, config, skill_manager, agent_manager
    )

    # The bulky authoring prose is gone from the always-on prompt...
    assert "## Local discovery comes first" not in prompt
    assert "dedup_by" not in prompt
    # Raw script authoring remains available as an advanced escape hatch, while
    # the common Le Chaton path is the adaptive strategy contract.
    assert "`work_strategy`" in prompt
    assert "launch_workflow" in tool_manager.available_tools

    # The full guide is preserved in the skill body.
    skill = BUILTIN_SKILLS["workflow-authoring"]
    assert "## Local discovery comes first" in skill.prompt
    assert "dedup_by" in skill.prompt
    # authoring guide must steer toward one-question-per-agent fan-out
    assert "One question per agent" in skill.prompt
    assert "fan out for breadth" in skill.prompt
