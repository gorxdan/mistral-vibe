from __future__ import annotations

from vibe.core.agents.models import (
    BUILTIN_AGENTS,
    AgentProfile,
    AgentSafety,
    AgentType,
    BuiltinAgentName,
    profile_requires_isolation,
)


def test_write_profiles_isolate() -> None:
    assert profile_requires_isolation(BUILTIN_AGENTS[BuiltinAgentName.WORKER])
    assert profile_requires_isolation(BUILTIN_AGENTS[BuiltinAgentName.AUTO_APPROVE])
    assert profile_requires_isolation(BUILTIN_AGENTS[BuiltinAgentName.EDITOR])


def test_read_only_profiles_stay_in_process() -> None:
    assert not profile_requires_isolation(BUILTIN_AGENTS[BuiltinAgentName.EXPLORE])
    assert not profile_requires_isolation(BUILTIN_AGENTS[BuiltinAgentName.RESEARCH])
    assert not profile_requires_isolation(BUILTIN_AGENTS[BuiltinAgentName.PLANNER])


def test_read_jailed_bash_profiles_stay_in_process() -> None:
    # reviewer/debugger/security carry bash but ship a denylist jail that
    # hard-NEVERs rm/git reset/etc. Their bash is read-only by enforcement,
    # not convention, so they do not need isolation.
    assert not profile_requires_isolation(BUILTIN_AGENTS[BuiltinAgentName.REVIEWER])
    assert not profile_requires_isolation(BUILTIN_AGENTS[BuiltinAgentName.DEBUGGER])
    assert not profile_requires_isolation(BUILTIN_AGENTS[BuiltinAgentName.SECURITY])


def _profile(overrides: dict | None = None) -> AgentProfile:
    return AgentProfile(
        name="x",
        display_name="x",
        description="x",
        safety=AgentSafety.NEUTRAL,
        agent_type=AgentType.SUBAGENT,
        overrides=overrides or {},
    )


def test_unjailed_bash_isolates() -> None:
    # A profile with bash but no denylist jail can run rm -rf -> isolate.
    p = _profile({"enabled_tools": ["read", "grep", "bash"]})
    assert profile_requires_isolation(p)


def test_write_tool_in_allowlist_isolates() -> None:
    p = _profile({"enabled_tools": ["read", "grep", "write_file"]})
    assert profile_requires_isolation(p)
    p_edit = _profile({"enabled_tools": ["read", "grep", "edit"]})
    assert profile_requires_isolation(p_edit)


def test_read_only_allowlist_stays_in_process() -> None:
    p = _profile({"enabled_tools": ["read", "grep", "web_search", "web_fetch"]})
    assert not profile_requires_isolation(p)
