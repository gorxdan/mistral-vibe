from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from vibe.core.agents.models import BUILTIN_AGENTS, AgentType
from vibe.core.workflows.manager import WorkflowManager
from vibe.core.workflows.runtime import WorkflowRuntime
from vibe.core.workflows.security import validate_script

pytestmark = pytest.mark.asyncio


def _make_config() -> MagicMock:
    config = MagicMock()
    config.workflow_paths = []
    return config


def _source() -> str:
    mgr = WorkflowManager(lambda: _make_config())
    info = mgr.get_workflow("adversarial-review")
    assert info is not None, "adversarial-review workflow not discovered"
    return info.source


@dataclass
class _Stats:
    session_prompt_tokens: int = 10
    session_completion_tokens: int = 5


def _factory(response_text: str) -> Any:
    from vibe.core.types import AssistantEvent

    @dataclass
    class _Loop:
        stats: _Stats = field(default_factory=_Stats)

        async def act(
            self, prompt: str, *, response_format: Any = None
        ) -> AsyncGenerator[AssistantEvent, None]:
            yield AssistantEvent(content=response_text, message_id="a1")

    def factory(
        prompt: str, *, agent: str, parent_context: Any | None = None
    ) -> Any:
        return _Loop()

    return factory


def _superset(verdict: str, files: list[str] | None = None) -> str:
    """One JSON object that validates against the scope, findings and verdict
    schemas at once (each schema only requires its own keys; extras allowed)."""
    return json.dumps({
        "files": ["a.py"] if files is None else files,
        "findings": [
            {
                "title": "t",
                "severity": "high",
                "file": "a.py",
                "evidence": "e",
                "suggested_fix": "f",
            }
        ],
        "verdict": verdict,
        "reasoning": "because",
        "corrected_severity": "high",
    })


def test_adversarial_review_discovered() -> None:
    mgr = WorkflowManager(lambda: _make_config())
    assert "adversarial-review" in mgr.get_workflow_names()
    info = mgr.get_workflow("adversarial-review")
    assert info is not None
    assert info.is_bundled is True
    assert "async def main" in info.source
    assert info.description


def test_adversarial_review_sandbox_clean() -> None:
    violations = validate_script(_source())
    assert not violations, f"sandbox violations: {[str(v) for v in violations]}"


def test_reviewer_agent_is_bash_capable_subagent() -> None:
    rev = BUILTIN_AGENTS["reviewer"]
    assert rev.agent_type == AgentType.SUBAGENT
    assert rev.overrides["enabled_tools"] == ["read", "grep", "bash"]


async def test_review_runs_find_verify_synthesize() -> None:
    rt = WorkflowRuntime(
        agent_loop_factory=_factory(_superset("refuted")),
        max_agents=100,
        budget_total=1_000_000,
    )
    result = await rt.run(_source(), args="HEAD")
    assert result.run.status.value == "completed"
    rv = result.return_value
    # 4 lenses each return 1 finding -> 4 candidates; all refuted -> 0 confirmed.
    assert rv["candidates"] == 4
    assert rv["confirmed"] == 0
    # All four phases ran in order (find -> verify -> synthesize).
    phase_names = [p.name for p in result.run.phases]
    assert phase_names == ["Scope", "Review", "Verify", "Synthesize"]
    # Each lens spawned its own finder.
    review_phase = next(p for p in result.run.phases if p.name == "Review")
    assert len(review_phase.agent_results) == 4


async def test_review_confirms_real_findings() -> None:
    rt = WorkflowRuntime(
        agent_loop_factory=_factory(_superset("confirmed")),
        max_agents=100,
        budget_total=1_000_000,
    )
    result = await rt.run(_source(), args="HEAD")
    rv = result.return_value
    assert rv["candidates"] == 4
    assert rv["confirmed"] == 4


async def test_review_short_circuits_on_no_changed_files() -> None:
    # scope returns an empty file list -> no finders/verifiers spawned.
    rt = WorkflowRuntime(
        agent_loop_factory=_factory(json.dumps({"files": []})),
        max_agents=100,
        budget_total=1_000_000,
    )
    result = await rt.run(_source(), args="HEAD")
    assert result.run.status.value == "completed"
    assert result.return_value["candidates"] == 0
    assert rt._agent_count == 1  # only the scope agent ran
