from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from vibe.core.types import AssistantEvent, ReasoningEvent, UserMessageEvent
from vibe.core.workflows.citations import CitationFailure
from vibe.core.workflows.runtime import WorkflowRuntime

pytestmark = pytest.mark.asyncio


@dataclass
class MockStats:
    session_prompt_tokens: int = 1000
    session_completion_tokens: int = 500


@dataclass
class MockAgentLoop:
    response_text: str = "mock response"
    stats: MockStats = field(default_factory=MockStats)

    async def act(
        self, prompt: str, *, response_format: Any = None
    ) -> AsyncGenerator[AssistantEvent | ReasoningEvent | UserMessageEvent, None]:
        yield UserMessageEvent(content=prompt, message_id="u1")
        yield ReasoningEvent(content="thinking", message_id="r1")
        yield AssistantEvent(content=self.response_text, message_id="a1")


def _factory(response_text: str) -> Any:
    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        return MockAgentLoop(response_text=response_text)

    return factory


FINDINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "evidence": {"type": "string"},
                },
                "required": ["file"],
            },
        }
    },
    "required": ["findings"],
}

CITATIONS = {
    "items_path": "findings",
    "path_field": "file",
    "line_field": "line",
    "snippet_field": "evidence",
}


@pytest.fixture
def repo_files(tmp_working_directory: Path) -> Path:
    (tmp_working_directory / "auth.py").write_text("def login():\n    return token\n")
    (tmp_working_directory / "models.py").write_text(
        "class User:\n    pass\n\nclass Admin:\n    pass\n"
    )
    return tmp_working_directory


async def test_live_agent_all_citations_verify(repo_files: Path) -> None:
    response = (
        '{"findings": ['
        '{"file": "auth.py", "line": 1, "evidence": "def login()"},'
        '{"file": "models.py", "line": 2, "evidence": "class User"}'
        "]}"
    )
    rt = WorkflowRuntime(agent_loop_factory=_factory(response))
    result = await rt.spawn_agent("audit", schema=FINDINGS_SCHEMA, citations=CITATIONS)
    assert isinstance(result, dict)
    assert len(result["findings"]) == 2
    assert result["citation_report"]["passed"] is True
    assert result["citation_report"]["items_verified"] == 2


async def test_live_agent_drops_bad_citation_keeps_good(repo_files: Path) -> None:
    response = (
        '{"findings": ['
        '{"file": "auth.py", "line": 1, "evidence": "def login()"},'
        '{"file": "fabricated.py", "line": 99, "evidence": "does not exist"},'
        '{"file": "models.py", "line": 2, "evidence": "class User"}'
        "]}"
    )
    rt = WorkflowRuntime(agent_loop_factory=_factory(response))
    result = await rt.spawn_agent("audit", schema=FINDINGS_SCHEMA, citations=CITATIONS)
    assert isinstance(result, dict)
    assert len(result["findings"]) == 2
    assert result["findings"][0]["file"] == "auth.py"
    assert result["findings"][1]["file"] == "models.py"
    assert result["citation_report"]["passed"] is False
    assert result["citation_report"]["items_verified"] == 2
    assert result["citation_report"]["dropped_indices"] == [1]


async def test_live_agent_strict_returns_citation_failure(repo_files: Path) -> None:
    response = (
        '{"findings": ['
        '{"file": "auth.py", "line": 1},'
        '{"file": "fabricated.py", "line": 99}'
        "]}"
    )
    strict_citations = {**CITATIONS, "strict": True}
    rt = WorkflowRuntime(agent_loop_factory=_factory(response))
    result = await rt.spawn_agent(
        "audit", schema=FINDINGS_SCHEMA, citations=strict_citations
    )
    assert isinstance(result, CitationFailure)
    assert not result
    assert result.report["items_checked"] == 2


async def test_live_agent_no_citations_passes_through(repo_files: Path) -> None:
    response = '{"findings": [{"file": "auth.py", "line": 1}]}'
    rt = WorkflowRuntime(agent_loop_factory=_factory(response))
    result = await rt.spawn_agent("audit", schema=FINDINGS_SCHEMA)
    assert isinstance(result, dict)
    assert result["findings"] == [{"file": "auth.py", "line": 1}]
    assert "citation_report" not in result


async def test_fabricated_citation_dropped_real_one_kept(repo_files: Path) -> None:
    response = (
        '{"findings": ['
        '{"file": "auth.py", "line": 1, "evidence": "def login()"},'
        '{"file": "totally_made_up.py", "line": 1, "evidence": "fake"}'
        "]}"
    )
    rt = WorkflowRuntime(agent_loop_factory=_factory(response))
    result = await rt.spawn_agent("audit", schema=FINDINGS_SCHEMA, citations=CITATIONS)
    assert isinstance(result, dict)
    files = [f["file"] for f in result["findings"]]
    assert "auth.py" in files
    assert "totally_made_up.py" not in files
