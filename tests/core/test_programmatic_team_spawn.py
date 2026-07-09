from __future__ import annotations

import json

from tests.conftest import build_test_vibe_config
from tests.mock.mock_backend_factory import mock_backend_factory
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.agents.models import BuiltinAgentName
from vibe.core.programmatic import ProgrammaticOptions, run_programmatic
from vibe.core.types import Backend, FunctionCall, OutputFormat, ToolCall


def test_programmatic_team_spawn_tool_creates_active_team(
    monkeypatch, tmp_path, telemetry_events: list[dict]
) -> None:
    monkeypatch.setenv("VIBE_HOME", str(tmp_path / "vibe-home"))

    spawned: list[tuple[str, str, str, int, bool]] = []

    async def fake_spawn(
        self,
        name: str,
        prompt: str,
        *,
        agent: str,
        max_turns: int,
        worker: bool = False,
    ):
        spawned.append((name, prompt, agent, max_turns, worker))
        return name

    monkeypatch.setattr(
        "vibe.core.teams.manager.TeamManager.spawn_teammate", fake_spawn
    )
    tool_call = ToolCall(
        id="call_1",
        index=0,
        function=FunctionCall(
            name="team_spawn",
            arguments=json.dumps({
                "name": "reviewer",
                "prompt": "Review the perf changes.",
                "agent": "explore",
                "max_turns": 2,
            }),
        ),
    )

    with mock_backend_factory(
        Backend.MISTRAL,
        lambda provider, **kwargs: FakeBackend([
            [mock_llm_chunk(content="Spawning.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="Done.")],
        ]),
    ):
        config = build_test_vibe_config(
            enabled_tools=["team_spawn", "team_message"],
            include_config_reference=False,
            include_humanizer_guidance=False,
            include_model_info=False,
            include_prompt_detail=False,
        )
        run_programmatic(
            config=config,
            prompt="Spawn a reviewer teammate.",
            options=ProgrammaticOptions(
                output_format=OutputFormat.TEXT,
                agent_name=BuiltinAgentName.AUTO_APPROVE,
            ),
        )

    assert spawned == [("reviewer", "Review the perf changes.", "explore", 2, False)]
