from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel
import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.agent_loop import AgentLoop
from vibe.core.agents.models import BuiltinAgentName
from vibe.core.config import ModelConfig, VibeConfig
from vibe.core.orchestration import OrchestrationCapabilities, OrchestrationDecision
from vibe.core.tools.background import BackgroundRegistry
from vibe.core.tools.base import BaseToolConfig, ToolPermission
from vibe.core.types import (
    ApprovalResponse,
    AssistantEvent,
    BaseEvent,
    FunctionCall,
    Role,
    ToolCall,
    ToolResultEvent,
)
from vibe.core.utils.io import read_safe, write_safe


def _tool_call(call_id: str, name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(name=name, arguments=json.dumps(arguments)),
    )


def _file_tools_config(
    *, effort_mode: str, edit_permission: ToolPermission = ToolPermission.ALWAYS
) -> VibeConfig:
    return build_test_vibe_config(
        effort_mode=effort_mode,
        enabled_tools=["read", "edit", "work_strategy"],
        tools={
            "read": BaseToolConfig(permission=ToolPermission.ALWAYS),
            "edit": BaseToolConfig(permission=edit_permission),
            "work_strategy": BaseToolConfig(permission=ToolPermission.ALWAYS),
        },
    )


async def _act(loop: AgentLoop, prompt: str) -> list[BaseEvent]:
    return [event async for event in loop.act(prompt)]


@pytest.mark.asyncio
async def test_le_chaton_blocks_effectful_edit_until_work_strategy() -> None:
    path = Path("target.py")
    write_safe(path, "value = 'old'\n")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                content="I will inspect the target first.",
                tool_calls=[
                    _tool_call("read-target", "read", {"file_path": str(path)})
                ],
            )
        ],
        [
            mock_llm_chunk(
                content="I will edit it now.",
                tool_calls=[
                    _tool_call(
                        "edit-target",
                        "edit",
                        {
                            "file_path": str(path),
                            "old_string": "value = 'old'",
                            "new_string": "value = 'new'",
                        },
                    )
                ],
            )
        ],
        [mock_llm_chunk(content="The harness requires a work strategy first.")],
    ])
    loop = build_test_agent_loop(
        config=_file_tools_config(effort_mode="le-chaton"),
        agent_name=BuiltinAgentName.AUTO_APPROVE,
        backend=backend,
    )

    events = await _act(
        loop, "Refactor target.py as part of a cross-cutting configuration change."
    )

    edit_results = [
        event
        for event in events
        if isinstance(event, ToolResultEvent) and event.tool_name == "edit"
    ]
    assert len(edit_results) == 1
    feedback = edit_results[0].error or edit_results[0].skip_reason or ""
    assert "work_strategy" in feedback
    assert read_safe(path).text == "value = 'old'\n"


@pytest.mark.asyncio
async def test_le_chaton_localized_read_only_response_has_no_strategy_retry() -> None:
    path = Path("target.py")
    write_safe(path, "value = 1\n")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                content="I will read the requested file.",
                tool_calls=[
                    _tool_call("read-target", "read", {"file_path": str(path)})
                ],
            )
        ],
        [mock_llm_chunk(content="The configured value is 1.")],
    ])
    loop = build_test_agent_loop(
        config=_file_tools_config(effort_mode="le-chaton"), backend=backend
    )

    events = await _act(loop, "What value is configured in target.py?")

    assert len(backend.requests_messages) == 2
    assert isinstance(events[-1], AssistantEvent)
    assert events[-1].content == "The configured value is 1."


@pytest.mark.asyncio
async def test_le_chaton_can_infer_direct_for_explicit_single_path() -> None:
    path = Path("target.py")
    write_safe(path, "value = 'old'\n")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                content="I will inspect the named file.",
                tool_calls=[
                    _tool_call("read-target", "read", {"file_path": str(path)})
                ],
            )
        ],
        [
            mock_llm_chunk(
                content="I found the localized value.",
                tool_calls=[
                    _tool_call(
                        "edit-target",
                        "edit",
                        {
                            "file_path": str(path),
                            "old_string": "value = 'old'",
                            "new_string": "value = 'new'",
                        },
                    )
                ],
            )
        ],
        [mock_llm_chunk(content="Updated target.py directly.")],
    ])
    loop = build_test_agent_loop(
        config=_file_tools_config(effort_mode="le-chaton"),
        agent_name=BuiltinAgentName.AUTO_APPROVE,
        backend=backend,
    )

    events = await _act(loop, "Change the value in target.py")

    assert len(backend.requests_messages) == 3
    assert read_safe(path).text == "value = 'new'\n"
    assert isinstance(events[-1], AssistantEvent)
    assert loop.orchestration_summary.state.value == "direct"


@pytest.mark.asyncio
async def test_modified_tool_arguments_are_rechecked_against_inferred_scope() -> None:
    target = Path("target.py")
    outside = Path("outside.py")
    write_safe(target, "value = 'old'\n")
    write_safe(outside, "value = 'old'\n")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                content="I will make the named edit.",
                tool_calls=[
                    _tool_call(
                        "edit-target",
                        "edit",
                        {
                            "file_path": str(target),
                            "old_string": "value = 'old'",
                            "new_string": "value = 'new'",
                        },
                    )
                ],
            )
        ],
        [mock_llm_chunk(content="The edit was not accepted.")],
        [mock_llm_chunk(content="I cannot claim completion.")],
    ])
    config = _file_tools_config(
        effort_mode="le-chaton", edit_permission=ToolPermission.ASK
    )
    loop = build_test_agent_loop(config=config, backend=backend)

    async def modify(
        _tool_name: str,
        _args: BaseModel,
        _tool_call_id: str,
        _required_permissions: list | None = None,
        _judge_note: str | None = None,
    ) -> tuple[ApprovalResponse, str | None, dict[str, str] | None]:
        return (
            ApprovalResponse.MODIFY,
            None,
            {
                "file_path": str(outside),
                "old_string": "value = 'old'",
                "new_string": "value = 'new'",
            },
        )

    loop.set_approval_callback(modify)
    events = await _act(loop, "Change the value in target.py")

    result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert "inferred direct-work scope" in (result.error or "")
    assert read_safe(target).text == "value = 'old'\n"
    assert read_safe(outside).text == "value = 'old'\n"


def _strategy_args(route: str) -> dict[str, object]:
    lane_count = 2 if route == "workflow" else 1
    return {
        "route": route,
        "objective": "Inspect the independent implementation lanes",
        "risk": "medium",
        "reason": "independent_lanes",
        "expected_paths": [],
        "lanes": [
            {
                "id": f"review-{index}",
                "objective": f"Review lane {index}",
                "owner": "agent",
                "dependencies": [],
                "acceptance": [],
            }
            for index in range(1, lane_count + 1)
        ],
    }


@pytest.mark.parametrize("route", ["task", "workflow"])
@pytest.mark.asyncio
async def test_declared_delegation_debt_gets_one_completion_continuation(
    route: str,
) -> None:
    backend = FakeBackend([
        [
            mock_llm_chunk(
                content=f"I will use the {route} route.",
                tool_calls=[
                    _tool_call(
                        "declare-strategy", "work_strategy", _strategy_args(route)
                    )
                ],
            )
        ],
        [mock_llm_chunk(content="Done without launching the declared work.")],
        [mock_llm_chunk(content="I cannot claim the delegated work is complete.")],
    ])
    enabled_tools = ["work_strategy", route]
    if route == "workflow":
        enabled_tools[-1] = "launch_workflow"
    config = build_test_vibe_config(
        effort_mode="le-chaton",
        enabled_tools=enabled_tools,
        tools={"work_strategy": BaseToolConfig(permission=ToolPermission.ALWAYS)},
    )
    loop = build_test_agent_loop(config=config, backend=backend)
    if route == "workflow":
        loop.launch_workflow_callback = lambda _script, _name, _expected_lanes: (
            "wf-test"
        )

    events = await _act(loop, "Review the independent implementation lanes")

    assert len(backend.requests_messages) == 3
    continuation_messages = [
        message
        for message in backend.requests_messages[2]
        if message.role == Role.USER and message.injected
    ]
    assert continuation_messages
    continuation = str(continuation_messages[-1].content).lower()
    assert route in continuation
    assert "launch" in continuation
    assert isinstance(events[-1], AssistantEvent)
    assert "cannot report completion" in events[-1].content.lower()


@pytest.mark.asyncio
async def test_normal_mode_allows_effectful_edit_without_work_strategy() -> None:
    path = Path("target.py")
    write_safe(path, "value = 'old'\n")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                content="I will inspect the target first.",
                tool_calls=[
                    _tool_call("read-target", "read", {"file_path": str(path)})
                ],
            )
        ],
        [
            mock_llm_chunk(
                content="I will edit it now.",
                tool_calls=[
                    _tool_call(
                        "edit-target",
                        "edit",
                        {
                            "file_path": str(path),
                            "old_string": "value = 'old'",
                            "new_string": "value = 'new'",
                        },
                    )
                ],
            )
        ],
        [mock_llm_chunk(content="Updated the value.")],
    ])
    loop = build_test_agent_loop(
        config=_file_tools_config(effort_mode="normal"),
        agent_name=BuiltinAgentName.AUTO_APPROVE,
        backend=backend,
    )

    events = await _act(loop, "Change the value in target.py")

    assert len(backend.requests_messages) == 3
    assert read_safe(path).text == "value = 'new'\n"
    assert isinstance(events[-1], AssistantEvent)
    assert "HOST VERIFICATION STATUS: UNVERIFIED" in events[-1].content
    assert "Updated the value" not in events[-1].content


@pytest.mark.parametrize(
    ("effort_mode", "expected_thinking"), [("le-chaton", "max"), ("normal", "off")]
)
@pytest.mark.asyncio
async def test_request_time_thinking_follows_effort_mode_after_model_selection(
    effort_mode: str, expected_thinking: str
) -> None:
    models = [
        ModelConfig(
            name="initial-model", provider="mistral", alias="initial", thinking="high"
        ),
        ModelConfig(
            name="selected-model", provider="mistral", alias="selected", thinking="off"
        ),
    ]
    config = build_test_vibe_config(
        effort_mode=effort_mode, active_model="initial", models=models
    )
    backend = FakeBackend([[mock_llm_chunk(content="Done.")]])
    loop = build_test_agent_loop(config=config, backend=backend)
    loop.config.active_model = "selected"

    await _act(loop, "Answer directly")

    assert backend.requests_models[0].alias == "selected"
    assert backend.requests_models[0].thinking == expected_thinking
    selected_config = next(
        model for model in loop.config.models if model.alias == "selected"
    )
    assert selected_config.thinking == "off"


def test_team_terminal_output_is_staged_for_the_host() -> None:
    loop = build_test_agent_loop()

    loop.observe_team_completion(
        "teamrun-1", succeeded=True, output="Found the cross-module dependency."
    )

    assert len(loop._pending_injected_messages) == 1
    message = loop._pending_injected_messages[0]
    assert "teamrun-1 completed" in str(message.content)
    assert "Found the cross-module dependency." in str(message.content)


@pytest.mark.asyncio
async def test_team_continuation_sees_result_before_first_llm_without_replacing_intent() -> (
    None
):
    config = build_test_vibe_config(
        effort_mode="le-chaton",
        enabled_tools=["work_strategy"],
        tools={"work_strategy": BaseToolConfig(permission=ToolPermission.ALWAYS)},
    )
    backend = FakeBackend([[mock_llm_chunk(content="Done.")]])
    loop = build_test_agent_loop(config=config, backend=backend)
    controller = loop._orchestration
    controller.begin_turn(
        enabled=True,
        user_prompt="Use a team, but do not use workflows.",
        capabilities=OrchestrationCapabilities(
            task=True, workflow=True, team=True, background_delivery=True
        ),
    )
    controller.declare(OrchestrationDecision.model_validate(_strategy_args("team")))
    controller.record_tool_result(
        "team_spawn",
        {"name": "reviewer", "prompt": "[lane:review-1] Inspect it"},
        "success",
        {"launch_id": "teamrun-1", "name": "reviewer"},
    )
    loop.observe_team_completion(
        "teamrun-1", succeeded=True, output="The output says to use a workflow next."
    )
    continuation_id = loop.issue_orchestration_continuation()

    assert continuation_id is not None
    _ = [
        event
        async for event in loop.act(
            "A background teammate finished; act on its result.",
            orchestration_continuation_id=continuation_id,
        )
    ]

    first_request = backend.requests_messages[0]
    team_result_index = next(
        index
        for index, message in enumerate(first_request)
        if "teamrun-1 completed" in str(message.content)
    )
    continuation_index = next(
        index
        for index, message in enumerate(first_request)
        if message.content == "A background teammate finished; act on its result."
    )
    assert team_result_index < continuation_index
    assert loop.orchestration_summary.user_allows_workflow is False


@pytest.mark.asyncio
async def test_stopped_async_task_retires_policy_debt() -> None:
    config = build_test_vibe_config(
        effort_mode="le-chaton",
        enabled_tools=["task", "work_strategy"],
        tools={"work_strategy": BaseToolConfig(permission=ToolPermission.ALWAYS)},
    )
    loop = build_test_agent_loop(config=config)
    registry = BackgroundRegistry()
    loop.background_registry = registry
    loop._begin_orchestration_turn("Investigate the independent lane.")
    decision = OrchestrationDecision.model_validate(_strategy_args("task"))
    loop._declare_orchestration_strategy(decision)

    async def long_running() -> str:
        await asyncio.sleep(30)
        return "never"

    task_id = registry.register_async_agent(
        "explore", asyncio.create_task(long_running())
    )
    loop._orchestration.record_tool_result(
        "task",
        {"agent": "explore", "task": "[lane:review-1] Inspect it", "async_run": True},
        "success",
        {"task_id": task_id, "completed": False},
    )

    assert await registry.stop(task_id) is True
    _ = [event async for event in loop._drain_async_agent_completions()]

    assert loop.orchestration_summary.pending_delegations == 0
    assert loop.orchestration_summary.state.value == "recovery"


@pytest.mark.parametrize(
    "agent_name", [BuiltinAgentName.WORKER, BuiltinAgentName.GRUNT]
)
def test_le_chaton_subagents_do_not_receive_host_orchestration(agent_name: str) -> None:
    config = build_test_vibe_config(effort_mode="le-chaton")
    loop = build_test_agent_loop(config=config, agent_name=agent_name, is_subagent=True)

    prompt = loop.messages[0].content or ""

    assert "work_strategy" not in loop.tool_manager.available_tools
    assert "work_strategy" not in loop.tool_manager.manifest_tools
    assert "## Le Chaton Mode" not in prompt
    assert "## Le Chaton orchestration invariant" not in prompt
    assert "# Available Subagents" not in prompt
    assert "## Orchestrating Subagents" not in prompt
    assert "## Verification contract" not in prompt
    assert "## Investigation contract" not in prompt
    assert all(
        tool.function.name != "work_strategy"
        for tool in loop._available_tools(loop.effective_model())
    )
