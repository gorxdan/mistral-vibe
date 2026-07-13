from __future__ import annotations

import pytest

from tests.conftest import build_test_vibe_config
from tests.mock.utils import collect_result
from vibe.core.orchestration import (
    LaneOwner,
    OrchestrationLane,
    OrchestrationRoute,
    StrategyReason,
    WorkRisk,
)
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError, ToolPermission
from vibe.core.tools.builtins.work_strategy import (
    OrchestrationDecision,
    OrchestrationState,
    StrategyReceipt,
    WorkStrategy,
    WorkStrategyArgs,
    WorkStrategyConfig,
)
from vibe.core.tools.manager import NoSuchToolError, ToolManager


def _make_tool() -> WorkStrategy:
    return WorkStrategy(
        config_getter=lambda: WorkStrategyConfig(), state=BaseToolState()
    )


def _workflow_args() -> WorkStrategyArgs:
    return WorkStrategyArgs(
        route=OrchestrationRoute.WORKFLOW,
        objective="Trace and correct Le Chaton orchestration behavior",
        risk=WorkRisk.HIGH,
        reason=StrategyReason.INDEPENDENT_LANES,
        expected_paths=[
            "vibe/core/system_prompt.py",
            "vibe/core/tools/builtins/",
            "tests/",
        ],
        lanes=[
            OrchestrationLane(
                id="policy",
                objective="Define and test the host orchestration policy",
                owner=LaneOwner.AGENT,
                profile="planner",
                dependencies=[],
                acceptance=["Small, medium, and large prompt tiers are covered"],
            ),
            OrchestrationLane(
                id="tool",
                objective="Implement the strategy receipt tool",
                owner=LaneOwner.AGENT,
                profile="worker",
                dependencies=["policy"],
                acceptance=["The receipt is validated and recorded"],
            ),
            OrchestrationLane(
                id="integration",
                objective="Integrate and verify the candidate",
                owner=LaneOwner.HOST,
                dependencies=["policy", "tool"],
                acceptance=["Focused tests and static checks pass"],
            ),
        ],
    )


@pytest.mark.asyncio
async def test_requires_strategy_recorder_callback() -> None:
    with pytest.raises(ToolError, match="not available"):
        await collect_result(
            _make_tool().run(
                _workflow_args(), ctx=InvokeContext(tool_call_id="strategy")
            )
        )


@pytest.mark.asyncio
async def test_records_validated_decision_and_returns_receipt() -> None:
    seen: list[OrchestrationDecision] = []
    receipt = StrategyReceipt(
        route=OrchestrationRoute.WORKFLOW,
        state=OrchestrationState.DELEGATION_PENDING,
        message="Workflow strategy recorded; launch the orchestration primitive.",
        required_delegations=1,
    )

    def record(decision: OrchestrationDecision) -> StrategyReceipt:
        seen.append(decision)
        return receipt

    result = await collect_result(
        _make_tool().run(
            _workflow_args(),
            ctx=InvokeContext(tool_call_id="strategy", work_strategy_callback=record),
        )
    )

    assert result == receipt
    assert result.state is OrchestrationState.DELEGATION_PENDING
    assert result.model_dump(mode="json")["state"] == "delegation_pending"
    assert result.required_delegations == 1
    assert len(seen) == 1
    decision = seen[0]
    assert isinstance(decision, OrchestrationDecision)
    assert decision.route == "workflow"
    assert decision.risk == "high"
    assert decision.reason == "independent_lanes"
    assert decision.expected_paths == [
        "vibe/core/system_prompt.py",
        "vibe/core/tools/builtins/",
        "tests/",
    ]
    assert [lane.id for lane in decision.lanes] == ["policy", "tool", "integration"]
    assert [lane.owner for lane in decision.lanes] == ["agent", "agent", "host"]
    assert decision.lanes[1].dependencies == ["policy"]
    assert decision.lanes[2].profile is None


def test_available_only_in_le_chaton_mode() -> None:
    normal = build_test_vibe_config(effort_mode="normal")
    le_chaton = build_test_vibe_config(effort_mode="le-chaton")
    le_chaton_without_workflows = build_test_vibe_config(
        effort_mode="le-chaton", disable_workflows=True
    )

    assert WorkStrategy.is_available(normal) is False
    assert WorkStrategy.is_available(le_chaton) is True
    assert WorkStrategy.is_available(le_chaton_without_workflows) is True


def test_permission_is_always() -> None:
    tool = _make_tool()

    assert WorkStrategyConfig().permission is ToolPermission.ALWAYS
    permission = tool.resolve_permission(_workflow_args())
    assert permission is not None
    assert permission.permission is ToolPermission.ALWAYS


def test_host_only_tool_is_removed_from_subagent_manager() -> None:
    config = build_test_vibe_config(effort_mode="le-chaton")
    host = ToolManager(lambda: config)
    subagent = ToolManager(lambda: config, host=False)

    assert "work_strategy" in host.available_tools
    assert "work_strategy" not in subagent.available_tools
    assert "work_strategy" not in subagent.manifest_tools
    with pytest.raises(NoSuchToolError):
        subagent.get("work_strategy")


def test_le_chaton_policy_control_survives_a_host_capability_allowlist() -> None:
    config = build_test_vibe_config(effort_mode="le-chaton", enabled_tools=["read"])

    manager = ToolManager(lambda: config)

    assert set(manager.available_tools) == {"read", "work_strategy"}
