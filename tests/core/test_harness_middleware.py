from __future__ import annotations

import pytest

from tests.conftest import build_test_vibe_config
from vibe.core.agent_loop import _managed_runtime_limits
from vibe.core.config import (
    TrustedExecutionTopologyConfig,
    TrustedVerificationCheckConfig,
    TrustedVerificationRecipeConfig,
)
from vibe.core.harness_middleware import CapabilityFailureCircuitBreaker
from vibe.core.middleware import ConversationContext, MiddlewareAction
from vibe.core.types import AgentStats, LLMMessage, MessageList, Role


def _managed_config(*, max_turns: int = 80, max_tokens: int = 2_000_000):
    recipe = TrustedVerificationRecipeConfig(
        recipe_version="managed-v1",
        task_brief="Run the packet",
        acceptance_contract="All checks pass",
        allowed_paths=("candidate.py",),
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
            packet_path="docs/packet.md",
            state="active",
            control_worktree="/maintenance/control",
            control_sha="1" * 40,
            candidate_worktree="/maintenance/candidate",
            candidate_branch="maintenance/i00-p01",
            baseline_sha="2" * 40,
            upstream_sha="3" * 40,
            evidence_workspace="/maintenance/evidence",
            run_id="managed-run",
            runner_id="managed-runner",
            max_turns=max_turns,
            max_session_tokens=max_tokens,
        ),
    )
    return build_test_vibe_config(trusted_verification_recipe=recipe)


def test_managed_runtime_applies_finite_limits_when_caller_omits_them() -> None:
    assert _managed_runtime_limits(_managed_config(), None, None) == (80, 2_000_000)


def test_managed_runtime_cannot_raise_host_limits() -> None:
    assert _managed_runtime_limits(
        _managed_config(max_turns=40, max_tokens=500_000), 400, 5_000_000
    ) == (40, 500_000)


def test_managed_runtime_preserves_stricter_caller_limits() -> None:
    assert _managed_runtime_limits(
        _managed_config(max_turns=40, max_tokens=500_000), 10, 100_000
    ) == (10, 100_000)


def _context(contents: list[str]) -> ConversationContext:
    messages = MessageList(
        initial=[
            LLMMessage(role=Role.USER, content="Run the maintenance packet."),
            *[
                LLMMessage(
                    role=Role.TOOL,
                    name="bash",
                    content=(
                        content
                        if "returncode: 0" in content
                        else f"<tool_error>{content}</tool_error>"
                    ),
                )
                for content in contents
            ],
        ]
    )
    return ConversationContext(
        messages=messages, stats=AgentStats(), config=build_test_vibe_config()
    )


def _tool(name: str, content: str, *, error: bool = False) -> LLMMessage:
    if error:
        content = f"<tool_error>{content}</tool_error>"
    return LLMMessage(role=Role.TOOL, name=name, content=content)


def _message_context(messages: list[LLMMessage]) -> ConversationContext:
    return ConversationContext(
        messages=MessageList(
            initial=[
                LLMMessage(role=Role.USER, content="Run the maintenance packet."),
                *messages,
            ]
        ),
        stats=AgentStats(),
        config=build_test_vibe_config(),
    )


@pytest.mark.asyncio
async def test_three_filesystem_capability_failures_force_blocked_handoff() -> None:
    middleware = CapabilityFailureCircuitBreaker()
    context = _context([
        "open: Read-only file system",
        "mkdir: Operation not permitted",
        "open: Permission denied",
    ])

    result = await middleware.before_turn(context)

    assert result.action is MiddlewareAction.STOP
    assert "HOST CAPABILITY STATUS: BLOCKED" in (result.reason or "")
    assert "3 consecutive filesystem confinement failures" in (result.reason or "")
    assert "stopped before another model or tool call" in (result.reason or "")


@pytest.mark.asyncio
async def test_two_capability_failures_do_not_trip_breaker() -> None:
    middleware = CapabilityFailureCircuitBreaker()

    result = await middleware.before_turn(
        _context(["Read-only file system", "Permission denied"])
    )

    assert result.action is MiddlewareAction.CONTINUE


@pytest.mark.asyncio
async def test_bwrap_namespace_denials_are_sandbox_startup_failures() -> None:
    message = "bwrap: Creating new namespace failed: Operation not permitted"
    result = await CapabilityFailureCircuitBreaker().before_turn(
        _context([message, message, message])
    )

    assert result.action is MiddlewareAction.STOP
    assert "3 consecutive sandbox startup failures" in (result.reason or "")


@pytest.mark.asyncio
async def test_unrelated_substantive_error_resets_capability_failure_run() -> None:
    context = _message_context([
        _tool("bash", "Read-only file system", error=True),
        _tool("bash", "compiler reported an invalid type", error=True),
        _tool("edit", "Operation not permitted", error=True),
        _tool("write_file", "Permission denied", error=True),
    ])

    result = await CapabilityFailureCircuitBreaker().before_turn(context)

    assert result.action is MiddlewareAction.CONTINUE


@pytest.mark.asyncio
async def test_success_between_failures_resets_consecutive_window() -> None:
    middleware = CapabilityFailureCircuitBreaker()
    context = _context([
        "Read-only file system",
        "Permission denied",
        "returncode: 0\nstdout: clean status",
        "Operation not permitted",
        "Read-only file system",
        "Permission denied",
    ])

    result = await middleware.before_turn(context)

    assert result.action is MiddlewareAction.STOP

    context.messages.append(
        LLMMessage(role=Role.TOOL, content="returncode: 0\nstdout: recovered")
    )
    reset = await middleware.before_turn(context)
    assert reset.action is MiddlewareAction.CONTINUE


@pytest.mark.asyncio
async def test_mixed_capability_classes_do_not_trip_breaker() -> None:
    middleware = CapabilityFailureCircuitBreaker()

    result = await middleware.before_turn(
        _context([
            "Read-only file system",
            "Tool execution not permitted",
            "bwrap: failed to start",
        ])
    )

    assert result.action is MiddlewareAction.CONTINUE


@pytest.mark.asyncio
async def test_successful_tool_output_with_failure_words_does_not_trip() -> None:
    context = _context([])
    for _ in range(3):
        context.messages.append(
            LLMMessage(
                role=Role.TOOL,
                name="read",
                content="The fixture says Permission denied and Read-only file system.",
            )
        )

    result = await CapabilityFailureCircuitBreaker().before_turn(context)

    assert result.action is MiddlewareAction.CONTINUE


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["read", "glob", "grep", "lsp"])
async def test_read_capability_failures_trip_breaker(tool_name: str) -> None:
    context = _message_context([
        _tool(tool_name, "Permission denied", error=True),
        _tool(tool_name, "Operation not permitted", error=True),
        _tool(tool_name, "Read-only file system", error=True),
    ])

    result = await CapabilityFailureCircuitBreaker().before_turn(context)

    assert result.action is MiddlewareAction.STOP
    assert "3 consecutive filesystem confinement failures" in (result.reason or "")


@pytest.mark.asyncio
async def test_orchestration_denials_cross_control_receipts() -> None:
    context = _message_context([
        _tool(
            "edit",
            "Le Chaton requires an adaptive work_strategy before the first "
            "substantive mutating tool.",
            error=True,
        ),
        _tool(
            "work_strategy",
            "route: direct\nstate: direct\naccepted: True\nrequired_delegations: 0",
        ),
        _tool(
            "bash",
            "The direct route reached its bounded mutation envelope. "
            "Reassess with work_strategy before another mutation.",
            error=True,
        ),
        _tool("background", "response: 1 background task: asub-2 running"),
        _tool(
            "manage_memory",
            "The declared delegation has not launched yet. Launch its productive "
            "task, workflow, or team lane before substantive host mutation.",
            error=True,
        ),
    ])

    result = await CapabilityFailureCircuitBreaker().before_turn(context)

    assert result.action is MiddlewareAction.STOP
    assert "3 consecutive orchestration policy failures" in (result.reason or "")


@pytest.mark.asyncio
async def test_observational_success_does_not_hide_denial_cycle() -> None:
    context = _message_context([
        _tool("edit", "The declared delegation has not launched yet.", error=True),
        _tool("read", "file_path: src/app.py\ncontent: pass"),
        _tool("bash", "The declared delegation has not launched yet.", error=True),
        _tool("todo", "message: Updated 4 todos"),
        _tool(
            "write_file", "The declared delegation has not launched yet.", error=True
        ),
    ])

    result = await CapabilityFailureCircuitBreaker().before_turn(context)

    assert result.action is MiddlewareAction.STOP


@pytest.mark.asyncio
async def test_memory_list_receipt_does_not_hide_denial_cycle() -> None:
    context = _message_context([
        _tool("edit", "The declared delegation has not launched yet.", error=True),
        _tool("bash", "The declared delegation has not launched yet.", error=True),
        _tool("manage_memory", "action: list\nmessage: 2 memories"),
        _tool(
            "write_file", "The declared delegation has not launched yet.", error=True
        ),
    ])

    result = await CapabilityFailureCircuitBreaker().before_turn(context)

    assert result.action is MiddlewareAction.STOP


@pytest.mark.asyncio
async def test_rejected_strategy_receipts_count_as_orchestration_denials() -> None:
    context = _message_context([
        _tool("work_strategy", "route: direct\nstate: route_required\naccepted: False"),
        _tool(
            "work_strategy", "route: workflow\nstate: route_required\naccepted: False"
        ),
        _tool("work_strategy", "route: task\nstate: route_required\naccepted: False"),
    ])

    result = await CapabilityFailureCircuitBreaker().before_turn(context)

    assert result.action is MiddlewareAction.STOP


@pytest.mark.asyncio
async def test_substantive_success_resets_orchestration_denial_cycle() -> None:
    context = _message_context([
        _tool("edit", "The declared delegation has not launched yet.", error=True),
        _tool("bash", "The declared delegation has not launched yet.", error=True),
        _tool("edit", "file: src/app.py\nmessage: updated successfully"),
        _tool("edit", "The declared delegation has not launched yet.", error=True),
        _tool(
            "write_file", "The declared delegation has not launched yet.", error=True
        ),
    ])

    result = await CapabilityFailureCircuitBreaker().before_turn(context)

    assert result.action is MiddlewareAction.CONTINUE


@pytest.mark.asyncio
async def test_new_user_turn_resets_orchestration_denial_cycle() -> None:
    context = _message_context([
        _tool("edit", "The declared delegation has not launched yet.", error=True),
        _tool("bash", "The declared delegation has not launched yet.", error=True),
        LLMMessage(role=Role.USER, content="Try the smaller fix instead."),
        _tool("edit", "The declared delegation has not launched yet.", error=True),
    ])

    result = await CapabilityFailureCircuitBreaker().before_turn(context)

    assert result.action is MiddlewareAction.CONTINUE
