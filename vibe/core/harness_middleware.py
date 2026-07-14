from __future__ import annotations

from vibe.core.middleware import (
    ConversationContext,
    MiddlewareAction,
    MiddlewareResult,
    ResetReason,
)
from vibe.core.types import Role

_CAPABILITY_FAILURE_THRESHOLD = 3
_CAPABILITY_TOOLS = frozenset({
    "bash",
    "edit",
    "glob",
    "grep",
    "land_work",
    "launch_workflow",
    "lsp",
    "manage_memory",
    "read",
    "schedule",
    "task",
    "task_checks",
    "team",
    "team_spawn",
    "verify_work",
    "work_strategy",
    "workflow_stop",
    "write_file",
})
_CAPABILITY_MARKERS = {
    "orchestration policy": (
        "record a strategy scoped to workspace-relative paths",
        "record work_strategy before",
        "le chaton requires an adaptive work_strategy",
        "the declared delegation has not launched yet",
        "every declared productive lane has already launched",
        "the recorded strategy requires route",
        "the direct route reached its bounded mutation envelope",
        "the direct route expanded beyond",
        "outside the declared direct-work scope",
        "outside the inferred direct-work scope",
        "the planned delegation failed",
        "strategy lane",
        "workflow lane",
        "bind every declared workflow lane",
        "bind this delegation",
    ),
    "sandbox startup": (
        "sandbox wrapper failed",
        "strict model control requires",
        "bwrap:",
    ),
    "policy denial": (
        "tool execution not permitted",
        "permanently disabled",
        "policy denied",
    ),
    "filesystem confinement": (
        "read-only file system",
        "operation not permitted",
        "permission denied",
    ),
}
_NON_SUBSTANTIVE_RECEIPT_TOOLS = frozenset({
    "ask_user_question",
    "background",
    "enter_plan_mode",
    "exit_plan_mode",
    "glob",
    "grep",
    "lsp",
    "read",
    "skill",
    "team_message",
    "todo",
    "tool_search",
    "webfetch",
    "websearch",
    "workflow_results",
    "workflow_status",
    "work_strategy",
})
_REJECTED_STRATEGY_FIELDS = frozenset({"accepted: false", '"accepted": false'})


class CapabilityFailureCircuitBreaker:
    def __init__(self, threshold: int = _CAPABILITY_FAILURE_THRESHOLD) -> None:
        self._threshold = threshold

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        capability = self._repeated_capability(context)
        if capability is None:
            return MiddlewareResult()
        return MiddlewareResult(
            action=MiddlewareAction.STOP,
            reason=(
                "HOST CAPABILITY STATUS: BLOCKED\n"
                f"The harness observed {self._threshold} consecutive {capability} "
                "failures. This turn was stopped before another model or tool call. "
                "No control-plane substitution or cleanup attempt is authorized."
            ),
        )

    def _repeated_capability(self, context: ConversationContext) -> str | None:
        failures: list[str] = []
        for message in reversed(context.messages):
            if message.role == Role.USER and not message.injected:
                break
            if message.role != Role.TOOL:
                continue
            capability = _classify_capability_failure(
                message.content, tool_name=message.name
            )
            if capability is not None:
                failures.append(capability)
                if len(failures) >= self._threshold:
                    break
                continue
            if _is_substantive_boundary(message.content, tool_name=message.name):
                break
        if len(failures) < self._threshold or len(set(failures)) != 1:
            return None
        return failures[0]

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        pass


def _classify_capability_failure(
    content: str | None, *, tool_name: str | None
) -> str | None:
    folded = (content or "").casefold()
    if tool_name == "work_strategy" and any(
        line.strip().rstrip(",") in _REJECTED_STRATEGY_FIELDS
        for line in folded.splitlines()
    ):
        return "orchestration policy"
    if tool_name not in _CAPABILITY_TOOLS or "<tool_error>" not in folded:
        return None
    return next(
        (
            capability
            for capability, markers in _CAPABILITY_MARKERS.items()
            if any(marker in folded for marker in markers)
        ),
        None,
    )


def _is_substantive_boundary(content: str | None, *, tool_name: str | None) -> bool:
    folded = (content or "").casefold()
    if tool_name == "manage_memory" and any(
        line.strip() == "action: list" for line in folded.splitlines()
    ):
        return False
    return tool_name not in _NON_SUBSTANTIVE_RECEIPT_TOOLS


__all__ = ["CapabilityFailureCircuitBreaker"]
