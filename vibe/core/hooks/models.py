from __future__ import annotations

from collections.abc import Callable
from enum import auto
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vibe.core.types import BaseEvent, StrEnum


class HookMessageSeverity(StrEnum):
    OK = auto()
    WARNING = auto()
    ERROR = auto()


class HookType(StrEnum):
    POST_AGENT_TURN = auto()
    BEFORE_TOOL = auto()
    AFTER_TOOL = auto()
    TEAMMATE_IDLE = auto()
    TASK_CREATED = auto()
    TASK_COMPLETED = auto()
    PRE_COMPACT = auto()
    USER_PROMPT_SUBMIT = auto()
    STOP = auto()
    SESSION_START = auto()
    SESSION_END = auto()
    NOTIFICATION = auto()


ToolStatus = Literal["success", "failure", "cancelled"]


_DEFAULT_HOOK_TIMEOUT = 60.0


class HookConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    type: HookType
    command: str
    match: str | None = None
    timeout: float | None = None
    strict: bool = False
    description: str | None = None

    @field_validator("command")
    @classmethod
    def command_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("command must not be empty")
        return v

    @field_validator("match")
    @classmethod
    def match_not_blank(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("match must not be empty")
        return v

    @model_validator(mode="after")
    def _apply_defaults_and_constraints(self) -> Self:
        if self.match is not None and self.type == HookType.POST_AGENT_TURN:
            raise ValueError(
                "match is only valid for tool hooks (before_tool / after_tool)"
            )
        if self.strict and self.type == HookType.POST_AGENT_TURN:
            raise ValueError(
                "strict is only valid for tool hooks (before_tool / after_tool)"
            )
        if self.timeout is None:
            self.timeout = _DEFAULT_HOOK_TIMEOUT
        return self


class HookConfigIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file: Path
    message: str


class HookConfigResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hooks: list[HookConfig]
    issues: list[HookConfigIssue]


class HookSessionContext(BaseModel):
    """Shared session fields passed to every hook invocation."""

    model_config = ConfigDict(extra="forbid")
    session_id: str
    transcript_path: str
    cwd: str
    parent_session_id: str | None = None


class PostAgentTurnInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.POST_AGENT_TURN] = HookType.POST_AGENT_TURN


class BeforeToolInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.BEFORE_TOOL] = HookType.BEFORE_TOOL
    tool_name: str
    tool_call_id: str
    tool_input: dict[str, Any]


class AfterToolInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.AFTER_TOOL] = HookType.AFTER_TOOL
    tool_name: str
    tool_call_id: str
    tool_input: dict[str, Any]
    tool_status: ToolStatus
    tool_output: dict[str, Any] | None
    tool_output_text: str
    tool_error: str | None
    duration_ms: float


class TeammateIdleInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.TEAMMATE_IDLE] = HookType.TEAMMATE_IDLE
    teammate_name: str
    teammate_session_id: str | None = None


class TaskCreatedInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.TASK_CREATED] = HookType.TASK_CREATED
    task_id: str
    task_description: str
    assignee: str | None = None


class TaskCompletedInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.TASK_COMPLETED] = HookType.TASK_COMPLETED
    task_id: str
    teammate_name: str
    result: str | None = None


class PreCompactInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.PRE_COMPACT] = HookType.PRE_COMPACT
    trigger: str  # "auto" | "emergency" | "manual"
    current_context_tokens: int
    threshold: int


class UserPromptSubmitInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.USER_PROMPT_SUBMIT] = HookType.USER_PROMPT_SUBMIT
    prompt: str
    message_id: str | None = None
    has_images: bool = False


class StopInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.STOP] = HookType.STOP
    # True when this Stop fires inside a Stop-induced continuation; lets a hook
    # short-circuit to avoid an infinite continue loop.
    stop_hook_active: bool = False


class SessionStartInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.SESSION_START] = HookType.SESSION_START
    source: str = "startup"  # "startup" | "resume" | "clear"


class SessionEndInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.SESSION_END] = HookType.SESSION_END
    reason: str = "exit"  # "exit" | "clear" | "logout"


class NotificationInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.NOTIFICATION] = HookType.NOTIFICATION
    notification_type: str  # "permission_required" | "question" | "input_idle"
    message: str = ""
    tool_name: str | None = None


HookInvocation = (
    PostAgentTurnInvocation
    | BeforeToolInvocation
    | AfterToolInvocation
    | TeammateIdleInvocation
    | TaskCreatedInvocation
    | TaskCompletedInvocation
    | PreCompactInvocation
    | UserPromptSubmitInvocation
    | StopInvocation
    | SessionStartInvocation
    | SessionEndInvocation
    | NotificationInvocation
)


def _build_before_tool(base: dict[str, Any], f: dict[str, Any]) -> BeforeToolInvocation:
    if f.get("tool_name") is None or f.get("tool_call_id") is None:
        raise ValueError(
            "tool_name and tool_call_id are required for before_tool hooks"
        )
    return BeforeToolInvocation(
        **base,
        tool_name=f["tool_name"],
        tool_call_id=f["tool_call_id"],
        tool_input=f.get("tool_input") or {},
    )


def _build_after_tool(base: dict[str, Any], f: dict[str, Any]) -> AfterToolInvocation:
    if (
        f.get("tool_name") is None
        or f.get("tool_call_id") is None
        or f.get("tool_status") is None
    ):
        raise ValueError(
            "tool_name, tool_call_id, and tool_status are required for after_tool hooks"
        )
    return AfterToolInvocation(
        **base,
        tool_name=f["tool_name"],
        tool_call_id=f["tool_call_id"],
        tool_input=f.get("tool_input") or {},
        tool_status=f["tool_status"],
        tool_output=f.get("tool_output"),
        tool_output_text=f.get("tool_output_text", ""),
        tool_error=f.get("tool_error"),
        duration_ms=f.get("duration_ms", 0.0),
    )


def _build_teammate_idle(
    base: dict[str, Any], f: dict[str, Any]
) -> TeammateIdleInvocation:
    if f.get("teammate_name") is None:
        raise ValueError("teammate_name is required for teammate_idle hooks")
    return TeammateIdleInvocation(
        **base,
        teammate_name=f["teammate_name"],
        teammate_session_id=f.get("teammate_session_id"),
    )


def _build_task_created(
    base: dict[str, Any], f: dict[str, Any]
) -> TaskCreatedInvocation:
    if f.get("task_id") is None or f.get("task_description") is None:
        raise ValueError(
            "task_id and task_description are required for task_created hooks"
        )
    return TaskCreatedInvocation(
        **base,
        task_id=f["task_id"],
        task_description=f["task_description"],
        assignee=f.get("assignee"),
    )


def _build_task_completed(
    base: dict[str, Any], f: dict[str, Any]
) -> TaskCompletedInvocation:
    if f.get("task_id") is None or f.get("teammate_name") is None:
        raise ValueError(
            "task_id and teammate_name are required for task_completed hooks"
        )
    return TaskCompletedInvocation(
        **base,
        task_id=f["task_id"],
        teammate_name=f["teammate_name"],
        result=f.get("result"),
    )


def _build_pre_compact(base: dict[str, Any], f: dict[str, Any]) -> PreCompactInvocation:
    if f.get("trigger") is None:
        raise ValueError("trigger is required for pre_compact hooks")
    return PreCompactInvocation(
        **base,
        trigger=f["trigger"],
        current_context_tokens=f.get("current_context_tokens") or 0,
        threshold=f.get("threshold") or 0,
    )


def _build_user_prompt_submit(
    base: dict[str, Any], f: dict[str, Any]
) -> UserPromptSubmitInvocation:
    if f.get("prompt") is None:
        raise ValueError("prompt is required for user_prompt_submit hooks")
    return UserPromptSubmitInvocation(
        **base,
        prompt=f["prompt"],
        message_id=f.get("message_id"),
        has_images=f.get("has_images", False),
    )


def _build_stop(base: dict[str, Any], f: dict[str, Any]) -> StopInvocation:
    return StopInvocation(**base, stop_hook_active=f.get("stop_hook_active", False))


def _build_session_start(
    base: dict[str, Any], f: dict[str, Any]
) -> SessionStartInvocation:
    return SessionStartInvocation(**base, source=f.get("source") or "startup")


def _build_session_end(base: dict[str, Any], f: dict[str, Any]) -> SessionEndInvocation:
    return SessionEndInvocation(**base, reason=f.get("reason") or "exit")


def _build_notification(
    base: dict[str, Any], f: dict[str, Any]
) -> NotificationInvocation:
    if f.get("notification_type") is None:
        raise ValueError("notification_type is required for notification hooks")
    return NotificationInvocation(
        **base,
        notification_type=f["notification_type"],
        message=f.get("message", ""),
        tool_name=f.get("tool_name"),
    )


_INVOCATION_BUILDERS: dict[
    HookType, Callable[[dict[str, Any], dict[str, Any]], HookInvocation]
] = {
    HookType.POST_AGENT_TURN: lambda base, f: PostAgentTurnInvocation(**base),
    HookType.BEFORE_TOOL: _build_before_tool,
    HookType.AFTER_TOOL: _build_after_tool,
    HookType.TEAMMATE_IDLE: _build_teammate_idle,
    HookType.TASK_CREATED: _build_task_created,
    HookType.TASK_COMPLETED: _build_task_completed,
    HookType.PRE_COMPACT: _build_pre_compact,
    HookType.USER_PROMPT_SUBMIT: _build_user_prompt_submit,
    HookType.STOP: _build_stop,
    HookType.SESSION_START: _build_session_start,
    HookType.SESSION_END: _build_session_end,
    HookType.NOTIFICATION: _build_notification,
}


def build_invocation(
    hook_type: HookType, ctx: HookSessionContext, **fields: Any
) -> HookInvocation:
    """Build the right HookInvocation subclass for *hook_type*."""
    base = ctx.model_dump()
    builder = _INVOCATION_BUILDERS.get(hook_type)
    if builder is None:
        raise ValueError(f"Unknown hook type: {hook_type!r}")
    return builder(base, fields)


class HookExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hook_name: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool


class HookSpecificOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # before_tool only.
    tool_input: dict[str, Any] | None = None
    # after_tool only.
    additional_context: str | None = None


class HookStructuredResponse(BaseModel):
    """The hook spec is "exit 0 + JSON object on stdout". ``decision:
    "deny"`` has per-type effect (denial / text replacement / retry
    injection). Unknown fields at any level are tolerated.
    """

    model_config = ConfigDict(extra="ignore")

    decision: Literal["allow", "deny"] = "allow"
    reason: str | None = None
    system_message: str | None = None
    hook_specific_output: HookSpecificOutput = Field(default_factory=HookSpecificOutput)


class HookUserMessage(BaseModel):
    """post_agent_turn deny: ``content`` is injected as a retry user
    message.
    """

    model_config = ConfigDict(extra="forbid")
    content: str


class HookPromptBlock(BaseModel):
    """user_prompt_submit deny: the prompt is blocked; ``content`` is the
    reason surfaced to the user. No LLM turn runs.
    """

    model_config = ConfigDict(extra="forbid")
    hook_name: str
    content: str


class HookToolDenial(BaseModel):
    """before_tool deny: ``content`` becomes the tool error returned to
    the LLM.
    """

    model_config = ConfigDict(extra="forbid")
    hook_name: str
    content: str


class HookToolInputRewrite(BaseModel):
    """before_tool: one per rewriting hook in the chain. The agent loop
    validates each as it arrives — the first invalid rewrite aborts the
    chain and synthesizes a denial.
    """

    model_config = ConfigDict(extra="forbid")
    hook_name: str
    tool_input: dict[str, Any]


class HookTextReplacement(BaseModel):
    """after_tool: ``text`` is the cumulative LLM-bound output after the
    handler applied its replacement or append.
    """

    model_config = ConfigDict(extra="forbid")

    text: str


class HookEvent(BaseEvent):
    pass


class HookRunStartEvent(HookEvent):
    scope: HookType = HookType.POST_AGENT_TURN
    tool_name: str | None = None
    tool_call_id: str | None = None


class HookRunEndEvent(HookEvent):
    scope: HookType = HookType.POST_AGENT_TURN
    tool_call_id: str | None = None


# scope / tool_call_id let consumers route events when concurrent tool-call
# chains interleave on the wire.
class HookStartEvent(HookEvent):
    hook_name: str
    scope: HookType = HookType.POST_AGENT_TURN
    tool_call_id: str | None = None


class HookEndEvent(HookEvent):
    hook_name: str
    status: HookMessageSeverity
    content: str | None = None
    scope: HookType = HookType.POST_AGENT_TURN
    tool_call_id: str | None = None
