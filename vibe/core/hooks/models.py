from __future__ import annotations

from enum import auto
from pathlib import Path
from typing import Any, Literal, Self, assert_never

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vibe.core.types import BaseEvent, StrEnum

# --- Types & enums ---


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


ToolStatus = Literal["success", "failure", "cancelled"]


_DEFAULT_HOOK_TIMEOUT = 60.0


# --- Declarative hook config (TOML on disk) ---


class HookConfig(BaseModel):
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
    file: Path
    message: str


class HookConfigResult(BaseModel):
    hooks: list[HookConfig]
    issues: list[HookConfigIssue]


# --- Subprocess execution ---


class HookSessionContext(BaseModel):
    """Shared session fields passed to every hook invocation."""

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
    hook_event_name: Literal[HookType.USER_PROMPT_SUBMIT] = (
        HookType.USER_PROMPT_SUBMIT
    )
    prompt: str
    message_id: str | None = None
    has_images: bool = False


HookInvocation = (
    PostAgentTurnInvocation
    | BeforeToolInvocation
    | AfterToolInvocation
    | TeammateIdleInvocation
    | TaskCreatedInvocation
    | TaskCompletedInvocation
    | PreCompactInvocation
    | UserPromptSubmitInvocation
)


def build_invocation(  # noqa: PLR0913
    hook_type: HookType,
    ctx: HookSessionContext,
    *,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    tool_input: dict[str, Any] | None = None,
    tool_status: ToolStatus | None = None,
    tool_output: dict[str, Any] | None = None,
    tool_output_text: str = "",
    tool_error: str | None = None,
    duration_ms: float = 0.0,
    teammate_name: str | None = None,
    teammate_session_id: str | None = None,
    task_id: str | None = None,
    task_description: str | None = None,
    assignee: str | None = None,
    result: str | None = None,
    trigger: str | None = None,
    current_context_tokens: int | None = None,
    threshold: int | None = None,
    prompt: str | None = None,
    message_id: str | None = None,
    has_images: bool = False,
) -> HookInvocation:
    """Build the right HookInvocation subclass for *hook_type*."""
    base = ctx.model_dump()
    match hook_type:
        case HookType.POST_AGENT_TURN:
            return PostAgentTurnInvocation(**base)
        case HookType.BEFORE_TOOL:
            if tool_name is None or tool_call_id is None:
                raise ValueError(
                    "tool_name and tool_call_id are required for before_tool hooks"
                )
            return BeforeToolInvocation(
                **base,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_input=tool_input or {},
            )
        case HookType.AFTER_TOOL:
            if tool_name is None or tool_call_id is None or tool_status is None:
                raise ValueError(
                    "tool_name, tool_call_id, and tool_status are required"
                    " for after_tool hooks"
                )
            return AfterToolInvocation(
                **base,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_input=tool_input or {},
                tool_status=tool_status,
                tool_output=tool_output,
                tool_output_text=tool_output_text,
                tool_error=tool_error,
                duration_ms=duration_ms,
            )
        case HookType.TEAMMATE_IDLE:
            if teammate_name is None:
                raise ValueError("teammate_name is required for teammate_idle hooks")
            return TeammateIdleInvocation(
                **base,
                teammate_name=teammate_name,
                teammate_session_id=teammate_session_id,
            )
        case HookType.TASK_CREATED:
            if task_id is None or task_description is None:
                raise ValueError(
                    "task_id and task_description are required for task_created hooks"
                )
            return TaskCreatedInvocation(
                **base,
                task_id=task_id,
                task_description=task_description,
                assignee=assignee,
            )
        case HookType.TASK_COMPLETED:
            if task_id is None or teammate_name is None:
                raise ValueError(
                    "task_id and teammate_name are required for task_completed hooks"
                )
            return TaskCompletedInvocation(
                **base, task_id=task_id, teammate_name=teammate_name, result=result
            )
        case HookType.PRE_COMPACT:
            if trigger is None:
                raise ValueError("trigger is required for pre_compact hooks")
            return PreCompactInvocation(
                **base,
                trigger=trigger,
                current_context_tokens=current_context_tokens or 0,
                threshold=threshold or 0,
            )
        case HookType.USER_PROMPT_SUBMIT:
            if prompt is None:
                raise ValueError("prompt is required for user_prompt_submit hooks")
            return UserPromptSubmitInvocation(
                **base,
                prompt=prompt,
                message_id=message_id,
                has_images=has_images,
            )
        case _:
            assert_never(hook_type)


class HookExecutionResult(BaseModel):
    hook_name: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool


# --- Structured stdout response (exit 0 + JSON) ---


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


# --- Decision values (consumed by the agent loop) ---


class HookUserMessage(BaseModel):
    """post_agent_turn deny: ``content`` is injected as a retry user
    message.
    """

    content: str


class HookPromptBlock(BaseModel):
    """user_prompt_submit deny: the prompt is blocked; ``content`` is the
    reason surfaced to the user. No LLM turn runs.
    """

    hook_name: str
    content: str


class HookToolDenial(BaseModel):
    """before_tool deny: ``content`` becomes the tool error returned to
    the LLM.
    """

    hook_name: str
    content: str


class HookToolInputRewrite(BaseModel):
    """before_tool: one per rewriting hook in the chain. The agent loop
    validates each as it arrives — the first invalid rewrite aborts the
    chain and synthesizes a denial.
    """

    hook_name: str
    tool_input: dict[str, Any]


class HookTextReplacement(BaseModel):
    """after_tool: ``text`` is the cumulative LLM-bound output after the
    handler applied its replacement or append.
    """

    text: str


# --- Transcript / UI events (BaseEvent) ---


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
