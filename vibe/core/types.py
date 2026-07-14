from __future__ import annotations

from abc import ABC
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
import copy
from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, overload
from uuid import uuid4

if TYPE_CHECKING:
    from vibe.core.tools.base import BaseTool
    from vibe.core.tools.permissions import RequiredPermission
else:
    BaseTool = Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    model_validator,
)

from vibe.core.experiments.models import EvalResponse
from vibe.core.logger import logger
from vibe.core.tasking import TaskOutcome


class ScheduledLoop(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    interval_seconds: int
    prompt: str
    next_fire_at: float
    created_at: float
    # Recurring loops re-arm after firing; one-shot loops fire once then drop.
    recurring: bool = True


class Backend(StrEnum):
    MISTRAL = auto()
    GENERIC = auto()


class AgentStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: int = 0
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0
    tool_calls_agreed: int = 0
    tool_calls_rejected: int = 0
    tool_calls_hook_denied: int = 0
    tool_calls_failed: int = 0
    tool_calls_succeeded: int = 0

    context_tokens: int = 0
    # Active model's soft compaction limit (mirrors update_pricing); 0 = disabled.
    auto_compact_threshold: int = 0

    # Prompt tokens served from the provider's cache (subset of prompt tokens).
    session_cached_tokens: int = 0
    last_turn_cached_tokens: int = 0
    # Prompt tokens written to a provider cache (also a subset of prompt tokens).
    session_cache_write_tokens: int = 0
    last_turn_cache_write_tokens: int = 0
    # Reasoning tokens are a subset of completion tokens when reported.
    session_reasoning_tokens: int = 0
    last_turn_reasoning_tokens: int = 0

    last_turn_prompt_tokens: int = 0
    last_turn_completion_tokens: int = 0
    last_turn_duration: float = 0.0
    tokens_per_second: float = 0.0

    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0

    # Exact per-call cost; session_cost returns this so a model switch can't reprice the session.
    accumulated_cost_usd: float = 0.0
    # Distinguishes an exact $0 session from one that has not yet recorded a quote.
    accumulated_cost_initialized: bool = False
    # A fallback quote is safe for local budget enforcement but must not be sent
    # as an exact ACP cost.
    cost_is_estimated: bool = False

    _listeners: dict[str, Callable[[AgentStats], None]] = PrivateAttr(
        default_factory=dict
    )

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name in self._listeners:
            self._listeners[name](self)

    def trigger_listeners(self) -> None:
        for listener in self._listeners.values():
            listener(self)

    def add_listener(
        self, attr_name: str, listener: Callable[[AgentStats], None]
    ) -> None:
        self._listeners[attr_name] = listener

    @staticmethod
    def create_fresh(previous: AgentStats) -> AgentStats:
        fresh = AgentStats()
        fresh._listeners = previous._listeners.copy()
        return fresh

    @computed_field
    @property
    def session_total_llm_tokens(self) -> int:
        return self.session_prompt_tokens + self.session_completion_tokens

    @computed_field
    @property
    def last_turn_total_tokens(self) -> int:
        return self.last_turn_prompt_tokens + self.last_turn_completion_tokens

    @computed_field
    @property
    def cache_hit_ratio(self) -> float:
        """Fraction of prompt tokens served from cache (0..1), clamped."""
        if self.session_prompt_tokens <= 0:
            return 0.0
        cached = min(self.session_cached_tokens, self.session_prompt_tokens)
        return cached / self.session_prompt_tokens

    @computed_field
    @property
    def session_cost(self) -> float:
        """Total session cost in USD.

        Prefer the per-call accumulator (`accumulated_cost_usd`), which is exact
        and immune to mid-session model switches. Fall back to the pricing-fields
        recompute for stats constructed without going through `_update_stats`
        (tests, manual construction) so the field stays meaningful there too.
        """
        if self.accumulated_cost_initialized or self.accumulated_cost_usd > 0.0:
            return self.accumulated_cost_usd
        input_cost = (
            self.session_prompt_tokens / 1_000_000
        ) * self.input_price_per_million
        output_cost = (
            self.session_completion_tokens / 1_000_000
        ) * self.output_price_per_million
        return input_cost + output_cost

    @computed_field
    @property
    def tokens_until_compaction(self) -> int:
        """Tokens remaining before auto-compaction fires (>=0).

        Derived from the one-turn-behind context_tokens, so treat as advisory.
        """
        if self.auto_compact_threshold <= 0:
            return 0
        return max(0, self.auto_compact_threshold - self.context_tokens)

    def update_pricing(self, input_price: float, output_price: float) -> None:
        """Update pricing info when model changes.

        Emits a one-line cache-impact warning when a real pricing change coincides
        with an established cache: switching providers mid-session invalidates the
        prefix cache (observed cache-hit ratio collapsing ~90% -> ~47% in
        multi-model sessions), roughly doubling effective input cost. Surfacing the
        pre-switch ratio at the switch site keeps the trade-off visible.
        """
        price_changed = (
            input_price != self.input_price_per_million
            or output_price != self.output_price_per_million
        )
        if (
            price_changed
            and self.session_prompt_tokens > 0
            and self.session_cached_tokens > 0
        ):
            logger.warning(
                "Model/pricing switch at %.0f%% cache hit (%s cached / %s prompt "
                "tokens) — the new model's prefix cache starts cold, so expect a "
                "cache-hit drop and higher effective input cost over the next turns",
                self.cache_hit_ratio * 100,
                self.session_cached_tokens,
                self.session_prompt_tokens,
            )
        self.input_price_per_million = input_price
        self.output_price_per_million = output_price

    def update_model_bounds(self, auto_compact_threshold: int) -> None:
        """Sync the active model's auto-compact threshold (mirrors update_pricing)."""
        self.auto_compact_threshold = max(0, auto_compact_threshold)

    def reset_context_state(self) -> None:
        """Reset context-related fields while preserving cumulative session stats.

        Used after config reload or similar operations where the context
        changes but we want to preserve session totals.
        """
        self.context_tokens = 0
        self.last_turn_prompt_tokens = 0
        self.last_turn_completion_tokens = 0
        self.last_turn_cached_tokens = 0
        self.last_turn_cache_write_tokens = 0
        self.last_turn_reasoning_tokens = 0
        self.last_turn_duration = 0.0
        self.tokens_per_second = 0.0


class SessionInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    session_id: str
    start_time: str
    message_count: int
    stats: AgentStats
    save_dir: str


class SessionMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")
    session_id: str
    parent_session_id: str | None = None
    start_time: str
    end_time: str | None
    git_commit: str | None
    git_branch: str | None
    environment: dict[str, str | None]
    username: str
    loops: list[ScheduledLoop] = Field(default_factory=list)
    title: str | None = None
    title_source: Literal["auto", "manual"] = "auto"
    experiments: EvalResponse | None = None
    workflow_snapshots: list[dict[str, Any]] = Field(default_factory=list)


StrToolChoice = Literal["auto", "none", "any", "required"]


class AvailableFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, Any]


class AvailableTool(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["function"] = "function"
    function: AvailableFunction


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str | None = None
    arguments: str | None = None


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str | None = None
    index: int | None = None
    function: FunctionCall = Field(default_factory=FunctionCall)
    type: Literal["function"] = "function"


def _content_before(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        parts: list[str] = []
        for p in v:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(v)


Content = Annotated[str, BeforeValidator(_content_before)]


class Role(StrEnum):
    SYSTEM = auto()
    USER = auto()
    ASSISTANT = auto()
    TOOL = auto()


class InjectedMessageKind(StrEnum):
    USER_CONTEXT = auto()
    STAGED = auto()
    SESSION_START = auto()
    USER_PROMPT_HOOK = auto()
    STOP_HOOK = auto()
    POST_AGENT_TURN_HOOK = auto()
    MIDDLEWARE = auto()
    PLAN_UPDATE = auto()
    BACKGROUND_TASK = auto()
    COMPACTION_CONTEXT = auto()
    MEMORY = auto()


class ApprovalResponse(StrEnum):
    YES = "y"
    NO = "n"
    MODIFY = "m"


IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})
MAX_IMAGE_BYTES: int = 10 * 1024 * 1024
MAX_IMAGES_PER_MESSAGE: int = 8


class FileImageSource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["file"] = "file"
    path: Path


class InlineImageSource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["inline"] = "inline"
    # Raw base64-encoded bytes (no `data:` prefix). Used when the image has no
    # durable file on disk (session logging disabled): memory-only, never
    # persisted to a session transcript.
    data: str


class ImageAttachment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: Annotated[FileImageSource | InlineImageSource, Field(discriminator="kind")]
    alias: str
    mime_type: str

    @model_validator(mode="before")
    @classmethod
    def _migrate_flat_source(cls, value: Any) -> Any:
        # Accept and migrate the legacy flat shape `{path|data, ...}` from older
        # session transcripts.
        if not isinstance(value, dict) or "source" in value:
            return value
        if value.get("path") is not None:
            return {**value, "source": {"kind": "file", "path": value["path"]}}
        if value.get("data") is not None:
            return {**value, "source": {"kind": "inline", "data": value["data"]}}
        return value


class LLMMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Role
    content: Content | None = None
    images: list[ImageAttachment] | None = None
    injected: bool = False
    injected_kind: InjectedMessageKind | None = None
    reasoning_content: Content | None = None
    reasoning_state: list[str] | None = None
    reasoning_signature: str | None = None
    reasoning_message_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    message_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_any(cls, v: Any) -> dict[str, Any] | Any:
        if isinstance(v, dict):
            v.setdefault("content", "")
            v.setdefault("role", "assistant")
            if v.get("message_id") is None and v.get("role") != "tool":
                v["message_id"] = str(uuid4())
            if v.get("reasoning_message_id") is None and v.get("reasoning_content"):
                v["reasoning_message_id"] = str(uuid4())
            return v
        role = str(getattr(v, "role", "assistant"))
        reasoning_content = getattr(v, "reasoning_content", None)
        return {
            "role": role,
            "content": getattr(v, "content", ""),
            "reasoning_content": reasoning_content,
            "reasoning_state": getattr(v, "reasoning_state", None),
            "reasoning_signature": getattr(v, "reasoning_signature", None),
            "reasoning_message_id": getattr(v, "reasoning_message_id", None)
            or (str(uuid4()) if reasoning_content else None),
            "tool_calls": getattr(v, "tool_calls", None),
            "name": getattr(v, "name", None),
            "tool_call_id": getattr(v, "tool_call_id", None),
            "images": getattr(v, "images", None),
            "injected": getattr(v, "injected", False),
            "injected_kind": getattr(v, "injected_kind", None),
            "message_id": getattr(v, "message_id", None)
            or (str(uuid4()) if role != "tool" else None),
        }

    def __add__(self, other: LLMMessage) -> LLMMessage:
        """Careful: this is not commutative!"""
        if self.role != other.role:
            raise ValueError("Can't accumulate messages with different roles")

        if self.name != other.name:
            raise ValueError("Can't accumulate messages with different names")

        if self.tool_call_id != other.tool_call_id:
            raise ValueError("Can't accumulate messages with different tool_call_ids")

        content = (self.content or "") + (other.content or "")
        if not content:
            content = None

        reasoning_content = (self.reasoning_content or "") + (
            other.reasoning_content or ""
        )
        if not reasoning_content:
            reasoning_content = None

        reasoning_signature = (self.reasoning_signature or "") + (
            other.reasoning_signature or ""
        )
        if not reasoning_signature:
            reasoning_signature = None

        reasoning_state: list[str] | None = None
        if self.reasoning_state or other.reasoning_state:
            reasoning_state = [
                *(self.reasoning_state or []),
                *(other.reasoning_state or []),
            ]

        tool_calls_map = OrderedDict[int, ToolCall]()
        for tool_calls in [self.tool_calls or [], other.tool_calls or []]:
            for tc in tool_calls:
                if tc.index is None:
                    raise ValueError("Tool call chunk missing index")
                if tc.index not in tool_calls_map:
                    tool_calls_map[tc.index] = copy.deepcopy(tc)
                else:
                    existing_name = tool_calls_map[tc.index].function.name
                    new_name = tc.function.name
                    if existing_name and new_name and existing_name != new_name:
                        raise ValueError(
                            "Can't accumulate messages with different tool call names"
                        )
                    if new_name and not existing_name:
                        tool_calls_map[tc.index].function.name = new_name
                    new_args = (tool_calls_map[tc.index].function.arguments or "") + (
                        tc.function.arguments or ""
                    )
                    tool_calls_map[tc.index].function.arguments = new_args

        return LLMMessage(
            role=self.role,
            content=content,
            images=self.images if self.images is not None else other.images,
            injected=self.injected or other.injected,
            injected_kind=self.injected_kind or other.injected_kind,
            reasoning_content=reasoning_content,
            reasoning_state=reasoning_state,
            reasoning_signature=reasoning_signature,
            reasoning_message_id=self.reasoning_message_id
            or other.reasoning_message_id,
            tool_calls=list(tool_calls_map.values()) or None,
            name=self.name,
            tool_call_id=self.tool_call_id,
            message_id=self.message_id,
        )


class LLMMessageAccumulator:
    """Accumulates streamed delta messages into a single LLMMessage in O(n) total.

    Folding a stream with ``LLMMessage.__add__`` per chunk re-concatenates the
    whole accumulated content (and every tool-call argument string, plus a
    deepcopy of every tool call) on each delta, which is O(n^2) over a response.
    This builds the same final message by appending fragments and joining once.

    It mirrors ``__add__``'s left-fold semantics exactly: identical validation,
    content/reasoning/signature concatenated in arrival order, first-seen value
    wins for ids/images/name/role/tool_call_id, and tool calls merged by index
    with their argument strings concatenated. ``build()`` constructs the result
    via the same ``LLMMessage(...)`` constructor, so the model validators run as
    they would for ``__add__``.
    """

    def __init__(self) -> None:
        self._started = False
        self._role: Role | None = None
        self._name: str | None = None
        self._tool_call_id: str | None = None
        self._reasoning_message_id: str | None = None
        self._message_id: str | None = None
        self._images: list[ImageAttachment] | None = None
        self._images_set = False
        self._content: list[str] = []
        self._reasoning_content: list[str] = []
        self._reasoning_signature: list[str] = []
        self._reasoning_state: list[str] = []
        self._tool_calls = OrderedDict[int, ToolCall]()
        self._tool_call_arg_parts = OrderedDict[int, list[str]]()

    @property
    def empty(self) -> bool:
        return not self._started

    def add(self, message: LLMMessage) -> None:
        if not self._started:
            self._init_header(message)
        else:
            self._merge_header(message)

        if message.content:
            self._content.append(message.content)
        if message.reasoning_content:
            self._reasoning_content.append(message.reasoning_content)
        if message.reasoning_signature:
            self._reasoning_signature.append(message.reasoning_signature)
        if message.reasoning_state:
            self._reasoning_state.extend(message.reasoning_state)

        for tc in message.tool_calls or []:
            self._merge_tool_call(tc)

    def _init_header(self, message: LLMMessage) -> None:
        self._started = True
        self._role = message.role
        self._name = message.name
        self._tool_call_id = message.tool_call_id
        self._reasoning_message_id = message.reasoning_message_id
        self._message_id = message.message_id
        self._images = message.images
        self._images_set = message.images is not None

    def _merge_header(self, message: LLMMessage) -> None:
        if self._role != message.role:
            raise ValueError("Can't accumulate messages with different roles")
        if self._name != message.name:
            raise ValueError("Can't accumulate messages with different names")
        if self._tool_call_id != message.tool_call_id:
            raise ValueError("Can't accumulate messages with different tool_call_ids")
        if self._reasoning_message_id is None:
            self._reasoning_message_id = message.reasoning_message_id
        if not self._images_set and message.images is not None:
            self._images = message.images
            self._images_set = True

    def _merge_tool_call(self, tc: ToolCall) -> None:
        if tc.index is None:
            raise ValueError("Tool call chunk missing index")
        if tc.index not in self._tool_calls:
            self._tool_calls[tc.index] = copy.deepcopy(tc)
            self._tool_call_arg_parts[tc.index] = []
            return
        existing = self._tool_calls[tc.index]
        existing_name = existing.function.name
        new_name = tc.function.name
        if existing_name and new_name and existing_name != new_name:
            raise ValueError("Can't accumulate messages with different tool call names")
        if new_name and not existing_name:
            existing.function.name = new_name
        self._tool_call_arg_parts[tc.index].append(tc.function.arguments or "")

    def build(self) -> LLMMessage:
        content = "".join(self._content) or None
        reasoning_content = "".join(self._reasoning_content) or None
        reasoning_signature = "".join(self._reasoning_signature) or None
        reasoning_state = list(self._reasoning_state) if self._reasoning_state else None

        for index, tc in self._tool_calls.items():
            extra = self._tool_call_arg_parts[index]
            if extra:
                tc.function.arguments = (tc.function.arguments or "") + "".join(extra)

        return LLMMessage(
            role=self._role or Role.ASSISTANT,
            content=content,
            images=self._images,
            reasoning_content=reasoning_content,
            reasoning_state=reasoning_state,
            reasoning_signature=reasoning_signature,
            reasoning_message_id=self._reasoning_message_id,
            tool_calls=list(self._tool_calls.values()) or None,
            name=self._name,
            tool_call_id=self._tool_call_id,
            message_id=self._message_id,
        )


class LLMUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", allow_inf_nan=False)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    # Prompt tokens served from the provider's cache (subset of prompt_tokens).
    # Lets cache effectiveness be measured for OpenAI-compatible providers that
    # auto-cache (e.g. Kimi, GLM) without Vibe managing cache breakpoints.
    cached_tokens: int = Field(default=0, ge=0)
    # Prompt tokens written to a provider cache. They are distinct from cache
    # reads because providers can bill writes at a different rate.
    cache_write_tokens: int = Field(default=0, ge=0)
    # Reasoning tokens (subset of completion_tokens for o-series / GLM / Kimi
    # thinking models). Surfaced from completion_tokens_details.reasoning_tokens
    # (OpenAI shape) so totals reflect the API's actual billed usage.
    reasoning_tokens: int = Field(default=0, ge=0)
    # Authoritative provider-reported charge for the call (for example,
    # OpenRouter's usage.cost). None means Vibe must quote from token rates.
    reported_cost_usd: float | None = Field(default=None, ge=0.0)

    def __add__(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            reported_cost_usd=(
                None
                if self.reported_cost_usd is None and other.reported_cost_usd is None
                else (self.reported_cost_usd or 0.0) + (other.reported_cost_usd or 0.0)
            ),
        )


class StopReason(StrEnum):
    REFUSAL = "refusal"
    LENGTH = "length"


# Finish reasons signalling the response hit its output-token ceiling. OpenAI
# chat uses "length"; some compatible servers use "max_tokens".
_TRUNCATION_REASONS = frozenset({StopReason.LENGTH, "max_tokens"})


class StopInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    reason: str | None = None
    category: str | None = None
    explanation: str | None = None

    @property
    def is_refusal(self) -> bool:
        return self.reason == StopReason.REFUSAL

    @property
    def is_truncated(self) -> bool:
        return self.reason in _TRUNCATION_REASONS

    @staticmethod
    def from_chat_choices(data: dict[str, Any]) -> StopInfo | None:
        choices = data.get("choices")
        if not choices:
            return None
        reason = choices[0].get("finish_reason")
        return StopInfo(reason=reason) if reason else None


class LLMChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    message: LLMMessage
    usage: LLMUsage | None = None
    correlation_id: str | None = None
    stop: StopInfo | None = None

    def __add__(self, other: LLMChunk) -> LLMChunk:
        if self.usage is None and other.usage is None:
            new_usage = None
        else:
            new_usage = (self.usage or LLMUsage()) + (other.usage or LLMUsage())
        return LLMChunk(
            message=self.message + other.message,
            usage=new_usage,
            correlation_id=other.correlation_id or self.correlation_id,
            stop=other.stop or self.stop,
        )


class LLMChunkAccumulator:
    """Accumulates streamed LLMChunks into one final chunk in O(n) total.

    The streaming equivalent of folding with ``LLMChunk.__add__`` per delta, but
    without the per-chunk O(n^2) message rebuild (see LLMMessageAccumulator).
    Usage is summed; the last non-None stop wins. ``usage`` exposes the running
    total so callers don't need to track it separately.
    """

    def __init__(self) -> None:
        self._message = LLMMessageAccumulator()
        self._usage = LLMUsage()
        self._saw_usage = False
        self._stop: StopInfo | None = None
        self._correlation_id: str | None = None

    @property
    def empty(self) -> bool:
        return self._message.empty

    @property
    def usage(self) -> LLMUsage:
        return self._usage

    def add(self, chunk: LLMChunk) -> None:
        self._message.add(chunk.message)
        if chunk.usage is not None:
            self._usage += chunk.usage
            self._saw_usage = True
        if chunk.stop is not None:
            self._stop = chunk.stop
        if chunk.correlation_id:
            self._correlation_id = chunk.correlation_id

    def build(self) -> LLMChunk | None:
        if self._message.empty:
            return None
        return LLMChunk(
            message=self._message.build(),
            usage=self._usage if self._saw_usage else None,
            correlation_id=self._correlation_id,
            stop=self._stop,
        )


class BaseEvent(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class UserMessageEvent(BaseEvent):
    content: str
    message_id: str


class AssistantEvent(BaseEvent):
    content: str
    stopped_by_middleware: bool = False
    message_id: str | None = None

    def __add__(self, other: AssistantEvent) -> AssistantEvent:
        return AssistantEvent(
            content=self.content + other.content,
            stopped_by_middleware=self.stopped_by_middleware
            or other.stopped_by_middleware,
            message_id=self.message_id or other.message_id,
        )


class ReasoningEvent(BaseEvent):
    content: str
    message_id: str | None = None


class ToolCallEvent(BaseEvent):
    tool_call_id: str
    tool_name: str
    tool_class: type[BaseTool]
    tool_call_index: int | None = None
    args: BaseModel | None = None


class ToolResultEvent(BaseEvent):
    tool_name: str
    tool_class: type[BaseTool] | None
    result: BaseModel | None = None
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    cancelled: bool = False
    duration: float | None = None
    tool_call_id: str
    # Set when the LLM safety judge auto-approved this call (shown in-session).
    approval_note: str | None = None


class ToolStreamEvent(BaseEvent):
    tool_name: str
    message: str
    tool_call_id: str


class WaitingForInputEvent(BaseEvent):
    task_id: str
    label: str | None = None
    predefined_answers: list[str] | None = None


class CompactStartEvent(BaseEvent):
    current_context_tokens: int
    threshold: int
    # WORKAROUND: Using tool_call to communicate compact events to the client.
    # This should be revisited when the ACP protocol defines how compact events
    # should be represented.
    # [RFD](https://agentclientprotocol.com/rfds/session-usage)
    tool_call_id: str


class CompactionOrigin(BaseModel):
    """Snapshot of what shaped the conversation when a compaction ran.

    Pure observability — recorded on CompactEndEvent for debugging, session
    analytics, and downstream tooling. Not consulted by the turn loop.
    """

    model_config = ConfigDict(extra="forbid")
    model_alias: str
    agent_profile: str
    system_prompt_hash: str


class CompactEndEvent(BaseEvent):
    summary_length: int
    old_session_id: str | None = None
    new_session_id: str | None = None
    origin: CompactionOrigin | None = None
    # WORKAROUND: Using tool_call to communicate compact events to the client.
    # This should be revisited when the ACP protocol defines how compact events
    # should be represented.
    # [RFD](https://agentclientprotocol.com/rfds/session-usage)
    tool_call_id: str


class PlanReviewRequestedEvent(BaseEvent):
    file_path: Path


class PlanReviewEndedEvent(BaseEvent):
    pass


class AgentProfileChangedEvent(BaseEvent):
    """Emitted when the active agent profile changes during a turn."""

    agent_name: str


class SessionTitleUpdatedEvent(BaseEvent):
    title: str


class BackgroundTaskCompletedEvent(BaseEvent):
    """Surfaces completion of an async/background subagent to the parent loop.

    Emitted at the top of the next parent turn after the async subagent
    finishes (the registry queues completions; the agent loop drains them).
    A matching user-role ``LLMMessage`` is appended alongside so the model
    sees the result as context for the upcoming turn.
    """

    task_id: str
    agent: str
    response: str
    completed: bool
    worktree_path: str | None = None
    branch: str | None = None
    error: str | None = None
    outcome: TaskOutcome | None = None


class OutputFormat(StrEnum):
    TEXT = auto()
    JSON = auto()
    STREAMING = auto()


type ApprovalCallback = Callable[
    [str, BaseModel, str, list[RequiredPermission] | None, str | None],
    Awaitable[tuple[ApprovalResponse, str | None, dict[str, Any] | None]],
]


type UserInputCallback = Callable[[BaseModel], Awaitable[BaseModel]]

type SwitchAgentCallback = Callable[[str], Awaitable[None]]

# Asks the host to let the user pick a model when a turn is rate-limited and no
# automatic fallback is available. Given (provider, model, candidate_aliases),
# returns the chosen model alias to switch to and retry, or None to surface the
# rate-limit error (no callback / user declined).
type RateLimitCallback = Callable[[str, str, list[str]], Awaitable[str | None]]


class MessageList(Sequence[LLMMessage]):
    def __init__(
        self,
        initial: list[LLMMessage] | None = None,
        observer: Callable[[LLMMessage], None] | None = None,
    ) -> None:
        self._data: list[LLMMessage] = list(initial) if initial else []
        self._observer = observer
        self._reset_hooks: list[Callable[[], None]] = []
        self._silent = False
        if self._observer:
            for msg in self._data:
                self._observer(msg)

    def _notify(self, msg: LLMMessage) -> None:
        if not self._silent and self._observer is not None:
            self._observer(msg)

    def append(self, msg: LLMMessage) -> None:
        self._data.append(msg)
        self._notify(msg)

    def insert(self, i: int, msg: LLMMessage) -> None:
        self._data.insert(i, msg)

    def extend(self, msgs: list[LLMMessage]) -> None:
        for msg in msgs:
            self.append(msg)

    def on_reset(self, hook: Callable[[], None]) -> None:
        """Register a callback that fires whenever the list is reset."""
        self._reset_hooks.append(hook)

    def reset(self, new: list[LLMMessage]) -> None:
        """Replace contents silently (never notifies)."""
        self._data = list(new)
        for hook in self._reset_hooks:
            hook()

    def replace_at(self, index: int, msg: LLMMessage) -> None:
        """Replace the message at ``index`` in place, silently.

        Used by context-shaper middlewares to rewrite (snip/compress) an old
        message without firing observer notifications or reset hooks. The caller
        builds the replacement message; structure (role, tool linkage) is its
        responsibility. Atomic under CPython's GIL like update_system_prompt.
        """
        self._data[index] = msg

    def notify_at(self, index: int) -> None:
        self._notify(self._data[index])

    def update_system_prompt(self, new: str, *, notify: bool = False) -> None:
        """Replace the system prompt, or insert it if none exists yet.

        Called from a background thread during deferred init.  Under deferred
        init the prompt can land after messages were already appended, so
        insert at the front rather than clobber slot 0.  A single list-item
        assignment is atomic under CPython's GIL, and the ``@requires_init``
        decorator ensures no ``act()`` call reads the prompt concurrently, so
        no additional lock is needed here.
        """
        msg = LLMMessage(role=Role.SYSTEM, content=new)
        if self._data and self._data[0].role == Role.SYSTEM:
            self._data[0] = msg
        else:
            self._data.insert(0, msg)
        if notify:
            self._notify(msg)

    @contextmanager
    def silent(self) -> Iterator[None]:
        """Context manager that suppresses notifications."""
        prev = self._silent
        self._silent = True
        try:
            yield
        finally:
            self._silent = prev

    def __len__(self) -> int:
        return len(self._data)

    @overload
    def __getitem__(self, index: int) -> LLMMessage: ...
    @overload
    def __getitem__(self, index: slice) -> list[LLMMessage]: ...
    def __getitem__(self, index: int | slice) -> LLMMessage | list[LLMMessage]:
        return self._data[index]

    def __iter__(self) -> Iterator[LLMMessage]:
        return iter(self._data)

    def __contains__(self, item: object) -> bool:
        return item in self._data

    def __bool__(self) -> bool:
        return bool(self._data)


class RateLimitError(Exception):
    failover_hint: str | None = None

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__(
            "Rate limits exceeded. Please wait a moment before trying again."
        )


class ContextTooLongError(Exception):
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__(
            "The conversation context exceeds the model's maximum limit. "
            "Use /rewind to undo recent actions, then /compact to summarize the conversation."
        )


class ResponseTooLongError(Exception):
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__(
            "The model's response exceeded the maximum output token limit."
        )


class ContentFilterError(Exception):
    failover_hint: str | None = None

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__(f"The request was blocked by the {provider} content filter.")


class ServerError(Exception):
    failover_hint: str | None = None

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__(f"The {provider} backend returned a persistent server error.")


class TransportError(Exception):
    failover_hint: str | None = None

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__(
            f"The {provider} backend dropped the connection before responding."
        )


class UnclassifiedBackendError(Exception):
    """API error that did not match a typed recovery class.

    Raised by ``_raise_for_backend_error`` for residual failures so the
    conversation loop can attempt failover instead of aborting the turn with a
    bare ``RuntimeError``.
    """

    failover_hint: str | None = None

    def __init__(self, provider: str, model: str, detail: str) -> None:
        self.provider = provider
        self.model = model
        self.detail = detail
        super().__init__(f"API error from {provider} (model: {model}): {detail}")


class RefusalError(Exception):
    def __init__(
        self,
        provider: str,
        model: str,
        category: str | None = None,
        explanation: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.category = category
        self.explanation = explanation
        super().__init__(self._fmt())

    def _fmt(self) -> str:
        lead = "The model declined to respond to this request and stopped early."
        if self.category:
            lead += f" (category: {self.category})"
        detail = self.explanation or (
            "Try rephrasing your request or starting a new conversation."
        )
        return f"{lead} {detail}"
