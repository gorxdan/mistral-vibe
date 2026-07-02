from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, Protocol

from vibe.core.types import (
    AvailableTool,
    InjectedMessageKind,
    LLMChunk,
    LLMMessage,
    StrToolChoice,
)

if TYPE_CHECKING:
    from vibe.core.config import ProviderConfig


def trailing_ephemeral_count(messages: Sequence[LLMMessage]) -> int:
    """Count the trailing ephemeral MEMORY messages (the late-memory tail).

    The tail rides the absolute end of every request and is gone from that
    position in the next one; a cache entry that includes it can never
    prefix-match a later request, so history cache breakpoints must land on
    the last persisted message instead. Invariant relied on by the adapters:
    trailing MEMORY messages are user-role, non-empty string content, no
    images — each maps 1:1 to one trailing converted dict in every adapter.
    """
    count = 0
    for msg in reversed(messages):
        if not (msg.injected and msg.injected_kind == InjectedMessageKind.MEMORY):
            break
        count += 1
    return count


class PreparedRequest(NamedTuple):
    endpoint: str
    headers: dict[str, str]
    body: bytes
    base_url: str = ""


@dataclass(frozen=True)
class RequestParams:
    model_name: str
    messages: Sequence[LLMMessage]
    temperature: float | None
    tools: list[AvailableTool] | None
    max_tokens: int | None
    tool_choice: StrToolChoice | AvailableTool | None
    enable_streaming: bool
    provider: ProviderConfig
    api_key: str | None = None
    thinking: str = "off"
    verbosity: str | None = None
    response_format: dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None
    # Stable per-conversation id used as the OpenAI ``prompt_cache_key`` routing
    # pin (mirrors codex's thread_id). When absent, the OpenAI paths fall back to
    # a content hash of the prefix. Non-OpenAI providers ignore it.
    cache_session_id: str | None = None


class APIAdapter(Protocol):
    endpoint: ClassVar[str]
    # False when the adapter can't put a larger max_tokens on the wire (codex
    # rejects the param), so the agent loop must skip the escalation retries.
    supports_max_output_escalation: ClassVar[bool] = True

    def prepare_request(self, params: RequestParams) -> PreparedRequest: ...

    def parse_response(
        self, data: dict[str, Any], provider: ProviderConfig
    ) -> LLMChunk: ...
