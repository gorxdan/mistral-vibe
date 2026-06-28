from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, Protocol

from vibe.core.types import AvailableTool, LLMChunk, LLMMessage, StrToolChoice

if TYPE_CHECKING:
    from vibe.core.config import ProviderConfig


class PreparedRequest(NamedTuple):
    endpoint: str
    headers: dict[str, str]
    body: bytes
    base_url: str = ""


@dataclass(frozen=True)
class RequestParams:
    model_name: str
    messages: Sequence[LLMMessage]
    temperature: float
    tools: list[AvailableTool] | None
    max_tokens: int | None
    tool_choice: StrToolChoice | AvailableTool | None
    enable_streaming: bool
    provider: ProviderConfig
    api_key: str | None = None
    thinking: str = "off"
    response_format: dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None
    # Stable per-conversation id used as the OpenAI ``prompt_cache_key`` routing
    # pin (mirrors codex's thread_id). When absent, the OpenAI paths fall back to
    # a content hash of the prefix. Non-OpenAI providers ignore it.
    cache_session_id: str | None = None


class APIAdapter(Protocol):
    endpoint: ClassVar[str]

    def prepare_request(self, params: RequestParams) -> PreparedRequest: ...

    def parse_response(
        self, data: dict[str, Any], provider: ProviderConfig
    ) -> LLMChunk: ...
