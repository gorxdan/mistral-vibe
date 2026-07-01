from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.fake_backend import FakeBackend
from vibe.core.config import (
    MaxOutputEscalationConfig,
    ModelConfig,
    ProviderConfig,
    VibeConfig,
)
from vibe.core.types import (
    BaseEvent,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    ResponseTooLongError,
    Role,
    StopInfo,
)

# --------------------------------------------------------------------------- #
# _escalate_max_output numeric behavior                                        #
# --------------------------------------------------------------------------- #


def test_escalation_grows_geometrically_then_caps() -> None:
    loop = build_test_agent_loop()  # defaults: base 8192, factor 2, cap 65536, 3 tries
    assert loop._escalate_max_output() == 16384
    assert loop._max_output_override == 16384
    assert loop._escalate_max_output() == 32768
    assert loop._escalate_max_output() == 65536  # hits cap
    assert loop._escalate_max_output() is None  # attempts exhausted


def test_escalation_clamped_to_cap_then_stops() -> None:
    cfg = build_test_vibe_config(
        max_output_escalation=MaxOutputEscalationConfig(cap=10000)
    )
    loop = build_test_agent_loop(config=cfg)
    assert loop._escalate_max_output() == 10000  # min(16384, cap)
    assert loop._escalate_max_output() is None  # pinned at cap, can't grow


def test_escalation_disabled_returns_none() -> None:
    cfg = build_test_vibe_config(
        max_output_escalation=MaxOutputEscalationConfig(enabled=False)
    )
    loop = build_test_agent_loop(config=cfg)
    assert loop._escalate_max_output() is None


def test_per_model_max_output_tokens_caps_escalation() -> None:
    cfg = build_test_vibe_config()
    cfg.models[0].max_output_tokens = 12000
    cfg.active_model = cfg.models[0].alias
    loop = build_test_agent_loop(config=cfg)
    assert loop._escalate_max_output() == 12000  # model cap wins over global cap
    assert loop._escalate_max_output() is None


# --------------------------------------------------------------------------- #
# Loop integration: retry on ResponseTooLongError                              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_response_too_long_escalates_and_retries() -> None:
    loop = build_test_agent_loop()
    calls = {"turn": 0}

    async def fake_turn() -> AsyncGenerator[BaseEvent, None]:
        calls["turn"] += 1
        if calls["turn"] == 1:
            raise ResponseTooLongError("prov", "model")
        return
        yield  # pragma: no cover

    loop._perform_llm_turn = fake_turn  # type: ignore[method-assign]

    events = [e async for e in loop._conversation_loop("hi")]
    assert calls["turn"] == 2, "retried once after truncation"
    assert loop._max_output_override == 16384
    assert events


@pytest.mark.asyncio
async def test_response_too_long_exhausts_then_surfaces() -> None:
    loop = build_test_agent_loop()  # max_attempts 3

    async def always_truncate() -> AsyncGenerator[BaseEvent, None]:
        raise ResponseTooLongError("prov", "model")
        yield  # pragma: no cover

    loop._perform_llm_turn = always_truncate  # type: ignore[method-assign]

    with pytest.raises(ResponseTooLongError):
        _ = [e async for e in loop._conversation_loop("hi")]
    # 3 escalations consumed (attempts 1..3), the 4th returns None -> surface
    assert loop._response_too_long_attempts == 4


@pytest.mark.asyncio
async def test_disabled_surfaces_immediately() -> None:
    cfg = build_test_vibe_config(
        max_output_escalation=MaxOutputEscalationConfig(enabled=False)
    )
    loop = build_test_agent_loop(config=cfg)

    async def always_truncate() -> AsyncGenerator[BaseEvent, None]:
        raise ResponseTooLongError("prov", "model")
        yield  # pragma: no cover

    loop._perform_llm_turn = always_truncate  # type: ignore[method-assign]

    with pytest.raises(ResponseTooLongError):
        _ = [e async for e in loop._conversation_loop("hi")]
    assert loop._max_output_override is None


@pytest.mark.asyncio
async def test_override_resets_between_turns() -> None:
    loop = build_test_agent_loop()
    state = {"turn": 0}

    async def truncate_once_per_first_turn() -> AsyncGenerator[BaseEvent, None]:
        state["turn"] += 1
        if state["turn"] == 1:
            raise ResponseTooLongError("prov", "model")
        return
        yield  # pragma: no cover

    loop._perform_llm_turn = truncate_once_per_first_turn  # type: ignore[method-assign]
    _ = [e async for e in loop._conversation_loop("first")]
    assert loop._max_output_override == 16384

    # Second user turn must start fresh.
    state["turn"] = 10  # so it succeeds immediately
    _ = [e async for e in loop._conversation_loop("second")]
    assert loop._max_output_override is None
    assert loop._response_too_long_attempts == 0


@pytest.mark.asyncio
async def test_chat_passes_override_to_backend_but_not_compaction() -> None:
    from vibe.core.types import LLMChunk, LLMMessage, LLMUsage, Role

    loop = build_test_agent_loop()
    seen: list[int | None] = []

    class _Spy:
        async def __aenter__(self) -> _Spy:
            return self

        async def __aexit__(self, *a: object) -> None:
            return None

        async def complete(self, request, *, response_headers_sink=None):
            seen.append(request.max_tokens)
            return LLMChunk(
                message=LLMMessage(role=Role.ASSISTANT, content="ok"),
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1),
            )

    loop.backend = _Spy()  # type: ignore[assignment]
    loop._max_output_override = 16384

    await loop._chat()  # main-turn call inherits override
    await loop._chat(model_override=loop.config.get_active_model())  # compaction-like

    assert seen == [16384, None]


@pytest.mark.asyncio
async def test_chat_raises_on_truncated_finish_reason_without_committing() -> None:
    from vibe.core.types import LLMChunk, LLMMessage, LLMUsage, Role, StopInfo

    loop = build_test_agent_loop()

    class _Trunc:
        async def __aenter__(self) -> _Trunc:
            return self

        async def __aexit__(self, *a: object) -> None:
            return None

        async def complete(self, request, *, response_headers_sink=None):
            return LLMChunk(
                message=LLMMessage(role=Role.ASSISTANT, content="partial..."),
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1),
                stop=StopInfo(reason="length"),
            )

    loop.backend = _Trunc()  # type: ignore[assignment]
    before = len(loop.messages)
    with pytest.raises(ResponseTooLongError):
        await loop._chat()
    assert len(loop.messages) == before, "truncated turn must not enter history"


@pytest.mark.asyncio
async def test_chat_raises_on_openai_responses_incomplete_max_output_tokens() -> None:
    from vibe.core.config import ProviderConfig
    from vibe.core.llm.backend.openai_responses import OpenAIResponsesAdapter

    loop = build_test_agent_loop()
    provider = ProviderConfig(
        name="openai",
        api_base="https://api.openai.com/v1",
        api_key_env_var="OPENAI_API_KEY",
        api_style="openai-responses",
    )
    incomplete_body = {
        "id": "resp_123",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "partial..."}],
                "role": "assistant",
            }
        ],
        "usage": {"input_tokens": 50, "output_tokens": 25},
    }

    class _ResponsesTrunc:
        async def __aenter__(self) -> _ResponsesTrunc:
            return self

        async def __aexit__(self, *a: object) -> None:
            return None

        async def complete(self, request, *, response_headers_sink=None):
            return OpenAIResponsesAdapter().parse_response(incomplete_body, provider)

    loop.backend = _ResponsesTrunc()  # type: ignore[assignment]
    before = len(loop.messages)
    with pytest.raises(ResponseTooLongError):
        await loop._chat()
    assert len(loop.messages) == before, "truncated turn must not enter history"


# --- Adapter capability: codex cannot raise the output cap ------------------ #


def _truncated_chunk() -> LLMChunk:
    return LLMChunk(
        message=LLMMessage(role=Role.ASSISTANT, content="partial..."),
        usage=LLMUsage(prompt_tokens=1, completion_tokens=1),
        stop=StopInfo(reason="length"),
    )


def _single_provider_config(api_style: str) -> VibeConfig:
    provider = ProviderConfig(
        name="prov",
        api_base="https://api.example.com/v1",
        api_key_env_var="",
        api_style=api_style,
    )
    model = ModelConfig(name="test-model", provider="prov", alias="test-model")
    return build_test_vibe_config(providers=[provider], models=[model])


@pytest.mark.asyncio
async def test_codex_truncation_goes_terminal_without_retrying() -> None:
    backend = FakeBackend([[_truncated_chunk()] for _ in range(5)])
    loop = build_test_agent_loop(
        config=_single_provider_config("openai-chatgpt"), backend=backend
    )
    with pytest.raises(ResponseTooLongError):
        _ = [e async for e in loop._conversation_loop("hi")]
    # Codex strips max_output_tokens from the wire, so an escalated retry would
    # re-send an identical request: exactly one attempt, then terminal.
    assert backend.requests_max_tokens == [None]
    assert loop._max_output_override is None


@pytest.mark.asyncio
async def test_escalation_capable_adapter_still_retries_with_larger_caps() -> None:
    backend = FakeBackend([[_truncated_chunk()] for _ in range(5)])
    loop = build_test_agent_loop(
        config=_single_provider_config("openai-responses"), backend=backend
    )
    with pytest.raises(ResponseTooLongError):
        _ = [e async for e in loop._conversation_loop("hi")]
    assert backend.requests_max_tokens == [None, 16384, 32768, 65536]


def test_adapter_capability_flags() -> None:
    from vibe.core.llm.backend.openai_responses import (
        ChatGPTResponsesAdapter,
        OpenAIResponsesAdapter,
    )

    assert OpenAIResponsesAdapter.supports_max_output_escalation
    assert not ChatGPTResponsesAdapter.supports_max_output_escalation
