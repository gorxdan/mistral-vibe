from __future__ import annotations

import json
from typing import Any

from vibe.core.config import ProviderCacheConfig, ProviderConfig
from vibe.core.llm.backend.adapter_port import (
    RequestParams,
    memory_tail_relocated_before_user,
    trailing_ephemeral_count,
)
from vibe.core.llm.backend.anthropic import AnthropicMapper
from vibe.core.llm.backend.cache_hints import build_cache_hint
from vibe.core.llm.backend.generic import OpenAIAdapter
from vibe.core.types import AgentStats, InjectedMessageKind, LLMMessage, Role


def _provider(cache: ProviderCacheConfig | None = None) -> ProviderConfig:
    return ProviderConfig(
        name="p", api_base="http://x/v1", cache=cache or ProviderCacheConfig()
    )


# --------------------------------------------------------------------------- #
# build_cache_hint                                                             #
# --------------------------------------------------------------------------- #


def test_default_passthrough_is_inert_empty_fragment() -> None:
    # Default is mode="explicit", style="passthrough" with no extra_body/cache_key,
    # so the hint is an empty fragment. The generic caller merges hint only when
    # truthy (`if hint:`), so an empty fragment is a no-op on the request body.
    assert build_cache_hint(_provider(), [{"role": "user", "content": "hi"}]) == {}


def test_openai_default_gets_stable_per_conversation_prompt_cache_key() -> None:
    # OpenAI's prefix cache load-balances across machines and misses without a
    # routing key; an OpenAI provider must auto-get a prompt_cache_key even with
    # the default (no explicit cache_key) config. See codex (sends thread id).
    p = ProviderConfig(name="openai", api_base="https://api.openai.com/v1")
    msgs = [
        {"role": "system", "content": "You are vibe."},
        {"role": "user", "content": "hello"},
    ]
    hint = build_cache_hint(p, msgs)
    assert hint is not None
    assert "prompt_cache_key" in hint
    key = hint["prompt_cache_key"]

    # Stable across the conversation's turns (prefix unchanged as history grows).
    grown = msgs + [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "follow up"},
    ]
    grown_hint = build_cache_hint(p, grown)
    assert grown_hint is not None
    assert grown_hint["prompt_cache_key"] == key

    # Distinct per conversation (different opening turn -> different partition).
    other = [
        {"role": "system", "content": "You are vibe."},
        {"role": "user", "content": "a different opener"},
    ]
    other_hint = build_cache_hint(p, other)
    assert other_hint is not None
    assert other_hint["prompt_cache_key"] != key


def test_non_openai_provider_gets_no_auto_cache_key() -> None:
    # A generic provider with default cache (no session_keyed) stays key-less;
    # the routing pin is opt-in so a provider that rejects unknown body fields,
    # or whose cache already works, is left untouched.
    p = ProviderConfig(name="zai", api_base="https://api.z.ai/api/coding/paas/v4")
    hint = build_cache_hint(
        p, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    )
    assert hint is not None
    assert "prompt_cache_key" not in hint


def test_session_keyed_provider_prefers_session_id() -> None:
    # An OpenAI-compatible provider that opts in via cache.session_keyed gets the
    # same per-conversation pin as OpenAI (zai/GLM, whose cache scatters under
    # concurrency). session_id is the routing key when threaded through.
    p = ProviderConfig(
        name="zai",
        api_base="https://api.z.ai/api/coding/paas/v4",
        cache=ProviderCacheConfig(session_keyed=True),
    )
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    hint = build_cache_hint(p, msgs, session_id="sess-zai-1")
    assert hint is not None
    assert hint["prompt_cache_key"] == "sess-zai-1"
    # No session id (one-shot caller) => stable content-hash pin instead.
    fallback = build_cache_hint(p, msgs)
    assert fallback is not None
    assert fallback["prompt_cache_key"].startswith("vibe-")


def test_openai_prefers_session_id_over_content_hash() -> None:
    # When the conversation's stable session id is threaded through, it is the
    # routing pin (mirrors codex's thread_id) — not the content hash. This makes
    # the key unique per conversation even when two sessions share an opening.
    p = ProviderConfig(name="openai", api_base="https://api.openai.com/v1")
    msgs = [
        {"role": "system", "content": "You are vibe."},
        {"role": "user", "content": "hello"},
    ]
    hint = build_cache_hint(p, msgs, session_id="sess-abc-123")
    assert hint is not None
    assert hint["prompt_cache_key"] == "sess-abc-123"
    # Identical opening but a different session id => a different partition.
    other = build_cache_hint(p, msgs, session_id="sess-def-456")
    assert other is not None
    assert other["prompt_cache_key"] == "sess-def-456"


def test_non_openai_provider_ignores_session_id() -> None:
    # Without session_keyed, a non-OpenAI provider stays key-less even when a
    # session id is threaded through (the opt-in path is covered above).
    p = ProviderConfig(name="zai", api_base="https://api.z.ai/api/coding/paas/v4")
    hint = build_cache_hint(
        p,
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        session_id="sess-abc-123",
    )
    assert hint is not None
    assert "prompt_cache_key" not in hint


def test_openai_falls_back_to_content_hash_without_session_id() -> None:
    # One-shot callers (memory, summary) thread no session id; the OpenAI path
    # still pins, via a content hash of the stable prefix.
    p = ProviderConfig(name="openai", api_base="https://api.openai.com/v1")
    hint = build_cache_hint(
        p, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    )
    assert hint is not None
    assert hint["prompt_cache_key"].startswith("vibe-")


def test_sakana_gets_stable_per_conversation_prompt_cache_key() -> None:
    # Sakana uses the OpenAI Responses wire format and needs the same partition
    # pinning that OpenAI does.
    p = ProviderConfig(name="sakana", api_base="https://api.sakana.ai/v1")
    msgs = [
        {"role": "system", "content": "You are vibe."},
        {"role": "user", "content": "hello"},
    ]
    hint = build_cache_hint(p, msgs)
    assert hint is not None
    assert "prompt_cache_key" in hint
    key = hint["prompt_cache_key"]

    # Stable across the conversation's turns (prefix unchanged as history grows).
    grown = msgs + [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "follow up"},
    ]
    grown_hint = build_cache_hint(p, grown)
    assert grown_hint is not None
    assert grown_hint["prompt_cache_key"] == key

    # Distinct per conversation (different opening turn -> different partition).
    other = [
        {"role": "system", "content": "You are vibe."},
        {"role": "user", "content": "a different opener"},
    ]
    other_hint = build_cache_hint(p, other)
    assert other_hint is not None
    assert other_hint["prompt_cache_key"] != key


def test_explicit_cache_key_overrides_auto_openai_key() -> None:
    cache = ProviderCacheConfig(cache_key="agent-main")
    p = ProviderConfig(name="openai", api_base="https://api.openai.com/v1", cache=cache)
    hint = build_cache_hint(
        p, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    )
    assert hint is not None
    assert hint["prompt_cache_key"] == "agent-main"


def test_passthrough_merges_extra_body_and_cache_key() -> None:
    cache = ProviderCacheConfig(
        mode="explicit",
        style="passthrough",
        extra_body={"cache": {"ttl": "1h"}},
        cache_key="agent-main",
    )
    hint = build_cache_hint(_provider(cache), [])
    assert hint == {"cache": {"ttl": "1h"}, "prompt_cache_key": "agent-main"}


def test_anthropic_compat_tags_last_system_and_user_str_content() -> None:
    cache = ProviderCacheConfig(mode="explicit", style="anthropic-compat")
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "mid"},
        {"role": "user", "content": "last"},
    ]
    hint = build_cache_hint(_provider(cache), msgs)
    assert hint == {}  # in-place mutation
    content0 = msgs[0]["content"]
    assert isinstance(content0, list)
    assert content0[0]["cache_control"] == {"type": "ephemeral"}
    content3 = msgs[3]["content"]
    assert isinstance(content3, list)
    assert content3[0]["cache_control"] == {"type": "ephemeral"}
    assert msgs[1]["content"] == "first"  # earlier user untouched
    assert msgs[2]["content"] == "mid"  # assistant untouched


def test_anthropic_compat_handles_list_content() -> None:
    cache = ProviderCacheConfig(mode="explicit", style="anthropic-compat")
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    build_cache_hint(_provider(cache), msgs)
    assert msgs[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_compat_skip_trailing_tags_below_tail_str_content() -> None:
    cache = ProviderCacheConfig(mode="explicit", style="anthropic-compat")
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "real question"},
        {"role": "user", "content": "<memories>tail</memories>"},
    ]
    hint = build_cache_hint(_provider(cache), msgs, skip_trailing=1)
    assert hint == {}
    tagged = msgs[1]["content"]
    assert isinstance(tagged, list)
    assert tagged[0]["cache_control"] == {"type": "ephemeral"}
    assert msgs[2]["content"] == "<memories>tail</memories>"  # tail untouched
    sys_content = msgs[0]["content"]
    assert isinstance(sys_content, list)
    assert sys_content[0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_compat_skip_trailing_list_content() -> None:
    cache = ProviderCacheConfig(mode="explicit", style="anthropic-compat")
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "real"}]},
        {"role": "user", "content": [{"type": "text", "text": "tail"}]},
    ]
    build_cache_hint(_provider(cache), msgs, skip_trailing=1)
    assert msgs[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in msgs[1]["content"][-1]


# trailing_ephemeral_count + generic wiring


def _mem_msg(content: str = "<memories>m</memories>") -> LLMMessage:
    return LLMMessage(
        role=Role.USER,
        content=content,
        injected=True,
        injected_kind=InjectedMessageKind.MEMORY,
    )


def test_trailing_ephemeral_count_counts_only_trailing_memory() -> None:
    user = LLMMessage(role=Role.USER, content="u")
    hook = LLMMessage(
        role=Role.USER,
        content="h",
        injected=True,
        injected_kind=InjectedMessageKind.USER_PROMPT_HOOK,
    )
    assert trailing_ephemeral_count([]) == 0
    assert trailing_ephemeral_count([user]) == 0
    assert trailing_ephemeral_count([user, hook]) == 0
    assert trailing_ephemeral_count([user, _mem_msg()]) == 1
    assert trailing_ephemeral_count([_mem_msg(), user, _mem_msg(), _mem_msg()]) == 2


def test_memory_tail_relocates_before_user_only_after_tool() -> None:
    sys_msg = LLMMessage(role=Role.SYSTEM, content="sys")
    user = LLMMessage(role=Role.USER, content="u")
    assistant = LLMMessage(role=Role.ASSISTANT, content="", tool_calls=None)
    tool = LLMMessage(role=Role.TOOL, content="result", tool_call_id="c1")
    mem = _mem_msg()

    mid_turn = [sys_msg, user, assistant, tool, mem]
    relocated = list(memory_tail_relocated_before_user(mid_turn))
    assert relocated == [sys_msg, mem, user, assistant, tool]

    turn_start = [sys_msg, user, mem]
    assert list(memory_tail_relocated_before_user(turn_start)) == turn_start

    no_tail = [sys_msg, user, assistant, tool]
    assert list(memory_tail_relocated_before_user(no_tail)) == no_tail
    assert list(memory_tail_relocated_before_user([])) == []


def test_memory_tail_relocation_preserves_tail_order() -> None:
    sys_msg = LLMMessage(role=Role.SYSTEM, content="sys")
    user = LLMMessage(role=Role.USER, content="u")
    assistant = LLMMessage(role=Role.ASSISTANT, content="", tool_calls=None)
    tool = LLMMessage(role=Role.TOOL, content="result", tool_call_id="c1")
    mem_a, mem_b = (
        _mem_msg("<memories>a</memories>"),
        _mem_msg("<memories>b</memories>"),
    )

    relocated = list(
        memory_tail_relocated_before_user([
            sys_msg,
            user,
            assistant,
            tool,
            mem_a,
            mem_b,
        ])
    )
    assert relocated == [sys_msg, mem_a, mem_b, user, assistant, tool]


def test_generic_anthropic_compat_prepare_request_skips_memory_tail() -> None:
    cache = ProviderCacheConfig(mode="explicit", style="anthropic-compat")
    req = OpenAIAdapter().prepare_request(
        RequestParams(
            model_name="m",
            messages=[
                LLMMessage(role=Role.SYSTEM, content="sys"),
                LLMMessage(role=Role.USER, content="question"),
                _mem_msg(),
            ],
            temperature=0.2,
            tools=None,
            max_tokens=100,
            tool_choice=None,
            enable_streaming=False,
            provider=_provider(cache),
        )
    )
    wire = json.loads(req.body)["messages"]
    assert wire[-1]["content"] == "<memories>m</memories>"  # tail untouched
    assert wire[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert wire[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_generic_payload_excludes_injected_markers() -> None:
    req = OpenAIAdapter().prepare_request(
        RequestParams(
            model_name="m",
            messages=[LLMMessage(role=Role.USER, content="q"), _mem_msg()],
            temperature=0.2,
            tools=None,
            max_tokens=100,
            tool_choice=None,
            enable_streaming=False,
            provider=_provider(),
        )
    )
    wire = json.loads(req.body)["messages"]
    assert all("injected_kind" not in m for m in wire)
    assert all("injected" not in m for m in wire)


# --------------------------------------------------------------------------- #
# Telemetry: cached_tokens                                                     #
# --------------------------------------------------------------------------- #


def test_anthropic_parse_populates_cached_tokens() -> None:
    data = {
        "content": [{"type": "text", "text": "ok"}],
        "usage": {
            "input_tokens": 10,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 80,
            "output_tokens": 3,
        },
    }
    usage = AnthropicMapper().parse_response(data).usage
    assert usage is not None
    assert usage.cached_tokens == 80
    assert usage.prompt_tokens == 95  # 10 + 5 + 80, no double count


def test_cache_hit_ratio() -> None:
    assert (
        AgentStats(session_prompt_tokens=100, session_cached_tokens=80).cache_hit_ratio
        == 0.8
    )
    # clamp at 1.0 if a provider over-reports
    assert (
        AgentStats(session_prompt_tokens=100, session_cached_tokens=200).cache_hit_ratio
        == 1.0
    )
    # div-by-zero guard
    assert (
        AgentStats(session_prompt_tokens=0, session_cached_tokens=0).cache_hit_ratio
        == 0.0
    )


def test_llm_usage_sums_cached_tokens() -> None:
    from vibe.core.types import LLMUsage

    total = LLMUsage(prompt_tokens=10, cached_tokens=8) + LLMUsage(
        prompt_tokens=5, cached_tokens=4
    )
    assert total.cached_tokens == 12


# --------------------------------------------------------------------------- #
# Stable-prefix invariant                                                      #
# --------------------------------------------------------------------------- #


def test_generic_payload_is_deterministic_for_identical_input() -> None:
    adapter = OpenAIAdapter()
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    p1 = adapter.build_payload("m", list(msgs), 0.2, None, 100, None)
    p2 = adapter.build_payload("m", list(msgs), 0.2, None, 100, None)
    # Byte-identical serialization => the auto-cache prefix is stable turn-over-turn.
    assert json.dumps(p1, sort_keys=False) == json.dumps(p2, sort_keys=False)
