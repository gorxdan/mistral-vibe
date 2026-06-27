from __future__ import annotations

import json

from vibe.core.config import ProviderCacheConfig, ProviderConfig
from vibe.core.llm.backend.anthropic import AnthropicMapper
from vibe.core.llm.backend.cache_hints import build_cache_hint
from vibe.core.llm.backend.generic import OpenAIAdapter
from vibe.core.types import AgentStats


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
    assert "prompt_cache_key" in hint
    key = hint["prompt_cache_key"]

    # Stable across the conversation's turns (prefix unchanged as history grows).
    grown = msgs + [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "follow up"},
    ]
    assert build_cache_hint(p, grown)["prompt_cache_key"] == key

    # Distinct per conversation (different opening turn -> different partition).
    other = [
        {"role": "system", "content": "You are vibe."},
        {"role": "user", "content": "a different opener"},
    ]
    assert build_cache_hint(p, other)["prompt_cache_key"] != key


def test_non_openai_provider_gets_no_auto_cache_key() -> None:
    # GLM/zai, DeepSeek, etc. auto-cache reliably; do not perturb their path.
    p = ProviderConfig(name="zai", api_base="https://api.z.ai/api/coding/paas/v4")
    hint = build_cache_hint(
        p,
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
    )
    assert "prompt_cache_key" not in hint


def test_explicit_cache_key_overrides_auto_openai_key() -> None:
    cache = ProviderCacheConfig(cache_key="agent-main")
    p = ProviderConfig(
        name="openai", api_base="https://api.openai.com/v1", cache=cache
    )
    hint = build_cache_hint(
        p,
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
    )
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
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "mid"},
        {"role": "user", "content": "last"},
    ]
    hint = build_cache_hint(_provider(cache), msgs)
    assert hint == {}  # in-place mutation
    assert msgs[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert msgs[3]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert msgs[1]["content"] == "first"  # earlier user untouched
    assert msgs[2]["content"] == "mid"  # assistant untouched


def test_anthropic_compat_handles_list_content() -> None:
    cache = ProviderCacheConfig(mode="explicit", style="anthropic-compat")
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    build_cache_hint(_provider(cache), msgs)
    assert msgs[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}


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
