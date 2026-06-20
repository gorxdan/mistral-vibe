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


def test_hint_off_by_default_returns_none() -> None:
    assert build_cache_hint(_provider(), [{"role": "user", "content": "hi"}]) is None


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
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]
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
    assert AgentStats(session_prompt_tokens=100, session_cached_tokens=80).cache_hit_ratio == 0.8
    # clamp at 1.0 if a provider over-reports
    assert AgentStats(session_prompt_tokens=100, session_cached_tokens=200).cache_hit_ratio == 1.0
    # div-by-zero guard
    assert AgentStats(session_prompt_tokens=0, session_cached_tokens=0).cache_hit_ratio == 0.0


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
