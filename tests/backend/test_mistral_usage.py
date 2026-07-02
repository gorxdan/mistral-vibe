from __future__ import annotations

from mistralai.client.models import UsageInfo

from vibe.core.llm.backend.mistral import cached_tokens_from_usage


def test_none_usage_is_zero() -> None:
    assert cached_tokens_from_usage(None) == 0


def test_no_cache_fields_is_zero() -> None:
    usage = UsageInfo(prompt_tokens=100, completion_tokens=10, total_tokens=110)
    assert cached_tokens_from_usage(usage) == 0


def test_openai_style_prompt_tokens_details() -> None:
    usage = UsageInfo.model_validate({
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 80},
    })
    assert cached_tokens_from_usage(usage) == 80


def test_flat_cached_tokens_extra() -> None:
    usage = UsageInfo.model_validate({
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "cached_tokens": 64,
    })
    assert cached_tokens_from_usage(usage) == 64


def test_deepseek_style_prompt_cache_hit_tokens() -> None:
    usage = UsageInfo.model_validate({
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "prompt_cache_hit_tokens": 32,
    })
    assert cached_tokens_from_usage(usage) == 32


def test_malformed_values_are_zero() -> None:
    usage = UsageInfo.model_validate({
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": "80"},
        "cached_tokens": None,
    })
    assert cached_tokens_from_usage(usage) == 0


def test_details_take_precedence_over_flat() -> None:
    usage = UsageInfo.model_validate({
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 80},
        "cached_tokens": 5,
    })
    assert cached_tokens_from_usage(usage) == 80
