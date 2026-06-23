from __future__ import annotations

from vibe.core.config import ProviderConfig
from vibe.core.llm.backend.generic import OpenAIAdapter

_PROVIDER = ProviderConfig(
    name="kimi",
    api_base="https://api.kimi.com/coding/v1",
    api_key_env_var="KIMI_API_KEY",
    api_style="openai",
    backend="generic",
    reasoning_field_name="reasoning_content",
)


def _parse(usage: dict) -> int:
    data = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "usage": usage,
    }
    return OpenAIAdapter().parse_response(data, _PROVIDER).usage.cached_tokens


def test_cached_tokens_from_prompt_tokens_details() -> None:
    assert (
        _parse({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 80}})
        == 80
    )


def test_cached_tokens_top_level_fallback() -> None:
    assert _parse({"prompt_tokens": 100, "cached_tokens": 30}) == 30


def test_cached_tokens_absent_defaults_zero() -> None:
    assert _parse({"prompt_tokens": 100, "completion_tokens": 5}) == 0
