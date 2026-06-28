from __future__ import annotations

from pathlib import Path

from vibe.cli.textual_ui.widgets._status_render import (
    format_cost,
    format_tokens_compact,
    render_status_card,
)
from vibe.core.types import AgentStats, LLMUsage
from vibe.core.usage import UsageRecord, summarize


def _stats() -> AgentStats:
    s = AgentStats()
    s.session_prompt_tokens = 1_400_000
    s.session_completion_tokens = 600_000
    s.session_cached_tokens = 900_000
    s.context_tokens = 2200
    return s


def _records() -> list[UsageRecord]:
    now = 1_000_000.0
    return [
        UsageRecord.from_usage(
            timestamp=now - 30,
            provider="mistral",
            model="mistral-large",
            usage=LLMUsage(
                prompt_tokens=1_400_000,
                completion_tokens=500_000,
                cached_tokens=900_000,
            ),
            cost_usd=2.10,
            duration_s=12.0,
            session_id="sess-1",
        ),
        UsageRecord.from_usage(
            timestamp=now - 30,
            provider="openai-chatgpt",
            model="gpt-5.3-codex-spark",
            usage=LLMUsage(
                prompt_tokens=300_000, completion_tokens=80_000, cached_tokens=0
            ),
            cost_usd=1.40,
            duration_s=8.0,
            session_id="sess-1",
        ),
        UsageRecord.from_usage(
            timestamp=now - 200_000,
            provider="zai",
            model="glm-5.2",
            usage=LLMUsage(
                prompt_tokens=50_000, completion_tokens=10_000, cached_tokens=0
            ),
            cost_usd=0.12,
            duration_s=5.0,
            session_id="sess-0",
        ),
    ]


def test_format_tokens_compact():
    assert format_tokens_compact(0) == "0"
    assert format_tokens_compact(999) == "999"
    assert format_tokens_compact(1500) == "1.5K"
    assert format_tokens_compact(2_200_000) == "2.2M"
    assert format_tokens_compact(3_000_000_000) == "3B"


def test_format_cost():
    assert format_cost(0.004) == "$0.0040"
    assert format_cost(0.01) == "$0.01"
    assert format_cost(12.5) == "$12.50"


def test_render_status_card_snapshot():
    summary = summarize(_records(), now=1_000_000.0)
    text = render_status_card(
        stats=_stats(),
        summary=summary,
        version="0.1.1",
        model_name="mistral-large",
        provider_name="mistral",
        workdir=Path("/home/dan/work/my-project"),
        session_id="sess-1",
        context_window=131072,
    )
    plain = text.plain
    # Structural assertions (content), not exact whitespace.
    assert "Vibe (v0.1.1)" in plain
    assert "mistral-large" in plain
    assert "Session usage" in plain
    assert "By provider" in plain
    assert "gpt-5.3-codex-spark" in plain
    assert "glm-5.2" in plain
    assert "Last hour" in plain
    assert "Last 7 days" in plain
    # Provider ordered by total tokens: mistral first, then openai, then zai.
    assert plain.index("mistral") < plain.index("openai-chatgpt")
    # Bordered top and bottom.
    assert plain.startswith("╭")
    assert plain.rstrip().endswith("╯")
