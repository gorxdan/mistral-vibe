from __future__ import annotations

from pathlib import Path

from vibe.cli.textual_ui.widgets._status_render import (
    StatusCardData,
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


def test_unpriced_usage_shows_em_dash_not_zero():
    # Models with real usage but no configured pricing (input_price=0) must not
    # claim $0.0000 — that reads as "free" when it really means "unknown".
    from vibe.cli.textual_ui.widgets._status_render import _cost_or_unknown

    assert _cost_or_unknown(0.0, has_usage=True) == "—"
    # Genuine zero usage → $0.0000 is correct.
    assert _cost_or_unknown(0.0, has_usage=False) == "$0.0000"
    # Priced usage → real cost.
    assert _cost_or_unknown(2.50, has_usage=True) == "$2.50"


def test_render_hides_cost_for_unpriced_model():
    # Mirrors the live report: glm-5.2 with 39.6M tokens, input_price=0.
    s = AgentStats()
    summary = summarize(
        [
            UsageRecord.from_usage(
                timestamp=1_000_000.0,
                provider="zai",
                model="glm-5.2",
                usage=LLMUsage(prompt_tokens=36_000_000, completion_tokens=3_600_000),
                cost_usd=0.0,  # unpriced
                duration_s=1.0,
                session_id="s1",
            )
        ],
        now=1_000_000.0,
    )
    text = render_status_card(
        StatusCardData(
            stats=s,
            summary=summary,
            version="0.1.1",
            model_name="glm-5.2",
            provider_name="zai",
            workdir=Path("/home/dan/p"),
            session_id="s1",
        )
    )
    plain = text.plain
    assert "glm-5.2" in plain
    # The model row and windows must show — (pricing not configured), not $0.0000.
    assert "—" in plain
    assert "$0.0000" not in plain.split("By provider")[1]


def test_render_status_card_snapshot():
    summary = summarize(_records(), now=1_000_000.0)
    text = render_status_card(
        StatusCardData(
            stats=_stats(),
            summary=summary,
            version="0.1.1",
            model_name="mistral-large",
            provider_name="mistral",
            workdir=Path("/home/dan/work/my-project"),
            session_id="sess-1",
            context_window=131072,
        )
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
