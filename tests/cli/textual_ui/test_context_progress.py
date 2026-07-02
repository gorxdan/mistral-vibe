from __future__ import annotations

from vibe.cli.textual_ui.widgets.context_progress import (
    ContextProgress,
    TokenState,
    build_context_text,
)


def test_context_progress_shows_empty_gauge_when_zero() -> None:
    widget = ContextProgress()

    widget.watch_tokens(TokenState(max_tokens=200_000, current_tokens=0))

    assert str(widget.render()) == "0/200k ░░░░░░░░░░ 0%"


def test_context_progress_draws_partial_gauge() -> None:
    widget = ContextProgress()

    widget.watch_tokens(TokenState(max_tokens=200_000, current_tokens=12_500))

    assert str(widget.render()) == "12k/200k █░░░░░░░░░ 6%"


def test_context_progress_uses_compact_k_format() -> None:
    widget = ContextProgress()

    widget.watch_tokens(TokenState(max_tokens=568_000, current_tokens=170_000))

    assert str(widget.render()) == "170k/568k ███░░░░░░░ 30%"


def test_context_progress_warns_near_threshold() -> None:
    widget = ContextProgress()

    widget.watch_tokens(TokenState(max_tokens=40_000_000, current_tokens=35_900_000))

    assert str(widget.render()) == "35.9M/40.0M █████████░ 90%"
    assert widget.has_class("ctx-warn")
    assert not widget.has_class("ctx-crit")


def test_context_progress_crit_when_nearly_full() -> None:
    widget = ContextProgress()

    widget.watch_tokens(TokenState(max_tokens=200_000, current_tokens=194_000))

    assert str(widget.render()) == "194k/200k ██████████ 97%"
    assert widget.has_class("ctx-crit")
    assert not widget.has_class("ctx-warn")


def test_context_progress_clears_escalation_when_dropping() -> None:
    widget = ContextProgress()

    widget.watch_tokens(TokenState(max_tokens=200_000, current_tokens=194_000))
    widget.watch_tokens(TokenState(max_tokens=200_000, current_tokens=20_000))

    assert not widget.has_class("ctx-crit")
    assert not widget.has_class("ctx-warn")


def test_context_progress_caches_segment_is_dim_green_prefix() -> None:
    state = TokenState(
        max_tokens=568_000, current_tokens=170_000, cached_tokens=119_000
    )

    rendered = build_context_text(state)
    # 70% of 3 filled cells = 2 cached cells, drawn first with the ▓ glyph.
    assert str(rendered) == "170k/568k ▓▓█░░░░░░░ 30% (70% cached)"
    cached_span = next(sp for sp in rendered.spans if "#26a641" in str(sp.style))
    assert "dim" in str(cached_span.style)


def test_context_progress_zero_cache_omits_segment_and_suffix() -> None:
    widget = ContextProgress()

    widget.watch_tokens(
        TokenState(max_tokens=568_000, current_tokens=170_000, cached_tokens=0)
    )

    assert str(widget.render()) == "170k/568k ███░░░░░░░ 30%"


def test_context_progress_cache_clamps_to_total_fill() -> None:
    widget = ContextProgress()

    widget.watch_tokens(
        TokenState(max_tokens=568_000, current_tokens=170_000, cached_tokens=1_000_000)
    )

    # cached exceeds current; share clamps to 100% so all filled cells render ▓
    # and the suffix shows the clamped (100%) value, not the raw ratio.
    assert str(widget.render()) == "170k/568k ▓▓▓░░░░░░░ 30% (100% cached)"
