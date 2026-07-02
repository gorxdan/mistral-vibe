from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.style import Style
from rich.text import Text
from textual.reactive import reactive

from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic

_THOUSAND = 1_000
_MILLION = 1_000_000
_GAUGE_CELLS = 10
_WARN_RATIO = 0.80
_CRIT_RATIO = 0.95
_BAR_FILLED = "█"
_CACHE_FILLED = "▓"
_BAR_EMPTY = "░"
# Cached reads are near-free on providers that cache, so they get a calm
# dim-green regardless of the widget's severity class. Uncached and empty
# cells carry no inline style and inherit the widget color (muted → warn
# amber → crit red), so severity still tracks the costly (uncached) share.
_CACHE_STYLE = Style(color="#26a641", dim=True)


@dataclass
class TokenState:
    max_tokens: int = 0
    current_tokens: int = 0
    cached_tokens: int = 0


def _format_token_count(tokens: int) -> str:
    if tokens >= _MILLION:
        return f"{tokens / _MILLION:.1f}M"
    if tokens >= _THOUSAND:
        return f"{tokens // _THOUSAND}k"
    return str(tokens)


def build_context_text(state: TokenState) -> Text:
    """Pure renderer: TokenState → styled Text for the context gauge.

    Three segments, drawn left to right: cached (▓, dim-green — near-free),
    uncached (█, inherits widget color — the costly share), empty (░, dim).
    Bar length stays total window fill, so severity still reflects what trips
    auto-compaction; the cache split only disambiguates *why* it looks full.
    """
    current = state.current_tokens
    max_tokens = state.max_tokens
    ratio = min(1, current / max_tokens)
    filled = min(_GAUGE_CELLS, round(ratio * _GAUGE_CELLS))
    cached_share = min(1.0, state.cached_tokens / current) if current > 0 else 0.0
    cached_cells = max(0, min(filled, round(filled * cached_share)))
    uncached_cells = filled - cached_cells
    empty_cells = _GAUGE_CELLS - filled

    text = Text()
    text.append(f"{_format_token_count(current)}/")
    text.append(f"{_format_token_count(max_tokens)} ")
    if cached_cells:
        text.append(_CACHE_FILLED * cached_cells, style=_CACHE_STYLE)
    if uncached_cells:
        text.append(_BAR_FILLED * uncached_cells)
    text.append(_BAR_EMPTY * empty_cells)
    text.append(f" {ratio:.0%}")
    if state.cached_tokens > 0:
        text.append(f" ({cached_share:.0%} cached)", style="dim")
    return text


class ContextProgress(NoMarkupStatic):
    tokens = reactive(TokenState())

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._render_state: TokenState | None = None

    def watch_tokens(self, new_state: TokenState) -> None:
        self.remove_class("ctx-warn", "ctx-crit")
        if new_state.max_tokens == 0:
            self._render_state = None
            self.refresh()
            return

        ratio = min(1, new_state.current_tokens / new_state.max_tokens)
        # Severity still tracks total window fill (what trips auto-compaction).
        if ratio >= _CRIT_RATIO:
            self.add_class("ctx-crit")
        elif ratio >= _WARN_RATIO:
            self.add_class("ctx-warn")

        self._render_state = new_state
        self.refresh()

    def render(self) -> Text:
        # Override render() rather than update(): Static.update(Rich Text)
        # forces app-console resolution (Content.from_rich_text), which raises
        # NoActiveAppError in app-less test contexts. Returning the Text here
        # keeps styling in the live app while str(render()) stays testable.
        if self._render_state is None:
            return Text()
        return build_context_text(self._render_state)
