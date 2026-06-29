from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.reactive import reactive

from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic

_THOUSAND = 1_000
_MILLION = 1_000_000
_GAUGE_CELLS = 10
_WARN_RATIO = 0.80
_CRIT_RATIO = 0.95


@dataclass
class TokenState:
    max_tokens: int = 0
    current_tokens: int = 0


def _format_token_count(tokens: int) -> str:
    if tokens >= _MILLION:
        return f"{tokens / _MILLION:.1f}M"
    if tokens >= _THOUSAND:
        return f"{tokens // _THOUSAND}k"
    return str(tokens)


class ContextProgress(NoMarkupStatic):
    tokens = reactive(TokenState())

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def watch_tokens(self, new_state: TokenState) -> None:
        self.remove_class("ctx-warn", "ctx-crit")
        if new_state.max_tokens == 0:
            self.update("")
            return

        ratio = min(1, new_state.current_tokens / new_state.max_tokens)
        if ratio >= _CRIT_RATIO:
            self.add_class("ctx-crit")
        elif ratio >= _WARN_RATIO:
            self.add_class("ctx-warn")

        filled = min(_GAUGE_CELLS, round(ratio * _GAUGE_CELLS))
        bar = "█" * filled + "░" * (_GAUGE_CELLS - filled)
        text = (
            f"{_format_token_count(new_state.current_tokens)}/"
            f"{_format_token_count(new_state.max_tokens)} {bar} {ratio:.0%}"
        )
        self.update(text)
