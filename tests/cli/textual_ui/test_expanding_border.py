from __future__ import annotations

from textual.widget import _RenderCache

from vibe.cli.textual_ui.widgets.messages import ExpandingBorder


def test_expanding_border_keeps_textual_render_cache_intact() -> None:
    border = ExpandingBorder(classes="tool-result-border")
    border.set_row_colors({0: "red", 1: "green"})

    border.render()

    cache = border._render_cache
    assert cache is None or isinstance(cache, _RenderCache)
    assert border._border_cache is not None
