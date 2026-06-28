from __future__ import annotations

from pathlib import Path

from rich.cells import cell_len
from rich.text import Text

from vibe.core.types import AgentStats
from vibe.core.usage import (
    ProviderBreakdown,
    RateLimitSnapshot,
    UsageSummary,
    WindowRollup,
)

_BAR_SEGMENTS = 20
_BAR_FILLED = "█"
_BAR_EMPTY = "░"
_CARD_WIDTH = 72
_LABEL_WIDTH = 16
_THOUSAND = 1_000
_MILLION = 1_000_000
_BILLION = 1_000_000_000
_CENT_THRESHOLD = 0.01
_DECIMAL_BOUND_2 = 10.0
_DECIMAL_BOUND_1 = 100.0
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600
_MODEL_COL = 24
_TOKENS_COL = 7


def format_tokens_compact(value: int) -> str:
    """2K / 1.4M / 3.2B — matches Codex's compact token formatting."""
    value = max(value, 0)
    if value == 0:
        return "0"
    if value < _THOUSAND:
        return str(value)
    f = float(value)
    if value >= _BILLION:
        scaled, suffix = f / _BILLION, "B"
    elif value >= _MILLION:
        scaled, suffix = f / _MILLION, "M"
    else:
        scaled, suffix = f / _THOUSAND, "K"
    decimals = (
        2 if scaled < _DECIMAL_BOUND_2 else (1 if scaled < _DECIMAL_BOUND_1 else 0)
    )
    formatted = f"{scaled:.{decimals}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return f"{formatted}{suffix}"


def format_cost(value: float) -> str:
    if value < _CENT_THRESHOLD:
        return f"${value:.4f}"
    return f"${value:.2f}"


def _progress_bar(ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    filled = min(round(ratio * _BAR_SEGMENTS), _BAR_SEGMENTS)
    empty = _BAR_SEGMENTS - filled
    return f"[{_BAR_FILLED * filled}{_BAR_EMPTY * empty}]"


def _label_line(label: str, value: Text) -> Text:
    line = Text()
    line.append(f"  {label}:".ljust(_LABEL_WIDTH + 3), style="dim")
    line.append_text(value)
    return line


def _shorten_dir(path: Path, max_len: int = 40) -> str:
    home = Path.home()
    try:
        rel = path.relative_to(home)
        s = f"~/{rel}"
    except ValueError:
        s = str(path)
    if len(s) <= max_len:
        return s
    return "…" + s[-(max_len - 1) :]


def _header(version: str) -> Text:
    line = Text()
    line.append("  >_ ", style="dim")
    line.append("Vibe", style="bold")
    line.append(f" (v{version})", style="dim")
    return line


def _config_section(
    model_name: str, provider_name: str, workdir: Path, session_id: str
) -> list[Text]:
    lines = [
        _label_line("Model", Text(model_name)),
        _label_line("Provider", Text(provider_name)),
        _label_line("Directory", Text(_shorten_dir(workdir))),
    ]
    if session_id:
        lines.append(_label_line("Session", Text(session_id)))
    return lines


def _session_section(stats: AgentStats, context_window: int | None) -> list[Text]:
    lines: list[Text] = []
    total = stats.session_total_llm_tokens
    usage_val = Text()
    usage_val.append(format_tokens_compact(total))
    usage_val.append(" total ", style="dim")
    usage_val.append(" (", style="dim")
    usage_val.append(format_tokens_compact(stats.session_prompt_tokens), style="dim")
    usage_val.append(" input", style="dim")
    usage_val.append(" + ", style="dim")
    usage_val.append(
        format_tokens_compact(stats.session_completion_tokens), style="dim"
    )
    usage_val.append(" output)", style="dim")
    lines.append(_label_line("Session usage", usage_val))

    cache_val = Text()
    cache_val.append(f"{stats.cache_hit_ratio:.0%} hit")
    if stats.session_cached_tokens > 0:
        cache_val.append(
            f" ({format_tokens_compact(stats.session_cached_tokens)} cached)",
            style="dim",
        )
    lines.append(_label_line("Cache", cache_val))
    lines.append(_label_line("Cost", Text(format_cost(stats.session_cost))))

    if context_window and context_window > 0 and stats.context_tokens > 0:
        ratio = min(stats.context_tokens / context_window, 1.0)
        ctx_val = Text()
        ctx_val.append(
            f"{format_tokens_compact(stats.context_tokens)} / "
            f"{format_tokens_compact(context_window)} "
        )
        ctx_val.append(f"({ratio:.0%})", style="dim")
        lines.append(_label_line("Context", ctx_val))
    return lines


def _provider_section(providers: list[ProviderBreakdown]) -> list[Text]:
    lines: list[Text] = [Text("  ── By provider (all-time) ──", style="dim"), Text()]
    for prov in providers:
        lines.append(Text(f"  {prov.provider}", style="bold"))
        prov_total = prov.total_tokens or 1
        for mb in prov.models:
            share = mb.total_tokens / prov_total
            row = Text()
            row.append(f"    {mb.model}".ljust(_MODEL_COL))
            row.append(format_tokens_compact(mb.total_tokens).rjust(_TOKENS_COL) + " ")
            row.append(_progress_bar(share) + " ")
            row.append(f"{share:.0%}".rjust(3) + " ", style="dim")
            row.append(format_cost(mb.cost_usd).rjust(_TOKENS_COL))
            lines.append(row)
        prov_row = Text()
        prov_row.append("    ".ljust(_MODEL_COL), style="dim")
        prov_row.append(
            format_tokens_compact(prov.total_tokens).rjust(_TOKENS_COL) + " ",
            style="bold",
        )
        calls_word = "call" if prov.calls == 1 else "calls"
        prov_row.append(f"{prov.calls} {calls_word}", style="dim")
        lines.append(prov_row)
    return lines


def _windows_section(windows: list[WindowRollup]) -> list[Text]:
    lines: list[Text] = [Text("  ── Time windows ──", style="dim"), Text()]
    for win in windows:
        if win.calls == 0:
            continue
        val = Text()
        val.append(f"{format_tokens_compact(win.total_tokens)} tokens")
        val.append(f" · {format_cost(win.cost_usd)}", style="dim")
        calls_word = "call" if win.calls == 1 else "calls"
        val.append(f" · {win.calls} {calls_word}", style="dim")
        if win.sessions > 1:
            val.append(f" · {win.sessions} sessions", style="dim")
        lines.append(_label_line(win.label, val))
    return lines


def _format_reset(seconds: float | None) -> str | None:
    if seconds is None or seconds <= 0:
        return None
    if seconds < _SECONDS_PER_MINUTE:
        return f"resets in {int(seconds)}s"
    if seconds < _SECONDS_PER_HOUR:
        return f"resets in {int(seconds / _SECONDS_PER_MINUTE)}m"
    return f"resets in {seconds / _SECONDS_PER_HOUR:.1f}h"


def _limits_section(snapshots: dict[str, RateLimitSnapshot]) -> list[Text]:
    lines: list[Text] = [Text("  ── Provider limits (live) ──", style="dim"), Text()]
    for snap in snapshots.values():
        lines.append(Text(f"  {snap.provider}", style="bold"))
        if snap.limit_tokens and snap.limit_tokens > 0:
            used = snap.limit_tokens - (snap.remaining_tokens or 0)
            ratio_used = min(used / snap.limit_tokens, 1.0)
            pct_left = round((1.0 - ratio_used) * 100)
            val = Text()
            val.append(_progress_bar(1.0 - ratio_used) + " ")
            val.append(
                f"{format_tokens_compact(snap.remaining_tokens or 0)} of "
                f"{format_tokens_compact(snap.limit_tokens)} left ({pct_left}%)"
            )
            reset = _format_reset(snap.reset_tokens_in_s)
            if reset:
                val.append(f" · {reset}", style="dim")
            lines.append(_label_line("Tokens", val))
        if snap.limit_requests and snap.limit_requests > 0:
            used = snap.limit_requests - (snap.remaining_requests or 0)
            ratio_used = min(used / snap.limit_requests, 1.0)
            pct_left = round((1.0 - ratio_used) * 100)
            val = Text()
            val.append(_progress_bar(1.0 - ratio_used) + " ")
            val.append(
                f"{snap.remaining_requests or 0} of {snap.limit_requests} left "
                f"({pct_left}%)"
            )
            reset = _format_reset(snap.reset_requests_in_s)
            if reset:
                val.append(f" · {reset}", style="dim")
            lines.append(_label_line("Requests", val))
    return lines


def render_status_card(
    *,
    stats: AgentStats,
    summary: UsageSummary,
    version: str,
    model_name: str,
    provider_name: str,
    workdir: Path,
    session_id: str,
    context_window: int | None = None,
    rate_limits: dict[str, RateLimitSnapshot] | None = None,
    width: int = _CARD_WIDTH,
) -> Text:
    """Build the full status card as a Rich ``Text`` (box-drawing included).

    Pure function: same inputs → identical output, so it snapshots cleanly.
    """
    lines: list[Text] = [_header(version), Text()]
    lines.extend(_config_section(model_name, provider_name, workdir, session_id))
    lines.append(Text())
    lines.extend(_session_section(stats, context_window))
    lines.append(Text())

    if summary.providers:
        lines.extend(_provider_section(summary.providers))
        lines.append(Text())
    if rate_limits:
        lines.extend(_limits_section(rate_limits))
        lines.append(Text())
    if summary.windows:
        lines.extend(_windows_section(summary.windows))
    return _box(lines, width)


def _box(lines: list[Text], width: int) -> Text:
    """Wrap rendered lines in a rounded border, padding to ``width``."""
    top = Text(f"╭{'─' * (width - 2)}╮", style="dim")
    bottom = Text(f"╰{'─' * (width - 2)}╯", style="dim")
    out = Text()
    out.append_text(top)
    out.append("\n")
    inner_w = width - 4
    for line in lines:
        # cell_len measures terminal cell width (CJK/emoji = 2), so the right
        # border stays aligned even when a model/dir name contains wide glyphs.
        pad = max(0, inner_w - cell_len(line.plain))
        bordered = Text()
        bordered.append("│ ", style="dim")
        bordered.append_text(line)
        bordered.append(" " * pad)
        bordered.append(" │", style="dim")
        out.append_text(bordered)
        out.append("\n")
    out.append_text(bottom)
    return out
