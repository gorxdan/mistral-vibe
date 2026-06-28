from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING

from rich.cells import cell_len
from rich.text import Text

if TYPE_CHECKING:
    from vibe.core.types import AgentStats
    from vibe.core.usage import (
        CodexCredits,
        CodexMonthlyLimit,
        CodexQuotaSnapshot,
        CodexQuotaWindow,
        DailyBucket,
        HarnessSplit,
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
_HOURS_PER_WEEK = 24 * 7
_MINS_PER_HOUR = 60
_FIVE_HOURS = 5
_FLOAT_EQ_EPSILON = 1e-9
_MODEL_COL = 24
_TOKENS_COL = 7
_PCT_COL = 4
# Card grows to fit long model names up to a hard cap; every line is then
# cell-clipped to the inner width, so the border can never be pushed out.
_MIN_CARD_WIDTH = _CARD_WIDTH
_MAX_CARD_WIDTH = 84
_MODEL_COL_MIN = _MODEL_COL
# Non-name tail of a provider row: tokens + sep + bar + sep + pct + sep + cost.
_PROVIDER_TAIL = _TOKENS_COL + 1 + (_BAR_SEGMENTS + 2) + 1 + _PCT_COL + 1 + _TOKENS_COL
# Activity heatmap palette: index 0 = no activity, 1 dark → 5 bright (green).
_HEAT_STYLES = ("dim", "#0e4429", "#006d32", "#26a641", "#39d353", "#5ae57a")
_HEAT_CELL = "██"
_HEAT_EMPTY = "░░"


@dataclass(frozen=True)
class StatusCardData:
    """Bundle of inputs to ``render_status_card`` — keeps arg count under the
    linter cap and makes the call site readable as the card grows.
    """

    stats: AgentStats
    summary: UsageSummary
    version: str
    model_name: str
    provider_name: str
    workdir: Path
    session_id: str
    context_window: int | None = None
    rate_limits: dict[str, RateLimitSnapshot] | None = None
    codex_quota: CodexQuotaSnapshot | None = None
    width: int = _CARD_WIDTH


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


def _cost_or_unknown(cost: float, has_usage: bool) -> str:
    """Formatted cost, or '—' when pricing is unset.

    A zero cost with real usage means the model's ``input_price``/``output_price``
    weren't configured (or it's a flat-rate subscription like the ChatGPT plan),
    not that the usage was free. Showing '—' keeps the card honest instead of
    displaying a misleading ``$0.0000``.
    """
    if cost <= 0.0 and has_usage:
        return "—"
    return format_cost(cost)


def _progress_bar(ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    filled = min(round(ratio * _BAR_SEGMENTS), _BAR_SEGMENTS)
    empty = _BAR_SEGMENTS - filled
    return f"[{_BAR_FILLED * filled}{_BAR_EMPTY * empty}]"


def _heat_index(ratio: float) -> int:
    """Map a 0..1 ratio to a 1..5 palette index (1 dark, 5 bright)."""
    if ratio <= 0.0:
        return 1
    return min(len(_HEAT_STYLES) - 1, max(1, math.ceil(ratio * 5)))


def _share_bar(ratio: float) -> Text:
    """Provider share bar — filled cells coloured by intensity (cohesive with
    the activity heatmap), empty cells dim.
    """
    ratio = max(0.0, min(1.0, ratio))
    filled = min(round(ratio * _BAR_SEGMENTS), _BAR_SEGMENTS)
    out = Text("[")
    if filled:
        out.append(_BAR_FILLED * filled, style=_HEAT_STYLES[_heat_index(ratio)])
    if _BAR_SEGMENTS - filled:
        out.append(_BAR_EMPTY * (_BAR_SEGMENTS - filled), style="dim")
    out.append("]")
    return out


def _fit_field(text: str, width: int) -> str:
    """Left-justify ``text`` to exactly ``width`` cells, truncating with '…'."""
    if cell_len(text) <= width:
        return text.ljust(width)
    if width <= 1:
        return "…"[:width]
    return text[: width - 1].rstrip() + "…"


def _pick_model_col(providers: list[ProviderBreakdown]) -> int:
    """First-column width (incl. 4-space indent) sized to the longest model
    name, clamped so a provider row always fits the max card width.
    """
    widest = _MODEL_COL_MIN
    for prov in providers:
        for mb in prov.models:
            w = cell_len(f"    {mb.model}")
            widest = max(widest, w)
    return min(widest, _MAX_CARD_WIDTH - 4 - _PROVIDER_TAIL)


def _center(line: Text, inner_w: int) -> Text:
    pad = max(0, (inner_w - cell_len(line.plain)) // 2)
    out = Text(" " * pad)
    out.append_text(line)
    return out


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
    lines.append(
        _label_line(
            "Cost",
            Text(
                _cost_or_unknown(stats.session_cost, stats.session_total_llm_tokens > 0)
            ),
        )
    )

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


def _provider_section(providers: list[ProviderBreakdown], model_col: int) -> list[Text]:
    lines: list[Text] = [Text("  ── By provider (all-time) ──", style="dim"), Text()]
    for prov in providers:
        lines.append(Text(f"  {prov.provider}", style="bold"))
        prov_total = prov.total_tokens or 1
        for mb in prov.models:
            share = mb.total_tokens / prov_total
            row = Text()
            row.append(_fit_field(f"    {mb.model}", model_col))
            row.append(format_tokens_compact(mb.total_tokens).rjust(_TOKENS_COL) + " ")
            row.append_text(_share_bar(share))
            row.append(" ")
            row.append(f"{share:.0%}".rjust(_PCT_COL) + " ", style="dim")
            row.append(
                _cost_or_unknown(mb.cost_usd, mb.total_tokens > 0).rjust(_TOKENS_COL)
            )
            lines.append(row)
        prov_row = Text()
        prov_row.append(_fit_field("    ", model_col), style="dim")
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
        val.append(f" · {_cost_or_unknown(win.cost_usd, win.calls > 0)}", style="dim")
        calls_word = "call" if win.calls == 1 else "calls"
        val.append(f" · {win.calls} {calls_word}", style="dim")
        if win.sessions > 1:
            val.append(f" · {win.sessions} sessions", style="dim")
        lines.append(_label_line(win.label, val))
    return lines


def _harness_section(split: HarnessSplit) -> list[Text]:
    if split.harness_tokens == 0 and split.user_tokens == 0:
        return []
    lines: list[Text] = [Text("  ── User vs harness ──", style="dim"), Text()]
    if split.user_tokens > 0:
        val = Text()
        val.append(f"{format_tokens_compact(split.user_tokens)} tokens")
        val.append(f" · {_cost_or_unknown(split.user_cost, True)}", style="dim")
        lines.append(_label_line("You", val))
    if split.harness_tokens > 0:
        val = Text()
        val.append(f"{format_tokens_compact(split.harness_tokens)} tokens")
        val.append(f" · {_cost_or_unknown(split.harness_cost, True)}", style="dim")
        lines.append(_label_line("Harness", val))
    return lines


def _heatmap_section(daily: list[DailyBucket], inner_w: int) -> list[Text]:
    """14-day activity heatmap: a 7-column grid coloured by token volume.

    Replaces the old single-row sparkline. Only shown when there's activity.
    """
    if not any(d.total_tokens > 0 for d in daily):
        return []
    cells = daily[-14:]
    max_tokens = max((d.total_tokens for d in daily), default=0) or 1
    lines: list[Text] = [Text("  ── Activity (last 14 days) ──", style="dim"), Text()]
    per_row = 7
    for start in range(0, len(cells), per_row):
        chunk = cells[start : start + per_row]
        row = Text()
        for i, d in enumerate(chunk):
            if d.total_tokens == 0:
                row.append(_HEAT_EMPTY, style=_HEAT_STYLES[0])
            else:
                idx = _heat_index(d.total_tokens / max_tokens)
                row.append(_HEAT_CELL, style=_HEAT_STYLES[idx])
            if i < len(chunk) - 1:
                row.append(" ")
        lines.append(_center(row, inner_w))
    lines.append(Text())
    legend = Text()
    legend.append("less ", style="dim")
    for idx in range(1, len(_HEAT_STYLES)):
        if idx > 1:
            legend.append(" ")
        legend.append(_HEAT_CELL, style=_HEAT_STYLES[idx])
    legend.append(" more", style="dim")
    lines.append(_center(legend, inner_w))
    active = [d for d in daily if d.total_tokens > 0]
    stats = Text(
        f"peak {format_tokens_compact(max_tokens)} · "
        f"latest {format_tokens_compact(daily[-1].total_tokens)} · "
        f"earliest {format_tokens_compact(active[0].total_tokens)}",
        style="dim",
    )
    lines.append(_center(stats, inner_w))
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


def _format_reset_at(resets_at: int | None) -> str | None:
    """Format a unix-seconds reset timestamp as a local HH:MM (or +Md)."""
    if resets_at is None or resets_at <= 0:
        return None
    import time

    delta = resets_at - time.time()
    if delta <= 0:
        return "resets soon"
    if delta < _SECONDS_PER_HOUR:
        return f"resets in {int(delta / _SECONDS_PER_MINUTE)}m"
    if delta < 24 * _SECONDS_PER_HOUR:
        return f"resets in {delta / _SECONDS_PER_HOUR:.1f}h"
    days = int(delta / (24 * _SECONDS_PER_HOUR))
    return f"resets in {days}d"


def _window_label(window: CodexQuotaWindow) -> str:
    """5h / weekly / etc. — derived from window_minutes when present."""
    if window.window_minutes is None:
        return "Limit"
    mins = window.window_minutes
    hours = mins / _MINS_PER_HOUR
    # Folded to one return per branch via early assignments to keep return-count
    # under the linter cap; special-cases below override the default.
    label: str
    if mins <= _MINS_PER_HOUR:
        label = f"{mins}m limit"
    elif abs(hours - _HOURS_PER_WEEK) < 1:
        label = "Weekly limit"
    elif hours.is_integer():
        whole = int(hours)
        label = "5h limit" if whole == _FIVE_HOURS else f"{whole}h limit"
    elif abs(hours % 24) < _FLOAT_EQ_EPSILON:
        label = f"{int(hours // 24)}d limit"
    else:
        label = f"{mins}m limit"
    return label


def _codex_quota_section(quota: CodexQuotaSnapshot) -> list[Text]:
    lines: list[Text] = [
        Text("  ── Codex quota (ChatGPT plan) ──", style="dim"),
        Text(),
    ]
    if quota.primary is not None:
        w = quota.primary
        val = Text()
        val.append(_progress_bar(w.percent_left / 100.0) + " ")
        val.append(f"{w.percent_left:.0f}% left")
        reset = _format_reset_at(w.resets_at)
        if reset:
            val.append(f" · {reset}", style="dim")
        lines.append(_label_line(_window_label(w), val))
    if quota.secondary is not None:
        w = quota.secondary
        val = Text()
        val.append(_progress_bar(w.percent_left / 100.0) + " ")
        val.append(f"{w.percent_left:.0f}% left")
        reset = _format_reset_at(w.resets_at)
        if reset:
            val.append(f" · {reset}", style="dim")
        lines.append(_label_line(_window_label(w), val))
    if quota.credits is not None:
        _append_credits_line(lines, quota.credits)
    if quota.monthly_limit is not None:
        _append_monthly_line(lines, quota.monthly_limit)
    return lines


def _append_credits_line(lines: list[Text], credits: CodexCredits) -> None:
    if not credits.has_credits:
        return
    val = Text()
    if credits.unlimited:
        val.append("Unlimited")
    elif credits.balance:
        val.append(f"{credits.balance} credits")
    else:
        return
    lines.append(_label_line("Credits", val))


def _append_monthly_line(lines: list[Text], monthly: CodexMonthlyLimit) -> None:
    val = Text()
    val.append(_progress_bar(monthly.percent_left / 100.0) + " ")
    val.append(f"{monthly.percent_left:.0f}% left ")
    val.append(f"({monthly.used}/{monthly.limit})", style="dim")
    reset = _format_reset_at(monthly.resets_at)
    if reset:
        val.append(f" · {reset}", style="dim")
    lines.append(_label_line("Monthly limit", val))


def render_status_card(data: StatusCardData) -> Text:
    """Build the full status card as a Rich ``Text`` (box-drawing included).

    Pure function: same inputs → identical output, so it snapshots cleanly.
    """
    # Adapt the card width to fit long model names (clamped); every line is
    # cell-clipped to the inner width afterwards, so the border is inviolable.
    if data.summary.providers:
        model_col = _pick_model_col(data.summary.providers)
    else:
        model_col = _MODEL_COL_MIN
    width = min(_MAX_CARD_WIDTH, max(data.width, model_col + _PROVIDER_TAIL + 4))
    inner_w = width - 4
    lines: list[Text] = [_header(data.version), Text()]
    lines.extend(
        _config_section(
            data.model_name, data.provider_name, data.workdir, data.session_id
        )
    )
    lines.append(Text())
    lines.extend(_session_section(data.stats, data.context_window))
    lines.append(Text())

    if data.summary.providers:
        lines.extend(_provider_section(data.summary.providers, model_col))
        lines.append(Text())
    if data.codex_quota is not None:
        lines.extend(_codex_quota_section(data.codex_quota))
        lines.append(Text())
    if data.rate_limits:
        lines.extend(_limits_section(data.rate_limits))
        lines.append(Text())
    harness = _harness_section(data.summary.harness)
    if harness:
        lines.extend(harness)
        lines.append(Text())
    heatmap = _heatmap_section(data.summary.daily, inner_w)
    if heatmap:
        lines.extend(heatmap)
        lines.append(Text())
    if data.summary.windows:
        lines.extend(_windows_section(data.summary.windows))
    return _box(lines, width)


def _box(lines: list[Text], width: int) -> Text:
    """Wrap rendered lines in a rounded border, padding to ``width``.

    Every line is cell-clipped to the inner width before the right border is
    drawn, so no input — however long — can push the border out of alignment.
    """
    top = Text(f"╭{'─' * (width - 2)}╮", style="dim")
    bottom = Text(f"╰{'─' * (width - 2)}╯", style="dim")
    out = Text()
    out.append_text(top)
    out.append("\n")
    inner_w = width - 4
    for line in lines:
        # cell_len measures terminal cell width (CJK/emoji = 2), so the right
        # border stays aligned even when a model/dir name contains wide glyphs.
        if cell_len(line.plain) > inner_w:
            line = line.copy()
            line.truncate(inner_w, overflow="ellipsis")
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
