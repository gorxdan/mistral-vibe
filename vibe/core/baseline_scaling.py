"""Window-tiered baseline scaling policy.

Pure classification + gating for shrinking the irreducible baseline (system
prompt prose, tool-schema text, project context) on small-window models. The
single opt-in is ModelConfig.context_window: a model with no declared window is
always tier LARGE, so every gate below collapses to today's behaviour.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibe.core.config._settings import ModelConfig, VibeConfig


class BaselineTier(StrEnum):
    LARGE = "large"
    MEDIUM = "medium"
    SMALL = "small"


# Optional system-prompt prose sections, keyed by a stable name. A tier emits a
# section iff that tier is in the section's set. A section NOT listed here is
# always emitted (not gateable). LARGE is in every set, so a LARGE model renders
# every section verbatim — byte-identical to before baseline scaling existed.
_L = BaselineTier.LARGE
_M = BaselineTier.MEDIUM
_S = BaselineTier.SMALL
_SECTION_TIERS: dict[str, frozenset[BaselineTier]] = {
    # Largest optional blocks — dropped first (MEDIUM already sheds these two).
    "config_reference": frozenset({_L}),
    "le_chaton_long": frozenset({_L}),
    # Mid-cost prose — kept through MEDIUM, dropped at SMALL.
    "model_routing_list": frozenset({_L, _M}),
    "orchestration_prose": frozenset({_L, _M}),
    "verification_contract": frozenset({_L, _M}),
    "investigation_contract": frozenset({_L, _M}),
    "humanizer": frozenset({_L, _M}),
    "worktree_detail": frozenset({_L, _M}),
    "skills_summaries": frozenset({_L, _M}),
}


def baseline_tier_for(model: ModelConfig, config: VibeConfig) -> BaselineTier:
    """Classify a model's baseline tier. Opt-in: LARGE unless the model declares
    a context_window AND baseline_scaling is enabled.
    """
    bs = config.baseline_scaling
    if not bs.enabled or model.context_window is None:
        return BaselineTier.LARGE
    window = model.effective_context_window
    if window < bs.small_max:
        return BaselineTier.SMALL
    if window < bs.medium_max:
        return BaselineTier.MEDIUM
    return BaselineTier.LARGE


def section_enabled(tier: BaselineTier, section: str) -> bool:
    """Whether an optional system-prompt section is emitted at this tier.
    Unknown sections are always emitted (only the registered ones are gateable).
    """
    tiers = _SECTION_TIERS.get(section)
    return True if tiers is None else tier in tiers


def trim_tool_descriptions(tier: BaselineTier, config: VibeConfig) -> bool:
    """Whether builtin tool-schema descriptions are trimmed at this tier."""
    return tier is BaselineTier.SMALL and config.baseline_scaling.trim_tool_descriptions_small


def agents_md_byte_budget(tier: BaselineTier, config: VibeConfig) -> int | None:
    """Per-doc byte cap for injected AGENTS.md on SMALL (None = unlimited)."""
    if tier is BaselineTier.SMALL:
        return config.baseline_scaling.small_agents_md_bytes
    return None


def budget_doc(body: str, budget: int | None) -> str:
    """Apply a byte budget to a doc body: None = unlimited, 0 = drop (empty),
    >0 = truncate to that many bytes with a marker.
    """
    if budget is None:
        return body
    if budget <= 0:
        return ""
    if len(body) <= budget:
        return body
    return body[:budget].rstrip() + "\n…[truncated for small context window]"


def scaled_guard_tokens(
    config: VibeConfig, model: ModelConfig, tier: BaselineTier
) -> int:
    """The cache-prefix guard band, scaled to the window for tiered models. LARGE
    returns the live config value unchanged (preserving byte-identity); smaller
    tiers shrink it so the fixed 4000 isn't a large fraction of a small window.
    """
    raw = config.context_shaping.cache_prefix_guard_tokens
    if tier is BaselineTier.LARGE:
        return raw
    bs = config.baseline_scaling
    window = model.effective_context_window
    return min(raw, max(bs.guard_floor, int(window * bs.guard_window_fraction)))
