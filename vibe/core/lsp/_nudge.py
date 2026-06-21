from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vibe.core.lsp._defaults import (
    ServerPreset,
    available_presets,
    preset_for_extension,
)

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig

_CACHE_SECTION = "lsp_nudge"
# After the user declines, show gentle reminders no more often than every N
# agent turns. The first reminder is the explicit "you can use /lspstall" line;
# subsequent ones are short toasts.
REMINDER_INTERVAL_TURNS = 15
# Hard cap so we eventually stop nagging people who clearly don't want it.
MAX_REMINDERS = 5


@dataclass(frozen=True)
class NudgeDecision:
    """Outcome of evaluating whether to surface an LSP install nudge.

    kind is one of:
      - "skip"        : conditions not met (LSP on, not a code file, no binary)
      - "first_prompt": first time — show the full prompt offering to install
      - "reminder"    : user previously declined; show a gentle reminder
      - "silent"      : user declined and we've hit the reminder cap
    """

    kind: str
    preset_display_name: str = ""
    install_hint: str = ""


def _read_nudge_state(cache_path: Path) -> dict[str, Any]:
    from vibe.cli.cache import read_cache

    return read_cache(cache_path).get(_CACHE_SECTION, {})


def _write_nudge_state(cache_path: Path, **updates: Any) -> None:
    from vibe.cli.cache import write_cache

    write_cache(cache_path, _CACHE_SECTION, updates)


def _resolvable_preset(
    file_path: str | Path, config: VibeConfig
) -> ServerPreset | None:
    """The preset we could offer for this file, or None if no useful offer.

    None when: LSP already installed, file has no extension, no preset matches,
    or the preset's binary isn't on PATH (nothing to enable).
    """
    if "lsp" in getattr(config, "installed_components", []):
        return None
    ext = Path(file_path).suffix
    if not ext:
        return None
    preset = preset_for_extension(ext)
    if preset is None:
        return None
    available_keys = {p.key for p in available_presets()}
    return preset if preset.key in available_keys else None


def evaluate_nudge(
    file_path: str | Path,
    config: VibeConfig,
    cache_path: Path,
    *,
    turns_since_last: int = 0,
) -> NudgeDecision:
    """Decide whether editing ``file_path`` should surface an LSP nudge.

    Returns ``skip`` when LSP is already installed, the file has no preset,
    or the matching server binary isn't on PATH (nothing useful to offer).
    Otherwise returns ``first_prompt`` on first sight, ``reminder`` on the
    cadence above, or ``silent`` once the cap is hit.
    """
    preset = _resolvable_preset(file_path, config)
    if preset is None:
        return NudgeDecision(kind="skip")
    offer = NudgeDecision(
        kind="first_prompt",
        preset_display_name=preset.display_name,
        install_hint=preset.install_hint,
    )

    state = _read_nudge_state(cache_path)
    if not state.get("offered_once") or state.get("declined") is False:
        return offer
    # Declined path: gentle reminders on a cadence, capped.
    reminders_shown = int(state.get("reminders_shown", 0))
    if reminders_shown >= MAX_REMINDERS:
        return NudgeDecision(kind="silent")
    if turns_since_last >= REMINDER_INTERVAL_TURNS:
        return NudgeDecision(kind="reminder", preset_display_name=preset.display_name)
    return NudgeDecision(kind="silent")


def record_first_prompted(cache_path: Path) -> None:
    _write_nudge_state(cache_path, offered_once=True)


def record_declined(cache_path: Path) -> None:
    _write_nudge_state(cache_path, declined=True, last_reminder_turn=0)


def record_reminder_shown(cache_path: Path, current_turn: int) -> None:
    state = _read_nudge_state(cache_path)
    _write_nudge_state(
        cache_path,
        reminders_shown=int(state.get("reminders_shown", 0)) + 1,
        last_reminder_turn=current_turn,
    )


def reset_nudge_state(cache_path: Path) -> None:
    """Clear nudge state — used when LSP is installed via /lspstall."""
    from vibe.cli.cache import write_cache

    write_cache(cache_path, _CACHE_SECTION, {"offered_once": True, "declined": False})
