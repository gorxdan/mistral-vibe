from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vibe.core.lsp._defaults import (
    ServerPreset,
    available_presets,
    broken_presets,
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


def _matching_preset(file_path: str | Path, config: VibeConfig) -> ServerPreset | None:
    """Preset matching ``file_path`` if a nudge could help, else None.

    None when LSP is already installed, the file has no extension, or no
    preset matches the extension. The preset's binary need NOT be on PATH —
    an absent binary surfaces an ``install_hint`` nudge so the user learns
    what to install at the moment they edit a file in that language.
    """
    if "lsp" in getattr(config, "installed_components", []):
        return None
    ext = Path(file_path).suffix
    if not ext:
        return None
    return preset_for_extension(ext)


def evaluate_nudge(
    file_path: str | Path,
    config: VibeConfig,
    cache_path: Path,
    *,
    turns_since_last: int = 0,
) -> NudgeDecision:
    """Decide whether editing ``file_path`` should surface an LSP nudge.

    Two independent paths, gated on whether the matching server binary is on
    PATH:

    - Server available, LSP feature off: ``first_prompt`` (offer to enable
      LSP), then ``reminder`` on a cadence, then ``silent`` once the cap hits.
    - Server absent: ``install_hint`` (carry the install command) so the user
      learns what to run. Respects its own declined cap so we don't nag users
      who can't or won't install the toolchain.

    ``skip`` when LSP is already installed, the file has no preset, or the
    matching preset's binary is broken (half-installed) — broken states belong
    in /lsp status, not in a passive nudge.
    """
    preset = _matching_preset(file_path, config)
    if preset is None:
        return NudgeDecision(kind="skip")

    available_keys = {p.key for p in available_presets()}
    if preset.key in available_keys:
        return _enable_nudge(preset, cache_path, turns_since_last)
    if preset.key in {p.preset.key for p in broken_presets()}:
        return NudgeDecision(kind="skip")
    return _install_hint_nudge(preset, cache_path, turns_since_last)


def _enable_nudge(
    preset: ServerPreset, cache_path: Path, turns_since_last: int
) -> NudgeDecision:
    offer = NudgeDecision(
        kind="first_prompt",
        preset_display_name=preset.display_name,
        install_hint=preset.install_hint,
    )
    state = _read_nudge_state(cache_path)
    if not state.get("offered_once") or state.get("declined") is False:
        return offer
    reminders_shown = int(state.get("reminders_shown", 0))
    if reminders_shown >= MAX_REMINDERS:
        return NudgeDecision(kind="silent")
    if turns_since_last >= REMINDER_INTERVAL_TURNS:
        return NudgeDecision(kind="reminder", preset_display_name=preset.display_name)
    return NudgeDecision(kind="silent")


def _install_hint_nudge(
    preset: ServerPreset, cache_path: Path, turns_since_last: int
) -> NudgeDecision:
    state = _read_nudge_state(cache_path)
    hint_key = f"hint_declined:{preset.key}"
    if state.get(hint_key) is True:
        hints_shown = int(state.get(f"hint_shown:{preset.key}", 0))
        if hints_shown >= MAX_REMINDERS:
            return NudgeDecision(kind="silent")
        if turns_since_last >= REMINDER_INTERVAL_TURNS:
            return NudgeDecision(
                kind="install_hint",
                preset_display_name=preset.display_name,
                install_hint=preset.install_hint,
            )
        return NudgeDecision(kind="silent")
    return NudgeDecision(
        kind="install_hint",
        preset_display_name=preset.display_name,
        install_hint=preset.install_hint,
    )


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


def record_install_hint_shown(preset_key: str, cache_path: Path) -> None:
    state = _read_nudge_state(cache_path)
    key = f"hint_shown:{preset_key}"
    updates = {key: int(state.get(key, 0)) + 1}
    if not state.get(f"hint_declined:{preset_key}"):
        updates[f"hint_first_seen:{preset_key}"] = True
    _write_nudge_state(cache_path, **updates)


def record_install_hint_declined(preset_key: str, cache_path: Path) -> None:
    _write_nudge_state(cache_path, **{f"hint_declined:{preset_key}": True})


def reset_nudge_state(cache_path: Path) -> None:
    """Clear nudge state — used when LSP is installed via /lspstall."""
    from vibe.cli.cache import write_cache

    write_cache(cache_path, _CACHE_SECTION, {"offered_once": True, "declined": False})
