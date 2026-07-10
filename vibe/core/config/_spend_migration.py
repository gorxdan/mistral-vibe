from __future__ import annotations

from pathlib import Path
import tomllib

import tomli_w

from vibe.core.config.harness_files import HarnessFilesManager
from vibe.core.utils.io import read_safe, write_safe

SPEND_DYNAMIC_DEFAULTS_MIGRATION = "spend_dynamic_token_defaults_v1"
LEGACY_GENERATED_SPEND_DEFAULTS: dict[str, int | float] = {
    "max_prompt_tokens": 400_000,
    "max_completion_tokens": 100_000,
    "max_total_tokens": 500_000,
    "max_cost_usd": 10.0,
    "max_calls": 128,
    "max_concurrent_calls": 2,
    "max_retries": 16,
    "default_max_output_tokens": 32_768,
    "unpriced_input_usd_per_million": 10.0,
    "unpriced_output_usd_per_million": 30.0,
}
LEGACY_SPEND_TOKEN_LIMITS = (
    "max_prompt_tokens",
    "max_completion_tokens",
    "max_total_tokens",
)


def migrate_legacy_generated_spend_defaults(file: Path) -> None:
    try:
        data = tomllib.loads(read_safe(file, raise_on_error=True).text)
    except (FileNotFoundError, OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return

    applied = data.get("applied_migrations", [])
    if not isinstance(applied, list) or not all(
        isinstance(item, str) for item in applied
    ):
        return
    if SPEND_DYNAMIC_DEFAULTS_MIGRATION in applied:
        return
    spend = data.get("spend")
    if not isinstance(spend, dict) or spend != LEGACY_GENERATED_SPEND_DEFAULTS:
        return

    for key in LEGACY_SPEND_TOKEN_LIMITS:
        spend.pop(key)
    data["applied_migrations"] = [*applied, SPEND_DYNAMIC_DEFAULTS_MIGRATION]
    write_safe(file, tomli_w.dumps(data))


def prepare_spend_migration(manager: HarnessFilesManager) -> Path | None:
    if not manager.persist_allowed:
        return None
    user_config = manager.user_config_file
    active_config = manager.config_file
    migrate_legacy_generated_spend_defaults(user_config)
    if active_config is not None and active_config != user_config:
        migrate_legacy_generated_spend_defaults(active_config)
    return active_config


__all__ = [
    "LEGACY_GENERATED_SPEND_DEFAULTS",
    "LEGACY_SPEND_TOKEN_LIMITS",
    "SPEND_DYNAMIC_DEFAULTS_MIGRATION",
    "migrate_legacy_generated_spend_defaults",
    "prepare_spend_migration",
]
