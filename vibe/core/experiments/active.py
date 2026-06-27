from __future__ import annotations

from enum import StrEnum
from typing import Final


class ExperimentName(StrEnum):
    SYSTEM_PROMPT = "vibe_cli_system_prompt"
    PROMPT_COMPRESSION = "vibe_prompt_compression"


DEFAULT_VARIANTS: Final[dict[ExperimentName, str]] = {
    ExperimentName.SYSTEM_PROMPT: "cli",
    ExperimentName.PROMPT_COMPRESSION: "compressed",
}

assert all(name in DEFAULT_VARIANTS for name in ExperimentName), (
    "Every ExperimentName must have a default in DEFAULT_VARIANTS"
)
