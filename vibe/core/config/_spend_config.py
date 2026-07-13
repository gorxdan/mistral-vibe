from __future__ import annotations

from enum import StrEnum, auto

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["PromptEstimatorMode", "SpendConfig"]


class PromptEstimatorMode(StrEnum):
    ADAPTIVE = auto()
    STRICT = auto()


class SpendConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", allow_inf_nan=False)

    enforce_limits: bool = Field(
        default=False,
        description=(
            "Enforce spend limits (cost/calls/tokens/concurrency/deadline/retries) "
            "by blocking calls before dispatch. When False (default), spend is "
            "tracked for /spend display but limits are advisory only."
        ),
    )
    max_prompt_tokens: int | None = Field(default=None, ge=0)
    max_completion_tokens: int | None = Field(default=None, ge=0)
    max_total_tokens: int | None = Field(default=None, ge=0)
    max_cost_usd: float = Field(default=10.0, ge=0.0)
    max_calls: int = Field(default=512, ge=0)
    max_concurrent_calls: int = Field(default=2, ge=0)
    max_retries: int = Field(default=16, ge=0)
    deadline_seconds: float | None = Field(default=None, gt=0.0)
    default_max_output_tokens: int = Field(default=32_768, gt=0)
    minimum_admitted_output_tokens: int = Field(default=256, gt=0)
    prompt_estimator_mode: PromptEstimatorMode = PromptEstimatorMode.ADAPTIVE
    unpriced_input_usd_per_million: float = Field(default=10.0, ge=0.0)
    unpriced_output_usd_per_million: float = Field(default=30.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_token_bounds(self) -> SpendConfig:
        if (
            self.max_total_tokens is not None
            and self.max_prompt_tokens is not None
            and self.max_prompt_tokens > self.max_total_tokens
        ):
            raise ValueError("max_prompt_tokens cannot exceed max_total_tokens")
        if (
            self.max_total_tokens is not None
            and self.max_completion_tokens is not None
            and self.max_completion_tokens > self.max_total_tokens
        ):
            raise ValueError("max_completion_tokens cannot exceed max_total_tokens")
        return self
