from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["SpendConfig"]


class SpendConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", allow_inf_nan=False)

    max_prompt_tokens: int = Field(default=400_000, ge=0)
    max_completion_tokens: int = Field(default=100_000, ge=0)
    max_total_tokens: int = Field(default=500_000, ge=0)
    max_cost_usd: float = Field(default=10.0, ge=0.0)
    max_calls: int = Field(default=128, ge=0)
    max_concurrent_calls: int = Field(default=2, ge=0)
    max_retries: int = Field(default=16, ge=0)
    deadline_seconds: float | None = Field(default=None, gt=0.0)
    default_max_output_tokens: int = Field(default=32_768, gt=0)
    unpriced_input_usd_per_million: float = Field(default=10.0, ge=0.0)
    unpriced_output_usd_per_million: float = Field(default=30.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_token_bounds(self) -> SpendConfig:
        if self.max_prompt_tokens > self.max_total_tokens:
            raise ValueError("max_prompt_tokens cannot exceed max_total_tokens")
        if self.max_completion_tokens > self.max_total_tokens:
            raise ValueError("max_completion_tokens cannot exceed max_total_tokens")
        return self
