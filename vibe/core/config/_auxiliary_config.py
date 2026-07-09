from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuxiliaryBudgetConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    max_tokens: int = Field(default=50_000, ge=0)
    max_calls: int = Field(default=24, ge=0)
    max_cost_usd: float = Field(default=1.0, ge=0)
