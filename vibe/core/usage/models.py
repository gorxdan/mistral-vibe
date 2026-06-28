from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from vibe.core.types import LLMUsage


class UsageRecord(BaseModel):
    """One persisted LLM call, the atom the status usage windows aggregate over."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: float
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    # Worst-case cost in USD (no caching discount applied); matches AgentStats.session_cost.
    cost_usd: float = 0.0
    duration_s: float = 0.0
    session_id: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def non_cached_input(self) -> int:
        return max(self.prompt_tokens - self.cached_tokens, 0)

    @classmethod
    def from_usage(
        cls,
        *,
        timestamp: float,
        provider: str,
        model: str,
        usage: LLMUsage,
        cost_usd: float,
        duration_s: float,
        session_id: str,
    ) -> UsageRecord:
        return cls(
            timestamp=timestamp,
            provider=provider,
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_tokens=usage.cached_tokens,
            cost_usd=cost_usd,
            duration_s=duration_s,
            session_id=session_id,
        )
