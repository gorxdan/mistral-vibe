from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TurnSummaryData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_message: str
    message_id: str | None = None
    assistant_fragments: list[str] = Field(default_factory=list)
    error: str | None = None


class TurnSummaryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation: int
    summary: str | None
