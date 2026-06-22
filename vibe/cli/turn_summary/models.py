from __future__ import annotations

from pydantic import BaseModel, Field


class TurnSummaryData(BaseModel):
    user_message: str
    message_id: str | None = None
    assistant_fragments: list[str] = Field(default_factory=list)
    error: str | None = None


class TurnSummaryResult(BaseModel):
    generation: int
    summary: str | None
