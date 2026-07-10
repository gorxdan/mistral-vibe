from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.failure_diagnostic import FailureDiagnostic
from vibe.core.tools.base import BaseTool


class ParsedToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    tool_name: str
    raw_args: dict[str, Any]
    raw_text: str = ""
    parse_error: FailureDiagnostic | None = None
    repaired: bool = False
    call_id: str = ""


class ResolvedToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")
    tool_name: str
    tool_class: type[BaseTool]
    validated_args: BaseModel
    call_id: str = ""

    @property
    def args_dict(self) -> dict[str, Any]:
        return self.validated_args.model_dump()


class FailedToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    tool_name: str
    call_id: str
    error: str
    diagnostic: FailureDiagnostic | None = None


class ParsedMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    tool_calls: list[ParsedToolCall]


class ResolvedMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    tool_calls: list[ResolvedToolCall]
    failed_calls: list[FailedToolCall] = Field(default_factory=list)
