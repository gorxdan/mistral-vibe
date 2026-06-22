from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.tools.base import BaseTool


class ParsedToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    tool_name: str
    raw_args: dict[str, Any]
    call_id: str = ""


class ResolvedToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    tool_name: str
    tool_class: type[BaseTool]
    validated_args: BaseModel
    call_id: str = ""

    @property
    def args_dict(self) -> dict[str, Any]:
        return self.validated_args.model_dump()


class FailedToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    tool_name: str
    call_id: str
    error: str


class ParsedMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    tool_calls: list[ParsedToolCall]


class ResolvedMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    tool_calls: list[ResolvedToolCall]
    failed_calls: list[FailedToolCall] = Field(default_factory=list)
