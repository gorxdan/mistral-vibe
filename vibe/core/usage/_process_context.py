from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from vibe.core.usage._context import SpendPurpose

__all__ = [
    "SPEND_PROCESS_CONTEXT_ENV",
    "SpendProcessContext",
    "SpendProcessContextError",
    "decode_spend_process_context",
    "encode_spend_process_context",
    "install_spend_process_context",
    "load_spend_process_context",
]

SPEND_PROCESS_CONTEXT_ENV = "VIBE_SPEND_CONTEXT"


class SpendProcessContextError(ValueError):
    pass


class SpendProcessContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    ledger_path: str = Field(min_length=1, max_length=4096)
    session_scope_id: str = Field(min_length=1, max_length=256)
    agent_scope_id: str = Field(min_length=1, max_length=256)
    purpose: SpendPurpose
    task_brief_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("ledger_path")
    @classmethod
    def validate_ledger_path(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("spend ledger path must be absolute")
        return str(path.resolve())


def encode_spend_process_context(context: SpendProcessContext) -> str:
    return context.model_dump_json()


def decode_spend_process_context(value: str) -> SpendProcessContext:
    try:
        return SpendProcessContext.model_validate_json(value)
    except ValidationError as e:
        raise SpendProcessContextError("invalid spend process context") from e


def load_spend_process_context(
    env: Mapping[str, str] | None = None,
) -> SpendProcessContext | None:
    value = (os.environ if env is None else env).get(SPEND_PROCESS_CONTEXT_ENV)
    if value is None:
        return None
    return decode_spend_process_context(value)


def install_spend_process_context(
    env: MutableMapping[str, str], context: SpendProcessContext | None
) -> None:
    env.pop(SPEND_PROCESS_CONTEXT_ENV, None)
    if context is not None:
        env[SPEND_PROCESS_CONTEXT_ENV] = encode_spend_process_context(context)
