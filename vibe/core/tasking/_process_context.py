from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import hashlib
import os
from typing import Literal

import orjson
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from vibe.core.tasking.models import TaskBrief

TASK_PROCESS_CONTEXT_ENV = "VIBE_TASK_CONTEXT"


class TaskProcessContextError(ValueError):
    pass


def task_brief_hash(brief: TaskBrief) -> str:
    payload = orjson.dumps(brief.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(payload).hexdigest()


class TaskProcessContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    brief: TaskBrief
    brief_hash: str

    @model_validator(mode="after")
    def validate_brief_hash(self) -> TaskProcessContext:
        if self.brief_hash != task_brief_hash(self.brief):
            raise ValueError("task process context brief hash does not match")
        return self

    @classmethod
    def from_brief(cls, brief: TaskBrief) -> TaskProcessContext:
        snapshot = brief.model_copy(deep=True)
        return cls(brief=snapshot, brief_hash=task_brief_hash(snapshot))


def encode_task_process_context(context: TaskProcessContext) -> str:
    return context.model_dump_json()


def decode_task_process_context(value: str) -> TaskProcessContext:
    try:
        return TaskProcessContext.model_validate_json(value)
    except ValidationError as e:
        raise TaskProcessContextError("invalid task process context") from e


def load_task_process_context(
    env: Mapping[str, str] | None = None,
) -> TaskProcessContext | None:
    value = (os.environ if env is None else env).get(TASK_PROCESS_CONTEXT_ENV)
    if value is None:
        return None
    return decode_task_process_context(value)


def install_task_process_context(
    env: MutableMapping[str, str], context: TaskProcessContext | None
) -> None:
    env.pop(TASK_PROCESS_CONTEXT_ENV, None)
    if context is not None:
        env[TASK_PROCESS_CONTEXT_ENV] = encode_task_process_context(context)


__all__ = [
    "TASK_PROCESS_CONTEXT_ENV",
    "TaskProcessContext",
    "TaskProcessContextError",
    "decode_task_process_context",
    "encode_task_process_context",
    "install_task_process_context",
    "load_task_process_context",
    "task_brief_hash",
]
