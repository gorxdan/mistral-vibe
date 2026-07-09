from __future__ import annotations

from enum import StrEnum, auto
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vibe.core.tasking import (
    TaskBrief,
    TaskOutcome,
    TaskOutcomeStatus,
    compile_task_brief,
)

LEGACY_TASK_PROTOCOL_VERSION = 1
STRUCTURED_TASK_PROTOCOL_VERSION = 2


class TaskStatus(StrEnum):
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    BLOCKED = auto()


class TeamSafetyMode(StrEnum):
    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[str]
    ) -> str:
        del start, count, last_values
        return name.lower().replace("_", "-")

    SHARED = auto()
    SHARED_ASK = auto()


class MessageKind(StrEnum):
    """Structured message kinds for typed teammate ↔ lead communication.

    ``TEXT`` is the default and covers all legacy free-form prose traffic.
    The structured kinds support typed request/response cycles between a
    teammate subprocess and the lead:

    - ``PERMISSION_REQUEST``: a teammate asks the lead to approve a destructive
      action it would otherwise auto-approve in isolation (e.g. a destructive
      bash command). ``payload`` carries the tool name and a description.
    - ``PERMISSION_RESPONSE``: the lead's reply to a ``PERMISSION_REQUEST``.
      ``payload`` carries the original request id, the decision (allow/deny),
      and an optional reason.
    - ``PLAN_APPROVAL`` / ``SHUTDOWN``: reserved for future use (defined so the
      enum is stable); not yet emitted by any teammate code.
    """

    TEXT = auto()
    PERMISSION_REQUEST = auto()
    PERMISSION_RESPONSE = auto()
    PLAN_APPROVAL = auto()
    SHUTDOWN = auto()


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    protocol_version: Literal[1, 2] = LEGACY_TASK_PROTOCOL_VERSION
    brief: TaskBrief | None = None
    status: TaskStatus = TaskStatus.PENDING
    outcome: TaskOutcome | None = None
    assignee: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    created_at: float = 0.0
    claimed_at: float | None = None
    completed_at: float | None = None
    result: str | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_record(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if brief_data := data.get("brief"):
            brief = TaskBrief.model_validate(brief_data)
            data.setdefault("description", brief.objective)
            data["protocol_version"] = STRUCTURED_TASK_PROTOCOL_VERSION
        if data.get("outcome") is not None:
            return data
        status = data.get("status", TaskStatus.PENDING)
        status_value = status.value if isinstance(status, TaskStatus) else status
        result = data.get("result")
        summary = result.strip() if isinstance(result, str) and result.strip() else None
        if status_value == TaskStatus.COMPLETED.value:
            data["outcome"] = {
                "status": TaskOutcomeStatus.SUCCEEDED,
                "summary": summary or "Legacy task completed",
            }
        elif status_value == TaskStatus.BLOCKED.value:
            data["outcome"] = {
                "status": TaskOutcomeStatus.BLOCKED,
                "summary": summary or "Legacy task blocked",
            }
        return data

    @model_validator(mode="after")
    def validate_protocol_state(self) -> Task:
        if self.brief is not None:
            self.protocol_version = STRUCTURED_TASK_PROTOCOL_VERSION
            if self.outcome is not None:
                if self.outcome.manifest is None:
                    self.outcome = self.outcome.model_copy(
                        update={"manifest": self.brief.manifest}
                    )
                elif self.outcome.manifest != self.brief.manifest:
                    raise ValueError("task outcome manifest does not match task brief")
        elif self.protocol_version == STRUCTURED_TASK_PROTOCOL_VERSION:
            raise ValueError("task protocol version 2 requires a task brief")
        if self.status in {TaskStatus.COMPLETED, TaskStatus.BLOCKED}:
            if self.outcome is None:
                raise ValueError("terminal task lifecycle requires an outcome")
        elif self.outcome is not None and not self.outcome.retryable:
            raise ValueError(
                "active task lifecycle may only retain a retryable outcome"
            )
        return self

    @property
    def structured(self) -> bool:
        return self.brief is not None

    @property
    def prompt(self) -> str:
        if self.brief is None:
            return self.description
        return compile_task_brief(self.brief)


class TeamMember(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    session_id: str | None = None
    agent_type: str = "default"
    status: str = "idle"
    pid: int | None = None
    spawn_prompt: str | None = None
    max_turns: int | None = None
    worker: bool = False
    safety_mode: TeamSafetyMode = TeamSafetyMode.SHARED
    last_task_id: str | None = None
    last_claimed_at: float | None = None


class TeamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    team_name: str
    created_at: float
    members: list[TeamMember] = Field(default_factory=list)
    team_dir: str = ""
    lead_session_id: str = ""


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    from_name: str
    to_name: str
    content: str
    timestamp: float
    read: bool = False
    kind: MessageKind = MessageKind.TEXT
    payload: dict = Field(default_factory=dict)
