from __future__ import annotations

from enum import StrEnum, auto

from pydantic import BaseModel, ConfigDict, Field


class TaskStatus(StrEnum):
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    BLOCKED = auto()


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
    status: TaskStatus = TaskStatus.PENDING
    assignee: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    created_at: float = 0.0
    completed_at: float | None = None
    result: str | None = None


class TeamMember(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    session_id: str | None = None
    agent_type: str = "default"
    status: str = "idle"
    pid: int | None = None


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
