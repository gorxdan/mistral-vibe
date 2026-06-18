from __future__ import annotations

from enum import StrEnum, auto

from pydantic import BaseModel, ConfigDict, Field


class TaskStatus(StrEnum):
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    BLOCKED = auto()


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
