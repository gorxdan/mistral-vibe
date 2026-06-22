from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_NUAGE_PROJECT_NAME = "Vibe CLI"


class NuageTextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "text"
    text: str


class NuageMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = "user"
    parts: list[NuageTextPart]


class NuageDiff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["git-diff"] = "git-diff"
    encoding: Literal["base64"] = "base64"
    compression: Literal["zstd"] = "zstd"
    content: str


class NuageRepository(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_url: str = Field(serialization_alias="repoUrl")
    branch: str | None = None
    commit_sha: str | None = Field(default=None, serialization_alias="commitSha")
    diff: NuageDiff | None = None


class NuageContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repositories: list[NuageRepository]


class NuageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str = Field(
        default=DEFAULT_NUAGE_PROJECT_NAME, serialization_alias="project_name"
    )
    source: str = "vibe_code_cli"
    idempotency_key: str = Field(serialization_alias="idempotencyKey")
    message: NuageMessage
    context: NuageContext


class TeleportSession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, object]
    messages: list[dict[str, object]]


class NuageResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    nuage_session_id: str = Field(validation_alias="sessionId")
    nuage_web_session_id: str = Field(validation_alias="webSessionId")
    nuage_project_id: str = Field(validation_alias="projectId")
    status: str
    url: str
