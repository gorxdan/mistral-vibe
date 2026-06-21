from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_SLUG = r"^[a-z0-9]+(-[a-z0-9]+)*$"


class MemoryMetadata(BaseModel):
    """Frontmatter of a memory file (the cheap header the selector scans)."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(pattern=_SLUG)
    title: str
    description: str = Field(default="", max_length=300)
    tags: list[str] = Field(default_factory=list)
    scope: Literal["user", "project"] = "user"
    created: str = ""
    updated: str = ""
    source: Literal["tool", "auto", "manual"] = "manual"


class MemoryEntry(BaseModel):
    """A parsed memory: frontmatter metadata + full markdown body."""

    metadata: MemoryMetadata
    body: str = ""

    @property
    def id(self) -> str:
        return self.metadata.id

    def index_line(self) -> str:
        m = self.metadata
        tags = f" (tags: {', '.join(m.tags)})" if m.tags else ""
        desc = f": {m.description}" if m.description else ""
        scope = " (project)" if m.scope == "project" else ""
        return f"- [{m.id}] {m.title}{desc}{tags}{scope}"
