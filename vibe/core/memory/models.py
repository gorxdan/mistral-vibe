from __future__ import annotations

import datetime as _dt
from enum import StrEnum
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SLUG = r"^[a-z0-9]+(-[a-z0-9]+)*$"

# Memories older than this (days) get a "verify before relying on it" caveat
# when their full body is injected: file:line and API details drift over time.
_STALE_DAYS = 7


class MemoryType(StrEnum):
    # Values are a serialization contract (persisted to/read from YAML
    # frontmatter), so they are explicit lowercase strings rather than auto().
    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "memory"


def freshness_note(updated: str, today: _dt.date | None = None) -> str:
    # Returns '' for fresh/unknown dates so callers add no noise. A memory citing
    # code state can go stale; the age cue nudges verification over assertion.
    if not updated:
        return ""
    try:
        d = _dt.date.fromisoformat(updated)
    except ValueError:
        return ""
    today = today or _dt.date.today()
    age = (today - d).days
    if age <= _STALE_DAYS:
        return ""
    return (
        f"_(updated {age} days ago; verify against current code before relying "
        "on file:line or API details)_"
    )


class MemoryMetadata(BaseModel):
    """Frontmatter of a memory file (the cheap header the selector scans)."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(pattern=_SLUG)
    title: str
    description: str = Field(default="", max_length=300)
    tags: list[str] = Field(default_factory=list)
    type: MemoryType | None = None
    scope: Literal["user", "project"] = "user"
    created: str = ""
    updated: str = ""
    source: Literal["tool", "auto", "manual"] = "manual"

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, v: object) -> MemoryType | None:
        # Unknown/legacy type values degrade to None instead of rejecting the
        # whole file, so a taxonomy change doesn't brick old memories.
        if v is None:
            return None
        if not isinstance(v, str):
            v = str(v)
        try:
            return MemoryType(v)
        except ValueError:
            return None


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
        type_tag = f" [{m.type.value}]" if m.type is not None else ""
        return f"- [{m.id}]{type_tag} {m.title}{desc}{tags}{scope}"
