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
# Age-bucket boundaries for the compact recency cue in index_line().
_WEEK_DAYS = 7
_MONTH_DAYS = 30
_YEAR_DAYS = 365


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
    age = _age_days(updated, today)
    if age is None or age <= _STALE_DAYS:
        return ""
    return (
        f"_(updated {age} days ago; verify against current code before relying "
        "on file:line or API details)_"
    )


def age_label(updated: str, today: _dt.date | None = None) -> str:
    # Compact recency cue for the selector's index line (e.g. "3d", "2w", "1mo"),
    # so recall can weigh freshness alongside relevance. Empty for unknown dates
    # (no frontmatter `updated`) so legacy entries render unchanged.
    age = _age_days(updated, today)
    if age is None or age < 0:
        return ""
    if age == 0:
        return "today"
    if age < _WEEK_DAYS:
        return f"{age}d"
    if age < _MONTH_DAYS:
        return f"{age // _WEEK_DAYS}w"
    if age < _YEAR_DAYS:
        return f"{age // _MONTH_DAYS}mo"
    return f"{age // _YEAR_DAYS}y"


def _age_days(updated: str, today: _dt.date | None = None) -> int | None:
    if not updated:
        return None
    try:
        d = _dt.date.fromisoformat(updated)
    except ValueError:
        return None
    return ((today or _dt.date.today()) - d).days


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
        age = age_label(m.updated)
        # Fold age into the bracketed tag so it stays one token: `[project, 3d]`.
        # When there is no type AND no age, omit the brackets entirely (legacy
        # shape); with either, include both comma-separated.
        parts = [p for p in (m.type.value if m.type is not None else None, age) if p]
        type_tag = f" [{', '.join(parts)}]" if parts else ""
        return f"- [{m.id}]{type_tag} {m.title}{desc}{tags}{scope}"
