from __future__ import annotations

import datetime as _dt
from enum import StrEnum
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Engine-neutral anchors (^/$ work in both Python `re` and the Rust regex
# pydantic uses for Field validation; \A/\Z and \z are NOT shared, so neither
# can anchor both). The path-gating code in store.py pairs this with
# _ID_RE.fullmatch() — NOT .match() — because Python `$` also matches just
# before a trailing newline, so "slug\n" would otherwise pass and interpolate a
# newline into a filename.
_SLUG = r"^[a-z0-9]+(-[a-z0-9]+)*$"

# Max length of a memory's frontmatter description: a recall-length summary only
# (the full text lives in the body). Over-length values are truncated by the
# _clamp_description validator rather than rejected, so a long model-authored
# description never fails the save.
_DESC_MAX = 300

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
    age = memory_age_days(updated, today)
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
    age = memory_age_days(updated, today)
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


def memory_age_days(updated: str, today: _dt.date | None = None) -> int | None:
    # Age of a memory in days from its frontmatter `updated` date. None for
    # missing/unparseable dates so callers can treat "unknown age" distinctly
    # from "fresh" (0).
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
    description: str = Field(default="", max_length=_DESC_MAX)
    tags: list[str] = Field(default_factory=list)
    type: MemoryType | None = None
    scope: Literal["user", "project"] = "user"
    created: str = ""
    updated: str = ""
    source: Literal["tool", "auto", "manual"] = "manual"
    # Originating session id for auditability — lets a surfaced memory be traced
    # back to the session/turn that produced it (auto-extracted memories
    # especially). Empty for legacy/manual memories; never used for recall.
    session_id: str = ""

    @field_validator("description", mode="before")
    @classmethod
    def _clamp_description(cls, v: object) -> object:
        # Truncate rather than reject: the body holds the full text, so the
        # frontmatter description only needs a recall-length summary. Guards
        # long model-authored descriptions and legacy over-length files alike.
        if isinstance(v, str) and len(v) > _DESC_MAX:
            return v[:_DESC_MAX]
        return v

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

    def index_line(self, today: _dt.date | None = None) -> str:
        m = self.metadata
        tags = f" (tags: {', '.join(m.tags)})" if m.tags else ""
        desc = f": {m.description}" if m.description else ""
        scope = " (project)" if m.scope == "project" else ""
        age = age_label(m.updated, today)
        # Fold age into the bracketed tag so it stays one token: `[project, 3d]`.
        # When there is no type AND no age, omit the brackets entirely (legacy
        # shape); with either, include both comma-separated.
        parts = [p for p in (m.type.value if m.type is not None else None, age) if p]
        type_tag = f" [{', '.join(parts)}]" if parts else ""
        return f"- [{m.id}]{type_tag} {m.title}{desc}{tags}{scope}"
