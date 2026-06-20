from __future__ import annotations

from collections.abc import AsyncGenerator
import datetime as _dt
import re
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from vibe.core.config import VibeConfig
from vibe.core.memory.models import MemoryEntry, MemoryMetadata
from vibe.core.memory.store import MemoryStore
from vibe.core.paths import VIBE_HOME
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.types import ToolStreamEvent


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "memory"


def user_memory_store() -> MemoryStore:
    return MemoryStore(user_dir=VIBE_HOME.path / "memory")


class ManageMemoryArgs(BaseModel):
    action: Literal["add", "update", "list", "delete"]
    id: str | None = None
    title: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    body: str | None = None


class ManageMemoryResult(BaseModel):
    action: str
    id: str | None = None
    message: str
    entries: list[str] = Field(default_factory=list)


class ManageMemoryConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


class ManageMemory(
    BaseTool[ManageMemoryArgs, ManageMemoryResult, ManageMemoryConfig, BaseToolState],
):
    description: ClassVar[str] = (
        "Manage durable cross-session memories (markdown files under ~/.vibe/memory). "
        "Actions: add (new memory), update (patch existing), list, delete. Memories are "
        "relevance-selected into context in later sessions."
    )

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        return config is not None and config.memory.enabled

    async def run(
        self, args: ManageMemoryArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | ManageMemoryResult, None]:
        store = user_memory_store()
        today = _dt.date.today().isoformat()

        if args.action == "list":
            index = store.index()
            yield ManageMemoryResult(
                action="list",
                message=f"{len(index)} memories",
                entries=index,
            )
            return

        if args.action == "delete":
            if not args.id:
                raise ToolError("delete requires 'id'")
            ok = store.delete(args.id)
            yield ManageMemoryResult(
                action="delete",
                id=args.id,
                message="deleted" if ok else "not found",
            )
            return

        if args.action == "add":
            if not args.title:
                raise ToolError("add requires 'title'")
            mem_id = args.id or _slugify(args.title)
            if store.get(mem_id) is not None:
                raise ToolError(
                    f"Memory '{mem_id}' already exists; use action=update instead."
                )
            entry = MemoryEntry(
                metadata=MemoryMetadata(
                    id=mem_id,
                    title=args.title,
                    description=args.description or "",
                    tags=args.tags,
                    created=today,
                    updated=today,
                    source="tool",
                ),
                body=args.body or "",
            )
            path = store.upsert(entry)
            yield ManageMemoryResult(
                action="add", id=mem_id, message=f"created {path.name}"
            )
            return

        # update
        if not args.id:
            raise ToolError("update requires 'id'")
        existing = store.get(args.id)
        if existing is None:
            raise ToolError(f"Memory '{args.id}' not found; use action=add.")
        meta = existing.metadata.model_copy(
            update={
                k: v
                for k, v in {
                    "title": args.title,
                    "description": args.description,
                    "tags": args.tags or None,
                }.items()
                if v is not None
            }
        )
        meta = meta.model_copy(update={"updated": today})
        body = args.body if args.body is not None else existing.body
        store.upsert(MemoryEntry(metadata=meta, body=body))
        yield ManageMemoryResult(action="update", id=args.id, message="updated")
