from __future__ import annotations

from collections.abc import AsyncGenerator
import datetime as _dt
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from vibe.core.config import VibeConfig
from vibe.core.memory.models import MemoryEntry, MemoryMetadata, MemoryType, slugify
from vibe.core.memory.store import MemoryStore, project_memory_dir
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


def _memory_store() -> MemoryStore:
    project_dirs = [d] if (d := project_memory_dir()) else []
    return MemoryStore(user_dir=VIBE_HOME.path / "memory", project_dirs=project_dirs)


class ManageMemoryArgs(BaseModel):
    action: Literal["add", "update", "list", "delete"]
    id: str | None = None
    title: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    body: str | None = None
    type: MemoryType | None = None
    # add: defaults to "user". update: None preserves the existing tier.
    scope: Literal["user", "project"] | None = None


class ManageMemoryResult(BaseModel):
    action: str
    id: str | None = None
    message: str
    entries: list[str] = Field(default_factory=list)


class ManageMemoryConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


class ManageMemory(
    BaseTool[ManageMemoryArgs, ManageMemoryResult, ManageMemoryConfig, BaseToolState]
):
    description: ClassVar[str] = (
        "Manage durable cross-session memories (markdown files under ~/.vibe/memory). "
        "Actions: add (new memory), update (patch existing), list, delete. Memories are "
        "relevance-selected into context in later sessions. By default memories are "
        "global (shared across all projects); pass scope='project' to write to the "
        "current project's private namespace (~/.vibe/memory/projects/<hash>, never "
        "committed, isolated per trusted project path). Project memories shadow "
        "same-id global ones for that project only. Prefer a 'type' (user, feedback, "
        "project, reference): user = who the user is; feedback = how they want you to "
        "work (with the why); project = ongoing work/decisions not in code/git; "
        "reference = pointers to external systems. Do not save code patterns, "
        "architecture, git history, or fix recipes — those are derivable."
    )

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        return config is not None and config.memory.enabled

    async def run(
        self, args: ManageMemoryArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | ManageMemoryResult, None]:
        store = _memory_store()
        today = _dt.date.today().isoformat()
        project_dir = project_memory_dir()

        if args.action == "list":
            index = store.index()
            note = (
                f" (project namespace {project_dir.name} active)"
                if project_dir is not None
                else ""
            )
            yield ManageMemoryResult(
                action="list", message=f"{len(index)} memories{note}", entries=index
            )
            return

        if args.action == "delete":
            if not args.id:
                raise ToolError("delete requires 'id'")
            ok = store.delete(args.id)
            yield ManageMemoryResult(
                action="delete", id=args.id, message="deleted" if ok else "not found"
            )
            return

        if args.action == "add":
            if not args.title:
                raise ToolError("add requires 'title'")
            scope = args.scope or "user"
            if scope == "project" and project_dir is None:
                raise ToolError(
                    "scope=project requires a trusted project directory; "
                    "run from a trusted folder or use scope=user."
                )
            mem_id = args.id or slugify(args.title)
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
                    type=args.type,
                    scope=scope,
                    created=today,
                    updated=today,
                    source="tool",
                ),
                body=args.body or "",
            )
            if scope == "project":
                project_memory_dir(create=True)
            path = store.upsert(entry, project=(scope == "project"))
            yield ManageMemoryResult(
                action="add", id=mem_id, message=f"created {path.name} ({scope})"
            )
            return

        # update
        if not args.id:
            raise ToolError("update requires 'id'")
        existing = store.get(args.id)
        if existing is None:
            raise ToolError(f"Memory '{args.id}' not found; use action=add.")
        # Omitting scope preserves the existing tier (no silent re-tier on edit).
        scope = args.scope if args.scope is not None else existing.metadata.scope
        if scope == "project" and project_dir is None:
            raise ToolError(
                "scope=project requires a trusted project directory; "
                "run from a trusted folder or use scope=user."
            )
        meta = existing.metadata.model_copy(
            update={
                k: v
                for k, v in {
                    "title": args.title,
                    "description": args.description,
                    "tags": args.tags or None,
                    "type": args.type,
                }.items()
                if v is not None
            }
        )
        meta = meta.model_copy(update={"updated": today, "scope": scope})
        body = args.body if args.body is not None else existing.body
        # Tier change: unlink the old tier's file so it can't shadow the new one
        # (project shadows user by id; a stale project file would win over a
        # freshly-written user file and make the re-scope invisible).
        if scope != existing.metadata.scope:
            store.remove_from_tier(
                args.id, project=(existing.metadata.scope == "project")
            )
        if scope == "project":
            project_memory_dir(create=True)
        store.upsert(
            MemoryEntry(metadata=meta, body=body), project=(scope == "project")
        )
        yield ManageMemoryResult(
            action="update", id=args.id, message=f"updated ({scope})"
        )
