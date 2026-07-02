from __future__ import annotations

from collections.abc import AsyncGenerator
import datetime as _dt
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vibe.core.config import VibeConfig
from vibe.core.memory.models import (
    _DESC_MAX,
    MemoryEntry,
    MemoryMetadata,
    MemoryType,
    slugify,
)
from vibe.core.memory.store import (
    MemoryStore,
    project_memory_dir,
    project_memory_dir_for,
)
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


def _memory_store(project_dir: Path | None = None) -> MemoryStore:
    # An explicit project_dir (from project_path) overrides the running
    # project's namespace so the store reads/writes the TARGET repo's tier.
    active = project_dir if project_dir is not None else project_memory_dir()
    project_dirs = [active] if active is not None else []
    return MemoryStore(user_dir=VIBE_HOME.path / "memory", project_dirs=project_dirs)


def _ensure_project_namespace(project_root: Path | None) -> Path | None:
    # Materialize (mkdir + .origin stamp) the resolved project namespace.
    # project_root=None -> the running project; else the explicit target.
    if project_root is not None:
        return project_memory_dir_for(project_root, create=True)
    return project_memory_dir(create=True)


def _resolve_target_namespace(
    args: ManageMemoryArgs,
) -> tuple[Path | None, Path | None]:
    """Resolve ``(project_dir, project_root)`` for this invocation.

    ``project_path`` targets another repo's namespace (cross-project memory);
    otherwise the running project's namespace (or ``(None, None)`` outside a
    trusted project). ``project_root`` is the explicit root to materialize on
    write, or None for the running project.
    """
    if args.project_path is None:
        return project_memory_dir(), None
    root = Path(args.project_path).expanduser()
    if not root.exists():
        raise ToolError(f"project_path does not exist: {root}")
    return project_memory_dir_for(root), root


def _derive_title(body: str | None) -> str:
    """Derive a short title from a memory body when the caller omits one.

    Takes the first non-empty line, strips markdown heading/list markers, and
    caps the length. Returns "" when nothing usable is present.
    """
    if not body:
        return ""
    for raw in body.splitlines():
        line = raw.strip().lstrip("#->*").strip()
        if line:
            return line[:60]
    return ""


def _default_add_scope(
    requested: Literal["user", "project"] | None,
    mem_type: MemoryType | None,
    project_dir: Path | None,
) -> Literal["user", "project"]:
    """Resolve the scope for a new memory when the caller omits one.

    Mirrors the auto-extractor's type routing (see ``_extract_memories`` in
    agent_loop): project/reference-typed memories are project-scoped, user/
    feedback are global. An explicit scope always wins. An untyped save
    defaults to the active project namespace — a fact captured while working in
    a project is usually about that project, and a global default leaks it into
    every other project's context.
    """
    if requested is not None:
        return requested
    if mem_type in {MemoryType.USER, MemoryType.FEEDBACK}:
        return "user"
    if project_dir is not None:
        return "project"
    return "user"


class ManageMemoryArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    action: Literal["add", "update", "list", "delete"]
    id: str | None = None
    title: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    body: str | None = None
    type: MemoryType | None = None
    # add: defaults to the active project namespace, else user (see
    # _default_add_scope). update: None preserves the existing tier.
    scope: Literal["user", "project"] | None = None
    # Target a DIFFERENT project's namespace than the running one (cross-project
    # memory, e.g. leaving a resume-memory for a repo the agent isn't in).
    # Applies to every action: add/update write there, list/delete operate on
    # that namespace. Resolved via the same identity hash the harness uses, so
    # it matches what an agent running inside project_path would see.
    project_path: str | None = None

    @field_validator("description", mode="before")
    @classmethod
    def _clamp_description(cls, v: object) -> object:
        # Truncate at input so the update path's model_copy(update=...) — which
        # bypasses MemoryMetadata validation — also stays within the limit, not
        # just the add path that constructs MemoryMetadata directly.
        if isinstance(v, str) and len(v) > _DESC_MAX:
            return v[:_DESC_MAX]
        return v


class ManageMemoryResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    action: str
    id: str | None = None
    message: str
    entries: list[str] = Field(default_factory=list)


class ManageMemoryConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


class ManageMemory(
    BaseTool[ManageMemoryArgs, ManageMemoryResult, ManageMemoryConfig, BaseToolState]
):
    manifest_deferrable: ClassVar[bool] = True
    description: ClassVar[str] = (
        "Manage durable cross-session memories (markdown files under ~/.vibe/memory). "
        "Actions: add (new memory), update (patch existing), list, delete. Memories are "
        "relevance-selected into context in later sessions. add defaults to the current "
        "project's private namespace (~/.vibe/memory/projects/<hash>, never committed, "
        "isolated per trusted project path) when one is active, so project-specific "
        "facts don't leak into other projects; pass scope='user' to write a global "
        "memory shared across all projects, reserved for cross-project identity, "
        "preferences, and feedback. Project memories shadow same-id global ones for "
        "that project only. Prefer a 'type' (user, feedback, "
        "project, reference): user = who the user is; feedback = how they want you to "
        "work (with the why); project = ongoing work/decisions not in code/git; "
        "reference = pointers to external systems. Do not save code patterns, "
        "architecture, git history, or fix recipes — those are derivable. "
        "project_path (add/update) targets a DIFFERENT repo's project namespace "
        "than the one you're running in — use it to leave a resume-memory for a "
        "project you are not currently inside."
    )

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        return config is not None and config.memory.enabled

    async def run(
        self, args: ManageMemoryArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | ManageMemoryResult, None]:
        # Resolve the effective project namespace: an explicit project_path
        # targets another repo's namespace (cross-project memory); otherwise the
        # running project's namespace (or None outside a trusted project).
        project_dir, project_root = _resolve_target_namespace(args)
        store = _memory_store(project_dir)
        today = _dt.date.today().isoformat()

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
            # `title` is the only hard requirement for add, but the schema marks
            # it optional, so models often omit it and hit a dead-end error.
            # Derive one from the body's first line when missing rather than
            # failing the call.
            title = args.title or _derive_title(args.body)
            if not title:
                raise ToolError("add requires 'title' (or a non-empty 'body')")
            scope = _default_add_scope(args.scope, args.type, project_dir)
            if scope == "project" and project_dir is None:
                raise ToolError(
                    "scope=project requires a trusted project directory; "
                    "run from a trusted folder or use scope=user."
                )
            mem_id = args.id or slugify(title)
            if store.get(mem_id) is not None:
                raise ToolError(
                    f"Memory '{mem_id}' already exists; use action=update instead."
                )
            entry = MemoryEntry(
                metadata=MemoryMetadata(
                    id=mem_id,
                    title=title,
                    description=args.description or "",
                    tags=args.tags,
                    type=args.type,
                    scope=scope,
                    created=today,
                    updated=today,
                    source="tool",
                    session_id=(ctx.session_id if ctx is not None else "") or "",
                ),
                body=args.body or "",
            )
            if scope == "project":
                _ensure_project_namespace(project_root)
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
            _ensure_project_namespace(project_root)
        store.upsert(
            MemoryEntry(metadata=meta, body=body), project=(scope == "project")
        )
        yield ManageMemoryResult(
            action="update", id=args.id, message=f"updated ({scope})"
        )
