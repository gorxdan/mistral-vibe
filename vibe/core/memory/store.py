"""File-based memory store: human-editable markdown with YAML frontmatter.

One file per memory under a memory dir (user: ``~/.vibe/memory/*.md``; project:
``<root>/.vibe/memory/*.md``). No embeddings, no DB — discovery is a file scan,
relevance is an LLM header-scan (see selector.py). Project memories shadow user
memories by id. Writes are atomic (temp + os.replace) so concurrent agents on a
shared tree never tear a sibling file.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from pydantic import ValidationError
import yaml

from vibe.core.memory.models import (
    _SLUG,
    MemoryEntry,
    MemoryMetadata,
    freshness_note,
    memory_age_days,
)
from vibe.core.paths import VIBE_HOME
from vibe.core.skills.parser import SkillParseError, parse_skill_markdown

# Compiled slug pattern (same source as MemoryMetadata.id) for the delete path,
# which bypasses the pydantic model and interpolates the id into a path.
_ID_RE = re.compile(_SLUG)


class MemoryStore:
    def __init__(self, user_dir: Path, project_dirs: list[Path] | None = None) -> None:
        self._user_dir = user_dir
        self._project_dirs = project_dirs or []
        self._cache: dict[str, MemoryEntry] | None = None
        self._mtimes: dict[Path, float] = {}
        self.issues: list[str] = []

    # --- discovery / read -------------------------------------------------- #

    def _search_dirs(self) -> list[Path]:
        # User first so project entries (loaded after) shadow by id.
        return [self._user_dir, *self._project_dirs]

    def _current_mtimes(self) -> dict[Path, float]:
        out: dict[Path, float] = {}
        for d in self._search_dirs():
            if not d.is_dir():
                continue
            for f in d.glob("*.md"):
                try:
                    out[f] = f.stat().st_mtime
                except OSError:
                    continue
        return out

    def _entries(self) -> dict[str, MemoryEntry]:
        mtimes = self._current_mtimes()
        if self._cache is not None and mtimes == self._mtimes:
            return self._cache
        entries: dict[str, MemoryEntry] = {}
        self.issues = []
        # Iterate in _search_dirs order (user first, then project dirs): the
        # project entry must shadow a same-id user entry. Sorting by path (the
        # old behavior) tied precedence to directory names alphabetically.
        for d in self._search_dirs():
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.md")):
                if f not in mtimes:  # vanished between scans; skip
                    continue
                entry = self._load_file(f)
                if entry is not None:
                    entries[entry.id] = entry  # later dirs (project) shadow user
        self._cache = entries
        self._mtimes = mtimes
        return entries

    def _load_file(self, path: Path) -> MemoryEntry | None:
        from vibe.core.utils.io import read_safe

        try:
            content = read_safe(path).text
        except OSError as e:
            self.issues.append(f"{path.name}: unreadable ({e})")
            return None
        try:
            frontmatter, body = parse_skill_markdown(content)
        except SkillParseError as e:
            self.issues.append(f"{path.name}: {e}")
            return None
        frontmatter.setdefault("id", path.stem)
        frontmatter.setdefault("title", frontmatter["id"])
        try:
            meta = MemoryMetadata.model_validate(frontmatter)
        except ValidationError as e:
            self.issues.append(f"{path.name}: invalid frontmatter ({e})")
            return None
        return MemoryEntry(metadata=meta, body=body.strip())

    def index(self, limit: int = 200) -> list[str]:
        entries = sorted(
            self._entries().values(), key=lambda e: e.metadata.updated, reverse=True
        )
        return [e.index_line() for e in entries[:limit]]

    def index_markdown(self, limit: int = 200) -> str:
        return "\n".join(self.index(limit))

    def ids(self) -> list[str]:
        return list(self._entries().keys())

    def get(self, memory_id: str) -> MemoryEntry | None:
        return self._entries().get(memory_id)

    def bodies(self, ids: list[str], max_chars: int) -> str:
        """Concatenate selected bodies in given order, capped at max_chars
        (whole-entry drop — never a partial body).
        """
        blocks: list[str] = []
        used = 0
        for mid in ids:
            entry = self.get(mid)
            if entry is None:
                continue
            note = freshness_note(entry.metadata.updated)
            body_text = f"{note}\n{entry.body}" if note else entry.body
            block = f"### {entry.metadata.title}\n{body_text}"
            if used + len(block) > max_chars:
                continue
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    # --- write (for the manage_memory tool) -------------------------------- #

    def upsert(self, entry: MemoryEntry, *, project: bool = False) -> Path:
        target_dir = (
            self._project_dirs[0] if project and self._project_dirs else self._user_dir
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{entry.id}.md"
        fm = entry.metadata.model_dump(mode="json")
        doc = f"---\n{yaml.safe_dump(fm, sort_keys=False)}---\n{entry.body}\n"
        self._atomic_write(path, doc)
        self._cache = None  # invalidate
        return path

    def remove_from_tier(self, memory_id: str, *, project: bool) -> bool:
        """Remove the memory from ONE tier only (user xor project).

        Used on tier change so the old tier's file does not shadow the new one
        (project shadows user by id; without this, a project->user re-scope is
        invisible because the stale project file keeps winning).
        """
        if not _ID_RE.match(memory_id):
            return False
        target = (
            self._project_dirs[0] if project and self._project_dirs else self._user_dir
        )
        path = target / f"{memory_id}.md"
        if path.exists():
            path.unlink()
            self._cache = None
            return True
        return False

    def delete(self, memory_id: str) -> bool:
        # Validate against the slug pattern before interpolating into a path:
        # the add/update paths enforce this via MemoryMetadata, but delete()
        # built `{memory_id}.md` directly, so an id like "../../x" could unlink
        # a .md file outside the memory dir.
        if not _ID_RE.match(memory_id):
            return False
        # Clear every tier: a project memory shadows a same-id user one, so a
        # first-match early-return would leave the shadowed id still visible.
        removed = False
        for d in self._search_dirs():
            path = d / f"{memory_id}.md"
            if path.exists():
                path.unlink()
                removed = True
        if removed:
            self._cache = None
        return removed

    # --- consolidation support (reversible trash + ledger) ------------- #

    def _effective_path(self, memory_id: str) -> Path | None:
        # The file that actually backs this id under shadowing: project dirs
        # override user (loaded last in _entries), so the effective file is the
        # first hit scanning project dirs first, then user. None if absent.
        if not _ID_RE.match(memory_id):
            return None
        for d in reversed(self._search_dirs()):
            if not d.is_dir():
                continue
            path = d / f"{memory_id}.md"
            if path.exists():
                return path
        return None

    def trash(self, memory_id: str, *, reason: str, into: str | None = None) -> bool:
        """Move a memory's effective file into a per-dir ``.trash/`` tree with
        a ledger entry. Never hard-deletes — the file stays recoverable and the
        ledger records why it moved (consolidation merge/delete). Returns False
        if the id has no effective file (nothing to trash).
        """
        src = self._effective_path(memory_id)
        if src is None:
            return False
        trash_dir = src.parent / ".trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        dest = trash_dir / f"{memory_id}-{ts}.md"
        try:
            src.replace(dest)  # atomic rename within one filesystem
        except OSError:
            # Cross-device mount of ~/.vibe: copy then unlink so the recoverable
            # trash copy still lands inside the trash tree.
            dest.write_bytes(src.read_bytes())
            src.unlink()
        entry = {
            "id": memory_id,
            "ts": ts,
            "reason": reason,
            "into": into,
            "file": dest.name,
            "from": src.name,
        }
        with (trash_dir / "ledger.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self._cache = None
        return True

    def apply_merge(
        self, into_id: str, source_ids: list[str], merged_body: str, today: str
    ) -> int:
        """Rewrite ``into_id`` with a reconciled body and trash its sources.

        The target keeps its metadata (scope/type/tags); only ``updated`` is
        bumped to ``today`` so the recency signal reflects the reconciliation,
        and ``source`` is marked ``auto``. Returns the count of source files
        actually trashed. A source equal to the target is skipped (not trashed).
        """
        target = self.get(into_id)
        if target is None:
            return 0
        meta = target.metadata.model_copy(update={"updated": today, "source": "auto"})
        self.upsert(
            MemoryEntry(metadata=meta, body=merged_body),
            project=(target.metadata.scope == "project"),
        )
        trashed = 0
        for sid in source_ids:
            if sid == into_id:
                continue
            if self.trash(sid, reason="merge", into=into_id):
                trashed += 1
        return trashed

    def consolidation_candidates(
        self, *, min_age_days: int, today: datetime.date | None = None
    ) -> list[MemoryEntry]:
        """Effective entries older than ``min_age_days`` — the consolidation
        set. Fresh memories and undated entries (unknown age) are excluded: a
        consolidate pass should never merge away something just learned.
        """
        out: list[MemoryEntry] = []
        for e in self._entries().values():
            age = memory_age_days(e.metadata.updated, today)
            if age is not None and age >= min_age_days:
                out.append(e)
        return out

    def last_consolidation(self) -> datetime.date | None:
        # Throttle marker for consolidation runs. Stored under the user dir so
        # one throttle covers the whole corpus (user + active project tier).
        marker = self._user_dir / ".last_consolidation"
        if not marker.exists():
            return None
        try:
            return datetime.date.fromisoformat(marker.read_text().strip())
        except (OSError, ValueError):
            return None

    def stamp_consolidation(self, today_iso: str) -> None:
        self._user_dir.mkdir(parents=True, exist_ok=True)
        with (self._user_dir / ".last_consolidation").open("w", encoding="utf-8") as f:
            f.write(today_iso)

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, path)
        except OSError:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise


def _project_identity(workdir: Path) -> Path:
    """Stable per-project identity for *workdir*.

    Git's common dir is shared by the main worktree and every linked worktree
    of one repo, so sessions across all of them collapse to one memory
    namespace (the multi-agent/multi-worktree case). Older git returns it
    relative, so resolve against workdir. Falls back to the workdir itself
    outside git (per-path isolation).
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(workdir), "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return workdir.resolve()
    common = Path(out.stdout.strip())
    common = common.resolve() if common.is_absolute() else (workdir / common).resolve()
    return common if common.exists() else workdir.resolve()


def project_memory_dir(*, create: bool = False) -> Path | None:
    """Current project's memory namespace under ``~/.vibe``, or ``None``.

    Per-project memories live UNDER ``~/.vibe`` (never in the repo) so they
    can't be committed — this is why project memory is on by default. The
    namespace is keyed by the project identity (see ``_project_identity``): all
    sessions and worktrees of one git repo share it; different repos (and
    non-git dirs) stay isolated. Returns ``None`` when there's no trusted
    project (caller falls back to global only).
    """
    try:
        from vibe.core.config.harness_files import get_harness_files_manager

        roots = get_harness_files_manager().project_roots
    except Exception:
        return None
    if not roots:
        return None
    identity = _project_identity(roots[0])
    digest = hashlib.sha256(str(identity).encode("utf-8")).hexdigest()[:16]
    ns = VIBE_HOME.path / "memory" / "projects" / digest
    if create:
        ns.mkdir(parents=True, exist_ok=True)
        # Stamp the identity so an opaque hash dir stays debuggable.
        origin = ns / ".origin"
        if not origin.exists():
            with contextlib.suppress(OSError):
                origin.write_text(f"{identity}\n", encoding="utf-8")
    return ns
