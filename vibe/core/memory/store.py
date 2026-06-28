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
# which bypasses the pydantic model and interpolates the id into a path. MUST be
# used with .fullmatch() (not .match()): the shared ^...$ pattern still lets a
# trailing newline through under .match(), and an id like "slug\n" must not
# reach a filename interpolation. See models._SLUG for the engine constraint.
_ID_RE = re.compile(_SLUG)


class MemoryStore:
    def __init__(self, user_dir: Path, project_dirs: list[Path] | None = None) -> None:
        self._user_dir = user_dir
        self._project_dirs = project_dirs or []
        self._cache: dict[str, MemoryEntry] | None = None
        self._mtimes: dict[Path, float] = {}
        self.issues: list[str] = []

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

    def _sorted_entries(self) -> list[MemoryEntry]:
        return sorted(
            self._entries().values(), key=lambda e: e.metadata.updated, reverse=True
        )

    def index(self, limit: int = 200) -> list[str]:
        return [e.index_line() for e in self._sorted_entries()[:limit]]

    def index_markdown(self, limit: int = 200) -> str:
        # The selector consumes index() (clean list); this markdown form is the
        # always-on display the model reads. When the corpus exceeds the cap,
        # silently dropping the tail makes those memories recall-invisible with
        # no signal — surface a footer so the model knows to raise the cap or
        # grep the store rather than assume the index is exhaustive.
        entries = self._sorted_entries()
        shown = entries[:limit]
        lines = [e.index_line() for e in shown]
        hidden = len(entries) - len(shown)
        if hidden > 0:
            noun = "memory" if hidden == 1 else "memories"
            lines.append(
                f"... and {hidden} more {noun} not shown "
                "(raise memory.max_entries_scanned to surface them)"
            )
        return "\n".join(lines)

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
        if not _ID_RE.fullmatch(memory_id):
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
        if not _ID_RE.fullmatch(memory_id):
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

    def _effective_path(self, memory_id: str) -> Path | None:
        # The file that actually backs this id under shadowing: project dirs
        # override user (loaded last in _entries), so the effective file is the
        # first hit scanning project dirs first, then user. None if absent.
        if not _ID_RE.fullmatch(memory_id):
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
        a ledger entry. Never hard-deletes — the file stays recoverable (see
        ``restore``) and the ledger records why it moved. Returns False if the
        id has no effective file (nothing to trash).
        """
        src = self._effective_path(memory_id)
        if src is None:
            return False
        trash_dir = src.parent / ".trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        # Collision-safe suffix: trashing the same id twice within a second
        # (re-create + re-trash, or a concurrent pass) would otherwise make an
        # identical ``{id}-{ts}.md`` and ``replace`` would clobber the first
        # copy. A short random tail keeps both recoverable.
        suffix = os.urandom(3).hex()
        dest = trash_dir / f"{memory_id}-{ts}-{suffix}.md"
        try:
            src.replace(dest)  # atomic rename within one filesystem
        except OSError:
            # Cross-device mount of ~/.vibe: copy then unlink so the recoverable
            # trash copy still lands inside the trash tree.
            dest.write_bytes(src.read_bytes())
            src.unlink()
        self._append_ledger(
            trash_dir,
            {
                "id": memory_id,
                "ts": ts,
                "reason": reason,
                "into": into,
                "file": dest.name,
                "from": src.name,
            },
        )
        self._cache = None
        return True

    def restore(self, memory_id: str) -> Path | None:
        """Restore the most recently trashed copy of ``memory_id`` back to a
        live memory file, returning its path. The undo is itself ledger-audited.

        Recovery scans ``.trash/`` trees by filename, so it does NOT depend on
        the ledger being intact — a trashed file is recoverable even if its
        ledger line was lost to a crash. Returns None when no trashed copy
        exists, or when a live file already occupies the id: restore refuses to
        clobber (delete it first if you really want the old copy back).
        """
        if not _ID_RE.fullmatch(memory_id):
            return None
        # Refuse to clobber a live memory: restoring over it would silently
        # destroy the current one. The caller resolves the conflict explicitly.
        if self._effective_path(memory_id) is not None:
            return None
        for d in reversed(self._search_dirs()):
            trash_dir = d / ".trash"
            if not trash_dir.is_dir():
                continue
            copies = sorted(
                trash_dir.glob(f"{memory_id}-*.md"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not copies:
                continue
            src = copies[0]
            dest = d / f"{memory_id}.md"
            dest.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dest)
            self._append_ledger(
                trash_dir,
                {
                    "id": memory_id,
                    "ts": datetime.datetime.now().strftime("%Y%m%dT%H%M%S"),
                    "reason": "restore",
                    "into": None,
                    "file": src.name,
                    "from": dest.name,
                },
            )
            self._cache = None
            return dest
        return None

    def apply_merge(
        self,
        into_id: str,
        source_ids: list[str],
        merged_body: str,
        today: str,
        *,
        extra_tags: list[str] | None = None,
    ) -> int:
        """Rewrite ``into_id`` with a reconciled body and trash its sources.

        The target keeps its metadata (scope/type); only ``updated`` is bumped
        to ``today`` so the recency signal reflects the reconciliation, and
        ``source`` is marked ``auto``. Tags from the folded-in sources are
        unioned in via ``extra_tags`` so a merge does not silently lose a source
        tag (e.g. folding a [commits] memory into a [git] one). Returns the
        count of source files actually trashed. A source equal to the target is
        skipped (not trashed).

        The survivor's PRE-merge body is backed up to trash before the rewrite,
        so a merge is reversible end to end — not just its sources. Use
        ``restore(into_id)`` to recover the pre-merge state (after deleting the
        reconciled copy, since restore refuses to clobber a live file).
        """
        target = self.get(into_id)
        if target is None:
            return 0
        merged_tags = sorted(set(target.metadata.tags) | set(extra_tags or []))
        meta = target.metadata.model_copy(
            update={"updated": today, "source": "auto", "tags": merged_tags}
        )
        # Back up the surviving memory's current body before overwriting it: the
        # only consolidation mutation that was previously non-reversible.
        self.trash(into_id, reason="merge-backup", into=into_id)
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
        from vibe.core.utils.io import read_safe

        try:
            return datetime.date.fromisoformat(read_safe(marker).text.strip())
        except (OSError, ValueError):
            return None

    def stamp_consolidation(self, today_iso: str) -> None:
        # Atomic like every other state mutation: a plain open("w") could leave
        # a truncated marker on crash. The read side tolerates corruption
        # (returns None -> re-run), but a clean write keeps the throttle honest.
        self._user_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write(self._user_dir / ".last_consolidation", today_iso)

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

    @staticmethod
    def _append_ledger(trash_dir: Path, entry: dict[str, object]) -> None:
        # Crash-safe append: flush + fsync the JSON line so a crash can't leave a
        # partial trailing record (a half-written line would corrupt the JSONL).
        # The file move already happened by the time we get here, so a ledger
        # failure leaves a file in .trash/ with no ledger entry — still
        # recoverable via restore() (which scans by filename, not the ledger),
        # just not audited. We accept that: durability of the move beats the
        # ledger, and the ledger is best-effort audit.
        trash_dir.mkdir(parents=True, exist_ok=True)
        ledger = trash_dir / "ledger.jsonl"
        line = json.dumps(entry) + "\n"
        with ledger.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def sweep_trash(self, max_age_days: int) -> int:
        """Delete trash entries older than ``max_age_days`` and compact the ledger.

        Ages trashed files by the timestamp encoded in their filename (written by
        ``trash()``), not file mtime — ``os.replace`` preserves the source mtime,
        so mtime would report the memory's original write time rather than when
        it was trashed. Files whose filename timestamp is unparseable are LEFT
        (conservative: never delete what cannot be dated). After unlinking, the
        per-directory ledger is compacted to drop entries referencing removed
        files. ``max_age_days <= 0`` disables the sweep (no-op). Returns the
        count of files removed.
        """
        if max_age_days <= 0:
            return 0
        now = datetime.datetime.now()
        cutoff = now - datetime.timedelta(days=max_age_days)
        removed = 0
        for d in self._search_dirs():
            trash_dir = d / ".trash"
            if not trash_dir.is_dir():
                continue
            survivors: set[str] = set()
            for f in trash_dir.glob("*.md"):
                trashed_at = _parse_trash_ts(f.name)
                if trashed_at is None:
                    survivors.add(f.name)  # undated — leave it
                    continue
                if trashed_at < cutoff:
                    with contextlib.suppress(OSError):
                        f.unlink()
                    removed += 1
                else:
                    survivors.add(f.name)
            self._compact_ledger(trash_dir, survivors)
        if removed:
            self._cache = None
        return removed

    @staticmethod
    def _compact_ledger(trash_dir: Path, survivors: set[str]) -> None:
        # Drop ledger lines whose "file" no longer exists in this trash dir so
        # the audit trail stays honest after a sweep. Unparseable lines are kept
        # (never silently drop audit data); a missing or unreadable ledger is a
        # no-op. Atomic rewrite so a crash can't truncate the ledger.
        ledger = trash_dir / "ledger.jsonl"
        if not ledger.exists():
            return
        from vibe.core.utils.io import read_safe

        try:
            text = read_safe(ledger).text
        except OSError:
            return
        kept: list[str] = []
        dropped_any = False
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                entry = json.loads(s)
            except (json.JSONDecodeError, ValueError):
                kept.append(s)  # keep unparseable audit lines verbatim
                continue
            fname = entry.get("file") if isinstance(entry, dict) else None
            if isinstance(fname, str) and fname not in survivors:
                dropped_any = True
                continue
            kept.append(s)
        if dropped_any:
            content = ("\n".join(kept) + "\n") if kept else ""
            MemoryStore._atomic_write(ledger, content)


# Trash filename timestamp format (see MemoryStore.trash): the trashed copy is
# named "{id}-{YYYYMMDDTHHMMSS}-{hex}.md", and sweep_trash ages files by THIS ts
# rather than file mtime — os.replace preserves the source mtime (the memory's
# original write time), so mtime would misreport the trash time as much older.
_TRASH_TS_FORMAT = "%Y%m%dT%H%M%S"
# trash() names files "{id}-{ts}-{hex}" — exactly three tail parts after
# rsplit("-", 2). Guards _parse_trash_ts before it indexes into the parts.
_TRASH_NAME_PARTS = 3


def _parse_trash_ts(filename: str) -> datetime.datetime | None:
    # Filename shape from trash(): "{id}-{YYYYMMDDTHHMMSS}-{hex}.md". The id is a
    # slug that may itself contain hyphens, so split from the RIGHT: the trailing
    # hex suffix, then the 15-char timestamp, with everything before being the id.
    stem = filename[:-3] if filename.endswith(".md") else filename
    parts = stem.rsplit("-", 2)
    if len(parts) != _TRASH_NAME_PARTS:
        return None
    try:
        return datetime.datetime.strptime(parts[1], _TRASH_TS_FORMAT)
    except ValueError:
        return None


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


def _namespace_for(identity: Path, *, create: bool = False) -> Path:
    """Memory namespace dir under ``~/.vibe`` for a resolved project identity.

    Shared by the harness-rooted resolver (:func:`project_memory_dir`) and the
    explicit-root resolver (:func:`project_memory_dir_for`). ``create`` also
    stamps a ``.origin`` file so an opaque hash dir stays debuggable.
    """
    digest = hashlib.sha256(str(identity).encode("utf-8")).hexdigest()[:16]
    ns = VIBE_HOME.path / "memory" / "projects" / digest
    if create:
        from vibe.core.utils.io import write_safe

        ns.mkdir(parents=True, exist_ok=True)
        origin = ns / ".origin"
        if not origin.exists():
            with contextlib.suppress(OSError):
                write_safe(origin, f"{identity}\n")
    return ns


def project_memory_dir_for(root: Path, *, create: bool = False) -> Path:
    """Memory namespace for an *explicit* project root, not the running one.

    Lets a caller target another repo's namespace (e.g. leaving a resume-memory
    for a project the agent is not currently running in). The identity is the
    same ``_project_identity`` the harness uses, so this resolves to the SAME
    namespace that an agent running inside ``root`` would see.
    """
    return _namespace_for(_project_identity(root), create=create)


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
    return _namespace_for(_project_identity(roots[0]), create=create)
