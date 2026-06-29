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
                    entries[entry.id] = entry
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
        self._cache = None
        return path

    def remove_from_tier(self, memory_id: str, *, project: bool) -> bool:
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
        out: list[MemoryEntry] = []
        for e in self._entries().values():
            age = memory_age_days(e.metadata.updated, today)
            if age is not None and age >= min_age_days:
                out.append(e)
        return out

    def verification_candidates(
        self, *, min_age_days: int, today: datetime.date | None = None
    ) -> list[MemoryEntry]:
        out: list[MemoryEntry] = []
        for e in self._entries().values():
            stamp = e.metadata.last_verified
            age = memory_age_days(stamp if stamp else e.metadata.updated, today)
            if age is not None and age >= min_age_days:
                out.append(e)
        return out

    def last_verification(self) -> datetime.date | None:
        marker = self._user_dir / ".last_verification"
        if not marker.exists():
            return None
        from vibe.core.utils.io import read_safe

        try:
            return datetime.date.fromisoformat(read_safe(marker).text.strip())
        except (OSError, ValueError):
            return None

    def stamp_verification(self, today_iso: str) -> None:
        self._user_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write(self._user_dir / ".last_verification", today_iso)

    def apply_verification_result(
        self, memory_id: str, state: str, today_iso: str, *, reconcile_tags: bool = True
    ) -> bool:
        from vibe.core.memory.models import VerificationState

        entry = self.get(memory_id)
        if entry is None:
            return False
        m = entry.metadata
        tags = m.tags
        if reconcile_tags and state == "stale":
            from vibe.core.memory.models import _STALE_TAGS

            tags = [t for t in tags if t not in _STALE_TAGS]
        meta = m.model_copy(
            update={
                "last_verified": today_iso,
                "verification_state": VerificationState(state),
                "tags": tags,
            }
        )
        self.trash(memory_id, reason="verification-backup", into=memory_id)
        self.upsert(
            MemoryEntry(metadata=meta, body=entry.body), project=(m.scope == "project")
        )
        return True

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
        # the audit trail stays honest after a sweep. Unparsable lines are kept
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
                kept.append(s)  # keep unparsable audit lines verbatim
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
    return _namespace_for(_project_identity(root), create=create)


def project_memory_dir(*, create: bool = False) -> Path | None:
    try:
        from vibe.core.config.harness_files import get_harness_files_manager

        roots = get_harness_files_manager().project_roots
    except Exception:
        return None
    if not roots:
        return None
    return _namespace_for(_project_identity(roots[0]), create=create)
