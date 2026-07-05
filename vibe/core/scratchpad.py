from __future__ import annotations

import atexit
from pathlib import Path
import shutil
import tempfile
import time

from vibe.core.logger import logger
from vibe.core.session.session_id import shorten_session_id

SCRATCHPAD_PREFIX = "vibe-scratchpad-"

_active_scratchpads: dict[str, Path] = {}
_atexit_registered = False
# Stranded scratchpads (process crashed/SIGKILLed before atexit) are reclaimed by
# age. Generous so a long-running session never loses its own working dir.
_SCRATCHPAD_GC_AGE_S = 24 * 3600


def init_scratchpad(session_id: str) -> Path | None:
    """Create a session-scoped scratchpad directory.

    Each session gets its own scratchpad. Idempotent per session_id.
    """
    if session_id in _active_scratchpads:
        return _active_scratchpads[session_id]

    global _atexit_registered
    if not _atexit_registered:
        atexit.register(cleanup_all_scratchpads)
        # Once per process: reap crash-stranded scratchpads (and the bash
        # bg-logs / sandbox profiles parked inside them) from dead sessions.
        gc_stale_scratchpads()
        _atexit_registered = True

    try:
        dir_path = Path(
            tempfile.mkdtemp(
                prefix=f"vibe-scratchpad-{shorten_session_id(session_id)}-"
            )
        )
        _active_scratchpads[session_id] = dir_path
        logger.debug("Scratchpad initialized at %s", dir_path)
        return dir_path
    except OSError:
        logger.warning("Failed to create scratchpad directory")
        return None


def cleanup_all_scratchpads() -> None:
    """Remove this process's scratchpad dirs. Registered atexit so a clean exit
    leaves nothing behind (covers the bash bg-logs / sandbox profiles inside).
    """
    for dir_path in list(_active_scratchpads.values()):
        shutil.rmtree(dir_path, ignore_errors=True)
    _active_scratchpads.clear()


def gc_stale_scratchpads(max_age_s: int = _SCRATCHPAD_GC_AGE_S) -> None:
    """Reclaim ``vibe-scratchpad-*`` dirs left by dead sessions, by age.

    Skips this process's active dirs and anything written to recently, so an
    in-flight session (its own or another process's) is never pulled out. Best
    effort — never raises.
    """
    try:
        active = {p.resolve() for p in _active_scratchpads.values()}
        cutoff = time.time() - max_age_s
        for d in Path(tempfile.gettempdir()).glob(f"{SCRATCHPAD_PREFIX}*"):
            try:
                if not d.is_dir() or d.resolve() in active:
                    continue
                if d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass
    except Exception:
        logger.debug("scratchpad gc skipped", exc_info=True)


def get_scratchpad_dir(session_id: str) -> Path | None:
    return _active_scratchpads.get(session_id)


def is_scratchpad_path(path_str: str) -> bool:
    """Return True if the resolved path is inside any active scratchpad.

    Uses Path.resolve() to defeat path traversal and symlink attacks.
    """
    if not _active_scratchpads:
        return False
    try:
        resolved = Path(path_str).expanduser().resolve()
        return any(
            _is_subpath(resolved, sp_dir.resolve())
            for sp_dir in _active_scratchpads.values()
        )
    except (ValueError, OSError):
        return False


def is_foreign_scratchpad_path(path_str: str) -> bool:
    """Return True if the path is inside a scratchpad-shaped dir that is NOT
    this process's active scratchpad.

    Companion to ``is_scratchpad_path``: that returns True only for THIS
    process's scratchpads; this returns True for scratchpad-shaped paths
    belonging to other concurrent vibe processes, which share the global
    ``/tmp`` namespace. A file tool targeting such a path is almost always
    a dropped ``task_id`` (the caller re-derived a verifier-log path by
    search instead of carrying the ``asub-N`` id) — surfacing it logs the
    real mechanism rather than leaving it indistinguishable from any other
    outside-workdir read.
    """
    try:
        resolved = Path(path_str).expanduser().resolve()
    except (ValueError, OSError):
        return False
    active = {sp.resolve() for sp in _active_scratchpads.values()}
    return any(
        p.name.startswith(SCRATCHPAD_PREFIX) and p not in active
        for p in (resolved, *resolved.parents)
    )


def _is_subpath(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
