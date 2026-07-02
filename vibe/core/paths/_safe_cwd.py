from __future__ import annotations

import os
from pathlib import Path

_last_good_cwd: Path | None = None


def safe_cwd() -> Path:
    """``Path.cwd()`` that survives the directory being deleted out from under
    the process (worktree reap, tmpdir cleanup, concurrent-session GC).

    Falls back to the last successful lookup, then a still-existing ``$PWD``,
    then home — callers get a usable anchor instead of ENOENT in paths that
    must never crash (hook context, session attribution, telemetry).
    """
    global _last_good_cwd
    try:
        cwd = Path.cwd()
    except OSError:
        if _last_good_cwd is not None and _last_good_cwd.is_dir():
            return _last_good_cwd
        env_pwd = os.environ.get("PWD")
        if env_pwd and Path(env_pwd).is_dir():
            return Path(env_pwd)
        return Path.home()
    _last_good_cwd = cwd
    return cwd
