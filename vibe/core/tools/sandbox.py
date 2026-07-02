"""OS-level sandbox wrappers for the bash tool (opt-in, defense-in-depth).

These are pure helpers: backend detection (which sandbox binary is available),
env scrubbing, and argv construction for each platform wrapper. The bash tool
composes them at spawn time. The textual permission gate + safety judge run
first and are never relaxed by the sandbox — this only *adds* containment to
commands the upper layers already permitted.

Backends, by platform:
- Linux: bubblewrap (`bwrap`) preferred, else `unshare` (weaker fallback).
- macOS: `sandbox-exec` (seatbelt).
- Windows / none available: no sandbox (callers decide fail-open vs fail-closed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from vibe.core.logger import logger
from vibe.core.utils import is_windows

# Env vars allowed through when scrubbing (everything else — API keys, tokens,
# cloud creds — is dropped). LC_* is allowed by prefix below.
_ENV_ALLOWLIST = frozenset({
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "TERM",
    "SHELL",
    "TMPDIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "CI",
    "NONINTERACTIVE",
    "NO_TTY",
    "DEBIAN_FRONTEND",
    "GIT_PAGER",
    "PAGER",
    "LESS",
})

# Kept through the scrub for the HOST session's bash only (ssh/https push, gh
# CLI, commit signing); NOT for isolated subagents — that scrub is the boundary.
HOST_GIT_ENV_PASSTHROUGH = frozenset({
    "SSH_AUTH_SOCK",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GIT_SSH_COMMAND",
    "GNUPGHOME",
    "GPG_TTY",
    "XDG_RUNTIME_DIR",
    "DBUS_SESSION_BUS_ADDRESS",
})


@dataclass
class SandboxSpec:
    write_roots: list[Path]
    allow_network: bool = True
    env: dict[str, str] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)


def _bwrap_usable() -> bool:
    """Whether bwrap can actually create namespaces here, not just exist.

    Docker/CI often deny unprivileged user-namespace creation, so a present
    bwrap dies with a cryptic 'bwrap:' error on every invocation. Probe once
    with a trivial sandbox; the result is cached by _detect_auto_backend.
    """
    exe = shutil.which("bwrap")
    if exe is None:
        return False
    try:
        proc = subprocess.run(
            [
                exe,
                "--ro-bind",
                "/",
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--unshare-pid",
                "--",
                "true",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


@functools.lru_cache(maxsize=1)
def _detect_auto_backend() -> str:
    """Resolve the auto-detected sandbox backend. Process-stable, so cached.

    Tests that monkeypatch is_windows/shutil.which call _detect_auto_backend.cache_clear().
    """
    if is_windows():
        return "none"
    if sys.platform == "darwin":
        return "sandbox-exec" if shutil.which("sandbox-exec") else "none"
    # Linux / other POSIX. bwrap must be usable, not merely present: a present
    # but namespace-denied bwrap is treated exactly like a missing backend.
    if _bwrap_usable():
        return "bwrap"
    if shutil.which("unshare"):
        return "unshare"
    return "none"


def detect_backend(override: str = "auto") -> str:
    """Resolve the sandbox backend name, or 'none' when unavailable."""
    if override != "auto":
        return override
    return _detect_auto_backend()


# BUBBLEWRAP_INSTALL_NUDGE is surfaced to the user (UI toast / startup issue)
# when the sandbox is enabled with containment but only the `unshare` backend
# is available. The goal is to convert a silent limitation into a one-time
# actionable prompt: install bubblewrap for real filesystem confinement.
BUBBLEWRAP_INSTALL_NUDGE = (
    "Sandbox is enabled, but bubblewrap (bwrap) isn't installed — only the "
    "'unshare' fallback is available, which provides namespace isolation but "
    "NO filesystem write confinement: sandboxed commands can still read/write "
    "anywhere you can. Install bubblewrap for real containment:\n"
    "  Debian/Ubuntu: sudo apt install bubblewrap\n"
    "  Fedora/RHEL:   sudo dnf install bubblewrap\n"
    "  Arch:          sudo pacman -S bubblewrap\n"
    "  macOS:         brew install bubblewrap\n"
    "(or disable the sandbox / set backend='unshare' to silence this)"
)


def unshare_confinement_nudge(
    *, sandbox_enabled: bool, backend_override: str
) -> str | None:
    """Return the install-bubblewrap nudge when the sandbox is enabled but the
    resolved backend is `unshare`; else None.

    Fires regardless of explicit write_dirs/allow_network: with the sandbox
    default-on, even a plain config confines writes to cwd+scratchpad, and the
    unshare backend cannot enforce that — so an unshare-only host would otherwise
    run silently with weaker confinement than the user believes. Pure (no side
    effects) so callers can invoke it freely to decide whether to surface a prompt.
    """
    if not sandbox_enabled:
        return None
    if detect_backend(backend_override) != "unshare":
        return None
    return BUBBLEWRAP_INSTALL_NUDGE


def scrub_env(base: dict[str, str], passthrough: list[str]) -> dict[str, str]:
    """Keep only an allowlist of env vars (drops secrets), plus passthrough."""
    allowed = _ENV_ALLOWLIST | set(passthrough)
    return {k: v for k, v in base.items() if k in allowed or k.startswith("LC_")}


def _canonical_roots(roots: list[Path]) -> list[str]:
    # Skip roots that aren't existing dirs: bwrap --bind on a missing source
    # aborts the whole sandbox with "Can't find source path".
    seen: set[str] = set()
    for r in roots:
        try:
            resolved = Path(r).expanduser().resolve()
            if resolved.is_dir():
                seen.add(str(resolved))
        except (OSError, RuntimeError):
            continue
    return sorted(seen)


# Sensitive subpaths layered read-only over each writable root so a sandboxed
# command can read them but not rewrite agent config or secrets. Applied AFTER
# the writable --bind (bwrap is left-to-right, so the later mount wins) and as
# explicit seatbelt denies after the write-allow. NB: `.git` is deliberately NOT
# here — a coding agent must be able to `git commit` (see _git_bind_dirs, which
# keeps the repo's git metadata writable but re-protects `hooks/`).
_PROTECTED_SUBPATHS = (".vibe", ".env")


def _worktree_gitdir(root: Path) -> Path | None:
    """The external gitdir a linked-worktree ``.git`` file points to, else None.

    A normal checkout has a ``.git`` *directory* (returns None — it lives under
    the already-writable root). A linked worktree has a ``.git`` *file* holding
    ``gitdir: <abs>/.git/worktrees/<name>``; return that path resolved.
    """
    dotgit = root / ".git"
    try:
        if not dotgit.is_file():
            return None
        for line in dotgit.read_text().splitlines():
            if line.startswith("gitdir:"):
                p = Path(line.split(":", 1)[1].strip())
                return p if p.is_absolute() else (root / p).resolve()
    except OSError:
        return None
    return None


# Git metadata kept read-only over the writable gitdir: hooks/ blocks a planted
# hook; config/config.worktree block a core.hooksPath (or sibling) escape.
_GIT_READONLY_METADATA = ("hooks", "config", "config.worktree")


def _readonly_git_targets(base: Path) -> list[str]:
    """Existing git metadata under *base* that must stay read-only. Skips
    symlinks (a link could point outside and be mounted through).
    """
    found: list[str] = []
    for name in _GIT_READONLY_METADATA:
        target = base / name
        try:
            if target.exists() and not target.is_symlink():
                found.append(str(target))
        except OSError:
            continue
    return found


def _git_bind_dirs(root: Path) -> tuple[list[str], list[str]]:
    """(writable_git_dirs, readonly_git_metadata) for one write root.

    A sandboxed command must be able to commit (write index/refs/objects/logs),
    so git metadata stays writable — but ``hooks/`` and ``config`` are re-layered
    read-only so a command can't drop a hook or repoint ``core.hooksPath`` to run
    code *outside* the sandbox later. For a linked worktree the real gitdir and
    shared object store live outside the checkout under the main repo's ``.git``,
    so that dir is bound writable explicitly.
    """
    writable: list[str] = []
    readonly: list[str] = []
    gitdir = _worktree_gitdir(root)
    if gitdir is not None:
        # gitdir == <common>/worktrees/<name>; <common> holds objects + refs and
        # contains the gitdir, so binding it writable covers the whole commit.
        common = gitdir.parent.parent
        if common.is_dir():
            writable.append(str(common))
        # Shared hooks/config live under <common>; the per-worktree config.worktree
        # (the `git config --worktree` target) lives under the worktree's gitdir.
        readonly += _readonly_git_targets(common)
        readonly += _readonly_git_targets(gitdir)
        readonly += _sibling_worktree_readonly_targets(common, skip=gitdir)
    else:
        readonly += _readonly_git_targets(root / ".git")
        readonly += _sibling_worktree_readonly_targets(root / ".git", skip=None)
    return writable, readonly


def _sibling_worktree_readonly_targets(common: Path, skip: Path | None) -> list[str]:
    """Metadata and admin dir of OTHER worktrees under ``<common>/worktrees``.

    A writable sibling ``config.worktree`` is a cross-worktree hooksPath escape,
    and a writable sibling admin dir lets a sandboxed ``git worktree remove``
    delete another session's registration (the husk mechanism from the
    2026-07-02 incident). Re-layer the entire sibling dir read-only; the
    agent's own admin dir is excluded via *skip* and stays writable.
    """
    worktrees = common / "worktrees"
    found: list[str] = []
    try:
        entries = list(worktrees.iterdir()) if worktrees.is_dir() else []
    except OSError:
        return found
    for entry in entries:
        if entry == skip or entry.is_symlink() or not entry.is_dir():
            continue
        found.append(str(entry))
        found += _readonly_git_targets(entry)
    return found


def build_sandbox_command(
    spec: SandboxSpec, backend: str
) -> tuple[list[str] | None, str, Path | None]:
    """Return (argv_prefix, backend_name, profile_path).

    The caller appends ``<shell> -c <command>`` to ``argv_prefix``. ``profile_path``
    is a temp file to unlink after the run (seatbelt only), else None. Returns
    (None, 'none', None) when the backend cannot build a command.
    """
    if backend == "bwrap":
        return _bwrap_argv(spec), "bwrap", None
    if backend == "unshare":
        # The unshare backend provides PID/IPC/net namespace isolation but NO
        # filesystem write confinement: it does not remount / read-only or
        # bind-mount write_roots. If the spec asks for containment the user can
        # reasonably believe is enforced, warn loudly so they know it is not —
        # this is the common case on minimal containers/CI without bubblewrap,
        # i.e. exactly the hosts where people reach for sandboxing.
        if spec.write_roots or not spec.allow_network:
            logger.warning(
                "sandbox backend 'unshare' provides namespace isolation but NO "
                "filesystem write confinement or network enforcement: write_roots "
                "and allow_network are IGNORED. Commands can still read/write "
                "anywhere the running user can. Install bubblewrap (bwrap) for "
                "real containment, or accept this by setting backend='unshare' "
                "explicitly."
            )
        return _unshare_argv(spec), "unshare", None
    if backend == "sandbox-exec":
        argv, profile = _seatbelt_argv(spec)
        return argv, "sandbox-exec", profile
    return None, "none", None


def _protected_subpaths_for(root: str) -> list[str]:
    """Existing sensitive subpaths (``.git``/``.vibe``/``.env``) under *root*.

    Skips symlinks: a symlinked ``.git`` could point outside the root, and
    bind-mounting through it would mount the link target read-only rather than
    the metadata dir. Real dirs/files only.
    """
    found: list[str] = []
    base = Path(root)
    for name in _PROTECTED_SUBPATHS:
        candidate = base / name
        try:
            if candidate.exists() and not candidate.is_symlink():
                found.append(str(candidate))
        except OSError:
            continue
    return found


def _bwrap_argv(spec: SandboxSpec) -> list[str]:
    # bwrap applies operations left to right inside the new namespace, so the
    # read-only root bind MUST precede the pseudo-filesystem overlays. Placing
    # --ro-bind / / after --dev/--proc/--tmpfs layers the read-only root over
    # them and makes /tmp (etc.) read-only, breaking any command that writes to
    # /tmp (mktemp, pip, compilers, editors, sort, ...).
    argv = [
        "bwrap",
        "--die-with-parent",
        "--unshare-pid",
        "--unshare-uts",
        "--unshare-ipc",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
    ]
    if not spec.allow_network:
        argv.append("--unshare-net")
    for root in _canonical_roots(spec.write_roots):
        # Writable bind first, then the (possibly external) git metadata writable,
        # then layer read-only over sensitive metadata + git hooks last (bwrap is
        # left-to-right, so the later --ro-bind wins for that subpath).
        argv += ["--bind", root, root]
        writable_git, readonly_git = _git_bind_dirs(Path(root))
        for gitdir in writable_git:
            argv += ["--bind", gitdir, gitdir]
        for sub in _protected_subpaths_for(root):
            argv += ["--ro-bind", sub, sub]
        for meta in readonly_git:
            argv += ["--ro-bind", meta, meta]
    argv += ["--chdir", str(Path.cwd())]
    argv += spec.extra_args
    argv.append("--")
    return argv


def _unshare_argv(spec: SandboxSpec) -> list[str]:
    # Weaker fallback: namespace isolation without bind-mount confinement of /.
    argv = ["unshare", "--user", "--map-root-user", "--mount"]
    if not spec.allow_network:
        argv.append("--net")
    argv += spec.extra_args
    argv.append("--")
    return argv


def build_seatbelt_profile(spec: SandboxSpec) -> str:
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-exec)",
        "(allow process-fork)",
        "(allow sysctl-read)",
        "(allow file-read*)",
    ]
    for root in _canonical_roots(spec.write_roots):
        if '"' in root or "\n" in root:
            continue  # never inject into the profile string
        lines.append(f'(allow file-write* (subpath "{root}"))')
        writable_git, readonly_git = _git_bind_dirs(Path(root))
        for gitdir in writable_git:
            if '"' not in gitdir and "\n" not in gitdir:
                lines.append(f'(allow file-write* (subpath "{gitdir}"))')
        # Re-deny secrets + git hooks/config (last match wins): a command can
        # commit but not plant a hook or repoint core.hooksPath outside.
        for sub in [*_protected_subpaths_for(root), *readonly_git]:
            if '"' in sub or "\n" in sub:
                continue
            lines.append(f'(deny file-write* (subpath "{sub}"))')
    lines.append("(allow network*)" if spec.allow_network else "(deny network*)")
    return "\n".join(lines) + "\n"


def _seatbelt_argv(spec: SandboxSpec) -> tuple[list[str], Path]:
    profile = build_seatbelt_profile(spec)
    fd, path = tempfile.mkstemp(suffix=".sb", prefix="vibe-sandbox-")
    os.close(fd)
    Path(path).write_text(profile)
    return ["sandbox-exec", "-f", path], Path(path)
