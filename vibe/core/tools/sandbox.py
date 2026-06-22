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
import os
from pathlib import Path
import shutil
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


@dataclass
class SandboxSpec:
    write_roots: list[Path]
    allow_network: bool = True
    env: dict[str, str] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)


def detect_backend(override: str = "auto") -> str:
    """Resolve the sandbox backend name, or 'none' when unavailable."""
    if override != "auto":
        return override
    if is_windows():
        return "none"
    if sys.platform == "darwin":
        return "sandbox-exec" if shutil.which("sandbox-exec") else "none"
    # Linux / other POSIX
    if shutil.which("bwrap"):
        return "bwrap"
    if shutil.which("unshare"):
        return "unshare"
    return "none"


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
    *,
    sandbox_enabled: bool,
    backend_override: str,
    write_dirs: list[str],
    allow_network: bool,
) -> str | None:
    """Return the install-bubblewrap nudge message when the resolved backend is
    `unshare` and the user asked for containment it cannot enforce; else None.

    Pure (no side effects) so callers can invoke it freely to decide whether to
    surface a UI prompt. Mirrors the runtime WARNING in build_sandbox_command,
    but lifted to a one-time startup/first-use prompt instead of a per-command
    log line.
    """
    if not sandbox_enabled:
        return None
    if detect_backend(backend_override) != "unshare":
        return None
    if not write_dirs and allow_network:
        return None
    return BUBBLEWRAP_INSTALL_NUDGE


def scrub_env(base: dict[str, str], passthrough: list[str]) -> dict[str, str]:
    """Keep only an allowlist of env vars (drops secrets), plus passthrough."""
    allowed = _ENV_ALLOWLIST | set(passthrough)
    return {k: v for k, v in base.items() if k in allowed or k.startswith("LC_")}


def _canonical_roots(roots: list[Path]) -> list[str]:
    seen: set[str] = set()
    for r in roots:
        try:
            seen.add(str(Path(r).expanduser().resolve()))
        except (OSError, RuntimeError):
            continue
    return sorted(seen)


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
        argv += ["--bind", root, root]
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
    lines.append("(allow network*)" if spec.allow_network else "(deny network*)")
    return "\n".join(lines) + "\n"


def _seatbelt_argv(spec: SandboxSpec) -> tuple[list[str], Path]:
    profile = build_seatbelt_profile(spec)
    fd, path = tempfile.mkstemp(suffix=".sb", prefix="vibe-sandbox-")
    os.close(fd)
    Path(path).write_text(profile)
    return ["sandbox-exec", "-f", path], Path(path)
