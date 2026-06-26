"""Argument-aware command safety for the bash tool.

Two independent checks, both pure (no I/O, no config):

* :func:`destructive_command_reason` flags inherently destructive invocations
  (``rm -rf``, ``chmod 777``, writing a block device, formatting a filesystem)
  regardless of the allowlist. It unwraps ``sudo``/``env``/``nohup``/``time``
  wrappers so ``sudo rm -rf /`` is caught the same as ``rm -rf /``. A destructive
  command is never auto-approved (it forces ASK) and carries a danger-specific
  reason so the user/LLM understands the escalation.

* :func:`allowlisted_argument_is_unsafe` is the argument-aware gate for commands
  that already match an allowlist prefix: a binary that is safe in one form can
  be dangerous in another (``find -delete`` vs ``find .``). When it returns a
  reason, the command is not unconditionally allowed even though its leading
  words are allowlisted.

These run *after* hard denials (denylist) and never relax an existing NEVER.
"""

from __future__ import annotations

# Command prefixes that wrap another command. Unwrapped before danger analysis
# so `sudo rm -rf /` is treated as `rm -rf /`. `env` also swallows VAR=val
# assignments and its own flags between the keyword and the real command.
_WRAPPERS = {"sudo", "nohup", "time", "command", "exec", "nice", "ionice"}

# chmod modes that broadly open permissions or wipe them — a security smell worth
# an explicit prompt rather than silent auto-approval.
_CHMOD_DANGEROUS_MODES = frozenset({
    "777",
    "666",
    "000",
    "0777",
    "0666",
    "0000",
    "a+rwx",
    "ugo+rwx",
})

# Block-device stems written via `dd of=/dev/<stem>...`. /dev/null and /dev/stdout
# are safe sinks; everything matching these stems can destroy a disk.
_DEVICE_STEMS = ("sd", "nvme", "vd", "hd", "disk", "mmcblk", "loop")

# find predicates that write/execute beyond a read-only search. The -exec family
# (-exec/-execdir/-ok/-okdir) is handled separately by the bash guardrail path
# (it routes them through a session-approvable RequiredPermission); these are the
# write/output predicates that should likewise not be auto-approved.
_FIND_WRITE_PREDICATES = {"-delete", "-fls", "-fprint", "-fprint0", "-fprintf"}


def _unwrap(tokens: list[str]) -> list[str]:
    """Drop leading privilege/wrapper prefixes and env VAR=val assignments."""
    out = list(tokens)
    while out:
        head = out[0]
        if head in _WRAPPERS:
            out = out[1:]
            continue
        if head == "env":
            out = out[1:]
            while out and (out[0].startswith("-") or "=" in out[0]):
                out = out[1:]
            continue
        # Bare `VAR=val` assignment prefixing the real command (no `env`).
        if "=" in head and not head.startswith("-"):
            out = out[1:]
            continue
        break
    return out


def _rm_is_destructive(args: list[str]) -> bool:
    """rm is destructive when forceful and/or recursive."""
    for arg in args:
        if arg == "--":
            break
        if arg.startswith("--"):
            if arg in {"--force", "--recursive"}:
                return True
            continue
        if arg.startswith("-") and len(arg) > 1:
            if any(flag in arg for flag in ("r", "R", "f")):
                return True
    return False


def _chmod_is_destructive(args: list[str]) -> bool:
    """chmod is flagged when the mode broadly opens or wipes permissions."""
    if not args:
        return False
    recursive = any(a in {"-R", "--recursive"} for a in args)
    mode = next((a for a in args if not a.startswith("-")), None)
    if mode is None:
        return False
    # Strip common prefixes: `u+rwx`, `a+rwx`, `=777`, leading `0`.
    candidate = mode.lstrip("ugoab+=")
    if candidate in _CHMOD_DANGEROUS_MODES:
        return True
    return recursive and candidate in _CHMOD_DANGEROUS_MODES


def _dd_is_destructive(args: list[str]) -> bool:
    """dd is destructive when writing to a real block device (not /dev/null)."""
    for arg in args:
        if not arg.startswith("of="):
            continue
        target = arg[len("of=") :]
        if target in {"/dev/null", "/dev/stdout", "/dev/stderr"}:
            continue
        if target.startswith("/dev/") and any(stem in target for stem in _DEVICE_STEMS):
            return True
    return False


def _single_command_destructive_reason(command: str) -> str | None:
    tokens = _unwrap(command.split())
    if not tokens:
        return None
    name = tokens[0]
    rest = tokens[1:]
    basename = name.rsplit("/", 1)[-1]

    if basename == "rm" and _rm_is_destructive(rest):
        return (
            f"`{command}` is destructive (rm with force/recursive flags). "
            "It can delete many files without prompting and is not auto-approved."
        )
    if basename == "chmod" and _chmod_is_destructive(rest):
        return (
            f"`{command}` broadly changes permissions (chmod to an open/wide "
            "mode). Confirm before it runs."
        )
    if basename == "dd" and _dd_is_destructive(rest):
        return (
            f"`{command}` writes to a block device (dd of=/dev/...) and can "
            "destroy a disk. It is not auto-approved."
        )
    if basename.startswith("mkfs") or basename in {"fdformat", "shred", "wipefs"}:
        return (
            f"`{command}` destroys or reformats data ({basename}) and is not "
            "auto-approved."
        )
    return None


def destructive_command_reason(command_parts: list[str]) -> str | None:
    """Return a danger reason if any sub-command is inherently destructive.

    Scans every parsed sub-command (the bash tool splits compound commands with
    tree-sitter before calling this), unwrapping wrappers so privileged or
    environment-prefixed destruction is caught. Returns the first reason found
    or None when nothing is destructive.
    """
    for part in command_parts:
        if reason := _single_command_destructive_reason(part):
            return reason
    return None


def allowlisted_argument_is_unsafe(command: str) -> str | None:
    """Argument gate for a command whose leading words match the allowlist.

    Returns a reason when the arguments turn an otherwise-allowlisted binary
    into something that should not be auto-approved (currently: ``find`` with a
    write/output predicate). None means the arguments are safe to auto-approve.
    """
    tokens = _unwrap(command.split())
    if not tokens:
        return None
    name = tokens[0].rsplit("/", 1)[-1]
    if name == "find" and any(pred in tokens for pred in _FIND_WRITE_PREDICATES):
        return (
            f"`{command}` uses a find write/output predicate that can create, "
            "modify, or delete files; it is not auto-approved even though `find` "
            "is allowlisted."
        )
    return None
