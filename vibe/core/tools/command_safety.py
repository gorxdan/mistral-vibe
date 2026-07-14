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

import re
import shlex

from vibe.core.tools._command_tokens import split_bash_tokens, unwrap_command_tokens

_SHORT_OPTION_LENGTH = 2

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
_RUFF_GLOBAL_FLAGS = {
    "-h",
    "--help",
    "-V",
    "--version",
    "-v",
    "--verbose",
    "-q",
    "--quiet",
    "-s",
    "--silent",
    "--isolated",
}
_RUFF_GLOBAL_VALUE_OPTIONS = {"--config", "--color"}
_RUFF_WRITE_OPTIONS = {
    "--fix",
    "--fix-only",
    "--add-noqa",
    "--output-file",
    "-o",
    "--watch",
    "-w",
}
_OUTPUT_OPTION_COMMANDS = {
    "diff": (("--output",), ()),
    "sort": (("--output",), ("-o",)),
    "tree": ((), ("-o",)),
}
_UNIQ_VALUE_OPTIONS = frozenset({
    "--check-chars",
    "--skip-chars",
    "--skip-fields",
    "-f",
    "-s",
    "-w",
})
_DATE_VALUE_OPTIONS = frozenset({"--date", "--file", "--reference", "-d", "-f", "-r"})
_DATE_SET_OPERAND = re.compile(r"\d{8}(?:\d{2}|\d{4})?(?:\.\d{2})?")


def unwrapped_command(command: str) -> str | None:
    try:
        result = unwrap_command_tokens(split_bash_tokens(command))
    except ValueError:
        return None
    if result.ambiguous or not result.tokens:
        return None
    return shlex.join(result.tokens)


def _ruff_subcommand(args: list[str]) -> tuple[str, list[str]] | None:
    index = 0
    while index < len(args):
        token = args[index]
        if token in _RUFF_GLOBAL_FLAGS:
            index += 1
            continue
        if token in _RUFF_GLOBAL_VALUE_OPTIONS:
            if index + 1 >= len(args):
                return None
            index += 2
            continue
        if any(token.startswith(f"{option}=") for option in _RUFF_GLOBAL_VALUE_OPTIONS):
            index += 1
            continue
        if token.startswith("-"):
            return None
        return token, args[index + 1 :]
    return None


def _ruff_is_read_only(args: list[str]) -> bool:
    parsed = _ruff_subcommand(args)
    if parsed is None:
        return False
    subcommand, subcommand_args = parsed
    if any(
        arg in _RUFF_WRITE_OPTIONS
        or arg.startswith("--fix=")
        or arg.startswith("--fix-only=")
        or arg.startswith("--add-noqa=")
        or arg.startswith("--output-file=")
        for arg in subcommand_args
    ):
        return False
    if subcommand == "check":
        return "--no-fix" in subcommand_args or "--diff" in subcommand_args
    if subcommand == "format":
        return "--check" in subcommand_args or "--diff" in subcommand_args
    return False


def _has_output_option(
    args: list[str], long_options: tuple[str, ...], short_options: tuple[str, ...]
) -> bool:
    for argument in args:
        option = argument.partition("=")[0]
        if any(
            option == candidate
            or (
                option.startswith("--")
                and len(option) > _SHORT_OPTION_LENGTH
                and candidate.startswith(option)
            )
            for candidate in long_options
        ):
            return True
        if any(
            argument == candidate
            or (argument.startswith(candidate) and len(argument) > len(candidate))
            for candidate in short_options
        ):
            return True
    return False


def _less_can_write(args: list[str]) -> bool:
    for argument in args:
        option = argument.partition("=")[0]
        folded = option.casefold()
        if (
            folded.startswith("--")
            and len(folded) > _SHORT_OPTION_LENGTH
            and any(
                candidate.startswith(folded)
                for candidate in {"--log-file", "--log-file2"}
            )
        ):
            return True
        if argument in {"-o", "-O"} or (
            argument.startswith(("-o", "-O")) and len(argument) > _SHORT_OPTION_LENGTH
        ):
            return True
    return False


def _date_can_set_clock(args: list[str]) -> bool:
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "--":
            return any(_DATE_SET_OPERAND.fullmatch(item) for item in args[index + 1 :])
        option, separator, _value = argument.partition("=")
        if option in _DATE_VALUE_OPTIONS:
            index += 1 if separator or argument != option else _SHORT_OPTION_LENGTH
            continue
        if option.startswith("--") and len(option) > _SHORT_OPTION_LENGTH:
            if "--set".startswith(option):
                return True
            index += 1
            continue
        if argument.startswith("-") and argument != "-":
            if "s" in argument[1:]:
                return True
            index += 1
            continue
        if _DATE_SET_OPERAND.fullmatch(argument):
            return True
        index += 1
    return False


def _uniq_has_output_operand(args: list[str]) -> bool:
    operands: list[str] = []
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "--":
            operands.extend(args[index + 1 :])
            break
        option = argument.partition("=")[0]
        if option in _UNIQ_VALUE_OPTIONS:
            if "=" not in argument and argument == option:
                index += 2
            else:
                index += 1
            continue
        if argument.startswith("-") and argument != "-":
            index += 1
            continue
        operands.append(argument)
        index += 1
    return len(operands) > 1


def _allowlisted_effect_reason(
    command: str, name: str, arguments: list[str]
) -> str | None:
    output_options = _OUTPUT_OPTION_COMMANDS.get(name)
    if output_options and _has_output_option(arguments, *output_options):
        return f"`{command}` selects an output file and is not read-only."
    if name == "less" and _less_can_write(arguments):
        return f"`{command}` enables less logging and can write a file."
    if name == "date" and _date_can_set_clock(arguments):
        return f"`{command}` can change the system clock."
    if name == "uniq" and _uniq_has_output_operand(arguments):
        return f"`{command}` supplies uniq's output-file operand."
    return None


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


def _destructive_reason_for_tokens(command: str, tokens: list[str]) -> str | None:
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


def _single_command_destructive_reason(command: str) -> str | None:
    try:
        result = unwrap_command_tokens(split_bash_tokens(command))
    except ValueError:
        return f"`{command}` could not be tokenized safely; it is not auto-approved."
    if result.ambiguous:
        return f"`{command}` uses a command wrapper that could not be classified safely; it is not auto-approved."
    return _destructive_reason_for_tokens(command, list(result.tokens))


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
    into something that should not be auto-approved (``find`` with a
    write/output predicate or a mutating ``ruff`` mode). None means the
    arguments are safe to auto-approve.
    """
    try:
        result = unwrap_command_tokens(split_bash_tokens(command))
    except ValueError:
        return f"`{command}` could not be tokenized safely; it is not auto-approved."
    if result.ambiguous:
        return f"`{command}` uses a command wrapper that could not be classified safely; it is not auto-approved."
    tokens = list(result.tokens)
    if not tokens:
        return None
    name = tokens[0].rsplit("/", 1)[-1]
    if name == "find" and any(pred in tokens for pred in _FIND_WRITE_PREDICATES):
        return (
            f"`{command}` uses a find write/output predicate that can create, "
            "modify, or delete files; it is not auto-approved even though `find` "
            "is allowlisted."
        )
    if name == "ruff":
        args = tokens[1:]
        if not _ruff_is_read_only(args):
            return (
                f"`{command}` is not an explicitly read-only ruff invocation; "
                "use `ruff check --no-fix ...`, `ruff check --diff ...`, or "
                "`ruff format --check/--diff ...`."
            )
    return _allowlisted_effect_reason(command, name, tokens[1:])
