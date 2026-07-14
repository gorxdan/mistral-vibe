"""Conservative token normalization for policy checks, never shell execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
import shlex

_EXECUTABLE_SUFFIXES = (".exe", ".cmd", ".bat", ".com")
_DYNAMIC_MARKERS = ("$", "`")
_SHORT_OPTION_LENGTH = 2
_CONTROL_ESCAPE_LENGTH = 2
_MAX_UNICODE_CODEPOINT = 0x10FFFF
_SURROGATE_RANGE = range(0xD800, 0xE000)
_BYTE_MASK = 0xFF
_ANSI_C_ESCAPE = re.compile(
    r"\\(?:[0-7]{1,3}|x[0-9A-Fa-f]{1,2}|u[0-9A-Fa-f]{1,4}|U[0-9A-Fa-f]{1,8}|c.|.)"
)


@dataclass(frozen=True, slots=True)
class OptionSpec:
    flags: frozenset[str] = frozenset()
    values: frozenset[str] = frozenset()
    optional_values: frozenset[str] = frozenset()
    optional_numeric_values: frozenset[str] = frozenset()
    cargo_toolchain: bool = False


@dataclass(frozen=True, slots=True)
class ParsedLeadingCommand:
    token: str | None
    arguments: tuple[str, ...]
    ambiguous: bool = False


@dataclass(frozen=True, slots=True)
class ParsedPythonModule:
    module: str | None
    arguments: tuple[str, ...]
    ambiguous: bool = False


@dataclass(frozen=True, slots=True)
class UnwrappedCommandTokens:
    tokens: tuple[str, ...]
    changed: bool
    ambiguous: bool = False
    dynamic: bool = False
    shell_payload: str | None = None
    environment_assignments: tuple[str, ...] = ()


def split_bash_tokens(command: str) -> list[str]:
    return shlex.split(normalize_bash_ansi_c(command), posix=True)


def normalize_bash_ansi_c(command: str) -> str:
    output: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(command):
        character = command[index]
        if quote is not None:
            output.append(character)
            if character == "\\" and quote == '"' and index + 1 < len(command):
                index += 1
                output.append(command[index])
            elif character == quote:
                quote = None
            index += 1
            continue
        if character == "\\" and index + 1 < len(command):
            output.extend((character, command[index + 1]))
            index += 2
            continue
        if character in {"'", '"'}:
            quote = character
            output.append(character)
            index += 1
            continue
        if not command.startswith("$'", index):
            output.append(character)
            index += 1
            continue
        body, end = _ansi_c_body(command, index + 2)
        if body is None:
            output.append(command[index:])
            break
        output.append(shlex.quote(_decode_ansi_c(body)))
        index = end
    return "".join(output)


def _ansi_c_body(command: str, start: int) -> tuple[str | None, int]:
    body: list[str] = []
    escaped = False
    for index in range(start, len(command)):
        character = command[index]
        if escaped:
            body.extend(("\\", character))
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "'":
            return "".join(body), index + 1
        body.append(character)
    return None, len(command)


def _decode_ansi_c(value: str) -> str:
    def decode(match: re.Match[str]) -> str:
        escape = match.group(0)[1:]
        if escape.startswith(("x", "u", "U")):
            try:
                codepoint = int(escape[1:], 16)
                if escape.startswith(("u", "U")) and (
                    codepoint > _MAX_UNICODE_CODEPOINT or codepoint in _SURROGATE_RANGE
                ):
                    raise ValueError
                return chr(codepoint)
            except (ValueError, OverflowError):
                raise ValueError("invalid ANSI-C escape") from None
        if escape[0] in "01234567":
            return chr(int(escape, 8) & _BYTE_MASK)
        if escape.startswith("c") and len(escape) == _CONTROL_ESCAPE_LENGTH:
            return chr(127 if escape[1] == "?" else ord(escape[1].upper()) & 0x1F)
        simple = {
            "a": "\a",
            "b": "\b",
            "e": "\x1b",
            "E": "\x1b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
            "v": "\v",
            "\\": "\\",
            "'": "'",
            '"': '"',
            "?": "?",
        }
        return simple.get(escape, match.group(0))

    return _ANSI_C_ESCAPE.sub(decode, value)


UV_OPTION_SPEC = OptionSpec(
    flags=frozenset({
        "--all-extras",
        "--all-groups",
        "--all-packages",
        "--compile-bytecode",
        "--exact",
        "--frozen",
        "--help",
        "--managed-python",
        "--native-tls",
        "--no-cache",
        "--no-config",
        "--no-dev",
        "--no-managed-python",
        "--no-project",
        "--no-progress",
        "--no-python-downloads",
        "--offline",
        "--refresh",
        "--verbose",
        "--version",
        "-q",
        "-v",
    }),
    values=frozenset({
        "--allow-insecure-host",
        "--cache-dir",
        "--config-file",
        "--config-setting",
        "--default-index",
        "--directory",
        "--env-file",
        "--exclude-newer",
        "--extra",
        "--find-links",
        "--fork-strategy",
        "--group",
        "--index",
        "--index-strategy",
        "--keyring-provider",
        "--link-mode",
        "--only-group",
        "--package",
        "--prerelease",
        "--project",
        "--python",
        "--python-platform",
        "--python-preference",
        "--resolution",
        "--with",
        "--with-editable",
        "--with-requirements",
    }),
)

PYTHON_OPTION_SPEC = OptionSpec(
    flags=frozenset({
        "--help",
        "--isolated",
        "--quiet",
        "--version",
        "-B",
        "-E",
        "-I",
        "-O",
        "-OO",
        "-P",
        "-S",
        "-V",
        "-b",
        "-d",
        "-i",
        "-q",
        "-s",
        "-u",
        "-v",
    }),
    values=frozenset({"--check-hash-based-pycs", "-W", "-X"}),
)

_WRAPPER_OPTION_SPECS = {
    "command": OptionSpec(flags=frozenset({"-p", "-V", "-v"})),
    "exec": OptionSpec(flags=frozenset({"-c", "-l"}), values=frozenset({"-a"})),
    "ionice": OptionSpec(
        flags=frozenset({"--ignore", "-t"}),
        values=frozenset({
            "--class",
            "--classdata",
            "--pid",
            "--pgid",
            "--uid",
            "-c",
            "-n",
            "-p",
            "-P",
            "-u",
        }),
    ),
    "nice": OptionSpec(values=frozenset({"--adjustment", "-n"})),
    "nohup": OptionSpec(flags=frozenset({"--help", "--version"})),
    "stdbuf": OptionSpec(
        values=frozenset({"--error", "--input", "--output", "-e", "-i", "-o"})
    ),
    "sudo": OptionSpec(
        flags=frozenset({
            "--askpass",
            "--background",
            "--edit",
            "--help",
            "--login",
            "--non-interactive",
            "--preserve-environment",
            "--remove-timestamp",
            "--reset-timestamp",
            "--set-home",
            "--shell",
            "--stdin",
            "--validate",
            "--version",
            "-A",
            "-B",
            "-E",
            "-H",
            "-K",
            "-S",
            "-V",
            "-b",
            "-e",
            "-i",
            "-k",
            "-l",
            "-n",
            "-s",
            "-v",
        }),
        optional_values=frozenset({"--preserve-env"}),
        values=frozenset({
            "--chdir",
            "--chroot",
            "--close-from",
            "--command-timeout",
            "--group",
            "--host",
            "--other-user",
            "--prompt",
            "--role",
            "--type",
            "--user",
            "-C",
            "-D",
            "-R",
            "-T",
            "-U",
            "-g",
            "-h",
            "-p",
            "-r",
            "-t",
            "-u",
        }),
    ),
    "time": OptionSpec(
        flags=frozenset({"--append", "--portable", "--verbose", "-a", "-p", "-v"}),
        values=frozenset({"--format", "--output", "-f", "-o"}),
    ),
}
_TIMEOUT_OPTION_SPEC = OptionSpec(
    flags=frozenset({"--foreground", "--preserve-status", "--verbose"}),
    values=frozenset({"--kill-after", "--signal", "-k", "-s"}),
)
_ENV_FLAGS = frozenset({"--debug", "--ignore-environment", "--null", "-0", "-i", "-v"})
_ENV_VALUES = frozenset({"--argv0", "--chdir", "--unset", "-C", "-a", "-u"})
_ENV_SPLIT_OPTIONS = frozenset({"--split-string", "-S"})
_ENV_OPTION_SPEC = OptionSpec(flags=_ENV_FLAGS, values=_ENV_VALUES | _ENV_SPLIT_OPTIONS)
_SHELL_VALUE_OPTIONS = frozenset({"--init-file", "--rcfile", "+O", "+o", "-O", "-o"})
_SHELL_STARTUP_OPTIONS = frozenset({"--init-file", "--login", "--rcfile"})
_SHELL_FLAGS = frozenset({
    "--debugger",
    "--help",
    "--login",
    "--noediting",
    "--noprofile",
    "--norc",
    "--posix",
    "--pretty-print",
    "--restricted",
    "--verbose",
    "--version",
})
_UNIX_SHELLS = frozenset({"ash", "bash", "dash", "fish", "ksh", "sh", "zsh"})
_CMD_PROCESSORS = frozenset({"cmd"})
_POWERSHELL_PROCESSORS = frozenset({"powershell", "pwsh"})


def command_name(token: str) -> str:
    name = re.split(r"[/\\]", token)[-1].casefold()
    for suffix in _EXECUTABLE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def token_is_dynamic(token: str) -> bool:
    return any(marker in token for marker in _DYNAMIC_MARKERS) or bool(
        re.search(r"%[^%]+%|![A-Za-z_][A-Za-z0-9_]*!", token)
    )


def _is_windows_command_variable(token: str) -> bool:
    return bool(
        re.fullmatch(r"(?:%[A-Za-z_][A-Za-z0-9_]*%|![A-Za-z_][A-Za-z0-9_]*!)", token)
    )


def token_is_assignment(token: str) -> bool:
    return bool(
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]+\])?\+?=.*", token, re.DOTALL)
    )


def _attached_value_option(argument: str, options: frozenset[str]) -> bool:
    return any(
        option.startswith("-")
        and not option.startswith("--")
        and len(option) == _SHORT_OPTION_LENGTH
        and argument.startswith(option)
        and len(argument) > len(option)
        for option in options
    )


def _combined_short_flags(argument: str, flags: frozenset[str]) -> bool:
    return bool(
        len(argument) > _SHORT_OPTION_LENGTH
        and argument.startswith("-")
        and not argument.startswith("--")
        and "=" not in argument
        and all(f"-{flag}" in flags for flag in argument[1:])
    )


def _canonical_long_option(option: str, spec: OptionSpec) -> tuple[str, bool]:
    known = (
        spec.flags | spec.values | spec.optional_values | spec.optional_numeric_values
    )
    if not option.startswith("--") or option in known:
        return option, False
    matches = [candidate for candidate in known if candidate.startswith(option)]
    if len(matches) == 1:
        return matches[0], False
    return option, len(matches) > 1


def parse_leading_command(
    arguments: list[str] | tuple[str, ...], spec: OptionSpec
) -> ParsedLeadingCommand:
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if spec.cargo_toolchain and argument.startswith("+") and len(argument) > 1:
            index += 1
            continue
        if argument == "--":
            index += 1
            break
        if not argument.startswith("-") or argument == "-":
            break
        option, separator, _ = argument.partition("=")
        option, ambiguous_long = _canonical_long_option(option, spec)
        if ambiguous_long:
            return ParsedLeadingCommand(None, (), ambiguous=True)
        value_option = (
            option
            if option in spec.values
            else next(
                (
                    candidate
                    for candidate in spec.values
                    if _attached_value_option(argument, frozenset({candidate}))
                ),
                None,
            )
        )
        if option in spec.optional_numeric_values:
            if separator or _attached_value_option(argument, frozenset({option})):
                index += 1
                continue
            if index + 1 < len(arguments) and arguments[index + 1].isdigit():
                index += 2
            else:
                index += 1
            continue
        optional_value = option in spec.optional_values or _attached_value_option(
            argument, spec.optional_values
        )
        if (
            optional_value
            or option in spec.flags
            or _combined_short_flags(argument, spec.flags)
        ):
            if separator and not optional_value:
                return ParsedLeadingCommand(None, (), ambiguous=True)
            index += 1
            continue
        if value_option is not None:
            if separator or value_option != option:
                index += 1
                continue
            if index + 1 >= len(arguments):
                return ParsedLeadingCommand(None, (), ambiguous=True)
            index += 2
            continue
        return ParsedLeadingCommand(None, (), ambiguous=True)
    if index >= len(arguments):
        return ParsedLeadingCommand(None, ())
    return ParsedLeadingCommand(arguments[index], tuple(arguments[index + 1 :]))


def parse_python_module(arguments: list[str] | tuple[str, ...]) -> ParsedPythonModule:
    module: str | None = None
    remaining: tuple[str, ...] = ()
    ambiguous = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "-m":
            if index + 1 >= len(arguments):
                ambiguous = True
            else:
                module = arguments[index + 1]
                remaining = tuple(arguments[index + 2 :])
            break
        if argument in {"-c", "--"} or not argument.startswith("-"):
            break
        option, separator, _ = argument.partition("=")
        if option in PYTHON_OPTION_SPEC.flags:
            ambiguous = bool(separator)
            index += 1
        elif option in PYTHON_OPTION_SPEC.values:
            attached = (
                len(option) == _SHORT_OPTION_LENGTH
                and argument.startswith(option)
                and len(argument) > len(option)
            )
            if separator or attached:
                index += 1
            elif index + 1 < len(arguments):
                index += 2
            else:
                ambiguous = True
        else:
            ambiguous = True
        if ambiguous:
            break
    return ParsedPythonModule(module, remaining, ambiguous)


def _split_payload(payload: str) -> tuple[tuple[str, ...], bool]:
    if token_is_dynamic(payload):
        return (payload,), True
    try:
        return tuple(split_bash_tokens(payload)), False
    except ValueError:
        return (), True


def _shell_payload_result(
    payload: str, *, ambiguous: bool = False
) -> UnwrappedCommandTokens:
    split, dynamic = _split_payload(payload)
    return UnwrappedCommandTokens(
        split,
        True,
        ambiguous=ambiguous or not split,
        dynamic=dynamic,
        shell_payload=payload,
    )


def _next_shell_value_index(
    arguments: list[str], index: int, option: str
) -> int | None:
    argument = arguments[index]
    if "=" in argument or _attached_value_option(argument, frozenset({option})):
        return index + 1
    return index + 2 if index + 1 < len(arguments) else None


def _unwrap_env_split(
    arguments: list[str], index: int, option: str, separator: str, value: str
) -> UnwrappedCommandTokens:
    argument = arguments[index]
    if option in _ENV_SPLIT_OPTIONS and separator:
        payload = value
        remainder = arguments[index + 1 :]
    elif argument.startswith("-S") and argument != "-S":
        payload = argument[2:]
        remainder = arguments[index + 1 :]
    elif index + 1 < len(arguments):
        payload = arguments[index + 1]
        remainder = arguments[index + 2 :]
    else:
        return UnwrappedCommandTokens((), True, ambiguous=True)
    split, dynamic = _split_payload(payload)
    command_payload = shlex.join((*split, *remainder)) if split else payload
    return UnwrappedCommandTokens(
        (*split, *remainder),
        True,
        ambiguous=not split,
        dynamic=dynamic,
        shell_payload=command_payload,
    )


def _unwrap_env(arguments: list[str]) -> UnwrappedCommandTokens:
    index = 0
    argv0_changed = False
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            index += 1
            break
        if not argument.startswith("-") or argument == "-":
            break
        option, separator, value = argument.partition("=")
        option, ambiguous_long = _canonical_long_option(option, _ENV_OPTION_SPEC)
        if ambiguous_long:
            return UnwrappedCommandTokens((), True, ambiguous=True)
        if option in _ENV_SPLIT_OPTIONS or argument.startswith("-S"):
            return _unwrap_env_split(arguments, index, option, separator, value)
        if option in _ENV_FLAGS:
            if separator:
                return UnwrappedCommandTokens((), True, ambiguous=True)
            index += 1
            continue
        value_option = (
            option
            if option in _ENV_VALUES
            else next(
                (
                    candidate
                    for candidate in _ENV_VALUES
                    if _attached_value_option(argument, frozenset({candidate}))
                ),
                None,
            )
        )
        if value_option is not None:
            argv0_changed = argv0_changed or value_option in {"--argv0", "-a"}
            if separator or value_option != option:
                index += 1
                continue
            if index + 1 >= len(arguments):
                return UnwrappedCommandTokens((), True, ambiguous=True)
            index += 2
            continue
        return UnwrappedCommandTokens((), True, ambiguous=True)
    assignment_start = index
    while index < len(arguments) and token_is_assignment(arguments[index]):
        index += 1
    return UnwrappedCommandTokens(
        tuple(arguments[index:]),
        True,
        ambiguous=argv0_changed,
        environment_assignments=tuple(arguments[assignment_start:index]),
    )


def _unwrap_shell(name: str, arguments: list[str]) -> UnwrappedCommandTokens:
    return _unwrap_executing_shell(name, arguments)


def _unwrap_executing_shell(name: str, arguments: list[str]) -> UnwrappedCommandTokens:
    index = 0
    startup_risk = False
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            index += 1
            break
        if not argument.startswith(("-", "+")) or argument in {"-", "+"}:
            break
        option = argument.partition("=")[0]
        startup_risk = (
            startup_risk
            or option in _SHELL_STARTUP_OPTIONS
            or (
                argument.startswith("-")
                and not argument.startswith("--")
                and bool(set(argument[1:]) & {"i", "l"})
            )
        )
        if argument == "-c" or (
            argument.startswith("-")
            and not argument.startswith("--")
            and "c" in argument[1:]
        ):
            if index + 1 >= len(arguments):
                return UnwrappedCommandTokens((), True, ambiguous=True)
            return _shell_payload_result(arguments[index + 1], ambiguous=startup_risk)
        if option in _SHELL_VALUE_OPTIONS:
            next_index = _next_shell_value_index(arguments, index, option)
            if next_index is None:
                return UnwrappedCommandTokens((), True, ambiguous=True)
            index = next_index
            continue
        if option in _SHELL_FLAGS or (
            argument.startswith("-") and not argument.startswith("--")
        ):
            index += 1
            continue
        return UnwrappedCommandTokens((), True, ambiguous=True)
    if index < len(arguments):
        result = UnwrappedCommandTokens((), True, ambiguous=True)
    elif startup_risk:
        result = UnwrappedCommandTokens((), True, ambiguous=True)
    else:
        result = UnwrappedCommandTokens((name, *arguments), False)
    return result


def _cmd_switch_payload(arguments: list[str]) -> tuple[bool, str]:
    for index, argument in enumerate(arguments):
        option = argument.casefold()
        if option in {"/c", "/k"}:
            return True, " ".join(arguments[index + 1 :])
        if len(argument) > _SHORT_OPTION_LENGTH and option[:_SHORT_OPTION_LENGTH] in {
            "/c",
            "/k",
        }:
            return True, " ".join((
                argument[_SHORT_OPTION_LENGTH:],
                *arguments[index + 1 :],
            ))
    return False, ""


def _unwrap_cmd(arguments: list[str]) -> UnwrappedCommandTokens:
    found, payload = _cmd_switch_payload(arguments)
    if found:
        split, dynamic = _split_payload(payload)
        return UnwrappedCommandTokens(
            split, True, ambiguous=not split, dynamic=dynamic, shell_payload=payload
        )
    return UnwrappedCommandTokens(("cmd", *arguments), False)


def _unwrap_powershell(arguments: list[str]) -> UnwrappedCommandTokens:
    for index, argument in enumerate(arguments):
        option = argument.casefold()
        if len(option) > 1 and "-encodedcommand".startswith(option):
            return UnwrappedCommandTokens((), True, ambiguous=True, dynamic=True)
        if option not in {"-", "--"} and "-commandwithargs".startswith(option):
            command_option = True
        else:
            command_option = option in {"-command", "-c"}
        if command_option:
            if index + 1 >= len(arguments):
                return UnwrappedCommandTokens((), True, ambiguous=True)
            payload = arguments[index + 1]
            split, dynamic = _split_payload(payload)
            return UnwrappedCommandTokens(
                split, True, ambiguous=not split, dynamic=dynamic, shell_payload=payload
            )
        if len(option) > 1 and "-file".startswith(option):
            return UnwrappedCommandTokens((), True, ambiguous=True)
    return UnwrappedCommandTokens(
        ("pwsh", *arguments), False, ambiguous=bool(arguments)
    )


def _unwrap_uv(_: str, arguments: list[str]) -> UnwrappedCommandTokens:
    parsed = parse_leading_command(arguments, UV_OPTION_SPEC)
    if parsed.ambiguous:
        return UnwrappedCommandTokens((), True, ambiguous=True)
    if command_name(parsed.token or "") != "run":
        return UnwrappedCommandTokens(("uv", *arguments), False)
    run = parse_leading_command(list(parsed.arguments), UV_OPTION_SPEC)
    if run.ambiguous:
        return UnwrappedCommandTokens((), True, ambiguous=True)
    child = () if run.token is None else (run.token, *run.arguments)
    return UnwrappedCommandTokens(child, True, ambiguous=not child)


def _unwrap_env_command(_: str, arguments: list[str]) -> UnwrappedCommandTokens:
    return _unwrap_env(arguments)


def _unwrap_cmd_command(_: str, arguments: list[str]) -> UnwrappedCommandTokens:
    return _unwrap_cmd(arguments)


def _unwrap_powershell_command(_: str, arguments: list[str]) -> UnwrappedCommandTokens:
    return _unwrap_powershell(arguments)


def _unwrap_sudo(name: str, arguments: list[str]) -> UnwrappedCommandTokens:
    normalized = [
        "--preserve-env" if argument.startswith("--preserve-env=") else argument
        for argument in arguments
    ]
    return _unwrap_with_option_spec(name, normalized)


def _unwrap_timeout(_: str, arguments: list[str]) -> UnwrappedCommandTokens:
    parsed = parse_leading_command(arguments, _TIMEOUT_OPTION_SPEC)
    if parsed.ambiguous or parsed.token is None:
        return UnwrappedCommandTokens((), True, ambiguous=True)
    child = list(parsed.arguments)
    if child[:1] == ["--"]:
        child = child[1:]
    return UnwrappedCommandTokens(tuple(child), True, ambiguous=not child)


def _unwrap_eval(_: str, arguments: list[str]) -> UnwrappedCommandTokens:
    if not arguments:
        return UnwrappedCommandTokens((), True, ambiguous=True)
    payload = " ".join(arguments)
    split, dynamic = _split_payload(payload)
    return UnwrappedCommandTokens(
        split, True, ambiguous=not split, dynamic=dynamic, shell_payload=payload
    )


def _unwrap_with_option_spec(name: str, arguments: list[str]) -> UnwrappedCommandTokens:
    parsed = parse_leading_command(arguments, _WRAPPER_OPTION_SPECS[name])
    child = () if parsed.token is None else (parsed.token, *parsed.arguments)
    argv0_changed = name == "exec" and any(
        argument == "-a" or argument.startswith("-a") for argument in arguments
    )
    return UnwrappedCommandTokens(
        child, True, ambiguous=parsed.ambiguous or not child or argv0_changed
    )


def _unwrap_nice(name: str, arguments: list[str]) -> UnwrappedCommandTokens:
    remaining = (
        arguments[1:]
        if arguments and re.fullmatch(r"-\d+", arguments[0])
        else arguments
    )
    return _unwrap_with_option_spec(name, remaining)


type _UnwrapHandler = Callable[[str, list[str]], UnwrappedCommandTokens]

_UNWRAP_HANDLERS: dict[str, _UnwrapHandler] = {
    **dict.fromkeys(_UNIX_SHELLS, _unwrap_shell),
    **dict.fromkeys(_CMD_PROCESSORS, _unwrap_cmd_command),
    **dict.fromkeys(_POWERSHELL_PROCESSORS, _unwrap_powershell_command),
    "env": _unwrap_env_command,
    "eval": _unwrap_eval,
    "nice": _unwrap_nice,
    "sudo": _unwrap_sudo,
    "timeout": _unwrap_timeout,
    "uv": _unwrap_uv,
}


def _unwrap_once(tokens: list[str]) -> UnwrappedCommandTokens:
    if not tokens:
        return UnwrappedCommandTokens((), False)
    raw_name = tokens[0].casefold()
    name = command_name(tokens[0])
    arguments = tokens[1:]
    cmd_switch, _ = _cmd_switch_payload(arguments)
    if _is_windows_command_variable(raw_name) and cmd_switch:
        result = _unwrap_cmd(arguments)
    elif handler := _UNWRAP_HANDLERS.get(name):
        result = handler(name, arguments)
    elif name in _WRAPPER_OPTION_SPECS:
        result = _unwrap_with_option_spec(name, arguments)
    elif token_is_assignment(tokens[0]):
        result = UnwrappedCommandTokens(
            tuple(tokens[1:]), True, ambiguous=len(tokens) == 1
        )
    else:
        result = UnwrappedCommandTokens(tuple(tokens), False)
    return result


def unwrap_command_once(tokens: list[str] | tuple[str, ...]) -> UnwrappedCommandTokens:
    return _unwrap_once(list(tokens))


def unwrap_command_tokens(
    tokens: list[str] | tuple[str, ...],
) -> UnwrappedCommandTokens:
    current = list(tokens)
    changed = False
    dynamic = False
    shell_payload: str | None = None
    environment_assignments: tuple[str, ...] = ()
    for _ in range(12):
        result = _unwrap_once(current)
        changed = changed or result.changed
        dynamic = dynamic or result.dynamic
        environment_assignments += result.environment_assignments
        if shell_payload is None and result.shell_payload is not None:
            shell_payload = result.shell_payload
        if result.ambiguous:
            return UnwrappedCommandTokens(
                (),
                changed,
                ambiguous=True,
                dynamic=dynamic,
                shell_payload=shell_payload,
                environment_assignments=environment_assignments,
            )
        if not result.changed:
            return UnwrappedCommandTokens(
                tuple(current),
                changed,
                dynamic=dynamic,
                shell_payload=shell_payload,
                environment_assignments=environment_assignments,
            )
        current = list(result.tokens)
    return UnwrappedCommandTokens(
        (),
        True,
        ambiguous=True,
        dynamic=dynamic,
        shell_payload=shell_payload,
        environment_assignments=environment_assignments,
    )
