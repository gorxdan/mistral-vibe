from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
import hashlib
from pathlib import PurePosixPath
import re
import shlex
import unicodedata

from vibe.core.tools._command_tokens import (
    UV_OPTION_SPEC,
    OptionSpec,
    command_name,
    parse_leading_command,
    parse_python_module,
)
from vibe.core.utils.platform import is_windows


class VerificationVerdict(StrEnum):
    PASS = auto()
    FAIL = auto()
    PARTIAL = auto()


@dataclass(frozen=True)
class CommandEvidence:
    check: str
    command: str
    output: str
    result: VerificationVerdict


@dataclass(frozen=True)
class VerificationReport:
    verdict: VerificationVerdict
    evidence: tuple[CommandEvidence, ...]

    @property
    def passed(self) -> bool:
        return self.verdict == VerificationVerdict.PASS

    def summary(self) -> str:
        commands = "; ".join(item.command.splitlines()[0] for item in self.evidence)
        return f"VERDICT: {self.verdict.value.upper()} ({len(self.evidence)} checks: {commands})"


class VerificationReportError(ValueError):
    pass


def verification_command_hash(command: str) -> str:
    normalized = "\n".join(line.strip() for line in command.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


_SHELL_CONTROL_TOKENS = frozenset({
    "&",
    "&&",
    "(",
    ")",
    ";",
    "<",
    "<<",
    ">",
    ">>",
    "|",
    "|&",
    "||",
    "`",
})
_DOTNET_TEST_COUNTS = (
    re.compile(r"\bTotal tests:\s*(?P<count>\d+)\b", re.IGNORECASE),
    re.compile(r"\bTotal:\s*(?P<count>\d+)\b", re.IGNORECASE),
)
_PYTEST_PASSED_COUNTS = re.compile(
    r"(?<![\w.])(?P<count>\d+)\s+passed\b", re.IGNORECASE
)
_UNITTEST_RUN_COUNTS = re.compile(
    r"^\s*Ran\s+(?P<count>\d+)\s+tests?\b", re.IGNORECASE | re.MULTILINE
)
_CARGO_PASSED_COUNTS = re.compile(
    r"^\s*test result:\s+ok\.\s+(?P<count>\d+)\s+passed;", re.IGNORECASE | re.MULTILINE
)
_GO_PASSED_TEST = re.compile(r"^\s*--- PASS:", re.MULTILINE)
_NONEXECUTING_FLAGS = frozenset({
    "--collect-only",
    "--co",
    "--dry-run",
    "--fixtures",
    "--fixtures-per-test",
    "--help",
    "--list",
    "--list-sessions",
    "--list-tests",
    "--listenvs",
    "--listenvs-all",
    "--markers",
    "--no-run",
    "--setup-plan",
    "--showconfig",
    "--showconfig-json",
    "--trace-config",
    "--version",
})
_SHORT_NONEXECUTING_FLAGS = frozenset({"-?", "-h", "-V", "/?"})
_FAILURE_MASKING_FLAGS = frozenset({
    "--exit-zero",
    "--no-test",
    "--notest",
    "--pass-with-no-tests",
    "--passwithnotests",
    "--suppress-no-test-exit-code",
})
_MAKE_NONEXECUTING_FLAGS = frozenset({
    "--dry-run",
    "--just-print",
    "--question",
    "--recon",
    "-n",
    "-q",
})
_PYTHON_MODULE_PREFIX_LENGTH = 3
_PYTHON_MODULE_RUNNERS = {
    "mypy": "mypy",
    "pyright": "pyright",
    "pytest": "pytest",
    "ruff": "ruff",
    "unittest": "unittest",
}
_DIRECT_RUNNERS = frozenset({
    "bandit",
    "eslint",
    "flake8",
    "jest",
    "mypy",
    "nox",
    "py.test",
    "pyright",
    "pytest",
    "tox",
    "tsc",
    "vitest",
})
_RUNNER_SUBCOMMANDS = {
    "cargo": frozenset({"build", "check", "clippy", "test"}),
    "dotnet": frozenset({"build", "restore", "test"}),
    "go": frozenset({"build", "test", "vet"}),
    "make": frozenset({"check", "lint", "test", "typecheck"}),
    "pre-commit": frozenset({"run"}),
    "ruff": frozenset({"check", "format"}),
}
_SCRIPT_RUNNERS = frozenset({"bun", "npm", "pnpm", "yarn"})
_VERIFICATION_SCRIPTS = frozenset({"lint", "test", "typecheck"})
_RUNNER_OPTION_SPECS = {
    "cargo": OptionSpec(
        flags=frozenset({
            "--frozen",
            "--locked",
            "--offline",
            "--quiet",
            "--verbose",
            "-q",
            "-v",
        }),
        values=frozenset({
            "--color",
            "--config",
            "--manifest-path",
            "--target-dir",
            "-Z",
        }),
        cargo_toolchain=True,
    ),
    "dotnet": OptionSpec(
        flags=frozenset({"--diagnostics", "--no-logo"}),
        values=frozenset({
            "--architecture",
            "--roll-forward",
            "--runtime",
            "--verbosity",
            "-a",
            "-r",
            "-v",
        }),
    ),
    "make": OptionSpec(
        flags=frozenset({"--dry-run", "--question", "--silent", "-n", "-q", "-s"}),
        values=frozenset({"--directory", "--eval", "--file", "-C", "-f"}),
        optional_numeric_values=frozenset({"--jobs", "-j"}),
    ),
    "npm": OptionSpec(
        flags=frozenset({"--silent", "--verbose", "-s"}),
        values=frozenset({
            "--loglevel",
            "--prefix",
            "--userconfig",
            "--workspace",
            "-w",
        }),
    ),
    "pnpm": OptionSpec(
        flags=frozenset({"--silent", "--workspace-root", "-s", "-w"}),
        values=frozenset({"--dir", "--filter", "-C"}),
    ),
    "pre-commit": OptionSpec(values=frozenset({"--config", "-c"})),
    "ruff": OptionSpec(
        flags=frozenset({
            "--isolated",
            "--quiet",
            "--silent",
            "--verbose",
            "-q",
            "-s",
            "-v",
        }),
        values=frozenset({"--config", "--output-format"}),
    ),
    "yarn": OptionSpec(
        flags=frozenset({"--silent", "--verbose", "-s"}), values=frozenset({"--cwd"})
    ),
}
_LAUNCHER_OPTION_SPEC = OptionSpec(
    flags=frozenset({"--no-install", "--yes", "-y"}),
    values=frozenset({
        "--cache",
        "--cache-dir",
        "--package",
        "--registry",
        "--userconfig",
        "-p",
    }),
)


def is_verification_command(command: str) -> bool:
    tokens = _direct_command_tokens(command)
    return tokens is not None and _verification_runner_tokens(tokens) is not None


def verification_observation_hashes(
    command: str, stdout: str, stderr: str
) -> tuple[str, ...]:
    if is_windows():
        return ()
    tokens = _direct_command_tokens(command)
    output = "\n".join(part for part in (stdout, stderr) if part)
    if tokens is None or not _command_output_is_eligible(tokens, output):
        return ()
    command_hash = verification_command_hash(command)
    return tuple(
        hashlib.sha256(f"{command_hash}\0{line}".encode()).hexdigest()
        for line in dict.fromkeys(_normalized_output_lines(output))
    )


def verification_command_output_diagnostics(
    argv: Sequence[str],
    output: str,
    *,
    custom_runner: bool = False,
    has_test_count_contract: bool = False,
) -> tuple[str, ...]:
    raw_command = list(argv)
    command = _verification_runner_tokens(raw_command)
    if _command_is_nonexecuting(raw_command) or (
        command is not None and _command_is_nonexecuting(command)
    ):
        return (
            f"verification command does not execute checks: {shlex.join(raw_command)}",
        )
    if command is None:
        if custom_runner:
            return ()
        return (
            f"verification command is not a recognized check runner: "
            f"{shlex.join(raw_command)}",
        )
    return _positive_execution_diagnostics(
        command, output, has_test_count_contract=has_test_count_contract
    )


def _positive_execution_diagnostics(
    command: list[str], output: str, *, has_test_count_contract: bool
) -> tuple[str, ...]:
    if command[:2] == ["dotnet", "test"]:
        return _dotnet_test_output_diagnostics(output)
    if has_test_count_contract:
        return ()
    executable = command[0]
    runner_counts: tuple[str, tuple[int, ...]] | None = None
    if executable in {"py.test", "pytest"}:
        runner_counts = (
            "pytest",
            tuple(_pattern_counts(_PYTEST_PASSED_COUNTS, output)),
        )
    elif executable == "unittest":
        runner_counts = (
            "unittest",
            tuple(_pattern_counts(_UNITTEST_RUN_COUNTS, output)),
        )
    elif command[:2] == ["cargo", "test"]:
        runner_counts = (
            "cargo test",
            tuple(_pattern_counts(_CARGO_PASSED_COUNTS, output)),
        )
    elif command[:2] == ["go", "test"]:
        runner_counts = ("go test", (len(_GO_PASSED_TEST.findall(output)),))
    if runner_counts is not None:
        return _positive_count_diagnostics(*runner_counts)
    if _requires_explicit_test_count_contract(command):
        return (
            "verification command requires an explicit positive test-count "
            f"contract: {shlex.join(command)}",
        )
    return ()


def _positive_count_diagnostics(
    runner: str, counts: tuple[int, ...]
) -> tuple[str, ...]:
    if not counts:
        return (f"{runner} output did not report an executed test count",)
    if sum(counts) < 1:
        return (f"{runner} reported zero executed tests",)
    return ()


def _pattern_counts(pattern: re.Pattern[str], output: str) -> list[int]:
    return [int(match.group("count")) for match in pattern.finditer(output)]


def _requires_explicit_test_count_contract(command: list[str]) -> bool:
    if command[0] in {"jest", "nox", "tox", "vitest"}:
        return True
    if command[:2] == ["make", "test"]:
        return True
    if command[0] not in _SCRIPT_RUNNERS:
        return False
    script = command[2] if command[1:2] == ["run"] else command[1]
    return script == "test"


def _dotnet_test_output_diagnostics(output: str) -> tuple[str, ...]:
    counts = _dotnet_test_counts(output)
    if not counts:
        return ("dotnet test output did not report an executed test count",)
    if len(set(counts)) > 1:
        rendered = ", ".join(str(count) for count in sorted(set(counts)))
        return (f"dotnet test output reported conflicting test counts: {rendered}",)
    if any(count == 0 for count in counts):
        return ("dotnet test reported zero executed tests",)
    return ()


def report_evidence_was_observed(
    report: VerificationReport, observation_hashes: tuple[str, ...]
) -> bool:
    remaining = list(observation_hashes)
    for evidence in report.evidence:
        command_hash = verification_command_hash(evidence.command)
        lines = _normalized_output_lines(evidence.output)
        if not lines:
            return False
        for line in lines:
            digest = hashlib.sha256(f"{command_hash}\0{line}".encode()).hexdigest()
            if digest not in remaining:
                return False
            remaining.remove(digest)
    return bool(report.evidence)


def _direct_command_tokens(command: str) -> list[str] | None:
    if not command.strip() or "\n" in command or "\r" in command:
        return None
    if "$(" in command or "${" in command:
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>()`")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return None
    if not tokens or any(
        token in _SHELL_CONTROL_TOKENS or token.startswith((">", "<"))
        for token in tokens
    ):
        return None
    return tokens


def _command_output_is_eligible(tokens: list[str], output: str) -> bool:
    command = _verification_runner_tokens(tokens)
    if command is None or not _normalized_output_lines(output):
        return False
    return not verification_command_output_diagnostics(tokens, output)


def _dotnet_test_counts(output: str) -> tuple[int, ...]:
    return tuple(
        int(match.group("count"))
        for pattern in _DOTNET_TEST_COUNTS
        for match in pattern.finditer(output)
    )


def _command_is_nonexecuting(command: list[str]) -> bool:
    arguments = command[1:]
    if any(_is_nonexecuting_flag(argument) for argument in arguments) or any(
        argument.casefold() in _FAILURE_MASKING_FLAGS for argument in arguments
    ):
        return True
    return _runner_specific_nonexecuting(command)


def _runner_specific_nonexecuting(command: list[str]) -> bool:
    executable = command[0]
    match executable:
        case "make":
            return any(argument in _MAKE_NONEXECUTING_FLAGS for argument in command[1:])
        case "go" if command[1:2] == ["test"]:
            return _go_test_is_nonexecuting(command[2:])
        case "nox" | "tox":
            return command[1:2] in (["l"], ["list"])
        case "ruff" if command[1:2] == ["check"]:
            return any(
                argument in {"--show-files", "--show-settings"}
                for argument in command[2:]
            )
        case "ruff" if command[1:2] == ["format"]:
            return "--diff" in command[2:] and "--check" not in command[2:]
        case _:
            return False


def _go_test_is_nonexecuting(arguments: list[str]) -> bool:
    if any(
        argument in {"-c", "-list"} or argument.startswith("-list=")
        for argument in arguments
    ):
        return True
    return _go_test_selects_no_checks(arguments)


def _is_nonexecuting_flag(argument: str) -> bool:
    if argument in _SHORT_NONEXECUTING_FLAGS:
        return True
    return any(
        argument == flag or argument.startswith(f"{flag}=")
        for flag in _NONEXECUTING_FLAGS
    )


def _go_test_selects_no_checks(arguments: list[str]) -> bool:
    run_values = _option_values(arguments, "-run")
    if "^$" not in run_values:
        return False
    return not _option_values(arguments, "-bench") and not _option_values(
        arguments, "-fuzz"
    )


def _option_values(arguments: list[str], option: str) -> tuple[str, ...]:
    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument.startswith(f"{option}="):
            values.append(argument.partition("=")[2])
        elif argument == option and index + 1 < len(arguments):
            values.append(arguments[index + 1])
    return tuple(values)


def _verification_runner_tokens(tokens: list[str]) -> list[str] | None:
    normalized = list(tokens)
    if (
        not normalized
        or (executable := _evidence_executable_name(normalized[0])) is None
    ):
        return None
    normalized[0] = executable
    if normalized[0] == "uv":
        uv = parse_leading_command(normalized[1:], UV_OPTION_SPEC)
        if uv.ambiguous or command_name(uv.token or "") != "run":
            return None
        run = parse_leading_command(list(uv.arguments), UV_OPTION_SPEC)
        if run.ambiguous or run.token is None:
            return None
        normalized = [run.token, *run.arguments]
    if (
        not normalized
        or (executable := _evidence_executable_name(normalized[0])) is None
    ):
        return None
    normalized[0] = executable
    if len(normalized) >= _PYTHON_MODULE_PREFIX_LENGTH and re.fullmatch(
        r"python(?:3(?:\.\d+)*)?", normalized[0]
    ):
        return _python_module_runner_tokens(normalized)
    return _native_runner_tokens(normalized)


def _evidence_executable_name(token: str) -> str | None:
    normalized = token.casefold()
    if command_name(token) != normalized:
        return None
    if re.fullmatch(r"[a-z0-9][a-z0-9.+_-]*", normalized) is None:
        return None
    return normalized


def _python_module_runner_tokens(tokens: list[str]) -> list[str] | None:
    parsed = parse_python_module(tokens[1:])
    if parsed.ambiguous:
        return None
    runner = _PYTHON_MODULE_RUNNERS.get(parsed.module or "")
    if runner is None:
        return None
    return [runner, *parsed.arguments]


def _native_runner_tokens(normalized: list[str]) -> list[str] | None:
    executable = normalized[0]
    arguments = normalized[1:]
    if executable in {"bunx", "npx", "uvx"}:
        return _launched_runner_tokens(arguments)
    if executable in {"bun", "npm", "pnpm", "yarn"}:
        manager_tokens = _normalized_script_manager_tokens(executable, arguments)
        if manager_tokens is None:
            return None
        normalized = manager_tokens
        if child := _script_launcher_tokens(executable, manager_tokens[1:]):
            return _verification_runner_tokens(child)
        arguments = manager_tokens[1:]
    if executable in _DIRECT_RUNNERS:
        return normalized
    return _subcommand_runner_tokens(executable, arguments, normalized)


def _launched_runner_tokens(arguments: list[str]) -> list[str] | None:
    launched = parse_leading_command(arguments, _LAUNCHER_OPTION_SPEC)
    if launched.ambiguous or launched.token is None:
        return None
    return _verification_runner_tokens([launched.token, *launched.arguments])


def _normalized_script_manager_tokens(
    executable: str, arguments: list[str]
) -> list[str] | None:
    manager = parse_leading_command(
        arguments, _RUNNER_OPTION_SPECS.get(executable, OptionSpec())
    )
    if manager.ambiguous:
        return None
    manager_args = [] if manager.token is None else [manager.token, *manager.arguments]
    return [executable, *manager_args]


def _subcommand_runner_tokens(
    executable: str, arguments: list[str], normalized: list[str]
) -> list[str] | None:
    runner = parse_leading_command(
        arguments, _RUNNER_OPTION_SPECS.get(executable, OptionSpec())
    )
    if runner.ambiguous:
        return None
    runner_args = [] if runner.token is None else [runner.token, *runner.arguments]
    allowed_subcommands = _RUNNER_SUBCOMMANDS.get(executable, frozenset())
    if runner_args[:1] and runner_args[0] in allowed_subcommands:
        return [executable, *runner_args]
    if executable in _SCRIPT_RUNNERS and _script_runner_is_supported(normalized):
        return normalized
    return None


def _script_launcher_tokens(executable: str, arguments: list[str]) -> list[str] | None:
    if executable == "npm" and arguments[:1] in (["exec"], ["x"]):
        parsed = parse_leading_command(arguments[1:], _LAUNCHER_OPTION_SPEC)
    elif executable in {"pnpm", "yarn"} and arguments[:1] == ["dlx"]:
        parsed = parse_leading_command(arguments[1:], _LAUNCHER_OPTION_SPEC)
    elif executable == "bun" and arguments[:1] == ["x"]:
        parsed = parse_leading_command(arguments[1:], _LAUNCHER_OPTION_SPEC)
    else:
        return None
    if parsed.ambiguous or parsed.token is None:
        return None
    return [parsed.token, *parsed.arguments]


def _script_runner_is_supported(tokens: list[str]) -> bool:
    if tokens[1:2] and tokens[1] in _VERIFICATION_SCRIPTS:
        return True
    return (
        tokens[1:2] == ["run"]
        and bool(tokens[2:3])
        and tokens[2] in _VERIFICATION_SCRIPTS
    )


def _normalized_output_lines(output: str) -> tuple[str, ...]:
    return tuple(
        normalized
        for line in output.splitlines()
        if (
            normalized := " ".join(
                unicodedata.normalize("NFKC", line).casefold().split()
            )
        )
    )


_CHECK_RE = re.compile(
    r"^\s*#{2,6}\s+Check(?:\s+(?:[A-Z0-9]{1,40}|\([^)]{1,40}\)))?:"
    r"\s*(?P<check>\S.*)\s*$",
    re.IGNORECASE,
)
_CHECK_LIKE_RE = re.compile(r"^\s*#{2,6}\s+Check\b.*$", re.IGNORECASE)
_HEADING_RE = re.compile(r"^\s*#{2,6}\s+\S.*$")
_COMMAND_RE = re.compile(
    r"^\s*(?:\*\*)?Command(?: run)?:(?:\*\*)?(?P<inline>.*)$", re.IGNORECASE
)
_OUTPUT_RE = re.compile(
    r"^\s*(?:\*\*)?Output(?: observed)?:(?:\*\*)?(?P<inline>.*)$", re.IGNORECASE
)
_RESULT_RE = re.compile(
    r"^\s*(?:\*\*)?Result:\s*(?P<result>PASS|FAIL)(?:\*\*)?"
    r"(?:\s+(?:[-—:]|\().*)?\s*$",
    re.IGNORECASE,
)
_VERDICT_RE = re.compile(
    r"^\s*VERDICT:\s*(?P<verdict>PASS|FAIL|PARTIAL)\s*$", re.IGNORECASE
)
_FENCED_BLOCK_MIN_LINES = 2


def parse_verification_report(response: str) -> VerificationReport:
    lines = response.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    if not nonempty:
        raise VerificationReportError("verification report is empty")

    verdict_matches = [
        (index, match)
        for index, line in enumerate(lines)
        if (match := _VERDICT_RE.fullmatch(line)) is not None
    ]
    if len(verdict_matches) != 1:
        raise VerificationReportError(
            "verification report must contain exactly one verdict line"
        )

    verdict_index, verdict_match = verdict_matches[0]
    if verdict_index != nonempty[-1]:
        raise VerificationReportError("verification verdict must be the final line")

    verdict = VerificationVerdict(verdict_match.group("verdict").lower())
    evidence = _parse_evidence(lines[:verdict_index])
    if not evidence:
        raise VerificationReportError("verification report has no command evidence")
    if verdict == VerificationVerdict.PASS and any(
        item.result != VerificationVerdict.PASS for item in evidence
    ):
        raise VerificationReportError("PASS verdict contains a failed check")
    if verdict == VerificationVerdict.FAIL and all(
        item.result == VerificationVerdict.PASS for item in evidence
    ):
        raise VerificationReportError("FAIL verdict contains no failed check")
    return VerificationReport(verdict=verdict, evidence=tuple(evidence))


def _parse_evidence(lines: list[str]) -> list[CommandEvidence]:
    if any(
        _CHECK_LIKE_RE.fullmatch(line) and not _CHECK_RE.fullmatch(line)
        for line in lines
    ):
        raise VerificationReportError("invalid check heading")

    check_indexes = [
        index for index, line in enumerate(lines) if _CHECK_RE.fullmatch(line)
    ]
    evidence: list[CommandEvidence] = []
    for start in check_indexes:
        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if _HEADING_RE.fullmatch(lines[index])
            ),
            len(lines),
        )
        evidence.append(_parse_check(lines[start:end]))
    return evidence


def _parse_check(lines: list[str]) -> CommandEvidence:
    check_match = _CHECK_RE.fullmatch(lines[0])
    if check_match is None:
        raise VerificationReportError("invalid check heading")

    command_index, command_match = _find_heading(lines, _COMMAND_RE, 1, "Command run")
    output_index, output_match = _find_heading(
        lines, _OUTPUT_RE, command_index + 1, "Output observed"
    )
    result_indexes = [
        index
        for index in range(output_index + 1, len(lines))
        if _RESULT_RE.fullmatch(lines[index])
    ]
    if len(result_indexes) != 1:
        raise VerificationReportError(
            f"check '{check_match.group('check')}' must contain exactly one result"
        )
    result_index = result_indexes[0]

    command = _clean_value(
        command_match.group("inline"), lines[command_index + 1 : output_index]
    )
    output = _clean_value(
        output_match.group("inline"), lines[output_index + 1 : result_index]
    )
    if not command:
        raise VerificationReportError(
            f"check '{check_match.group('check')}' has an empty command"
        )
    if not output:
        raise VerificationReportError(
            f"check '{check_match.group('check')}' has empty observed output"
        )

    result_match = _RESULT_RE.fullmatch(lines[result_index])
    if result_match is None:
        raise VerificationReportError("invalid check result")
    return CommandEvidence(
        check=check_match.group("check").strip(),
        command=command,
        output=output,
        result=VerificationVerdict(result_match.group("result").lower()),
    )


def _find_heading(
    lines: list[str], pattern: re.Pattern[str], start: int, heading: str
) -> tuple[int, re.Match[str]]:
    matches = [
        (index, match)
        for index in range(start, len(lines))
        if (match := pattern.fullmatch(lines[index])) is not None
    ]
    if len(matches) != 1:
        raise VerificationReportError(
            f"each check must contain exactly one '{heading}' heading"
        )
    return matches[0]


def _clean_value(inline: str, block: list[str]) -> str:
    values = [value for value in (_clean_inline(inline), _clean_block(block)) if value]
    return "\n".join(values)


def _clean_inline(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("`") and cleaned.endswith("`"):
        return cleaned[1:-1].strip()
    return cleaned


def _clean_block(lines: list[str]) -> str:
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    if len(lines) >= _FENCED_BLOCK_MIN_LINES and lines[0].strip().startswith("```"):
        if lines[-1].strip() != "```":
            return ""
        lines = lines[1:-1]
    return "\n".join(line.strip() for line in lines).strip()


_DIRECT_VERIFICATION_ACTIONS = {
    "check",
    "exercise",
    "prove",
    "test",
    "testing",
    "validate",
    "validation",
    "verification",
    "verify",
}
_RUNNABLE_CHECKS = {
    "build",
    "checks",
    "integration",
    "lint",
    "pyright",
    "suite",
    "test",
    "tests",
    "typecheck",
    "validation",
    "verification",
}


def is_verification_todo(content: str) -> bool:
    normalized = unicodedata.normalize("NFKC", content).casefold()
    words = re.findall(r"[a-z0-9]+", normalized)
    if not words:
        return False
    if words[0] in _DIRECT_VERIFICATION_ACTIONS:
        return True
    return words[0] in {"execute", "perform", "run"} and any(
        word in _RUNNABLE_CHECKS for word in words[1:]
    )


def is_trivial_verification_note(note: str) -> bool:
    prefix, separator, reason = note.partition(":")
    return bool(separator and prefix.strip().casefold() == "trivial" and reason.strip())


_TRIVIAL_DOC_ROOTS = {"docs", "openwiki"}
_TRIVIAL_DOC_FILES = {"CHANGELOG.md", "CONTRIBUTING.md", "README.md"}
_TRIVIAL_DOC_SUFFIXES = {".md", ".rst", ".txt"}


def is_trivial_change_set(paths: list[str]) -> bool:
    if not paths:
        return False
    for raw_path in paths:
        path = PurePosixPath(raw_path)
        if ".." in path.parts or path.suffix.casefold() not in _TRIVIAL_DOC_SUFFIXES:
            return False
        if len(path.parts) == 1:
            if path.name not in _TRIVIAL_DOC_FILES:
                return False
            continue
        if path.parts[0] not in _TRIVIAL_DOC_ROOTS:
            return False
    return True
