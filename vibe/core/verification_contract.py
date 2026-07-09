from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import PurePosixPath
import re
import unicodedata


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


_CHECK_RE = re.compile(r"^\s*#{2,6}\s+Check:\s*(?P<check>\S.*)\s*$", re.IGNORECASE)
_COMMAND_RE = re.compile(r"^\s*(?:\*\*)?Command run:(?:\*\*)?\s*$", re.IGNORECASE)
_OUTPUT_RE = re.compile(r"^\s*(?:\*\*)?Output observed:(?:\*\*)?\s*$", re.IGNORECASE)
_RESULT_RE = re.compile(
    r"^\s*(?:\*\*)?Result:\s*(?P<result>PASS|FAIL)(?:\*\*)?\s*$", re.IGNORECASE
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
    check_indexes = [
        index for index, line in enumerate(lines) if _CHECK_RE.fullmatch(line)
    ]
    evidence: list[CommandEvidence] = []
    for position, start in enumerate(check_indexes):
        end = (
            check_indexes[position + 1]
            if position + 1 < len(check_indexes)
            else len(lines)
        )
        evidence.append(_parse_check(lines[start:end]))
    return evidence


def _parse_check(lines: list[str]) -> CommandEvidence:
    check_match = _CHECK_RE.fullmatch(lines[0])
    if check_match is None:
        raise VerificationReportError("invalid check heading")

    command_index = _find_heading(lines, _COMMAND_RE, 1, "Command run")
    output_index = _find_heading(
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

    command = _clean_block(lines[command_index + 1 : output_index])
    output = _clean_block(lines[output_index + 1 : result_index])
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
) -> int:
    indexes = [
        index for index in range(start, len(lines)) if pattern.fullmatch(lines[index])
    ]
    if len(indexes) != 1:
        raise VerificationReportError(
            f"each check must contain exactly one '{heading}' heading"
        )
    return indexes[0]


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
