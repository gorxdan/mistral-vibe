from __future__ import annotations

import pytest

from vibe.core.verification_contract import (
    VerificationReportError,
    VerificationVerdict,
    is_trivial_change_set,
    is_trivial_verification_note,
    is_verification_todo,
    parse_verification_report,
)


def _report(
    *,
    result: str = "PASS",
    verdict: str = "PASS",
    command: str = "uv run pytest tests/tools/test_todo.py",
    output: str = "5 passed in 0.12s",
) -> str:
    return (
        "Verification notes before the evidence.\n\n"
        "### Check: focused tests\n"
        "**Command run:**\n"
        f"  {command}\n"
        "**Output observed:**\n"
        f"  {output}\n"
        f"**Result: {result}**\n\n"
        f"VERDICT: {verdict}"
    )


def test_parse_pass_report_with_command_evidence() -> None:
    report = parse_verification_report(_report())

    assert report.verdict == VerificationVerdict.PASS
    assert report.passed is True
    assert report.evidence[0].check == "focused tests"
    assert report.evidence[0].command.startswith("uv run pytest")
    assert report.evidence[0].output == "5 passed in 0.12s"


def test_parse_accepts_fenced_multiline_command() -> None:
    report = parse_verification_report(
        _report(command="```bash\nuv run pytest\n```", output="ok")
    )

    assert report.evidence[0].command == "uv run pytest"


def test_parse_accepts_common_verifier_markdown_variants() -> None:
    report = parse_verification_report(
        "### Check 1: focused tests\n"
        "**Command run:** `uv run pytest -q`\n"
        "**Output observed:** 8 passed\n"
        "**Result: PASS** — Expected a green suite and got one.\n\n"
        "VERDICT: PASS"
    )

    assert report.evidence[0].check == "focused tests"
    assert report.evidence[0].command == "uv run pytest -q"
    assert report.evidence[0].output == "8 passed"


def test_parse_ignores_non_evidence_sections_after_checks() -> None:
    report = parse_verification_report(
        "### Check: focused tests\n"
        "**Command run:** `uv run pytest -q`\n"
        "**Output observed:** 8 passed\n"
        "**Result: PASS**\n\n"
        "### Check (adversarial): no-op behavior\n"
        "**Command run:** `uv run pytest -k noop`\n"
        "**Output observed:** 1 passed\n"
        "**Result: PASS**\n\n"
        "### Observations\n"
        "No blocking findings.\n\n"
        "VERDICT: PASS"
    )

    assert len(report.evidence) == 2
    assert report.evidence[1].check == "no-op behavior"


@pytest.mark.parametrize(
    ("response", "error"),
    [
        ("VERDICT: PASS", "no command evidence"),
        (_report(command="   "), "empty command"),
        (_report(output="   "), "empty observed output"),
        (_report() + "\nafter", "final line"),
        (_report() + "\nVERDICT: FAIL", "exactly one verdict"),
        (_report(result="FAIL"), "PASS verdict contains a failed check"),
        (_report(verdict="FAIL"), "FAIL verdict contains no failed check"),
    ],
)
def test_parse_rejects_invalid_or_inconsistent_reports(
    response: str, error: str
) -> None:
    with pytest.raises(VerificationReportError, match=error):
        parse_verification_report(response)


def test_parse_accepts_fail_when_a_check_failed() -> None:
    report = parse_verification_report(_report(result="FAIL", verdict="FAIL"))

    assert report.verdict == VerificationVerdict.FAIL
    assert report.passed is False


@pytest.mark.parametrize(
    "content",
    [
        "Verify the implementation end-to-end",
        "VERIFY: adversarial behavior",
        "Validation of the API",
        "Run focused tests",
        "Execute pyright and lint",
        "Perform independent verification",
        "Check the implementation",
    ],
)
def test_verification_todo_recognizes_normalized_actions(content: str) -> None:
    assert is_verification_todo(content) is True


@pytest.mark.parametrize(
    "content",
    [
        "Add the verifier profile",
        "Fix verification parser",
        "Document test commands",
        "Investigate verification failures",
    ],
)
def test_verification_todo_rejects_incidental_keywords(content: str) -> None:
    assert is_verification_todo(content) is False


def test_trivial_note_requires_an_explicit_reason() -> None:
    assert is_trivial_verification_note("trivial: docs only") is True
    assert is_trivial_verification_note(" TRIVIAL : read-only ") is True
    assert is_trivial_verification_note("trivial:") is False
    assert is_trivial_verification_note("not trivial: typo") is False


@pytest.mark.parametrize(
    "paths",
    [["README.md"], ["docs/design/harness.md", "openwiki/config-models/overview.md"]],
)
def test_trivial_change_set_accepts_documentation_only(paths: list[str]) -> None:
    assert is_trivial_change_set(paths) is True


@pytest.mark.parametrize(
    "paths",
    [
        [],
        ["vibe/core/agent_loop.py"],
        ["docs/example.py"],
        ["vibe/core/prompts/agent.md"],
        ["docs/../vibe/core/agent_loop.txt"],
    ],
)
def test_trivial_change_set_rejects_unknown_or_runtime_changes(
    paths: list[str],
) -> None:
    assert is_trivial_change_set(paths) is False
