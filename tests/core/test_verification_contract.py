from __future__ import annotations

import pytest

from vibe.core.verification_contract import (
    VerificationReportError,
    VerificationVerdict,
    is_trivial_change_set,
    is_trivial_verification_note,
    is_verification_command,
    is_verification_todo,
    parse_verification_report,
    report_evidence_was_observed,
    verification_observation_hashes,
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


def test_report_evidence_requires_host_observed_output() -> None:
    report = parse_verification_report(_report(output="5 passed in 0.12s"))
    observed = verification_observation_hashes(
        "uv run pytest tests/tools/test_todo.py", "5 passed in 0.12s\n", ""
    )

    assert report_evidence_was_observed(report, observed)
    assert not report_evidence_was_observed(
        parse_verification_report(_report(output="151 passed in 0.12s")), observed
    )


@pytest.mark.parametrize(
    "command",
    [
        "uv run pytest -q && echo passed",
        "uv run pytest -q | tail -1",
        "uv run pytest -q > result.txt",
        "uv run pytest -q\necho passed",
    ],
)
def test_compound_shell_commands_are_not_verification_evidence(command: str) -> None:
    assert not verification_observation_hashes(command, "5 passed\n", "")


@pytest.mark.parametrize(
    "command",
    [
        "echo '5 passed'",
        "printf '5 passed\\n'",
        "uv run python -c 'print(\"5 passed\")'",
        "sh ./candidate_owned_reporter.sh",
    ],
)
def test_arbitrary_output_producers_are_not_verification_evidence(command: str) -> None:
    assert not verification_observation_hashes(command, "5 passed\n", "")


@pytest.mark.parametrize(
    "command",
    [
        "./pytest",
        "./ruff check .",
        "tools/pytest",
        r".\pytest.exe",
        r"tools\\pytest.exe",
        "pytest.cmd",
        "pytest.bat",
        "pytest.com",
        "uv run ./pytest",
        "npx --no-install ./pytest",
        "npm exec -- pytest.cmd",
    ],
)
def test_candidate_owned_runner_names_are_not_verification_evidence(
    command: str,
) -> None:
    assert not is_verification_command(command)
    assert not verification_observation_hashes(command, "5 passed\n", "")


def test_windows_bash_output_never_becomes_verification_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("vibe.core.verification_contract.is_windows", lambda: True)

    assert not verification_observation_hashes("pytest", "5 passed\n", "")


@pytest.mark.parametrize(
    "command",
    [
        "uv run --offline pytest",
        "uv --project . run --frozen pytest",
        "python -I -m pytest",
        "npm --silent test",
        "ruff --config pyproject.toml check .",
        "make -j test",
        "npx --no-install pytest",
        "uvx ruff check .",
        "npm exec -- pytest",
        "pnpm dlx vitest",
        "cargo +nightly test",
        "cargo --color always test",
        "dotnet --diagnostics test --no-restore",
        "make -j 4 test",
    ],
)
def test_verification_recognition_normalizes_common_runner_options(
    command: str,
) -> None:
    assert is_verification_command(command)


@pytest.mark.parametrize(
    "command",
    [
        "cargo run",
        "dotnet publish --no-restore",
        "dotnet run --no-restore",
        "go generate ./...",
        "npm run build",
    ],
)
def test_project_execution_is_not_accepted_as_verification_evidence(
    command: str,
) -> None:
    assert not is_verification_command(command)
    assert not verification_observation_hashes(
        command, "candidate says checks passed\n", ""
    )


def test_dotnet_test_requires_a_nonzero_observed_test_count() -> None:
    command = "dotnet test src/Fcc.Core/Fcc.Core.csproj"

    assert not verification_observation_hashes(
        command, "Build succeeded.\n    0 Warning(s)\n    0 Error(s)\n", ""
    )
    assert not verification_observation_hashes(
        command, "Passed! - Failed: 0, Passed: 0, Skipped: 0, Total: 0\n", ""
    )
    assert verification_observation_hashes(
        command, "Passed! - Failed: 0, Passed: 151, Skipped: 0, Total: 151\n", ""
    )
    assert not verification_observation_hashes(
        command,
        (
            "candidate says Total: 151\n"
            "Passed! - Failed: 0, Passed: 0, Skipped: 0, Total: 0\n"
        ),
        "",
    )


@pytest.mark.parametrize(
    "command",
    [
        "pytest --version",
        "uv run --help pytest",
        "uv run pytest --collect-only",
        "python -m pytest --fixtures",
        "python3 -m unittest --help",
        "dotnet test --list-tests",
        "go test -list TestFeature ./...",
        "go test -run '^$' ./...",
        "cargo test --no-run",
        "cargo test -- --list",
        "npm run test -- --help",
        "tox --showconfig",
        "nox --list",
        "ruff check --exit-zero .",
        "make test -n",
        "npm --help test",
    ],
)
def test_nonexecuting_verification_modes_are_not_evidence(command: str) -> None:
    assert not verification_observation_hashes(command, "command succeeded\n", "")


@pytest.mark.parametrize(
    "command",
    [
        "dotnet build src/Fcc.Core/Fcc.Core.csproj",
        "dotnet restore src/Fcc.Core/Fcc.Core.csproj",
        "pnpm run lint",
        "yarn run typecheck",
        "cargo build",
        "cargo check",
        "go build ./...",
        "python3.12 -m pyright",
        "python3.12 -m ruff check .",
    ],
)
def test_direct_build_and_check_commands_are_eligible(command: str) -> None:
    assert verification_observation_hashes(command, "command succeeded\n", "")


@pytest.mark.parametrize(
    ("command", "output"),
    [
        ("pytest", "no tests ran in 0.01s"),
        ("python -m unittest", "Ran 0 tests in 0.000s\n\nOK"),
        ("cargo test", "test result: ok. 0 passed; 0 failed; 0 ignored"),
        ("go test ./...", "ok  example.test/pkg  0.002s"),
        ("npm test", "checks passed"),
        ("make test", "nothing to do"),
        ("tox", "congratulations :)"),
    ],
)
def test_test_runners_require_positive_execution_evidence(
    command: str, output: str
) -> None:
    assert not verification_observation_hashes(command, output, "")


@pytest.mark.parametrize(
    ("command", "output"),
    [
        ("pytest", "5 passed in 0.12s"),
        ("python -m unittest", "Ran 3 tests in 0.002s\n\nOK"),
        ("cargo test", "test result: ok. 2 passed; 0 failed; 0 ignored"),
        ("go test -v ./...", "--- PASS: TestFeature (0.00s)\nPASS"),
    ],
)
def test_test_runners_accept_positive_builtin_counts(command: str, output: str) -> None:
    assert verification_observation_hashes(command, output, "")


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


def test_parse_accepts_alphanumeric_check_labels_and_heading_aliases() -> None:
    checks = []
    for label in ("A", "B2", "C", "D4", "E"):
        checks.append(
            f"### Check {label}: probe {label}\n"
            f"**Command:** `pytest -k probe_{label.lower()}`\n"
            "**Output:** 1 passed\n"
            "**Result: PASS**"
        )

    report = parse_verification_report("\n\n".join(checks) + "\n\nVERDICT: PASS")

    assert [item.check for item in report.evidence] == [
        "probe A",
        "probe B2",
        "probe C",
        "probe D4",
        "probe E",
    ]
    assert report.evidence[1].command == "pytest -k probe_b2"
    assert report.evidence[1].output == "1 passed"


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
    "heading",
    [
        "### Check A - missing colon",
        "### Check alpha beta: unsupported label",
        "### Check:",
    ],
)
def test_parse_rejects_malformed_check_like_headings(heading: str) -> None:
    response = _report().replace("Verification notes before the evidence.", heading)

    with pytest.raises(VerificationReportError, match="invalid check heading"):
        parse_verification_report(response)


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
