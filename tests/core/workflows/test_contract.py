from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.workflows.contract import ContractFailure, ContractSpec, verify_contract


@pytest.fixture
def root(tmp_path: Path) -> Path:
    tree = tmp_path / "wt"
    tree.mkdir()
    (tree / "auth.py").write_text("JWT_TOKEN = generate()\n# no plaintext\n")
    (tree / "PLAN.md").write_text("# Plan\nUse OAuth2 and JWT.\n" * 20)
    (tree / "big.bin").write_text("x" * 500)
    return tree


def _spec(**overrides: object) -> ContractSpec:
    return ContractSpec.model_validate(overrides)


def test_passes_when_all_checks_hold(root: Path) -> None:
    report = verify_contract(
        root,
        _spec(
            outputs=[
                {
                    "path": "auth.py",
                    "must_contain": ["JWT"],
                    "must_not_contain": ["password ="],
                }
            ],
            invariants=[{"grep": "JWT", "must_match": True}],
        ),
    )
    assert report.passed
    assert report.violations == []


def test_missing_output_fails(root: Path) -> None:
    report = verify_contract(root, _spec(outputs=[{"path": "missing.py"}]))
    assert not report.passed
    assert report.violations[0].category == "output"
    assert "missing.py" in report.violations[0].message


def test_missing_substring_fails(root: Path) -> None:
    report = verify_contract(
        root, _spec(outputs=[{"path": "auth.py", "must_contain": ["does-not-exist"]}])
    )
    assert not report.passed
    assert "does-not-exist" in report.violations[0].message


def test_forbidden_substring_fails(root: Path) -> None:
    report = verify_contract(
        root, _spec(outputs=[{"path": "auth.py", "must_not_contain": ["JWT"]}])
    )
    assert not report.passed
    assert "forbidden" in report.violations[0].message


def test_min_size_enforced(root: Path) -> None:
    report = verify_contract(
        root, _spec(outputs=[{"path": "auth.py", "min_size": 99999}])
    )
    assert not report.passed
    assert "min_size" in report.violations[0].message


def test_max_size_enforced(root: Path) -> None:
    report = verify_contract(root, _spec(outputs=[{"path": "big.bin", "max_size": 10}]))
    assert not report.passed
    assert "max_size" in report.violations[0].message


def test_must_match_regex(root: Path) -> None:
    report = verify_contract(
        root, _spec(outputs=[{"path": "auth.py", "must_match": [r"JWT_\w+"]}])
    )
    assert report.passed


def test_invariant_forbidden_pattern_present(root: Path) -> None:
    report = verify_contract(
        root,
        _spec(
            invariants=[
                {"grep": "JWT", "must_match": False, "description": "no JWT literals"}
            ]
        ),
    )
    assert not report.passed
    assert report.violations[0].category == "invariant"
    assert "no JWT literals" in report.violations[0].message


def test_invariant_required_pattern_absent(root: Path) -> None:
    report = verify_contract(
        root, _spec(invariants=[{"grep": "NEVER_PRESENT", "must_match": True}])
    )
    assert not report.passed


def test_invariant_invalid_regex_is_a_violation(root: Path) -> None:
    report = verify_contract(root, _spec(invariants=[{"grep": "(unclosed"}]))
    assert not report.passed
    assert "invalid grep pattern" in report.violations[0].message


def test_test_runs_and_passes(root: Path) -> None:
    (root / "ok.sh").write_text("#!/bin/sh\necho hello\n")
    (root / "ok.sh").chmod(0o755)
    report = verify_contract(
        root, _spec(tests=[{"command": "./ok.sh", "expect": "hello"}])
    )
    assert report.passed


def test_test_nonzero_exit_fails(root: Path) -> None:
    report = verify_contract(root, _spec(tests=[{"command": "false"}]))
    assert not report.passed
    assert report.violations[0].category == "test"
    assert "exited" in report.violations[0].message


def test_test_missing_expected_stdout_fails(root: Path) -> None:
    (root / "mute.sh").write_text("#!/bin/sh\necho somethingelse\n")
    (root / "mute.sh").chmod(0o755)
    report = verify_contract(
        root, _spec(tests=[{"command": "./mute.sh", "expect": "hello"}])
    )
    assert not report.passed


def test_output_path_escape_is_rejected(root: Path) -> None:
    report = verify_contract(root, _spec(outputs=[{"path": "../../../etc/passwd"}]))
    assert not report.passed
    assert "escapes the worktree root" in report.violations[0].message


def test_summary_truncates_long_violation_lists(root: Path) -> None:
    report = verify_contract(
        root,
        _spec(
            outputs=[
                {"path": "a.py", "must_contain": ["nope1"]},
                {"path": "b.py", "must_contain": ["nope2"]},
                {"path": "c.py", "must_contain": ["nope3"]},
                {"path": "d.py", "must_contain": ["nope4"]},
            ]
        ),
    )
    summary = report.summary()
    assert "+1 more" in summary


def test_contract_failure_is_falsy_and_dict_like(root: Path) -> None:
    report = verify_contract(root, _spec(outputs=[{"path": "absent.py"}]))
    failure = ContractFailure(report=report, error="contract failed")
    assert not failure
    assert failure.get("anything", "default") == "default"
    assert failure.report is report
    assert not failure.report.passed
