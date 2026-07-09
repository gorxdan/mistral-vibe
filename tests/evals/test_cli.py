from __future__ import annotations

from pathlib import Path

from evals.cli import main
from evals.models import ComparisonReport, TaskCategory
from tests.evals._factories import make_dataset, make_run, make_trials
from vibe.core.utils.io import read_safe, write_safe


def _write_dataset(path: Path, dataset) -> None:
    write_safe(path, dataset.model_dump_json(indent=2) + "\n")


def _release_trials(harness_revision: str, total_cost_usd: float):
    return (
        *make_trials(harness_revision, task_name="core", total_cost_usd=total_cost_usd),
        *make_trials(
            harness_revision,
            task_name="policy",
            task_category=TaskCategory.POLICY,
            total_cost_usd=total_cost_usd,
        ),
        *make_trials(
            harness_revision,
            task_name="security",
            task_category=TaskCategory.SECURITY,
            total_cost_usd=total_cost_usd,
        ),
    )


def test_cli_writes_machine_readable_report_and_returns_success(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "reports" / "comparison.json"
    _write_dataset(
        baseline_path, make_dataset(_release_trials("baseline", total_cost_usd=10.0))
    )
    _write_dataset(
        candidate_path, make_dataset(_release_trials("candidate", total_cost_usd=6.0))
    )

    result = main((
        "--baseline",
        str(baseline_path),
        "--candidate",
        str(candidate_path),
        "--output",
        str(output_path),
        "--release-gate",
    ))

    report = ComparisonReport.model_validate_json(read_safe(output_path).text)
    assert result == 0
    assert report.passed
    assert report.release_gate


def test_cli_returns_nonzero_and_still_emits_failed_report(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "comparison.json"
    _write_dataset(
        baseline_path, make_dataset(make_trials("baseline", total_cost_usd=10.0))
    )
    candidate_runs = (
        make_run(
            1,
            harness_revision="candidate",
            verified=False,
            false_done=True,
            total_cost_usd=6.0,
        ),
        *make_trials("candidate", count=5, total_cost_usd=6.0)[1:],
    )
    _write_dataset(candidate_path, make_dataset(candidate_runs))

    result = main((
        "--baseline",
        str(baseline_path),
        "--candidate",
        str(candidate_path),
        "--output",
        str(output_path),
    ))

    report = ComparisonReport.model_validate_json(read_safe(output_path).text)
    assert result == 1
    assert not report.passed
    assert any(
        gate.name == "false_done_rate" and not gate.passed for gate in report.gates
    )


def test_cli_rejects_malformed_input_without_output(tmp_path: Path, capsys) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "comparison.json"
    write_safe(baseline_path, "{}")
    _write_dataset(
        candidate_path, make_dataset(make_trials("candidate", total_cost_usd=6.0))
    )

    result = main((
        "--baseline",
        str(baseline_path),
        "--candidate",
        str(candidate_path),
        "--output",
        str(output_path),
    ))

    assert result == 2
    assert not output_path.exists()
    assert "evaluation failed" in capsys.readouterr().err
