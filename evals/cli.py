from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from pydantic import ValidationError

from evals.gates import compare_datasets
from evals.models import ComparisonReport, EvaluationDataset
from vibe.core.utils.io import read_safe, write_durable


def load_dataset(path: Path) -> EvaluationDataset:
    return EvaluationDataset.model_validate_json(
        read_safe(path, raise_on_error=True).text
    )


def render_report(report: ComparisonReport) -> str:
    return report.model_dump_json(indent=2) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare offline harness evaluation datasets"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--release-gate",
        action="store_true",
        help=(
            "Require five trials per group plus policy and security fixture coverage"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        baseline = load_dataset(args.baseline)
        candidate = load_dataset(args.candidate)
        report = compare_datasets(baseline, candidate, release_gate=args.release_gate)
        rendered = render_report(report)
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_durable(args.output, rendered.encode("utf-8"), suffix=".eval.tmp")
    except (OSError, UnicodeError, ValidationError, ValueError) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 2
    return 0 if report.passed else 1


__all__ = ["build_parser", "load_dataset", "main", "render_report"]
