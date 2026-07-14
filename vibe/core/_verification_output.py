from __future__ import annotations

from collections.abc import Sequence
import json
import os
import subprocess
import sys

OUTPUT_PATTERN_LIMIT = 4_096
OUTPUT_PATTERN_COUNT_LIMIT = 64

_REGEX_TIMEOUT_SECONDS = 1.0
_REGEX_WORKER_SOURCE = r"""
import json
import re
import sys

request = json.load(sys.stdin)
output = request["output"]
diagnostics = []
for pattern in request["required"]:
    if re.search(pattern, output) is None:
        diagnostics.append(f"required output pattern did not match: {pattern}")
for pattern in request["forbidden"]:
    if re.search(pattern, output) is not None:
        diagnostics.append(f"forbidden output pattern matched: {pattern}")
pattern = request["count_pattern"]
minimum = request["minimum_count"]
if pattern is not None and minimum is not None:
    raw_counts = [match.group("count") for match in re.finditer(pattern, output)]
    try:
        counts = [int(count) for count in raw_counts]
    except (TypeError, ValueError):
        diagnostics.append("test count pattern produced a non-integer count")
    else:
        if not counts:
            diagnostics.append("test count pattern did not match output")
        else:
            distinct = sorted(set(counts))
            if len(distinct) > 1:
                rendered = ", ".join(str(count) for count in distinct)
                diagnostics.append(f"conflicting test counts observed: {rendered}")
            below = sorted({count for count in counts if count < minimum})
            for count in below:
                diagnostics.append(
                    f"observed test count {count} is below required minimum {minimum}"
                )
json.dump(diagnostics, sys.stdout)
"""


def validate_output_patterns(patterns: Sequence[str]) -> None:
    if len(patterns) > OUTPUT_PATTERN_COUNT_LIMIT:
        raise ValueError(
            f"at most {OUTPUT_PATTERN_COUNT_LIMIT} verification output patterns are allowed"
        )
    for pattern in patterns:
        if not pattern:
            raise ValueError("verification output patterns must be nonempty")
        if len(pattern) > OUTPUT_PATTERN_LIMIT:
            raise ValueError(
                f"verification output patterns may not exceed {OUTPUT_PATTERN_LIMIT} characters"
            )


def validate_custom_runner_contract(
    *,
    custom_runner: bool,
    executable_sha256: str | None,
    required_output_patterns: Sequence[str],
    test_count_pattern: str | None,
    minimum_test_count: int | None,
) -> None:
    if not custom_runner:
        return
    if (
        executable_sha256 is None
        or not required_output_patterns
        or test_count_pattern is None
        or minimum_test_count is None
    ):
        raise ValueError(
            "custom runner requires executable_sha256, required_output_patterns, "
            "and a positive test-count contract"
        )


def output_regex_diagnostics(
    *,
    required_patterns: Sequence[str],
    forbidden_patterns: Sequence[str],
    test_count_pattern: str | None,
    minimum_test_count: int | None,
    output: str,
) -> tuple[str, ...]:
    if (
        not required_patterns
        and not forbidden_patterns
        and test_count_pattern is None
        and minimum_test_count is None
    ):
        return ()
    request = json.dumps({
        "required": list(required_patterns),
        "forbidden": list(forbidden_patterns),
        "count_pattern": test_count_pattern,
        "minimum_count": minimum_test_count,
        "output": output,
    })
    result = _run_regex_worker(request)
    if isinstance(result, tuple):
        return result
    return _parse_regex_worker_output(result)


def _run_regex_worker(
    request: str,
) -> subprocess.CompletedProcess[str] | tuple[str, ...]:
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", _REGEX_WORKER_SOURCE],
            check=False,
            input=request,
            capture_output=True,
            text=True,
            timeout=_REGEX_TIMEOUT_SECONDS,
            env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": os.defpath},
        )
    except subprocess.TimeoutExpired:
        return ("verification output pattern evaluation timed out",)
    except OSError as exc:
        return (f"verification output pattern evaluation failed to start: {exc}",)
    if result.returncode != 0:
        return ("verification output pattern evaluation failed",)
    return result


def _parse_regex_worker_output(
    result: subprocess.CompletedProcess[str],
) -> tuple[str, ...]:
    try:
        parsed = json.loads(result.stdout)
    except (TypeError, ValueError):
        parsed = None
    if not isinstance(parsed, list) or any(
        not isinstance(item, str) for item in parsed
    ):
        return ("verification output pattern evaluation returned invalid data",)
    return tuple(item for item in parsed if isinstance(item, str))


__all__ = [
    "OUTPUT_PATTERN_COUNT_LIMIT",
    "OUTPUT_PATTERN_LIMIT",
    "output_regex_diagnostics",
    "validate_custom_runner_contract",
    "validate_output_patterns",
]
