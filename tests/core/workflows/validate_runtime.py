"""T8: Live validation harness for the workflow runtime.

Spawns 3 explore agents in parallel on different subdirs of the vibe repo,
merges their summaries, and prints the result. Validates:
  - parallel execution works
  - results merge correctly
  - token/cost totals are sane
  - no semaphore deadlock

Usage:
  MISTRAL_API_KEY=... uv run python tests/core/workflows/validate_runtime.py

Or from the repo root:
  MISTRAL_API_KEY=... uv run python -m tests.core.workflows.validate_runtime
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import time

from vibe.core.workflows.runtime import WorkflowRuntime

REPO_ROOT = Path(__file__).resolve().parents[3]

SCRIPT = """
import json

async def main():
    phase("Explore")

    dirs = ["vibe/core/tools", "vibe/core/llm", "vibe/cli/textual_ui"]

    async def explore_dir(d):
        return await agent(
            f"Explore the {d} directory in this codebase. "
            f"Summarize the architecture: what are the main modules, "
            f"what patterns are used, what are the key abstractions. "
            f"Keep it under 200 words.",
            label=f"explore:{d}",
            phase="Explore",
        )

    summaries = await pipeline(dirs, explore_dir)

    phase("Synthesize")
    report = await agent(
        "Synthesize these three directory summaries into a single "
        "architecture overview. Highlight cross-cutting patterns and "
        "dependencies between the three areas.\\n\\n"
        + "\\n\\n---\\n\\n".join(summaries),
        label="synthesize",
        phase="Synthesize",
    )

    return {"report": report, "dir_count": len(summaries)}
"""


async def main() -> int:
    print(f"Repo root: {REPO_ROOT}")
    print("Starting workflow runtime validation...")
    print()

    runtime = WorkflowRuntime(max_concurrent=4, max_agents=100, budget_total=500_000)

    events: list[str] = []
    runtime.set_event_sink(lambda msg: events.append(msg))

    start = time.monotonic()
    result = await runtime.run(SCRIPT)
    elapsed = time.monotonic() - start

    print(f"Status: {result.run.status.value}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Agents spawned: {result.run.agent_count}")
    print(f"Tokens: {result.run.tokens_total}")
    print(f"Budget spent: {result.run.budget.spent}")
    print(f"Budget reserved: {result.run.budget.reserved}")
    print()

    for phase in result.run.phases:
        print(f"Phase: {phase.name}")
        for ar in phase.agent_results:
            status = "completed" if ar.completed else "FAILED"
            print(f"  {ar.label}: {ar.tokens_total} tokens [{status}]")
            if ar.error:
                print(f"    error: {ar.error}")
        print()

    print("Events:")
    for e in events:
        print(f"  {e}")
    print()

    if result.return_value:
        print("Report:")
        print(result.return_value.get("report", "(no report)"))
    else:
        print("No return value (workflow may have failed)")
        print(f"Summary: {result.summary}")
        return 1

    assert result.run.status.value == "completed", (
        f"Expected completed, got {result.run.status.value}"
    )
    assert result.run.agent_count == 4, (
        f"Expected 4 agents, got {result.run.agent_count}"
    )
    assert result.run.budget.reserved == 0, (
        f"Expected 0 reserved, got {result.run.budget.reserved}"
    )
    assert result.run.budget.spent > 0, "Expected non-zero spent"

    print()
    print("All assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
