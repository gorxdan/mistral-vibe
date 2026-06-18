from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import time
from typing import Any

from textual.widget import Widget

from vibe.cli.textual_ui.widgets.messages import ErrorMessage, UserCommandMessage
from vibe.core.logger import logger
from vibe.core.workflows.models import (
    BudgetSnapshot,
    PhaseReport,
    WorkflowResult,
    WorkflowRunSnapshot,
    WorkflowStatus,
)
from vibe.core.workflows.runtime import WorkflowRuntime

_MIN_PARTS_FOR_STOP = 2


@dataclass
class WorkflowRunEntry:
    run_id: str
    script_source: str
    started_at: float
    runtime: WorkflowRuntime
    task: asyncio.Task[WorkflowResult] | None = None
    result: WorkflowResult | None = None
    error: str | None = None

    @property
    def status(self) -> WorkflowStatus:
        if self.result is not None:
            return self.result.run.status
        if self.error is not None:
            return WorkflowStatus.FAILED
        if self.task is not None and self.task.done():
            return WorkflowStatus.FAILED
        return WorkflowStatus.RUNNING

    @property
    def elapsed(self) -> float:
        if self.result is not None and self.result.run.finished_at:
            return self.result.run.finished_at - self.started_at
        return time.monotonic() - self.started_at

    @property
    def agent_count(self) -> int:
        if self.result is not None:
            return self.result.run.agent_count
        return self.runtime._agent_count

    @property
    def tokens_total(self) -> int:
        if self.result is not None:
            return self.result.run.tokens_total
        return sum(p.tokens_total for p in self.runtime._phases.values())

    @property
    def phases(self) -> list[str]:
        if self.result is not None:
            return [p.name for p in self.result.run.phases]
        return list(self.runtime._phase_order)

    @property
    def phase_reports(self) -> list[PhaseReport]:
        """Ordered phase reports with agent results, live during execution."""
        if self.result is not None:
            return self.result.run.phases
        return [
            self.runtime._phases[name]
            for name in self.runtime._phase_order
            if name in self.runtime._phases
        ]

    @property
    def budget_snapshot(self) -> BudgetSnapshot:
        if self.result is not None:
            return self.result.run.budget
        return self.runtime._budget.snapshot()


def _format_run_list(runs: list[WorkflowRunEntry]) -> str:
    if not runs:
        return "No workflow runs."

    rows = [
        "| ID | Status | Agents | Tokens | Elapsed | Phases |",
        "|----|--------|--------|--------|---------|--------|",
    ]
    for entry in runs:
        elapsed_s = f"{entry.elapsed:.1f}s"
        phases = ", ".join(entry.phases) or "(none)"
        rows.append(
            f"| `{entry.run_id}` | {entry.status.value} | {entry.agent_count} | "
            f"{entry.tokens_total} | {elapsed_s} | {phases} |"
        )
    return "\n".join(rows)


class WorkflowRunner:
    def __init__(
        self,
        *,
        mount: Callable[[Widget], Awaitable[None]],
        on_complete: Callable[[WorkflowResult], Awaitable[None]] | None = None,
        persist_callback: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._mount = mount
        self._on_complete = on_complete
        self._persist_callback = persist_callback
        self._runs: list[WorkflowRunEntry] = []
        self._next_id = 1

    @property
    def runs(self) -> list[WorkflowRunEntry]:
        return list(self._runs)

    @property
    def active_runs(self) -> list[WorkflowRunEntry]:
        return [r for r in self._runs if r.status == WorkflowStatus.RUNNING]

    @property
    def completed_runs(self) -> list[WorkflowRunEntry]:
        return [r for r in self._runs if r.status != WorkflowStatus.RUNNING]

    def launch(
        self, script_source: str, *, runtime: WorkflowRuntime, args: Any = None
    ) -> str:
        run_id = f"wf-{self._next_id}"
        self._next_id += 1

        entry = WorkflowRunEntry(
            run_id=run_id,
            script_source=script_source,
            started_at=time.monotonic(),
            runtime=runtime,
        )

        runtime.set_event_sink(self._make_event_sink(run_id))

        entry.task = asyncio.create_task(self._run_workflow(entry, args))
        self._runs.append(entry)
        return run_id

    async def _run_workflow(self, entry: WorkflowRunEntry, args: Any) -> WorkflowResult:
        try:
            result = await entry.runtime.run(entry.script_source, args=args)
            entry.result = result
            if self._on_complete:
                await self._on_complete(result)
            if self._persist_callback:
                await self._persist_callback()
            return result
        except asyncio.CancelledError:
            entry.error = "Cancelled"
            raise
        except Exception as e:
            entry.error = str(e)
            logger.error("Workflow run failed", exc_info=e)
            await self._mount(ErrorMessage(f"Workflow `{entry.run_id}` failed: {e}"))
            raise

    def get_snapshot(self, run_id: str) -> WorkflowRunSnapshot | None:
        entry = self._find_run(run_id)
        if entry is None:
            return None
        return entry.runtime.snapshot(
            run_id=entry.run_id, script_source=entry.script_source, args=None
        )

    def resume(
        self, run_id: str, snapshot: WorkflowRunSnapshot, *, runtime: WorkflowRuntime
    ) -> str:
        runtime.restore_from_snapshot(snapshot)
        new_id = f"wf-{self._next_id}"
        self._next_id += 1

        entry = WorkflowRunEntry(
            run_id=new_id,
            script_source=snapshot.script_source,
            started_at=time.monotonic(),
            runtime=runtime,
        )

        runtime.set_event_sink(self._make_event_sink(new_id))
        entry.task = asyncio.create_task(self._run_workflow(entry, snapshot.args))
        self._runs.append(entry)
        return new_id

    @staticmethod
    def _make_event_sink(run_id: str) -> Callable[[str], None]:
        def sink(msg: str) -> None:
            logger.info("workflow %s: %s", run_id, msg)

        return sink

    async def stop(self, run_id: str) -> bool:
        entry = self._find_run(run_id)
        if entry is None or entry.task is None:
            return False
        if entry.task.done():
            return False
        entry.task.cancel()
        try:
            await entry.task
        except asyncio.CancelledError:
            pass
        return True

    async def stop_all(self) -> None:
        for entry in list(self._runs):
            if entry.task is not None and not entry.task.done():
                entry.task.cancel()
                try:
                    await entry.task
                except asyncio.CancelledError:
                    pass

    def _find_run(self, run_id: str) -> WorkflowRunEntry | None:
        return next((r for r in self._runs if r.run_id == run_id), None)

    async def handle_command(self, cmd_args: str) -> Widget:  # noqa: PLR0911
        cmd_args = cmd_args.strip()
        if not cmd_args or cmd_args in {"list", "ls"}:
            return UserCommandMessage(_format_run_list(self._runs))

        parts = cmd_args.split(None, 1)
        verb = parts[0].lower()

        match verb:
            case "stop" | "cancel" | "kill":
                if len(parts) < _MIN_PARTS_FOR_STOP:
                    return ErrorMessage("Usage: /workflows stop <run-id>")
                target_id = parts[1].strip()
                if target_id == "all":
                    await self.stop_all()
                    return UserCommandMessage("Stopped all workflow runs.")
                stopped = await self.stop(target_id)
                if stopped:
                    return UserCommandMessage(f"Stopped workflow `{target_id}`.")
                return ErrorMessage(
                    f"Could not stop `{target_id}` — not found or already finished."
                )

            case "snapshot" | "snap":
                if len(parts) < _MIN_PARTS_FOR_STOP:
                    return ErrorMessage("Usage: /workflows snapshot <run-id>")
                target_id = parts[1].strip()
                snap = self.get_snapshot(target_id)
                if snap is None:
                    return ErrorMessage(f"Run `{target_id}` not found.")
                return UserCommandMessage(
                    f"Snapshot of `{target_id}`: {snap.cached_count} cached results, "
                    f"{snap.budget_spent} tokens spent, status: {snap.status.value}"
                )

            case _:
                return ErrorMessage(
                    f"Unknown /workflows subcommand: `{verb}`.\n"
                    "Usage: /workflows [list|stop <id|all>|snapshot <id>]"
                )
