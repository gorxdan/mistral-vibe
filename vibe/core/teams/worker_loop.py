"""Claim-loop for long-lived team workers (VIBE_TEAM_WORKER=1).

One-shot teammates still run a single ``-p`` prompt and exit. Workers poll
TaskStore, claim available work, run one programmatic turn per task, and
complete. Crashed workers leave IN_PROGRESS claims that ``reclaim_stale``
returns to PENDING.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import os
from pathlib import Path

from vibe.core.logger import logger
from vibe.core.tasking import TaskOutcome, TaskOutcomeStatus
from vibe.core.tasking._policy import BoundTaskContract, TaskContractError
from vibe.core.tasking._process_context import task_brief_hash
from vibe.core.teams._retry_context import bounded_retry_context
from vibe.core.teams._structured_attempt import evaluate_structured_attempt
from vibe.core.teams.mailbox import Mailbox
from vibe.core.teams.models import MessageKind, Task, TaskStatus
from vibe.core.teams.task_store import DEFAULT_TASK_LEASE_S, TaskStore
from vibe.core.usage._session import (
    SpendAdmissionBlockedError,
    SpendBudgetExceededError,
)

# Env set by TeamManager when spawn_teammate(worker=True).
TEAM_WORKER_ENV = "VIBE_TEAM_WORKER"
TEAM_LEASE_ENV = "VIBE_TEAM_LEASE_S"

# Idle poll when the queue is empty (seconds).
_IDLE_POLL_S = 2.0
# Max consecutive empty polls before exit when no SHUTDOWN (bounded worker).
# 0 = run until SHUTDOWN / process signal only.
_DEFAULT_IDLE_EXITS = 0


@dataclass(frozen=True, slots=True)
class WorkerTaskAttempt:
    summary: str | None
    contract: BoundTaskContract | None = None
    outcome: TaskOutcome | None = None


TaskRunner = Callable[[Task], Awaitable[WorkerTaskAttempt]]


def is_team_worker() -> bool:
    return os.environ.get(TEAM_WORKER_ENV) == "1"


def team_lease_s() -> float:
    raw = os.environ.get(TEAM_LEASE_ENV)
    if raw is None or raw == "":
        return DEFAULT_TASK_LEASE_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TASK_LEASE_S
    return value if value > 0 else DEFAULT_TASK_LEASE_S


def _team_identity() -> tuple[Path, str] | None:
    team_dir = os.environ.get("VIBE_TEAM_DIR")
    name = os.environ.get("VIBE_TEAMMATE_NAME")
    if not team_dir or not name:
        return None
    return Path(team_dir), name


def _shutdown_pending(mailbox: Mailbox, name: str) -> bool:
    for msg in mailbox.get_unread(name):
        if msg.kind is MessageKind.SHUTDOWN:
            mailbox.read(name, mark_read=True)
            return True
    return False


def worker_task_prompt(task: Task) -> str:
    if task.structured:
        retry_context = bounded_retry_context(task.outcome)
        retry_section = f"\n\nRetry context:\n{retry_context}" if retry_context else ""
        return (
            "You are a team worker. Complete this claimed task and then stop "
            "(the worker loop will claim the next one).\n\n"
            f"Task id: {task.id}\n"
            f"Structured contract:\n{task.prompt}"
            f"{retry_section}\n\n"
            "Do not call team complete_task. The harness owns completion and "
            "will run the bound acceptance checks. End your response with exactly "
            "one terminal line: TASK_OUTCOME: SUCCEEDED, TASK_OUTCOME: FAILED, "
            "TASK_OUTCOME: BLOCKED, or TASK_OUTCOME: RETRYABLE. Do not claim "
            "other tasks yourself."
        )
    return (
        f"You are a team worker. Complete this claimed task and then stop "
        f"(the worker loop will claim the next one).\n\n"
        f"Task id: {task.id}\n"
        f"Description:\n{task.prompt}\n\n"
        f"When finished, call the team tool action=complete_task with "
        f"task_id={task.id!r} and a short result summary. Do not claim other "
        f"tasks yourself — the harness does that."
    )


async def run_team_worker_loop(
    run_task: TaskRunner,
    *,
    idle_poll_s: float = _IDLE_POLL_S,
    max_idle_rounds: int = _DEFAULT_IDLE_EXITS,
    lease_s: float | None = None,
) -> str | None:
    """Drive available_tasks → claim → run_task → complete until stop.

    ``run_task`` receives the claimed Task and returns the worker attempt.
    """
    identity = _team_identity()
    if identity is None:
        raise RuntimeError(
            "Team worker requires VIBE_TEAM_DIR and VIBE_TEAMMATE_NAME in the env."
        )
    team_dir, name = identity
    store = TaskStore(team_dir)
    mailbox = Mailbox(team_dir)
    lease = team_lease_s() if lease_s is None else lease_s
    logger.info(
        "Team worker %s starting (lease_s=%s team_dir=%s)", name, lease, team_dir
    )
    return await _worker_main(
        run_task,
        store=store,
        mailbox=mailbox,
        name=name,
        lease=lease,
        idle_poll_s=idle_poll_s,
        max_idle_rounds=max_idle_rounds,
    )


async def _worker_main(
    run_task: TaskRunner,
    *,
    store: TaskStore,
    mailbox: Mailbox,
    name: str,
    lease: float,
    idle_poll_s: float,
    max_idle_rounds: int,
) -> str | None:
    idle_rounds = 0
    last_summary: str | None = None
    while True:
        if _shutdown_pending(mailbox, name):
            logger.info("Team worker %s received SHUTDOWN", name)
            break

        reclaimed = await asyncio.to_thread(store.reclaim_stale, lease)
        if reclaimed:
            logger.info(
                "Team worker %s reclaimed stale tasks: %s", name, ",".join(reclaimed)
            )

        await asyncio.to_thread(store.reload)
        available = await asyncio.to_thread(store.get_available_tasks)
        if not available:
            idle_rounds += 1
            if max_idle_rounds > 0 and idle_rounds >= max_idle_rounds:
                logger.info(
                    "Team worker %s exiting after %s idle rounds", name, idle_rounds
                )
                break
            await asyncio.sleep(idle_poll_s)
            continue

        idle_rounds = 0
        claimed = await asyncio.to_thread(store.claim_task, available[0].id, name)
        if claimed is None:
            continue

        last_summary = await _run_claimed(run_task, store, name, claimed)
        await asyncio.to_thread(store.reload)
        current = store.get_task(claimed.id)
        if (
            current is not None
            and current.status is TaskStatus.PENDING
            and current.outcome is not None
            and current.outcome.retryable
        ):
            logger.info("Team worker %s queued %s for retry", name, claimed.id)
            break
    return last_summary


async def _run_claimed(
    run_task: TaskRunner, store: TaskStore, name: str, claimed: Task
) -> str | None:
    logger.info("Team worker %s claimed %s", name, claimed.id)
    try:
        attempt = await run_task(claimed)
    except (
        SpendAdmissionBlockedError,
        SpendBudgetExceededError,
        TaskContractError,
    ) as e:
        logger.warning(
            "Team worker %s blocked deterministic task %s: %s", name, claimed.id, e
        )
        if claimed.structured:
            outcome = TaskOutcome(
                status=TaskOutcomeStatus.BLOCKED,
                summary="Structured task cannot proceed under its bound contract",
                diagnostics=[f"{type(e).__name__}: {str(e)[:1_000]}"],
                manifest=claimed.brief.manifest if claimed.brief else None,
            )
            await asyncio.to_thread(
                store.complete_task, claimed.id, outcome, actor=name, authoritative=True
            )
        return None
    except Exception as e:
        logger.exception(
            "Team worker %s failed while running task %s", name, claimed.id
        )
        if claimed.structured:
            outcome = TaskOutcome(
                status=TaskOutcomeStatus.RETRYABLE,
                summary="Structured worker execution failed",
                diagnostics=[f"{type(e).__name__}: {str(e)[:1_000]}"],
                manifest=claimed.brief.manifest if claimed.brief else None,
            )
            await asyncio.to_thread(
                store.complete_task, claimed.id, outcome, actor=name, authoritative=True
            )
        return None

    await asyncio.to_thread(store.reload)
    current = store.get_task(claimed.id)
    if current is None or current.status != TaskStatus.IN_PROGRESS:
        return attempt.summary
    if claimed.structured:
        return await _complete_structured_attempt(store, name, claimed, attempt)
    result_text = (attempt.summary or "").strip() or "completed by worker loop"
    completed = await asyncio.to_thread(
        store.complete_task, claimed.id, result_text, actor=name
    )
    if completed is None:
        logger.warning(
            "Team worker %s could not complete %s (lost claim?)", name, claimed.id
        )
        return attempt.summary
    return completed.result


async def _complete_structured_attempt(
    store: TaskStore, name: str, claimed: Task, attempt: WorkerTaskAttempt
) -> str | None:
    brief = claimed.brief
    if brief is None:
        return attempt.summary
    contract = attempt.contract
    if contract is None or contract.brief_hash != task_brief_hash(brief):
        outcome = TaskOutcome(
            status=TaskOutcomeStatus.RETRYABLE,
            summary="Structured worker contract was not bound to the claimed task",
            diagnostics=["Missing or mismatched host-bound task contract"],
            manifest=brief.manifest,
        )
    elif attempt.outcome is not None:
        if attempt.outcome.manifest != brief.manifest:
            outcome = TaskOutcome(
                status=TaskOutcomeStatus.RETRYABLE,
                summary="Structured worker outcome changed its manifest identity",
                diagnostics=["Host-evaluated outcome manifest did not match the task"],
                manifest=brief.manifest,
            )
        else:
            outcome = attempt.outcome
    else:
        outcome = await evaluate_structured_attempt(brief, contract, attempt.summary)
    completed = await asyncio.to_thread(
        store.complete_task, claimed.id, outcome, actor=name, authoritative=True
    )
    if completed is None:
        logger.warning(
            "Team worker harness could not complete %s (lost claim?)", claimed.id
        )
        return attempt.summary
    return completed.result


def worker_bootstrap_prompt(user_prompt: str) -> str:
    """Prompt used when spawning a worker (queue driver, not a single task)."""
    base = (
        "You are a long-lived team worker. The harness claims tasks from the "
        "shared TaskStore and injects each one as a turn. Prefer the team tool "
        "(available_tasks / claim / structured completion are harness-owned). "
        "Wait for the next task prompt after completing one."
    )
    extra = (user_prompt or "").strip()
    if not extra:
        return base
    return f"{base}\n\nLead notes:\n{extra}"
