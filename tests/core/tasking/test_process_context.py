from __future__ import annotations

import orjson
import pytest

from vibe.core.tasking._process_context import (
    TASK_PROCESS_CONTEXT_ENV,
    TaskProcessContext,
    TaskProcessContextError,
    decode_task_process_context,
    install_task_process_context,
    load_task_process_context,
)
from vibe.core.tasking.models import TaskBrief, TaskManifestIdentity


def _brief() -> TaskBrief:
    return TaskBrief(
        objective="Update the parser",
        allowed_paths=["vibe/core/parser.py"],
        denied_paths=["vibe/core/agent_loop.py"],
        acceptance_checks=["focused"],
        manifest=TaskManifestIdentity(name="implement-verify", version="1"),
    )


def test_process_context_round_trip_and_environment_replacement() -> None:
    context = TaskProcessContext.from_brief(_brief())
    env = {TASK_PROCESS_CONTEXT_ENV: "stale"}

    install_task_process_context(env, context)

    assert load_task_process_context(env) == context
    install_task_process_context(env, None)
    assert TASK_PROCESS_CONTEXT_ENV not in env


def test_process_context_rejects_tampered_brief() -> None:
    context = TaskProcessContext.from_brief(_brief())
    payload = orjson.loads(context.model_dump_json())
    payload["brief"]["allowed_paths"] = ["**"]

    with pytest.raises(TaskProcessContextError):
        decode_task_process_context(orjson.dumps(payload).decode())
