from __future__ import annotations

from typing import Any

import pytest

from tests.conftest import build_test_vibe_app
from vibe.cli.textual_ui.widgets.messages import SubagentResponseMessage
from vibe.core.workflows.models import WorkflowResult, WorkflowRun, WorkflowStatus


@pytest.mark.asyncio
async def test_blocked_workflow_is_not_labeled_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app()
    mounted: list[Any] = []

    async def capture(widget: Any) -> None:
        mounted.append(widget)

    monkeypatch.setattr(app, "_mount_and_scroll", capture)
    app._agent_running = True
    result = WorkflowResult(
        run=WorkflowRun(status=WorkflowStatus.BLOCKED),
        summary="Workflow blocked: spend limit reached",
    )

    await app._on_workflow_complete(result)

    message = mounted[0]
    assert isinstance(message, SubagentResponseMessage)
    assert "(blocked)" in message._label
    assert "(completed)" not in message._label
    assert message.collapsed is False
