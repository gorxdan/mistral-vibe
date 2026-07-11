from __future__ import annotations

import time

import pytest

from tests.conftest import build_test_vibe_app
from vibe.cli.textual_ui.widgets.messages import UserCommandMessage
from vibe.cli.textual_ui.workflow_runner import WorkflowRunEntry
from vibe.core.tools.base import InvokeContext
from vibe.core.workflows.runtime import WorkflowRuntime


@pytest.mark.asyncio
async def test_spend_status_reports_current_envelope() -> None:
    app = build_test_vibe_app()

    async with app.run_test() as pilot:
        handled = await app._handle_command("/spend")
        await pilot.pause()
        messages = [message._content for message in app.query(UserCommandMessage)]

    assert handled is True
    assert any("### Spend budget" in message for message in messages)
    assert any("**Calls**" in message for message in messages)


@pytest.mark.asyncio
async def test_spend_reset_rebinds_existing_team_manager() -> None:
    app = build_test_vibe_app()

    async with app.run_test() as pilot:
        manager = app._build_team_manager()
        app._team_manager = manager
        previous_adapter = manager._spend_adapter
        try:
            handled = await app._handle_command("/spend reset")
            await pilot.pause()

            assert handled is True
            assert manager._spend_adapter is app.agent_loop.spend_adapter
            assert manager._spend_adapter is not previous_adapter
            assert any(
                "Spend ledger reset" in message._content
                for message in app.query(UserCommandMessage)
            )
        finally:
            app._team_manager = None
            manager.cleanup()


@pytest.mark.asyncio
async def test_spend_reset_rebinds_active_workflow_runtime() -> None:
    app = build_test_vibe_app()

    async with app.run_test() as pilot:
        previous_adapter = app.agent_loop.spend_adapter
        runtime = WorkflowRuntime(
            parent_context=InvokeContext(
                tool_call_id="workflow", spend_adapter=previous_adapter
            )
        )
        entry = WorkflowRunEntry(
            run_id="wf-reset",
            script_source="",
            started_at=time.monotonic(),
            runtime=runtime,
        )
        app._workflow_runner._runs.append(entry)
        try:
            handled = await app._handle_command("/spend reset")
            await pilot.pause()

            assert handled is True
            assert runtime.parent_context is not None
            assert runtime.parent_context.spend_adapter is app.agent_loop.spend_adapter
            assert runtime.parent_context.spend_adapter is not previous_adapter
        finally:
            app._workflow_runner._runs.remove(entry)


@pytest.mark.asyncio
async def test_clear_rebinds_existing_team_manager() -> None:
    app = build_test_vibe_app()

    async with app.run_test() as pilot:
        manager = app._build_team_manager()
        app._team_manager = manager
        previous_adapter = manager._spend_adapter
        try:
            handled = await app._handle_command("/clear")
            await pilot.pause()

            assert handled is True
            assert manager._spend_adapter is app.agent_loop.spend_adapter
            assert manager._spend_adapter is not previous_adapter
        finally:
            app._team_manager = None
            manager.cleanup()
