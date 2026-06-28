from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from tests.conftest import build_test_vibe_app, build_test_vibe_config
from vibe.cli.textual_ui.widgets.session_picker import SessionPickerApp
from vibe.core.config import SessionLoggingConfig


@pytest.mark.asyncio
async def test_startup_prompt_waits_for_startup_resume_picker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app(
        initial_prompt="continue the work",
        config=build_test_vibe_config(
            session_logging=SessionLoggingConfig(enabled=True)
        ),
    )
    app._show_resume_picker = True
    process_prompt = Mock()

    monkeypatch.setattr(
        "vibe.cli.textual_ui.app.list_local_resume_sessions",
        lambda *_args, **_kwargs: ["session-1"],
    )
    monkeypatch.setattr(app, "_build_picker", Mock(return_value=object()))
    monkeypatch.setattr(app, "_switch_from_input", AsyncMock())
    monkeypatch.setattr(app, "_process_initial_prompt", process_prompt)

    await app._show_session_picker()

    # Picker shown ⇒ the initial prompt is deferred until a session is selected.
    process_prompt.assert_not_called()
    assert app._show_resume_picker is True


@pytest.mark.asyncio
async def test_startup_prompt_runs_after_startup_resume_picker_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app(initial_prompt="continue the work")
    app._show_resume_picker = True
    process_prompt = Mock()

    monkeypatch.setattr(app, "_switch_to_input_app", AsyncMock())
    monkeypatch.setattr(app, "_resume_local_session", AsyncMock())
    monkeypatch.setattr(app, "_process_initial_prompt", process_prompt)

    await app.on_session_picker_app_session_selected(
        SessionPickerApp.SessionSelected("local:session-1", "session-1")
    )

    assert app._show_resume_picker is False
    process_prompt.assert_called_once_with()


@pytest.mark.asyncio
async def test_startup_teleport_routes_after_session_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app(initial_prompt="continue the work")
    app._show_resume_picker = True
    app._teleport_on_start = True
    run_worker = Mock()
    handle_teleport = Mock(return_value=object())
    handle_user_message = Mock(return_value=object())

    monkeypatch.setattr(app, "_switch_to_input_app", AsyncMock())
    monkeypatch.setattr(app, "_resume_local_session", AsyncMock())
    monkeypatch.setattr(app, "run_worker", run_worker)
    monkeypatch.setattr(app, "_handle_teleport_command", handle_teleport)
    monkeypatch.setattr(app, "_handle_user_message", handle_user_message)
    monkeypatch.setattr(app.commands, "has_command", lambda name: name == "teleport")

    await app.on_session_picker_app_session_selected(
        SessionPickerApp.SessionSelected("local:session-1", "session-1")
    )

    assert app._show_resume_picker is False
    handle_teleport.assert_called_once_with("continue the work")
    handle_user_message.assert_not_called()
    run_worker.assert_called_once_with(handle_teleport.return_value, exclusive=False)
