from __future__ import annotations

from typing import cast

import pytest

from tests.conftest import build_test_vibe_app, build_test_vibe_config
from vibe.cli.textual_ui.app import BottomApp
from vibe.cli.textual_ui.widgets.messages import ErrorMessage, UserCommandMessage
from vibe.cli.textual_ui.widgets.provider_login_app import ProviderLoginApp
from vibe.setup.auth.zai_protocol_handler import ZaiProtocolHandlerInstallResult
from vibe.setup.auth.zai_sign_in import ZaiSignInService


class _StubZaiSignIn:
    receive_code = None

    async def authenticate(self, *, on_url: object = None) -> str:
        return "zai-id.zai-secret"


def _noop_zai_protocol_handler_installer() -> ZaiProtocolHandlerInstallResult:
    return ZaiProtocolHandlerInstallResult(status="installed")


@pytest.mark.asyncio
async def test_login_command_opens_provider_login_app() -> None:
    app = build_test_vibe_app()

    async with app.run_test() as pilot:
        handled = await app._handle_command("/login zai")
        await pilot.pause()
        assert handled is True
        assert app._current_bottom_app == BottomApp.ProviderLogin
        login_app = app.query_one(ProviderLoginApp)
        assert login_app._target is not None
        assert login_app._target.key == "zai"


@pytest.mark.asyncio
async def test_login_command_rejects_extra_arguments() -> None:
    app = build_test_vibe_app()

    async with app.run_test() as pilot:
        handled = await app._handle_command("/login zai extra")
        await pilot.pause()
        assert handled is True
        errors = app.query(ErrorMessage)
        assert any(error._error == "Usage: /login [provider]" for error in errors)


@pytest.mark.asyncio
async def test_provider_login_success_refreshes_config_and_reports_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app()
    refreshed = False

    def fake_refresh_config() -> None:
        nonlocal refreshed
        refreshed = True

    async def fake_refresh_system_prompt() -> None:
        return None

    monkeypatch.setattr(app.agent_loop, "refresh_config", fake_refresh_config)
    monkeypatch.setattr(
        app.agent_loop, "refresh_system_prompt", fake_refresh_system_prompt
    )

    message = ProviderLoginApp.ProviderLoginClosed(
        authenticated=True, provider_name="zai"
    )

    async with app.run_test() as pilot:
        await app.on_provider_login_app_provider_login_closed(message)
        await pilot.pause()
        assert any(
            message._content == "Logged in to zai."
            for message in app.query(UserCommandMessage)
        )

    assert refreshed is True


@pytest.mark.asyncio
async def test_provider_login_zai_browser_flow_posts_success() -> None:
    login_app = ProviderLoginApp(
        build_test_vibe_config(),
        provider_name="zai",
        zai_sign_in_service_factory=lambda: cast(ZaiSignInService, _StubZaiSignIn()),
        zai_protocol_handler_installer=_noop_zai_protocol_handler_installer,
    )
    target = next(target for target in login_app._targets if target.key == "zai")

    result = await login_app._run_zai_login(target)

    assert result.authenticated is True
    assert result.provider_name == "zai"
