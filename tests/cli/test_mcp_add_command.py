from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.conftest import build_test_vibe_app, build_test_vibe_config
from vibe.cli.textual_ui.mcp_commands import parse_mcp_add_args, parse_mcp_subcommand
from vibe.cli.textual_ui.widgets.mcp_add_app import MCPAddApp
from vibe.core.config import MCPHttp, MCPOAuth, MCPStreamableHttp, VibeConfig


def test_parse_mcp_add_args_accepts_url_and_options() -> None:
    args = parse_mcp_add_args(
        "https://mcp.example.com/mcp --name docs --scope read --scope write "
        "--transport http --no-login"
    )

    assert args.url == "https://mcp.example.com/mcp"
    assert args.name == "docs"
    assert args.scopes == ["read", "write"]
    assert args.transport == "http"
    assert args.login is False


def test_parse_mcp_add_args_defaults_to_login() -> None:
    args = parse_mcp_add_args("https://mcp.example.com/mcp")

    assert args.transport == "streamable-http"
    assert args.login is True


@pytest.mark.parametrize(
    "raw_args",
    [
        "",
        "https://mcp.example.com/mcp extra",
        "https://mcp.example.com/mcp --unknown",
        "https://mcp.example.com/mcp --name",
        "https://mcp.example.com/mcp --name a --name b",
        "https://mcp.example.com/mcp --scope",
        "https://mcp.example.com/mcp --login",
        "https://mcp.example.com/mcp --transport",
        "https://mcp.example.com/mcp --transport sse",
        "https://mcp.example.com/mcp --transport http --transport streamable-http",
        "'unterminated",
    ],
)
def test_parse_mcp_add_args_rejects_invalid_args(raw_args: str) -> None:
    with pytest.raises(ValueError):
        parse_mcp_add_args(raw_args)


def test_parse_mcp_subcommand_recognizes_supported_subcommands() -> None:
    parsed = parse_mcp_subcommand("add https://mcp.linear.app/mcp --no-login")

    assert parsed is not None
    assert parsed.name == "add"
    assert parsed.args == "https://mcp.linear.app/mcp --no-login"


def test_parse_mcp_subcommand_ignores_unknown_subcommands() -> None:
    assert parse_mcp_subcommand("tools linear") is None


# The fork replaced upstream's string-based `/mcp add <url> --opts` flow (which
# parsed args, saved the server, and optionally ran OAuth login inline) with an
# interactive bottom-panel form, MCPAddApp. `_mcp_add()` now takes no arguments
# and just opens that form; `_dispatch_mcp_subcommand("add ...")` routes to it.
# The tests below assert that new behavior and exercise MCPAddApp's save logic
# directly, which is where the persistence coverage now lives.


def _make_add_app(**values: str) -> MCPAddApp:
    fields = {
        "name": "",
        "transport": "http",
        "url": "",
        "command": "",
        "auth": "static",
        "api_key_env": "",
        "scopes": "",
    }
    fields.update(values)
    app = MCPAddApp()
    app.inputs = {key: MagicMock(value=val) for key, val in fields.items()}
    app.post_message = MagicMock()  # type: ignore[method-assign]
    return app


@pytest.mark.asyncio
async def test_mcp_add_opens_interactive_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app(config=build_test_vibe_config())
    switch = AsyncMock()
    monkeypatch.setattr(app, "_switch_from_input", switch)

    await app._mcp_add()

    switch.assert_awaited_once()
    assert isinstance(switch.await_args.args[0], MCPAddApp)


@pytest.mark.asyncio
async def test_dispatch_mcp_subcommand_routes_add_to_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app(config=build_test_vibe_config())
    switch = AsyncMock()
    monkeypatch.setattr(app, "_switch_from_input", switch)

    handled = await app._dispatch_mcp_subcommand("add")

    assert handled is True
    switch.assert_awaited_once()
    assert isinstance(switch.await_args.args[0], MCPAddApp)


@pytest.mark.asyncio
async def test_dispatch_mcp_subcommand_ignores_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app(config=build_test_vibe_config())

    assert await app._dispatch_mcp_subcommand("frobnicate linear") is False


def test_mcp_add_form_saves_oauth_streamable_http_server() -> None:
    app = _make_add_app(
        name="linear",
        transport="streamable-http",
        url="https://mcp.linear.app/mcp",
        auth="oauth",
    )

    app._save_and_close()

    server = VibeConfig.load().mcp_servers[0]
    assert isinstance(server, MCPStreamableHttp)
    assert server.name == "linear"
    assert isinstance(server.auth, MCPOAuth)
    assert server.auth.scopes == []
    closed = app.post_message.call_args.args[0]
    assert isinstance(closed, MCPAddApp.MCPAddClosed)
    assert closed.added is True
    assert closed.name == "linear"


def test_mcp_add_form_saves_name_and_scopes() -> None:
    app = _make_add_app(
        name="docs",
        transport="streamable-http",
        url="https://mcp.example.com/mcp",
        auth="oauth",
        scopes="read, write",
    )

    app._save_and_close()

    server = VibeConfig.load().mcp_servers[0]
    assert isinstance(server, MCPStreamableHttp)
    assert server.name == "docs"
    assert isinstance(server.auth, MCPOAuth)
    assert server.auth.scopes == ["read", "write"]


def test_mcp_add_form_saves_http_transport() -> None:
    app = _make_add_app(
        name="docs",
        transport="http",
        url="https://mcp.example.com/mcp",
        auth="oauth",
    )

    app._save_and_close()

    server = VibeConfig.load().mcp_servers[0]
    assert isinstance(server, MCPHttp)
    assert server.transport == "http"
    assert isinstance(server.auth, MCPOAuth)


def test_mcp_add_form_requires_name() -> None:
    app = _make_add_app(
        name="",
        transport="http",
        url="https://mcp.example.com/mcp",
    )

    app._save_and_close()

    assert VibeConfig.load().mcp_servers == []
    closed = app.post_message.call_args.args[0]
    assert closed.added is False
    assert closed.error == "Server name is required."
