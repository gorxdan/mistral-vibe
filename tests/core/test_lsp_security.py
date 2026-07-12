from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

from vibe.core.lsp import _server as server_module
from vibe.core.lsp._environment import language_server_env
from vibe.core.lsp._server import LanguageServer, ServerConfig
from vibe.core.tools.base import ToolError, ToolPermission
from vibe.core.tools.builtins.lsp import Lsp, LspArgs, LspConfig, LspOperation, LspState
from vibe.core.tools.permissions import PermissionScope


def _tool(config: LspConfig | None = None) -> Lsp:
    resolved = config or LspConfig()
    return Lsp(config_getter=lambda: resolved, state=LspState())


def _document_symbols(path: str) -> LspArgs:
    return LspArgs(operation=LspOperation.DOCUMENT_SYMBOL, file_path=path)


def test_resolve_path_rejects_team_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    team_dir = tmp_path / "team"
    team_dir.mkdir()
    metadata = team_dir / "tasks.json"
    metadata.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("VIBE_TEAM_DIR", str(team_dir))

    with pytest.raises(ToolError, match="team coordination metadata"):
        _tool()._resolve_path(str(metadata))


def test_resolve_path_rejects_isolated_worktree_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    outside = tmp_path / "host.py"
    outside.write_text("secret = True\n", encoding="utf-8")
    monkeypatch.setenv("VIBE_ISOLATED_WORKTREE_ROOT", str(worktree))

    with pytest.raises(ToolError, match="isolated subagent is confined"):
        _tool()._resolve_path(str(outside))


def test_file_permission_honors_denylist(tmp_path: Path) -> None:
    path = tmp_path / "secret.py"
    permission = _tool(LspConfig(denylist=["*/secret.py"])).resolve_permission(
        _document_symbols(str(path))
    )

    assert permission is not None
    assert permission.permission is ToolPermission.NEVER


def test_file_permission_asks_for_sensitive_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    permission = _tool().resolve_permission(_document_symbols(".env"))

    assert permission is not None
    assert permission.permission is ToolPermission.ASK
    assert {required.scope for required in permission.required_permissions} == {
        PermissionScope.FILE_PATTERN
    }


def test_file_permission_asks_outside_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    permission = _tool().resolve_permission(
        _document_symbols(str(tmp_path / "outside.py"))
    )

    assert permission is not None
    assert permission.permission is ToolPermission.ASK
    assert PermissionScope.OUTSIDE_DIRECTORY in {
        required.scope for required in permission.required_permissions
    }


def test_workspace_symbol_retains_configured_permission() -> None:
    permission = _tool(LspConfig(permission=ToolPermission.NEVER)).resolve_permission(
        LspArgs(operation=LspOperation.WORKSPACE_SYMBOL, query="Widget")
    )

    assert permission is not None
    assert permission.permission is ToolPermission.NEVER


def test_language_server_env_scrubs_credentials_and_keeps_toolchains() -> None:
    inherited = language_server_env(
        {},
        {
            "PATH": "/opt/tools:/usr/bin",
            "VIRTUAL_ENV": "/workspace/.venv",
            "GOPRIVATE": "corp.example/*",
            "GONOPROXY": "corp.example/*",
            "GOPROXY": "https://proxy.example.com,direct",
            "CPATH": "/opt/headers",
            "CPLUS_INCLUDE_PATH": "/opt/cpp-headers",
            "COMPILER_PATH": "/opt/llvm/bin",
            "DEVELOPER_DIR": "/Applications/Xcode.app/Contents/Developer",
            "SDKROOT": "/opt/swift-sdk",
            "TOOLCHAINS": "swift",
            "GRADLE_USER_HOME": "/workspace/.gradle",
            "HTTPS_PROXY": "http://proxy.example.com:8080",
            "NO_PROXY": "localhost,.example.com",
            "RUSTUP_TOOLCHAIN": "stable",
            "MISTRAL_API_KEY": "ambient-provider-secret",
            "MAVEN_OPTS": "-Dtoken=ambient-build-secret",
            "JAVA_TOOL_OPTIONS": "-Dtoken=ambient-java-secret",
        },
    )

    assert inherited == {
        "PATH": "/opt/tools:/usr/bin",
        "VIRTUAL_ENV": "/workspace/.venv",
        "GOPRIVATE": "corp.example/*",
        "GONOPROXY": "corp.example/*",
        "GOPROXY": "https://proxy.example.com,direct",
        "CPATH": "/opt/headers",
        "CPLUS_INCLUDE_PATH": "/opt/cpp-headers",
        "COMPILER_PATH": "/opt/llvm/bin",
        "DEVELOPER_DIR": "/Applications/Xcode.app/Contents/Developer",
        "SDKROOT": "/opt/swift-sdk",
        "TOOLCHAINS": "swift",
        "GRADLE_USER_HOME": "/workspace/.gradle",
        "HTTPS_PROXY": "http://proxy.example.com:8080",
        "NO_PROXY": "localhost,.example.com",
        "RUSTUP_TOOLCHAIN": "stable",
    }


def test_language_server_env_rejects_credentialed_proxy_unless_explicit() -> None:
    proxy = "https://build-user:build-password@proxy.example.com"
    assert "GOPROXY" not in language_server_env({}, {"GOPROXY": proxy})
    assert language_server_env({"GOPROXY": proxy}, {})["GOPROXY"] == proxy
    assert "HTTPS_PROXY" not in language_server_env({}, {"HTTPS_PROXY": proxy})


@pytest.mark.asyncio
async def test_language_server_spawn_uses_scrubbed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe_env = {"PATH": "/synthetic/bin"}
    helper_calls: list[dict[str, str]] = []
    spawn_env_matches: list[bool] = []

    def fake_language_server_env(configured: dict[str, str]) -> dict[str, str]:
        helper_calls.append(configured)
        return safe_env

    async def capture_spawn(*_args: str, **kwargs: Any) -> Any:
        spawn_env_matches.append(kwargs.get("env") is safe_env)
        raise OSError("spawn captured")

    monkeypatch.setattr(server_module, "language_server_env", fake_language_server_env)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_spawn)
    server = LanguageServer(
        ServerConfig(
            name="test",
            command=["test-language-server"],
            languages={".test": "test"},
            env={"OPENAI_API_KEY": "explicit-server-secret", "LSP_SETTING": "enabled"},
        )
    )

    with pytest.raises(OSError, match="spawn captured"):
        await server._spawn()

    assert helper_calls == [
        {"OPENAI_API_KEY": "explicit-server-secret", "LSP_SETTING": "enabled"}
    ]
    assert spawn_env_matches == [True]


def test_preset_probe_uses_scrubbed_environment(monkeypatch) -> None:
    from types import SimpleNamespace

    from vibe.core.lsp import _defaults
    from vibe.core.lsp._defaults import PRESETS

    safe_env = {"PATH": "/synthetic/bin"}
    helper_calls: list[dict[str, str]] = []
    probe_env_matches: list[bool] = []

    def fake_language_server_env(configured: dict[str, str]) -> dict[str, str]:
        helper_calls.append(configured)
        return safe_env

    def fake_run(*_args: object, **kwargs: object) -> SimpleNamespace:
        probe_env_matches.append(kwargs.get("env") is safe_env)
        return SimpleNamespace(returncode=0, stderr="", stdout="ok")

    monkeypatch.setattr(
        _defaults, "language_server_env", fake_language_server_env, raising=False
    )
    monkeypatch.setattr(_defaults, "_resolve_binary", lambda *_args: "/fake/pyright")
    monkeypatch.setattr(_defaults.subprocess, "run", fake_run)

    result = _defaults._probe(PRESETS["pyright"])

    assert result.status == "available"
    assert helper_calls == [{}]
    assert probe_env_matches == [True]


@pytest.mark.asyncio
async def test_language_server_spawn_log_omits_command_arguments(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def reject_spawn(*_args: str, **_kwargs: Any) -> Any:
        raise OSError("spawn rejected")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", reject_spawn)
    monkeypatch.setattr(
        server_module,
        "language_server_env",
        lambda _configured: {"PATH": "/synthetic/bin"},
    )
    caplog.set_level(logging.DEBUG, logger="vibe")
    server = LanguageServer(
        ServerConfig(
            name="test",
            command=["test-language-server", "--token", "sensitive-test-value"],
            languages={".test": "test"},
        )
    )

    with pytest.raises(OSError, match="spawn rejected"):
        await server._spawn()

    messages = [record.getMessage() for record in caplog.records]
    assert any("test-language-server" in message for message in messages)
    assert all("sensitive-test-value" not in message for message in messages)
