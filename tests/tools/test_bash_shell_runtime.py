from __future__ import annotations

import asyncio
from typing import cast

import pytest

from tests.mock.utils import collect_result
from vibe.core._trusted_command import TrustedCommandError
from vibe.core.config import SandboxConfig
from vibe.core.tools._shell import get_base_shell_env
from vibe.core.tools.background import BackgroundRegistry
from vibe.core.tools.base import (
    BaseToolState,
    InvokeContext,
    ToolAuthorizationSource,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.builtins.bash import (
    Bash,
    BashArgs,
    BashToolConfig,
    _get_shell_executable,
)
from vibe.core.tools.permissions import (
    PermissionContext,
    authorization_context_fingerprint,
)


@pytest.fixture(autouse=True)
def clear_shell_executable_cache():
    _get_shell_executable.cache_clear()
    yield
    _get_shell_executable.cache_clear()


def test_linux_shell_executable_is_frozen_and_ignores_login_shell(monkeypatch) -> None:
    resolved_names: list[str] = []

    def fake_resolve(name: str):
        resolved_names.append(name)
        return "/opt/bash-a"

    monkeypatch.setattr("vibe.core.tools.builtins.bash.is_windows", lambda: False)
    monkeypatch.setenv("SHELL", "/usr/bin/fish")
    monkeypatch.setenv("PATH", "/workspace/.venv/bin")
    monkeypatch.setattr(
        "vibe.core.tools._shell.resolve_trusted_system_executable", fake_resolve
    )
    monkeypatch.setattr("vibe.core.tools._shell.is_windows", lambda: False)
    first = _get_shell_executable()
    monkeypatch.setattr(
        "vibe.core.tools._shell.resolve_trusted_system_executable",
        lambda _name: "/opt/bash-b",
    )

    assert first == "/opt/bash-a"
    assert _get_shell_executable() == first
    assert resolved_names == ["bash"]


def test_bash_environment_drops_startup_code(monkeypatch) -> None:
    monkeypatch.setattr(
        "vibe.core.tools._shell.get_bash_executable", lambda: "/usr/bin/bash"
    )
    monkeypatch.setenv("BASH_ENV", "/tmp/bootstrap.sh")
    monkeypatch.setenv("BASH_COMPAT", "4.2")
    monkeypatch.setenv("ENV", "/tmp/posix-bootstrap.sh")
    monkeypatch.setenv("POSIXLY_CORRECT", "1")
    monkeypatch.setenv("POSIX_PEDANTIC", "1")
    monkeypatch.setenv("SHELLOPTS", "xtrace")
    monkeypatch.setenv("BASHOPTS", "extdebug")
    monkeypatch.setenv("BASH_XTRACEFD", "9")
    monkeypatch.setenv("PS4", "$(run-hidden-code)")
    monkeypatch.setenv("BASH_FUNC_shadow%%", "() { run-hidden-code; }")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/inject.so")
    monkeypatch.setenv("LD_AUDIT", "/tmp/audit.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/cuda/lib64")
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/tmp/inject.dylib")
    monkeypatch.setenv("LESSOPEN", "|run-hidden-code %s")
    monkeypatch.setenv("LESSCLOSE", "run-hidden-code %s %s")
    monkeypatch.setenv("LESSKEY", "/tmp/lesskey")
    monkeypatch.setenv("LESSKEYIN", "/tmp/lesskey-src")
    monkeypatch.setenv("LESSKEYIN_SYSTEM", "/tmp/system-lesskey-src")
    monkeypatch.setenv("LESSKEY_CONTENT", "x shell run-hidden-code")
    monkeypatch.setenv("LESSKEY_SYSTEM", "/tmp/system-lesskey")
    monkeypatch.setenv("LESSSECURE_ALLOW", "shell,pipe")
    monkeypatch.setenv("LESSSECURE", "0")
    monkeypatch.setenv("SHELL", "/usr/bin/fish")
    monkeypatch.setenv("VIBE_TEST_VISIBLE", "yes")

    environment = get_base_shell_env()

    assert environment["VIBE_TEST_VISIBLE"] == "yes"
    assert "BASH_ENV" not in environment
    assert "BASH_COMPAT" not in environment
    assert "ENV" not in environment
    assert "POSIXLY_CORRECT" not in environment
    assert "POSIX_PEDANTIC" not in environment
    assert "SHELLOPTS" not in environment
    assert "BASHOPTS" not in environment
    assert "BASH_XTRACEFD" not in environment
    assert "PS4" not in environment
    assert not any(name.startswith("BASH_FUNC_") for name in environment)
    assert "LD_PRELOAD" not in environment
    assert "LD_AUDIT" not in environment
    assert environment["LD_LIBRARY_PATH"] == "/opt/cuda/lib64"
    assert "DYLD_INSERT_LIBRARIES" not in environment
    assert "LESSOPEN" not in environment
    assert "LESSCLOSE" not in environment
    assert "LESSKEY" not in environment
    assert "LESSKEYIN" not in environment
    assert "LESSKEYIN_SYSTEM" not in environment
    assert "LESSKEY_CONTENT" not in environment
    assert "LESSKEY_SYSTEM" not in environment
    assert "LESSSECURE_ALLOW" not in environment
    assert environment["LESSSECURE"] == "1"
    assert environment["SHELL"] == "/usr/bin/bash"


def test_autoapproved_execution_uses_only_trusted_system_path(monkeypatch) -> None:
    trusted_path = "/usr/bin:/bin"
    monkeypatch.setenv("LD_LIBRARY_PATH", "/workspace/injected-libs")
    monkeypatch.setenv("PYTHONPATH", "/workspace/injected-python")
    monkeypatch.setenv("TEXTDOMAIN", "injected-catalog")
    monkeypatch.setenv("TEXTDOMAINDIR", "/workspace/catalogs")
    monkeypatch.setattr(
        "vibe.core.tools._shell.get_trusted_system_path", lambda: trusted_path
    )
    bash_tool = Bash(
        config_getter=lambda: BashToolConfig(sandbox=SandboxConfig(enabled=False)),
        state=BaseToolState(),
    )

    _argv, _profile, environment, _fd = bash_tool._resolve_sandbox(
        None, "cat README.md", trusted_system_path_only=True
    )

    assert environment["PATH"] == trusted_path
    assert environment["HOME"] == "/nonexistent"
    assert "LD_LIBRARY_PATH" not in environment
    assert "PYTHONPATH" not in environment
    assert "TEXTDOMAIN" not in environment
    assert "TEXTDOMAINDIR" not in environment


@pytest.mark.asyncio
async def test_automated_authorization_drift_never_reaches_spawn(monkeypatch) -> None:
    async def fail_if_spawned(*_args, **_kwargs):
        raise AssertionError("spawn must not be reached")

    bash_tool = Bash(
        config_getter=lambda: BashToolConfig(permission=ToolPermission.ALWAYS),
        state=BaseToolState(),
    )
    monkeypatch.setattr(
        bash_tool,
        "resolve_permission",
        lambda _args: PermissionContext(
            permission=ToolPermission.ASK, requires_explicit_user_approval=True
        ),
    )
    monkeypatch.setattr(bash_tool, "_start_foreground", fail_if_spawned)
    approved = PermissionContext(permission=ToolPermission.ALWAYS)
    ctx = InvokeContext(
        tool_call_id="drift",
        authorization_source=ToolAuthorizationSource.POLICY,
        authorization_fingerprint=authorization_context_fingerprint(
            "bash", BashArgs(command="cat README.md"), approved
        ),
    )

    with pytest.raises(ToolError, match="authorization context changed"):
        await collect_result(bash_tool.run(BashArgs(command="cat README.md"), ctx))


@pytest.mark.parametrize(
    "command", ["/usr/bin/git log -1", "env git log -1", "git log -1 && true"]
)
@pytest.mark.asyncio
async def test_unhardenable_automated_git_never_reaches_spawn(
    monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    async def fail_if_spawned(*_args, **_kwargs):
        raise AssertionError("spawn must not be reached")

    args = BashArgs(command=command)
    bash_tool = Bash(
        config_getter=lambda: BashToolConfig(permission=ToolPermission.ALWAYS),
        state=BaseToolState(),
    )
    permission = bash_tool.resolve_permission(args)
    assert permission is not None
    assert permission.requires_explicit_user_approval
    monkeypatch.setattr(bash_tool, "_start_foreground", fail_if_spawned)
    ctx = InvokeContext(
        tool_call_id="unhardenable-git",
        authorization_source=ToolAuthorizationSource.POLICY,
        authorization_fingerprint=authorization_context_fingerprint(
            "bash", args, permission
        ),
    )

    with pytest.raises(ToolError, match="authorization changed after approval"):
        await collect_result(bash_tool.run(args, ctx))


@pytest.mark.parametrize(
    ("source", "permission", "message"),
    [
        (
            ToolAuthorizationSource.STORED_USER,
            PermissionContext(
                permission=ToolPermission.ASK, requires_explicit_user_approval=True
            ),
            "no longer covers",
        ),
        (
            ToolAuthorizationSource.USER,
            PermissionContext(permission=ToolPermission.NEVER),
            "disabled after approval",
        ),
    ],
)
@pytest.mark.asyncio
async def test_user_authorization_drift_never_reaches_spawn(
    monkeypatch,
    source: ToolAuthorizationSource,
    permission: PermissionContext,
    message: str,
) -> None:
    async def fail_if_spawned(*_args, **_kwargs):
        raise AssertionError("spawn must not be reached")

    bash_tool = Bash(
        config_getter=lambda: BashToolConfig(permission=ToolPermission.ASK),
        state=BaseToolState(),
    )
    monkeypatch.setattr(bash_tool, "resolve_permission", lambda _args: permission)
    monkeypatch.setattr(bash_tool, "_start_foreground", fail_if_spawned)
    ctx = InvokeContext(
        tool_call_id="drift",
        authorization_source=source,
        authorization_fingerprint=authorization_context_fingerprint(
            "bash", BashArgs(command="cat README.md"), permission
        ),
    )

    with pytest.raises(ToolError, match=message):
        await collect_result(bash_tool.run(BashArgs(command="cat README.md"), ctx))


@pytest.mark.asyncio
async def test_bypass_authorization_drift_never_reaches_spawn(monkeypatch) -> None:
    async def fail_if_spawned(*_args, **_kwargs):
        raise AssertionError("spawn must not be reached")

    permission = PermissionContext(permission=ToolPermission.ASK)
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())
    monkeypatch.setattr(bash_tool, "resolve_permission", lambda _args: permission)
    monkeypatch.setattr(bash_tool, "_start_foreground", fail_if_spawned)
    ctx = InvokeContext(
        tool_call_id="bypass-drift",
        authorization_source=ToolAuthorizationSource.BYPASS,
        authorization_fingerprint=authorization_context_fingerprint(
            "bash", BashArgs(command="cat README.md"), permission
        ),
    )

    with pytest.raises(ToolError, match="auto-approve authority changed"):
        await collect_result(bash_tool.run(BashArgs(command="cat README.md"), ctx))


@pytest.mark.asyncio
async def test_authorization_is_bound_to_working_directory(
    tmp_path, monkeypatch
) -> None:
    async def fail_if_spawned(*_args, **_kwargs):
        raise AssertionError("spawn must not be reached")

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    permission = PermissionContext(permission=ToolPermission.ASK)
    args = BashArgs(command="cat README.md")
    monkeypatch.chdir(first)
    fingerprint = authorization_context_fingerprint("bash", args, permission)
    monkeypatch.chdir(second)
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())
    monkeypatch.setattr(bash_tool, "resolve_permission", lambda _args: permission)
    monkeypatch.setattr(bash_tool, "_start_foreground", fail_if_spawned)
    ctx = InvokeContext(
        tool_call_id="cwd-drift",
        authorization_source=ToolAuthorizationSource.USER,
        authorization_fingerprint=fingerprint,
    )

    with pytest.raises(ToolError, match="authorization context changed"):
        await collect_result(bash_tool.run(args, ctx))


def test_linux_without_bash_fails_closed(monkeypatch) -> None:
    def missing(_name: str):
        raise TrustedCommandError("missing")

    monkeypatch.setattr("vibe.core.tools.builtins.bash.is_windows", lambda: False)
    monkeypatch.setattr("vibe.core.tools._shell.is_windows", lambda: False)
    monkeypatch.setattr(
        "vibe.core.tools._shell.resolve_trusted_system_executable", missing
    )
    bash_tool = Bash(
        config_getter=lambda: BashToolConfig(permission=ToolPermission.ALWAYS),
        state=BaseToolState(),
    )

    permission = bash_tool.resolve_permission(BashArgs(command="echo hello"))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER
    assert "Bash executable" in (permission.reason or "")


@pytest.mark.asyncio
async def test_linux_without_bash_never_reaches_spawn(monkeypatch) -> None:
    async def fail_if_spawned(*_args, **_kwargs):
        raise AssertionError("spawn must not be reached")

    monkeypatch.setattr("vibe.core.tools.builtins.bash.is_windows", lambda: False)
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash._get_shell_executable", lambda: None
    )
    monkeypatch.setattr(asyncio, "create_subprocess_shell", fail_if_spawned)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_if_spawned)
    bash_tool = Bash(
        config_getter=lambda: BashToolConfig(permission=ToolPermission.ALWAYS),
        state=BaseToolState(),
    )

    with pytest.raises(ToolError, match="Bash executable"):
        await collect_result(bash_tool.run(BashArgs(command="echo hello")))


@pytest.mark.asyncio
async def test_foreground_spawn_uses_frozen_bash_executable(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_shell(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("vibe.core.tools.builtins.bash.is_windows", lambda: False)
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash._get_shell_executable",
        lambda: "/opt/frozen/bash",
    )
    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_shell)
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())
    monkeypatch.setattr(
        bash_tool, "_resolve_sandbox", lambda _ctx, _command: (None, None, {}, None)
    )

    await bash_tool._start_foreground("echo hello", None)

    assert captured["command"] == "echo hello"
    assert captured["executable"] == "/opt/frozen/bash"


@pytest.mark.asyncio
async def test_sandbox_and_fallback_use_same_frozen_bash(monkeypatch) -> None:
    exec_argv: tuple[str, ...] = ()
    fallback: dict[str, object] = {}

    async def failing_exec(*argv, **_kwargs):
        nonlocal exec_argv
        exec_argv = argv
        raise OSError("sandbox unavailable")

    async def fake_shell(command, **kwargs):
        fallback["command"] = command
        fallback.update(kwargs)
        return object()

    monkeypatch.setattr("vibe.core.tools.builtins.bash.is_windows", lambda: False)
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash._get_shell_executable",
        lambda: "/opt/frozen/bash",
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", failing_exec)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_shell)
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())
    monkeypatch.setattr(
        bash_tool,
        "_resolve_sandbox",
        lambda _ctx, _command: (["sandbox", "--"], None, {}, None),
    )

    await bash_tool._start_foreground("echo hello", None)

    assert exec_argv == ("sandbox", "--", "/opt/frozen/bash", "-c", "echo hello")
    assert fallback["executable"] == "/opt/frozen/bash"


@pytest.mark.asyncio
async def test_strict_sandbox_wrapper_failure_never_falls_back(monkeypatch) -> None:
    async def failing_exec(*_argv, **_kwargs):
        raise OSError("sandbox unavailable")

    async def fail_if_unsandboxed(*_args, **_kwargs):
        raise AssertionError("unsandboxed fallback must not be reached")

    monkeypatch.setattr("vibe.core.tools.builtins.bash.is_windows", lambda: False)
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash._get_shell_executable",
        lambda: "/opt/frozen/bash",
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", failing_exec)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", fail_if_unsandboxed)
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())
    monkeypatch.setattr(
        bash_tool,
        "_resolve_sandbox",
        lambda _ctx, _command: (
            ["sandbox", "--"],
            None,
            {"VIBE_STRICT_MODEL_CONTROL": "1"},
            None,
        ),
    )

    with pytest.raises(ToolError, match="Sandbox wrapper failed to start"):
        await bash_tool._start_foreground("echo hello", None)


@pytest.mark.asyncio
async def test_background_spawn_uses_frozen_bash_executable(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 123

    class FakeRegistry:
        async def register_process(self, _proc, **kwargs):
            kwargs["log_handle"].close()
            return "proc-1"

    async def fake_shell(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr("vibe.core.tools.builtins.bash.is_windows", lambda: False)
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash._get_shell_executable",
        lambda: "/opt/frozen/bash",
    )
    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_shell)
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())
    monkeypatch.setattr(
        bash_tool, "_resolve_sandbox", lambda _ctx, _command: (None, None, {}, None)
    )
    ctx = InvokeContext(
        tool_call_id="bash-1",
        scratchpad_dir=tmp_path,
        background_registry=cast(BackgroundRegistry, FakeRegistry()),
    )

    result = await collect_result(
        bash_tool._run_background(BashArgs(command="echo hello", background=True), ctx)
    )

    assert captured["command"] == "echo hello"
    assert captured["executable"] == "/opt/frozen/bash"
    assert result.background_task_id == "proc-1"
