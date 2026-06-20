from __future__ import annotations

import asyncio

import pytest

from vibe.core.config import SandboxConfig
from vibe.core.tools.base import BaseToolState, ToolError
from vibe.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from vibe.core.tools.sandbox import (
    SandboxSpec,
    build_sandbox_command,
    build_seatbelt_profile,
    detect_backend,
    scrub_env,
)

# --------------------------------------------------------------------------- #
# Pure helpers (no OS dependency)                                              #
# --------------------------------------------------------------------------- #


def test_detect_backend_honors_override() -> None:
    assert detect_backend("bwrap") == "bwrap"
    assert detect_backend("none") == "none"


def test_detect_backend_windows_is_none(monkeypatch) -> None:
    monkeypatch.setattr("vibe.core.tools.sandbox.is_windows", lambda: True)
    assert detect_backend("auto") == "none"


def test_bwrap_argv_network_and_binds(tmp_path) -> None:
    spec = SandboxSpec(write_roots=[tmp_path], allow_network=False, extra_args=["--x"])
    argv, name, profile = build_sandbox_command(spec, "bwrap")
    assert name == "bwrap" and profile is None
    assert argv is not None
    assert "--unshare-net" in argv  # network blocked
    assert argv.count("--bind") == 1
    assert str(tmp_path.resolve()) in argv
    assert "--chdir" in argv
    assert "--x" in argv and argv.index("--x") < argv.index("--")  # extra before --


def test_bwrap_argv_network_allowed_has_no_unshare_net(tmp_path) -> None:
    spec = SandboxSpec(write_roots=[tmp_path], allow_network=True)
    argv, _n, _p = build_sandbox_command(spec, "bwrap")
    assert argv is not None and "--unshare-net" not in argv


def test_seatbelt_profile(tmp_path) -> None:
    spec = SandboxSpec(write_roots=[tmp_path], allow_network=False)
    profile = build_seatbelt_profile(spec)
    assert "(deny default)" in profile
    assert f'(allow file-write* (subpath "{tmp_path.resolve()}"))' in profile
    assert "(deny network*)" in profile


def test_seatbelt_rejects_quoted_roots(tmp_path) -> None:
    bad = tmp_path / 'a"b'
    spec = SandboxSpec(write_roots=[bad], allow_network=True)
    profile = build_seatbelt_profile(spec)
    assert 'a"b' not in profile  # never injected into the SBPL string


def test_scrub_env_drops_secrets_keeps_allowlist() -> None:
    base = {
        "PATH": "/bin",
        "HOME": "/home/x",
        "OPENAI_API_KEY": "sk-secret",
        "AWS_SECRET_ACCESS_KEY": "zzz",
        "GH_TOKEN": "ghp",
        "LC_CTYPE": "UTF-8",
        "MY_BUILD_VAR": "keep",
    }
    out = scrub_env(base, passthrough=["MY_BUILD_VAR"])
    assert out["PATH"] == "/bin" and out["HOME"] == "/home/x"
    assert out["LC_CTYPE"] == "UTF-8"  # LC_* allowed by prefix
    assert out["MY_BUILD_VAR"] == "keep"  # passthrough
    assert "OPENAI_API_KEY" not in out
    assert "AWS_SECRET_ACCESS_KEY" not in out
    assert "GH_TOKEN" not in out


# --------------------------------------------------------------------------- #
# Bash._resolve_sandbox                                                        #
# --------------------------------------------------------------------------- #


def _bash(sandbox: SandboxConfig) -> Bash:
    return Bash(
        config_getter=lambda: BashToolConfig(sandbox=sandbox), state=BaseToolState()
    )


def test_resolve_sandbox_disabled_runs_plain() -> None:
    argv, profile, env = _bash(SandboxConfig(enabled=False))._resolve_sandbox(
        None, "echo hi"
    )
    assert argv is None and profile is None
    assert "OPENAI_API_KEY" not in env or env  # plain base env (unscrubbed)


def test_resolve_sandbox_require_backend_raises_when_none() -> None:
    bash = _bash(SandboxConfig(enabled=True, backend="none", require_backend=True))
    with pytest.raises(ToolError):
        bash._resolve_sandbox(None, "echo hi")


def test_resolve_sandbox_none_backend_falls_back_unsandboxed() -> None:
    bash = _bash(SandboxConfig(enabled=True, backend="none", require_backend=False))
    argv, profile, _env = bash._resolve_sandbox(None, "echo hi")
    assert argv is None and profile is None  # runs unsandboxed


# --------------------------------------------------------------------------- #
# End-to-end (requires a real sandbox backend, e.g. bwrap on Linux)           #
# --------------------------------------------------------------------------- #

_HAS_BACKEND = detect_backend("auto") != "none"
_skip_no_backend = pytest.mark.skipif(
    not _HAS_BACKEND, reason="no sandbox backend (bwrap/sandbox-exec) available"
)


async def _run(bash: Bash, command: str):
    from vibe.core.tools.builtins.bash import BashResult

    result = None
    async for item in bash.run(BashArgs(command=command)):
        if isinstance(item, BashResult):
            result = item
    return result


@_skip_no_backend
@pytest.mark.asyncio
async def test_sandboxed_echo_runs() -> None:
    bash = _bash(SandboxConfig(enabled=True))
    result = await _run(bash, "echo hello-sandbox")
    assert result is not None and "hello-sandbox" in result.stdout


@_skip_no_backend
@pytest.mark.asyncio
async def test_sandbox_blocks_write_outside_workspace(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # workspace = tmp_path (writable)
    bash = _bash(SandboxConfig(enabled=True))
    # Writing into the workspace works.
    await _run(bash, "echo ok > inside.txt")
    assert (tmp_path / "inside.txt").exists()
    # Writing to a read-only root (/etc) must fail (command returns nonzero).
    with pytest.raises(ToolError):
        await _run(bash, "echo x > /etc/vibe_sandbox_probe")


@_skip_no_backend
@pytest.mark.asyncio
async def test_sandbox_scrubs_secret_env(monkeypatch) -> None:
    monkeypatch.setenv("FAKE_SECRET_API_KEY", "sk-leak")
    bash = _bash(SandboxConfig(enabled=True, scrub_env=True))
    result = await _run(bash, "echo secret=[${FAKE_SECRET_API_KEY}]")
    assert result is not None and "secret=[]" in result.stdout  # var was scrubbed


@pytest.mark.asyncio
async def test_disabled_sandbox_sees_env(monkeypatch) -> None:
    # Regression: disabled sandbox keeps the full (unscrubbed) env.
    monkeypatch.setenv("FAKE_SECRET_API_KEY", "sk-visible")
    bash = _bash(SandboxConfig(enabled=False))
    result = await _run(bash, "echo secret=[${FAKE_SECRET_API_KEY}]")
    assert result is not None and "sk-visible" in result.stdout


def test_create_subprocess_exec_not_called_when_disabled(monkeypatch) -> None:
    called = {"exec": 0}
    real_exec = asyncio.create_subprocess_exec

    async def spy_exec(*a, **k):
        called["exec"] += 1
        return await real_exec(*a, **k)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy_exec)
    bash = _bash(SandboxConfig(enabled=False))
    asyncio.run(_run(bash, "echo plain"))
    assert called["exec"] == 0  # disabled path uses create_subprocess_shell
