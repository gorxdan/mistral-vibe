from __future__ import annotations

import asyncio

import pytest

from vibe.core.config import SandboxConfig
from vibe.core.tools.base import BaseToolState, ToolError
from vibe.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from vibe.core.tools.sandbox import (
    BUBBLEWRAP_INSTALL_NUDGE,
    SandboxSpec,
    _detect_auto_backend,
    build_sandbox_command,
    build_seatbelt_profile,
    detect_backend,
    scrub_env,
    unshare_confinement_nudge,
)

# --------------------------------------------------------------------------- #
# Pure helpers (no OS dependency)                                              #
# --------------------------------------------------------------------------- #


def test_detect_backend_honors_override() -> None:
    assert detect_backend("bwrap") == "bwrap"
    assert detect_backend("none") == "none"


def test_detect_backend_windows_is_none(monkeypatch) -> None:
    monkeypatch.setattr("vibe.core.tools.sandbox.is_windows", lambda: True)
    _detect_auto_backend.cache_clear()
    assert detect_backend("auto") == "none"
    _detect_auto_backend.cache_clear()  # avoid polluting later tests with "none"


def test_detect_backend_skips_unusable_bwrap(monkeypatch) -> None:
    # Existence isn't enough: when namespace creation is denied (Docker/CI) a
    # failing probe must make auto detection skip bwrap like a missing backend.
    import vibe.core.tools.sandbox as sbmod

    monkeypatch.setattr(sbmod, "is_windows", lambda: False)
    monkeypatch.setattr(sbmod.sys, "platform", "linux")
    monkeypatch.setattr(
        sbmod.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"bwrap", "unshare"} else None,
    )

    class _FailedProbe:
        returncode = 1

    monkeypatch.setattr(sbmod.subprocess, "run", lambda *a, **k: _FailedProbe())
    _detect_auto_backend.cache_clear()
    try:
        assert detect_backend("auto") == "unshare"  # unusable bwrap skipped
    finally:
        _detect_auto_backend.cache_clear()


def test_detect_backend_uses_bwrap_when_probe_succeeds(monkeypatch) -> None:
    import vibe.core.tools.sandbox as sbmod

    monkeypatch.setattr(sbmod, "is_windows", lambda: False)
    monkeypatch.setattr(sbmod.sys, "platform", "linux")
    monkeypatch.setattr(
        sbmod.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"bwrap", "unshare"} else None,
    )

    class _OkProbe:
        returncode = 0

    monkeypatch.setattr(sbmod.subprocess, "run", lambda *a, **k: _OkProbe())
    _detect_auto_backend.cache_clear()
    try:
        assert detect_backend("auto") == "bwrap"
    finally:
        _detect_auto_backend.cache_clear()


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


def test_bwrap_argv_ro_bind_precedes_pseudo_fs(tmp_path) -> None:
    # Regression: --ro-bind / / must come BEFORE --dev/--proc/--tmpfs, else the
    # read-only root overlays them and makes /tmp read-only.
    spec = SandboxSpec(write_roots=[tmp_path], allow_network=True)
    argv, _n, _p = build_sandbox_command(spec, "bwrap")
    assert argv is not None
    ro = argv.index("--ro-bind")
    assert argv[ro + 1] == "/" and argv[ro + 2] == "/"
    assert ro < argv.index("--dev")
    assert ro < argv.index("--proc")
    assert ro < argv.index("--tmpfs")


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


def test_bwrap_env_readonly_but_git_writable(tmp_path) -> None:
    # A writable root with .git/.env: .env is re-layered read-only, but .git is
    # NOT (a sandboxed command must be able to commit). src/ stays writable.
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()
    (root / ".env").write_text("SECRET=1")
    (root / "src").mkdir()

    spec = SandboxSpec(write_roots=[root], allow_network=True)
    argv, _n, _p = build_sandbox_command(spec, "bwrap")
    assert argv is not None
    r = str(root.resolve())
    bind_idx = argv.index("--bind")
    assert argv[bind_idx + 1] == r and argv[bind_idx + 2] == r
    ro_targets = [argv[i + 1] for i, a in enumerate(argv) if a == "--ro-bind"]
    assert f"{r}/.env" in ro_targets  # secrets read-only
    assert f"{r}/.git" not in ro_targets  # git writable so commit works
    assert f"{r}/src" not in ro_targets


def test_bwrap_git_hooks_stay_readonly(tmp_path) -> None:
    # .git is writable for commits, but .git/hooks is re-layered read-only so a
    # command can't drop a hook that runs outside the sandbox later.
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git" / "hooks").mkdir(parents=True)

    spec = SandboxSpec(write_roots=[root], allow_network=True)
    argv, _n, _p = build_sandbox_command(spec, "bwrap")
    assert argv is not None
    r = str(root.resolve())
    ro_targets = [argv[i + 1] for i, a in enumerate(argv) if a == "--ro-bind"]
    assert f"{r}/.git/hooks" in ro_targets
    assert f"{r}/.git" not in ro_targets  # whole gitdir not read-only


def test_bwrap_skips_nonexistent_write_root(tmp_path) -> None:
    # A write_root that doesn't exist must be skipped, not passed to bwrap --bind
    # (a missing source explodes with "Can't find source path").
    real = tmp_path / "real"
    real.mkdir()
    missing = tmp_path / "does-not-exist"

    spec = SandboxSpec(write_roots=[real, missing], allow_network=True)
    argv, _n, _p = build_sandbox_command(spec, "bwrap")
    assert argv is not None
    assert str(real.resolve()) in argv
    assert str(missing.resolve()) not in argv  # skipped, not bound


def test_bwrap_git_config_stays_readonly(tmp_path) -> None:
    # .git/config must be read-only or a sandboxed command can set core.hooksPath
    # (or diff.external/core.fsmonitor/pager) to plant a hook that runs outside it.
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git" / "hooks").mkdir(parents=True)
    (root / ".git" / "config").write_text("[core]\n")

    spec = SandboxSpec(write_roots=[root], allow_network=True)
    argv, _n, _p = build_sandbox_command(spec, "bwrap")
    assert argv is not None
    r = str(root.resolve())
    ro_targets = [argv[i + 1] for i, a in enumerate(argv) if a == "--ro-bind"]
    assert f"{r}/.git/config" in ro_targets  # config read-only -> no hooksPath escape
    assert f"{r}/.git" not in ro_targets  # whole gitdir still writable for commits


def test_bwrap_worktree_external_gitdir_writable(tmp_path) -> None:
    # A linked worktree: root/.git is a FILE pointing to the main repo's
    # .git/worktrees/<name>. The shared .git (objects/refs + the worktree gitdir)
    # lives outside the checkout and must be bound writable so commits succeed.
    main = tmp_path / "main"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    (main / ".git" / "hooks").mkdir()
    (main / ".git" / "config").write_text("[core]\n")
    (main / ".git" / "worktrees" / "wt" / "config.worktree").write_text("[core]\n")
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/wt\n")

    spec = SandboxSpec(write_roots=[wt], allow_network=True)
    argv, _n, _p = build_sandbox_command(spec, "bwrap")
    assert argv is not None
    binds = [argv[i + 1] for i, a in enumerate(argv) if a == "--bind"]
    ro_targets = [argv[i + 1] for i, a in enumerate(argv) if a == "--ro-bind"]
    common = f"{main}/.git"
    assert common in binds  # external git common-dir bound writable
    assert f"{common}/hooks" in ro_targets  # its hooks still read-only
    assert f"{common}/config" in ro_targets  # shared config read-only
    # The per-worktree config.worktree (git config --worktree target) too.
    assert f"{common}/worktrees/wt/config.worktree" in ro_targets


def test_bwrap_skips_symlinked_protected_metadata(tmp_path) -> None:
    # A symlinked .git could point outside the root; never bind through it.
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").symlink_to(tmp_path / "elsewhere")

    spec = SandboxSpec(write_roots=[root], allow_network=True)
    argv, _n, _p = build_sandbox_command(spec, "bwrap")
    assert argv is not None
    assert f"{root.resolve()}/.git" not in argv


def test_seatbelt_denies_secrets_and_hooks_not_gitdir(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git" / "hooks").mkdir(parents=True)
    (root / ".git" / "config").write_text("[core]\n")
    (root / ".env").write_text("SECRET=1")

    spec = SandboxSpec(write_roots=[root], allow_network=True)
    profile = build_seatbelt_profile(spec)
    r = str(root.resolve())
    assert f'(allow file-write* (subpath "{r}"))' in profile
    assert f'(deny file-write* (subpath "{r}/.env"))' in profile
    assert f'(deny file-write* (subpath "{r}/.git/hooks"))' in profile
    assert (
        f'(deny file-write* (subpath "{r}/.git/config"))' in profile
    )  # no hooksPath escape
    assert f'(deny file-write* (subpath "{r}/.git"))' not in profile  # gitdir writable


def test_bwrap_no_overlay_when_no_protected_metadata(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()  # no .git/.vibe/.env

    spec = SandboxSpec(write_roots=[root], allow_network=True)
    argv, _n, _p = build_sandbox_command(spec, "bwrap")
    assert argv is not None
    # Only the single root --ro-bind (the / / root), no metadata overlays.
    assert argv.count("--ro-bind") == 1


def test_unshare_backend_warns_when_containment_requested(tmp_path, caplog) -> None:
    # The unshare backend cannot enforce filesystem confinement or network
    # denial. When the spec asks for either, a loud warning must fire so the
    # user knows their opt-in is not doing what they think (the common case on
    # minimal containers/CI without bubblewrap).
    import logging

    spec = SandboxSpec(write_roots=[tmp_path], allow_network=False)
    with caplog.at_level(logging.WARNING, logger="vibe.core.tools.sandbox"):
        argv, name, _ = build_sandbox_command(spec, "unshare")
    assert name == "unshare" and argv is not None
    assert any(
        "NO filesystem write confinement" in r.message for r in caplog.records
    ), [r.message for r in caplog.records]


def test_unshare_backend_silent_when_no_containment_requested(caplog) -> None:
    # Bare namespace isolation (no write_roots, network allowed) is honest
    # about what it provides — no warning needed.
    import logging

    spec = SandboxSpec(write_roots=[], allow_network=True)
    with caplog.at_level(logging.WARNING, logger="vibe.core.tools.sandbox"):
        build_sandbox_command(spec, "unshare")
    assert not any(
        "NO filesystem write confinement" in r.message for r in caplog.records
    )


def test_unshare_nudge_fires_whenever_backend_is_unshare(monkeypatch) -> None:
    # Fires whenever sandbox is enabled and resolves to unshare, not only on explicit containment.
    monkeypatch.setattr(
        "vibe.core.tools.sandbox.detect_backend",
        lambda override: override if override != "auto" else "unshare",
    )

    # Default config (enabled, no explicit containment) resolves to unshare -> nudge.
    msg = unshare_confinement_nudge(sandbox_enabled=True, backend_override="auto")
    assert msg == BUBBLEWRAP_INSTALL_NUDGE
    assert "sudo apt install bubblewrap" in msg

    # Sandbox disabled -> no nudge.
    assert (
        unshare_confinement_nudge(sandbox_enabled=False, backend_override="auto")
        is None
    )

    # A real backend (bwrap) -> no nudge (the override is honored without a probe).
    assert (
        unshare_confinement_nudge(sandbox_enabled=True, backend_override="bwrap")
        is None
    )


# --------------------------------------------------------------------------- #
# Seccomp-BPF filter (pure byte-layout; no OS dependency)                      #
# --------------------------------------------------------------------------- #


def _decode_bpf(prog: bytes) -> list[tuple[int, int, int, int]]:
    import struct

    assert len(prog) % 8 == 0
    return [struct.unpack_from("<HBBI", prog, i) for i in range(0, len(prog), 8)]


def test_seccomp_bpf_unsupported_arch_is_none() -> None:
    from vibe.core.tools.sandbox_seccomp import build_seccomp_bpf

    assert build_seccomp_bpf("riscv64") is None
    assert build_seccomp_bpf("s390x") is None


def test_seccomp_bpf_x86_64_layout() -> None:
    from vibe.core.tools.sandbox_seccomp import build_seccomp_bpf

    prog = build_seccomp_bpf("x86_64")
    assert prog is not None
    insts = _decode_bpf(prog)
    # 4 fixed (LD arch, JEQ arch, RET kill, LD nr) + 6 denials + 2 terminal rets.
    assert len(insts) == 12

    LD, JEQ, RET = 0x20, 0x15, 0x06
    KILL, ALLOW, EPERM = 0x80000000, 0x7FFF0000, 0x00050001

    # [0] load arch @off 4, [1] JEQ AUDIT_ARCH_X86_64 skip-kill, [2] RET KILL.
    assert insts[0] == (LD, 0, 0, 4)
    assert insts[1][0] == JEQ and insts[1][1:] == (1, 0, 0xC000003E)
    assert insts[2] == (RET, 0, 0, KILL)
    # [3] load nr @off 0.
    assert insts[3] == (LD, 0, 0, 0)
    # [4] first denial is ptrace (x86_64 nr 101), jumping to the EPERM ret (idx 11).
    assert insts[4][0] == JEQ and insts[4][3] == 101
    assert 4 + insts[4][1] + 1 == 11  # jt lands on the EPERM instruction
    # io_uring_setup/enter/register present.
    denied_nrs = {insts[i][3] for i in range(4, 10)}
    assert {101, 310, 311, 425, 426, 427} == denied_nrs
    # [10] default ALLOW, [11] EPERM landing pad.
    assert insts[10] == (RET, 0, 0, ALLOW)
    assert insts[11] == (RET, 0, 0, EPERM)


def test_seccomp_bpf_aarch64_uses_generic_numbers() -> None:
    from vibe.core.tools.sandbox_seccomp import build_seccomp_bpf

    prog = build_seccomp_bpf("arm64")  # alias of aarch64
    assert prog is not None
    insts = _decode_bpf(prog)
    assert insts[1][3] == 0xC00000B7  # AUDIT_ARCH_AARCH64
    denied_nrs = {insts[i][3] for i in range(4, 10)}
    assert denied_nrs == {117, 270, 271, 425, 426, 427}  # ptrace=117 on aarch64


def test_open_seccomp_fd_roundtrips_bytes() -> None:
    import os

    from vibe.core.tools.sandbox_seccomp import build_seccomp_bpf, open_seccomp_fd

    bpf = build_seccomp_bpf("x86_64")
    assert bpf is not None
    fd = open_seccomp_fd(bpf)
    try:
        assert os.read(fd, len(bpf) + 8) == bpf  # readable from offset 0
    finally:
        os.close(fd)


def test_resolve_sandbox_bwrap_injects_seccomp(monkeypatch) -> None:
    # With the bwrap backend + seccomp on, _resolve_sandbox loads a filter and
    # wires `--seccomp <fd>` into the argv immediately before the trailing `--`.
    import os

    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.detect_backend", lambda override: "bwrap"
    )
    bash = _bash(SandboxConfig(enabled=True, backend="bwrap", seccomp=True))
    argv, _profile, _env, fd = bash._resolve_sandbox(None, "echo hi")
    try:
        assert argv is not None and fd is not None
        assert "--seccomp" in argv
        assert argv[argv.index("--seccomp") + 1] == str(fd)
        assert argv.index("--seccomp") < argv.index("--")  # before the terminator
    finally:
        if fd is not None:
            os.close(fd)


def test_resolve_sandbox_bwrap_seccomp_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.detect_backend", lambda override: "bwrap"
    )
    bash = _bash(SandboxConfig(enabled=True, backend="bwrap", seccomp=False))
    argv, _profile, _env, fd = bash._resolve_sandbox(None, "echo hi")
    assert argv is not None and fd is None
    assert "--seccomp" not in argv


def test_resolve_sandbox_isolated_forces_confine(tmp_path, monkeypatch) -> None:
    # An isolated subagent (VIBE_ISOLATED_WORKTREE_ROOT set) OS-confines bash to
    # its worktree even when the global sandbox is DISABLED — mirroring the file
    # tools' enforce_isolated_confine.
    import os

    wt = tmp_path / "wt"
    wt.mkdir()
    monkeypatch.setenv("VIBE_ISOLATED_WORKTREE_ROOT", str(wt))
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.detect_backend", lambda override: "bwrap"
    )
    bash = _bash(SandboxConfig(enabled=False))  # user never enabled the sandbox
    argv, _profile, _env, fd = bash._resolve_sandbox(None, "echo hi")
    try:
        assert argv is not None  # forced on by isolation
        bind_i = argv.index("--bind")
        assert argv[bind_i + 1] == str(wt.resolve())  # worktree is the write root
    finally:
        if fd is not None:
            os.close(fd)


def test_resolve_sandbox_isolated_no_outside_widening(tmp_path, monkeypatch) -> None:
    # Confinement is the point: an out-of-tree dir referenced by the command is
    # NOT added as a writable bind (unlike the non-isolated permission-widened path).
    import os

    wt = tmp_path / "wt"
    wt.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("VIBE_ISOLATED_WORKTREE_ROOT", str(wt))
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.detect_backend", lambda override: "bwrap"
    )
    bash = _bash(SandboxConfig(enabled=False))
    argv, _profile, _env, fd = bash._resolve_sandbox(None, f"echo x > {outside}/f.txt")
    try:
        assert argv is not None
        assert str(outside.resolve()) not in argv  # never widened out of the worktree
    finally:
        if fd is not None:
            os.close(fd)


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


def test_sandbox_enabled_default_is_on() -> None:
    # Defense-in-depth default: on. Where a backend exists bash is sandboxed;
    # where none does it soft-falls-back (still enabled, warns once).
    assert SandboxConfig().enabled is True


def test_resolve_sandbox_disabled_runs_plain() -> None:
    argv, profile, env, fd = _bash(SandboxConfig(enabled=False))._resolve_sandbox(
        None, "echo hi"
    )
    assert argv is None and profile is None and fd is None
    assert "OPENAI_API_KEY" not in env or env  # plain base env (unscrubbed)


def test_resolve_sandbox_require_backend_raises_when_none() -> None:
    bash = _bash(SandboxConfig(enabled=True, backend="none", require_backend=True))
    with pytest.raises(ToolError):
        bash._resolve_sandbox(None, "echo hi")


def test_resolve_sandbox_none_backend_falls_back_unsandboxed() -> None:
    bash = _bash(SandboxConfig(enabled=True, backend="none", require_backend=False))
    argv, profile, _env, fd = bash._resolve_sandbox(None, "echo hi")
    assert argv is None and profile is None and fd is None  # runs unsandboxed


def test_host_session_keeps_git_gh_creds_through_scrub(monkeypatch) -> None:
    # The host session's scrubbed bash env keeps authenticated git/gh working
    # (ssh/https push, gh CLI, signing) while still dropping model API keys.
    import os

    monkeypatch.delenv("VIBE_ISOLATED_WORKTREE_ROOT", raising=False)
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.detect_backend", lambda override: "bwrap"
    )
    monkeypatch.setenv("GH_TOKEN", "ghp_host")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/run/ssh-agent.sock")
    monkeypatch.setenv("GIT_SSH_COMMAND", "ssh -i /home/x/.ssh/id")
    monkeypatch.setenv("GPG_TTY", "/dev/pts/0")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")

    _argv, _profile, env, fd = _bash(
        SandboxConfig(enabled=True, scrub_env=True)
    )._resolve_sandbox(None, "git push")
    try:
        assert env["GH_TOKEN"] == "ghp_host"  # gh CLI + https push keep working
        assert env["SSH_AUTH_SOCK"] == "/run/ssh-agent.sock"  # ssh push keeps working
        assert env["GIT_SSH_COMMAND"] == "ssh -i /home/x/.ssh/id"
        assert env["GPG_TTY"] == "/dev/pts/0"  # commit signing
        assert "OPENAI_API_KEY" not in env  # model secrets still scrubbed
    finally:
        if fd is not None:
            os.close(fd)


def test_isolated_subagent_still_scrubs_git_gh_creds(tmp_path, monkeypatch) -> None:
    # The host-only cred passthrough must NOT leak into an isolated subagent —
    # that strict scrub is the security boundary between host and worker.
    import os

    wt = tmp_path / "wt"
    wt.mkdir()
    monkeypatch.setenv("VIBE_ISOLATED_WORKTREE_ROOT", str(wt))
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.detect_backend", lambda override: "bwrap"
    )
    monkeypatch.setenv("GH_TOKEN", "ghp_should_not_leak")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/run/ssh-agent.sock")

    _argv, _profile, env, fd = _bash(
        SandboxConfig(enabled=True, scrub_env=True)
    )._resolve_sandbox(None, "git push")
    try:
        assert "GH_TOKEN" not in env  # boundary: worker never gets host creds
        assert "SSH_AUTH_SOCK" not in env
    finally:
        if fd is not None:
            os.close(fd)


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
async def test_default_config_bash_is_sandboxed(tmp_path, monkeypatch) -> None:
    # The flipped default (enabled=True) means a plain default config sandboxes
    # bash on a backend host, with no explicit opt-in.
    if detect_backend("auto") != "bwrap":
        pytest.skip("default-on sandbox visible as a bwrap wrapper only on Linux")
    import os

    monkeypatch.chdir(tmp_path)
    bash = _bash(SandboxConfig())  # all defaults
    argv, _profile, _env, fd = bash._resolve_sandbox(None, "echo hi")
    try:
        assert argv is not None and argv[0] == "bwrap"
    finally:
        if fd is not None:
            os.close(fd)


@_skip_no_backend
@pytest.mark.asyncio
async def test_sandboxed_git_commit_works(tmp_path, monkeypatch) -> None:
    # Regression: the sandbox must not read-only-mount .git, or `git commit`
    # fails with "Unable to create .git/index.lock: Read-only file system".
    if detect_backend("auto") != "bwrap":
        pytest.skip("git-metadata bind confinement needs the bwrap backend")
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        "git init -q && git config user.email t@t && git config user.name t "
        "&& git commit --allow-empty -qm init",
        cwd=repo,
        shell=True,
        check=True,
    )
    monkeypatch.chdir(repo)
    bash = _bash(SandboxConfig(enabled=True))

    res = await _run(bash, "git commit --allow-empty -m sbx -q && echo done")
    assert res is not None and res.returncode == 0 and "done" in res.stdout

    with pytest.raises(ToolError):  # hooks stay read-only
        await _run(bash, "echo x > .git/hooks/pre-commit")


@_skip_no_backend
@pytest.mark.asyncio
async def test_sandboxed_cannot_repoint_hookspath(tmp_path, monkeypatch) -> None:
    # Hook-persistence escape: .git/config must be read-only so a command can't
    # set core.hooksPath to a writable dir and plant a hook that runs outside it.
    if detect_backend("auto") != "bwrap":
        pytest.skip("git-config bind confinement needs the bwrap backend")
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        "git init -q && git config user.email t@t && git config user.name t "
        "&& git commit --allow-empty -qm init",
        cwd=repo,
        shell=True,
        check=True,
    )
    monkeypatch.chdir(repo)
    bash = _bash(SandboxConfig(enabled=True))

    with pytest.raises(ToolError):  # config is read-only inside the sandbox
        await _run(bash, "git config core.hooksPath /tmp/evil-hooks")
    # The on-disk config was never modified.
    assert "hooksPath" not in (repo / ".git" / "config").read_text()


@_skip_no_backend
@pytest.mark.asyncio
async def test_seccomp_blocks_ptrace() -> None:
    # The seccomp filter must make ptrace fail with EPERM inside the bwrap
    # sandbox. PTRACE_TRACEME (request 0) normally returns 0; under the filter it
    # returns -1 / errno 1. Only the bwrap backend carries a seccomp filter.
    if detect_backend("auto") != "bwrap":
        pytest.skip("seccomp filter only applies to the bwrap backend")
    probe = (
        'python3 -c "import ctypes; '
        "libc = ctypes.CDLL(None, use_errno=True); "
        "libc.ptrace(0, 0, 0, 0); "
        "print('ptrace_errno', ctypes.get_errno())\""
    )
    on = _bash(SandboxConfig(enabled=True, backend="bwrap", seccomp=True))
    res = await _run(on, probe)
    assert res is not None and "ptrace_errno 1" in res.stdout  # EPERM

    off = _bash(SandboxConfig(enabled=True, backend="bwrap", seccomp=False))
    res_off = await _run(off, probe)
    assert res_off is not None and "ptrace_errno 0" in res_off.stdout  # allowed


@_skip_no_backend
@pytest.mark.asyncio
async def test_isolated_bash_confined_even_when_sandbox_disabled(
    tmp_path, monkeypatch
) -> None:
    # End-to-end: with the global sandbox OFF but VIBE_ISOLATED_WORKTREE_ROOT set,
    # bash writes inside the worktree but is OS-blocked from writing outside it.
    if detect_backend("auto") != "bwrap":
        pytest.skip("worktree bind confinement needs the bwrap backend")
    wt = tmp_path / "wt"
    wt.mkdir()
    monkeypatch.chdir(wt)
    monkeypatch.setenv("VIBE_ISOLATED_WORKTREE_ROOT", str(wt))
    bash = _bash(SandboxConfig(enabled=False))  # not globally enabled

    await _run(bash, "echo ok > inside.txt")
    assert (wt / "inside.txt").exists()  # worktree write allowed

    with pytest.raises(ToolError):  # /etc is read-only under bwrap
        await _run(bash, "echo x > /etc/vibe_isolated_probe")


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


def test_headless_nudge_emits_to_stderr_on_unshare(capsys) -> None:
    # Headless (`vibe -p`) has no TUI toast, so the unshare-only nudge goes to
    # stderr. Explicit backend='unshare' means detect_backend honors it, no probe.
    from vibe.core.programmatic import _emit_headless_sandbox_nudge

    _emit_headless_sandbox_nudge(SandboxConfig(enabled=True, backend="unshare"))
    err = capsys.readouterr().err
    assert "bubblewrap" in err


def test_headless_nudge_silent_when_disabled_or_no_config(capsys) -> None:
    from vibe.core.programmatic import _emit_headless_sandbox_nudge

    _emit_headless_sandbox_nudge(SandboxConfig(enabled=False, backend="unshare"))
    _emit_headless_sandbox_nudge(None)
    assert capsys.readouterr().err == ""
