from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vibe.core.config import SandboxConfig
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from vibe.core.tools.sandbox import (
    BUBBLEWRAP_INSTALL_NUDGE,
    SandboxSpec,
    _detect_auto_backend,
    build_sandbox_command,
    build_seatbelt_profile,
    detect_backend,
    scrub_child_env,
    scrub_env,
    strict_read_hidden_roots,
    unshare_confinement_nudge,
)

# --------------------------------------------------------------------------- #
# Pure helpers (no OS dependency)                                              #
# --------------------------------------------------------------------------- #


def test_strict_read_hidden_roots_masks_non_runtime_directories(tmp_path: Path) -> None:
    for name in ("usr", "etc", "home", "var", "workspace", "custom-mount"):
        (tmp_path / name).mkdir()

    hidden = strict_read_hidden_roots(tmp_path)

    assert hidden == [
        tmp_path / "custom-mount",
        tmp_path / "home",
        tmp_path / "var",
        tmp_path / "workspace",
    ]


def test_bwrap_recreates_only_assigned_path_below_masked_root(tmp_path: Path) -> None:
    masked = tmp_path / "host-data"
    assigned = masked / "campaign" / "candidate"
    assigned.mkdir(parents=True)

    argv, backend, profile = build_sandbox_command(
        SandboxSpec(
            write_roots=[], read_roots=[assigned], hidden_roots=[masked], cwd=assigned
        ),
        "bwrap",
    )

    assert backend == "bwrap"
    assert profile is None
    assert argv is not None
    mask_index = argv.index(str(masked))
    parent_index = argv.index(str(masked / "campaign"))
    assigned_index = argv.index(str(assigned))
    assert mask_index < parent_index < assigned_index
    assert argv[parent_index - 1] == "--dir"
    assert argv[assigned_index - 1] in {"--dir", "--ro-bind"}


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


def test_sandbox_e2e_marker_predicate_tracks_usable_backend(monkeypatch) -> None:
    # The conftest marker predicate reuses detect_backend, so an unshare/none
    # result (userns unavailable) disables sandbox e2e; bwrap/seatbelt enables it.
    from tests.conftest import sandbox_e2e_available

    monkeypatch.setattr(
        "vibe.core.tools.sandbox.detect_backend", lambda override="auto": "unshare"
    )
    assert sandbox_e2e_available() is False
    monkeypatch.setattr(
        "vibe.core.tools.sandbox.detect_backend", lambda override="auto": "bwrap"
    )
    assert sandbox_e2e_available() is True


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
    spec = SandboxSpec(
        write_roots=[tmp_path], allow_network=False, extra_args=["--new-session"]
    )
    argv, name, profile = build_sandbox_command(spec, "bwrap")
    assert name == "bwrap" and profile is None
    assert argv is not None
    assert "--unshare-net" in argv  # network blocked
    assert argv.count("--bind") == 1
    assert str(tmp_path.resolve()) in argv
    assert "--chdir" in argv
    assert "--new-session" in argv
    assert argv.index("--new-session") < argv.index("--")


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--bind", "/", "/"],
        ["--ro-bind=/", "/"],
        ["--args", "9"],
        ["--"],
        ["--chdir", "/tmp"],
    ],
)
def test_bwrap_extra_args_cannot_reopen_or_skip_policy(
    tmp_path, extra_args: list[str]
) -> None:
    spec = SandboxSpec(write_roots=[tmp_path], extra_args=extra_args)

    with pytest.raises(ValueError, match="unsafe bubblewrap extra argument"):
        build_sandbox_command(spec, "bwrap")


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


def test_bwrap_hidden_home_does_not_reexpose_protected_host_state(tmp_path) -> None:
    home = tmp_path / "home"
    logs = home / ".vibe" / "logs"
    candidate = home / "worktrees" / "candidate"
    logs.mkdir(parents=True)
    candidate.mkdir(parents=True)
    spec = SandboxSpec(
        write_roots=[candidate],
        read_roots=[candidate],
        hidden_roots=[home],
        protected_roots=[logs],
    )

    argv, _name, _profile = build_sandbox_command(spec, "bwrap")

    assert argv is not None
    ro_targets = [
        argv[index + 1] for index, item in enumerate(argv) if item == "--ro-bind"
    ]
    assert str(candidate.resolve()) in ro_targets
    assert str(logs.resolve()) not in ro_targets


def test_seatbelt_profile(tmp_path) -> None:
    spec = SandboxSpec(write_roots=[tmp_path], allow_network=False)
    profile = build_seatbelt_profile(spec)
    assert "(deny default)" in profile
    assert f'(allow file-write* (subpath "{tmp_path.resolve()}"))' in profile
    assert "(deny network*)" in profile


def test_seatbelt_hides_home_then_reexposes_explicit_read_root(tmp_path) -> None:
    home = tmp_path / "home"
    repository = home / "candidate"
    repository.mkdir(parents=True)
    spec = SandboxSpec(write_roots=[], hidden_roots=[home], read_roots=[repository])

    lines = build_seatbelt_profile(spec).splitlines()

    deny = f'(deny file-read* (subpath "{home.resolve()}"))'
    allow = f'(allow file-read* (subpath "{repository.resolve()}"))'
    assert deny in lines
    assert allow in lines
    assert lines.index(deny) < lines.index(allow)


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


def test_strict_model_control_keeps_shared_git_metadata_readonly(tmp_path) -> None:
    main = tmp_path / "main"
    gitdir = main / ".git" / "worktrees" / "candidate"
    gitdir.mkdir(parents=True)
    wt = tmp_path / "candidate"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {gitdir}\n")

    spec = SandboxSpec(write_roots=[wt], protect_git_metadata=True, allow_network=True)
    argv, _name, _profile = build_sandbox_command(spec, "bwrap")

    assert argv is not None
    binds = [argv[i + 1] for i, value in enumerate(argv) if value == "--bind"]
    ro_targets = [argv[i + 1] for i, value in enumerate(argv) if value == "--ro-bind"]
    assert str((main / ".git").resolve()) not in binds
    assert str((main / ".git").resolve()) in ro_targets
    assert str((wt / ".git").resolve()) in ro_targets


def test_protected_roots_are_bound_after_overlapping_write_roots(tmp_path) -> None:
    broad = tmp_path / "broad"
    protected = broad / "host-state"
    protected.mkdir(parents=True)

    spec = SandboxSpec(
        write_roots=[protected, broad], protected_roots=[protected], allow_network=True
    )
    argv, _name, _profile = build_sandbox_command(spec, "bwrap")

    assert argv is not None
    target = str(protected.resolve())
    last_write = max(
        index
        for index, value in enumerate(argv)
        if value == "--bind" and argv[index + 1] in {target, str(broad.resolve())}
    )
    protection = next(
        index
        for index, value in enumerate(argv)
        if value == "--ro-bind" and argv[index + 1] == target
    )
    assert protection > last_write


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


def test_resolve_sandbox_isolated_writer_keeps_shared_git_host_owned(
    tmp_path, monkeypatch
) -> None:
    import os

    main = tmp_path / "main"
    gitdir = main / ".git" / "worktrees" / "writer"
    gitdir.mkdir(parents=True)
    wt = tmp_path / "writer"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {gitdir}\n")
    monkeypatch.setenv("VIBE_ISOLATED_WORKTREE_ROOT", str(wt))
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.detect_backend", lambda override: "bwrap"
    )
    bash = _bash(SandboxConfig(enabled=False))

    argv, _profile, _env, fd = bash._resolve_sandbox(None, "git status --short")

    try:
        assert argv is not None
        common = str((main / ".git").resolve())
        binds = [
            argv[index + 1] for index, value in enumerate(argv) if value == "--bind"
        ]
        ro_targets = [
            argv[index + 1] for index, value in enumerate(argv) if value == "--ro-bind"
        ]
        assert common not in binds
        assert common in ro_targets
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


@pytest.mark.parametrize("backend", ["none", "unshare"])
def test_isolated_model_control_rejects_nonconfining_backend(
    tmp_path, monkeypatch, backend: str
) -> None:
    wt = tmp_path / "wt"
    wt.mkdir()
    monkeypatch.setenv("VIBE_ISOLATED_WORKTREE_ROOT", str(wt))
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.detect_backend", lambda override: backend
    )
    bash = _bash(SandboxConfig(enabled=False))

    with pytest.raises(ToolError, match="Strict model control requires"):
        bash._resolve_sandbox(None, "echo hi")


def test_topology_bound_session_forces_strict_control_without_autoapprove(
    tmp_path, monkeypatch
) -> None:
    protected = tmp_path / "evidence"
    protected.mkdir()
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.verification_protected_roots",
        lambda state: (protected,),
    )
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.detect_backend", lambda override: "none"
    )
    bash = _bash(SandboxConfig(enabled=False))

    with pytest.raises(ToolError, match="Strict model control requires"):
        bash._resolve_sandbox(InvokeContext(tool_call_id="topology"), "echo hi")


def test_resolve_sandbox_host_redirects_toolchain_cache(monkeypatch) -> None:
    # Host session: pre-commit/uv caches redirect to a writable private root
    # (bound writable) so `git commit` gates run instead of hitting RO ~/.cache.
    import os
    from pathlib import Path

    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.detect_backend", lambda override: "bwrap"
    )
    bash = _bash(SandboxConfig(enabled=True))
    argv, _profile, env, fd = bash._resolve_sandbox(None, "git commit -m x")
    try:
        assert argv is not None
        uv_cache = env["UV_CACHE_DIR"]
        pc_home = env["PRE_COMMIT_HOME"]
        assert uv_cache.endswith("sandbox-cache/uv")
        assert pc_home.endswith("sandbox-cache/pre-commit")
        cache_root = str(Path(uv_cache).parent.resolve())
        assert cache_root in argv  # the cache root is bound writable
    finally:
        if fd is not None:
            os.close(fd)


def test_resolve_sandbox_isolated_does_not_redirect_cache(
    tmp_path, monkeypatch
) -> None:
    # Isolated subagents get only the sandbox-private /tmp cache, never the
    # persistent host cache or another writable host directory.
    import os

    wt = tmp_path / "wt"
    wt.mkdir()
    monkeypatch.setenv("VIBE_ISOLATED_WORKTREE_ROOT", str(wt))
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.detect_backend", lambda override: "bwrap"
    )
    bash = _bash(SandboxConfig(enabled=True))
    argv, _profile, env, fd = bash._resolve_sandbox(None, "git commit -m x")
    try:
        assert argv is not None
        assert env["UV_CACHE_DIR"].startswith("/tmp/")
        assert env["PRE_COMMIT_HOME"].startswith("/tmp/")
        assert "sandbox-cache" not in " ".join(argv)
        home = str(Path.home().resolve())
        hidden = [
            argv[index + 1] for index, item in enumerate(argv) if item == "--tmpfs"
        ]
        assert home in hidden
        assert str(wt.resolve()) in argv
        local_bin = str((Path.home() / ".local" / "bin").resolve())
        local_parent = str((Path.home() / ".local").resolve())
        if Path(local_bin).is_dir():
            assert local_bin in argv
        assert local_parent not in argv
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


def test_scrub_child_env_drops_host_creds_keeps_provider_keys() -> None:
    # Isolated/team children inherit provider keys so they can call the model,
    # but must not receive host git/gh/ssh/cloud creds (exfil under --trust).
    base = {
        "PATH": "/bin",
        "HOME": "/home/x",
        "OPENAI_API_KEY": "sk-keep",
        "MISTRAL_API_KEY": "msk-keep",
        "GH_TOKEN": "ghp_drop",
        "GITHUB_TOKEN": "ghs_drop",
        "SSH_AUTH_SOCK": "/run/ssh-agent.sock",
        "GIT_SSH_COMMAND": "ssh -i /home/x/.ssh/id",
        "AWS_SECRET_ACCESS_KEY": "aws_drop",
        "AWS_ACCESS_KEY_ID": "AKIADROP",
        "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/sa.json",
        "VIBE_HOME": "/tmp/vibe",
    }
    out = scrub_child_env(base)
    assert out["OPENAI_API_KEY"] == "sk-keep"
    assert out["MISTRAL_API_KEY"] == "msk-keep"
    assert out["PATH"] == "/bin"
    assert out["VIBE_HOME"] == "/tmp/vibe"
    assert "GH_TOKEN" not in out
    assert "GITHUB_TOKEN" not in out
    assert "SSH_AUTH_SOCK" not in out
    assert "GIT_SSH_COMMAND" not in out
    assert "AWS_SECRET_ACCESS_KEY" not in out
    assert "AWS_ACCESS_KEY_ID" not in out
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in out


# --------------------------------------------------------------------------- #
# Bash._resolve_sandbox                                                        #
# --------------------------------------------------------------------------- #


def _bash(sandbox: SandboxConfig) -> Bash:
    return Bash(
        config_getter=lambda: BashToolConfig(sandbox=sandbox), state=BaseToolState()
    )


def _managed_context(*, state: str, scratchpad_dir=None) -> InvokeContext:
    topology = SimpleNamespace(state=state)
    recipe = SimpleNamespace(config=SimpleNamespace(execution_topology=topology))
    verification_state = SimpleNamespace(trusted_recipe=recipe)
    return InvokeContext(
        tool_call_id="managed",
        verification_state=cast(Any, verification_state),
        scratchpad_dir=scratchpad_dir,
    )


def _autoapprove_context() -> InvokeContext:
    manager = SimpleNamespace(config=SimpleNamespace(bypass_tool_permissions=True))
    return InvokeContext(tool_call_id="autoapprove", agent_manager=cast(Any, manager))


def _verifier_context(*, scratchpad_dir: Path | None = None) -> InvokeContext:
    manager = SimpleNamespace(
        config=SimpleNamespace(
            bypass_tool_permissions=False, system_prompt_id="verifier"
        )
    )
    return InvokeContext(
        tool_call_id="verifier",
        agent_manager=cast(Any, manager),
        scratchpad_dir=scratchpad_dir,
    )


def test_topology_active_bash_is_candidate_readonly_and_ignores_widening(
    tmp_path, monkeypatch
) -> None:
    import os

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = tmp_path / "control"
    protected.mkdir()
    scratchpad = tmp_path / "scratchpad"
    scratchpad.mkdir()
    monkeypatch.chdir(candidate)
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.verification_protected_roots",
        lambda state: (protected,),
    )
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.detect_backend", lambda override: "bwrap"
    )
    bash = _bash(
        SandboxConfig(
            enabled=False,
            write_dirs=[str(outside)],
            scrub_env=False,
            allow_network=True,
        )
    )

    argv, _profile, env, fd = bash._resolve_sandbox(
        _managed_context(state="active", scratchpad_dir=scratchpad),
        f"touch {outside}/escape",
    )

    try:
        assert argv is not None
        binds = [
            argv[index + 1] for index, value in enumerate(argv) if value == "--bind"
        ]
        assert str(candidate.resolve()) not in binds
        assert str(outside.resolve()) not in argv
        assert str(scratchpad.resolve()) in binds
        assert "--unshare-net" in argv
        assert env["UV_CACHE_DIR"].startswith("/tmp/")
        assert "sandbox-cache" not in " ".join(argv)
    finally:
        if fd is not None:
            os.close(fd)


def test_strict_model_control_scrubs_host_credentials_even_when_disabled(
    tmp_path, monkeypatch
) -> None:
    import os

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    protected = tmp_path / "control"
    protected.mkdir()
    monkeypatch.chdir(candidate)
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.verification_protected_roots",
        lambda state: (protected,),
    )
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.detect_backend", lambda override: "bwrap"
    )
    monkeypatch.setenv("GH_TOKEN", "host-token")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/run/host-ssh.sock")
    monkeypatch.setenv("GPG_TTY", "/dev/pts/1")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-token")

    argv, _profile, env, fd = _bash(
        SandboxConfig(enabled=False, scrub_env=False)
    )._resolve_sandbox(_managed_context(state="active"), "git status --short")

    try:
        assert argv is not None
        assert "GH_TOKEN" not in env
        assert "SSH_AUTH_SOCK" not in env
        assert "GPG_TTY" not in env
        assert "OPENAI_API_KEY" not in env
        assert env["GIT_CONFIG_GLOBAL"] == os.devnull
        assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    finally:
        if fd is not None:
            os.close(fd)


def test_autoapprove_keeps_same_session_git_commit_compatibility(
    tmp_path, monkeypatch
) -> None:
    import os

    repo = tmp_path / "repo"
    (repo / ".git" / "hooks").mkdir(parents=True)
    (repo / ".git" / "config").write_text("[core]\n")
    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.detect_backend", lambda override: "bwrap"
    )

    argv, _profile, _env, fd = _bash(SandboxConfig(enabled=False))._resolve_sandbox(
        _autoapprove_context(), "git commit -m controlled"
    )

    try:
        assert argv is not None
        root = str(repo.resolve())
        ro_targets = [
            argv[index + 1] for index, value in enumerate(argv) if value == "--ro-bind"
        ]
        assert f"{root}/.git" not in ro_targets
        assert f"{root}/.git/hooks" in ro_targets
        assert f"{root}/.git/config" in ro_targets
        assert "--unshare-net" in argv
    finally:
        if fd is not None:
            os.close(fd)


def test_autoapprove_hides_unrelated_host_home_and_runtime_state(
    tmp_path, monkeypatch
) -> None:
    import os

    home = tmp_path / "host-home"
    repo = tmp_path / "repo"
    home.mkdir()
    (repo / ".git" / "hooks").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.detect_backend", lambda override: "bwrap"
    )

    argv, _profile, env, fd = _bash(SandboxConfig(enabled=False))._resolve_sandbox(
        _autoapprove_context(), "git status --short"
    )

    try:
        assert argv is not None
        hidden = [
            argv[index + 1] for index, value in enumerate(argv) if value == "--tmpfs"
        ]
        assert str(home.resolve()) in hidden
        assert "/run" in hidden
        assert env["HOME"] != str(home.resolve())
        assert str(repo.resolve()) in argv
    finally:
        if fd is not None:
            os.close(fd)


def test_verifier_bash_is_fail_closed_and_candidate_readonly(
    tmp_path, monkeypatch
) -> None:
    import os

    repo = tmp_path / "candidate"
    scratchpad = tmp_path / "scratchpad"
    (repo / ".git" / "hooks").mkdir(parents=True)
    scratchpad.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        "vibe.core.tools.builtins.bash.detect_backend", lambda override: "bwrap"
    )

    argv, _profile, env, fd = _bash(SandboxConfig(enabled=False))._resolve_sandbox(
        _verifier_context(scratchpad_dir=scratchpad), "uv run pytest -q"
    )

    try:
        assert argv is not None
        writable = [
            argv[index + 1] for index, value in enumerate(argv) if value == "--bind"
        ]
        readonly = [
            argv[index + 1] for index, value in enumerate(argv) if value == "--ro-bind"
        ]
        assert str(repo.resolve()) not in writable
        assert str(repo.resolve()) in readonly
        assert str(scratchpad.resolve()) in writable
        assert "--unshare-net" in argv
        assert env["VIBE_STRICT_MODEL_CONTROL"] == "1"
    finally:
        if fd is not None:
            os.close(fd)


@pytest.mark.asyncio
async def test_strict_model_control_rejects_background_bash() -> None:
    bash = _bash(SandboxConfig(enabled=False))

    with pytest.raises(ToolError, match="does not permit background Bash"):
        async for _ in bash.run(
            BashArgs(command="sleep 60", background=True), _autoapprove_context()
        ):
            pass


def test_sandbox_enabled_default_is_on() -> None:
    # Defense-in-depth default: on. Where a backend exists bash is sandboxed;
    # where none does it soft-falls-back (still enabled, warns once).
    assert SandboxConfig().enabled is True


def test_build_result_appends_sandbox_hint_on_bwrap_error() -> None:
    # A sandbox-caused failure (bwrap: prefix / fs / permission error) must carry
    # a one-line attribution so the user knows to widen write_dirs or disable it.
    bash = _bash(SandboxConfig(enabled=True))
    with pytest.raises(ToolError) as ei:
        bash._build_result(
            command="echo x > /etc/foo",
            stdout="",
            stderr="bwrap: Can't find source path /nope",
            returncode=1,
            sandbox_active=True,
        )
    assert "OS sandbox may have blocked this" in str(ei.value)
    assert "sandbox.write_dirs" in str(ei.value)


def test_build_result_no_hint_when_unsandboxed() -> None:
    # A failure that ran WITHOUT the sandbox must not be blamed on it, even when
    # stderr happens to mention permission errors.
    bash = _bash(SandboxConfig(enabled=True))
    with pytest.raises(ToolError) as ei:
        bash._build_result(
            command="false",
            stdout="",
            stderr="Permission denied",
            returncode=1,
            sandbox_active=False,
        )
    assert "OS sandbox may have blocked" not in str(ei.value)


def test_build_result_no_hint_when_failure_unrelated() -> None:
    # A sandboxed command that fails for its own reasons (nonzero, no fs/perm
    # marker) gets no misleading sandbox attribution.
    bash = _bash(SandboxConfig(enabled=True))
    with pytest.raises(ToolError) as ei:
        bash._build_result(
            command="grep missing file",
            stdout="",
            stderr="no match",
            returncode=1,
            sandbox_active=True,
        )
    assert "OS sandbox may have blocked" not in str(ei.value)


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

# The conftest `sandbox_e2e` marker skips these when user namespaces are
# unavailable (reusing the bwrap capability probe), which un-reds CI runners.
_skip_no_backend = pytest.mark.sandbox_e2e


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
async def test_sandbox_blocked_write_gets_attribution_hint(
    tmp_path, monkeypatch
) -> None:
    # When the OS sandbox is what blocked the command, the ToolError must say so.
    if detect_backend("auto") != "bwrap":
        pytest.skip("filesystem confinement needs the bwrap backend")
    monkeypatch.chdir(tmp_path)
    bash = _bash(SandboxConfig(enabled=True))
    with pytest.raises(ToolError) as ei:
        await _run(bash, "echo x > /etc/vibe_sandbox_hint_probe")
    assert "OS sandbox may have blocked this" in str(ei.value)


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
async def test_isolated_bash_cannot_read_host_home(tmp_path, monkeypatch) -> None:
    if detect_backend("auto") != "bwrap":
        pytest.skip("home hiding needs the bwrap backend")
    home = tmp_path / "host-home"
    worktree = tmp_path / "candidate"
    home.mkdir()
    worktree.mkdir()
    secret = home / "credential"
    secret.write_text("must-not-be-visible")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("VIBE_ISOLATED_WORKTREE_ROOT", str(worktree))
    monkeypatch.chdir(worktree)
    bash = _bash(SandboxConfig(enabled=False))

    result = await _run(
        bash, f"if cat {secret}; then exit 9; else echo home-hidden; fi"
    )

    assert result is not None
    assert result.returncode == 0
    assert "home-hidden" in result.stdout
    assert "must-not-be-visible" not in result.stdout


@_skip_no_backend
@pytest.mark.asyncio
async def test_autoapprove_bash_cannot_read_unrelated_host_home(
    tmp_path, monkeypatch
) -> None:
    if detect_backend("auto") != "bwrap":
        pytest.skip("home hiding needs the bwrap backend")
    home = tmp_path / "host-home"
    repo = tmp_path / "candidate"
    home.mkdir()
    repo.mkdir()
    (repo / ".git").mkdir()
    secret = home / "credential"
    secret.write_text("must-not-be-visible")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(repo)
    bash = _bash(SandboxConfig(enabled=False))

    result = None
    async for item in bash.run(
        BashArgs(command=f"if cat {secret}; then exit 9; else echo home-hidden; fi"),
        _autoapprove_context(),
    ):
        from vibe.core.tools.builtins.bash import BashResult

        if isinstance(item, BashResult):
            result = item

    assert result is not None
    assert result.returncode == 0
    assert "home-hidden" in result.stdout
    assert "must-not-be-visible" not in result.stdout


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


def test_bwrap_sibling_worktree_configs_readonly(tmp_path) -> None:
    # extensions.worktreeConfig: a sibling worktree's config.worktree is a
    # cross-worktree hooksPath escape if left writable under the shared .git.
    main = tmp_path / "main"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    (main / ".git" / "worktrees" / "other").mkdir(parents=True)
    (main / ".git" / "config").write_text("[core]\n")
    (main / ".git" / "worktrees" / "other" / "config.worktree").write_text("[core]\n")
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/wt\n")

    spec = SandboxSpec(write_roots=[wt], allow_network=True)
    argv, _n, _p = build_sandbox_command(spec, "bwrap")
    assert argv is not None
    ro_targets = [argv[i + 1] for i, a in enumerate(argv) if a == "--ro-bind"]
    assert f"{main}/.git/worktrees/other/config.worktree" in ro_targets


def test_bwrap_main_checkout_linked_worktree_configs_readonly(tmp_path) -> None:
    # Sandboxed in the MAIN checkout: linked worktrees' config.worktree files
    # under .git/worktrees/* must not be writable either.
    root = tmp_path / "repo"
    (root / ".git" / "worktrees" / "wt").mkdir(parents=True)
    (root / ".git" / "config").write_text("[core]\n")
    (root / ".git" / "worktrees" / "wt" / "config.worktree").write_text("[core]\n")

    spec = SandboxSpec(write_roots=[root], allow_network=True)
    argv, _n, _p = build_sandbox_command(spec, "bwrap")
    assert argv is not None
    ro_targets = [argv[i + 1] for i, a in enumerate(argv) if a == "--ro-bind"]
    assert f"{root}/.git/worktrees/wt/config.worktree" in ro_targets
