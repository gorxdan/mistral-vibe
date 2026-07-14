from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from vibe.core._verification_runner import TrustedCheck
from vibe.core.teams._task_checks import run_trusted_task_checks
from vibe.core.tools.sandbox import ResolvedSandboxBackend


def _backend(name: str) -> ResolvedSandboxBackend:
    executable = None if name == "none" else Path(f"/usr/bin/{name}")
    return ResolvedSandboxBackend(name, executable)


def _check() -> TrustedCheck:
    return TrustedCheck(
        name="focused",
        argv=(sys.executable, "-c", "raise SystemExit(0)"),
        cwd=".",
        timeout_seconds=5,
    )


def test_trusted_checks_fail_closed_without_filesystem_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        "vibe.core.teams._task_checks.resolve_backend",
        lambda _override: _backend("none"),
    )
    monkeypatch.setattr(subprocess, "run", forbidden_run)

    [evidence] = run_trusted_task_checks((_check(),), tmp_path)

    assert not called
    assert not evidence.passed
    assert "filesystem-containment sandbox" in evidence.stderr


def test_bwrap_reexposes_temporary_workspace_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[str] = []

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        captured.extend(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(
        "vibe.core.teams._task_checks.resolve_backend",
        lambda _override: _backend("bwrap"),
    )
    monkeypatch.setattr(subprocess, "run", fake_run)

    [evidence] = run_trusted_task_checks((_check(),), tmp_path)

    assert evidence.passed
    tmpfs_index = captured.index("--tmpfs")
    chdir_index = captured.index("--chdir")
    readonly_pairs = list(zip(captured, captured[1:], strict=False))
    assert (str(tmp_path), str(tmp_path)) in readonly_pairs
    workspace_index = captured.index(str(tmp_path))
    assert tmpfs_index < workspace_index < chdir_index
