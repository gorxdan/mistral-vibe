from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from pydantic import BaseModel, ConfigDict, Field

from vibe.core._verification_runner import TrustedCheck
from vibe.core.tools.sandbox import (
    SandboxSpec,
    build_sandbox_command,
    resolve_backend,
    scrub_env,
)
from vibe.core.utils.io import decode_safe

__all__ = [
    "TaskCheckEvidence",
    "run_guarded_task_checks",
    "run_trusted_task_checks",
    "task_check_diagnostics",
]

_EVIDENCE_CHARS = 1_500


class TaskCheckEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    argv: tuple[str, ...]
    cwd: str
    exit_code: int | None
    timed_out: bool
    duration_ms: int = Field(ge=0)
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return not self.timed_out and self.exit_code == 0


def _bounded_output(raw: bytes | str | None) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        text = raw
    else:
        text = decode_safe(raw, from_subprocess=True).text
    if len(text) <= _EVIDENCE_CHARS:
        return text
    omitted = len(text) - _EVIDENCE_CHARS
    marker = f"\n...[{omitted} characters omitted]"
    return text[: _EVIDENCE_CHARS - len(marker)] + marker


def _check_cwd(workspace_root: Path, requested: str) -> Path:
    path = Path(requested)
    if not path.is_absolute():
        path = workspace_root / path
    resolved = path.resolve()
    if not resolved.is_relative_to(workspace_root):
        raise ValueError(f"check cwd escapes the workspace: {requested!r}")
    if not resolved.is_dir():
        raise ValueError(f"check cwd is not a directory: {requested!r}")
    return resolved


def _temporary_executable_root(argv0: str) -> Path | None:
    executable = shutil.which(argv0)
    if executable is None:
        return None
    lexical = Path(os.path.abspath(executable))
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if not lexical.is_relative_to(temporary_root):
        return None
    relative = lexical.relative_to(temporary_root)
    if not relative.parts:
        return None
    return temporary_root / relative.parts[0]


def _run_check(check: TrustedCheck, workspace_root: Path) -> TaskCheckEvidence:
    started = time.monotonic()
    cwd = workspace_root
    exit_code: int | None = None
    timed_out = False
    stdout: bytes | str | None = None
    stderr: bytes | str | None = None
    profile: Path | None = None
    try:
        cwd = _check_cwd(workspace_root, check.cwd)
        backend = resolve_backend("auto")
        if backend.name not in {"bwrap", "sandbox-exec"}:
            raise ValueError("trusted checks require a filesystem-containment sandbox")
        temp_dir = Path(tempfile.mkdtemp(prefix="vibe-task-check-"))
        try:
            env = scrub_env(dict(os.environ), [])
            env.update({
                "HOME": str(temp_dir),
                "TMPDIR": str(temp_dir),
                "UV_CACHE_DIR": f"{temp_dir}/uv-cache",
                "XDG_CACHE_HOME": f"{temp_dir}/cache",
                "PYTHONDONTWRITEBYTECODE": "1",
            })
            argv = list(check.argv)
            read_roots = [workspace_root]
            if executable_root := _temporary_executable_root(check.argv[0]):
                read_roots.append(executable_root)
            prefix, _, profile = build_sandbox_command(
                SandboxSpec(
                    write_roots=[Path(temp_dir)],
                    read_roots=read_roots,
                    allow_network=False,
                    env=env,
                    cwd=cwd,
                ),
                backend,
            )
            if prefix is None:
                raise ValueError("trusted check sandbox could not be constructed")
            argv = [*prefix, *argv]
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=env,
                shell=False,
                capture_output=True,
                timeout=check.timeout_seconds,
                check=False,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout
        stderr = e.stderr
        timed_out = True
    except (OSError, ValueError) as e:
        stderr = str(e)
    finally:
        if profile is not None:
            profile.unlink(missing_ok=True)
    return TaskCheckEvidence(
        name=check.name,
        argv=check.argv,
        cwd=str(cwd),
        exit_code=exit_code,
        timed_out=timed_out,
        duration_ms=max(0, round((time.monotonic() - started) * 1_000)),
        stdout=_bounded_output(stdout),
        stderr=_bounded_output(stderr),
    )


def run_trusted_task_checks(
    checks: Sequence[TrustedCheck], workspace_root: Path
) -> tuple[TaskCheckEvidence, ...]:
    root = workspace_root.resolve()
    return tuple(_run_check(check, root) for check in checks)


def run_guarded_task_checks(
    checks: Sequence[TrustedCheck], workspace_root: Path
) -> tuple[tuple[TaskCheckEvidence, ...], str | None]:
    from vibe.core._workspace_verification import workspace_fingerprint

    root = workspace_root.resolve()
    before = workspace_fingerprint(root)
    evidence = run_trusted_task_checks(checks, root)
    after = workspace_fingerprint(root)
    if before is not None and after != before:
        return evidence, "trusted acceptance checks modified the candidate workspace"
    return evidence, None


def task_check_diagnostics(evidence: Sequence[TaskCheckEvidence]) -> tuple[str, ...]:
    diagnostics: list[str] = []
    for item in evidence:
        status = (
            "timed out"
            if item.timed_out
            else f"exit {item.exit_code}"
            if item.exit_code is not None
            else "could not start"
        )
        lines = [f"check {item.name!r}: {status}"]
        if item.stdout:
            lines.append(f"stdout:\n{item.stdout}")
        if item.stderr:
            lines.append(f"stderr:\n{item.stderr}")
        diagnostics.append("\n".join(lines))
    return tuple(diagnostics)
