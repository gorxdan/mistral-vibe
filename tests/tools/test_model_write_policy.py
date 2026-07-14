from __future__ import annotations

from pathlib import Path
import sys
from typing import Literal

import pytest

from vibe.core.config import (
    TrustedExecutionTopologyConfig,
    TrustedVerificationCheckConfig,
    TrustedVerificationRecipeConfig,
)
from vibe.core.config.fingerprint import file_fingerprint
from vibe.core.tools._model_write_policy import (
    managed_candidate_write_reason,
    protected_model_write_reason,
    verification_protected_roots,
)
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError, ToolPermission
from vibe.core.tools.builtins.edit import Edit, EditArgs, EditConfig, EditResult
from vibe.core.tools.builtins.task import _with_scratchpad_context
from vibe.core.tools.builtins.write_file import (
    WriteFile,
    WriteFileArgs,
    WriteFileConfig,
    WriteFileResult,
)
from vibe.core.utils.io import read_safe, write_safe
from vibe.core.verification_state import VerificationState


def _managed_state(
    tmp_path: Path,
    *,
    phase: Literal["active", "verification"] = "active",
    allowed_paths: tuple[str, ...] = ("candidate.py",),
) -> VerificationState:
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    evidence = tmp_path / "evidence"
    for path in (control, candidate, evidence):
        path.mkdir()
    topology = TrustedExecutionTopologyConfig(
        packet_id="I00-P01",
        packet_path="docs/packet.md",
        state=phase,
        control_worktree=str(control),
        control_sha="1" * 40,
        candidate_worktree=str(candidate),
        candidate_branch="candidate",
        baseline_sha="2" * 40,
        candidate_sha="4" * 40 if phase == "verification" else None,
        upstream_sha="3" * 40,
        evidence_workspace=str(evidence),
        run_id="run-1",
        runner_id="runner-1",
        evidence_manifest_sha256="5" * 64 if phase == "verification" else None,
    )
    recipe = TrustedVerificationRecipeConfig(
        recipe_version="v1",
        task_brief="managed task",
        acceptance_contract="focused check passes",
        allowed_paths=allowed_paths,
        checks=(
            TrustedVerificationCheckConfig(
                name="focused",
                argv=(sys.executable, "-c", "print('ok')"),
                executable_sha256="0" * 64,
                environment_attestation_path="/usr/bin/true",
                environment_attestation_sha256="1" * 64,
            ),
        ),
        execution_topology=topology,
    )
    return VerificationState.from_recipe(recipe)


def test_candidate_source_path_remains_writable(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate" / "src" / "module.py"

    assert protected_model_write_reason(candidate) is None


def test_git_metadata_is_host_owned(tmp_path: Path) -> None:
    target = tmp_path / "candidate" / ".git" / "refs" / "heads" / "main"

    reason = protected_model_write_reason(target)

    assert reason is not None
    assert "Git control metadata" in reason


def test_symlink_alias_to_git_metadata_is_rejected(tmp_path: Path) -> None:
    gitdir = tmp_path / "repo" / ".git"
    gitdir.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(gitdir, target_is_directory=True)

    reason = protected_model_write_reason(alias / "config")

    assert reason is not None
    assert "Git control metadata" in reason


def test_managed_control_and_evidence_roots_are_host_owned(tmp_path: Path) -> None:
    state = _managed_state(tmp_path)
    roots = verification_protected_roots(state)

    assert protected_model_write_reason(
        tmp_path / "control" / "status.yaml", extra_roots=roots
    )
    assert protected_model_write_reason(
        tmp_path / "evidence" / "manifest.json", extra_roots=roots
    )
    assert (
        protected_model_write_reason(
            tmp_path / "candidate" / "candidate.py", extra_roots=roots
        )
        is None
    )


def test_active_managed_file_writes_are_limited_to_recipe_paths(tmp_path: Path) -> None:
    state = _managed_state(tmp_path)

    assert (
        managed_candidate_write_reason(tmp_path / "candidate" / "candidate.py", state)
        is None
    )
    reason = managed_candidate_write_reason(
        tmp_path / "candidate" / "unrelated.py", state
    )

    assert reason is not None
    assert "outside the managed candidate allowlist" in reason


def test_allowed_file_cannot_be_reinterpreted_as_a_directory(tmp_path: Path) -> None:
    state = _managed_state(tmp_path)

    reason = managed_candidate_write_reason(
        tmp_path / "candidate" / "candidate.py" / "payload.py", state
    )

    assert reason is not None
    assert "outside the managed candidate allowlist" in reason


@pytest.mark.parametrize(
    "relative",
    [
        ".vibe/config.toml",
        ".vibe/hooks.toml",
        ".agents/skills/local/SKILL.md",
        ".codex/plugin.json",
        ".claude/settings.json",
        "AGENTS.md",
        "src/AGENTS.md",
        "CLAUDE.md",
    ],
)
def test_managed_control_files_remain_host_owned_with_broad_allowlist(
    tmp_path: Path, relative: str
) -> None:
    state = _managed_state(tmp_path, allowed_paths=("**",))

    reason = managed_candidate_write_reason(tmp_path / "candidate" / relative, state)

    assert reason is not None
    assert "harness control files are host-owned" in reason


def test_active_managed_file_writes_allow_session_scratchpad(tmp_path: Path) -> None:
    state = _managed_state(tmp_path)
    scratchpad = tmp_path / "scratch"
    scratchpad.mkdir()

    assert (
        managed_candidate_write_reason(
            scratchpad / "notes.txt", state, scratchpad_dir=scratchpad
        )
        is None
    )


def test_frozen_verification_candidate_is_read_only_to_model_tools(
    tmp_path: Path,
) -> None:
    state = _managed_state(tmp_path, phase="verification")
    roots = verification_protected_roots(state)

    reason = protected_model_write_reason(
        tmp_path / "candidate" / "candidate.py", extra_roots=roots
    )

    assert reason is not None
    assert "read-only" in reason


@pytest.mark.asyncio
async def test_write_file_cannot_mutate_managed_evidence_with_always_permission(
    tmp_path: Path,
) -> None:
    state = _managed_state(tmp_path)
    tool = WriteFile(
        config_getter=lambda: WriteFileConfig(permission=ToolPermission.ALWAYS),
        state=BaseToolState(),
    )
    ctx = InvokeContext(tool_call_id="write", verification_state=state)

    with pytest.raises(ToolError, match="read-only"):
        await anext(
            tool.run(
                WriteFileArgs(
                    path=str(tmp_path / "evidence" / "manifest.json"), content="forged"
                ),
                ctx,
            )
        )


@pytest.mark.asyncio
async def test_write_file_cannot_escape_managed_candidate_allowlist(
    tmp_path: Path,
) -> None:
    state = _managed_state(tmp_path)
    tool = WriteFile(
        config_getter=lambda: WriteFileConfig(permission=ToolPermission.ALWAYS),
        state=BaseToolState(),
    )
    ctx = InvokeContext(tool_call_id="write", verification_state=state)

    with pytest.raises(ToolError, match="outside the managed candidate allowlist"):
        await anext(
            tool.run(
                WriteFileArgs(
                    path=str(tmp_path / "candidate" / "unrelated.py"), content="bad"
                ),
                ctx,
            )
        )


@pytest.mark.asyncio
async def test_managed_write_creates_allowed_file(tmp_path: Path) -> None:
    state = _managed_state(tmp_path, allowed_paths=("src/new.py",))
    target = tmp_path / "candidate" / "src" / "new.py"
    tool = WriteFile(
        config_getter=lambda: WriteFileConfig(permission=ToolPermission.ALWAYS),
        state=BaseToolState(),
    )
    ctx = InvokeContext(tool_call_id="write", verification_state=state, files_read={})

    result = await anext(
        tool.run(WriteFileArgs(path=str(target), content="managed\n"), ctx)
    )

    assert isinstance(result, WriteFileResult)
    assert result.path == str(target)
    assert read_safe(target).text == "managed\n"
    assert ctx.files_read is not None
    assert ctx.files_read[str(target)] == file_fingerprint(target)


@pytest.mark.asyncio
async def test_managed_edit_atomically_replaces_allowed_file(tmp_path: Path) -> None:
    state = _managed_state(tmp_path)
    target = tmp_path / "candidate" / "candidate.py"
    write_safe(target, "before\n")
    tool = Edit(
        config_getter=lambda: EditConfig(permission=ToolPermission.ALWAYS),
        state=BaseToolState(),
    )
    ctx = InvokeContext(
        tool_call_id="edit",
        verification_state=state,
        files_read={str(target): file_fingerprint(target)},
    )

    result = await anext(
        tool.run(
            EditArgs(file_path=str(target), old_string="before", new_string="after"),
            ctx,
        )
    )

    assert isinstance(result, EditResult)
    assert result.file == str(target)
    assert read_safe(target).text == "after\n"
    assert ctx.files_read is not None
    assert ctx.files_read[str(target)] == file_fingerprint(target)


@pytest.mark.asyncio
async def test_managed_write_rejects_ancestor_swap_during_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _managed_state(tmp_path, allowed_paths=("src/new.py",))
    source = tmp_path / "candidate" / "src"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    moved = tmp_path / "candidate" / "moved"

    async def swap_ancestor(*_args: object) -> None:
        source.rename(moved)
        source.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(
        "vibe.core.tools.builtins.write_file.enforce_shared_ask", swap_ancestor
    )
    tool = WriteFile(
        config_getter=lambda: WriteFileConfig(permission=ToolPermission.ALWAYS),
        state=BaseToolState(),
    )
    ctx = InvokeContext(tool_call_id="write", verification_state=state)

    with pytest.raises(ToolError, match="symlink|ancestor changed"):
        await anext(
            tool.run(WriteFileArgs(path=str(source / "new.py"), content="managed"), ctx)
        )

    assert not (outside / "new.py").exists()
    assert not (moved / "new.py").exists()


@pytest.mark.asyncio
async def test_managed_write_rejects_root_swap_during_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _managed_state(tmp_path)
    candidate = tmp_path / "candidate"
    moved = tmp_path / "moved-candidate"
    target = candidate / "candidate.py"

    async def swap_root(*_args: object) -> None:
        candidate.rename(moved)
        candidate.mkdir()

    monkeypatch.setattr(
        "vibe.core.tools.builtins.write_file.enforce_shared_ask", swap_root
    )
    tool = WriteFile(
        config_getter=lambda: WriteFileConfig(permission=ToolPermission.ALWAYS),
        state=BaseToolState(),
    )
    ctx = InvokeContext(tool_call_id="write", verification_state=state)

    with pytest.raises(ToolError, match="root changed"):
        await anext(tool.run(WriteFileArgs(path=str(target), content="managed"), ctx))

    assert not target.exists()
    assert not (moved / "candidate.py").exists()


@pytest.mark.asyncio
async def test_managed_write_does_not_replace_file_created_during_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _managed_state(tmp_path)
    target = tmp_path / "candidate" / "candidate.py"

    async def create_target(*_args: object) -> None:
        write_safe(target, "external\n")

    monkeypatch.setattr(
        "vibe.core.tools.builtins.write_file.enforce_shared_ask", create_target
    )
    tool = WriteFile(
        config_getter=lambda: WriteFileConfig(permission=ToolPermission.ALWAYS),
        state=BaseToolState(),
    )
    ctx = InvokeContext(tool_call_id="write", verification_state=state)

    with pytest.raises(ToolError, match="already exists"):
        await anext(tool.run(WriteFileArgs(path=str(target), content="managed"), ctx))

    assert read_safe(target).text == "external\n"
    assert not tuple((tmp_path / "candidate").glob(".candidate.py.tmp.*"))


@pytest.mark.asyncio
async def test_managed_edit_rejects_ancestor_swap_during_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _managed_state(tmp_path, allowed_paths=("src/file.py",))
    source = tmp_path / "candidate" / "src"
    source.mkdir()
    target = source / "file.py"
    write_safe(target, "before\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_target = outside / "file.py"
    write_safe(outside_target, "outside\n")
    moved = tmp_path / "candidate" / "moved"

    async def swap_ancestor(*_args: object) -> None:
        source.rename(moved)
        source.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(
        "vibe.core.tools.builtins.edit.enforce_shared_ask", swap_ancestor
    )
    tool = Edit(
        config_getter=lambda: EditConfig(permission=ToolPermission.ALWAYS),
        state=BaseToolState(),
    )
    ctx = InvokeContext(
        tool_call_id="edit",
        verification_state=state,
        files_read={str(target): file_fingerprint(target)},
    )

    with pytest.raises(ToolError, match="symlink|ancestor changed"):
        await anext(
            tool.run(
                EditArgs(
                    file_path=str(target), old_string="before", new_string="after"
                ),
                ctx,
            )
        )

    assert read_safe(moved / "file.py").text == "before\n"
    assert read_safe(outside_target).text == "outside\n"


@pytest.mark.asyncio
async def test_managed_edit_rejects_target_swap_during_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _managed_state(tmp_path)
    target = tmp_path / "candidate" / "candidate.py"
    original = tmp_path / "candidate" / "original.py"
    write_safe(target, "before\n")

    async def swap_target(*_args: object) -> None:
        target.rename(original)
        write_safe(target, "replacement\n")

    monkeypatch.setattr("vibe.core.tools.builtins.edit.enforce_shared_ask", swap_target)
    tool = Edit(
        config_getter=lambda: EditConfig(permission=ToolPermission.ALWAYS),
        state=BaseToolState(),
    )
    ctx = InvokeContext(
        tool_call_id="edit",
        verification_state=state,
        files_read={str(target): file_fingerprint(target)},
    )

    with pytest.raises(ToolError, match="target changed"):
        await anext(
            tool.run(
                EditArgs(
                    file_path=str(target), old_string="before", new_string="after"
                ),
                ctx,
            )
        )

    assert read_safe(original).text == "before\n"
    assert read_safe(target).text == "replacement\n"


def test_verifier_prompt_names_host_evidence_not_scratch_as_authority(
    tmp_path: Path,
) -> None:
    state = _managed_state(tmp_path)

    prompt = _with_scratchpad_context(
        "verifier", "Verify independently", tmp_path / "scratch", state
    )

    assert (
        f"Host-provisioned evidence workspace (read-only to model tools): {tmp_path / 'evidence'}"
        in prompt
    )
    assert "This path, not the scratchpad or parent prose" in prompt
