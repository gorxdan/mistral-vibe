from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, Mock

import pytest

from tests.mock.utils import collect_result
from vibe.core.config import (
    TrustedExecutionTopologyConfig,
    TrustedVerificationCheckConfig,
    TrustedVerificationRecipeConfig,
)
from vibe.core.tools import _model_read_policy as read_policy
from vibe.core.tools._model_read_policy import (
    ManagedReadPolicyError,
    managed_read_policy_active,
    resolve_managed_read_path,
)
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.glob import Glob, GlobArgs, GlobBackend, GlobToolConfig
from vibe.core.tools.builtins.grep import (
    Grep,
    GrepArgs,
    GrepBackend,
    GrepOutputMode,
    GrepToolConfig,
    _normalize_search_output,
)
from vibe.core.tools.builtins.lsp import Lsp, LspArgs, LspConfig, LspOperation, LspState
from vibe.core.tools.builtins.read import Read, ReadArgs, ReadConfig, ReadState
from vibe.core.utils.io import ReadSafeResult, write_safe
from vibe.core.verification_state import VerificationState


@dataclass(frozen=True)
class _ManagedSession:
    context: InvokeContext
    candidate: Path
    control: Path
    evidence: Path
    scratchpad: Path


def _managed_session(
    tmp_path: Path, *, phase: Literal["active", "verification"] = "active"
) -> _ManagedSession:
    candidate = tmp_path / "candidate"
    control = tmp_path / "control"
    evidence = tmp_path / "evidence"
    scratchpad = tmp_path / "scratchpad"
    for path in (candidate, control, evidence, scratchpad):
        path.mkdir(parents=True, exist_ok=True)
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
        allowed_paths=("candidate.py",),
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
    context = InvokeContext(
        tool_call_id="managed-read",
        verification_state=VerificationState.from_recipe(recipe),
        scratchpad_dir=scratchpad,
    )
    return _ManagedSession(context, candidate, control, evidence, scratchpad)


@pytest.mark.parametrize("phase", ["active", "verification"])
def test_managed_phases_allow_only_assigned_session_roots(
    tmp_path: Path, phase: Literal["active", "verification"]
) -> None:
    session = _managed_session(tmp_path, phase=phase)

    for root in (
        session.candidate,
        session.control,
        session.evidence,
        session.scratchpad,
    ):
        target = root / "nested" / "artifact.txt"
        assert resolve_managed_read_path(target, session.context) == target.resolve()

    assert managed_read_policy_active(session.context)
    with pytest.raises(ManagedReadPolicyError, match="assigned candidate"):
        resolve_managed_read_path(
            tmp_path / "unrelated" / "secret.txt", session.context
        )


def test_managed_policy_denies_host_state_and_runtime_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vibe_home = tmp_path / "home" / ".vibe"
    monkeypatch.setenv("VIBE_HOME", str(vibe_home))
    session = _managed_session(tmp_path / "session")

    denied = (
        vibe_home / "config.toml",
        vibe_home / "logs" / "vibe.log",
        vibe_home / "verification" / "receipts" / "receipt.json",
        Path("/run/secrets/model-token"),
    )
    for path in denied:
        with pytest.raises(ManagedReadPolicyError):
            resolve_managed_read_path(path, session.context)


def test_symlink_ancestry_cannot_escape_assigned_roots(tmp_path: Path) -> None:
    session = _managed_session(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = session.candidate / "linked"
    alias.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ManagedReadPolicyError, match="assigned candidate"):
        resolve_managed_read_path(alias / "secret.txt", session.context)


def test_nonmanaged_context_preserves_unrestricted_resolution(tmp_path: Path) -> None:
    context = InvokeContext(tool_call_id="ordinary")

    assert not managed_read_policy_active(context)
    assert resolve_managed_read_path(tmp_path / "outside.txt", context) is None


def test_registered_skill_directory_is_an_exact_host_resource(tmp_path: Path) -> None:
    session = _managed_session(tmp_path / "session")
    skill_dir = tmp_path / "host-skills" / "security-review"
    skill_dir.mkdir(parents=True)
    session.context.skill_manager = cast(
        Any,
        SimpleNamespace(
            available_skills={"security-review": SimpleNamespace(skill_dir=skill_dir)}
        ),
    )

    reference = skill_dir / "references" / "paths.md"
    assert resolve_managed_read_path(reference, session.context) == reference.resolve()
    with pytest.raises(ManagedReadPolicyError):
        resolve_managed_read_path(
            skill_dir.parent / "unregistered.txt", session.context
        )


def test_only_selected_prompt_files_are_host_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _managed_session(tmp_path / "session")
    prompt_dir = tmp_path / "host-prompts"
    prompt_dir.mkdir()
    selected = prompt_dir / "managed.md"
    compact = prompt_dir / "compact.md"
    sibling = prompt_dir / "unselected.md"
    for path in (selected, compact, sibling):
        write_safe(path, path.stem)
    manager = SimpleNamespace(project_prompts_dirs=[], user_prompts_dirs=[])
    monkeypatch.setattr(read_policy, "get_harness_files_manager", lambda: manager)
    session.context.agent_manager = cast(
        Any,
        SimpleNamespace(
            config=SimpleNamespace(
                system_prompt_id="managed",
                compaction_prompt_id="compact",
                prompt_paths=[prompt_dir],
            )
        ),
    )

    assert resolve_managed_read_path(selected, session.context) == selected.resolve()
    assert resolve_managed_read_path(compact, session.context) == compact.resolve()
    with pytest.raises(ManagedReadPolicyError):
        resolve_managed_read_path(sibling, session.context)


@pytest.mark.asyncio
async def test_read_uses_managed_canonical_path_and_rejects_symlink_escape(
    tmp_path: Path,
) -> None:
    session = _managed_session(tmp_path)
    allowed = session.candidate / "allowed.txt"
    outside = tmp_path / "outside.txt"
    alias = session.candidate / "alias.txt"
    write_safe(allowed, "allowed")
    write_safe(outside, "secret")
    alias.symlink_to(outside)
    tool = Read(config_getter=ReadConfig, state=ReadState())

    result = await collect_result(
        tool.run(ReadArgs(file_path=str(allowed)), session.context)
    )

    assert "allowed" in result.content
    with pytest.raises(ToolError, match="assigned candidate"):
        await collect_result(tool.run(ReadArgs(file_path=str(alias)), session.context))


@pytest.mark.asyncio
async def test_read_nonmanaged_behavior_is_unchanged(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    write_safe(outside, "ordinary session")
    tool = Read(config_getter=ReadConfig, state=ReadState())

    result = await collect_result(
        tool.run(
            ReadArgs(file_path=str(outside)),
            InvokeContext(tool_call_id="ordinary-read"),
        )
    )

    assert "ordinary session" in result.content


@pytest.mark.asyncio
async def test_glob_rejects_absolute_escape_and_filters_symlink_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _managed_session(tmp_path)
    monkeypatch.chdir(session.candidate)
    allowed = session.candidate / "allowed.txt"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside = outside_dir / "secret.txt"
    alias = session.candidate / "secret-link.txt"
    write_safe(allowed, "allowed")
    write_safe(outside, "secret")
    alias.symlink_to(outside)
    tool = Glob(config_getter=GlobToolConfig, state=BaseToolState())
    monkeypatch.setattr(tool, "_detect_backend", lambda: GlobBackend.WALK)

    result = await collect_result(
        tool.run(
            GlobArgs(pattern="*.txt", path=str(session.candidate)), session.context
        )
    )

    assert result.paths == [str(allowed)]
    with pytest.raises(ToolError, match="assigned candidate"):
        await collect_result(
            tool.run(
                GlobArgs(
                    pattern=str(outside_dir / "*.txt"), path=str(session.candidate)
                ),
                session.context,
            )
        )


@pytest.mark.asyncio
async def test_grep_rejects_escape_before_starting_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _managed_session(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    tool = Grep(config_getter=GrepToolConfig, state=BaseToolState())
    execute = AsyncMock(return_value=b"")
    monkeypatch.setattr(tool, "_detect_backend", lambda: GrepBackend.RIPGREP)
    monkeypatch.setattr(tool, "_execute_search", execute)

    with pytest.raises(ToolError, match="assigned candidate"):
        await collect_result(
            tool.run(GrepArgs(pattern="secret", path=str(outside)), session.context)
        )

    execute.assert_not_awaited()


def test_grep_drops_out_of_scope_backend_records(tmp_path: Path) -> None:
    session = _managed_session(tmp_path)
    allowed = session.candidate / "allowed.txt"
    outside = tmp_path / "outside.txt"
    payload = b"\0".join((str(allowed).encode(), str(outside).encode(), b""))

    normalized = _normalize_search_output(
        payload, GrepOutputMode.FILES_WITH_MATCHES, session.context
    )

    assert normalized == str(allowed)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "query"),
    [(LspOperation.STATUS, None), (LspOperation.WORKSPACE_SYMBOL, "Widget")],
)
async def test_lsp_managed_global_queries_fail_before_manager_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: LspOperation,
    query: str | None,
) -> None:
    session = _managed_session(tmp_path)
    tool = Lsp(config_getter=LspConfig, state=LspState())
    ensure_manager = Mock(side_effect=AssertionError("manager must not be accessed"))
    monkeypatch.setattr(tool, "_ensure_manager", ensure_manager)

    with pytest.raises(ToolError, match="requires an allowed file_path"):
        await collect_result(
            tool.run(LspArgs(operation=operation, query=query), session.context)
        )

    ensure_manager.assert_not_called()


@pytest.mark.asyncio
async def test_lsp_rejects_external_input_before_manager_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _managed_session(tmp_path)
    outside = tmp_path / "outside.py"
    write_safe(outside, "secret = True\n")
    tool = Lsp(config_getter=LspConfig, state=LspState())
    ensure_manager = Mock(side_effect=AssertionError("manager must not be accessed"))
    monkeypatch.setattr(tool, "_ensure_manager", ensure_manager)

    with pytest.raises(ToolError, match="assigned candidate"):
        await collect_result(
            tool.run(
                LspArgs(operation=LspOperation.DOCUMENT_SYMBOL, file_path=str(outside)),
                session.context,
            )
        )

    ensure_manager.assert_not_called()


@pytest.mark.asyncio
async def test_lsp_filters_external_uris_before_reading_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _managed_session(tmp_path)
    allowed = session.candidate / "allowed.py"
    outside = tmp_path / "outside.py"
    write_safe(allowed, "allowed = True\n")
    write_safe(outside, "secret = True\n")
    tool = Lsp(config_getter=LspConfig, state=LspState())

    async def read_allowed(path: Path) -> ReadSafeResult:
        if path != allowed:
            raise AssertionError("external URI must not be read")
        return ReadSafeResult("allowed = True\n", "utf-8")

    read = AsyncMock(side_effect=read_allowed)
    monkeypatch.setattr("vibe.core.tools.builtins.lsp.read_safe_async", read)
    range_ = {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 6}}
    locations_raw = [
        {"uri": allowed.as_uri(), "range": range_},
        {"uri": outside.as_uri(), "range": range_},
        {"data": "malformed", "range": range_},
    ]
    symbols_raw = [
        {"name": name, "kind": 13, "location": {"uri": path.as_uri(), "range": range_}}
        for name, path in (("allowed", allowed), ("secret", outside))
    ]

    locations = await tool._normalize_location_positions(locations_raw, session.context)
    symbols = await tool._normalize_symbols(symbols_raw, "", session.context)

    assert [location["uri"] for location in locations] == [allowed.as_uri()]
    assert [symbol.name for symbol in symbols] == ["allowed"]
    assert read.await_count == 2
