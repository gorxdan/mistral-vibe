from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from vibe.core.tools.base import InvokeContext
from vibe.core.verification_state import VerificationState
from vibe.core.workflows.runtime import (
    WorkflowError,
    WorkflowRuntime,
    _run_verifier_in_worktree,
)

pytestmark = pytest.mark.asyncio

_PASS_RESPONSE = (
    "Verification notes.\n\n"
    "### Check: focused tests\n"
    "**Command run:**\n"
    "  uv run pytest -q\n"
    "**Output observed:**\n"
    "  3 passed\n"
    "**Result: PASS**\n\n"
    "VERDICT: PASS"
)

_FAIL_RESPONSE = (
    "Verification notes.\n\n"
    "### Check: focused tests\n"
    "**Command run:**\n"
    "  uv run pytest -q\n"
    "**Output observed:**\n"
    "  1 failed\n"
    "**Result: FAIL**\n\n"
    "VERDICT: FAIL"
)

_PARTIAL_RESPONSE = (
    "Verification notes.\n\n"
    "### Check: focused tests\n"
    "**Command run:**\n"
    "  uv run pytest -q\n"
    "**Output observed:**\n"
    "  2 passed, 1 failed\n"
    "**Result: FAIL**\n\n"
    "VERDICT: PARTIAL"
)


def _make_runtime(state: VerificationState | None = None) -> WorkflowRuntime:
    from tests.core.workflows.test_runtime import make_factory

    rt = WorkflowRuntime(agent_loop_factory=make_factory(), budget_total=1_000_000)
    rt.parent_context = InvokeContext(tool_call_id="t1", verification_state=state)
    return rt


class _FakeResult:
    def __init__(self, output: str = "worker output") -> None:
        self.output = output
        self.stats: dict[str, int] | None = None
        self.worktree_path: str | None = None
        self.branch: str | None = None
        self.wt: Any = None


class _FakeWT:
    def __init__(self, path: str = "/tmp/wt") -> None:
        self.path = Path(path)
        self.branch = "wt-branch"
        self.base = Path("/tmp/base")


async def test_then_verifier_requires_worktree_isolation() -> None:
    rt = _make_runtime()
    with pytest.raises(WorkflowError, match="requires isolation='worktree'"):
        await rt.spawn_agent("task", agent="worker", then="verifier")


async def test_then_rejects_unknown_value() -> None:
    rt = _make_runtime()
    with pytest.raises(WorkflowError, match="supports only 'verifier'"):
        await rt.spawn_agent(
            "task", agent="worker", isolation="worktree", then="reviewer"
        )


async def test_run_verifier_pass_records_and_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = VerificationState()
    rt = _make_runtime(state)

    async def fake_spawn(
        wt: Any, prompt: str, agent: str, max_turns: int, **kw: Any
    ) -> Any:
        return _FakeResult(output=_PASS_RESPONSE)

    monkeypatch.setattr("vibe.core.workflows.runtime._spawn_isolated", fake_spawn)
    result = _FakeResult("did the work")
    wt = _FakeWT()

    verdict = await _run_verifier_in_worktree(rt, wt, result, None, None)

    assert verdict is True
    assert state.last_verifier_pass is not None


async def test_run_verifier_fail_blocks_and_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = VerificationState()
    rt = _make_runtime(state)

    async def fake_spawn(
        wt: Any, prompt: str, agent: str, max_turns: int, **kw: Any
    ) -> Any:
        return _FakeResult(output=_FAIL_RESPONSE)

    monkeypatch.setattr("vibe.core.workflows.runtime._spawn_isolated", fake_spawn)

    verdict = await _run_verifier_in_worktree(
        rt, _FakeWT(), _FakeResult("work"), None, None
    )

    assert verdict is False
    assert state.last_verifier_pass is None


async def test_run_verifier_spawn_failure_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = VerificationState()
    rt = _make_runtime(state)

    async def boom(wt: Any, prompt: str, agent: str, max_turns: int, **kw: Any) -> Any:
        raise RuntimeError("subprocess exploded")

    monkeypatch.setattr("vibe.core.workflows.runtime._spawn_isolated", boom)

    verdict = await _run_verifier_in_worktree(
        rt, _FakeWT(), _FakeResult("work"), None, None
    )

    assert verdict is False
    assert state.last_verifier_pass is None


async def test_run_verifier_noop_without_state(monkeypatch: pytest.MonkeyPatch) -> None:
    rt = _make_runtime(state=None)

    async def fake_spawn(
        wt: Any, prompt: str, agent: str, max_turns: int, **kw: Any
    ) -> Any:
        return _FakeResult(output=_PASS_RESPONSE)

    monkeypatch.setattr("vibe.core.workflows.runtime._spawn_isolated", fake_spawn)

    verdict = await _run_verifier_in_worktree(
        rt, _FakeWT(), _FakeResult("work"), None, None
    )

    assert verdict is True


async def test_run_verifier_partial_blocks() -> None:
    from unittest.mock import AsyncMock

    state = VerificationState()
    rt = _make_runtime(state)
    rt.parent_context = InvokeContext(tool_call_id="t1", verification_state=state)

    fake_result = _FakeResult(output=_PARTIAL_RESPONSE)
    mock_spawn = AsyncMock(return_value=fake_result)

    import vibe.core.workflows.runtime as rt_mod

    original = rt_mod._spawn_isolated
    rt_mod._spawn_isolated = mock_spawn
    try:
        verdict = await _run_verifier_in_worktree(
            rt, _FakeWT(), _FakeResult("work"), None, None
        )
    finally:
        rt_mod._spawn_isolated = original

    assert verdict is False
    assert state.last_verifier_pass is None


class _VerdictProc:
    pid = 4243
    returncode = 0

    def __init__(self, out: bytes) -> None:
        self._out = out

    async def communicate(self) -> tuple[bytes, bytes]:
        return (self._out, b"")


class _FakeProc:
    pid = 4242
    returncode = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        return (b"worker output", b"")


async def test_default_isolated_executor_standalone_then_verifier_pass_delivers_and_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibe.core.worktree.ephemeral as eph

    state = VerificationState()
    rt = _make_runtime(state)

    fake_wt = type("WT", (), {"path": Path("/tmp/iso-wt")})()
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    monkeypatch.setattr(eph, "remove_ephemeral_worktree", lambda wt, **k: None)
    delivered: list[Any] = []
    monkeypatch.setattr(
        eph, "deliver_ephemeral_worktree", lambda wt: delivered.append(wt) or True
    )

    spawn_count = 0

    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        nonlocal spawn_count
        spawn_count += 1
        if spawn_count == 2:
            return _VerdictProc(_PASS_RESPONSE.encode())
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    out, stats, report = await rt._default_isolated_executor(
        "do it", "worker", "lbl", 40, then="verifier"
    )

    assert out == "worker output"
    assert spawn_count == 2
    assert delivered == [fake_wt]
    assert report is not None
    assert report.passed is True
    assert report.delivered is True
    assert state.last_verifier_pass is not None
    assert state.last_verifier_pass.summary.startswith("VERDICT: PASS")


async def test_default_isolated_executor_standalone_then_verifier_fail_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibe.core.worktree.ephemeral as eph

    state = VerificationState()
    rt = _make_runtime(state)

    fake_wt = type("WT", (), {"path": Path("/tmp/iso-wt")})()
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    removed: list[Any] = []
    monkeypatch.setattr(
        eph,
        "remove_ephemeral_worktree",
        lambda wt, **k: removed.append(k.get("keep_if_changed")),
    )
    delivered: list[Any] = []
    monkeypatch.setattr(
        eph, "deliver_ephemeral_worktree", lambda wt: delivered.append(wt) or True
    )

    spawn_count = 0

    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        nonlocal spawn_count
        spawn_count += 1
        if spawn_count == 2:
            return _VerdictProc(_FAIL_RESPONSE.encode())
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    out, stats, report = await rt._default_isolated_executor(
        "do it", "worker", "lbl", 40, then="verifier"
    )

    assert out == "worker output"
    assert spawn_count == 2
    assert delivered == []
    assert report is None or not report.passed
    assert state.last_verifier_pass is None
    assert removed and removed[0] is True
