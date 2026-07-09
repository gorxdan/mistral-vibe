from __future__ import annotations

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
        return _FakeResult(output="All checks passed.\n\nVERDICT: PASS — green")

    monkeypatch.setattr("vibe.core.workflows.runtime._spawn_isolated", fake_spawn)
    result = _FakeResult("did the work")
    wt = _FakeWT()

    verdict = await _run_verifier_in_worktree(rt, wt, result, None)

    assert verdict is True
    assert state.last_verifier_pass is not None
    assert state.last_verifier_pass.summary.startswith("VERDICT: PASS")


async def test_run_verifier_fail_blocks_and_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = VerificationState()
    rt = _make_runtime(state)

    async def fake_spawn(
        wt: Any, prompt: str, agent: str, max_turns: int, **kw: Any
    ) -> Any:
        return _FakeResult(output="Found bugs.\nVERDICT: FAIL")

    monkeypatch.setattr("vibe.core.workflows.runtime._spawn_isolated", fake_spawn)

    verdict = await _run_verifier_in_worktree(rt, _FakeWT(), _FakeResult("work"), None)

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

    verdict = await _run_verifier_in_worktree(rt, _FakeWT(), _FakeResult("work"), None)

    assert verdict is False
    assert state.last_verifier_pass is None


async def test_run_verifier_noop_without_state(monkeypatch: pytest.MonkeyPatch) -> None:
    rt = _make_runtime(state=None)

    async def fake_spawn(
        wt: Any, prompt: str, agent: str, max_turns: int, **kw: Any
    ) -> Any:
        return _FakeResult(output="VERDICT: PASS")

    monkeypatch.setattr("vibe.core.workflows.runtime._spawn_isolated", fake_spawn)

    verdict = await _run_verifier_in_worktree(rt, _FakeWT(), _FakeResult("work"), None)

    assert verdict is True


async def test_run_verifier_partial_blocks() -> None:
    from unittest.mock import AsyncMock

    state = VerificationState()
    rt = _make_runtime(state)
    rt.parent_context = InvokeContext(tool_call_id="t1", verification_state=state)

    fake_result = _FakeResult(output="Some checks failed.\nVERDICT: PARTIAL")
    mock_spawn = AsyncMock(return_value=fake_result)

    import vibe.core.workflows.runtime as rt_mod

    original = rt_mod._spawn_isolated
    rt_mod._spawn_isolated = mock_spawn
    try:
        verdict = await _run_verifier_in_worktree(
            rt, _FakeWT(), _FakeResult("work"), None
        )
    finally:
        rt_mod._spawn_isolated = original

    assert verdict is False
    assert state.last_verifier_pass is None
