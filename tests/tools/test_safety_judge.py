from __future__ import annotations

from typing import Any

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from vibe.core.agent_loop import AgentLoop, ToolExecutionResponse
from vibe.core.config import DEFAULT_MODELS, SafetyJudgeConfig
from vibe.core.tools.base import BaseToolState
from vibe.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from vibe.core.tools.safety_judge import (
    _SYSTEM_PROMPT,
    _WORKFLOW_SYSTEM_PROMPT,
    JudgeVerdict,
    SafetyJudge,
    _system_prompt_for,
)
from vibe.core.types import ApprovalResponse


def _bash() -> Bash:
    return Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())


class _FakeJudge:
    """Stand-in for SafetyJudge with a fixed verdict that records its calls."""

    def __init__(self, *, safe: bool, reason: str = "stub") -> None:
        self.verdict = JudgeVerdict(safe=safe, reason=reason)
        self.calls: list[tuple[str, str, list[str]]] = []

    async def judge(
        self, tool_name: str, args_repr: str, flagged: list[str]
    ) -> JudgeVerdict:
        self.calls.append((tool_name, args_repr, flagged))
        return self.verdict


class _RecordingApproval:
    def __init__(self, response: ApprovalResponse) -> None:
        self.response = response
        self.called = False

    async def __call__(self, *args: Any, **kwargs: Any) -> tuple[ApprovalResponse, None]:
        self.called = True
        return self.response, None


# --------------------------------------------------------------------------- #
# Parser fails closed                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected_safe",
    [
        ('{"safe": true, "reason": "read only"}', True),
        ('{"safe": false, "reason": "destructive"}', False),
        ('```json\n{"safe": true, "reason": "x"}\n```', True),
        ('prefix {"safe": true, "reason": "x"} suffix', True),
        ("not json", False),
        ('{"reason": "no safe key"}', False),
        ('{"safe": "yes"}', False),  # wrong type → fail closed
        ("", False),
        (None, False),
    ],
)
def test_parser_fail_closed(raw: str | None, expected_safe: bool) -> None:
    assert SafetyJudge._parse(raw).safe is expected_safe


# --------------------------------------------------------------------------- #
# Resolver gating                                                             #
# --------------------------------------------------------------------------- #


def test_resolve_judge_none_when_disabled() -> None:
    config = build_test_vibe_config(
        safety_judge=SafetyJudgeConfig(enabled=False)
    )
    loop = build_test_agent_loop(config=config)
    assert loop.config.safety_judge.enabled is False
    assert loop._resolve_safety_judge() is None


def test_resolve_judge_none_when_enabled_but_no_model() -> None:
    # Enabled by default but no model alias -> judge cannot be built -> None.
    loop = build_test_agent_loop()
    assert loop.config.safety_judge.enabled is True
    assert loop._resolve_safety_judge() is None


def test_resolve_judge_none_when_model_alias_unknown() -> None:
    config = build_test_vibe_config(
        safety_judge=SafetyJudgeConfig(enabled=True, model="does-not-exist")
    )
    loop = build_test_agent_loop(config=config)
    assert loop._resolve_safety_judge() is None


# --------------------------------------------------------------------------- #
# Decision wiring in _should_execute_tool                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_judge_approves_executes_without_prompt() -> None:
    loop = build_test_agent_loop()
    fake = _FakeJudge(safe=True, reason="npm install is benign")
    loop._resolve_safety_judge = lambda: fake  # type: ignore[method-assign]
    approval = _RecordingApproval(ApprovalResponse.NO)
    loop.approval_callback = approval

    decision = await loop._should_execute_tool(
        _bash(), BashArgs(command="npm install"), "call-1"
    )

    assert decision.verdict == ToolExecutionResponse.EXECUTE
    assert fake.calls, "judge should have been consulted"
    assert approval.called is False, "must not prompt the user when judge approves"
    assert decision.judge_approved is True
    assert decision.feedback and "npm install is benign" in decision.feedback


@pytest.mark.asyncio
async def test_judge_rejects_falls_through_to_prompt() -> None:
    loop = build_test_agent_loop()
    fake = _FakeJudge(safe=False)
    loop._resolve_safety_judge = lambda: fake  # type: ignore[method-assign]
    approval = _RecordingApproval(ApprovalResponse.NO)
    loop.approval_callback = approval

    decision = await loop._should_execute_tool(
        _bash(), BashArgs(command="npm install"), "call-2"
    )

    assert decision.verdict == ToolExecutionResponse.SKIP
    assert fake.calls
    assert approval.called is True, "judge rejection must defer to the user"


class _NoteCapturingApproval:
    """Records the judge_note argument so a test can assert the judge's
    deferral reason was threaded through to the host-facing callback.

    Mirrors how a workflow/task subagent's denial must reach the host: the
    note travels as the callback's 5th argument, not via loop-local state
    (which is invisible across loop boundaries).
    """

    def __init__(self, response: ApprovalResponse) -> None:
        self.response = response
        self.judge_note: str | None = None
        self.called = False

    async def __call__(self, *args: Any, **kwargs: Any) -> tuple[ApprovalResponse, None]:
        self.called = True
        # The 5th positional argument is judge_note (see ApprovalCallback).
        if len(args) >= 5:
            self.judge_note = args[4]
        elif "judge_note" in kwargs:
            self.judge_note = kwargs["judge_note"]
        return self.response, None


@pytest.mark.asyncio
async def test_judge_deferral_reason_is_threaded_to_approval_callback() -> None:
    """The judge's reason must reach the host prompt even when the judged call
    originated from a subagent. _ask_approval passes pending_judge_deferral as
    the callback's 5th argument so it crosses loop boundaries.
    """
    loop = build_test_agent_loop()
    fake = _FakeJudge(safe=False, reason="could delete files")
    loop._resolve_safety_judge = lambda: fake  # type: ignore[method-assign]
    approval = _NoteCapturingApproval(ApprovalResponse.NO)
    loop.approval_callback = approval

    await loop._should_execute_tool(
        _bash(), BashArgs(command="rm -rf build"), "call-3"
    )

    assert approval.called
    assert approval.judge_note == "could delete files", (
        "judge deferral reason must be passed to the approval callback so the "
        "host can show WHY approval is needed"
    )


@pytest.mark.asyncio
async def test_denylisted_command_never_consults_judge() -> None:
    loop = build_test_agent_loop()
    fake = _FakeJudge(safe=True)  # would approve — must never be reached
    loop._resolve_safety_judge = lambda: fake  # type: ignore[method-assign]
    approval = _RecordingApproval(ApprovalResponse.YES)
    loop.approval_callback = approval

    decision = await loop._should_execute_tool(
        _bash(), BashArgs(command="vim secrets.txt"), "call-3"
    )

    assert decision.verdict == ToolExecutionResponse.SKIP
    assert fake.calls == [], "denylist (NEVER) must short-circuit before the judge"
    assert approval.called is False


@pytest.mark.asyncio
async def test_no_judge_configured_prompts_as_before() -> None:
    loop = build_test_agent_loop()  # judge disabled by default
    approval = _RecordingApproval(ApprovalResponse.YES)
    loop.approval_callback = approval

    decision = await loop._should_execute_tool(
        _bash(), BashArgs(command="npm install"), "call-4"
    )

    assert decision.verdict == ToolExecutionResponse.EXECUTE
    assert approval.called is True, "without a judge the user is still prompted"


# --------------------------------------------------------------------------- #
# SafetyJudge fails closed when the backend errors                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_safety_judge_fails_closed_on_backend_error(monkeypatch) -> None:
    model = DEFAULT_MODELS[0]

    class _BoomBackend:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _BoomBackend:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def complete(self, **kwargs: Any) -> Any:
            raise RuntimeError("backend exploded")

    fake_provider = type(
        "P", (), {"backend": "generic", "extra_headers": {}, "api_base": "", "name": "p"}
    )()
    monkeypatch.setattr(
        "vibe.core.tools.safety_judge.BACKEND_FACTORY",
        {"generic": _BoomBackend},
    )

    judge = SafetyJudge(
        model=model,
        provider=fake_provider,  # type: ignore[arg-type]
        config=SafetyJudgeConfig(enabled=True, model=model.alias),
    )
    verdict = await judge.judge("bash", '{"command":"rm -rf /"}', ["rm -rf /"])
    assert verdict.safe is False
    assert verdict.failed is True


# --------------------------------------------------------------------------- #
# Verdict cache                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_repeated_safe_call_hits_cache_and_skips_judge() -> None:
    loop = build_test_agent_loop()
    fake = _FakeJudge(safe=True, reason="benign")
    loop._resolve_safety_judge = lambda: fake  # type: ignore[method-assign]
    loop.approval_callback = _RecordingApproval(ApprovalResponse.NO)

    args = BashArgs(command="npm install")
    first = await loop._should_execute_tool(_bash(), args, "c1")
    second = await loop._should_execute_tool(_bash(), args, "c2")

    assert first.verdict == ToolExecutionResponse.EXECUTE
    assert second.verdict == ToolExecutionResponse.EXECUTE
    assert len(fake.calls) == 1, "second identical call must reuse the cached verdict"
    assert second.judge_approved is True


@pytest.mark.asyncio
async def test_repeated_unsafe_call_hits_cache_and_skips_judge() -> None:
    loop = build_test_agent_loop()
    fake = _FakeJudge(safe=False, reason="destructive")
    loop._resolve_safety_judge = lambda: fake  # type: ignore[method-assign]
    approval = _RecordingApproval(ApprovalResponse.NO)
    loop.approval_callback = approval

    args = BashArgs(command="rm -rf build")
    await loop._should_execute_tool(_bash(), args, "c1")
    await loop._should_execute_tool(_bash(), args, "c2")

    assert len(fake.calls) == 1, "cached unsafe verdict must skip re-querying"
    assert approval.called is True, "each call still defers to the user"


@pytest.mark.asyncio
async def test_distinct_args_miss_cache_and_query_judge() -> None:
    loop = build_test_agent_loop()
    fake = _FakeJudge(safe=True, reason="benign")
    loop._resolve_safety_judge = lambda: fake  # type: ignore[method-assign]
    loop.approval_callback = _RecordingApproval(ApprovalResponse.NO)

    await loop._should_execute_tool(_bash(), BashArgs(command="npm install"), "c1")
    await loop._should_execute_tool(_bash(), BashArgs(command="npm test"), "c2")

    assert len(fake.calls) == 2, "different args must not share a verdict"


@pytest.mark.asyncio
async def test_fail_closed_verdict_is_not_cached() -> None:
    loop = build_test_agent_loop()

    class _FailOnceJudge:
        """Returns a fail-closed verdict first, then a real safe one."""

        def __init__(self) -> None:
            self.calls = 0

        async def judge(self, tool_name, args_repr, flagged) -> JudgeVerdict:  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                return JudgeVerdict(safe=False, reason="timed out", failed=True)
            return JudgeVerdict(safe=True, reason="benign")

    fake = _FailOnceJudge()
    loop._resolve_safety_judge = lambda: fake  # type: ignore[method-assign]
    loop.approval_callback = _RecordingApproval(ApprovalResponse.YES)

    args = BashArgs(command="npm install")
    first = await loop._should_execute_tool(_bash(), args, "c1")  # fail-closed → prompt
    second = await loop._should_execute_tool(_bash(), args, "c2")  # retried → safe

    assert first.judge_approved is False
    assert second.judge_approved is True, "fail-closed must not poison the cache"
    assert fake.calls == 2


@pytest.mark.asyncio
async def test_cache_disabled_when_size_zero() -> None:
    config = build_test_vibe_config(
        safety_judge=SafetyJudgeConfig(
            enabled=True, model="any", verdict_cache_size=0
        )
    )
    loop = build_test_agent_loop(config=config)
    fake = _FakeJudge(safe=True, reason="benign")
    loop._resolve_safety_judge = lambda: fake  # type: ignore[method-assign]
    loop.approval_callback = _RecordingApproval(ApprovalResponse.NO)

    args = BashArgs(command="npm install")
    await loop._should_execute_tool(_bash(), args, "c1")
    await loop._should_execute_tool(_bash(), args, "c2")

    assert len(fake.calls) == 2, "verdict_cache_size=0 must disable caching"


@pytest.mark.asyncio
async def test_cache_evicts_oldest_at_capacity() -> None:
    config = build_test_vibe_config(
        safety_judge=SafetyJudgeConfig(
            enabled=True, model="any", verdict_cache_size=1
        )
    )
    loop = build_test_agent_loop(config=config)
    fake = _FakeJudge(safe=True, reason="benign")
    loop._resolve_safety_judge = lambda: fake  # type: ignore[method-assign]
    loop.approval_callback = _RecordingApproval(ApprovalResponse.NO)

    # Fill the single slot with call A, then a distinct call B evicts it.
    await loop._should_execute_tool(_bash(), BashArgs(command="npm install"), "c1")
    await loop._should_execute_tool(_bash(), BashArgs(command="npm test"), "c2")
    fake.calls.clear()
    # A was evicted by B, so re-querying A must miss and call the judge.
    await loop._should_execute_tool(_bash(), BashArgs(command="npm install"), "c3")

    assert len(fake.calls) == 1, "LRU must evict the least-recently-used entry"


@pytest.mark.asyncio
async def test_calls_differing_only_past_truncation_do_not_collide() -> None:
    """The cache key hashes the FULL args, so two calls whose 4000-char truncation
    is byte-identical but whose full content differs get distinct keys.

    Calls _judge_tool_safety directly with a fixed uncovered list so the result
    isolates cache keying from Bash's command-specific permission resolution.
    """
    loop = build_test_agent_loop()
    fake = _FakeJudge(safe=True, reason="benign")
    loop._resolve_safety_judge = lambda: fake  # type: ignore[method-assign]

    padding = "a" * 4500
    benign = BashArgs(command=f"echo {padding}")
    # Identical for the first 4000 chars of the JSON; the dangerous tail sits
    # past the truncation point, so the judge would see the same args_repr.
    dangerous = BashArgs(command=f"echo {padding}; rm -rf /important")

    args_key_a, repr_a = AgentLoop._serialize_args(benign)
    args_key_b, repr_b = AgentLoop._serialize_args(dangerous)
    assert repr_a == repr_b, "fixture premise: judge sees identical truncated args"
    assert args_key_a != args_key_b, "full-args fingerprint must distinguish them"

    await loop._judge_tool_safety("bash", benign, [])
    await loop._judge_tool_safety("bash", dangerous, [])

    assert len(fake.calls) == 2, (
        "calls differing past the 4000-char truncation must not share a verdict"
    )


@pytest.mark.asyncio
async def test_cache_cleared_when_judge_model_changes() -> None:
    config = build_test_vibe_config(
        safety_judge=SafetyJudgeConfig(enabled=True, model="alpha")
    )
    loop = build_test_agent_loop(config=config)
    fake = _FakeJudge(safe=True, reason="benign")
    loop._resolve_safety_judge = lambda: fake  # type: ignore[method-assign]
    loop.approval_callback = _RecordingApproval(ApprovalResponse.NO)

    args = BashArgs(command="npm install")
    await loop._should_execute_tool(_bash(), args, "c1")  # cached under "alpha"
    assert len(fake.calls) == 1
    # Swap the judge model mid-session.
    loop.config.safety_judge = SafetyJudgeConfig(enabled=True, model="beta")
    await loop._should_execute_tool(_bash(), args, "c2")  # must not reuse alpha's verdict

    assert len(fake.calls) == 2, (
        "cached verdict must be dropped when the judge model changes"
    )


# --------------------------------------------------------------------------- #
# Per-tool system prompt selection (workflow-aware judge)                     #
# --------------------------------------------------------------------------- #


class TestSystemPromptSelection:
    def test_launch_workflow_uses_workflow_prompt(self) -> None:
        prompt = _system_prompt_for("launch_workflow")
        assert prompt is _WORKFLOW_SYSTEM_PROMPT
        # The workflow prompt must reason about the planned agent surface,
        # not just literal command effects.
        assert "PLANNED SURFACE" in prompt
        assert "worker" in prompt

    def test_other_tools_use_default_prompt(self) -> None:
        assert _system_prompt_for("bash") is _SYSTEM_PROMPT
        assert _system_prompt_for("read") is _SYSTEM_PROMPT
        assert _system_prompt_for("unknown_tool") is _SYSTEM_PROMPT

    def test_workflow_and_default_prompts_differ(self) -> None:
        assert _WORKFLOW_SYSTEM_PROMPT != _SYSTEM_PROMPT
