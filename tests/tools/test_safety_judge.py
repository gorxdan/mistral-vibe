from __future__ import annotations

from typing import Any

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from vibe.core.agent_loop import ToolExecutionResponse
from vibe.core.config import DEFAULT_MODELS, SafetyJudgeConfig
from vibe.core.tools.base import BaseToolState
from vibe.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from vibe.core.tools.safety_judge import JudgeVerdict, SafetyJudge
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
    loop = build_test_agent_loop()
    assert loop.config.safety_judge.enabled is False
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
