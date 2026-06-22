from __future__ import annotations

import pytest

from tests.conftest import build_test_agent_loop
from vibe.core.hooks._handler import HookRetryState
from vibe.core.hooks._user_prompt_submit import UserPromptSubmitHandler
from vibe.core.hooks.manager import _HANDLERS
from vibe.core.hooks.models import (
    HookConfig,
    HookPromptBlock,
    HookSpecificOutput,
    HookStructuredResponse,
    HookType,
    HookUserMessage,
    UserPromptSubmitInvocation,
)


def _hook() -> HookConfig:
    return HookConfig(name="h", type=HookType.USER_PROMPT_SUBMIT, command="echo")


def _inv() -> UserPromptSubmitInvocation:
    return UserPromptSubmitInvocation(
        session_id="s", transcript_path="t", cwd="/x", prompt="hi"
    )


def test_registered() -> None:
    assert HookType.USER_PROMPT_SUBMIT in _HANDLERS


def test_deny_yields_prompt_block_and_breaks() -> None:
    action = UserPromptSubmitHandler().on_structured(
        _hook(),
        _inv(),
        HookStructuredResponse(decision="deny", reason="contains a secret"),
        HookRetryState(),
    )
    assert action.should_break is True
    blocks = [e for e in action.events if isinstance(e, HookPromptBlock)]
    assert blocks and blocks[0].content == "contains a secret"


def test_allow_with_additional_context_injects() -> None:
    action = UserPromptSubmitHandler().on_structured(
        _hook(),
        _inv(),
        HookStructuredResponse(
            decision="allow",
            hook_specific_output=HookSpecificOutput(additional_context="ctx note"),
        ),
        HookRetryState(),
    )
    assert action.should_break is False
    injects = [e for e in action.events if isinstance(e, HookUserMessage)]
    assert injects and injects[0].content == "ctx note"


# --------------------------------------------------------------------------- #
# Dispatch + loop integration                                                  #
# --------------------------------------------------------------------------- #


class _FakeManager:
    def __init__(self, *events) -> None:
        self._events = events

    def reset_retry_count(self) -> None:
        pass

    async def run(self, invocation):
        for e in self._events:
            yield e


@pytest.mark.asyncio
async def test_dispatch_block_and_inject() -> None:
    loop = build_test_agent_loop()
    loop._hooks_manager = _FakeManager(HookPromptBlock(hook_name="h", content="no"))  # type: ignore[assignment]
    reason, injected, _ = await loop._dispatch_user_prompt_submit_hooks("p", "m", False)
    assert reason == "no" and injected == []

    loop._hooks_manager = _FakeManager(HookUserMessage(content="extra"))  # type: ignore[assignment]
    reason, injected, _ = await loop._dispatch_user_prompt_submit_hooks("p", "m", False)
    assert reason is None and injected == ["extra"]


@pytest.mark.asyncio
async def test_blocked_prompt_runs_no_llm_turn() -> None:
    loop = build_test_agent_loop()
    loop._hooks_manager = _FakeManager(  # type: ignore[assignment]
        HookPromptBlock(hook_name="h", content="blocked: secret detected")
    )
    called = {"turn": 0}

    async def fake_turn():
        called["turn"] += 1
        return
        yield  # pragma: no cover

    loop._perform_llm_turn = fake_turn  # type: ignore[method-assign]

    events = [e async for e in loop._conversation_loop("leak my key")]
    assert called["turn"] == 0, "no LLM turn when the prompt is blocked"
    # Transcript: the user prompt + a synthetic assistant 'blocked' reply.
    assert loop.messages[-1].role.value == "assistant"
    assert "blocked: secret detected" in (loop.messages[-1].content or "")
    assert events


@pytest.mark.asyncio
async def test_blocked_prompt_is_redacted_from_transcript() -> None:
    # A denied prompt (e.g. one containing a secret) must not be retained raw in
    # the transcript nor re-sent to the model on later turns. The slot stays for
    # coherence, but the content is redacted.
    loop = build_test_agent_loop()
    loop._hooks_manager = _FakeManager(  # type: ignore[assignment]
        HookPromptBlock(hook_name="h", content="secret detected")
    )

    async def fake_turn():
        raise AssertionError("no LLM turn when blocked")
        yield  # pragma: no cover

    loop._perform_llm_turn = fake_turn  # type: ignore[method-assign]

    await _drain(loop._conversation_loop("AWS_KEY=sk-supersecret"))

    user_msgs = [m for m in loop.messages if m.role.value == "user"]
    assert user_msgs, "user slot retained for transcript coherence"
    assert "sk-supersecret" not in "".join(m.content or "" for m in user_msgs), (
        "raw denied prompt must not persist"
    )
    assert any("redacted" in (m.content or "") for m in user_msgs)


async def _drain(gen) -> None:
    async for _ in gen:
        pass


class _StopLoop(Exception):
    pass


@pytest.mark.asyncio
async def test_injected_context_added_before_turn() -> None:
    loop = build_test_agent_loop()
    loop._hooks_manager = _FakeManager(HookUserMessage(content="REMEMBER: be terse"))  # type: ignore[assignment]

    async def fake_turn():
        # Injection already happened above the turn; bail before the loop's
        # post-turn token accounting (keeps the test fast + deterministic).
        raise _StopLoop
        yield  # pragma: no cover

    loop._perform_llm_turn = fake_turn  # type: ignore[method-assign]
    with pytest.raises(_StopLoop):
        async for _ in loop._conversation_loop("do it"):
            pass
    injected = [
        m for m in loop.messages if m.injected and "REMEMBER" in (m.content or "")
    ]
    assert injected, "additional_context injected as a user message"
