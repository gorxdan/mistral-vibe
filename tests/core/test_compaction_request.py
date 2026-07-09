from __future__ import annotations

from vibe.core._compaction_request import with_compaction_system_prompt
from vibe.core.types import LLMMessage, Role


def test_compaction_replaces_main_system_prompt() -> None:
    messages = [
        LLMMessage(role=Role.SYSTEM, content="large coding prompt"),
        LLMMessage(role=Role.USER, content="fix the bug"),
    ]

    result = with_compaction_system_prompt(messages)

    assert len(result) == 2
    assert result[0].role == Role.SYSTEM
    assert "Summarize a coding-agent transcript" in (result[0].content or "")
    assert "large coding prompt" not in (result[0].content or "")
    assert result[1] is messages[1]


def test_compaction_adds_system_prompt_when_missing() -> None:
    user = LLMMessage(role=Role.USER, content="continue")

    result = with_compaction_system_prompt([user])

    assert [message.role for message in result] == [Role.SYSTEM, Role.USER]
    assert result[1] is user
