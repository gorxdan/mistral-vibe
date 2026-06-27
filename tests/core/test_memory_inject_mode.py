from __future__ import annotations

import json
import os
from typing import Literal

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from vibe.core.agent_loop import AgentLoop
from vibe.core.config._settings import MemoryConfig
from vibe.core.types import LLMMessage, Role


def _loop(mode: Literal["system", "late"]) -> AgentLoop:
    config = build_test_vibe_config(memory=MemoryConfig(inject_mode=mode))
    loop = build_test_agent_loop(config=config)
    loop.messages.append(LLMMessage(role=Role.user, content="first question"))
    loop.messages.append(LLMMessage(role=Role.assistant, content="an answer"))
    loop.messages.append(LLMMessage(role=Role.user, content="latest question"))
    return loop


def _sent(loop: AgentLoop) -> list[LLMMessage]:
    return list(loop._messages_for_backend(loop.config.get_active_model()))


def test_system_mode_embeds_block_in_system_prompt():
    loop = _loop("system")
    loop._set_memory_section("RECALL-BODY")

    sys_content = loop.messages[0].content or ""
    assert "<memories>" in sys_content
    assert "RECALL-BODY" in sys_content
    # No extra ephemeral message: the backend sees exactly the history.
    assert len(_sent(loop)) == len(loop.messages)


def test_late_mode_keeps_system_stable_and_injects_ephemerally():
    loop = _loop("late")
    loop._set_memory_section("RECALL-BODY")

    # System prompt untouched ...
    assert "<memories>" not in (loop.messages[0].content or "")
    # ... block absent from persisted history (self.messages) ...
    assert not any("RECALL-BODY" in (m.content or "") for m in loop.messages)

    # ... but present in what the backend receives, right before the last user.
    sent = _sent(loop)
    assert len(sent) == len(loop.messages) + 1
    mem_idx = next(i for i, m in enumerate(sent) if "RECALL-BODY" in (m.content or ""))
    assert sent[mem_idx].role == Role.user
    assert sent[mem_idx + 1].content == "latest question"


def test_late_mode_empty_section_injects_nothing():
    loop = _loop("late")
    loop._set_memory_section("")
    assert len(_sent(loop)) == len(loop.messages)


def _divergence_vs_system_len(loop: AgentLoop) -> tuple[int, int]:
    """Return (common-prefix len across a selection change, serialized system len)."""

    def serialized() -> str:
        return "".join(
            json.dumps({"r": str(m.role), "c": m.content or ""}, sort_keys=True)
            for m in _sent(loop)
        )

    loop._set_memory_section("SELECTION-V1-aaaaaaaa")
    r1 = serialized()
    loop._set_memory_section("SELECTION-V2-bbbbbbbb")
    r2 = serialized()
    common = len(os.path.commonprefix([r1, r2]))
    sys_msg = loop.messages[0]
    sys_len = len(
        json.dumps({"r": str(sys_msg.role), "c": sys_msg.content or ""}, sort_keys=True)
    )
    return common, sys_len


def test_late_mode_protects_system_prefix_that_system_mode_busts():
    # late: divergence falls AFTER the system prompt -> system+history cached.
    common_late, sys_len_late = _divergence_vs_system_len(_loop("late"))
    assert common_late >= sys_len_late

    # system: a selection change diverges INSIDE the system prompt -> the whole
    # history behind it is no longer a cached prefix.
    common_sys, sys_len_sys = _divergence_vs_system_len(_loop("system"))
    assert common_sys < sys_len_sys
