from __future__ import annotations

import json
import os
from typing import Literal

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from vibe.core.agent_loop import AgentLoop
from vibe.core.compaction import render_compaction_context
from vibe.core.config._settings import ContextShapingConfig, MemoryConfig
from vibe.core.types import (
    FunctionCall,
    InjectedMessageKind,
    LLMMessage,
    Role,
    ToolCall,
)


def _loop(mode: Literal["system", "late"]) -> AgentLoop:
    config = build_test_vibe_config(memory=MemoryConfig(inject_mode=mode))
    loop = build_test_agent_loop(config=config)
    loop.messages.append(LLMMessage(role=Role.USER, content="first question"))
    loop.messages.append(LLMMessage(role=Role.ASSISTANT, content="an answer"))
    loop.messages.append(LLMMessage(role=Role.USER, content="latest question"))
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

    # ... but present in what the backend receives, at the absolute tail.
    sent = _sent(loop)
    assert len(sent) == len(loop.messages) + 1
    assert "RECALL-BODY" in (sent[-1].content or "")
    assert sent[-1].role == Role.USER
    assert sent[-2].content == "latest question"


def test_late_mode_empty_section_injects_nothing():
    loop = _loop("late")
    loop._set_memory_section("")
    assert len(_sent(loop)) == len(loop.messages)


def test_tail_anchor_keeps_history_prefix_across_turns():
    loop = _loop("late")
    loop._set_memory_section("SELECTION-V1")
    turn_n = _sent(loop)
    assert turn_n[-1].injected_kind == InjectedMessageKind.MEMORY

    loop.messages.append(LLMMessage(role=Role.ASSISTANT, content="answer N"))
    loop.messages.append(LLMMessage(role=Role.USER, content="question N+1"))
    loop._set_memory_section("SELECTION-V2")
    turn_n1 = _sent(loop)

    # Turn N's request minus its final mem message is a verbatim prefix of
    # turn N+1's request: no persisted message ever changes absolute position.
    persisted = [(m.role, m.content) for m in turn_n[:-1]]
    assert [(m.role, m.content) for m in turn_n1[: len(persisted)]] == persisted


def test_tail_anchor_rides_after_tool_results_intra_turn():
    loop = _loop("late")
    loop._set_memory_section("RECALL-BODY")
    loop.messages.append(
        LLMMessage(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[
                ToolCall(
                    id="tc1",
                    index=0,
                    function=FunctionCall(name="grep", arguments="{}"),
                )
            ],
        )
    )
    loop.messages.append(LLMMessage(role=Role.TOOL, content="out", tool_call_id="tc1"))

    sent = _sent(loop)
    assert "RECALL-BODY" in (sent[-1].content or "")
    assert sent[-1].role == Role.USER
    assert sent[-2].role == Role.TOOL


def test_before_user_anchor_restores_legacy_placement():
    config = build_test_vibe_config(
        memory=MemoryConfig(inject_mode="late", late_anchor="before-user")
    )
    loop = build_test_agent_loop(config=config)
    loop.messages.append(LLMMessage(role=Role.USER, content="first question"))
    loop.messages.append(LLMMessage(role=Role.ASSISTANT, content="an answer"))
    loop.messages.append(LLMMessage(role=Role.USER, content="latest question"))
    loop._set_memory_section("RECALL-BODY")

    sent = _sent(loop)
    mem_idx = next(i for i, m in enumerate(sent) if "RECALL-BODY" in (m.content or ""))
    assert sent[mem_idx].role == Role.USER
    assert sent[mem_idx + 1].content == "latest question"
    assert sent[-1].content == "latest question"


def test_injected_index_clipped_while_selector_view_unclipped(tmp_path):
    from vibe.core.memory.models import MemoryEntry, MemoryMetadata
    from vibe.core.memory.store import MemoryStore

    config = build_test_vibe_config(
        memory=MemoryConfig(inject_mode="late", index_entry_max_chars=100)
    )
    loop = build_test_agent_loop(config=config)
    store = MemoryStore(user_dir=tmp_path)
    store.upsert(
        MemoryEntry(
            metadata=MemoryMetadata(
                id="wordy",
                title="Wordy",
                description="detail " * 30,
                tags=["alpha", "beta"],
            ),
            body="b",
        )
    )

    injected = loop._injected_index_markdown(store)
    assert all(len(line) <= 100 for line in injected.splitlines())
    selector_lines = store.index(loop.config.memory.max_entries_scanned)
    assert any(len(line) > 100 for line in selector_lines)


def _divergence_vs_system_len(loop: AgentLoop) -> tuple[int, int]:
    # (common-prefix len across a selection change, serialized system-message len)
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


def test_late_memory_message_is_typed_as_injected_memory():
    loop = _loop("late")
    loop._set_memory_section("RECALL-BODY")

    sent = _sent(loop)
    mem_msg = next(m for m in sent if "RECALL-BODY" in (m.content or ""))

    assert mem_msg.injected is True
    assert mem_msg.injected_kind == InjectedMessageKind.MEMORY


def test_injected_context_is_capped_for_backend_without_mutating_history():
    config = build_test_vibe_config(
        memory=MemoryConfig(inject_mode="late"),
        context_shaping=ContextShapingConfig(max_injected_message_tokens=12),
    )
    loop = build_test_agent_loop(config=config)
    content = "HEAD-" + ("x" * 400) + "-TAIL"
    loop.messages.append(
        LLMMessage(
            role=Role.USER,
            content=content,
            injected=True,
            injected_kind=InjectedMessageKind.USER_CONTEXT,
        )
    )
    loop.messages.append(LLMMessage(role=Role.USER, content="latest"))

    sent = _sent(loop)
    capped = sent[1]

    assert "[... truncated ...]" in (capped.content or "")
    assert (capped.content or "").startswith("HEAD-")
    assert (capped.content or "").endswith("-TAIL")
    assert loop.messages[1].content == content


def test_compaction_context_cap_preserves_persisted_tool_outputs_tail():
    config = build_test_vibe_config(
        memory=MemoryConfig(inject_mode="late"),
        context_shaping=ContextShapingConfig(max_injected_message_tokens=24),
    )
    loop = build_test_agent_loop(config=config)
    compaction_context = render_compaction_context(
        [], "SUMMARY-" + ("y" * 800), ["/tmp/full-tool-output.txt"]
    )
    loop.messages.append(
        LLMMessage(
            role=Role.USER,
            content=compaction_context,
            injected=True,
            injected_kind=InjectedMessageKind.COMPACTION_CONTEXT,
        )
    )
    loop.messages.append(LLMMessage(role=Role.USER, content="latest"))

    sent = _sent(loop)
    capped = sent[1].content or ""

    assert len(capped) < len(compaction_context)
    assert "<persisted_tool_outputs>" in capped
    assert "/tmp/full-tool-output.txt" in capped
    assert "</persisted_tool_outputs>" in capped
