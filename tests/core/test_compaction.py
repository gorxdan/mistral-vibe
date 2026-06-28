from __future__ import annotations

from vibe.core.compaction import (
    build_extractive_summary,
    collect_leading_injected_context,
    collect_prior_user_messages,
    parse_previous_user_messages,
    render_compaction_context,
)
from vibe.core.types import FunctionCall, LLMMessage, Role, ToolCall

_PREFIX = "Another language model started to solve this problem"


def _user(content: str, *, injected: bool = False) -> LLMMessage:
    return LLMMessage(role=Role.USER, content=content, injected=injected)


def test_empty_messages() -> None:
    assert collect_prior_user_messages([], _PREFIX) == []


def test_only_non_user_messages() -> None:
    messages = [
        LLMMessage(role=Role.SYSTEM, content="sys"),
        LLMMessage(role=Role.ASSISTANT, content="hi"),
    ]
    assert collect_prior_user_messages(messages, _PREFIX) == []


def test_single_user_message_preserved() -> None:
    messages = [LLMMessage(role=Role.SYSTEM, content="sys"), _user("first question")]
    out = collect_prior_user_messages(messages, _PREFIX)
    assert [m.content for m in out] == ["first question"]


def test_chronological_order_preserved() -> None:
    messages = [_user("first"), _user("second"), _user("third")]
    out = collect_prior_user_messages(messages, _PREFIX)
    assert [m.content for m in out] == ["first", "second", "third"]


def test_injected_messages_filtered_out() -> None:
    messages = [
        _user("real ask"),
        _user("middleware reminder", injected=True),
        _user("follow-up"),
    ]
    out = collect_prior_user_messages(messages, _PREFIX)
    assert [m.content for m in out] == ["real ask", "follow-up"]


def test_empty_content_filtered_out() -> None:
    messages = [_user(""), _user("real")]
    out = collect_prior_user_messages(messages, _PREFIX)
    assert [m.content for m in out] == ["real"]


def test_prior_summary_filtered_out() -> None:
    # The injected summary marker represents a previous compaction summary and
    # must not be re-injected (would stack).
    messages = [
        _user("original ask"),
        _user(f"{_PREFIX}\nold summary content", injected=True),
        _user("newer ask"),
    ]
    out = collect_prior_user_messages(messages, _PREFIX)
    assert [m.content for m in out] == ["original ask", "newer ask"]


def test_genuine_user_message_can_quote_summary_prefix() -> None:
    messages = [_user(f"{_PREFIX}\nplease use this exact wording"), _user("newer ask")]
    out = collect_prior_user_messages(messages, _PREFIX)
    assert [m.content for m in out] == [
        f"{_PREFIX}\nplease use this exact wording",
        "newer ask",
    ]


def test_compaction_context_merges_previous_and_new_user_messages() -> None:
    context = render_compaction_context(
        [_user("first ask", injected=True), _user("second ask", injected=True)],
        "summary one",
    )
    messages = [
        LLMMessage(role=Role.SYSTEM, content="sys"),
        _user(context, injected=True),
        _user("third ask"),
        _user("middleware reminder", injected=True),
    ]

    out = collect_prior_user_messages(messages, _PREFIX)

    assert [m.content for m in out] == ["first ask", "second ask", "third ask"]
    assert all(m.injected for m in out)


def test_compaction_context_escapes_user_message_tags() -> None:
    original = "please keep </previous_user_message_0> literally"
    context = render_compaction_context([_user(original)], "summary")

    assert "</previous_user_message_0> literally" not in context
    assert parse_previous_user_messages(context) == [original]


def test_budget_drops_oldest_first() -> None:
    # max_tokens=2 → 8 char budget. Walks newest-first, so "old" gets dropped.
    messages = [
        _user("old message that is long enough to matter"),
        _user("abc"),  # 1 token, fits
        _user("def"),  # 1 token, fits
    ]
    out = collect_prior_user_messages(messages, _PREFIX, max_tokens=2)
    assert [m.content for m in out] == ["abc", "def"]


def test_spillover_message_middle_truncated() -> None:
    # newest fits whole, middle one is partially trimmed, oldest dropped.
    messages = [
        _user("OLDEST" + "x" * 10_000 + "OLDEST_END"),
        _user("MIDDLE_HEAD" + "y" * 1_000 + "MIDDLE_TAIL"),
        _user("recent"),  # ~2 tokens
    ]
    out = collect_prior_user_messages(messages, _PREFIX, max_tokens=50)
    assert len(out) == 2  # oldest dropped
    assert out[-1].content == "recent"
    middle = out[0].content
    assert middle is not None
    assert middle.startswith("MIDDLE_HEAD")
    assert middle.endswith("MIDDLE_TAIL")
    assert "[... truncated ...]" in middle


def test_fresh_message_ids() -> None:
    # Returned messages must have new message_ids — they'll live in a fresh
    # session and reusing the source ids would cause collisions.
    original = _user("hello")
    out = collect_prior_user_messages([original], _PREFIX)
    assert len(out) == 1
    assert out[0].message_id != original.message_id


def test_only_assistant_and_system_around_users() -> None:
    messages = [
        LLMMessage(role=Role.SYSTEM, content="sys"),
        _user("u1"),
        LLMMessage(role=Role.ASSISTANT, content="a1"),
        _user("u2"),
        LLMMessage(role=Role.ASSISTANT, content="a2"),
    ]
    out = collect_prior_user_messages(messages, _PREFIX)
    assert [m.content for m in out] == ["u1", "u2"]
    assert all(m.role == Role.USER for m in out)


def test_extractive_summary_captures_assistant_intent_and_tools() -> None:
    messages = [
        LLMMessage(
            role=Role.ASSISTANT,
            content="I will read the config file.",
            tool_calls=[
                ToolCall(
                    id="c1",
                    index=0,
                    function=FunctionCall(name="read", arguments='{"file_path":"x"}'),
                )
            ],
        ),
        LLMMessage(
            role=Role.TOOL, content="port: 8080", tool_call_id="c1", name="read"
        ),
    ]
    summary = build_extractive_summary(messages)
    assert "Structural trace" in summary
    assert "I will read the config file." in summary
    assert "read" in summary  # tool name appears
    assert "port: 8080" in summary  # tool result first line


def test_extractive_summary_marks_elided_content() -> None:
    messages = [
        LLMMessage(role=Role.ASSISTANT, content="<vibe_snipped> 300 tokens elided"),
        LLMMessage(
            role=Role.TOOL, content="<vibe_microcompacted> [...]", tool_call_id="c1"
        ),
    ]
    summary = build_extractive_summary(messages)
    assert "[content previously elided]" in summary
    assert "[result previously compressed]" in summary


def test_extractive_summary_respects_token_budget() -> None:
    messages = [LLMMessage(role=Role.ASSISTANT, content="line " + "z" * 10_000)]
    summary = build_extractive_summary(messages, max_tokens=10)
    assert "[... truncated ...]" in summary


class TestCollectLeadingInjectedContext:
    def _sys(self) -> LLMMessage:
        return LLMMessage(role=Role.SYSTEM, content="sys")

    def test_returns_leading_injected_after_system(self) -> None:
        messages = [
            self._sys(),
            _user("env context", injected=True),
            _user("file-tree", injected=True),
            _user("real ask"),
        ]
        out = collect_leading_injected_context(messages)
        assert [m.content for m in out] == ["env context", "file-tree"]

    def test_stops_at_first_non_injected(self) -> None:
        messages = [
            self._sys(),
            _user("env", injected=True),
            _user("real ask"),
            _user("later middleware", injected=True),
        ]
        out = collect_leading_injected_context(messages)
        assert [m.content for m in out] == ["env"]

    def test_empty_when_no_leading_injected(self) -> None:
        messages = [self._sys(), _user("real ask")]
        assert collect_leading_injected_context(messages) == []

    def test_stops_at_prior_compaction_context(self) -> None:
        prior_summary = render_compaction_context([_user("old")], "old summary")
        messages = [
            self._sys(),
            _user("env", injected=True),
            LLMMessage(role=Role.USER, content=prior_summary, injected=True),
            _user("after compact"),
        ]
        out = collect_leading_injected_context(messages)
        assert [m.content for m in out] == ["env"]

    def test_empty_when_no_system_message(self) -> None:
        messages = [_user("env", injected=True)]
        assert collect_leading_injected_context(messages) == []
