from __future__ import annotations

from http import HTTPStatus
import io
from pathlib import Path

import httpx
import pytest

import vibe.core.feedback as feedback_mod
from vibe.core.output_formatters import (
    JsonOutputFormatter,
    StreamingJsonOutputFormatter,
    TextOutputFormatter,
    create_formatter,
)
from vibe.core.plan_session import PlanSession
from vibe.core.teleport.types import (
    TeleportCheckingGitEvent,
    TeleportCompleteEvent,
    TeleportPushingEvent,
    TeleportPushRequiredEvent,
    TeleportStartingWorkflowEvent,
)
from vibe.core.types import AssistantEvent, LLMMessage, OutputFormat, Role

# --------------------------------------------------------------------------- #
# plan_session                                                                #
# --------------------------------------------------------------------------- #


def test_plan_session_read_returns_none_without_file() -> None:
    assert PlanSession().read() is None


def test_plan_session_read_returns_none_for_missing_file(tmp_path: Path) -> None:
    session = PlanSession()
    session._plan_file_path = tmp_path / "missing.md"  # type: ignore[attr-defined]
    assert session.read() is None


def test_plan_session_read_returns_content(tmp_path: Path) -> None:
    plan = tmp_path / "plan.md"
    plan.write_text("# My Plan")
    session = PlanSession()
    session._plan_file_path = plan  # type: ignore[attr-defined]
    assert session.read() == "# My Plan"


def test_plan_session_snapshot_and_change_detection(tmp_path: Path) -> None:
    plan = tmp_path / "plan.md"
    plan.write_text("v1")
    session = PlanSession()
    session._plan_file_path = plan  # type: ignore[attr-defined]
    session.snapshot_content_hash()
    assert session.has_content_changed() is False
    plan.write_text("v2")
    assert session.has_content_changed() is True


def test_plan_file_path_str_generated_once() -> None:
    session = PlanSession()
    p1 = session.plan_file_path_str
    p2 = session.plan_file_path_str
    assert p1 == p2
    assert p1.endswith(".md")


# --------------------------------------------------------------------------- #
# output_formatters                                                           #
# --------------------------------------------------------------------------- #


def _msg(content: str = "hi") -> LLMMessage:
    return LLMMessage(role=Role.assistant, content=content)


def test_text_formatter_handles_assistant_and_teleport_events() -> None:
    stream = io.StringIO()
    fmt = TextOutputFormatter(stream)
    fmt.on_message_added(_msg())
    fmt.on_event(AssistantEvent(content="answer", message_id="m1"))
    assert fmt._final_response == "answer"

    fmt.on_event(TeleportCheckingGitEvent())
    fmt.on_event(TeleportPushRequiredEvent(unpushed_count=2))
    fmt.on_event(TeleportPushingEvent())
    fmt.on_event(TeleportStartingWorkflowEvent())
    fmt.on_event(TeleportCompleteEvent(url="https://example.com"))
    assert fmt._final_response == "https://example.com"
    out = stream.getvalue()
    assert "Pushing 2 commit" in out
    assert "Syncing" in out
    assert "Teleporting" in out


def test_text_formatter_finalize_returns_final_response() -> None:
    fmt = TextOutputFormatter()
    assert fmt.finalize() is None
    fmt._final_response = "done"
    assert fmt.finalize() == "done"


def test_json_formatter_dumps_messages() -> None:
    stream = io.StringIO()
    fmt = JsonOutputFormatter(stream)
    fmt.on_message_added(_msg("payload"))
    assert fmt.finalize() is None
    out = stream.getvalue()
    assert "payload" in out


def test_streaming_json_formatter_emits_on_message_added() -> None:
    stream = io.StringIO()
    fmt = StreamingJsonOutputFormatter(stream)
    fmt.on_message_added(_msg("chunk"))
    assert "chunk" in stream.getvalue()
    assert fmt.finalize() is None


def test_create_formatter_returns_correct_type() -> None:
    assert isinstance(create_formatter(OutputFormat.TEXT), TextOutputFormatter)
    assert isinstance(create_formatter(OutputFormat.JSON), JsonOutputFormatter)
    assert isinstance(
        create_formatter(OutputFormat.STREAMING), StreamingJsonOutputFormatter
    )
    assert isinstance(create_formatter("unknown"), TextOutputFormatter)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# feedback.should_show_feedback                                               #
# --------------------------------------------------------------------------- #


def test_should_show_feedback_requires_telemetry_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feedback_mod, "FEEDBACK_PROBABILITY", 1.0)
    assert (
        feedback_mod.should_show_feedback(
            telemetry_active=False, is_mistral_model=True, user_message_count=10
        )
        is False
    )


def test_should_show_feedback_requires_mistral_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feedback_mod, "FEEDBACK_PROBABILITY", 1.0)
    assert (
        feedback_mod.should_show_feedback(
            telemetry_active=True, is_mistral_model=False, user_message_count=10
        )
        is False
    )


def test_should_show_feedback_requires_min_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feedback_mod, "FEEDBACK_PROBABILITY", 1.0)
    assert (
        feedback_mod.should_show_feedback(
            telemetry_active=True, is_mistral_model=True, user_message_count=2
        )
        is False
    )


def test_should_show_feedback_passes_when_all_conditions_met(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feedback_mod, "FEEDBACK_PROBABILITY", 1.0)
    monkeypatch.setattr(
        "vibe.core.feedback.read_cache",
        lambda _p: {"user_feedback": {"last_shown_at": 0}},
    )
    assert (
        feedback_mod.should_show_feedback(
            telemetry_active=True, is_mistral_model=True, user_message_count=5
        )
        is True
    )


def test_should_show_feedback_rejects_non_int_last_shown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feedback_mod, "FEEDBACK_PROBABILITY", 1.0)
    monkeypatch.setattr(
        "vibe.core.feedback.read_cache",
        lambda _p: {"user_feedback": {"last_shown_at": "not-int"}},
    )
    assert (
        feedback_mod.should_show_feedback(
            telemetry_active=True, is_mistral_model=True, user_message_count=5
        )
        is False
    )
