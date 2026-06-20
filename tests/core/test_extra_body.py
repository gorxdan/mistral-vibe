from __future__ import annotations

from vibe.core.llm.backend.generic import OpenAIAdapter


def _payload(extra_body: dict | None) -> dict:
    return OpenAIAdapter().build_payload(
        model_name="glm-5.2",
        converted_messages=[{"role": "user", "content": "hi"}],
        temperature=0.2,
        tools=None,
        max_tokens=100,
        tool_choice=None,
        response_format={"type": "json_object"},
        extra_body=extra_body,
    )


def test_extra_body_merged_into_payload() -> None:
    payload = _payload({"thinking": {"type": "disabled"}})
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["model"] == "glm-5.2"
    assert payload["response_format"] == {"type": "json_object"}


def test_no_extra_body_leaves_payload_unchanged() -> None:
    payload = _payload(None)
    assert "thinking" not in payload


def test_empty_extra_body_is_noop() -> None:
    payload = _payload({})
    assert "thinking" not in payload
