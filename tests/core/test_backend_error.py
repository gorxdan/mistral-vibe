from __future__ import annotations

import pytest

from tests.constants import CHAT_COMPLETIONS_PATH
from vibe.core.llm.exceptions import BackendError, PayloadSummary


def _make_payload_summary() -> PayloadSummary:
    return PayloadSummary(
        model="test-model",
        message_count=1,
        approx_chars=10,
        temperature=0.7,
        has_tools=False,
        tool_choice=None,
    )


def _make_error(
    *,
    status: int | None,
    headers: dict[str, str] | None = None,
    body_text: str = "body",
) -> BackendError:
    return BackendError(
        provider="test-provider",
        endpoint=CHAT_COMPLETIONS_PATH,
        status=status,
        reason="some reason",
        headers=headers or {},
        body_text=body_text,
        parsed_error=None,
        model="test-model",
        payload_summary=_make_payload_summary(),
    )


class TestContextTooLongClassification:
    def test_qwen_template_truncation_is_context_too_long(self) -> None:
        # When an over-window request is truncated to fit num_ctx, the oldest
        # messages (incl. the user query) drop out and strict chat templates
        # (Qwen3/Ornith) hard-400 with this body. The harness always sends a
        # user query, so this signature only arises from over-window truncation
        # -> route it to the compact-and-retry recovery, not a hard failure.
        body = (
            '{"error":{"code":400,"message":"Unable to generate parser for this '
            "template. Automatic parser generation failed: Jinja Exception: "
            'No user query found in messages."}}'
        )
        err = _make_error(status=400, body_text=body)
        assert err.is_context_too_long is True

    def test_unrelated_400_is_not_context_too_long(self) -> None:
        err = _make_error(status=400, body_text="some other bad request")
        assert err.is_context_too_long is False


class TestBackendErrorFmt:
    def test_standard_status_code(self) -> None:
        err = _make_error(status=500)
        msg = str(err)
        assert "500 Internal Server Error" in msg
        assert "test-provider" in msg

    def test_non_standard_status_code(self) -> None:
        """Status 529 is not in HTTPStatus and previously raised ValueError."""
        err = _make_error(status=529)
        msg = str(err)
        assert "529" in msg
        # Should not contain a phrase since 529 is not standard
        assert "LLM backend error [test-provider]" in msg

    def test_no_status(self) -> None:
        err = _make_error(status=None)
        msg = str(err)
        assert "status: N/A" in msg

    def test_unauthorized_short_circuits(self) -> None:
        err = _make_error(status=401)
        assert str(err) == "Invalid API key. Please check your API key and try again."

    def test_rate_limit_short_circuits(self) -> None:
        err = _make_error(status=429)
        assert (
            str(err) == "Rate limit exceeded. Please wait a moment before trying again."
        )

    def test_request_id_from_headers(self) -> None:
        err = _make_error(status=500, headers={"x-request-id": "req-123"})
        assert "req-123" in str(err)

    @pytest.mark.parametrize("code", [530, 599, 999])
    def test_other_non_standard_codes(self, code: int) -> None:
        err = _make_error(status=code)
        msg = str(err)
        assert str(code) in msg
        assert "LLM backend error" in msg


class TestBackendErrorIsContextTooLong:
    @pytest.mark.parametrize(
        ("status", "body_text"),
        [
            (400, "context too long"),
            (400, "prompt is too long"),
            # orchestral_runtime wraps context errors as 422
            (422, '{"error":{"type":"model_context_exceeded"}}'),
            (422, '{"error":{"type":"prompt_too_long"}}'),
            # kimi phrases overflow as "exceeded model token limit"
            (
                400,
                '{"error":{"message":"Invalid request: Your request exceeded '
                'model token limit: 262144 (requested: 262225)",'
                '"type":"invalid_request_error"}}',
            ),
        ],
    )
    def test_true(self, status: int, body_text: str) -> None:
        err = _make_error(status=status, body_text=body_text)
        assert err.is_context_too_long

    def test_false_on_unrelated_status(self) -> None:
        err = _make_error(status=500, body_text="context too long")
        assert not err.is_context_too_long

    def test_false_on_max_tokens(self) -> None:
        # max-tokens truncation must not be misread as context-too-long
        err = _make_error(status=422, body_text="max_tokens_exceeded")
        assert not err.is_context_too_long


class TestBackendErrorIsContentFiltered:
    @pytest.mark.parametrize(
        "body_text",
        [
            '{"contentFilter":[{"level":1,"role":"assistant"}],"error":{"code":"1301"}}',
            '{"error":{"code": "1301"}}',
            "request blocked by ContentFilter policy",
        ],
    )
    def test_true_on_400(self, body_text: str) -> None:
        err = _make_error(status=400, body_text=body_text)
        assert err.is_content_filtered

    def test_false_on_non_400(self) -> None:
        err = _make_error(status=422, body_text='{"error":{"code":"1301"}}')
        assert not err.is_content_filtered

    def test_false_without_filter_markers(self) -> None:
        err = _make_error(status=400, body_text="some unrelated 400")
        assert not err.is_content_filtered


class TestBackendErrorIsResponseTooLong:
    @pytest.mark.parametrize(
        "body_text",
        [
            '{"error":{"type":"max_tokens_exceeded"}}',
            "Generation truncated: finish_reason=length",
        ],
    )
    def test_true_on_422_with_substring(self, body_text: str) -> None:
        err = _make_error(status=422, body_text=body_text)
        assert err.is_response_too_long

    def test_false_when_status_not_422(self) -> None:
        err = _make_error(status=400, body_text="max_tokens_exceeded")
        assert not err.is_response_too_long

    def test_false_when_substring_missing(self) -> None:
        err = _make_error(status=422, body_text="some unrelated error")
        assert not err.is_response_too_long


class TestBackendErrorIsStructuredOutputRejected:
    @pytest.mark.parametrize(
        ("status", "body_text"),
        [
            # OpenAI Responses API rejecting the text.format payload
            (
                400,
                '{"error":{"message":"Missing required parameter: \'text.format.name\'."}}',
            ),
            # Chat Completions providers that reject response_format outright
            (400, "response_format is not supported by this model"),
            (422, "unknown parameter: json_schema"),
        ],
    )
    def test_true(self, status: int, body_text: str) -> None:
        err = _make_error(status=status, body_text=body_text)
        assert err.is_structured_output_rejected

    @pytest.mark.parametrize("status", [500, 401, 429])
    def test_false_on_unrelated_status(self, status: int) -> None:
        err = _make_error(status=status, body_text="response_format boom")
        assert not err.is_structured_output_rejected

    def test_false_on_400_without_format_substring(self) -> None:
        # A 400 about the message content, not the format, must not trigger
        # structured-output degradation.
        err = _make_error(status=400, body_text="invalid model id")
        assert not err.is_structured_output_rejected
