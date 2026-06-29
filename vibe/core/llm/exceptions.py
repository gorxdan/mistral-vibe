from __future__ import annotations

from collections.abc import Mapping, Sequence
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

import httpx
import orjson
from pydantic import BaseModel, ConfigDict, ValidationError

from vibe.core.types import AvailableTool, LLMMessage, StrToolChoice

if TYPE_CHECKING:
    from mistralai.client.errors import SDKError

type HttpError = SDKError | httpx.HTTPStatusError

_CONTEXT_TOO_LONG_SUBSTRINGS = (
    "context too long",
    "maximum context length",
    "input too large",
    "couldn't fit with truncation",
    "prompt is too long",
    # orchestral_runtime returns these as 422
    "model_context_exceeded",
    "prompt_too_long",
    # kimi: "exceeded model token limit: N (requested: M)"
    "model token limit",
)

_RESPONSE_TOO_LONG_SUBSTRINGS = ("max_tokens_exceeded", "finish_reason=length")

_STRUCTURED_OUTPUT_REJECTION_SUBSTRINGS = (
    "text.format",
    "response_format",
    "json_schema",
)


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")
    message: str | None = None


class PayloadSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    message_count: int
    approx_chars: int
    temperature: float | None
    has_tools: bool
    tool_choice: StrToolChoice | AvailableTool | None


class BackendError(RuntimeError):
    def __init__(
        self,
        *,
        provider: str,
        endpoint: str,
        status: int | None,
        reason: str | None,
        headers: Mapping[str, str] | None,
        body_text: str | None,
        parsed_error: str | None,
        model: str,
        payload_summary: PayloadSummary,
    ) -> None:
        self.provider = provider
        self.endpoint = endpoint
        self.status = status
        self.reason = reason
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.body_text = body_text or ""
        self.parsed_error = parsed_error
        self.model = model
        self.payload_summary = payload_summary
        super().__init__(self._fmt())

    @property
    def is_context_too_long(self) -> bool:
        if self.status not in {HTTPStatus.BAD_REQUEST, HTTPStatus.UNPROCESSABLE_ENTITY}:
            return False
        body = (self.body_text or "").lower()
        return any(s in body for s in _CONTEXT_TOO_LONG_SUBSTRINGS)

    @property
    def is_response_too_long(self) -> bool:
        if self.status != HTTPStatus.UNPROCESSABLE_ENTITY:
            return False
        body = (self.body_text or "").lower()
        return any(s in body for s in _RESPONSE_TOO_LONG_SUBSTRINGS)

    @property
    def is_structured_output_rejected(self) -> bool:
        # 400/422 where the provider rejected the response_format / text.format
        # payload itself (e.g. an adapter sent a schema shape the endpoint does
        # not understand). Retrying without response_format, relying on the
        # prompt-level JSON fallback, recovers gracefully.
        if self.status not in {HTTPStatus.BAD_REQUEST, HTTPStatus.UNPROCESSABLE_ENTITY}:
            return False
        body = (self.body_text or "").lower()
        return any(s in body for s in _STRUCTURED_OUTPUT_REJECTION_SUBSTRINGS)

    @property
    def is_content_filtered(self) -> bool:
        # 400 where the provider's content filter rejected the prompt or the
        # generation (e.g. zai code 1301, body {"contentFilter":[...]}). Not the
        # same as a context/structured-output rejection: retrying the same
        # backend won't help — only failing over to another backend will.
        if self.status != HTTPStatus.BAD_REQUEST:
            return False
        body = (self.body_text or "").lower()
        return (
            "contentfilter" in body
            or '"code":"1301"' in body
            or '"code": "1301"' in body
        )

    def _fmt(self) -> str:
        if self.status == HTTPStatus.UNAUTHORIZED:
            return "Invalid API key. Please check your API key and try again."

        if self.status == HTTPStatus.TOO_MANY_REQUESTS:
            return "Rate limit exceeded. Please wait a moment before trying again."

        rid = self.headers.get("x-request-id") or self.headers.get("request-id")
        if self.status:
            try:
                status_label = f"{self.status} {HTTPStatus(self.status).phrase}"
            except ValueError:
                status_label = str(self.status)
        else:
            status_label = "N/A"
        parts = [
            f"LLM backend error [{self.provider}]",
            f"  status: {status_label}",
            f"  reason: {self.reason or 'N/A'}",
            f"  request_id: {rid or 'N/A'}",
            f"  endpoint: {self.endpoint}",
            f"  model: {self.model}",
            f"  provider_message: {self.parsed_error or 'N/A'}",
            f"  body_excerpt: {self._excerpt(self.body_text)}",
            f"  payload_summary: {self.payload_summary.model_dump_json(exclude_none=True)}",
        ]
        return "\n".join(parts)

    @staticmethod
    def _excerpt(s: str, *, n: int = 400) -> str:
        s = s.strip().replace("\n", " ")
        return s[:n] + ("…" if len(s) > n else "")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    error: ErrorDetail | dict[str, Any] | None = None
    message: str | None = None
    detail: str | None = None

    @property
    def primary_message(self) -> str | None:
        if e := self.error:
            match e:
                case {"message": str(m)}:
                    return m
                case {"type": str(t)}:
                    return f"Error: {t}"
                case ErrorDetail(message=str(m)):
                    return m
        if m := self.message:
            return m
        if d := self.detail:
            return d
        return None


class BackendErrorBuilder:
    @classmethod
    def build_http_error(
        cls,
        *,
        provider: str,
        endpoint: str,
        error: HttpError,
        model: str,
        messages: Sequence[LLMMessage],
        temperature: float | None,
        has_tools: bool,
        tool_choice: StrToolChoice | AvailableTool | None,
    ) -> BackendError:
        from mistralai.client.errors import SDKError

        response = error.raw_response if isinstance(error, SDKError) else error.response
        body_text = cls._read_response_body(response, error)

        return BackendError(
            provider=provider,
            endpoint=endpoint,
            status=response.status_code,
            reason=response.reason_phrase,
            headers=response.headers,
            body_text=body_text,
            parsed_error=cls._parse_provider_error(body_text),
            model=model,
            payload_summary=cls._payload_summary(
                model, messages, temperature, has_tools, tool_choice
            ),
        )

    @classmethod
    def build_request_error(
        cls,
        *,
        provider: str,
        endpoint: str,
        error: httpx.RequestError,
        model: str,
        messages: Sequence[LLMMessage],
        temperature: float | None,
        has_tools: bool,
        tool_choice: StrToolChoice | AvailableTool | None,
    ) -> BackendError:
        return BackendError(
            provider=provider,
            endpoint=endpoint,
            status=None,
            reason=str(error) or repr(error),
            headers={},
            body_text=None,
            parsed_error="Network error",
            model=model,
            payload_summary=cls._payload_summary(
                model, messages, temperature, has_tools, tool_choice
            ),
        )

    @staticmethod
    def _read_response_body(response: httpx.Response, error: HttpError) -> str | None:
        try:
            response.read()
            return response.text
        except Exception:
            pass
        if body := getattr(error, "body", None):
            return body
        return str(error)

    @staticmethod
    def _parse_provider_error(body_text: str | None) -> str | None:
        if not body_text:
            return None
        try:
            data = orjson.loads(body_text)
            error_model = ErrorResponse.model_validate(data)
            return error_model.primary_message
        except (orjson.JSONDecodeError, ValidationError):
            return None

    @staticmethod
    def _payload_summary(
        model_name: str,
        messages: Sequence[LLMMessage],
        temperature: float | None,
        has_tools: bool,
        tool_choice: StrToolChoice | AvailableTool | None,
    ) -> PayloadSummary:
        total_chars = sum(len(m.content or "") for m in messages)
        return PayloadSummary(
            model=model_name,
            message_count=len(messages),
            approx_chars=total_chars,
            temperature=temperature,
            has_tools=has_tools,
            tool_choice=tool_choice,
        )
