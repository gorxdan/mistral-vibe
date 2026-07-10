from __future__ import annotations

from dataclasses import dataclass

import orjson

from vibe.core.config import ModelPurpose, VibeConfig
from vibe.core.llm.backend.factory import create_backend
from vibe.core.llm.types import CompletionRequest
from vibe.core.logger import logger
from vibe.core.types import LLMMessage, LLMUsage, Role
from vibe.core.usage._context import SpendPurpose
from vibe.core.usage._session import SessionSpendAdapter
from vibe.core.workflows._result_repair import (
    WorkflowRepairHandoff,
    WorkflowRepairRoute,
)

_MAX_RAW_CHARS = 4_000
_MAX_OUTPUT_TOKENS = 512


@dataclass(frozen=True, slots=True)
class FormatterRepairResult:
    attempted: bool
    content: str | None = None
    usage: LLMUsage | None = None


async def format_workflow_result(
    handoff: WorkflowRepairHandoff,
    *,
    config: VibeConfig,
    spend_adapter: SessionSpendAdapter | None,
) -> FormatterRepairResult:
    alias = config.model_routing.alias_for(ModelPurpose.FORMATTER)
    if handoff.route is not WorkflowRepairRoute.FORMATTER or alias is None:
        return FormatterRepairResult(attempted=False)
    if spend_adapter is None:
        logger.warning(
            "Skipping configured workflow formatter %s without a spend adapter", alias
        )
        return FormatterRepairResult(attempted=False)

    model = next((item for item in config.models if item.alias == alias), None)
    if model is None:
        logger.warning("Configured workflow formatter model %s was not found", alias)
        return FormatterRepairResult(attempted=False)

    try:
        provider = config.get_provider_for_model(model)
        request = CompletionRequest(
            model=model,
            messages=_formatter_messages(handoff),
            temperature=0.0 if model.temperature is not None else None,
            tools=None,
            tool_choice=None,
            max_tokens=_formatter_max_tokens(model.max_output_tokens),
            extra_headers=provider.extra_headers or None,
            metadata={"call_type": "workflow_json_repair"},
            response_format={"type": "json_object"},
        )
        backend = create_backend(
            provider=provider, timeout=config.api_timeout, retry_max_elapsed_time=0.0
        )
        async with backend:
            result = await spend_adapter.complete(
                backend, request, purpose=SpendPurpose.REPAIR, is_retry=True
            )
    except Exception as exc:
        logger.warning("Workflow formatter %s failed: %s", alias, exc)
        return FormatterRepairResult(attempted=True)

    content = result.message.content
    if result.message.tool_calls or not content:
        return FormatterRepairResult(attempted=True, usage=result.usage)
    return FormatterRepairResult(attempted=True, content=content, usage=result.usage)


def _formatter_messages(handoff: WorkflowRepairHandoff) -> list[LLMMessage]:
    raw = _bounded_raw(handoff.raw_response)
    schema = orjson.dumps(handoff.schema).decode()
    diagnostic = handoff.diagnostic.for_model()
    return [
        LLMMessage(
            role=Role.SYSTEM,
            content=(
                "Repair JSON syntax and object shape only. Do not infer facts or "
                "solve the underlying task. Return one JSON object with no prose."
            ),
        ),
        LLMMessage(
            role=Role.USER,
            content=(
                f"RAW OUTPUT:\n{raw}\n\n"
                f"RESULT SCHEMA:\n{schema}\n\n"
                f"EXACT DIAGNOSTIC:\n{diagnostic}"
            ),
        ),
    ]


def _bounded_raw(raw: str) -> str:
    if len(raw) <= _MAX_RAW_CHARS:
        return raw
    omitted = len(raw) - _MAX_RAW_CHARS
    return f"{raw[:_MAX_RAW_CHARS]}...[{omitted} chars omitted]"


def _formatter_max_tokens(model_limit: int | None) -> int:
    if model_limit is None or model_limit <= 0:
        return _MAX_OUTPUT_TOKENS
    return min(model_limit, _MAX_OUTPUT_TOKENS)


__all__ = ["FormatterRepairResult", "format_workflow_result"]
