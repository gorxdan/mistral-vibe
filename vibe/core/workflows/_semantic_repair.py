from __future__ import annotations

from dataclasses import dataclass

import orjson

from vibe.core.config import ModelPurpose, VibeConfig
from vibe.core.llm.backend.factory import create_backend
from vibe.core.llm.types import CompletionRequest
from vibe.core.logger import logger
from vibe.core.repair import RepairAction
from vibe.core.types import LLMMessage, LLMUsage, Role
from vibe.core.usage import SpendPurpose
from vibe.core.usage._session import SessionSpendAdapter
from vibe.core.workflows._result_repair import (
    WorkflowRepairHandoff,
    WorkflowRepairRoute,
)

_MAX_PROMPT_CHARS = 4_000
_MAX_RAW_CHARS = 4_000
_MAX_SCHEMA_CHARS = 8_000
_MAX_OUTPUT_TOKENS = 2_048


@dataclass(frozen=True, slots=True)
class SemanticRepairResult:
    attempted: bool
    content: str | None = None
    usage: LLMUsage | None = None
    error: str | None = None


async def repair_semantic_workflow_result(
    handoff: WorkflowRepairHandoff,
    *,
    original_prompt: str,
    config: VibeConfig,
    spend_adapter: SessionSpendAdapter | None,
) -> SemanticRepairResult:
    alias = config.model_routing.alias_for(ModelPurpose.SEMANTIC_ESCALATION)
    if (
        handoff.route is not WorkflowRepairRoute.SEMANTIC
        or handoff.decision.action is not RepairAction.ESCALATE
        or alias is None
    ):
        return SemanticRepairResult(attempted=False)
    if spend_adapter is None:
        logger.warning(
            "Skipping workflow semantic escalation %s without a spend adapter", alias
        )
        return SemanticRepairResult(attempted=False)

    model = next((item for item in config.models if item.alias == alias), None)
    if model is None:
        logger.warning("Configured semantic escalation model %s was not found", alias)
        return SemanticRepairResult(attempted=False)

    try:
        provider = config.get_provider_for_model(model)
        request = CompletionRequest(
            model=model,
            messages=_semantic_messages(handoff, original_prompt),
            temperature=0.0 if model.temperature is not None else None,
            tools=None,
            tool_choice=None,
            max_tokens=_semantic_max_tokens(model.max_output_tokens),
            extra_headers=provider.extra_headers or None,
            metadata={"call_type": "workflow_semantic_repair"},
            response_format={"type": "json_object"},
        )
        backend = create_backend(
            provider=provider,
            timeout=config.api_timeout,
            retry_max_elapsed_time=config.api_retry_max_elapsed_time,
        )
        async with backend:
            result = await spend_adapter.complete(
                backend, request, purpose=SpendPurpose.REPAIR, is_retry=True
            )
    except Exception as exc:
        logger.warning("Workflow semantic escalation %s failed: %s", alias, exc)
        return SemanticRepairResult(
            attempted=True, error=f"{type(exc).__name__}: {exc}"
        )

    content = result.message.content
    if result.message.tool_calls or not content:
        return SemanticRepairResult(
            attempted=True,
            usage=result.usage,
            error="semantic escalation returned no JSON content",
        )
    return SemanticRepairResult(attempted=True, content=content, usage=result.usage)


def _semantic_messages(
    handoff: WorkflowRepairHandoff, original_prompt: str
) -> list[LLMMessage]:
    schema = orjson.dumps(handoff.schema).decode()
    return [
        LLMMessage(
            role=Role.SYSTEM,
            content=(
                "Correct a semantically invalid JSON result using only the supplied "
                "task and prior result. Do not invent missing evidence. Return one "
                "JSON object with no prose."
            ),
        ),
        LLMMessage(
            role=Role.USER,
            content=(
                f"TASK:\n{_bounded(original_prompt, _MAX_PROMPT_CHARS)}\n\n"
                f"PRIOR RESULT:\n{_bounded(handoff.raw_response, _MAX_RAW_CHARS)}\n\n"
                f"RESULT SCHEMA:\n{_bounded(schema, _MAX_SCHEMA_CHARS)}\n\n"
                f"EXACT DIAGNOSTIC:\n{handoff.diagnostic.for_model()}\n\n"
                f"ESCALATION REASON:\n{handoff.decision.reason}"
            ),
        ),
    ]


def _bounded(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"{value[:limit]}...[{omitted} chars omitted]"


def _semantic_max_tokens(model_limit: int | None) -> int:
    if model_limit is None or model_limit <= 0:
        return _MAX_OUTPUT_TOKENS
    return min(model_limit, _MAX_OUTPUT_TOKENS)


__all__ = ["SemanticRepairResult", "repair_semantic_workflow_result"]
