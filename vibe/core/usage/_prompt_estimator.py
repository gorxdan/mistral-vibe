from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

import orjson
from pydantic_core import to_jsonable_python

from vibe.core.llm.types import CompletionRequest
from vibe.core.types import FileImageSource
from vibe.core.usage._context import PromptTokenEstimate

_BYTES_PER_BASE_TOKEN = 4
_COLD_FACTOR = 2.0
_MIN_FACTOR = 1.10
_MAX_FACTOR = 4.0
PROMPT_OBSERVATION_WINDOW = 32
_ESTIMATOR_VERSION = 1


@dataclass(frozen=True, slots=True)
class PromptObservation:
    base_tokens: int
    actual_tokens: int


@dataclass(frozen=True, slots=True)
class PromptFootprint:
    profile_key: str
    base_tokens: int
    strict_tokens: int


@dataclass(frozen=True, slots=True)
class PromptReservationPlan:
    footprint: PromptFootprint
    completion_tokens: int
    input_cost_usd_per_token: float
    completion_cost_usd: float
    adaptive: bool
    allow_completion_reduction: bool = False
    minimum_completion_tokens: int = 1


def request_prompt_footprint(request: CompletionRequest) -> PromptFootprint:
    normalized_messages = [
        message.model_dump(
            mode="json",
            exclude_none=True,
            exclude={
                "images",
                "injected",
                "injected_kind",
                "message_id",
                "reasoning_message_id",
            },
        )
        for message in request.messages
    ]
    raw_messages = [
        message.model_dump(mode="json", exclude_none=True, exclude={"images"})
        for message in request.messages
    ]
    tools = [
        tool.model_dump(mode="json", exclude_none=True) for tool in request.tools or []
    ]
    tool_choice = to_jsonable_python(request.tool_choice, fallback=str)
    normalized_payload = {
        "messages": normalized_messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "response_format": request.response_format,
        "extra_body": request.extra_body,
    }
    raw_payload = {
        "messages": raw_messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "response_format": request.response_format,
        "extra_body": request.extra_body,
    }
    image_bytes, has_images = _expanded_image_bytes(request)
    normalized_bytes = len(orjson.dumps(normalized_payload)) + image_bytes
    strict_tokens = max(1, len(orjson.dumps(raw_payload)) + image_bytes)
    base_tokens = max(1, math.ceil(normalized_bytes / _BYTES_PER_BASE_TOKEN))
    profile_identity = {
        "estimator_version": _ESTIMATOR_VERSION,
        "provider": request.model.provider,
        "model": request.model.name,
        "thinking": request.model.thinking,
        "preserve_reasoning": request.model.preserve_reasoning,
        "vision": has_images,
        "tools": bool(tools),
        "tool_choice": request.tool_choice is not None,
        "schema": request.response_format is not None,
        "extra_body": request.extra_body is not None,
    }
    profile_digest = sha256(
        orjson.dumps(profile_identity, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()
    profile_key = f"v{_ESTIMATOR_VERSION}:{profile_digest}"
    return PromptFootprint(
        profile_key=profile_key, base_tokens=base_tokens, strict_tokens=strict_tokens
    )


def estimate_prompt_tokens(
    plan: PromptReservationPlan, observations: list[PromptObservation]
) -> PromptTokenEstimate:
    footprint = plan.footprint
    if not plan.adaptive:
        return PromptTokenEstimate(
            estimator_version=_ESTIMATOR_VERSION,
            profile_key=footprint.profile_key,
            base_tokens=footprint.base_tokens,
            strict_tokens=footprint.strict_tokens,
            estimated_tokens=footprint.strict_tokens,
            factor=footprint.strict_tokens / footprint.base_tokens,
            sample_count=0,
            adaptive=False,
        )

    comparable = [
        sample
        for sample in observations[-PROMPT_OBSERVATION_WINDOW:]
        if footprint.base_tokens / 2 <= sample.base_tokens <= footprint.base_tokens * 2
    ]
    if comparable:
        uncertainty = max(0.15, 0.50 / math.sqrt(len(comparable)))
        observed_factor = max(
            sample.actual_tokens / sample.base_tokens for sample in comparable
        )
        factor = min(_MAX_FACTOR, max(_MIN_FACTOR, observed_factor * (1 + uncertainty)))
    else:
        factor = _COLD_FACTOR
    guard = max(256, math.ceil(footprint.base_tokens * 0.02))
    estimate = min(
        footprint.strict_tokens, math.ceil(footprint.base_tokens * factor) + guard
    )
    return PromptTokenEstimate(
        estimator_version=_ESTIMATOR_VERSION,
        profile_key=footprint.profile_key,
        base_tokens=footprint.base_tokens,
        strict_tokens=footprint.strict_tokens,
        estimated_tokens=max(1, estimate),
        factor=factor,
        sample_count=len(comparable),
        adaptive=True,
    )


def _expanded_image_bytes(request: CompletionRequest) -> tuple[int, bool]:
    expanded = 0
    has_images = False
    for message in request.messages:
        for image in message.images or []:
            has_images = True
            prefix = len(f"data:{image.mime_type};base64,".encode())
            if isinstance(image.source, FileImageSource):
                size = image.source.path.stat().st_size
                expanded += 4 * ((size + 2) // 3) + prefix
            else:
                expanded += len(image.source.data.encode()) + prefix
    return expanded, has_images


__all__ = [
    "PROMPT_OBSERVATION_WINDOW",
    "PromptFootprint",
    "PromptObservation",
    "PromptReservationPlan",
    "estimate_prompt_tokens",
    "request_prompt_footprint",
]
