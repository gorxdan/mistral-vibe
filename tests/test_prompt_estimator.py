from __future__ import annotations

import base64
import math

import pytest

from vibe.core.config import ModelConfig
from vibe.core.llm.types import CompletionRequest
from vibe.core.types import (
    FileImageSource,
    ImageAttachment,
    InlineImageSource,
    LLMMessage,
    Role,
)
from vibe.core.usage._prompt_estimator import (
    PromptFootprint,
    PromptObservation,
    PromptReservationPlan,
    estimate_prompt_tokens,
    request_prompt_footprint,
)
from vibe.core.utils.io import write_safe


def _request(
    content: str = "x" * 20_000,
    *,
    model_name: str = "model-a",
    provider: str = "provider-a",
    images: list[ImageAttachment] | None = None,
) -> CompletionRequest:
    return CompletionRequest(
        model=ModelConfig(
            name=model_name,
            provider=provider,
            alias="display-alias",
            context_window=100_000,
        ),
        messages=[LLMMessage(role=Role.USER, content=content, images=images)],
    )


def _plan(
    request: CompletionRequest, *, adaptive: bool = True
) -> PromptReservationPlan:
    return PromptReservationPlan(
        footprint=request_prompt_footprint(request),
        completion_tokens=100,
        input_cost_usd_per_token=0.000_001,
        completion_cost_usd=0.001,
        adaptive=adaptive,
    )


def test_cold_estimate_uses_base_factor_and_guard() -> None:
    plan = _plan(_request())

    estimate = estimate_prompt_tokens(plan, [])

    expected = min(
        plan.footprint.strict_tokens,
        math.ceil(plan.footprint.base_tokens * 2.0)
        + max(256, math.ceil(plan.footprint.base_tokens * 0.02)),
    )
    assert estimate.estimated_tokens == expected
    assert estimate.factor == 2.0
    assert estimate.sample_count == 0
    assert estimate.adaptive is True


def test_exact_observations_reduce_reservation_with_uncertainty() -> None:
    plan = _plan(_request())
    base = plan.footprint.base_tokens
    cold = estimate_prompt_tokens(plan, [])
    observations = [
        PromptObservation(base_tokens=base, actual_tokens=base) for _ in range(4)
    ]

    learned = estimate_prompt_tokens(plan, observations)

    assert learned.factor == 1.25
    assert learned.sample_count == 4
    assert learned.estimated_tokens < cold.estimated_tokens


def test_unrelated_request_sizes_do_not_calibrate_profile() -> None:
    plan = _plan(_request())
    observation = PromptObservation(
        base_tokens=max(1, plan.footprint.base_tokens // 4), actual_tokens=1
    )

    estimate = estimate_prompt_tokens(plan, [observation])

    assert estimate.factor == 2.0
    assert estimate.sample_count == 0


def test_profile_key_isolates_provider_model_and_request_shape() -> None:
    base = request_prompt_footprint(_request())
    other_model = request_prompt_footprint(_request(model_name="model-b"))
    other_provider = request_prompt_footprint(_request(provider="provider-b"))
    schema = request_prompt_footprint(
        CompletionRequest(
            model=_request().model,
            messages=_request().messages,
            response_format={"type": "json_object"},
        )
    )
    extra_body = request_prompt_footprint(
        CompletionRequest(
            model=_request().model,
            messages=_request().messages,
            extra_body={"vendor_prompt": "value"},
        )
    )

    assert (
        len({
            base.profile_key,
            other_model.profile_key,
            other_provider.profile_key,
            schema.profile_key,
            extra_body.profile_key,
        })
        == 5
    )
    assert "display-alias" not in base.profile_key


def test_strict_mode_preserves_semantic_byte_ceiling() -> None:
    plan = _plan(_request(), adaptive=False)
    request_with_extra_body = CompletionRequest(
        model=_request().model,
        messages=_request().messages,
        extra_body={"vendor_prompt": "x" * 1_000},
    )

    estimate = estimate_prompt_tokens(plan, [])
    with_extra_body = request_prompt_footprint(request_with_extra_body)

    assert estimate.estimated_tokens == plan.footprint.strict_tokens
    assert estimate.adaptive is False
    assert with_extra_body.strict_tokens > plan.footprint.strict_tokens + 900


def test_file_and_inline_images_have_equal_footprints(tmp_path) -> None:
    raw = b"image bytes" * 100
    image_path = tmp_path / "image.png"
    write_safe(image_path, raw.decode())
    encoded = base64.b64encode(raw).decode()
    file_image = ImageAttachment(
        source=FileImageSource(path=image_path),
        alias="image.png",
        mime_type="image/png",
    )
    inline_image = ImageAttachment(
        source=InlineImageSource(data=encoded), alias="image.png", mime_type="image/png"
    )

    file_footprint = request_prompt_footprint(_request(images=[file_image]))
    inline_footprint = request_prompt_footprint(_request(images=[inline_image]))
    text_footprint = request_prompt_footprint(_request())

    assert file_footprint.base_tokens == inline_footprint.base_tokens
    assert file_footprint.strict_tokens == inline_footprint.strict_tokens
    assert file_footprint.base_tokens > text_footprint.base_tokens


@pytest.mark.parametrize(
    (
        "prior_raw_estimates",
        "prior_actual_prompt_tokens",
        "next_raw_size",
        "expected_raw_total",
        "expected_actual_total",
    ),
    [
        (
            [87_178, 112_847, 136_467, 182_093, 184_464, 191_113],
            [21_069, 27_216, 32_801, 44_615, 44_953, 46_663],
            199_280,
            894_162,
            217_317,
        ),
        (
            [
                82_801,
                88_669,
                94_285,
                112_180,
                139_789,
                149_298,
                27_332,
                156_184,
                164_762,
            ],
            [19_986, 21_398, 22_653, 26_674, 32_698, 34_642, 6_659, 36_163, 37_646],
            173_367,
            1_015_300,
            238_519,
        ),
    ],
    ids=["ff3fae2e", "8e77d14c"],
)
def test_observed_sessions_admit_next_call_after_calibration(
    prior_raw_estimates: list[int],
    prior_actual_prompt_tokens: list[int],
    next_raw_size: int,
    expected_raw_total: int,
    expected_actual_total: int,
) -> None:
    observations = [
        PromptObservation(
            base_tokens=math.ceil(raw_estimate / 4), actual_tokens=actual_tokens
        )
        for raw_estimate, actual_tokens in zip(
            prior_raw_estimates, prior_actual_prompt_tokens, strict=True
        )
    ]
    footprint = PromptFootprint(
        profile_key="observed-session",
        base_tokens=math.ceil(next_raw_size / 4),
        strict_tokens=next_raw_size,
    )
    strict = estimate_prompt_tokens(
        PromptReservationPlan(
            footprint=footprint,
            completion_tokens=0,
            input_cost_usd_per_token=0.0,
            completion_cost_usd=0.0,
            adaptive=False,
        ),
        observations,
    )
    adaptive = estimate_prompt_tokens(
        PromptReservationPlan(
            footprint=footprint,
            completion_tokens=0,
            input_cost_usd_per_token=0.0,
            completion_cost_usd=0.0,
            adaptive=True,
        ),
        observations,
    )

    assert sum(prior_raw_estimates) == expected_raw_total
    assert sum(prior_actual_prompt_tokens) == expected_actual_total
    assert expected_actual_total + strict.estimated_tokens > 400_000
    assert expected_actual_total + adaptive.estimated_tokens < 400_000
