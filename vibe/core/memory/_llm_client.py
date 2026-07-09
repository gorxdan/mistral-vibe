"""Shared base for the fork's standalone LLM memory clients.

``MemorySelector`` / ``MemoryExtractor`` / ``MemoryConsolidator`` /
``MemoryVerifier`` each run on their OWN standalone backend (never the agent's
main one) so a failure can't trigger model failover or emergency compaction.
That standalone-backend shape — five common ``__init__`` fields plus the
``async with backend_cls(...) as backend: await backend.complete(...)`` block —
is identical across the four, so it lives here once.

Concrete subclasses build their own message list and parse the JSON response;
this base only owns model/provider/timeout storage and the completion call.
"""

from __future__ import annotations

import time
from typing import Any

from vibe.core.config import ModelConfig, ProviderConfig
from vibe.core.llm.backend.factory import BACKEND_FACTORY
from vibe.core.llm.types import CompletionRequest
from vibe.core.types import LLMMessage, LLMUsage
from vibe.core.usage import CallKind, UsageMeter, usage_cost
from vibe.core.utils.tokens import approx_token_count


class _MemoryLLMClient:
    """Standalone-backend LLM client shared by the memory collaborators.

    Subclasses set any extra fields (e.g. ``_max_selected``, ``_project_root``)
    in their own ``__init__`` after calling ``super().__init__``.
    """

    _model: ModelConfig
    _provider: ProviderConfig
    _timeout: float
    _extra_headers: dict[str, str]
    _extra_body: dict[str, Any] | None
    _call_kind: CallKind
    _usage_meter: UsageMeter | None

    def __init__(
        self,
        *,
        model: ModelConfig,
        provider: ProviderConfig,
        timeout: float,
        call_kind: CallKind,
        usage_meter: UsageMeter | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._provider = provider
        self._timeout = timeout
        self._call_kind = call_kind
        self._usage_meter = usage_meter
        self._extra_headers = extra_headers or {}
        self._extra_body = extra_body or None

    async def _complete_json(
        self, messages: list[LLMMessage], *, max_tokens: int, temperature: float | None
    ) -> str | None:
        """Run one completion on the standalone backend, returning raw content.

        ``temperature`` is required: pass ``self._model.temperature`` to forward
        the model's own value (None keeps a temperature-omitting model's wire
        contract), or an explicit float to override.
        """
        request = CompletionRequest(
            model=self._model,
            messages=messages,
            temperature=temperature,
            tools=None,
            tool_choice=None,
            max_tokens=max_tokens,
            extra_headers=self._extra_headers,
            response_format={"type": "json_object"},
            extra_body=self._extra_body,
        )
        if self._usage_meter is None:
            backend_cls = BACKEND_FACTORY[self._provider.backend]
            async with backend_cls(
                provider=self._provider, timeout=self._timeout
            ) as backend:
                result = await backend.complete(request)
            return result.message.content

        estimated_prompt_tokens = sum(
            approx_token_count(message.content or "") for message in messages
        )
        estimated_usage = LLMUsage(
            prompt_tokens=estimated_prompt_tokens, completion_tokens=max_tokens
        )
        reservation = self._usage_meter.try_reserve(
            estimated_prompt_tokens + max_tokens,
            estimated_cost_usd=usage_cost(self._model, estimated_usage),
        )
        if reservation is None:
            return None
        started = time.monotonic()
        try:
            backend_cls = BACKEND_FACTORY[self._provider.backend]
            async with backend_cls(
                provider=self._provider, timeout=self._timeout
            ) as backend:
                result = await backend.complete(request)
        except BaseException:
            self._usage_meter.reconcile(
                reservation,
                usage=None,
                model=self._model,
                provider=self._provider,
                call_kind=self._call_kind,
                duration_s=time.monotonic() - started,
            )
            raise
        self._usage_meter.reconcile(
            reservation,
            usage=result.usage,
            model=self._model,
            provider=self._provider,
            call_kind=self._call_kind,
            duration_s=time.monotonic() - started,
            result_used=True,
        )
        return result.message.content
