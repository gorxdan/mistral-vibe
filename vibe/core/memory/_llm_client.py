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

from typing import Any

from vibe.core.config import ModelConfig, ProviderConfig
from vibe.core.llm.backend.factory import BACKEND_FACTORY
from vibe.core.llm.types import CompletionRequest
from vibe.core.types import LLMMessage


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

    def __init__(
        self,
        *,
        model: ModelConfig,
        provider: ProviderConfig,
        timeout: float,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._provider = provider
        self._timeout = timeout
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
        backend_cls = BACKEND_FACTORY[self._provider.backend]
        async with backend_cls(
            provider=self._provider, timeout=self._timeout
        ) as backend:
            result = await backend.complete(
                CompletionRequest(
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
            )
        return result.message.content
