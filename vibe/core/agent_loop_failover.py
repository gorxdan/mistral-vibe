"""Model-failover mixin for AgentLoop.

Provides the rate-limit / overload / content-filter failover machinery: picking
a fallback from the configured pool, prompting the user (or auto-switching in
headless) on a rate-limit error, and finalising a failover attempt with a
recovery trace. Extracted from the loop module.

Implicit dependencies on the host class (AgentLoop):

Attributes (set by AgentLoop.__init__):
    backend                  (BackendLike — settable; _activate_model reassigns it)
    rate_limit_callback      (RateLimitCallback | None)
    stats                    (AgentStats — _activate_model re-prices it)

Properties (defined on AgentLoop):
    config                   (VibeConfig)

Methods (defined elsewhere on AgentLoop):
    _trace_recovery(*, error_type, action, **extra) -> None
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from vibe.core.llm.backend.factory import create_backend
from vibe.core.llm.types import BackendLike
from vibe.core.logger import logger
from vibe.core.types import (
    AgentStats,
    ContentFilterError,
    RateLimitCallback,
    RateLimitError,
    ServerError,
    TransportError,
    UnclassifiedBackendError,
)

if TYPE_CHECKING:
    from vibe.core.config import ModelConfig, VibeConfig


class AgentLoopFailoverMixin:
    """Mixin that adds model-failover to AgentLoop.

    See module docstring for the implicit contract with the host class.
    """

    # Declared for type-checking only; set by AgentLoop.__init__.
    _fallback_model_override: ModelConfig | None
    _tried_fallback_aliases: set[str]
    backend: BackendLike
    rate_limit_callback: RateLimitCallback | None
    stats: AgentStats

    @property
    def config(self) -> VibeConfig: ...

    def _trace_recovery(
        self, *, error_type: str, action: str, **extra: Any
    ) -> None: ...
    def _switch_to_fallback_model(self) -> ModelConfig | None:
        current_alias = (
            self._fallback_model_override.alias
            if self._fallback_model_override
            else self.config.active_model
        )
        self._tried_fallback_aliases.add(current_alias)
        for alias in self.config.fallback_models:
            if alias in self._tried_fallback_aliases:
                continue
            self._tried_fallback_aliases.add(alias)
            model = next((m for m in self.config.models if m.alias == alias), None)
            if model is None or not self.config.is_model_available(model):
                continue
            return self._activate_model(model)
        return None

    def _activate_model(self, model: ModelConfig) -> ModelConfig:
        self._tried_fallback_aliases.add(model.alias)
        provider = self.config.get_provider_for_model(model)
        self._fallback_model_override = model
        self.backend = create_backend(
            provider=provider, timeout=self.config.api_timeout
        )
        self.stats.update_pricing(model.input_price, model.output_price)
        self.stats.update_model_bounds(model.auto_compact_threshold)
        return model

    def _switchable_model_aliases(self) -> list[str]:
        return [
            m.alias
            for m in self.config.models
            if m.alias not in self._tried_fallback_aliases
            and self.config.is_model_available(m)
        ]

    def _switch_to_chosen_model(self, alias: str) -> ModelConfig | None:
        model = next((m for m in self.config.models if m.alias == alias), None)
        if model is None or not self.config.is_model_available(model):
            return None
        return self._activate_model(model)

    def _auto_fallback_headless(self) -> ModelConfig | None:
        candidates = self._switchable_model_aliases()
        if not candidates:
            return None
        return self._switch_to_chosen_model(candidates[0])

    async def _prompt_model_switch_on_rate_limit(
        self, error: RateLimitError
    ) -> ModelConfig | None:
        if self.rate_limit_callback is None:
            return None
        candidates = self._switchable_model_aliases()
        if not candidates:
            return None
        chosen = await self.rate_limit_callback(error.provider, error.model, candidates)
        if not chosen:
            return None
        return self._switch_to_chosen_model(chosen)

    def _failover_unavailable_hint(self, reason: str) -> str:
        if not self.config.fallback_models:
            hint = (
                f"{reason} and no fallback_models configured; set "
                "config.fallback_models to enable automatic failover."
            )
        else:
            hint = (
                f"{reason} and fallback pool exhausted (tried "
                f"{sorted(self._tried_fallback_aliases)})."
            )
        logger.warning("%s", hint)
        return hint

    def _apply_failover(
        self,
        exc: (
            RateLimitError
            | ContentFilterError
            | ServerError
            | TransportError
            | UnclassifiedBackendError
        ),
        fallback: ModelConfig | None,
        *,
        error_type: str,
        unavailable_reason: str,
        log_template: str,
        log_prefix_args: Sequence[object] = (),
    ) -> None:
        # Finish a failover attempt. On success, log the switch and record a
        # recovery trace; when no fallback resolved, attach an actionable hint
        # to `exc` and re-raise so the reason reaches the user-visible error
        # rather than only the log file. The caller resolves `fallback`
        # (configured pool, plus the rate-limit callback/headless path for
        # rate-limit errors) — the structural difference between the clauses.
        # `log_template`'s trailing %r is always the fallback alias; the helper
        # appends it after `log_prefix_args` so call sites stay closure-free.
        if fallback is None:
            exc.failover_hint = self._failover_unavailable_hint(unavailable_reason)
            raise exc
        logger.warning(log_template, *log_prefix_args, fallback.alias)
        self._trace_recovery(
            error_type=error_type, action="failover", fallback=fallback.alias
        )
