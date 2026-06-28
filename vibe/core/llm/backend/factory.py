from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vibe.core.types import Backend

if TYPE_CHECKING:
    from vibe.core.config import ProviderConfig
    from vibe.core.llm.types import BackendLike


def create_backend(*, provider: ProviderConfig, timeout: float = 720.0) -> BackendLike:
    backend = provider.backend
    if backend == Backend.MISTRAL:
        from vibe.core.llm.backend.mistral import MistralBackend

        return MistralBackend(provider=provider, timeout=timeout)
    if backend == Backend.GENERIC:
        from vibe.core.llm.backend.generic import GenericBackend

        return GenericBackend(provider=provider, timeout=timeout)
    raise ValueError(f"no backend registered for {backend!r}")


def __getattr__(name: str) -> Any:
    if name == "BACKEND_FACTORY":
        from vibe.core.llm.backend.generic import GenericBackend
        from vibe.core.llm.backend.mistral import MistralBackend

        factory: dict[Backend, type] = {
            Backend.MISTRAL: MistralBackend,
            Backend.GENERIC: GenericBackend,
        }
        globals()["BACKEND_FACTORY"] = factory
        return factory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
