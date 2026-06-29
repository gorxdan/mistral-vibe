"""``requires_init`` guard decorator for AgentLoop public methods.

Extracted from the loop module so subsystem mixins (e.g. ``session_mixin``) can
apply the same deferred-init gate without a circular import through ``_loop``.

Methods decorated with ``@requires_init`` await ``self.wait_until_ready()`` before
the body runs, so callers hitting a still-initializing loop (``defer_heavy_init``
path) block until heavy init finishes (or surfaces its error).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vibe.core.agent_loop._loop import AgentLoop


def requires_init(fn: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.isasyncgenfunction(fn):

        @wraps(fn)
        async def gen_wrapper(self: AgentLoop, *args: Any, **kwargs: Any) -> Any:
            await self.wait_until_ready()
            agen = fn(self, *args, **kwargs)
            sent: Any = None
            try:
                while True:
                    sent = yield await agen.asend(sent)
            except StopAsyncIteration:
                return
            finally:
                await agen.aclose()

        return gen_wrapper

    @wraps(fn)
    async def wrapper(self: AgentLoop, *args: Any, **kwargs: Any) -> Any:
        await self.wait_until_ready()
        return await fn(self, *args, **kwargs)

    return wrapper
