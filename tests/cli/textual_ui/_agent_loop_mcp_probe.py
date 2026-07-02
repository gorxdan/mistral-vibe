# Subprocess probe (never imported): builds an AgentLoop and exits non-zero if
# the mcp SDK leaked into sys.modules. Pass --defer for defer_heavy_init=True.
from __future__ import annotations

from collections.abc import AsyncGenerator
import sys
import types

from vibe.core.agent_loop import AgentLoop, AgentLoopParams
from vibe.core.config import SessionLoggingConfig, VibeConfig
from vibe.core.config.harness_files import (
    init_harness_files_manager,
    reset_harness_files_manager,
)
from vibe.core.llm.types import BackendLike, CompletionRequest, LLMChunk


class _Backend:
    async def __aenter__(self) -> BackendLike:
        raise AssertionError

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        return None

    async def complete(
        self,
        request: CompletionRequest,
        *,
        response_headers_sink: dict[str, str] | None = None,
    ) -> LLMChunk:
        raise AssertionError

    def complete_streaming(
        self,
        request: CompletionRequest,
        *,
        response_headers_sink: dict[str, str] | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        raise AssertionError


def main() -> None:
    defer = "--defer" in sys.argv[1:]
    init_harness_files_manager("user", "project")
    try:
        config = VibeConfig(
            enable_connectors=False, session_logging=SessionLoggingConfig(enabled=False)
        )
        loop = AgentLoop(
            config,
            backend=_Backend(),
            params=AgentLoopParams(defer_heavy_init=defer, headless=True),
        )
        if loop._deferred_init_thread is not None:
            loop._deferred_init_thread.join()
    finally:
        reset_harness_files_manager()

    blocked = ["vibe.core.tools.mcp.tools", "mcp"]
    loaded = [name for name in blocked if name in sys.modules]
    if loaded:
        raise SystemExit(f"unexpected agent loop modules loaded: {loaded}")


if __name__ == "__main__":
    main()
