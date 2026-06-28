from __future__ import annotations

from collections.abc import AsyncGenerator, Iterable, Sequence
import types
from typing import Any, cast

from tests.mock.utils import mock_llm_chunk
from vibe.core.config import ModelConfig
from vibe.core.types import AvailableTool, LLMChunk, LLMMessage, Role, StrToolChoice


class FakeBackend:
    """Minimal async backend stub to drive Agent.act without network.

    Provide a finite sequence of LLMResult objects to be returned by
    `complete`. When exhausted, returns an empty assistant message.
    """

    def __init__(
        self,
        chunks: LLMChunk
        | Iterable[LLMChunk]
        | Iterable[Iterable[LLMChunk]]
        | None = None,
        *,
        exception_to_raise: Exception | None = None,
    ) -> None:
        """Fake backend that will output the given chunks in the order they are given.

        chunks: A single chunk, a sequence of chunks, or a sequence of sequences of chunks.
        A single chunk would be outputted as such in complete / complete_streaming
        A sequence of chunks will is considered a single stream: a completion would output
        all chunks (either streaming or in an aggregated way)
        A sequence of sequences of chunks is considered a list of streams: each completion
        will output a stream (either streaming or in an aggregated way)
        """
        self._requests_messages: list[list[LLMMessage]] = []
        self._requests_extra_headers: list[dict[str, str] | None] = []
        self._requests_metadata: list[dict[str, str] | None] = []
        self._requests_models: list[ModelConfig] = []
        self._requests_max_tokens: list[int | None] = []
        self._exception_to_raise = exception_to_raise

        self._streams: list[list[LLMChunk]]
        if chunks is None:
            self._streams = []
            return
        if isinstance(chunks, LLMChunk):
            self._streams = [[chunks]]
            return
        if all(isinstance(chunk, LLMChunk) for chunk in chunks):
            self._streams = [[cast(LLMChunk, chunk) for chunk in chunks]]
            return
        if any(isinstance(chunk, LLMChunk) for chunk in chunks):
            raise TypeError(
                f"Invalid type for chunks, expected a value of type "
                f"LLMChunk | Iterable[LLMChunk] | Iterable[Iterable[LLMChunk]], got {chunks!r}"
            )
        chunks = cast(Iterable[Iterable[LLMChunk]], chunks)
        self._streams = [[chunk for chunk in stream] for stream in chunks]

    @property
    def requests_messages(self) -> list[list[LLMMessage]]:
        return self._requests_messages

    @property
    def requests_extra_headers(self) -> list[dict[str, str] | None]:
        return self._requests_extra_headers

    @property
    def requests_metadata(self) -> list[dict[str, str] | None]:
        return self._requests_metadata

    @property
    def requests_models(self) -> list[ModelConfig]:
        return self._requests_models

    @property
    def requests_max_tokens(self) -> list[int | None]:
        return self._requests_max_tokens

    async def __aenter__(self) -> FakeBackend:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        return None

    async def complete(
        self,
        *,
        model: ModelConfig,
        messages: Sequence[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        extra_headers: dict[str, str] | None,
        metadata: dict[str, str] | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        response_headers_sink: dict[str, str] | None = None,
    ) -> LLMChunk:
        if self._exception_to_raise:
            raise self._exception_to_raise

        self._requests_messages.append(list(messages))
        self._requests_models.append(model)
        self._requests_extra_headers.append(extra_headers)
        self._requests_metadata.append(metadata)
        self._requests_max_tokens.append(max_tokens)

        if self._streams:
            stream = self._streams.pop(0)
            chunk_agg = LLMChunk(message=LLMMessage(role=Role.ASSISTANT))
            for chunk in stream:
                chunk_agg += chunk
            return chunk_agg

        return mock_llm_chunk(content="")

    async def complete_streaming(
        self,
        *,
        model: ModelConfig,
        messages: Sequence[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        extra_headers: dict[str, str] | None,
        metadata: dict[str, str] | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        response_headers_sink: dict[str, str] | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        if self._exception_to_raise:
            raise self._exception_to_raise

        self._requests_messages.append(list(messages))
        self._requests_models.append(model)
        self._requests_extra_headers.append(extra_headers)
        self._requests_metadata.append(metadata)
        self._requests_max_tokens.append(max_tokens)

        if self._streams:
            stream = list(self._streams.pop(0))
        else:
            stream = [mock_llm_chunk(content="")]
        for chunk in stream:
            yield chunk
