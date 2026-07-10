from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum, auto
import time
from typing import Protocol

__all__ = [
    "ProviderRetryController",
    "RetryAttemptAdmission",
    "SpendRetryCause",
    "SpendRetryPolicyReason",
    "authorize_provider_retry",
    "bind_provider_retry_controller",
    "bind_retry_attempt_admission",
    "iterate_provider_stream",
]


class SpendRetryCause(StrEnum):
    HTTP_STATUS = auto()
    TRANSPORT = auto()
    REASONING_EFFORT = auto()


class SpendRetryPolicyReason(StrEnum):
    ATTEMPT_LIMIT = auto()
    ELAPSED_LIMIT = auto()


class RetryAttemptAdmission(Protocol):
    def authorize_retry(self, cause: SpendRetryCause) -> None: ...

    def reject_retry_policy(
        self,
        cause: SpendRetryCause,
        reason: SpendRetryPolicyReason,
        *,
        elapsed_s: float,
        max_elapsed_s: float,
        next_delay_s: float,
        max_retries: int,
    ) -> None: ...


_ACTIVE_ADMISSION: ContextVar[RetryAttemptAdmission | None] = ContextVar(
    "vibe_retry_attempt_admission", default=None
)
_ACTIVE_CONTROLLER: ContextVar[ProviderRetryController | None] = ContextVar(
    "vibe_provider_retry_controller", default=None
)


@contextmanager
def bind_retry_attempt_admission(admission: RetryAttemptAdmission) -> Iterator[None]:
    token = _ACTIVE_ADMISSION.set(admission)
    try:
        yield
    finally:
        _ACTIVE_ADMISSION.reset(token)


@contextmanager
def bind_provider_retry_controller(
    controller: ProviderRetryController,
) -> Iterator[None]:
    token = _ACTIVE_CONTROLLER.set(controller)
    try:
        yield
    finally:
        _ACTIVE_CONTROLLER.reset(token)


@dataclass(slots=True)
class ProviderRetryController:
    max_elapsed_time: float
    max_retries: int = 2
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    _started_at: float = field(init=False)
    _authorized_retries: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.max_elapsed_time < 0:
            raise ValueError("max_elapsed_time cannot be negative")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self._started_at = self.clock()

    async def authorize(self, cause: SpendRetryCause, *, delay_s: float) -> bool:
        if delay_s < 0:
            raise ValueError("retry delay cannot be negative")
        elapsed = self._elapsed()
        if self._authorized_retries >= self.max_retries:
            self._reject(
                cause,
                SpendRetryPolicyReason.ATTEMPT_LIMIT,
                elapsed_s=elapsed,
                next_delay_s=delay_s,
            )
            return False
        if elapsed + delay_s >= self.max_elapsed_time:
            self._reject(
                cause,
                SpendRetryPolicyReason.ELAPSED_LIMIT,
                elapsed_s=elapsed,
                next_delay_s=delay_s,
            )
            return False
        if delay_s > 0:
            await self.sleep(delay_s)
        elapsed = self._elapsed()
        if elapsed >= self.max_elapsed_time:
            self._reject(
                cause,
                SpendRetryPolicyReason.ELAPSED_LIMIT,
                elapsed_s=elapsed,
                next_delay_s=0.0,
            )
            return False
        admission = _ACTIVE_ADMISSION.get()
        if admission is not None:
            admission.authorize_retry(cause)
        self._authorized_retries += 1
        return True

    def _elapsed(self) -> float:
        return max(self.clock() - self._started_at, 0.0)

    def _reject(
        self,
        cause: SpendRetryCause,
        reason: SpendRetryPolicyReason,
        *,
        elapsed_s: float,
        next_delay_s: float,
    ) -> None:
        admission = _ACTIVE_ADMISSION.get()
        if admission is None:
            return
        admission.reject_retry_policy(
            cause,
            reason,
            elapsed_s=elapsed_s,
            max_elapsed_s=self.max_elapsed_time,
            next_delay_s=next_delay_s,
            max_retries=self.max_retries,
        )


async def authorize_provider_retry(cause: SpendRetryCause, *, delay_s: float) -> bool:
    controller = _ACTIVE_CONTROLLER.get()
    if controller is not None:
        return await controller.authorize(cause, delay_s=delay_s)
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    admission = _ACTIVE_ADMISSION.get()
    if admission is not None:
        admission.authorize_retry(cause)
    return True


async def iterate_provider_stream[T](
    stream: AsyncGenerator[T, None], controller: ProviderRetryController
) -> AsyncGenerator[T, None]:
    try:
        while True:
            try:
                with bind_provider_retry_controller(controller):
                    item = await anext(stream)
            except StopAsyncIteration:
                return
            yield item
    finally:
        with suppress(Exception):
            await stream.aclose()
